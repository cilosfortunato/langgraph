import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import httpx
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import redis
from contextlib import asynccontextmanager

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração do Redis para debounce
redis_client = None
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        decode_responses=True
    )
    redis_client.ping()
    logger.info("Conectado ao Redis")
except Exception as e:
    logger.warning(f"Redis não disponível: {e}")
    redis_client = None

# Configuração de segurança
security = HTTPBearer()
API_KEY = os.getenv('API_KEY', '151fb361-f295-4a4f-84c9-ec1f42599a67')

# Importar configuração do Cognee
try:
    from cognee_config import setup_cognee
    setup_cognee()
    logger.info("Cognee configurado com sucesso")
except Exception as e:
    logger.warning(f"Erro ao configurar Cognee: {e}")

# Importar agente
try:
    from agents.simple_agent import SimpleAgent
except ImportError:
    logger.error("Não foi possível importar SimpleAgent")
    SimpleAgent = None

# Modelos Pydantic
class MessageInput(BaseModel):
    mensagem: str
    agent_id: str
    debounce: int = 15000
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    cliente_id: Optional[str] = ""
    user_id: str
    id_conta: str

class AgentConfig(BaseModel):
    id: Optional[str] = None
    name: str
    description: str = ""
    instructions: str
    model: str = "openai/gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    skills: List[Dict[str, Any]] = Field(default_factory=list)
    webhook_url: str = ""

class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    instructions: str
    model: str
    temperature: float
    max_tokens: int
    skills: List[Dict[str, Any]]
    webhook_url: str
    created_at: str
    updated_at: str

# Armazenamento em memória para agentes (em produção, usar banco de dados)
agents_storage: Dict[str, Dict[str, Any]] = {}

# Armazenamento para debounce
debounce_storage: Dict[str, Dict[str, Any]] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gerencia o ciclo de vida da aplicação."""
    logger.info("Iniciando aplicação LangGraph")
    
    # Criar agente padrão se não existir
    if not agents_storage:
        default_agent = {
            "id": "1677dc47-20d0-442a-80a8-171f00d39d39",
            "name": "Assistente Padrão",
            "description": "Agente padrão para processamento de mensagens",
            "instructions": "Você é um assistente útil e prestativo. Responda de forma clara e objetiva.",
            "model": "openai/gpt-4o-mini",
            "temperature": 0.7,
            "max_tokens": 1000,
            "skills": [],
            "webhook_url": "",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        agents_storage[default_agent["id"]] = default_agent
        logger.info(f"Agente padrão criado: {default_agent['id']}")
    
    yield
    
    logger.info("Encerrando aplicação LangGraph")

# Criar aplicação FastAPI
app = FastAPI(
    title="LangGraph API",
    description="API para processamento de agentes multi-tenant com Cognee",
    version="1.0.0",
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Função de autenticação
def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verifica a chave de API."""
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API inválida")
    return credentials.credentials

# Função para obter chave de API do header
def get_api_key_from_header(request: Request):
    """Obtém a chave de API do header X-API-Key."""
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Chave de API inválida")
    return api_key

# Função para processar mensagens em lote
async def process_message_batch(batch_key: str, messages: List[MessageInput]):
    """Processa um lote de mensagens."""
    try:
        logger.info(f"Processando lote {batch_key} com {len(messages)} mensagens")
        
        # Agrupar mensagens por agente
        agent_groups = {}
        for msg in messages:
            if msg.agent_id not in agent_groups:
                agent_groups[msg.agent_id] = []
            agent_groups[msg.agent_id].append(msg)
        
        # Processar cada grupo de agente
        for agent_id, agent_messages in agent_groups.items():
            await process_agent_messages(agent_id, agent_messages)
        
        # Remover do armazenamento de debounce
        if batch_key in debounce_storage:
            del debounce_storage[batch_key]
        
        logger.info(f"Lote {batch_key} processado com sucesso")
        
    except Exception as e:
        logger.error(f"Erro ao processar lote {batch_key}: {e}")

async def process_agent_messages(agent_id: str, messages: List[MessageInput]):
    """Processa mensagens para um agente específico."""
    try:
        # Verificar se o agente existe
        if agent_id not in agents_storage:
            logger.error(f"Agente {agent_id} não encontrado")
            return
        
        agent_config = agents_storage[agent_id]
        
        # Criar instância do agente se SimpleAgent estiver disponível
        if SimpleAgent:
            agent = SimpleAgent(agent_config)
        else:
            logger.warning("SimpleAgent não disponível, usando processamento básico")
            agent = None
        
        # Processar cada mensagem
        for msg in messages:
            try:
                # Gerar session_id se não fornecido
                if not msg.session_id:
                    msg.session_id = str(uuid.uuid4())
                
                # Extrair tenant_id do id_conta
                tenant_id = f"tenant_{msg.id_conta}"
                
                # Tentar recuperar histórico do Cognee
                context = await get_cognee_context(tenant_id, msg.user_id, msg.session_id)
                
                # Processar mensagem
                if agent:
                    response = agent.process(
                        messages=[msg.mensagem],
                        user_id=msg.user_id,
                        session_id=msg.session_id,
                        tenant_id=tenant_id
                    )
                else:
                    # Processamento básico sem agente
                    response = {
                        "messages": ["Desculpe, o sistema está temporariamente indisponível."],
                        "transferir": False,
                        "session_id": msg.session_id,
                        "user_id": msg.user_id,
                        "agent_id": agent_id,
                        "custom": [],
                        "agent_usage": {}
                    }
                
                # Salvar no Cognee
                await save_to_cognee(tenant_id, msg.user_id, msg.session_id, msg.mensagem, response["messages"][0])
                
                # Enviar webhook se configurado
                if agent_config.get("webhook_url"):
                    await send_webhook(agent_config["webhook_url"], response)
                
                logger.info(f"Mensagem processada para agente {agent_id}, usuário {msg.user_id}")
                
            except Exception as e:
                logger.error(f"Erro ao processar mensagem individual: {e}")
        
    except Exception as e:
        logger.error(f"Erro ao processar mensagens do agente {agent_id}: {e}")

async def get_cognee_context(tenant_id: str, user_id: str, session_id: str) -> Dict[str, Any]:
    """Recupera contexto do Cognee."""
    try:
        # Tentar importar e usar Cognee
        import cognee
        
        # Configurar tenant
        await cognee.config.set_tenant(tenant_id)
        
        # Buscar histórico da conversa
        search_results = await cognee.search(f"user:{user_id} session:{session_id}")
        
        return {
            "history": search_results,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "session_id": session_id
        }
        
    except Exception as e:
        logger.warning(f"Erro ao recuperar contexto do Cognee: {e}")
        return {}

async def save_to_cognee(tenant_id: str, user_id: str, session_id: str, user_message: str, bot_response: str):
    """Salva conversa no Cognee."""
    try:
        import cognee
        
        # Configurar tenant
        await cognee.config.set_tenant(tenant_id)
        
        # Preparar dados da conversa
        conversation_data = {
            "user_id": user_id,
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "user_message": user_message,
            "bot_response": bot_response
        }
        
        # Adicionar ao Cognee
        await cognee.add([conversation_data])
        
        logger.info(f"Conversa salva no Cognee para tenant {tenant_id}")
        
    except Exception as e:
        logger.warning(f"Erro ao salvar no Cognee: {e}")

async def send_webhook(webhook_url: str, response: Dict[str, Any]):
    """Envia resposta via webhook."""
    try:
        async with httpx.AsyncClient() as client:
            webhook_response = await client.post(
                webhook_url,
                json=response,
                timeout=30.0
            )
            webhook_response.raise_for_status()
            logger.info(f"Webhook enviado com sucesso para {webhook_url}")
    except Exception as e:
        logger.error(f"Erro ao enviar webhook para {webhook_url}: {e}")

# Endpoints da API

@app.get("/")
async def root():
    """Endpoint raiz."""
    return {
        "message": "LangGraph API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Verificação de saúde."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "redis_connected": redis_client is not None,
        "agents_count": len(agents_storage)
    }

@app.post("/messages")
async def receive_messages(
    messages: List[MessageInput],
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key_from_header)
):
    """Recebe mensagens para processamento com debounce."""
    try:
        logger.info(f"Recebidas {len(messages)} mensagens")
        
        # Agrupar mensagens por chave de debounce
        debounce_groups = {}
        
        for msg in messages:
            # Criar chave de debounce baseada em agente, usuário e sessão
            debounce_key = f"{msg.agent_id}_{msg.user_id}_{msg.session_id or 'no_session'}"
            
            if debounce_key not in debounce_groups:
                debounce_groups[debounce_key] = []
            debounce_groups[debounce_key].append(msg)
        
        # Processar cada grupo de debounce
        for debounce_key, group_messages in debounce_groups.items():
            # Verificar se já existe um timer para esta chave
            if debounce_key in debounce_storage:
                # Cancelar timer anterior
                debounce_storage[debounce_key]["task"].cancel()
                # Adicionar novas mensagens
                debounce_storage[debounce_key]["messages"].extend(group_messages)
            else:
                # Criar novo grupo
                debounce_storage[debounce_key] = {
                    "messages": group_messages,
                    "task": None
                }
            
            # Criar nova task de debounce
            debounce_time = group_messages[0].debounce / 1000.0  # Converter para segundos
            
            async def delayed_process(key: str, delay: float):
                await asyncio.sleep(delay)
                if key in debounce_storage:
                    batch_messages = debounce_storage[key]["messages"]
                    await process_message_batch(key, batch_messages)
            
            task = asyncio.create_task(delayed_process(debounce_key, debounce_time))
            debounce_storage[debounce_key]["task"] = task
        
        return {
            "success": True,
            "message": f"Lote de {len(messages)} mensagens recebido e aceito",
            "debounce_groups": len(debounce_groups)
        }
        
    except Exception as e:
        logger.error(f"Erro ao processar mensagens: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents", response_model=List[AgentResponse])
async def list_agents(api_key: str = Depends(get_api_key_from_header)):
    """Lista todos os agentes."""
    return list(agents_storage.values())

@app.get("/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str, api_key: str = Depends(get_api_key_from_header)):
    """Obtém um agente específico."""
    if agent_id not in agents_storage:
        raise HTTPException(status_code=404, detail="Agente não encontrado")
    return agents_storage[agent_id]

@app.post("/agents", response_model=AgentResponse)
async def create_agent(agent: AgentConfig, api_key: str = Depends(get_api_key_from_header)):
    """Cria um novo agente."""
    agent_id = agent.id or str(uuid.uuid4())
    
    if agent_id in agents_storage:
        raise HTTPException(status_code=400, detail="Agente já existe")
    
    agent_data = {
        "id": agent_id,
        "name": agent.name,
        "description": agent.description,
        "instructions": agent.instructions,
        "model": agent.model,
        "temperature": agent.temperature,
        "max_tokens": agent.max_tokens,
        "skills": agent.skills,
        "webhook_url": agent.webhook_url,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    agents_storage[agent_id] = agent_data
    logger.info(f"Agente criado: {agent_id}")
    
    return agent_data

@app.put("/agents/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    agent: AgentConfig,
    api_key: str = Depends(get_api_key_from_header)
):
    """Atualiza um agente existente."""
    if agent_id not in agents_storage:
        raise HTTPException(status_code=404, detail="Agente não encontrado")
    
    agent_data = agents_storage[agent_id]
    agent_data.update({
        "name": agent.name,
        "description": agent.description,
        "instructions": agent.instructions,
        "model": agent.model,
        "temperature": agent.temperature,
        "max_tokens": agent.max_tokens,
        "skills": agent.skills,
        "webhook_url": agent.webhook_url,
        "updated_at": datetime.now().isoformat()
    })
    
    logger.info(f"Agente atualizado: {agent_id}")
    return agent_data

@app.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str, api_key: str = Depends(get_api_key_from_header)):
    """Remove um agente."""
    if agent_id not in agents_storage:
        raise HTTPException(status_code=404, detail="Agente não encontrado")
    
    del agents_storage[agent_id]
    logger.info(f"Agente removido: {agent_id}")
    
    return {"message": "Agente removido com sucesso"}

# Endpoints do Cognee

@app.get("/cognee/status")
async def cognee_status(api_key: str = Depends(get_api_key_from_header)):
    """Verifica o status do Cognee."""
    try:
        import cognee
        return {
            "status": "available",
            "version": getattr(cognee, '__version__', 'unknown')
        }
    except ImportError:
        return {
            "status": "not_available",
            "error": "Cognee não está instalado"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/cognee/tenants/{tenant_id}/search")
async def search_cognee(
    tenant_id: str,
    query: str,
    api_key: str = Depends(get_api_key_from_header)
):
    """Busca no Cognee para um tenant específico."""
    try:
        import cognee
        
        # Configurar tenant
        await cognee.config.set_tenant(tenant_id)
        
        # Realizar busca
        results = await cognee.search(query)
        
        return {
            "tenant_id": tenant_id,
            "query": query,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Erro na busca do Cognee: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info"
    )
