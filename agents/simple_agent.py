import json
import logging
import os
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuração do LiteLLM para OpenRouter
os.environ["LITELLM_LOG"] = "DEBUG"

class AgentState(BaseModel):
    """Estado do agente durante o processamento."""
    messages: List[str] = Field(default_factory=list)
    user_id: str = ""
    session_id: str = ""
    agent_id: str = ""
    tenant_id: str = ""
    context: Dict[str, Any] = Field(default_factory=dict)
    skills_used: List[str] = Field(default_factory=list)
    transferir: bool = False
    custom: List[Dict[str, str]] = Field(default_factory=list)
    agent_usage: Dict[str, Any] = Field(default_factory=dict)
    
class SimpleAgent:
    """Agente simples para processamento de mensagens."""
    
    def __init__(self, agent_config: Dict[str, Any]):
        self.agent_id = agent_config.get('id', str(uuid.uuid4()))
        self.name = agent_config.get('name', 'SimpleAgent')
        self.description = agent_config.get('description', '')
        self.instructions = agent_config.get('instructions', '')
        self.model = agent_config.get('model', 'openai/gpt-4o-mini')
        self.temperature = agent_config.get('temperature', 0.7)
        self.max_tokens = agent_config.get('max_tokens', 1000)
        self.skills = agent_config.get('skills', [])
        self.webhook_url = agent_config.get('webhook_url', '')
        
        # Configurar LLM
        self.llm = self._setup_llm()
        
        # Configurar grafo
        self.graph = self._build_graph()
        
    def _setup_llm(self):
        """Configura o LLM baseado no modelo especificado."""
        try:
            # Verificar se é um modelo OpenRouter
            if self.model.startswith('openai/') or self.model.startswith('anthropic/') or self.model.startswith('google/'):
                # Usar OpenRouter via LiteLLM
                from litellm import completion
                
                # Configurar chave da API
                openrouter_key = os.getenv('OPENROUTER_API_KEY')
                if not openrouter_key:
                    raise ValueError("OPENROUTER_API_KEY não encontrada")
                
                # Configurar LiteLLM para OpenRouter
                os.environ["OPENROUTER_API_KEY"] = openrouter_key
                
                # Criar wrapper para LiteLLM
                class LiteLLMWrapper:
                    def __init__(self, model, temperature, max_tokens):
                        self.model = model
                        self.temperature = temperature
                        self.max_tokens = max_tokens
                        
                    def invoke(self, messages):
                        # Converter mensagens para formato LiteLLM
                        litellm_messages = []
                        for msg in messages:
                            if hasattr(msg, 'content'):
                                if isinstance(msg, SystemMessage):
                                    litellm_messages.append({"role": "system", "content": msg.content})
                                elif isinstance(msg, HumanMessage):
                                    litellm_messages.append({"role": "user", "content": msg.content})
                            else:
                                litellm_messages.append({"role": "user", "content": str(msg)})
                        
                        try:
                            response = completion(
                                model=self.model,
                                messages=litellm_messages,
                                temperature=self.temperature,
                                max_tokens=self.max_tokens,
                                api_base="https://openrouter.ai/api/v1",
                                api_key=os.getenv('OPENROUTER_API_KEY')
                            )
                            
                            # Criar objeto de resposta compatível
                            class MockResponse:
                                def __init__(self, content, usage=None):
                                    self.content = content
                                    self.usage_metadata = usage or {}
                                    
                            usage = {
                                'input_tokens': response.usage.prompt_tokens if response.usage else 0,
                                'output_tokens': response.usage.completion_tokens if response.usage else 0,
                                'model': self.model
                            }
                            
                            return MockResponse(response.choices[0].message.content, usage)
                            
                        except Exception as e:
                            logger.error(f"Erro ao chamar LiteLLM: {e}")
                            # Fallback para resposta padrão
                            return MockResponse("Desculpe, não consegui processar sua mensagem no momento.")
                
                return LiteLLMWrapper(self.model, self.temperature, self.max_tokens)
            
            else:
                # Usar ChatOpenAI padrão para modelos OpenAI nativos
                return ChatOpenAI(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
        except Exception as e:
            logger.error(f"Erro ao configurar LLM: {e}")
            # Fallback para modelo padrão
            return ChatOpenAI(
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=1000
            )
    
    def _build_graph(self):
        """Constrói o grafo de processamento do agente."""
        workflow = StateGraph(AgentState)
        
        # Adicionar nós
        workflow.add_node("process_message", self._process_message)
        workflow.add_node("apply_skills", self._apply_skills)
        workflow.add_node("generate_response", self._generate_response)
        
        # Definir fluxo
        workflow.set_entry_point("process_message")
        workflow.add_edge("process_message", "apply_skills")
        workflow.add_edge("apply_skills", "generate_response")
        workflow.add_edge("generate_response", END)
        
        return workflow.compile()
    
    def _process_message(self, state: AgentState) -> AgentState:
        """Processa a mensagem inicial."""
        logger.info(f"Processando mensagem para agente {self.agent_id}")
        
        # Adicionar contexto básico
        state.context.update({
            "agent_name": self.name,
            "timestamp": datetime.now().isoformat(),
            "model": self.model
        })
        
        return state
    
    def _apply_skills(self, state: AgentState) -> AgentState:
        """Aplica skills relevantes baseadas na mensagem."""
        if not self.skills:
            return state
            
        message = state.messages[0] if state.messages else ""
        message_lower = message.lower()
        
        # Verificar quais skills são relevantes
        relevant_skills = []
        for skill in self.skills:
            keywords = skill.get('keywords', [])
            if any(keyword.lower() in message_lower for keyword in keywords):
                relevant_skills.append(skill)
                state.skills_used.append(skill.get('name', 'unknown'))
        
        # Adicionar contexto das skills
        if relevant_skills:
            skill_context = []
            for skill in relevant_skills:
                skill_context.append({
                    "name": skill.get('name', ''),
                    "description": skill.get('description', ''),
                    "context": skill.get('context', '')
                })
            state.context["skills"] = skill_context
        
        return state
    
    def _generate_response(self, state: AgentState) -> AgentState:
        """Gera a resposta usando o LLM."""
        try:
            # Preparar mensagens para o LLM
            messages = []
            
            # Adicionar instruções do sistema
            system_prompt = self.instructions
            if state.context.get("skills"):
                system_prompt += "\n\nSkills disponíveis:\n"
                for skill in state.context["skills"]:
                    system_prompt += f"- {skill['name']}: {skill['description']}\n"
                    if skill.get('context'):
                        system_prompt += f"  Contexto: {skill['context']}\n"
            
            messages.append(SystemMessage(content=system_prompt))
            
            # Adicionar mensagem do usuário
            user_message = state.messages[0] if state.messages else "Olá"
            messages.append(HumanMessage(content=user_message))
            
            # Criar LLM com configurações específicas do agente
            response = self.llm.invoke(messages)
            
            # Extrair resposta
            if hasattr(response, 'content'):
                response_text = response.content
            else:
                response_text = str(response)
            
            # Atualizar estado
            state.messages = [response_text]
            
            # Adicionar informações de uso se disponível
            if hasattr(response, 'usage_metadata'):
                state.agent_usage = response.usage_metadata
            
            logger.info(f"Resposta gerada para agente {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Erro ao gerar resposta: {e}")
            state.messages = ["Desculpe, ocorreu um erro ao processar sua mensagem."]
        
        return state
    
    def process(self, messages: List[str], user_id: str, session_id: str, tenant_id: str) -> Dict[str, Any]:
        """Processa mensagens e retorna resposta."""
        try:
            # Criar estado inicial
            initial_state = AgentState(
                messages=messages,
                user_id=user_id,
                session_id=session_id,
                agent_id=self.agent_id,
                tenant_id=tenant_id
            )
            
            # Executar grafo
            final_state = self.graph.invoke(initial_state)
            
            # Preparar resposta
            response = {
                "messages": final_state.messages,
                "transferir": final_state.transferir,
                "session_id": final_state.session_id,
                "user_id": final_state.user_id,
                "agent_id": final_state.agent_id,
                "custom": final_state.custom,
                "agent_usage": final_state.agent_usage
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Erro ao processar mensagem: {e}")
            return {
                "messages": ["Desculpe, ocorreu um erro ao processar sua mensagem."],
                "transferir": False,
                "session_id": session_id,
                "user_id": user_id,
                "agent_id": self.agent_id,
                "custom": [],
                "agent_usage": {}
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte o agente para dicionário."""
        return {
            "id": self.agent_id,
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "skills": self.skills,
            "webhook_url": self.webhook_url
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SimpleAgent':
        """Cria agente a partir de dicionário."""
        return cls(data)
