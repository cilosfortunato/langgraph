# LangGraph com Cognee

Sistema LangGraph com integração Cognee para processamento de agentes multi-tenant.

## Características

- **Multi-tenant**: Isolamento completo de dados por tenant
- **Memória Persistente**: Integração com Cognee para histórico de conversas
- **Debounce Inteligente**: Agrupamento de mensagens para otimização
- **Skills Dinâmicas**: Sistema de habilidades configuráveis
- **OpenRouter**: Suporte a múltiplos LLMs via OpenRouter
- **Webhooks**: Notificações em tempo real

## Tecnologias

- **FastAPI**: API REST de alta performance
- **LangGraph**: Orquestração de agentes
- **Cognee**: Camada de memória e conhecimento
- **Neo4j**: Banco de dados de grafos
- **Qdrant**: Banco de dados vetorial
- **Redis**: Cache e debounce
- **OpenRouter**: Gateway para LLMs

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/cilosfortunato/langgraph.git
cd langgraph
```

2. Instale as dependências:
```bash
pip install -r requirements.txt
```

3. Configure as variáveis de ambiente:
```bash
cp .env.example .env
# Edite o arquivo .env com suas configurações
```

4. Inicie os serviços:
```bash
# Para desenvolvimento
python start_server.py

# Para produção
python start_production.py
```

## Configuração

### Variáveis de Ambiente

```env
# API
API_KEY=sua-chave-api
WEBHOOK_URL=https://seu-webhook.com

# OpenRouter
OPENROUTER_API_KEY=sua-chave-openrouter

# Cognee
GRAPH_DATABASE_PROVIDER=neo4j
VECTOR_DB_PROVIDER=qdrant
VECTOR_DB_URL=http://localhost:6333
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Redis
REDIS_URL=redis://localhost:6379
```

## Uso

### Criar um Agente

```bash
curl -X POST "http://localhost:8000/agents" \
  -H "X-API-Key: sua-chave" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Assistente",
    "description": "Assistente virtual",
    "instructions": "Você é um assistente útil",
    "model": "openai/gpt-4o-mini",
    "skills": []
  }'
```

### Enviar Mensagem

```bash
curl -X POST "http://localhost:8000/messages" \
  -H "X-API-Key: sua-chave" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{
      "mensagem": "Olá!",
      "user_id": "user123",
      "session_id": "session123",
      "tenant_id": "tenant123",
      "agent_id": "agent-id"
    }]
  }'
```

## Arquitetura

### Fluxo de Processamento

1. **Recepção**: API recebe mensagem
2. **Debounce**: Agrupa mensagens por 15s
3. **Cognee**: Recupera histórico e contexto
4. **Skills**: Aplica habilidades relevantes
5. **LLM**: Processa via OpenRouter
6. **Webhook**: Envia resposta

### Estrutura de Dados

#### Input
```json
{
  "messages": [{
    "mensagem": "texto da mensagem",
    "user_id": "identificador do usuário",
    "session_id": "identificador da sessão",
    "tenant_id": "identificador do tenant",
    "agent_id": "identificador do agente"
  }]
}
```

#### Output
```json
{
  "messages": ["resposta do agente"],
  "transferir": false,
  "session_id": "session123",
  "user_id": "user123",
  "agent_id": "agent123",
  "custom": [],
  "agent_usage": {
    "input_tokens": 10,
    "output_tokens": 25,
    "model": "openai/gpt-4o-mini"
  }
}
```

## Desenvolvimento

### Testes

```bash
# Executar todos os testes
pytest

# Testes específicos
pytest tests/test_api.py
pytest tests/test_agents.py
```

### Monitoramento

```bash
# Dashboard de monitoramento
python monitoring/dashboard_server.py
```

## Produção

### Docker

```bash
# Build
docker build -t langgraph .

# Run
docker-compose up -d
```

### Deploy

Veja o arquivo `GUIA_PRODUCAO.md` para instruções detalhadas de deploy.

## Contribuição

1. Fork o projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## Licença

MIT License
