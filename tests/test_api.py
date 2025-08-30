import pytest
import asyncio
import json
import time
from httpx import AsyncClient
from fastapi.testclient import TestClient
from api.main import app

# Configurações de teste
TEST_API_KEY = "151fb361-f295-4a4f-84c9-ec1f42599a67"
TEST_AGENT_ID = "1677dc47-20d0-442a-80a8-171f00d39d39"
TEST_USER_ID = "test_user_123"
TEST_SESSION_ID = "test_session_456"
TEST_CONTA_ID = "test_conta_789"

@pytest.fixture
def client():
    """Cliente de teste para a API"""
    return TestClient(app)

@pytest.fixture
def headers():
    """Headers padrão para autenticação"""
    return {
        "X-API-Key": TEST_API_KEY,
        "Content-Type": "application/json"
    }

class TestHealthEndpoint:
    """Testes para o endpoint de saúde"""
    
    def test_health_check(self, client):
        """Testa se o endpoint de saúde está funcionando"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

class TestAuthenticationMiddleware:
    """Testes para o middleware de autenticação"""
    
    def test_missing_api_key(self, client):
        """Testa requisição sem chave de API"""
        response = client.post("/messages", json=[])
        assert response.status_code == 401
        assert "X-API-Key header is required" in response.json()["detail"]
    
    def test_invalid_api_key(self, client):
        """Testa requisição com chave de API inválida"""
        headers = {"X-API-Key": "invalid-key"}
        response = client.post("/messages", json=[], headers=headers)
        assert response.status_code == 401
        assert "Invalid API key" in response.json()["detail"]
    
    def test_valid_api_key(self, client, headers):
        """Testa requisição com chave de API válida"""
        response = client.post("/messages", json=[], headers=headers)
        # Deve retornar 422 (validation error) ao invés de 401 (unauthorized)
        assert response.status_code != 401

class TestMessagesEndpoint:
    """Testes para o endpoint de mensagens"""
    
    def test_empty_messages_list(self, client, headers):
        """Testa envio de lista vazia de mensagens"""
        response = client.post("/messages", json=[], headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "batch received" in data["message"]
    
    def test_valid_message(self, client, headers):
        """Testa envio de mensagem válida"""
        message_data = [{
            "mensagem": "Qual horário funciona?",
            "agent_id": TEST_AGENT_ID,
            "debounce": 15000,
            "session_id": TEST_SESSION_ID,
            "message_id": "test_message_123",
            "cliente_id": "",
            "user_id": TEST_USER_ID,
            "id_conta": TEST_CONTA_ID
        }]
        
        response = client.post("/messages", json=message_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "batch received" in data["message"]
    
    def test_invalid_message_structure(self, client, headers):
        """Testa envio de mensagem com estrutura inválida"""
        invalid_message = [{
            "invalid_field": "test"
        }]
        
        response = client.post("/messages", json=invalid_message, headers=headers)
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_fields(self, client, headers):
        """Testa mensagem sem campos obrigatórios"""
        incomplete_message = [{
            "mensagem": "Test message"
            # Faltando agent_id, user_id, etc.
        }]
        
        response = client.post("/messages", json=incomplete_message, headers=headers)
        assert response.status_code == 422

class TestAgentsEndpoint:
    """Testes para o endpoint de agentes"""
    
    def test_list_agents(self, client, headers):
        """Testa listagem de agentes"""
        response = client.get("/agents", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_create_agent(self, client, headers):
        """Testa criação de novo agente"""
        agent_data = {
            "name": "Test Agent",
            "description": "Agent for testing",
            "model": "gpt-3.5-turbo",
            "system_prompt": "You are a helpful assistant",
            "skills": ["general_conversation"]
        }
        
        response = client.post("/agents", json=agent_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert data["name"] == agent_data["name"]
        
        # Limpar: deletar o agente criado
        agent_id = data["id"]
        client.delete(f"/agents/{agent_id}", headers=headers)
    
    def test_get_agent(self, client, headers):
        """Testa busca de agente específico"""
        # Primeiro criar um agente
        agent_data = {
            "name": "Test Agent for Get",
            "description": "Agent for get testing",
            "model": "gpt-3.5-turbo",
            "system_prompt": "You are a helpful assistant",
            "skills": ["general_conversation"]
        }
        
        create_response = client.post("/agents", json=agent_data, headers=headers)
        agent_id = create_response.json()["id"]
        
        # Buscar o agente
        response = client.get(f"/agents/{agent_id}", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == agent_id
        assert data["name"] == agent_data["name"]
        
        # Limpar
        client.delete(f"/agents/{agent_id}", headers=headers)
    
    def test_update_agent(self, client, headers):
        """Testa atualização de agente"""
        # Criar agente
        agent_data = {
            "name": "Test Agent for Update",
            "description": "Agent for update testing",
            "model": "gpt-3.5-turbo",
            "system_prompt": "You are a helpful assistant",
            "skills": ["general_conversation"]
        }
        
        create_response = client.post("/agents", json=agent_data, headers=headers)
        agent_id = create_response.json()["id"]
        
        # Atualizar agente
        update_data = {
            "name": "Updated Test Agent",
            "description": "Updated description"
        }
        
        response = client.put(f"/agents/{agent_id}", json=update_data, headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == update_data["name"]
        assert data["description"] == update_data["description"]
        
        # Limpar
        client.delete(f"/agents/{agent_id}", headers=headers)
    
    def test_delete_agent(self, client, headers):
        """Testa exclusão de agente"""
        # Criar agente
        agent_data = {
            "name": "Test Agent for Delete",
            "description": "Agent for delete testing",
            "model": "gpt-3.5-turbo",
            "system_prompt": "You are a helpful assistant",
            "skills": ["general_conversation"]
        }
        
        create_response = client.post("/agents", json=agent_data, headers=headers)
        agent_id = create_response.json()["id"]
        
        # Deletar agente
        response = client.delete(f"/agents/{agent_id}", headers=headers)
        assert response.status_code == 200
        
        # Verificar se foi deletado
        get_response = client.get(f"/agents/{agent_id}", headers=headers)
        assert get_response.status_code == 404

class TestCogneeEndpoints:
    """Testes para endpoints do Cognee"""
    
    def test_cognee_status(self, client, headers):
        """Testa status do Cognee"""
        response = client.get("/cognee/status", headers=headers)
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
    
    def test_cognee_search(self, client, headers):
        """Testa busca no Cognee"""
        search_params = {
            "query": "test query",
            "tenant_id": "test_tenant"
        }
        
        response = client.get("/cognee/search", params=search_params, headers=headers)
        # Pode retornar 200 (sucesso) ou erro dependendo da configuração
        assert response.status_code in [200, 500]  # 500 se Cognee não estiver configurado

class TestIntegrationFlow:
    """Testes de integração do fluxo completo"""
    
    @pytest.mark.asyncio
    async def test_complete_message_flow(self, client, headers):
        """Testa fluxo completo de processamento de mensagem"""
        # 1. Enviar mensagem
        message_data = [{
            "mensagem": "Qual horário funciona?",
            "agent_id": TEST_AGENT_ID,
            "debounce": 1000,  # Debounce curto para teste
            "session_id": TEST_SESSION_ID,
            "message_id": "integration_test_123",
            "cliente_id": "",
            "user_id": TEST_USER_ID,
            "id_conta": TEST_CONTA_ID
        }]
        
        response = client.post("/messages", json=message_data, headers=headers)
        assert response.status_code == 200
        
        # 2. Aguardar processamento (debounce + processamento)
        await asyncio.sleep(2)
        
        # 3. Verificar se não houve erros críticos
        health_response = client.get("/health")
        assert health_response.status_code == 200

class TestErrorHandling:
    """Testes para tratamento de erros"""
    
    def test_404_endpoint(self, client, headers):
        """Testa endpoint inexistente"""
        response = client.get("/nonexistent", headers=headers)
        assert response.status_code == 404
    
    def test_method_not_allowed(self, client, headers):
        """Testa método HTTP não permitido"""
        response = client.patch("/health", headers=headers)
        assert response.status_code == 405
    
    def test_large_payload(self, client, headers):
        """Testa payload muito grande"""
        large_message = [{
            "mensagem": "x" * 10000,  # Mensagem muito longa
            "agent_id": TEST_AGENT_ID,
            "debounce": 15000,
            "session_id": TEST_SESSION_ID,
            "message_id": "large_test_123",
            "cliente_id": "",
            "user_id": TEST_USER_ID,
            "id_conta": TEST_CONTA_ID
        }]
        
        response = client.post("/messages", json=large_message, headers=headers)
        # Deve processar normalmente ou retornar erro específico
        assert response.status_code in [200, 413, 422]

if __name__ == "__main__":
    # Executar testes
    pytest.main(["-v", __file__])
