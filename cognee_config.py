import os
import logging
from typing import Optional

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_cognee():
    """Configura o Cognee com as configurações necessárias."""
    try:
        # Importar Cognee
        import cognee
        
        # Configurar variáveis de ambiente para Cognee
        os.environ["GRAPH_DATABASE_PROVIDER"] = "neo4j"
        os.environ["VECTOR_DB_PROVIDER"] = "qdrant"
        
        # Configurações do Neo4j
        neo4j_url = os.getenv("NEO4J_URL", "bolt://localhost:7687")
        neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        
        os.environ["NEO4J_URL"] = neo4j_url
        os.environ["NEO4J_USER"] = neo4j_user
        os.environ["NEO4J_PASSWORD"] = neo4j_password
        
        # Configurações do Qdrant
        qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        
        os.environ["VECTOR_DB_URL"] = qdrant_url
        if qdrant_api_key:
            os.environ["QDRANT_API_KEY"] = qdrant_api_key
        
        # Configurações do LLM
        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        
        if llm_provider == "openrouter":
            # Configurar OpenRouter
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            if openrouter_key:
                os.environ["OPENAI_API_KEY"] = openrouter_key
                os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"
                
                # Aplicar patch para LiteLLM reconhecer OpenRouter
                try:
                    from cognee_patches import apply_litellm_patch
                    apply_litellm_patch()
                    logger.info("Patch LiteLLM aplicado com sucesso")
                except ImportError:
                    logger.warning("Patch LiteLLM não disponível")
        
        elif llm_provider == "openai":
            # Configurar OpenAI
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                os.environ["OPENAI_API_KEY"] = openai_key
        
        # Configurar modelo padrão
        default_model = os.getenv("DEFAULT_LLM_MODEL", "gpt-3.5-turbo")
        os.environ["LLM_MODEL"] = default_model
        
        # Configurar embedding model
        embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
        os.environ["EMBEDDING_MODEL"] = embedding_model
        
        # Configurações de desenvolvimento
        if os.getenv("DISABLE_EXTERNAL_DBS") == "true":
            logger.warning("Bancos de dados externos desabilitados - modo desenvolvimento")
            return
        
        # Inicializar Cognee
        cognee.config.set_llm_provider(llm_provider)
        
        # Configurar provedores de banco de dados
        cognee.config.set_graph_database_provider("neo4j")
        cognee.config.set_vector_database_provider("qdrant")
        
        logger.info("Cognee configurado com sucesso")
        logger.info(f"LLM Provider: {llm_provider}")
        logger.info(f"Graph DB: Neo4j ({neo4j_url})")
        logger.info(f"Vector DB: Qdrant ({qdrant_url})")
        
        return True
        
    except ImportError as e:
        logger.error(f"Cognee não está instalado: {e}")
        return False
    except Exception as e:
        logger.error(f"Erro ao configurar Cognee: {e}")
        return False

def get_cognee_status() -> dict:
    """Retorna o status atual do Cognee."""
    try:
        import cognee
        
        status = {
            "available": True,
            "version": getattr(cognee, '__version__', 'unknown'),
            "graph_db_provider": os.getenv("GRAPH_DATABASE_PROVIDER", "not_set"),
            "vector_db_provider": os.getenv("VECTOR_DB_PROVIDER", "not_set"),
            "llm_provider": os.getenv("LLM_PROVIDER", "not_set"),
            "neo4j_url": os.getenv("NEO4J_URL", "not_set"),
            "qdrant_url": os.getenv("VECTOR_DB_URL", "not_set"),
            "external_dbs_disabled": os.getenv("DISABLE_EXTERNAL_DBS") == "true"
        }
        
        return status
        
    except ImportError:
        return {
            "available": False,
            "error": "Cognee não está instalado"
        }
    except Exception as e:
        return {
            "available": False,
            "error": str(e)
        }

def create_tenant_database(tenant_id: str) -> bool:
    """Cria banco de dados específico para um tenant."""
    try:
        import cognee
        
        # Configurar tenant
        cognee.config.set_tenant(tenant_id)
        
        logger.info(f"Banco de dados criado para tenant: {tenant_id}")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao criar banco para tenant {tenant_id}: {e}")
        return False

def validate_cognee_configuration() -> dict:
    """Valida a configuração do Cognee."""
    validation_results = {
        "cognee_available": False,
        "graph_db_configured": False,
        "vector_db_configured": False,
        "llm_configured": False,
        "errors": []
    }
    
    try:
        # Verificar se Cognee está disponível
        import cognee
        validation_results["cognee_available"] = True
        
        # Verificar configuração do banco de grafos
        if os.getenv("GRAPH_DATABASE_PROVIDER") == "neo4j":
            neo4j_url = os.getenv("NEO4J_URL")
            neo4j_user = os.getenv("NEO4J_USER")
            neo4j_password = os.getenv("NEO4J_PASSWORD")
            
            if neo4j_url and neo4j_user and neo4j_password:
                validation_results["graph_db_configured"] = True
            else:
                validation_results["errors"].append("Configuração Neo4j incompleta")
        
        # Verificar configuração do banco vetorial
        if os.getenv("VECTOR_DB_PROVIDER") == "qdrant":
            qdrant_url = os.getenv("VECTOR_DB_URL")
            
            if qdrant_url:
                validation_results["vector_db_configured"] = True
            else:
                validation_results["errors"].append("URL do Qdrant não configurada")
        
        # Verificar configuração do LLM
        llm_provider = os.getenv("LLM_PROVIDER", "openai")
        
        if llm_provider == "openrouter":
            if os.getenv("OPENROUTER_API_KEY"):
                validation_results["llm_configured"] = True
            else:
                validation_results["errors"].append("Chave OpenRouter não configurada")
        elif llm_provider == "openai":
            if os.getenv("OPENAI_API_KEY"):
                validation_results["llm_configured"] = True
            else:
                validation_results["errors"].append("Chave OpenAI não configurada")
        
    except ImportError:
        validation_results["errors"].append("Cognee não está instalado")
    except Exception as e:
        validation_results["errors"].append(f"Erro na validação: {str(e)}")
    
    return validation_results

if __name__ == "__main__":
    # Teste da configuração
    print("Configurando Cognee...")
    success = setup_cognee()
    
    if success:
        print("✅ Cognee configurado com sucesso")
        
        # Mostrar status
        status = get_cognee_status()
        print("\nStatus do Cognee:")
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        # Validar configuração
        validation = validate_cognee_configuration()
        print("\nValidação da configuração:")
        for key, value in validation.items():
            if key != "errors":
                status_icon = "✅" if value else "❌"
                print(f"  {status_icon} {key}: {value}")
        
        if validation["errors"]:
            print("\nErros encontrados:")
            for error in validation["errors"]:
                print(f"  ❌ {error}")
    else:
        print("❌ Falha ao configurar Cognee")
