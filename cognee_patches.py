import logging
import os
from typing import Dict, Any, Optional

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def apply_litellm_patch():
    """Aplica patch para corrigir problemas do LiteLLM com OpenRouter."""
    try:
        import litellm
        
        # Patch para forçar reconhecimento de chaves OpenRouter
        original_completion = litellm.completion
        
        def patched_completion(*args, **kwargs):
            """Versão corrigida da função completion do LiteLLM."""
            
            # Verificar se é uma chave OpenRouter
            api_key = kwargs.get('api_key') or os.getenv('OPENROUTER_API_KEY')
            model = kwargs.get('model', '')
            
            if api_key and api_key.startswith('sk-or-v1-'):
                # Forçar configurações para OpenRouter
                kwargs['api_base'] = kwargs.get('api_base', 'https://openrouter.ai/api/v1')
                kwargs['api_key'] = api_key
                
                # Mapear modelos para formato OpenRouter se necessário
                if not model.startswith(('openai/', 'anthropic/', 'google/')):
                    if 'gpt' in model.lower():
                        kwargs['model'] = f"openai/{model}"
                    elif 'claude' in model.lower():
                        kwargs['model'] = f"anthropic/{model}"
                    elif 'gemini' in model.lower():
                        kwargs['model'] = f"google/{model}"
                
                # Configurar headers específicos do OpenRouter
                headers = kwargs.get('headers', {})
                headers.update({
                    'HTTP-Referer': 'https://langgraph-cognee.local',
                    'X-Title': 'LangGraph with Cognee'
                })
                kwargs['headers'] = headers
                
                logger.info(f"Usando OpenRouter com modelo: {kwargs['model']}")
            
            return original_completion(*args, **kwargs)
        
        # Aplicar o patch
        litellm.completion = patched_completion
        
        logger.info("Patch LiteLLM aplicado com sucesso")
        return True
        
    except ImportError:
        logger.warning("LiteLLM não está disponível para patch")
        return False
    except Exception as e:
        logger.error(f"Erro ao aplicar patch LiteLLM: {e}")
        return False

def apply_cognee_logging_patch():
    """Aplica patch para melhorar logging do Cognee."""
    try:
        import cognee
        
        # Configurar logging mais detalhado
        cognee_logger = logging.getLogger('cognee')
        cognee_logger.setLevel(logging.INFO)
        
        # Adicionar handler se não existir
        if not cognee_logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            cognee_logger.addHandler(handler)
        
        logger.info("Patch de logging do Cognee aplicado")
        return True
        
    except ImportError:
        logger.warning("Cognee não está disponível para patch de logging")
        return False
    except Exception as e:
        logger.error(f"Erro ao aplicar patch de logging do Cognee: {e}")
        return False

def apply_neo4j_connection_patch():
    """Aplica patch para melhorar conexão com Neo4j."""
    try:
        from neo4j import GraphDatabase
        import time
        
        # Patch para retry automático em conexões Neo4j
        original_driver = GraphDatabase.driver
        
        def patched_driver(uri, auth=None, **config):
            """Driver Neo4j com retry automático."""
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    driver = original_driver(uri, auth=auth, **config)
                    # Testar conexão
                    with driver.session() as session:
                        session.run("RETURN 1")
                    logger.info(f"Conectado ao Neo4j: {uri}")
                    return driver
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Tentativa {attempt + 1} falhou, tentando novamente em {retry_delay}s: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"Falha ao conectar ao Neo4j após {max_retries} tentativas: {e}")
                        raise
        
        GraphDatabase.driver = patched_driver
        
        logger.info("Patch de conexão Neo4j aplicado")
        return True
        
    except ImportError:
        logger.warning("Neo4j driver não está disponível para patch")
        return False
    except Exception as e:
        logger.error(f"Erro ao aplicar patch Neo4j: {e}")
        return False

def apply_qdrant_connection_patch():
    """Aplica patch para melhorar conexão com Qdrant."""
    try:
        from qdrant_client import QdrantClient
        import time
        
        # Patch para retry automático em conexões Qdrant
        original_init = QdrantClient.__init__
        
        def patched_init(self, url=None, port=None, grpc_port=None, prefer_grpc=False, 
                        https=None, api_key=None, prefix=None, timeout=None, 
                        host=None, path=None, **kwargs):
            """QdrantClient com retry automático."""
            max_retries = 3
            retry_delay = 1
            
            for attempt in range(max_retries):
                try:
                    original_init(self, url=url, port=port, grpc_port=grpc_port,
                                prefer_grpc=prefer_grpc, https=https, api_key=api_key,
                                prefix=prefix, timeout=timeout, host=host, path=path, **kwargs)
                    
                    # Testar conexão
                    self.get_collections()
                    logger.info(f"Conectado ao Qdrant: {url or host}")
                    return
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Tentativa {attempt + 1} falhou, tentando novamente em {retry_delay}s: {e}")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                    else:
                        logger.error(f"Falha ao conectar ao Qdrant após {max_retries} tentativas: {e}")
                        raise
        
        QdrantClient.__init__ = patched_init
        
        logger.info("Patch de conexão Qdrant aplicado")
        return True
        
    except ImportError:
        logger.warning("Qdrant client não está disponível para patch")
        return False
    except Exception as e:
        logger.error(f"Erro ao aplicar patch Qdrant: {e}")
        return False

def apply_embedding_dimension_patch():
    """Aplica patch para corrigir problemas de dimensão de embeddings."""
    try:
        # Patch para garantir que as dimensões de embedding sejam consistentes
        import numpy as np
        
        def normalize_embedding_dimensions(embeddings, target_dim=1536):
            """Normaliza dimensões de embeddings para o tamanho esperado."""
            if isinstance(embeddings, list):
                embeddings = np.array(embeddings)
            
            current_dim = embeddings.shape[-1] if embeddings.ndim > 0 else 0
            
            if current_dim == target_dim:
                return embeddings
            elif current_dim < target_dim:
                # Pad com zeros
                padding = target_dim - current_dim
                if embeddings.ndim == 1:
                    return np.pad(embeddings, (0, padding), mode='constant')
                else:
                    return np.pad(embeddings, ((0, 0), (0, padding)), mode='constant')
            else:
                # Truncar
                if embeddings.ndim == 1:
                    return embeddings[:target_dim]
                else:
                    return embeddings[:, :target_dim]
        
        # Disponibilizar função globalmente
        globals()['normalize_embedding_dimensions'] = normalize_embedding_dimensions
        
        logger.info("Patch de dimensão de embeddings aplicado")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao aplicar patch de embeddings: {e}")
        return False

def apply_all_patches():
    """Aplica todos os patches disponíveis."""
    patches_applied = []
    
    patches = [
        ('LiteLLM', apply_litellm_patch),
        ('Cognee Logging', apply_cognee_logging_patch),
        ('Neo4j Connection', apply_neo4j_connection_patch),
        ('Qdrant Connection', apply_qdrant_connection_patch),
        ('Embedding Dimensions', apply_embedding_dimension_patch)
    ]
    
    for patch_name, patch_func in patches:
        try:
            if patch_func():
                patches_applied.append(patch_name)
                logger.info(f"✅ Patch {patch_name} aplicado com sucesso")
            else:
                logger.warning(f"⚠️ Patch {patch_name} não pôde ser aplicado")
        except Exception as e:
            logger.error(f"❌ Erro ao aplicar patch {patch_name}: {e}")
    
    logger.info(f"Patches aplicados: {', '.join(patches_applied)}")
    return patches_applied

def get_patch_status():
    """Retorna o status dos patches aplicados."""
    status = {
        'litellm_available': False,
        'cognee_available': False,
        'neo4j_available': False,
        'qdrant_available': False,
        'patches_applied': []
    }
    
    try:
        import litellm
        status['litellm_available'] = True
    except ImportError:
        pass
    
    try:
        import cognee
        status['cognee_available'] = True
    except ImportError:
        pass
    
    try:
        from neo4j import GraphDatabase
        status['neo4j_available'] = True
    except ImportError:
        pass
    
    try:
        from qdrant_client import QdrantClient
        status['qdrant_available'] = True
    except ImportError:
        pass
    
    return status

if __name__ == "__main__":
    print("Aplicando patches do Cognee...")
    
    # Mostrar status antes dos patches
    status_before = get_patch_status()
    print("\nStatus antes dos patches:")
    for key, value in status_before.items():
        if key != 'patches_applied':
            icon = "✅" if value else "❌"
            print(f"  {icon} {key}: {value}")
    
    # Aplicar todos os patches
    patches_applied = apply_all_patches()
    
    print(f"\n✅ {len(patches_applied)} patches aplicados com sucesso")
    
    if patches_applied:
        print("\nPatches aplicados:")
        for patch in patches_applied:
            print(f"  ✅ {patch}")
    else:
        print("\n⚠️ Nenhum patch foi aplicado")
