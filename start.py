#!/usr/bin/env python3
"""
Script de inicialização do projeto LangGraph

Este script facilita o setup e execução do projeto em diferentes modos:
- Desenvolvimento local
- Produção
- Testes
- Docker

Uso:
    python start.py [modo] [opções]
    
Modos disponíveis:
    dev         - Modo desenvolvimento (padrão)
    prod        - Modo produção
    test        - Executar testes
    docker      - Iniciar com Docker
    setup       - Configurar ambiente
"""

import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LangGraphStarter:
    """Classe para gerenciar a inicialização do projeto"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / ".env"
        self.env_example = self.project_root / ".env.example"
    
    def check_python_version(self):
        """Verifica se a versão do Python é compatível"""
        if sys.version_info < (3, 9):
            logger.error("Python 3.9+ é necessário. Versão atual: %s", sys.version)
            sys.exit(1)
        logger.info("Python %s.%s.%s detectado ✓", *sys.version_info[:3])
    
    def check_dependencies(self):
        """Verifica se as dependências estão instaladas"""
        requirements_file = self.project_root / "requirements.txt"
        if not requirements_file.exists():
            logger.error("Arquivo requirements.txt não encontrado")
            return False
        
        try:
            import fastapi
            import uvicorn
            import redis
            import cognee
            logger.info("Dependências principais encontradas ✓")
            return True
        except ImportError as e:
            logger.warning("Dependência não encontrada: %s", e)
            return False
    
    def setup_environment(self):
        """Configura o ambiente de desenvolvimento"""
        logger.info("Configurando ambiente...")
        
        # Verificar se .env existe
        if not self.env_file.exists():
            if self.env_example.exists():
                logger.info("Copiando .env.example para .env")
                import shutil
                shutil.copy(self.env_example, self.env_file)
                logger.warning("⚠️  Configure as variáveis em .env antes de continuar")
            else:
                logger.error("Arquivo .env.example não encontrado")
                return False
        
        # Instalar dependências se necessário
        if not self.check_dependencies():
            logger.info("Instalando dependências...")
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
                ], check=True)
                logger.info("Dependências instaladas ✓")
            except subprocess.CalledProcessError:
                logger.error("Erro ao instalar dependências")
                return False
        
        # Criar diretórios necessários
        directories = ["logs", "data", "tests"]
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.info("Diretório %s criado ✓", directory)
        
        return True
    
    def run_development(self, host="127.0.0.1", port=8000, reload=True):
        """Executa o servidor em modo desenvolvimento"""
        logger.info("Iniciando servidor de desenvolvimento...")
        
        # Configurar variáveis de ambiente para desenvolvimento
        env = os.environ.copy()
        env.update({
            "DISABLE_EXTERNAL_DBS": "true",
            "ALLOW_START_WITHOUT_DBS": "1",
            "LOG_LEVEL": "DEBUG"
        })
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", host,
            "--port", str(port),
            "--log-level", "info"
        ]
        
        if reload:
            cmd.append("--reload")
        
        try:
            subprocess.run(cmd, env=env, cwd=self.project_root)
        except KeyboardInterrupt:
            logger.info("Servidor interrompido pelo usuário")
        except Exception as e:
            logger.error("Erro ao executar servidor: %s", e)
    
    def run_production(self, host="0.0.0.0", port=8000, workers=4):
        """Executa o servidor em modo produção"""
        logger.info("Iniciando servidor de produção...")
        
        cmd = [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", host,
            "--port", str(port),
            "--workers", str(workers),
            "--log-level", "warning"
        ]
        
        try:
            subprocess.run(cmd, cwd=self.project_root)
        except KeyboardInterrupt:
            logger.info("Servidor interrompido pelo usuário")
        except Exception as e:
            logger.error("Erro ao executar servidor: %s", e)
    
    def run_tests(self, coverage=False, verbose=False):
        """Executa os testes"""
        logger.info("Executando testes...")
        
        cmd = [sys.executable, "-m", "pytest"]
        
        if verbose:
            cmd.append("-v")
        
        if coverage:
            cmd.extend(["--cov=api", "--cov=agents", "--cov-report=html"])
        
        try:
            result = subprocess.run(cmd, cwd=self.project_root)
            if result.returncode == 0:
                logger.info("Todos os testes passaram ✓")
            else:
                logger.error("Alguns testes falharam")
            return result.returncode == 0
        except Exception as e:
            logger.error("Erro ao executar testes: %s", e)
            return False
    
    def run_docker(self, build=False, detach=False):
        """Executa o projeto com Docker"""
        logger.info("Iniciando com Docker...")
        
        if build:
            logger.info("Construindo imagens Docker...")
            try:
                subprocess.run(["docker-compose", "build"], check=True, cwd=self.project_root)
            except subprocess.CalledProcessError:
                logger.error("Erro ao construir imagens Docker")
                return False
        
        cmd = ["docker-compose", "up"]
        if detach:
            cmd.append("-d")
        
        try:
            subprocess.run(cmd, cwd=self.project_root)
        except KeyboardInterrupt:
            logger.info("Docker interrompido pelo usuário")
            subprocess.run(["docker-compose", "down"], cwd=self.project_root)
        except Exception as e:
            logger.error("Erro ao executar Docker: %s", e)
    
    def show_status(self):
        """Mostra o status do projeto"""
        logger.info("Status do Projeto LangGraph")
        logger.info("=" * 40)
        
        # Verificar arquivos importantes
        files_to_check = [
            ".env", "requirements.txt", "api/main.py", 
            "agents/simple_agent.py", "cognee_config.py"
        ]
        
        for file_path in files_to_check:
            full_path = self.project_root / file_path
            status = "✓" if full_path.exists() else "✗"
            logger.info("%s %s", status, file_path)
        
        # Verificar dependências
        deps_ok = self.check_dependencies()
        logger.info("%s Dependências", "✓" if deps_ok else "✗")
        
        # Verificar Docker
        try:
            subprocess.run(["docker", "--version"], 
                         capture_output=True, check=True)
            logger.info("✓ Docker disponível")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.info("✗ Docker não disponível")

def main():
    """Função principal"""
    parser = argparse.ArgumentParser(
        description="Script de inicialização do LangGraph",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python start.py                    # Modo desenvolvimento
  python start.py dev --port 8001    # Desenvolvimento na porta 8001
  python start.py prod --workers 8   # Produção com 8 workers
  python start.py test --coverage    # Testes com cobertura
  python start.py docker --build     # Docker com rebuild
  python start.py setup              # Configurar ambiente
        """
    )
    
    parser.add_argument(
        "mode", 
        nargs="?", 
        default="dev",
        choices=["dev", "prod", "test", "docker", "setup", "status"],
        help="Modo de execução (padrão: dev)"
    )
    
    # Argumentos para modo dev/prod
    parser.add_argument("--host", default="127.0.0.1", help="Host do servidor")
    parser.add_argument("--port", type=int, default=8000, help="Porta do servidor")
    parser.add_argument("--workers", type=int, default=4, help="Número de workers (prod)")
    parser.add_argument("--no-reload", action="store_true", help="Desabilitar reload (dev)")
    
    # Argumentos para testes
    parser.add_argument("--coverage", action="store_true", help="Executar com cobertura")
    parser.add_argument("--verbose", action="store_true", help="Saída verbosa")
    
    # Argumentos para Docker
    parser.add_argument("--build", action="store_true", help="Rebuild das imagens")
    parser.add_argument("--detach", action="store_true", help="Executar em background")
    
    args = parser.parse_args()
    
    starter = LangGraphStarter()
    starter.check_python_version()
    
    if args.mode == "setup":
        success = starter.setup_environment()
        sys.exit(0 if success else 1)
    
    elif args.mode == "status":
        starter.show_status()
    
    elif args.mode == "dev":
        if not starter.setup_environment():
            sys.exit(1)
        starter.run_development(
            host=args.host, 
            port=args.port, 
            reload=not args.no_reload
        )
    
    elif args.mode == "prod":
        starter.run_production(
            host=args.host, 
            port=args.port, 
            workers=args.workers
        )
    
    elif args.mode == "test":
        success = starter.run_tests(
            coverage=args.coverage, 
            verbose=args.verbose
        )
        sys.exit(0 if success else 1)
    
    elif args.mode == "docker":
        starter.run_docker(
            build=args.build, 
            detach=args.detach
        )

if __name__ == "__main__":
    main()
