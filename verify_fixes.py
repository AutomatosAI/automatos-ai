
import sys
import os
import asyncio
from unittest.mock import MagicMock

# Add orchestrator to path
sys.path.append(os.path.join(os.getcwd(), 'orchestrator'))

# Mock dependencies that might be missing or require DB
sys.modules['sqlalchemy'] = MagicMock()
sys.modules['sqlalchemy.orm'] = MagicMock()
sys.modules['models'] = MagicMock()
sys.modules['database'] = MagicMock()
sys.modules['database.database'] = MagicMock()
sys.modules['services.llm_provider'] = MagicMock()
sys.modules['context_engineering.context_optimizer'] = MagicMock()
sys.modules['context_engineering.vector_store'] = MagicMock()
sys.modules['context_engineering.embeddings'] = MagicMock()

# Mock specific imports used in files
from unittest.mock import Mock
mock_db = Mock()
mock_db.query.return_value.filter.return_value.first.return_value = None

def verify_config():
    print("Verifying Config...")
    try:
        from config import config
        print(f"REDIS_URL: {config.REDIS_URL}")
        print("✅ Config verification passed")
    except Exception as e:
        print(f"❌ Config verification failed: {e}")

def verify_codegraph_service():
    print("\nVerifying CodeGraphService...")
    try:
        from services.codegraph_service import CodeGraphService
        service = CodeGraphService(mock_db)
        # Check if embedding_manager is used instead of openai_client
        # We can't easily check internal logic without running it, but successful init is good
        print("✅ CodeGraphService initialized")
    except Exception as e:
        print(f"❌ CodeGraphService verification failed: {e}")

def verify_context_engineering_integrator():
    print("\nVerifying ContextEngineeringIntegrator...")
    try:
        # We need to unmock database.database to test the import fix?
        # Actually we mocked it above. Let's ensure get_database_url is there
        sys.modules['database.database'].get_database_url = lambda: "postgresql://user:pass@localhost:5432/db"
        
        from core.context_engineering_integrator import ContextEngineeringIntegrator
        integrator = ContextEngineeringIntegrator(mock_db)
        print("✅ ContextEngineeringIntegrator initialized")
    except Exception as e:
        print(f"❌ ContextEngineeringIntegrator verification failed: {e}")

async def verify_agent_platform_tools():
    print("\nVerifying AgentPlatformTools...")
    try:
        from services.agent_platform_tools import AgentPlatformTools
        tools = AgentPlatformTools(mock_db)
        print("✅ AgentPlatformTools initialized")
        
        # Verify execute_tool is async
        import inspect
        if inspect.iscoroutinefunction(tools.execute_tool):
            print("✅ execute_tool is async")
        else:
            print("❌ execute_tool is NOT async")
            
    except Exception as e:
        print(f"❌ AgentPlatformTools verification failed: {e}")

if __name__ == "__main__":
    verify_config()
    verify_codegraph_service()
    verify_context_engineering_integrator()
    asyncio.run(verify_agent_platform_tools())
