#!/usr/bin/env python3
"""
Test Configuration Setup
=========================

Run this to verify all environment variables are properly configured
"""

import sys
from pathlib import Path

# Add orchestrator to path
sys.path.append(str(Path(__file__).parent / "orchestrator"))

def test_config():
    """Test configuration is properly loaded"""
    
    print("🔧 TESTING CONFIGURATION")
    print("=" * 60)
    
    try:
        from orchestrator.config import config
        
        # Print configuration
        config.print_config(show_secrets=False)
        
        print("\n📋 Configuration Check:")
        print(f"✅ Environment: {config.ENVIRONMENT}")
        print(f"✅ API Base URL: {config.API_BASE_URL}")
        print(f"✅ Database URL: postgresql://{config.POSTGRES_USER}@{config.POSTGRES_HOST}/{config.POSTGRES_DB}")
        print(f"✅ Redis URL: redis://{config.REDIS_HOST}:{config.REDIS_PORT}")
        
        # Check critical configurations
        print("\n🔍 Critical Checks:")
        
        if config.OPENAI_API_KEY:
            print(f"✅ OpenAI API Key is set ({len(config.OPENAI_API_KEY)} chars)")
        else:
            print("❌ OpenAI API Key is NOT set")
        
        if config.ANTHROPIC_API_KEY:
            print(f"✅ Anthropic API Key is set ({len(config.ANTHROPIC_API_KEY)} chars)")
        else:
            print("⚠️  Anthropic API Key is NOT set (optional)")
        
        if config.API_KEY:
            print(f"✅ API Key for authentication is set")
        else:
            print("⚠️  API Key is NOT set (authentication may fail)")
        
        # Validate configuration
        print("\n🎯 Validation Result:")
        if config.validate():
            print("✅ Configuration is VALID")
        else:
            print("❌ Configuration is INVALID - check errors above")
            return False
        
        # Test API URL construction
        print("\n🌐 API URLs:")
        print(f"Base URL: {config.API_BASE_URL}")
        print(f"Task Analyze: {config.API_BASE_URL}/api/orchestrator/task/analyze")
        print(f"Agent List: {config.API_BASE_URL}/api/agents")
        
        # Show feature flags
        print("\n🚩 Feature Flags:")
        print(f"Mock Data Enabled: {config.ENABLE_MOCK_DATA} {'⚠️ WARNING!' if config.ENABLE_MOCK_DATA else '✅'}")
        print(f"Batch API Enabled: {config.ENABLE_BATCH_API}")
        print(f"API Key Required: {config.REQUIRE_API_KEY}")
        
        print("\n✅ Configuration test PASSED!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import config: {e}")
        print("\nMake sure you're running from the automatos-ai directory")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_config()
    sys.exit(0 if success else 1)
