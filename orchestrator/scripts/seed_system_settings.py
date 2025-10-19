#!/usr/bin/env python3
"""
Seed System Settings from Environment Variables
===============================================

This script migrates .env configuration to the system_settings table.
Run this once to initialize the system settings database.
"""

import os
import sys
from sqlalchemy.orm import Session
from database.database import SessionLocal
from models.system_settings import SystemSetting, SettingCategory

def create_system_settings():
    """Create system settings from environment variables"""
    
    # Define all system settings with their categories and metadata
    settings_config = [
        # General Settings
        {
            "category": SettingCategory.GENERAL,
            "key": "environment",
            "value": os.getenv("ENVIRONMENT", "development"),
            "description": "Application environment (development, staging, production)",
            "is_required": True,
            "default_value": "development"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "log_level",
            "value": os.getenv("LOG_LEVEL", "INFO"),
            "description": "System logging level",
            "is_required": False,
            "default_value": "INFO"
        },
        
        # Orchestrator LLM Settings
        {
            "category": SettingCategory.ORCHESTRATOR_LLM,
            "key": "llm_provider",
            "value": os.getenv("LLM_PROVIDER", "openai"),
            "description": "Default LLM provider for orchestrator",
            "is_required": True,
            "default_value": "openai"
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM,
            "key": "llm_model",
            "value": os.getenv("LLM_MODEL", "gpt-4"),
            "description": "Default LLM model",
            "is_required": True,
            "default_value": "gpt-4"
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM,
            "key": "llm_temperature",
            "value": os.getenv("LLM_TEMPERATURE", "0.7"),
            "description": "LLM temperature setting",
            "is_required": False,
            "default_value": "0.7"
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM,
            "key": "llm_max_tokens",
            "value": os.getenv("LLM_MAX_TOKENS", "2000"),
            "description": "Maximum tokens for LLM responses",
            "is_required": False,
            "default_value": "2000"
        },
        
        # CodeGraph Settings
        {
            "category": SettingCategory.CODEGRAPH,
            "key": "enabled",
            "value": os.getenv("CODEGRAPH_ENABLED", "true"),
            "description": "Enable CodeGraph analysis",
            "is_required": False,
            "default_value": "true"
        },
        {
            "category": SettingCategory.CODEGRAPH,
            "key": "max_file_size",
            "value": os.getenv("CODEGRAPH_MAX_FILE_SIZE", "1000000"),
            "description": "Maximum file size for analysis (bytes)",
            "is_required": False,
            "default_value": "1000000"
        },
        {
            "category": SettingCategory.CODEGRAPH,
            "key": "supported_languages",
            "value": os.getenv("CODEGRAPH_SUPPORTED_LANGUAGES", "python,typescript,javascript,go,rust"),
            "description": "Supported programming languages",
            "is_required": False,
            "default_value": "python,typescript,javascript,go,rust"
        },
        {
            "category": SettingCategory.CODEGRAPH,
            "key": "cache_ttl",
            "value": os.getenv("CODEGRAPH_CACHE_TTL", "3600"),
            "description": "Cache TTL in seconds",
            "is_required": False,
            "default_value": "3600"
        },
        
        # API Rate Limiting Settings
        {
            "category": SettingCategory.API_RATE_LIMITING,
            "key": "requests_per_window",
            "value": os.getenv("RATE_LIMIT_REQUESTS", "100"),
            "description": "Maximum requests per time window",
            "is_required": False,
            "default_value": "100"
        },
        {
            "category": SettingCategory.API_RATE_LIMITING,
            "key": "window_seconds",
            "value": os.getenv("RATE_LIMIT_WINDOW", "60"),
            "description": "Rate limiting time window in seconds",
            "is_required": False,
            "default_value": "60"
        },
        
        # Backend API Keys
        {
            "category": SettingCategory.BACKEND_API_KEYS,
            "key": "api_key",
            "value": os.getenv("API_KEY", "test_api_key_for_backend_validation_2025"),
            "description": "Backend API authentication key",
            "is_sensitive": True,
            "is_required": True,
            "default_value": "test_api_key_for_backend_validation_2025"
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS,
            "key": "api_port",
            "value": os.getenv("API_PORT", "8000"),
            "description": "Backend API port",
            "is_required": True,
            "default_value": "8000"
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS,
            "key": "api_url",
            "value": os.getenv("API_URL", "api.automatos.app"),
            "description": "Backend API URL",
            "is_required": True,
            "default_value": "api.automatos.app"
        },
        
        # ML/AI Model Configuration
        {
            "category": SettingCategory.GENERAL,
            "key": "embedding_model",
            "value": os.getenv("EMBEDDING_MODEL", "disabled"),
            "description": "Embedding model configuration",
            "is_required": False,
            "default_value": "disabled"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "embedding_cache_dir",
            "value": os.getenv("EMBEDDING_CACHE_DIR", "./model_cache"),
            "description": "Directory for embedding model cache",
            "is_required": False,
            "default_value": "./model_cache"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "embedding_max_seq_length",
            "value": os.getenv("EMBEDDING_MAX_SEQ_LENGTH", "256"),
            "description": "Maximum sequence length for embeddings",
            "is_required": False,
            "default_value": "256"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "openai_embedding_model",
            "value": os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            "description": "OpenAI embedding model",
            "is_required": False,
            "default_value": "text-embedding-3-small"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "vector_store_type",
            "value": os.getenv("VECTOR_STORE_TYPE", "faiss"),
            "description": "Vector store type",
            "is_required": False,
            "default_value": "faiss"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "vector_store_dimensions",
            "value": os.getenv("VECTOR_STORE_DIMENSIONS", "384"),
            "description": "Vector store dimensions",
            "is_required": False,
            "default_value": "384"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "vector_store_index_type",
            "value": os.getenv("VECTOR_STORE_INDEX_TYPE", "IndexFlatIP"),
            "description": "Vector store index type",
            "is_required": False,
            "default_value": "IndexFlatIP"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "chunk_size",
            "value": os.getenv("CHUNK_SIZE", "512"),
            "description": "Text chunk size for processing",
            "is_required": False,
            "default_value": "512"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "chunk_overlap",
            "value": os.getenv("CHUNK_OVERLAP", "50"),
            "description": "Text chunk overlap",
            "is_required": False,
            "default_value": "50"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "max_context_length",
            "value": os.getenv("MAX_CONTEXT_LENGTH", "4000"),
            "description": "Maximum context length",
            "is_required": False,
            "default_value": "4000"
        },
        
        # SSH Deployment Settings
        {
            "category": SettingCategory.GENERAL,
            "key": "deploy_host",
            "value": os.getenv("DEPLOY_HOST", "mcp.automatos.app"),
            "description": "Deployment host",
            "is_required": False,
            "default_value": "mcp.automatos.app"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "deploy_port",
            "value": os.getenv("DEPLOY_PORT", "22"),
            "description": "Deployment SSH port",
            "is_required": False,
            "default_value": "22"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "deploy_user",
            "value": os.getenv("DEPLOY_USER", "root"),
            "description": "Deployment SSH user",
            "is_required": False,
            "default_value": "root"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "deploy_key_path",
            "value": os.getenv("DEPLOY_KEY_PATH", "/aRpp/keys/deploy_key"),
            "description": "Deployment SSH key path",
            "is_required": False,
            "default_value": "/aRpp/keys/deploy_key"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "deploy_enabled",
            "value": os.getenv("DEPLOY", "false"),
            "description": "Enable deployment",
            "is_required": False,
            "default_value": "false"
        },
        
        # NextJS Frontend Settings
        {
            "category": SettingCategory.GENERAL,
            "key": "nextauth_secret",
            "value": os.getenv("NEXTAUTH_SECRET", "your-nextauth-secret-here"),
            "description": "NextAuth secret key",
            "is_sensitive": True,
            "is_required": False,
            "default_value": "your-nextauth-secret-here"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "nextauth_url",
            "value": os.getenv("NEXTAUTH_URL", "https://8018adcb5.preview.abacusai.app"),
            "description": "NextAuth URL",
            "is_required": False,
            "default_value": "https://8018adcb5.preview.abacusai.app"
        },
        {
            "category": SettingCategory.GENERAL,
            "key": "next_public_api_url",
            "value": os.getenv("NEXT_PUBLIC_API_URL", "https://api.automatos.app"),
            "description": "Public API URL for frontend",
            "is_required": False,
            "default_value": "https://api.automatos.app"
        }
    ]
    
    db = SessionLocal()
    try:
        created_count = 0
        updated_count = 0
        
        for config in settings_config:
            # Check if setting already exists
            existing = db.query(SystemSetting).filter(
                SystemSetting.category == config["category"],
                SystemSetting.key == config["key"]
            ).first()
            
            if existing:
                # Update existing setting
                existing.value = config["value"]
                existing.description = config["description"]
                existing.is_sensitive = config.get("is_sensitive", False)
                existing.is_required = config.get("is_required", False)
                existing.default_value = config.get("default_value")
                updated_count += 1
            else:
                # Create new setting
                setting = SystemSetting(**config)
                db.add(setting)
                created_count += 1
        
        db.commit()
        print(f"✅ System settings initialized:")
        print(f"   Created: {created_count} settings")
        print(f"   Updated: {updated_count} settings")
        print(f"   Total: {created_count + updated_count} settings")
        
    except Exception as e:
        db.rollback()
        print(f"❌ Error initializing system settings: {e}")
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    print("🚀 Initializing system settings from environment variables...")
    create_system_settings()
    print("✅ System settings initialization complete!")
