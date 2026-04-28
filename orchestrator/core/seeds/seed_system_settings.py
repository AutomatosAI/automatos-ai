"""
Seed System Settings
===================

Seeds default system settings for all configuration categories.
PRD-26: System Settings Configuration - Comprehensive settings seed.

CRITICAL: This script NEVER overwrites existing values.
- If setting exists: Only updates default_value if changed, preserves existing value
- If setting doesn't exist: Creates with default_value as initial value
"""

import logging
from sqlalchemy.orm import Session

from config import config
from core.models.system_settings import SystemSetting, SettingCategory

logger = logging.getLogger(__name__)


def seed_system_settings(db: Session):
    """
    Seed default system settings for all configuration categories.
    
    Creates settings for:
    - general: General system settings (environment, logging, embeddings, deployment, frontend)
    - orchestrator_llm: Orchestrator LLM configuration
    - codegraph: CodeGraph LLM and embedding configuration
    - system_logging: Logging service configuration (if implemented)
    - api_rate_limiting: Rate limiting configuration (if implemented)
    - backend_api_keys: API keys configuration
    
    CRITICAL: Never overwrites existing user values. Only sets value for new settings.
    """
    
    settings_to_create = [
        # ========================================
        # GENERAL SETTINGS
        # ========================================
        
        # Environment Configuration
        {
            "category": SettingCategory.GENERAL.value,
            "key": "environment",
            "default_value": "development",
            "value_type": "string",
            "description": "Application environment (development, staging, production)",
            "is_required": True,
            "validation_rules": {
                "options": ["development", "staging", "production"]
            }
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "log_level",
            "default_value": "INFO",
            "value_type": "string",
            "description": "Python logging level",
            "is_required": False,
            "validation_rules": {
                "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            }
        },
        
        # ML/AI Model Configuration
        {
            "category": SettingCategory.GENERAL.value,
            "key": "embedding_model",
            "default_value": "qwen/qwen3-embedding-8b",
            "value_type": "string",
            "description": "Embedding model name (used by the selected embedding_provider)",
            "is_required": False
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "embedding_provider",
            "default_value": "openrouter",
            "value_type": "string",
            "description": "Embedding provider (openrouter, openai, google, cohere, huggingface_local, huggingface_api, disabled)",
            "is_required": False,
            "validation_rules": {
                "options": ["openrouter", "openai", "google", "cohere", "huggingface_local", "huggingface_api", "disabled"]
            }
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "embedding_cache_dir",
            "default_value": "./model_cache",
            "value_type": "string",
            "description": "Directory for embedding model cache",
            "is_required": False
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "embedding_max_seq_length",
            "default_value": "256",
            "value_type": "number",
            "description": "Maximum sequence length for embeddings",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 8192
            }
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "openai_embedding_model",
            "default_value": "qwen/qwen3-embedding-8b",
            "value_type": "string",
            "description": "Embedding model name (via configured provider)",
            "is_required": False,
            "validation_rules": {
                "options": [
                    "qwen/qwen3-embedding-8b",
                    "text-embedding-3-small",
                    "text-embedding-3-large",
                    "text-embedding-ada-002"
                ]
            }
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "vector_store_type",
            "default_value": "faiss",
            "value_type": "string",
            "description": "Vector store type",
            "is_required": False,
            "validation_rules": {
                "options": ["faiss", "chroma", "pinecone", "weaviate"]
            }
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "vector_store_dimensions",
            "default_value": "2048",
            "value_type": "number",
            "description": "Vector embedding dimensions (Qwen3-8B Matryoshka truncated to 2048)",
            "is_required": False,
            "validation_rules": {
                "min": 128,
                "max": 4096
            }
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "chunk_size",
            "default_value": "512",
            "value_type": "number",
            "description": "Document chunk size for embeddings",
            "is_required": False,
            "validation_rules": {
                "min": 64,
                "max": 4096
            }
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "chunk_overlap",
            "default_value": "50",
            "value_type": "number",
            "description": "Chunk overlap size",
            "is_required": False,
            "validation_rules": {
                "min": 0,
                "max": 256
            }
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "max_context_length",
            "default_value": "4000",
            "value_type": "number",
            "description": "Maximum context length",
            "is_required": False,
            "validation_rules": {
                "min": 512,
                "max": 128000
            }
        },
        
        # Reranking Configuration
        {
            "category": SettingCategory.GENERAL.value,
            "key": "rag_rerank_enabled",
            "default_value": "false",
            "value_type": "boolean",
            "description": "Enable Cohere reranking after vector search (improves precision 15-30%)",
            "is_required": False,
            "validation_rules": {
                "options": ["true", "false"]
            }
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "rag_rerank_model",
            "default_value": "rerank-v3.5",
            "value_type": "string",
            "description": "Cohere rerank model",
            "is_required": False,
            "validation_rules": {
                "options": ["rerank-v3.5", "rerank-english-v3.0", "rerank-multilingual-v3.0"]
            }
        },

        # Deployment Configuration (if deployment service exists)
        {
            "category": SettingCategory.GENERAL.value,
            "key": "deploy_host",
            "default_value": "",
            "value_type": "string",
            "description": "Deployment host (SSH)",
            "is_required": False
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "deploy_port",
            "default_value": "22",
            "value_type": "number",
            "description": "Deployment SSH port",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 65535
            }
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "deploy_user",
            "default_value": "root",
            "value_type": "string",
            "description": "Deployment SSH user",
            "is_required": False
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "deploy_key_path",
            "default_value": "",
            "value_type": "string",
            "description": "Path to deployment SSH key",
            "is_required": False,
            "is_sensitive": True
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "deploy_enabled",
            "default_value": "false",
            "value_type": "boolean",
            "description": "Enable automated deployment",
            "is_required": False,
            "validation_rules": {
                "options": ["true", "false"]
            }
        },
        
        # Frontend Configuration
        {
            "category": SettingCategory.GENERAL.value,
            "key": "nextauth_secret",
            "default_value": "",
            "value_type": "string",
            "description": "NextAuth secret key",
            "is_required": False,
            "is_sensitive": True
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "nextauth_url",
            "default_value": "",
            "value_type": "string",
            "description": "NextAuth callback URL",
            "is_required": False
        },
        {
            "category": SettingCategory.GENERAL.value,
            "key": "next_public_api_url",
            "default_value": config.NEXT_PUBLIC_API_URL or "",
            "value_type": "string",
            "description": "Public API URL for frontend",
            "is_required": True
        },
        # ========================================
        # ORCHESTRATOR LLM SETTINGS
        # ========================================
        
        # LLM Provider Configuration
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "provider",
            "default_value": "openrouter",
            "value_type": "string",
            "description": "LLM provider for orchestrator operations. OpenRouter recommended — single key for 100+ models.",
            "is_required": True,
            "validation_rules": {
                "options": ["openrouter", "openai", "anthropic", "google", "azure", "huggingface"]
            }
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "model",
            "default_value": "google/gemini-2.5-flash",
            "value_type": "string",
            "description": "LLM model for orchestrator operations. Use OpenRouter format: provider/model (e.g. google/gemini-2.5-flash).",
            "is_required": True,
            "validation_rules": {
                "depends_on": {"provider": "..."}
            }
        },
        
        # LLM Parameters
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "temperature",
            "default_value": "0.7",
            "value_type": "number",
            "description": "Temperature for LLM responses (0.0-2.0)",
            "is_required": False,
            "validation_rules": {
                "min": 0.0,
                "max": 2.0,
                "step": 0.1
            }
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "max_tokens",
            "default_value": "2000",
            "value_type": "number",
            "description": "Maximum tokens in LLM response",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 32000
            }
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "top_p",
            "default_value": "1.0",
            "value_type": "number",
            "description": "Top-p sampling parameter (0.0-1.0)",
            "is_required": False,
            "validation_rules": {
                "min": 0.0,
                "max": 1.0,
                "step": 0.1
            }
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "frequency_penalty",
            "default_value": "0.0",
            "value_type": "number",
            "description": "Frequency penalty (-2.0 to 2.0)",
            "is_required": False,
            "validation_rules": {
                "min": -2.0,
                "max": 2.0,
                "step": 0.1
            }
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "presence_penalty",
            "default_value": "0.0",
            "value_type": "number",
            "description": "Presence penalty (-2.0 to 2.0)",
            "is_required": False,
            "validation_rules": {
                "min": -2.0,
                "max": 2.0,
                "step": 0.1
            }
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "stop_sequences",
            "default_value": "",
            "value_type": "string",
            "description": "Comma-separated stop sequences",
            "is_required": False
        },
        
        # Performance Settings
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "timeout_seconds",
            "default_value": "30",
            "value_type": "number",
            "description": "Request timeout in seconds",
            "is_required": False,
            "validation_rules": {
                "min": 5,
                "max": 300
            }
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "retry_attempts",
            "default_value": "3",
            "value_type": "number",
            "description": "Number of retry attempts on failure",
            "is_required": False,
            "validation_rules": {
                "min": 0,
                "max": 5
            }
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "concurrent_requests",
            "default_value": "5",
            "value_type": "number",
            "description": "Maximum concurrent LLM requests",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 10
            }
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "cache_ttl",
            "default_value": "300",
            "value_type": "number",
            "description": "Cache TTL in seconds",
            "is_required": False,
            "validation_rules": {
                "min": 0,
                "max": 3600
            }
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "streaming_enabled",
            "default_value": "false",
            "value_type": "boolean",
            "description": "Enable streaming responses",
            "is_required": False,
            "validation_rules": {
                "options": ["true", "false"]
            }
        },
        
        # MVP: Credential Name Mappings (explicit credential name per provider)
        # These override the automatic credential name resolution
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "credential_name_openai",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for OpenAI (e.g., 'development_openai'). Leave empty for automatic resolution.",
            "is_required": False
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "credential_name_anthropic",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for Anthropic (e.g., 'development_anthropic'). Leave empty for automatic resolution.",
            "is_required": False
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "credential_name_google",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for Google (e.g., 'development_google'). Leave empty for automatic resolution.",
            "is_required": False
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "credential_name_azure",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for Azure OpenAI (e.g., 'development_azure'). Leave empty for automatic resolution.",
            "is_required": False
        },
        {
            "category": SettingCategory.ORCHESTRATOR_LLM.value,
            "key": "credential_name_huggingface",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for HuggingFace (e.g., 'development_huggingface'). Leave empty for automatic resolution.",
            "is_required": False
        },
        
        # ========================================
        # CHATBOT SETTINGS
        # ========================================
        
        # LLM Provider Configuration for Chatbot
        {
            "category": SettingCategory.CHATBOT.value,
            "key": "provider",
            "default_value": "huggingface",
            "value_type": "string",
            "description": "LLM provider for chatbot",
            "is_required": True,
            "validation_rules": {
                "options": ["openai", "anthropic", "google", "azure", "huggingface", "grok"]
            }
        },
        {
            "category": SettingCategory.CHATBOT.value,
            "key": "model",
            "default_value": "meta-llama/Llama-3.2-3B-Instruct",
            "value_type": "string",
            "description": "LLM model for chatbot",
            "is_required": True
        },
        {
            "category": SettingCategory.CHATBOT.value,
            "key": "temperature",
            "default_value": "0.7",
            "value_type": "number",
            "description": "Temperature for chatbot responses (0.0-2.0)",
            "is_required": False,
            "validation_rules": {
                "min": 0.0,
                "max": 2.0,
                "step": 0.1
            }
        },
        {
            "category": SettingCategory.CHATBOT.value,
            "key": "max_tokens",
            "default_value": "2000",
            "value_type": "number",
            "description": "Maximum tokens in chatbot response",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 32000
            }
        },
        {
            "category": SettingCategory.CHATBOT.value,
            "key": "credential_name_huggingface",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for HuggingFace. Leave empty for automatic resolution.",
            "is_required": False
        },
        {
            "category": SettingCategory.CHATBOT.value,
            "key": "credential_name_openai",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for OpenAI. Leave empty for automatic resolution.",
            "is_required": False
        },
        {
            "category": SettingCategory.CHATBOT.value,
            "key": "credential_name_grok",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for xAI/Grok. Leave empty for automatic resolution.",
            "is_required": False
        },
        
        # ========================================
        # CODEGRAPH SETTINGS
        # ========================================
        
        # LLM Provider Configuration
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "provider",
            "default_value": "openai",
            "value_type": "string",
            "description": "LLM provider for CodeGraph operations",
            "is_required": True,
            "validation_rules": {
                "options": ["openai", "anthropic", "google", "azure", "huggingface"]
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "model",
            "default_value": "gpt-3.5-turbo",
            "value_type": "string",
            "description": "LLM model for CodeGraph operations",
            "is_required": True,
            "validation_rules": {
                "depends_on": {"provider": "..."}
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "temperature",
            "default_value": "0.7",
            "value_type": "number",
            "description": "Temperature for CodeGraph LLM responses (0.0-2.0)",
            "is_required": False,
            "validation_rules": {
                "min": 0.0,
                "max": 2.0,
                "step": 0.1
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "max_tokens",
            "default_value": "2000",
            "value_type": "number",
            "description": "Maximum tokens in CodeGraph LLM response",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 32000
            }
        },
        
        # Embedding Model Configuration (inherits from general settings by default)
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "embedding_model",
            "default_value": "qwen/qwen3-embedding-8b",
            "value_type": "string",
            "description": "Embedding model for CodeGraph semantic search",
            "is_required": False,
            "validation_rules": {
                "options": [
                    "qwen/qwen3-embedding-8b",
                    "text-embedding-3-small",
                    "text-embedding-3-large"
                ]
            }
        },
        
        # CodeGraph Configuration
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "enabled",
            "default_value": "true",
            "value_type": "boolean",
            "description": "Enable CodeGraph analysis",
            "is_required": False,
            "validation_rules": {
                "options": ["true", "false"]
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "max_file_size",
            "default_value": "1000000",
            "value_type": "number",
            "description": "Maximum file size for analysis (bytes)",
            "is_required": False,
            "validation_rules": {
                "min": 1024,
                "max": 100000000
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "cache_ttl",
            "default_value": "3600",
            "value_type": "number",
            "description": "Cache TTL for analysis results (seconds)",
            "is_required": False,
            "validation_rules": {
                "min": 0,
                "max": 86400
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "supported_languages",
            "default_value": "python,typescript,javascript,go,rust",
            "value_type": "string",
            "description": "Comma-separated list of supported languages",
            "is_required": False
        },
        
        # Analysis Settings
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "max_depth",
            "default_value": "5",
            "value_type": "number",
            "description": "Maximum depth for symbol analysis",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 10
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "max_nodes",
            "default_value": "1000",
            "value_type": "number",
            "description": "Maximum graph nodes",
            "is_required": False,
            "validation_rules": {
                "min": 100,
                "max": 10000
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "analysis_timeout",
            "default_value": "60",
            "value_type": "number",
            "description": "Analysis timeout (seconds)",
            "is_required": False,
            "validation_rules": {
                "min": 10,
                "max": 300
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "concurrent_analyses",
            "default_value": "3",
            "value_type": "number",
            "description": "Maximum concurrent analyses",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 10
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "semantic_search_limit",
            "default_value": "10",
            "value_type": "number",
            "description": "Default limit for semantic search results",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 100
            }
        },
        
        # Performance Settings
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "memory_limit",
            "default_value": "512",
            "value_type": "number",
            "description": "Memory limit for CodeGraph (MB)",
            "is_required": False,
            "validation_rules": {
                "min": 100,
                "max": 2048
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "cpu_limit",
            "default_value": "50",
            "value_type": "number",
            "description": "CPU limit for CodeGraph (%)",
            "is_required": False,
            "validation_rules": {
                "min": 10,
                "max": 100
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "batch_size",
            "default_value": "10",
            "value_type": "number",
            "description": "Batch size for processing",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 100
            }
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "cleanup_interval",
            "default_value": "24",
            "value_type": "number",
            "description": "Cleanup interval (hours)",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 168
            }
        },
        
        # MVP: Credential Name Mappings for CodeGraph (explicit credential name per provider)
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "credential_name_openai",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for OpenAI (e.g., 'development_openai'). Leave empty for automatic resolution.",
            "is_required": False
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "credential_name_anthropic",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for Anthropic (e.g., 'development_anthropic'). Leave empty for automatic resolution.",
            "is_required": False
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "credential_name_google",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for Google (e.g., 'development_google'). Leave empty for automatic resolution.",
            "is_required": False
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "credential_name_azure",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for Azure OpenAI (e.g., 'development_azure'). Leave empty for automatic resolution.",
            "is_required": False
        },
        {
            "category": SettingCategory.CODEGRAPH.value,
            "key": "credential_name_huggingface",
            "default_value": "",
            "value_type": "string",
            "description": "Explicit credential name for HuggingFace (e.g., 'development_huggingface'). Leave empty for automatic resolution.",
            "is_required": False
        },
        
        # ========================================
        # SYSTEM LOGGING SETTINGS (if logging service exists)
        # ========================================
        
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_level",
            "default_value": "INFO",
            "value_type": "string",
            "description": "System logging level",
            "is_required": False,
            "validation_rules": {
                "options": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            }
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_format",
            "default_value": "json",
            "value_type": "string",
            "description": "Log format (json, text, structured)",
            "is_required": False,
            "validation_rules": {
                "options": ["json", "text", "structured"]
            }
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_file_path",
            "default_value": "/var/log/automatos/app.log",
            "value_type": "string",
            "description": "Path to log file",
            "is_required": False
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_max_size",
            "default_value": "100",
            "value_type": "number",
            "description": "Maximum log file size (MB)",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 1000
            }
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_backup_count",
            "default_value": "7",
            "value_type": "number",
            "description": "Number of backup log files to keep",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 50
            }
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_rotation_interval",
            "default_value": "24",
            "value_type": "number",
            "description": "Log rotation interval (hours)",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 168
            }
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_retention_days",
            "default_value": "30",
            "value_type": "number",
            "description": "Log retention period (days)",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 365
            }
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_compress",
            "default_value": "false",
            "value_type": "boolean",
            "description": "Enable log file compression",
            "is_required": False,
            "validation_rules": {
                "options": ["true", "false"]
            }
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_buffer_size",
            "default_value": "64",
            "value_type": "number",
            "description": "Log buffer size (KB)",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 1024
            }
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_flush_interval",
            "default_value": "5",
            "value_type": "number",
            "description": "Log flush interval (seconds)",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 60
            }
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_async",
            "default_value": "false",
            "value_type": "boolean",
            "description": "Enable async logging",
            "is_required": False,
            "validation_rules": {
                "options": ["true", "false"]
            }
        },
        {
            "category": SettingCategory.SYSTEM_LOGGING.value,
            "key": "log_queue_size",
            "default_value": "1000",
            "value_type": "number",
            "description": "Log queue size for async logging",
            "is_required": False,
            "validation_rules": {
                "min": 100,
                "max": 10000
            }
        },
        
        # ========================================
        # API RATE LIMITING SETTINGS (if rate limiting exists)
        # ========================================
        
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "enabled",
            "default_value": "true",
            "value_type": "boolean",
            "description": "Enable API rate limiting",
            "is_required": False,
            "validation_rules": {
                "options": ["true", "false"]
            }
        },
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "requests_per_window",
            "default_value": "100",
            "value_type": "number",
            "description": "Maximum requests per time window",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 10000
            }
        },
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "window_seconds",
            "default_value": "60",
            "value_type": "number",
            "description": "Rate limiting time window (seconds)",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 3600
            }
        },
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "burst_limit",
            "default_value": "200",
            "value_type": "number",
            "description": "Maximum burst requests",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 1000
            }
        },
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "recovery_time",
            "default_value": "300",
            "value_type": "number",
            "description": "Time to recover from rate limit (seconds)",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 3600
            }
        },
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "skip_successful_requests",
            "default_value": "false",
            "value_type": "boolean",
            "description": "Don't count successful requests in rate limit",
            "is_required": False,
            "validation_rules": {
                "options": ["true", "false"]
            }
        },
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "skip_failed_requests",
            "default_value": "false",
            "value_type": "boolean",
            "description": "Don't count failed requests in rate limit",
            "is_required": False,
            "validation_rules": {
                "options": ["true", "false"]
            }
        },
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "storage_backend",
            "default_value": "redis",
            "value_type": "string",
            "description": "Storage backend for rate limiting (redis, memory)",
            "is_required": False,
            "validation_rules": {
                "options": ["redis", "memory"]
            }
        },
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "cleanup_interval",
            "default_value": "300",
            "value_type": "number",
            "description": "Cleanup interval for rate limit keys (seconds)",
            "is_required": False,
            "validation_rules": {
                "min": 60,
                "max": 3600
            }
        },
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "max_keys",
            "default_value": "100000",
            "value_type": "number",
            "description": "Maximum number of rate limit keys to store",
            "is_required": False,
            "validation_rules": {
                "min": 1000,
                "max": 1000000
            }
        },
        {
            "category": SettingCategory.API_RATE_LIMITING.value,
            "key": "key_expiry",
            "default_value": "3600",
            "value_type": "number",
            "description": "Rate limit key expiry time (seconds)",
            "is_required": False,
            "validation_rules": {
                "min": 60,
                "max": 86400
            }
        },
        
        # ========================================
        # BACKEND API KEYS SETTINGS
        # ========================================
        
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "cohere_api_key",
            "default_value": "",
            "value_type": "string",
            "description": "Cohere API key for reranking (rerank-v3.5). Get one at dashboard.cohere.com",
            "is_required": False,
            "is_sensitive": True
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "api_key",
            "default_value": "",
            "value_type": "string",
            "description": "Backend API key for authentication",
            "is_required": False,
            "is_sensitive": True
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "api_port",
            "default_value": "8000",
            "value_type": "number",
            "description": "Backend API port",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 65535
            }
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "api_url",
            "default_value": "",
            "value_type": "string",
            "description": "Backend API URL",
            "is_required": False
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "api_key_length",
            "default_value": "32",
            "value_type": "number",
            "description": "API key length for generated keys",
            "is_required": False,
            "validation_rules": {
                "min": 16,
                "max": 128
            }
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "api_key_expiry",
            "default_value": "90",
            "value_type": "number",
            "description": "API key expiry period (days)",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 365
            }
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "max_api_keys_per_user",
            "default_value": "5",
            "value_type": "number",
            "description": "Maximum API keys per user",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 10
            }
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "api_key_prefix",
            "default_value": "ak_",
            "value_type": "string",
            "description": "Prefix for generated API keys",
            "is_required": False
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "api_timeout",
            "default_value": "30",
            "value_type": "number",
            "description": "API request timeout (seconds)",
            "is_required": False,
            "validation_rules": {
                "min": 5,
                "max": 300
            }
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "api_retry_attempts",
            "default_value": "3",
            "value_type": "number",
            "description": "API retry attempts on failure",
            "is_required": False,
            "validation_rules": {
                "min": 0,
                "max": 5
            }
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "api_rate_limit_per_key",
            "default_value": "1000",
            "value_type": "number",
            "description": "Rate limit per API key (requests per hour)",
            "is_required": False,
            "validation_rules": {
                "min": 1,
                "max": 10000
            }
        },
        {
            "category": SettingCategory.BACKEND_API_KEYS.value,
            "key": "api_monitoring_enabled",
            "default_value": "false",
            "value_type": "boolean",
            "description": "Enable API monitoring",
            "is_required": False,
            "validation_rules": {
                "options": ["true", "false"]
            }
        },

        # ========================================
        # PRD-68: COMPLEXITY ASSESSOR SETTINGS
        # ========================================

        {
            "category": SettingCategory.COMPLEXITY_ASSESSOR.value,
            "key": "provider",
            "default_value": "openrouter",
            "value_type": "string",
            "description": "LLM provider for complexity assessment routing (PRD-68). Use a fast, cheap model.",
            "is_required": True,
            "validation_rules": {
                "options": ["openai", "anthropic", "google", "openrouter", "huggingface", "grok"]
            }
        },
        {
            "category": SettingCategory.COMPLEXITY_ASSESSOR.value,
            "key": "model",
            "default_value": "meta-llama/llama-3.1-8b-instruct",
            "value_type": "string",
            "description": "Model for complexity assessment. Recommend: lightweight model (Llama 8B, Haiku, Flash).",
            "is_required": True
        },
        {
            "category": SettingCategory.COMPLEXITY_ASSESSOR.value,
            "key": "temperature",
            "default_value": "0.1",
            "value_type": "number",
            "description": "Low temperature for consistent routing decisions (0.0-1.0)",
            "is_required": False,
            "validation_rules": {
                "min": 0.0,
                "max": 1.0
            }
        },
        {
            "category": SettingCategory.COMPLEXITY_ASSESSOR.value,
            "key": "max_tokens",
            "default_value": "200",
            "value_type": "number",
            "description": "Max tokens for routing response (JSON output only, keep low)",
            "is_required": False,
            "validation_rules": {
                "min": 50,
                "max": 500
            }
        },

        # ========================================
        # COORDINATION SETTINGS (Mission planner, verifier, reconciler)
        # ========================================
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "provider",
            "default_value": "openrouter",
            "value_type": "string",
            "description": "LLM provider for coordination services (planner, verifier)",
            "is_required": True,
            "validation_rules": {
                "options": ["openrouter", "openai", "anthropic", "google", "deepseek"]
            }
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "model",
            "default_value": "openai/gpt-4o-mini",
            "value_type": "string",
            "description": "LLM model for mission planning/decomposition",
            "is_required": True,
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "planner_max_tokens",
            "default_value": "4000",
            "value_type": "number",
            "description": "Max tokens for planner LLM output",
            "is_required": False,
            "validation_rules": {"min": 500, "max": 32000}
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "planner_temperature",
            "default_value": "0.4",
            "value_type": "number",
            "description": "Temperature for planner (low = more deterministic plans)",
            "is_required": False,
            "validation_rules": {"min": 0.0, "max": 1.0, "step": 0.1}
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "task_max_tokens",
            "default_value": "4000",
            "value_type": "number",
            "description": "Max tokens for mission task agent execution (overrides agent default during missions)",
            "is_required": False,
            "validation_rules": {"min": 500, "max": 16000}
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "verifier_model_mapping",
            "default_value": "anthropic=openai/gpt-4o-mini,openai=anthropic/claude-haiku-4-5,google=openai/gpt-4o-mini,deepseek=openai/gpt-4o-mini,meta=openai/gpt-4o-mini",
            "value_type": "string",
            "description": "Cross-model verifier mapping (family=model pairs, comma-separated)",
            "is_required": False,
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "verifier_fallback_model",
            "default_value": "openai/gpt-4o-mini",
            "value_type": "string",
            "description": "Fallback model for verification when no mapping matches",
            "is_required": True,
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "verifier_max_tokens",
            "default_value": "2000",
            "value_type": "number",
            "description": "Max tokens for verification LLM output",
            "is_required": False,
            "validation_rules": {"min": 200, "max": 8000}
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "verification_pass_threshold",
            "default_value": "0.7",
            "value_type": "number",
            "description": "Score threshold for PASS verdict (above = pass)",
            "is_required": False,
            "validation_rules": {"min": 0.0, "max": 1.0, "step": 0.05}
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "verification_catastrophic_threshold",
            "default_value": "0.15",
            "value_type": "number",
            "description": "Score below this flags for human review (NOT auto-retry, just alert)",
            "is_required": False,
            "validation_rules": {"min": 0.0, "max": 0.5, "step": 0.05}
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "max_plan_retries",
            "default_value": "3",
            "value_type": "number",
            "description": "Max retries for plan validation failures",
            "is_required": False,
            "validation_rules": {"min": 1, "max": 5}
        },
        {
            "category": SettingCategory.COORDINATION.value,
            "key": "consistency_check_enabled",
            "default_value": "true",
            "value_type": "boolean",
            "description": "Enable cross-task consistency check (adds 1 extra LLM call per mission)",
            "is_required": False,
        },

        # ========================================
        # KNOWLEDGE GRAPH SETTINGS (Graph extraction LLM)
        # ========================================
        {
            "category": SettingCategory.KNOWLEDGE_GRAPH.value,
            "key": "provider",
            "default_value": "openrouter",
            "value_type": "string",
            "description": "LLM provider for knowledge graph extraction",
            "is_required": True,
            "validation_rules": {
                "options": ["openrouter", "openai", "anthropic", "google"]
            }
        },
        {
            "category": SettingCategory.KNOWLEDGE_GRAPH.value,
            "key": "model",
            "default_value": "google/gemini-2.5-flash",
            "value_type": "string",
            "description": "LLM model for knowledge graph entity/relation extraction",
            "is_required": True,
        },
        {
            "category": SettingCategory.KNOWLEDGE_GRAPH.value,
            "key": "max_tokens",
            "default_value": "8000",
            "value_type": "number",
            "description": "Max tokens for graph extraction LLM output (read by core.llm.manager)",
            "is_required": False,
            "validation_rules": {"min": 500, "max": 16000}
        },
        {
            "category": SettingCategory.KNOWLEDGE_GRAPH.value,
            "key": "temperature",
            "default_value": "0.1",
            "value_type": "number",
            "description": "Temperature for extraction (low = more consistent entities)",
            "is_required": False,
            "validation_rules": {"min": 0.0, "max": 1.0, "step": 0.1}
        },
        {
            "category": SettingCategory.KNOWLEDGE_GRAPH.value,
            "key": "max_concurrent_extractions",
            "default_value": "5",
            "value_type": "number",
            "description": "Max concurrent document extractions",
            "is_required": False,
            "validation_rules": {"min": 1, "max": 20}
        },

        # ========================================
        # LLM COST AUDIT SETTINGS (Budget alerts, tracking)
        # ========================================
        {
            "category": SettingCategory.LLM_COST_AUDIT.value,
            "key": "enabled",
            "default_value": "true",
            "value_type": "boolean",
            "description": "Enable LLM cost tracking and audit logging",
            "is_required": False,
        },
        {
            "category": SettingCategory.LLM_COST_AUDIT.value,
            "key": "mission_budget_alert_usd",
            "default_value": "2.00",
            "value_type": "number",
            "description": "Alert threshold per mission (USD). Logs WARNING when exceeded.",
            "is_required": False,
            "validation_rules": {"min": 0.10, "max": 100.00, "step": 0.50}
        },
        {
            "category": SettingCategory.LLM_COST_AUDIT.value,
            "key": "daily_budget_alert_usd",
            "default_value": "20.00",
            "value_type": "number",
            "description": "Alert threshold per day (USD). Logs CRITICAL when exceeded.",
            "is_required": False,
            "validation_rules": {"min": 1.00, "max": 500.00, "step": 1.00}
        },
        {
            "category": SettingCategory.LLM_COST_AUDIT.value,
            "key": "log_every_call",
            "default_value": "true",
            "value_type": "boolean",
            "description": "Log model, tokens, and estimated cost for every LLM call",
            "is_required": False,
        },

        # ── Memory Management ──────────────────────────────────────
        # Storage limits
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "store_max_chars",
            "default_value": "6000",
            "value_type": "number",
            "description": "Max characters per message (user + assistant) stored in Mem0. Higher = richer recall, more tokens per save.",
            "is_required": False,
            "validation_rules": {"min": 500, "max": 20000, "step": 500},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "daily_log_max_chars",
            "default_value": "2000",
            "value_type": "number",
            "description": "Max characters for daily activity log entries.",
            "is_required": False,
            "validation_rules": {"min": 500, "max": 10000, "step": 500},
        },
        # Context injection budgets (tokens)
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "context_budget_total",
            "default_value": "4000",
            "value_type": "number",
            "description": "Total token budget for memory context injected into prompts.",
            "is_required": False,
            "validation_rules": {"min": 1000, "max": 16000, "step": 500},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "context_budget_session",
            "default_value": "500",
            "value_type": "number",
            "description": "Token budget for L1 session summary in prompts.",
            "is_required": False,
            "validation_rules": {"min": 100, "max": 4000, "step": 100},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "context_budget_long_term",
            "default_value": "800",
            "value_type": "number",
            "description": "Token budget for L3 long-term memories in prompts.",
            "is_required": False,
            "validation_rules": {"min": 200, "max": 4000, "step": 100},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "context_budget_temporal",
            "default_value": "600",
            "value_type": "number",
            "description": "Token budget for L2 temporal/short-term memories in prompts.",
            "is_required": False,
            "validation_rules": {"min": 200, "max": 4000, "step": 100},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "context_budget_daily",
            "default_value": "400",
            "value_type": "number",
            "description": "Token budget for daily activity logs in prompts.",
            "is_required": False,
            "validation_rules": {"min": 100, "max": 2000, "step": 100},
        },
        # Retrieval limits
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "search_result_limit",
            "default_value": "8",
            "value_type": "number",
            "description": "Max number of memories returned per search query.",
            "is_required": False,
            "validation_rules": {"min": 1, "max": 50, "step": 1},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "long_term_search_limit",
            "default_value": "5",
            "value_type": "number",
            "description": "Max Mem0 memories fetched for context injection.",
            "is_required": False,
            "validation_rules": {"min": 1, "max": 20, "step": 1},
        },
        # Circuit breaker
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "circuit_breaker_threshold",
            "default_value": "5",
            "value_type": "number",
            "description": "Consecutive Mem0 failures before circuit breaker opens.",
            "is_required": False,
            "validation_rules": {"min": 2, "max": 20, "step": 1},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "circuit_breaker_cooldown",
            "default_value": "60",
            "value_type": "number",
            "description": "Seconds circuit breaker stays open before retrying Mem0.",
            "is_required": False,
            "validation_rules": {"min": 10, "max": 600, "step": 10},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "request_timeout",
            "default_value": "15",
            "value_type": "number",
            "description": "Timeout (seconds) for individual Mem0 API calls.",
            "is_required": False,
            "validation_rules": {"min": 5, "max": 60, "step": 5},
        },
        # Cache TTLs
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "cache_ttl",
            "default_value": "300",
            "value_type": "number",
            "description": "Seconds to cache Mem0 search results (avoids repeated queries).",
            "is_required": False,
            "validation_rules": {"min": 30, "max": 1800, "step": 30},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "session_ttl",
            "default_value": "86400",
            "value_type": "number",
            "description": "Seconds before L1 session memory expires in Redis (default 24h).",
            "is_required": False,
            "validation_rules": {"min": 3600, "max": 604800, "step": 3600},
        },
        # Decay & promotion
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "decay_rate",
            "default_value": "0.1",
            "value_type": "number",
            "description": "Memory importance decay rate per cycle (0.0-1.0).",
            "is_required": False,
            "validation_rules": {"min": 0.01, "max": 0.5, "step": 0.01},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "promotion_min_importance",
            "default_value": "0.7",
            "value_type": "number",
            "description": "Minimum importance score for L2 to L3 promotion.",
            "is_required": False,
            "validation_rules": {"min": 0.3, "max": 1.0, "step": 0.05},
        },
        {
            "category": SettingCategory.MEMORY_MANAGEMENT.value,
            "key": "promotion_min_access_count",
            "default_value": "3",
            "value_type": "number",
            "description": "Minimum access count before a memory can be promoted to L3.",
            "is_required": False,
            "validation_rules": {"min": 1, "max": 20, "step": 1},
        },
    ]
    
    # ── One-time cleanup: fix stale orchestrator_llm values ──────────
    # The env vars LLM_PROVIDER=openai / LLM_MODEL=openai/gpt-5.4 were
    # captured into system_settings before they were deleted from Railway.
    # Reset to the correct defaults so config.LLM_PROVIDER returns the right value.
    from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
    _STALE_FIXES = {
        ("orchestrator_llm", "provider"): DEFAULT_LLM_PROVIDER,
        ("orchestrator_llm", "model"): DEFAULT_LLM_MODEL,
    }
    for (cat, key), correct_value in _STALE_FIXES.items():
        stale = db.query(SystemSetting).filter(
            SystemSetting.category == cat,
            SystemSetting.key == key,
        ).first()
        if stale and stale.value and stale.value != correct_value:
            old_val = stale.value
            stale.value = correct_value
            stale.default_value = correct_value
            logger.info(
                "Fixed stale system_setting %s.%s: '%s' → '%s'",
                cat, key, old_val, correct_value,
            )

    created_count = 0
    updated_count = 0
    preserved_count = 0

    for setting_data in settings_to_create:
        # Check if setting already exists
        existing = db.query(SystemSetting).filter(
            SystemSetting.category == setting_data["category"],
            SystemSetting.key == setting_data["key"]
        ).first()
        
        if existing:
            # CRITICAL: Never overwrite existing value - preserve user's setting
            # Only update default_value and metadata if changed
            metadata_changed = False
            
            # Update default_value if different
            if existing.default_value != setting_data.get("default_value"):
                existing.default_value = setting_data.get("default_value")
                metadata_changed = True
            
            # Update description if different
            if existing.description != setting_data.get("description"):
                existing.description = setting_data.get("description")
                metadata_changed = True
            
            # Update validation_rules if different
            existing_vrules = existing.validation_rules or {}
            new_vrules = setting_data.get("validation_rules") or {}
            if existing_vrules != new_vrules:
                existing.validation_rules = new_vrules
                metadata_changed = True
            
            # Update is_required if different
            if existing.is_required != setting_data.get("is_required", False):
                existing.is_required = setting_data.get("is_required", False)
                metadata_changed = True
            
            # Update is_sensitive if different
            if existing.is_sensitive != setting_data.get("is_sensitive", False):
                existing.is_sensitive = setting_data.get("is_sensitive", False)
                metadata_changed = True
            
            if metadata_changed:
                updated_count += 1
                logger.debug(f"Updated metadata for {setting_data['category']}.{setting_data['key']}")
            else:
                preserved_count += 1
                logger.debug(f"Preserved existing setting {setting_data['category']}.{setting_data['key']}")
            
            # NEVER set existing.value = default_value - preserve user's choice!
            
        else:
            # Create new setting - set value to default_value initially
            # User can change it later via UI
            setting = SystemSetting(
                category=setting_data["category"],
                key=setting_data["key"],
                value=setting_data.get("default_value"),  # Initial value = default
                default_value=setting_data.get("default_value"),
                value_type=setting_data.get("value_type", "string"),
                description=setting_data.get("description"),
                is_sensitive=setting_data.get("is_sensitive", False),
                is_required=setting_data.get("is_required", False),
                validation_rules=setting_data.get("validation_rules"),
                created_by="system"
            )
            db.add(setting)
            created_count += 1
            logger.debug(f"Created setting {setting_data['category']}.{setting_data['key']}")
    
    try:
        db.commit()
        logger.info(
            f"✅ Seeded system settings: {created_count} created, {updated_count} metadata updated, "
            f"{preserved_count} preserved (user values not overwritten)"
        )
        return created_count, updated_count
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to seed system settings: {e}")
        raise
