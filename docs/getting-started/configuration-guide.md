# Configuration Guide

<details>
<summary>Relevant source files</summary>

The following files were used as context for generating this wiki page:

- [docker-compose.yml](docker-compose.yml)
- [frontend/.dockerignore](frontend/.dockerignore)
- [frontend/Dockerfile](frontend/Dockerfile)
- [frontend/app/admin/plugins/page.tsx](frontend/app/admin/plugins/page.tsx)
- [frontend/lib/api-client.ts](frontend/lib/api-client.ts)
- [orchestrator/.env.example](orchestrator/.env.example)
- [orchestrator/Dockerfile](orchestrator/Dockerfile)
- [orchestrator/api/agent_plugins.py](orchestrator/api/agent_plugins.py)
- [orchestrator/config.py](orchestrator/config.py)
- [orchestrator/core/database/load_seed_data.py](orchestrator/core/database/load_seed_data.py)
- [orchestrator/core/redis/client.py](orchestrator/core/redis/client.py)
- [orchestrator/core/seeds/seed_personas.py](orchestrator/core/seeds/seed_personas.py)
- [orchestrator/core/seeds/seed_plugin_categories.py](orchestrator/core/seeds/seed_plugin_categories.py)
- [orchestrator/core/services/plugin_cache.py](orchestrator/core/services/plugin_cache.py)
- [orchestrator/main.py](orchestrator/main.py)
- [orchestrator/requirements.txt](orchestrator/requirements.txt)
- [scripts/ralph/prd.json](scripts/ralph/prd.json)

</details>



This document covers all configuration options for Automatos AI, including environment variables, database settings, LLM providers, AWS S3, feature flags, and credential management. For instructions on initial installation and setup, see [Installation & Setup](#2.1). For deployment-specific configuration, see [Deployment & Infrastructure](#12).

---

## Configuration Architecture

Automatos AI uses a centralized configuration system with a single source of truth for all environment variables. The `Config` class in [orchestrator/config.py:28-285]() is the **only location** where `os.getenv()` is called, ensuring consistent configuration access across the codebase.

### Configuration Loading Flow

```mermaid
graph TB
    EnvFile[".env file<br/>(root/.env)"]
    EnvVars["Environment Variables<br/>(Shell/Docker)"]
    ConfigClass["Config class<br/>(config.py:28)"]
    Validation["Config.validate()<br/>(config.py:225)"]
    Services["Core Services<br/>(AgentFactory, LLM Manager, etc.)"]
    
    EnvFile -->|"load_dotenv()<br/>(config.py:26)"| ConfigClass
    EnvVars --> ConfigClass
    ConfigClass --> Validation
    Validation -->|"pass"| Services
    Validation -->|"fail"| Error["ConfigurationError<br/>Logged + Exit"]
    
    ConfigClass -->|"property access"| Dynamic["Dynamic Settings<br/>(LLM_PROVIDER, LLM_MODEL)"]
    Dynamic -->|"fallback"| Database["system_settings table<br/>(PostgreSQL)"]
```

**Sources:** [orchestrator/config.py:1-286]()

### Configuration Precedence

Configuration values are resolved in the following order (highest to lowest priority):

1. **Explicit environment variables** - Set in shell or Docker container
2. **`.env` file** - Loaded via `load_dotenv()` at [orchestrator/config.py:26]()
3. **Database system settings** - For dynamic settings like `LLM_PROVIDER` and `LLM_MODEL` ([config.py:89-106]())
4. **Default values** - Hardcoded defaults in property getters

**Sources:** [orchestrator/config.py:24-26](), [orchestrator/config.py:89-106]()

---

## Database Configuration

### PostgreSQL Settings

Automatos AI requires PostgreSQL 14+ with the `pgvector` extension for vector similarity search.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `POSTGRES_DB` | **Yes** | - | Database name |
| `POSTGRES_USER` | **Yes** | - | Database username |
| `POSTGRES_PASSWORD` | **Yes** | - | Database password |
| `POSTGRES_HOST` | **Yes** | - | Database host |
| `POSTGRES_PORT` | **Yes** | - | Database port |
| `DATABASE_URL` | No | - | Complete connection string (overrides individual params) |

**Configuration Location:** [orchestrator/config.py:35-42]()

**Example `.env` Configuration:**

```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=orchestrator_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_database_password_here
```

**Sources:** [orchestrator/config.py:35-42](), [orchestrator/.env.example:1-6]()

### Connection String Format

If `DATABASE_URL` is provided, it takes precedence over individual parameters:

```bash
DATABASE_URL=postgresql://user:password@host:port/database
```

**Validation:** The `Config.validate()` method ([config.py:225-247]()) checks for required database parameters at startup and logs errors if missing.

**Sources:** [orchestrator/config.py:42](), [orchestrator/config.py:233-234]()

---

## Redis Configuration

Redis is used for caching and pub/sub messaging. It is **optional** but highly recommended for production deployments.

### Redis Connection Parameters

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `REDIS_HOST` | No | - | Redis server host |
| `REDIS_PORT` | No | - | Redis server port |
| `REDIS_PASSWORD` | No | - | Redis authentication password |
| `REDIS_URL` | No | - | Complete Redis URL (overrides individual params) |

**Configuration Location:** [orchestrator/config.py:45-62]()

### Redis URL Construction

The `REDIS_URL` property dynamically constructs the connection URL if individual parameters are provided:

```python
# Property getter at config.py:52-62
@property
def REDIS_URL(self) -> str:
    url = os.getenv("REDIS_URL")
    if url:
        return url
    
    if self.REDIS_HOST and self.REDIS_PORT:
        auth = f":{self.REDIS_PASSWORD}@" if self.REDIS_PASSWORD else ""
        return f"redis://{auth}{self.REDIS_HOST}:{self.REDIS_PORT}/0"
    
    return None
```

**Example `.env` Configuration:**

```bash
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=your_redis_password_here
```

**Sources:** [orchestrator/config.py:45-62](), [orchestrator/.env.example:8-11]()

### Redis Usage in Services

```mermaid
graph LR
    Config["Config.REDIS_URL<br/>(config.py:52)"]
    RedisClient["RedisClient<br/>(core/redis/client.py)"]
    PluginCache["PluginContentCache<br/>(plugin_cache.py:22)"]
    PubSub["Workflow Pub/Sub<br/>(Real-time updates)"]
    
    Config --> RedisClient
    RedisClient --> PluginCache
    RedisClient --> PubSub
    
    PluginCache -->|"TTL: PLUGIN_CACHE_TTL_SECONDS"| CachedContent["Cached Plugin Content"]
```

**Sources:** [orchestrator/config.py:45-62](), [orchestrator/core/services/plugin_cache.py:42-47]()

---

## API Security

### API Key Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `API_KEY` | Conditional | - | Master API key for backend authentication |
| `REQUIRE_API_KEY` | No | `"true"` | Whether API key validation is enforced |

**Configuration Location:** [orchestrator/config.py:65-68]()

**Authentication Modes:**

The backend supports **hybrid authentication** via the `get_request_context_hybrid` dependency:

1. **Clerk JWT** - Primary authentication for logged-in users
2. **API Key** - Programmatic access via `x-api-key` header
3. **Anonymous Fallback** - Development mode when `REQUIRE_API_KEY=false`

**Validation Logic:**

```python
# config.py:237-238
if self.REQUIRE_API_KEY and not self.API_KEY:
    errors.append("API_KEY required when REQUIRE_API_KEY=true")
```

**Sources:** [orchestrator/config.py:65-68](), [orchestrator/config.py:237-238]()

### CORS Configuration

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CORS_ALLOW_ORIGINS` | No | `"http://localhost:3000,https://..."` | Comma-separated list of allowed origins |

**Configuration Location:** [orchestrator/config.py:71-79]()

**Example:**

```bash
CORS_ALLOW_ORIGINS=http://localhost:3000,https://ui.automatos.app
```

The configuration automatically strips whitespace and filters empty entries at [config.py:79]().

**Sources:** [orchestrator/config.py:71-79]()

---

## LLM Provider Configuration

### Provider API Keys

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OPENAI_API_KEY` | Conditional | - | OpenAI API key (required if using OpenAI) |
| `ANTHROPIC_API_KEY` | Conditional | - | Anthropic API key (required if using Claude) |

**Configuration Location:** [orchestrator/config.py:82-85]()

### Dynamic LLM Settings

The `LLM_PROVIDER` and `LLM_MODEL` settings are **dynamic properties** that load from the database first, falling back to environment variables:

```mermaid
graph TB
    Request["Service requests LLM config"]
    Property["Config.LLM_PROVIDER<br/>(config.py:89)"]
    Database["system_settings table"]
    EnvVar["LLM_PROVIDER env var"]
    Fallback["No default - must be set"]
    
    Request --> Property
    Property -->|"try"| GetSetting["get_system_setting()<br/>(llm/manager.py)"]
    GetSetting -->|"found"| Database
    GetSetting -->|"not found"| EnvVar
    EnvVar -->|"not set"| Fallback
```

**Property Implementation:**

```python
# config.py:88-96
@property
def LLM_PROVIDER(self) -> str:
    try:
        from core.llm.manager import get_system_setting
        return get_system_setting("orchestrator_llm", "provider", os.getenv("LLM_PROVIDER"))
    except Exception:
        return os.getenv("LLM_PROVIDER")  # No hardcoded default
```

**Sources:** [orchestrator/config.py:88-106]()

### LLM Model Parameters

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_PROVIDER` | **Yes** | - | Provider name: `"openai"` or `"anthropic"` |
| `LLM_MODEL` | **Yes** | - | Model name (e.g., `"gpt-4"`, `"claude-3-opus"`) |
| `LLM_TEMPERATURE` | No | `0.7` | Sampling temperature (0.0-1.0) |
| `LLM_MAX_TOKENS` | No | `2000` | Maximum tokens per response |

**Configuration Location:** [orchestrator/config.py:82-109]()

**Example Configuration:**

```bash
# OpenAI Configuration
LLM_PROVIDER=openai
LLM_MODEL=gpt-4
OPENAI_API_KEY=sk-...

# Anthropic Configuration
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-opus-20240229
ANTHROPIC_API_KEY=sk-ant-...
```

**Sources:** [orchestrator/config.py:82-109](), [orchestrator/.env.example:18-26]()

---

## AWS S3 Configuration

### Core S3 Settings

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `AWS_ACCESS_KEY_ID` | Conditional | - | AWS access key (required if using S3) |
| `AWS_SECRET_ACCESS_KEY` | Conditional | - | AWS secret key (required if using S3) |
| `AWS_REGION` | No | `"us-east-1"` | AWS region |

**Configuration Location:** [orchestrator/config.py:159-161](), [config.py:180-182]()

### Marketplace S3 (Plugin Storage)

The plugin marketplace uses S3 for storing plugin packages, manifests, and security scan results.

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `MARKETPLACE_S3_BUCKET` | No | `"automatos-marketplace"` | S3 bucket for plugin storage |
| `PLUGIN_MAX_UPLOAD_SIZE_MB` | No | `10` | Maximum plugin package size (MB) |
| `PLUGIN_LLM_SCAN_MODEL` | No | `"claude-haiku-4-20250414"` | Model for security scanning |
| `PLUGIN_CACHE_TTL_SECONDS` | No | `3600` | Plugin content cache TTL (1 hour) |

**Configuration Location:** [orchestrator/config.py:176-185]()

**S3 Path Structure:**

```
automatos-marketplace/
├── plugins/
│   ├── {slug}/
│   │   ├── {version}/
│   │   │   ├── manifest.json
│   │   │   ├── SKILL.md
│   │   │   └── ...other files
```

**Sources:** [orchestrator/config.py:176-185](), [orchestrator/.env.example:46-53]()

### Plugin Caching Configuration

The `PluginContentCache` service ([plugin_cache.py:22]()) uses Redis to cache S3 content with configurable TTL:

```python
# plugin_cache.py:42-47
def __init__(self, s3_service=None):
    try:
        from config import config
        self._ttl = config.PLUGIN_CACHE_TTL_SECONDS
    except Exception:
        self._ttl = 3600  # 1 hour default
```

**Cache Key Prefixes:**
- `plugin_content:{slug}:{version}` - Complete plugin file tree
- `plugin_manifest:{slug}:{version}` - Parsed manifest.json
- `plugin_files:{slug}:{version}:{file_path}` - Individual files

**Sources:** [orchestrator/core/services/plugin_cache.py:22-47](), [orchestrator/config.py:185]()

### Vector Storage (S3 Vectors)

For cloud document synchronization and vector storage:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `S3_VECTORS_ENABLED` | No | `"false"` | Enable S3-backed vector storage |
| `S3_VECTORS_BUCKET` | Conditional | - | S3 bucket for vector index |
| `S3_VECTORS_INDEX_NAME` | No | `"documents-index"` | Index name |
| `S3_VECTORS_DIMENSION` | No | `1024` | Vector dimension |
| `S3_VECTORS_METRIC` | No | `"cosine"` | Distance metric |

**Configuration Location:** [orchestrator/config.py:163-168]()

**Sources:** [orchestrator/config.py:163-168](), [orchestrator/.env.example:59-63]()

---

## Feature Flags

Feature flags enable/disable specific system capabilities.

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_BATCH_API` | `"false"` | Enable batch API endpoints |
| `S3_VECTORS_ENABLED` | `"false"` | Enable S3-backed vector storage |
| `JIRA_BUG_REPORTS_ENABLED` | `"true"` | Enable Jira bug report widget |

**Configuration Location:** [orchestrator/config.py:154](), [config.py:164](), [config.py:174]()

**Usage Pattern:**

```python
from config import config

if config.ENABLE_BATCH_API:
    # Register batch endpoints
    pass
```

**Sources:** [orchestrator/config.py:154-174]()

---

## Environment Settings

### Environment Mode

| Variable | Default | Description |
|----------|---------|-------------|
| `ENVIRONMENT` | `"development"` | Environment mode: `"development"` or `"production"` |
| `LOG_LEVEL` | `"INFO"` | Logging level: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"` |

**Configuration Location:** [orchestrator/config.py:112-123]()

**Convenience Properties:**

```python
# config.py:117-123
@property
def IS_PRODUCTION(self) -> bool:
    return self.ENVIRONMENT.lower() == "production"

@property
def IS_DEVELOPMENT(self) -> bool:
    return self.ENVIRONMENT.lower() == "development"
```

**Sources:** [orchestrator/config.py:112-123]()

---

## Credential Management Configuration

The credential management system provides secure storage for API keys, database credentials, and OAuth tokens. For details on using credentials, see [Credentials Management](#9.4).

### Encryption Configuration

Credentials are encrypted using the `EncryptionService` with a key derived from the `ENCRYPTION_KEY` environment variable (not shown in config.py but required for production).

### Credential Store Architecture

```mermaid
graph TB
    Config["Configuration"]
    CredStore["CredentialStore<br/>(credentials/service.py:42)"]
    EncService["EncryptionService<br/>(credentials/encryption.py)"]
    Database["credentials table<br/>(models/credentials.py:60)"]
    AuditLog["credential_audit_logs table<br/>(models/credentials.py:105)"]
    
    Config -->|"ENCRYPTION_KEY"| EncService
    CredStore -->|"uses"| EncService
    CredStore -->|"reads/writes"| Database
    CredStore -->|"logs access"| AuditLog
    
    Database -->|"encrypted_data column"| Encrypted["Encrypted JSON Blob"]
    EncService -->|"decrypt"| Plaintext["Plaintext Credentials"]
```

**Sources:** [orchestrator/core/credentials/service.py:42-56](), [orchestrator/core/models/credentials.py:60-103]()

### Credential Types

Credential types define schemas for different credential kinds (PostgreSQL, OpenAI, SSH, etc.). The `credential_types` table ([credentials.py:25-57]()) stores these definitions:

| Column | Type | Description |
|--------|------|-------------|
| `name` | String | Unique identifier (e.g., `"postgres_credentials"`) |
| `display_name` | String | UI label (e.g., `"PostgreSQL"`) |
| `category` | String | Category: `"database"`, `"ai"`, `"infrastructure"`, `"api"` |
| `schema_definition` | JSON | Field definitions (names, types, required flags) |
| `test_endpoint` | JSON | How to test the credential |

**Sources:** [orchestrator/core/models/credentials.py:25-57]()

### Credential Storage

The `credentials` table ([credentials.py:60-102]()) stores encrypted credential instances:

| Column | Type | Description |
|--------|------|-------------|
| `encrypted_data` | Text | Encrypted JSON blob of field values |
| `environment` | String | Environment: `"dev"`, `"staging"`, `"production"` |
| `test_status` | String | Test status: `"passed"`, `"failed"`, `"not_tested"` |
| `expires_at` | DateTime | Optional expiration timestamp |

**Encryption Flow:**

```python
# credentials/service.py:143-148
encrypted_data = self.encryption_service.encrypt_dict(credential_data.credential_data)

credential = Credential(
    encrypted_data=encrypted_data,
    # ...other fields
)
```

**Sources:** [orchestrator/core/credentials/service.py:99-181](), [orchestrator/core/models/credentials.py:60-102]()

### Credential Testing

Credentials can be tested to verify they work correctly. The test configuration is stored in the `test_endpoint` field of the credential type:

```python
# credentials/service.py:504-562
async def test_credential(self, credential_id: int, user_id: Optional[str] = None):
    # Decrypt credential data
    decrypted_data = self.encryption_service.decrypt_dict(credential.encrypted_data)
    
    # Perform test based on credential type
    test_result = await self._perform_credential_test(cred_type, decrypted_data)
    
    # Update test status
    credential.test_status = 'passed' if test_result.success else 'failed'
```

**Sources:** [orchestrator/core/credentials/service.py:504-562]()

### Audit Logging

All credential access is logged to the `credential_audit_logs` table ([credentials.py:105-130]()) for security compliance:

**Logged Actions:**
- `created` - Credential created
- `updated` - Credential modified
- `deleted` - Credential deleted
- `accessed` - Credential decrypted and used
- `tested` - Credential test performed
- `access_denied` - Access denied (inactive/expired)

**Audit Log Entry Creation:**

```python
# credentials/service.py:790-813
def _create_audit_log(
    self,
    credential: Credential,
    action: str,
    user_id: Optional[str] = None,
    success: Optional[bool] = True,
    metadata: Optional[Dict[str, Any]] = None
):
    audit_log = CredentialAuditLog(
        credential_id=credential.id,
        action=action,
        user_id=user_id,
        success=success,
        audit_metadata=metadata or {}
    )
```

**Sources:** [orchestrator/core/credentials/service.py:790-813](), [orchestrator/core/models/credentials.py:105-130]()

---

## RAG Configuration

Retrieval-Augmented Generation (RAG) settings control vector similarity search behavior. These are **dynamic settings** loaded from the database first, with environment variable fallbacks.

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_MIN_SIMILARITY` | `0.65` | Minimum cosine similarity threshold |
| `RAG_TOP_K` | `5` | Number of results to retrieve |
| `RAG_RERANK_ENABLED` | `"false"` | Enable result reranking |

**Configuration Location:** [orchestrator/config.py:192-223]()

**Dynamic Property Pattern:**

```python
# config.py:195-203
@property
def RAG_MIN_SIMILARITY(self) -> float:
    try:
        from core.llm.manager import get_system_setting
        val = get_system_setting("rag", "min_similarity", "0.65")
        return float(val) if val else 0.65
    except Exception:
        return float(os.getenv("RAG_MIN_SIMILARITY", "0.65"))
```

**Sources:** [orchestrator/config.py:192-223]()

---

## Routing Configuration

Settings for the Universal Orchestrator Router and webhook integrations:

| Variable | Default | Description |
|----------|---------|-------------|
| `COMPOSIO_WEBHOOK_SECRET` | - | Webhook verification secret from Composio |
| `ROUTING_CACHE_TTL_HOURS` | `24` | Agent routing cache TTL (hours) |
| `ROUTING_LLM_CONFIDENCE_THRESHOLD` | `0.5` | Minimum confidence for LLM routing decisions |

**Configuration Location:** [orchestrator/config.py:139-144]()

**Sources:** [orchestrator/config.py:139-144](), [orchestrator/.env.example:36-39]()

---

## GitHub Integration

For automated recipes that create pull requests:

| Variable | Default | Description |
|----------|---------|-------------|
| `GITHUB_REPO_OWNER` | `""` | GitHub repository owner |
| `GITHUB_REPO_NAME` | `""` | GitHub repository name |
| `GITHUB_DEFAULT_BRANCH` | `"main"` | Default branch for PRs |

**Configuration Location:** [orchestrator/config.py:146-149]()

**Sources:** [orchestrator/config.py:146-149](), [orchestrator/.env.example:41-44]()

---

## Configuration Validation

### Validation on Startup

The `Config.validate()` method ([config.py:225-247]()) checks required settings at application startup:

```python
def validate(self) -> bool:
    errors = []
    
    # Check database
    if not all([self.POSTGRES_DB, self.POSTGRES_USER, self.POSTGRES_HOST, self.POSTGRES_PORT]):
        errors.append("Database not configured")
    
    # Check API key
    if self.REQUIRE_API_KEY and not self.API_KEY:
        errors.append("API_KEY required when REQUIRE_API_KEY=true")
    
    if errors:
        logger.error("❌ Configuration errors:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    return True
```

**Sources:** [orchestrator/config.py:225-247]()

### Configuration Debugging

The `Config.print_config()` method ([config.py:249-272]()) prints current configuration with optional secret masking:

```python
config.print_config(show_secrets=False)
```

**Output:**

```
============================================================
AUTOMATOS AI CONFIGURATION
============================================================
Environment: production
Database: orchestrator_db@localhost:5432
Redis: localhost:6379
LLM Provider: openai (gpt-4)
OpenAI Key: ✅ Set
Anthropic Key: ❌ Not set
API Key: ✅ Set
API Key Required: True
============================================================
```

**Sources:** [orchestrator/config.py:249-272]()

---

## Complete Configuration Reference

### Minimal Production Configuration

```bash
# Database (Required)
POSTGRES_DB=orchestrator_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password_here
POSTGRES_HOST=db
POSTGRES_PORT=5432

# Redis (Recommended)
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=redis_password_here

# API Security (Required)
API_KEY=your_secure_api_key_here
REQUIRE_API_KEY=true

# LLM Provider (Required)
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai
LLM_MODEL=gpt-4

# AWS S3 (Required for Marketplace)
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
MARKETPLACE_S3_BUCKET=automatos-marketplace

# Environment
ENVIRONMENT=production
LOG_LEVEL=INFO
```

**Sources:** [orchestrator/.env.example:1-64](), [orchestrator/config.py:28-285]()

### Configuration File Locations

```mermaid
graph TB
    Root["/automatos-ai/"]
    RootEnv[".env<br/>(loaded by backend)"]
    OrchestratorDir["/orchestrator/"]
    OrchestratorEnv[".env<br/>(backend config)"]
    EnvExample[".env.example<br/>(template)"]
    FrontendDir["/frontend/"]
    FrontendEnv[".env.local<br/>(frontend config)"]
    
    Root --> RootEnv
    Root --> OrchestratorDir
    Root --> FrontendDir
    OrchestratorDir --> OrchestratorEnv
    OrchestratorDir --> EnvExample
    FrontendDir --> FrontendEnv
    
    OrchestratorEnv -->|"loaded by"| ConfigPy["config.py:26<br/>load_dotenv()"]
```

**Sources:** [orchestrator/config.py:24-26](), [orchestrator/.env.example:1-64]()

---

## Configuration Best Practices

### 1. Environment Separation

Use different configuration files for each environment:

- `.env.local` - Local development
- `.env.staging` - Staging environment
- `.env.production` - Production environment

**Never commit** `.env` files to version control. Use `.env.example` as a template.

**Sources:** [.gitignore:100-105]()

### 2. Secret Management

Store sensitive values (API keys, passwords) in:
- Environment variables for production
- Secret management systems (AWS Secrets Manager, HashiCorp Vault)
- Credential management system ([Credentials Management](#9.4))

**Sources:** [orchestrator/core/credentials/service.py:42-181]()

### 3. Database-Driven Settings

For settings that change frequently (LLM model, similarity thresholds), use the `system_settings` table instead of environment variables. The `Config` class automatically loads these via dynamic properties.

**Sources:** [orchestrator/config.py:88-106](), [orchestrator/config.py:195-223]()

### 4. Feature Flag Patterns

Use feature flags for gradual rollout:

```python
from config import config

if config.ENABLE_BATCH_API:
    router.include_router(batch_router, prefix="/batch")
```

**Sources:** [orchestrator/config.py:154]()

### 5. Validation Early

Call `config.validate()` at application startup to catch configuration errors before they cause runtime failures:

```python
from config import config

if not config.validate():
    logger.critical("Configuration validation failed - exiting")
    sys.exit(1)
```

**Sources:** [orchestrator/config.py:225-247]()

---