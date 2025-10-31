# PRD-26: System Settings Configuration

## Overview

This PRD addresses critical issues with the system settings implementation:
1. Settings are not being properly saved to database
2. Settings refresh/reset when page reloads
3. Many settings shown in UI are not actually used by services
4. Settings need to be properly wired to services that should use them
5. Need comprehensive audit of all settings to identify what's real vs UI-only

## Current State Analysis

### ✅ What Works

1. **Settings API** - Backend API endpoints exist:
   - `GET /api/system-settings/` - List all settings
   - `GET /api/system-settings/by-category` - Get by category
   - `PUT /api/system-settings/{setting_id}` - Update single setting
   - `POST /api/system-settings/bulk-update` - Bulk update
   - `POST /api/system-settings/reset-to-defaults` - Reset to defaults

2. **Frontend Save Handler** - `SystemSettingsTab.tsx` calls `bulkUpdateSettings()`

3. **LLM Settings Usage** - Orchestrator LLM settings ARE used:
   - `orchestrator_llm.provider` → Used in `llm_provider/manager.py`
   - `orchestrator_llm.model` → Used in `llm_provider/manager.py`
   - `orchestrator_llm.temperature` → Used in `llm_provider/manager.py` (line 245)
   - `orchestrator_llm.max_tokens` → Used in `llm_provider/manager.py` (line 246)

### ❌ What's Broken

1. **Settings Not Persisting**
   - Issue: Settings refresh after page reload shows defaults
   - Root Cause: Likely seed script re-running or settings being overwritten
   - Solution: Ensure seed script only creates if not exists, never overwrites existing values

2. **Settings Not Used by Services**
   - **General Settings:**
     - `environment` - NOT USED (hardcoded in config.py)
     - `log_level` - NOT USED (Python logging uses config.py LOG_LEVEL)
     - `embedding_model` - NOT USED (hardcoded in services)
     - `openai_embedding_model` - NOT USED (hardcoded in services)
     - `deploy_host`, `deploy_port` - NOT USED (no deployment service)
     - `nextauth_secret`, `nextauth_url` - PARTIALLY USED (frontend env vars)
     - `next_public_api_url` - USED (frontend API client)

   - **Orchestrator LLM Settings:**
     - `provider`, `model`, `temperature`, `max_tokens` - ✅ USED
     - Other parameters (top_p, frequency_penalty, etc.) - ❌ NOT USED
     - Performance settings - ❌ NOT USED
     - Model-specific settings - ❌ NOT USED

   - **CodeGraph Settings:**
     - `provider`, `model` - ❌ NOT USED (CodeGraph uses hardcoded OpenAI)
     - `embedding_model` - ❌ NOT USED (hardcoded `text-embedding-ada-002`)
     - Other CodeGraph settings - ❌ NOT USED

   - **Logging Settings:**
     - All logging settings - ❌ NOT USED (Python logging configured in code)

   - **Rate Limiting Settings:**
     - All rate limiting settings - ❌ NOT USED (no rate limiting middleware)

3. **Missing Settings Integration**
   - Services don't read from system settings
   - Services use hardcoded values or environment variables
   - No real-time settings reload capability

## Requirements

### Phase 1: Fix Settings Persistence ✅ CRITICAL

1. **Ensure Settings Save to Database**
   - Verify `bulkUpdateSettings()` API works correctly
   - Add database transaction logging
   - Add frontend success/error handling
   - Add verification: After save, immediately fetch and verify

2. **Fix Seed Script**
   - Seed script should ONLY create settings if they don't exist
   - Seed script should NEVER overwrite existing values
   - Seed script should only set `default_value`, not `value` if setting exists

3. **Add Settings Validation**
   - Validate settings before saving
   - Enforce validation rules from `validation_rules` field
   - Return clear error messages

### Phase 2: Wire Up LLM Settings ✅ HIGH PRIORITY

1. **Orchestrator LLM Settings**
   - ✅ Already working: `provider`, `model`, `temperature`, `max_tokens`
   - **TODO**: Add support for:
     - `top_p`, `frequency_penalty`, `presence_penalty`
     - `max_context_length` (for context window)
     - `streaming_enabled` (for streaming responses)
     - `timeout_seconds` (for request timeouts)

2. **CodeGraph LLM Settings**
   - **TODO**: Update `codegraph_service.py` to use:
     - `codegraph.provider` (currently uses hardcoded OpenAI)
     - `codegraph.model` (currently uses hardcoded GPT-3.5-turbo)
     - `codegraph.embedding_model` (currently uses hardcoded `text-embedding-ada-002`)
   - **TODO**: Update embedding generation to read from settings

3. **Chatbot LLM Settings**
   - **TODO**: Chatbot currently defaults to orchestrator settings
   - **OPTIONAL**: Add `chatbot.provider`, `chatbot.model` for per-service config

### Phase 3: Wire Up Other Settings ✅ MEDIUM PRIORITY

1. **General Settings**
   - `environment` - Wire to `config.py` ENVIRONMENT (used in IS_PRODUCTION checks)
   - `log_level` - Wire to Python logging (requires runtime log level change)
   - `next_public_api_url` - ✅ Already used in frontend API client
   - `nextauth_secret`, `nextauth_url` - Used in NextAuth config (if exists)

2. **Embedding Settings**
   - `codegraph.embedding_model` - Wire to CodeGraph embedding generation
   - `rag.embedding_model` - Wire to RAG service (create setting if missing)
   - Create unified embedding service that reads from settings

3. **Deployment Settings**
   - **DECISION**: Remove if not used, or create deployment service
   - If keeping, wire to deployment automation (if exists)

### Phase 4: Implement Missing Features ✅ LOW PRIORITY

1. **Logging Service**
   - **DECISION**: Do we need a logging service?
   - If yes, implement logging service that reads from `logging.*` settings
   - If no, remove logging settings tab

2. **Rate Limiting Service**
   - **DECISION**: Do we need rate limiting?
   - If yes, implement FastAPI rate limiting middleware
   - Read from `rate_limiting.*` settings
   - If no, remove rate limiting settings tab

3. **API Keys Tab**
   - **DECISION**: Merge with General settings or keep separate?
   - Currently duplicates General tab settings
   - Recommendation: Remove if redundant

### Phase 5: Settings Management ✅ ENHANCEMENTS

1. **Real-time Settings Reload**
   - Add API endpoint to reload settings without restart
   - Add signal handlers for graceful reload
   - Document which settings require restart

2. **Settings Validation**
   - Frontend validation before save
   - Backend validation on save
   - Clear error messages for invalid values

3. **Settings History**
   - Track settings changes (audit log)
   - Show who changed what and when
   - Allow rollback to previous values

4. **Settings Import/Export**
   - Export settings to JSON
   - Import settings from JSON
   - Useful for environment migrations

## Implementation Plan

### Step 1: Fix Seed Script ✅

**File**: `orchestrator/seeds/seed_system_settings.py`

**Changes**:
```python
# Current: Always sets value
setting.value = setting_data.get("default_value")

# New: Only set value if setting doesn't exist
if existing:
    # Only update default_value, never overwrite user's value
    if existing.default_value != setting_data.get("default_value"):
        existing.default_value = setting_data.get("default_value")
else:
    # New setting - set both value and default_value
    setting.value = setting_data.get("default_value", setting_data.get("value"))
    setting.default_value = setting_data.get("default_value")
```

### Step 2: Verify Settings Save ✅

**Test**:
1. Change a setting in UI
2. Save
3. Immediately refresh page
4. Verify setting persists

**Add Debugging**:
- Add console.log in frontend save handler
- Add logging in backend bulk update endpoint
- Add verification endpoint: GET `/api/system-settings/verify/{category}/{key}`

### Step 3: Wire LLM Parameters ✅

**File**: `orchestrator/services/llm_provider/manager.py`

**Changes**:
```python
# Current: Only reads temperature and max_tokens
temperature = float(get_system_setting(category, "temperature", "0.7"))
max_tokens = int(get_system_setting(category, "max_tokens", "2000"))

# New: Read all LLM parameters
temperature = float(get_system_setting(category, "temperature", "0.7"))
max_tokens = int(get_system_setting(category, "max_tokens", "2000"))
top_p = float(get_system_setting(category, "top_p", "1.0"))
frequency_penalty = float(get_system_setting(category, "frequency_penalty", "0.0"))
presence_penalty = float(get_system_setting(category, "presence_penalty", "0.0"))
streaming_enabled = get_system_setting(category, "streaming_enabled", "false").lower() == "true"
timeout_seconds = int(get_system_setting(category, "timeout_seconds", "30"))

# Pass to LLMConfig
config = LLMConfig(
    provider=provider,
    model=model,
    temperature=temperature,
    max_tokens=max_tokens,
    top_p=top_p,
    frequency_penalty=frequency_penalty,
    presence_penalty=presence_penalty,
    streaming_enabled=streaming_enabled,
    timeout_seconds=timeout_seconds,
    # ...
)
```

### Step 4: Wire CodeGraph Settings ✅

**File**: `orchestrator/services/codegraph_service.py`

**Changes**:
```python
# Current: Hardcoded OpenAI client
self.openai_client = OpenAI(api_key=openai_api_key)

# New: Use LLM service with CodeGraph settings
from services.llm_provider import create_llm_manager
self.llm_manager = create_llm_manager(service_name="codegraph")

# Update embedding generation
async def _store_symbols_with_embeddings(self, ...):
    # Get embedding model from settings
    from services.llm_provider.manager import get_system_setting
    embedding_model = get_system_setting("codegraph", "embedding_model", "text-embedding-ada-002")
    
    # Use LLM service for embeddings (if supported) or direct OpenAI
    if embedding_model.startswith("text-embedding"):
        # Use OpenAI embeddings
        response = self.openai_client.embeddings.create(
            model=embedding_model,  # From settings
            input=texts
        )
```

### Step 5: Wire General Settings ✅

**File**: `orchestrator/config.py`

**Changes**:
```python
# Current: Hardcoded from environment
ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

# New: Read from system settings with env fallback
@property
def ENVIRONMENT(self) -> str:
    from services.llm_provider.manager import get_system_setting
    return get_system_setting("general", "environment") or os.getenv("ENVIRONMENT", "development")

@property
def LOG_LEVEL(self) -> str:
    from services.llm_provider.manager import get_system_setting
    return get_system_setting("general", "log_level") or os.getenv("LOG_LEVEL", "INFO")
```

### Step 6: Clean Up Unused Settings ✅

**Action Items**:
1. **Remove or Wire Up**:
   - Logging Settings Tab - Either implement logging service or remove
   - Rate Limiting Settings Tab - Either implement rate limiting or remove
   - API Keys Tab - Merge with General or remove

2. **Document What's Used**:
   - Add comments in seed script showing which settings are active
   - Add validation rules indicating if setting is used

3. **Add Settings Status Badge**:
   - In UI, show badge: "Active" or "Not Used"
   - Help users understand which settings matter

## Testing Checklist

### Settings Persistence
- [ ] Change setting, save, refresh page → Setting persists
- [ ] Change setting, save, restart backend → Setting persists
- [ ] Run seed script → Does NOT overwrite existing values

### LLM Settings
- [ ] Change orchestrator LLM provider → Actually changes LLM used
- [ ] Change orchestrator temperature → Affects LLM responses
- [ ] Change codegraph LLM provider → CodeGraph uses new provider
- [ ] Change codegraph embedding model → Embeddings use new model

### General Settings
- [ ] Change environment setting → `config.IS_PRODUCTION` reflects change
- [ ] Change log level → Logging level actually changes
- [ ] Change API URL → Frontend uses new URL

### Settings Validation
- [ ] Invalid temperature value → Shows error
- [ ] Invalid provider name → Shows error
- [ ] Missing required setting → Shows error

## Success Criteria

1. ✅ **Settings Persist**: All settings save to database and persist across page refreshes
2. ✅ **Settings Used**: All settings shown in UI are actually used by services
3. ✅ **Clear Documentation**: Each setting clearly indicates if it's active or not
4. ✅ **Validation Works**: Invalid settings show clear error messages
5. ✅ **No Redundant Settings**: No duplicate settings across tabs
6. ✅ **Settings Auditable**: Can see when settings were changed and by whom (future)

## Files to Modify

### Backend
1. `orchestrator/seeds/seed_system_settings.py` - Fix seed script
2. `orchestrator/services/llm_provider/manager.py` - Wire up all LLM parameters
3. `orchestrator/services/codegraph_service.py` - Wire up CodeGraph settings
4. `orchestrator/config.py` - Wire up general settings
5. `orchestrator/api/system_settings.py` - Add verification endpoint

### Frontend
1. `frontend/components/settings/SystemSettingsTab.tsx` - Improve save feedback
2. `frontend/components/settings/GeneralSettingsTab.tsx` - Remove unused settings or wire them
3. `frontend/components/settings/OrchestratorLLMSettingsTab.tsx` - Wire up all parameters
4. `frontend/components/settings/CodeGraphSettingsTab.tsx` - Ensure settings are saved
5. `frontend/components/settings/SystemLoggingSettingsTab.tsx` - Remove or implement
6. `frontend/components/settings/APIRateLimitingSettingsTab.tsx` - Remove or implement
7. `frontend/components/settings/BackendAPIKeysSettingsTab.tsx` - Merge or remove

## Migration Strategy

1. **Phase 1** (Week 1): Fix seed script and verify settings save
2. **Phase 2** (Week 1): Wire up all LLM settings
3. **Phase 3** (Week 2): Wire up CodeGraph settings
4. **Phase 4** (Week 2): Wire up general settings
5. **Phase 5** (Week 3): Clean up unused settings tabs
6. **Phase 6** (Week 3): Add validation and error handling

## Notes

- **Critical**: Settings must persist across page refreshes
- **Important**: Only show settings that are actually used
- **Enhancement**: Add settings usage indicators in UI
- **Future**: Add settings audit log and rollback capability

