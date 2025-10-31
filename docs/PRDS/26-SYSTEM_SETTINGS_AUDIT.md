# PRD-26: System Settings Audit - Current Status

## Executive Summary

**Critical Issues Found:**
1. ❌ Settings are saved but may be reset by seed script
2. ❌ Many settings in UI are NOT actually used by services
3. ❌ CodeGraph settings show "Select LLM Provider" but still works (using hardcoded defaults)
4. ❌ Settings refresh/reset after page reload (likely seed script overwriting)

## Settings Usage Matrix

### ✅ General Settings Tab

| Setting | Used? | Where Used | Status |
|---------|-------|------------|--------|
| `environment` | ❌ NO | Config uses `os.getenv("ENVIRONMENT")` | Not wired |
| `log_level` | ❌ NO | Python logging uses `os.getenv("LOG_LEVEL")` | Not wired |
| `embedding_model` | ❌ NO | Services hardcoded | Not wired |
| `openai_embedding_model` | ❌ NO | Services hardcoded | Not wired |
| `embedding_cache_dir` | ❌ NO | Not used | Not implemented |
| `embedding_max_seq_length` | ❌ NO | Not used | Not implemented |
| `vector_store_type` | ❌ NO | Not used | Not implemented |
| `vector_store_dimensions` | ❌ NO | Not used | Not implemented |
| `chunk_size` | ❌ NO | Not used | Not implemented |
| `chunk_overlap` | ❌ NO | Not used | Not implemented |
| `max_context_length` | ❌ NO | Not used | Not implemented |
| `deploy_host` | ❌ NO | No deployment service | Not used |
| `deploy_port` | ❌ NO | No deployment service | Not used |
| `deploy_user` | ❌ NO | No deployment service | Not used |
| `deploy_key_path` | ❌ NO | No deployment service | Not used |
| `deploy_enabled` | ❌ NO | No deployment service | Not used |
| `nextauth_secret` | ⚠️ PARTIAL | Frontend env var (if NextAuth exists) | Partially used |
| `nextauth_url` | ⚠️ PARTIAL | Frontend env var (if NextAuth exists) | Partially used |
| `next_public_api_url` | ✅ YES | Frontend API client | Used |

### ✅ Orchestrator LLM Settings Tab

| Setting | Used? | Where Used | Status |
|---------|-------|------------|--------|
| `orchestrator_llm.provider` | ✅ YES | `llm_provider/manager.py` line 91 | **WORKING** |
| `orchestrator_llm.model` | ✅ YES | `llm_provider/manager.py` line 92 | **WORKING** |
| `orchestrator_llm.temperature` | ✅ YES | `llm_provider/manager.py` line 245 | **WORKING** |
| `orchestrator_llm.max_tokens` | ✅ YES | `llm_provider/manager.py` line 246 | **WORKING** |
| `orchestrator_llm.top_p` | ❌ NO | Not in LLMConfig | Not wired |
| `orchestrator_llm.frequency_penalty` | ❌ NO | Not in LLMConfig | Not wired |
| `orchestrator_llm.presence_penalty` | ❌ NO | Not in LLMConfig | Not wired |
| `orchestrator_llm.streaming_enabled` | ❌ NO | Not in LLMConfig | Not wired |
| `orchestrator_llm.timeout_seconds` | ❌ NO | Not in LLMConfig | Not wired |
| `orchestrator_llm.max_context_length` | ❌ NO | Not in LLMConfig | Not wired |
| `orchestrator_llm.cache_enabled` | ❌ NO | Not implemented | Not wired |
| `orchestrator_llm.retry_count` | ❌ NO | Not implemented | Not wired |
| `orchestrator_llm.retry_delay` | ❌ NO | Not implemented | Not wired |

### ❌ CodeGraph Settings Tab

| Setting | Used? | Where Used | Status |
|---------|-------|------------|--------|
| `codegraph.provider` | ❌ NO | CodeGraph uses hardcoded OpenAI | **NOT USED** |
| `codegraph.model` | ❌ NO | CodeGraph uses hardcoded GPT-3.5-turbo | **NOT USED** |
| `codegraph.embedding_model` | ❌ NO | CodeGraph uses hardcoded `text-embedding-ada-002` | **NOT USED** |
| `codegraph.temperature` | ❌ NO | CodeGraph doesn't use LLM for embeddings | Not applicable |
| `codegraph.max_tokens` | ❌ NO | CodeGraph doesn't use LLM for embeddings | Not applicable |
| `codegraph.analysis_settings.*` | ❌ NO | Not implemented | Not wired |
| `codegraph.performance_settings.*` | ❌ NO | Not implemented | Not wired |

**Note**: CodeGraph works because it uses hardcoded OpenAI client initialization. Settings are displayed but not consumed.

### ❌ Logging Settings Tab

| Setting | Used? | Where Used | Status |
|---------|-------|------------|--------|
| All logging settings | ❌ NO | Python logging configured in code | **NOT USED** |

**Note**: Python logging is configured in `utils/logging_utils.py` and `main.py`. Settings tab exists but nothing reads from it.

### ❌ Rate Limiting Settings Tab

| Setting | Used? | Where Used | Status |
|---------|-------|------------|--------|
| All rate limiting settings | ❌ NO | No rate limiting middleware | **NOT USED** |

**Note**: `mcp_bridge.py` has rate limiting but uses environment variables, not settings.

### ⚠️ API Keys Settings Tab

| Setting | Used? | Where Used | Status |
|---------|-------|------------|--------|
| All API key settings | ⚠️ DUPLICATE | Same as General tab | **REDUNDANT** |

**Note**: This tab duplicates General settings. Should be merged or removed.

## Root Cause Analysis

### Why Settings Don't Persist

1. **Seed Script May Overwrite**
   - Seed script runs on database init
   - If seed script runs after settings save, it may overwrite
   - Need to verify seed script logic

2. **Settings Save to Database**
   - `bulkUpdateSettings()` API endpoint exists and works
   - Settings ARE saved to database
   - Issue: Settings may be reset when:
     - Seed script runs again
     - Database is reinitialized
     - Settings are refreshed from seed data

3. **Frontend Reload Issue**
   - Frontend loads settings from `/api/system-settings/by-category`
   - If seed script runs, settings reset to defaults
   - Need to ensure seed script NEVER overwrites existing values

### Why Settings Don't Work

1. **Services Use Hardcoded Values**
   - CodeGraph: `OpenAI(api_key=openai_api_key)` hardcoded
   - RAG: `model="text-embedding-ada-002"` hardcoded
   - Config: `os.getenv("ENVIRONMENT")` instead of settings

2. **Settings Not Read by Services**
   - Services don't call `get_system_setting()`
   - Services use environment variables or hardcoded values
   - Need to wire services to read from settings

3. **Missing Settings Integration**
   - No middleware to inject settings
   - No settings reload mechanism
   - No real-time settings updates

## Immediate Action Items

### Critical (Fix First)

1. **Fix Seed Script** ✅ PRIORITY 1
   - Ensure seed script NEVER overwrites existing values
   - Only set `default_value`, never `value` if setting exists
   - Test: Save setting, run seed script, verify setting persists

2. **Verify Settings Save** ✅ PRIORITY 2
   - Add verification endpoint: `GET /api/system-settings/verify/{category}/{key}`
   - Add frontend verification after save
   - Add logging to track saves

3. **Wire CodeGraph Settings** ✅ PRIORITY 3
   - Update `codegraph_service.py` to use settings
   - Use LLM service instead of direct OpenAI
   - Read `codegraph.provider`, `codegraph.model`, `codegraph.embedding_model`

### High Priority

4. **Wire LLM Parameters** ✅ PRIORITY 4
   - Add `top_p`, `frequency_penalty`, `presence_penalty` to LLMConfig
   - Read from settings in `llm_provider/manager.py`
   - Pass to LLM clients

5. **Wire General Settings** ✅ PRIORITY 5
   - Update `config.py` to read `environment` from settings
   - Update logging to read `log_level` from settings
   - Wire `next_public_api_url` (already used in frontend)

### Medium Priority

6. **Remove Unused Settings Tabs** ✅ PRIORITY 6
   - Logging Tab: Remove or implement logging service
   - Rate Limiting Tab: Remove or implement rate limiting
   - API Keys Tab: Merge with General or remove

7. **Add Settings Usage Indicators** ✅ PRIORITY 7
   - Show badge: "Active" or "Not Used" in UI
   - Help users understand which settings matter
   - Add tooltips explaining usage

## Testing Checklist

### Settings Persistence
- [ ] Change setting → Save → Refresh page → Setting persists
- [ ] Change setting → Save → Restart backend → Setting persists
- [ ] Run seed script → Settings NOT overwritten
- [ ] Database reinit → Settings persist (if not wiped)

### Settings Usage
- [ ] Change orchestrator LLM provider → Actually changes LLM
- [ ] Change orchestrator temperature → Affects responses
- [ ] Change codegraph LLM provider → CodeGraph uses new provider
- [ ] Change codegraph embedding model → Embeddings use new model
- [ ] Change environment setting → Config reflects change
- [ ] Change log level → Logging level changes

### Settings Validation
- [ ] Invalid temperature → Shows error
- [ ] Invalid provider → Shows error
- [ ] Missing required setting → Shows error
- [ ] Save button disabled if invalid → Yes

## Files to Review/Modify

### Backend (Critical)
1. `orchestrator/seeds/seed_system_settings.py` - Fix overwrite issue
2. `orchestrator/services/llm_provider/manager.py` - Wire all LLM parameters
3. `orchestrator/services/codegraph_service.py` - Wire CodeGraph settings
4. `orchestrator/config.py` - Wire general settings
5. `orchestrator/api/system_settings.py` - Add verification endpoint

### Frontend (Critical)
1. `frontend/components/settings/SystemSettingsTab.tsx` - Add save verification
2. `frontend/components/settings/CodeGraphSettingsTab.tsx` - Ensure settings save
3. `frontend/components/settings/OrchestratorLLMSettingsTab.tsx` - Wire all parameters
4. `frontend/components/settings/GeneralSettingsTab.tsx` - Remove unused or wire them
5. `frontend/components/settings/SystemLoggingSettingsTab.tsx` - Remove or implement
6. `frontend/components/settings/APIRateLimitingSettingsTab.tsx` - Remove or implement
7. `frontend/components/settings/BackendAPIKeysSettingsTab.tsx` - Merge or remove

## Recommendations

### Short Term (This PR)
1. ✅ Fix seed script to never overwrite
2. ✅ Verify settings save correctly
3. ✅ Wire CodeGraph settings (critical for user experience)
4. ✅ Wire remaining LLM parameters

### Medium Term (Next PR)
1. Wire general settings (environment, log_level)
2. Remove unused settings tabs
3. Add settings usage indicators

### Long Term (Future)
1. Add settings audit log
2. Add settings rollback capability
3. Add real-time settings reload
4. Add settings import/export

## Conclusion

**Current State**: Settings infrastructure exists but is not fully utilized. Many settings are "UI-only" and not connected to services.

**Goal**: Make every setting shown in UI actually functional, or remove it if not needed.

**Priority**: Fix seed script first (prevents settings reset), then wire up critical settings (CodeGraph, LLM parameters).

