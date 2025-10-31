# LLM Service Integration Plan

## Executive Summary

This document outlines the plan to integrate a unified LLM service into Automatos, replacing direct API key usage with a credential-based, lazy-loading system that supports multiple LLM providers (OpenAI, Anthropic, Google, Azure, HuggingFace).

## Current State Analysis

### Files Using OPENAI_API_KEY (126 files found)
**Critical Orchestrator Files:**
1. `orchestrator/services/llm_provider.py` - Current LLM abstraction layer
2. `orchestrator/api/codegraph.py` - CodeGraph service (uses OpenAI for embeddings)
3. `orchestrator/api/chatbot_llm.py` - Chatbot API (uses Anthropic directly)
4. `orchestrator/services/codegraph_service.py` - CodeGraph service implementation
5. `orchestrator/context_engineering/embeddings.py` - Embedding generation
6. `orchestrator/api/documents.py` - Document processing
7. `orchestrator/config.py` - Configuration loader

### Files Using ANTHROPIC_API_KEY (37 files found)
**Critical Files:**
1. `orchestrator/services/llm_provider.py` - Already supports Anthropic
2. `orchestrator/api/chatbot_llm.py` - Direct Anthropic client usage
3. `orchestrator/services/credential_resolver.py` - Credential resolution

### Current LLM Architecture

#### Orchestrator LLM Provider (`services/llm_provider.py`)
- **Structure**: `LLMManager` → `BaseLLMProvider` → Provider implementations
- **Providers**: OpenAI, Anthropic
- **Features**: Async/sync support, tool calling, token tracking
- **Configuration**: Environment variables + credential system fallback
- **Limitation**: Only supports OpenAI/Anthropic, no Google/Azure/HuggingFace

#### LLM Service (Reference Implementation)
- **Structure**: `LLM` → `LLMClient` (base) → Provider clients
- **Providers**: OpenAI, Anthropic, Google, Azure, Bow (custom)
- **Interface**: `inference()`, `inference_stream()`, `test_connection()`
- **Location**: `bagofwords/backend/app/ai/llm/`

### Credential System (PRD-18)
- **Status**: ✅ Fully implemented
- **LLM Credential Types**: `openai_api`, `anthropic_api`, `huggingface_api` (exists!)
- **Storage**: Encrypted database storage
- **Resolver**: `CredentialResolver` with caching and fallback
- **UI**: Full credential management UI in Settings

### System Settings
- **Location**: `orchestrator/models/system_settings.py`
- **Category**: `orchestrator_llm`, `codegraph`
- **API**: `/api/system-settings` endpoints
- **Frontend**: `OrchestratorLLMSettingsTab.tsx`, `CodeGraphSettingsTab.tsx`

## Target Architecture

### Refactored LLM Provider Structure

**Key Decision**: Refactor existing `llm_provider.py` instead of creating new service!

```
orchestrator/services/
├── llm_provider.py                    # Main entry point (refactored, backward compatible)
└── llm_provider/                      # NEW: Module structure
    ├── __init__.py                     # Export main classes for backward compatibility
    ├── config.py                       # LLMConfig, LLMResponse dataclasses (moved)
    ├── manager.py                      # LLMManager (refactored with settings support)
    └── clients/
        ├── __init__.py
        ├── base.py                     # BaseLLMProvider (moved from llm_provider.py)
        ├── openai_client.py            # OpenAIProvider (moved from llm_provider.py)
        ├── anthropic_client.py         # AnthropicProvider (moved from llm_provider.py)
        ├── google_client.py            # GoogleProvider (NEW)
        ├── azure_client.py             # AzureProvider (NEW)
        └── huggingface_client.py       # HuggingFaceProvider (NEW)
```

**Backward Compatibility**: `services/llm_provider.py` will import from `llm_provider/` to maintain existing imports.

### Key Features
1. **Lazy Loading**: Service initializes on first use, not at startup
2. **Credential-Based**: Pulls API keys from credential system via settings
3. **Multi-Provider**: OpenAI, Anthropic, Google, Azure, HuggingFace
4. **Unified Interface**: `inference()`, `inference_stream()`, `test_connection()`
5. **Service-Specific**: Orchestrator and CodeGraph can use different providers
6. **Graceful Failure**: Returns None/error if credentials not configured

### Integration Points

#### 1. System Settings Integration
- **Orchestrator LLM Setting**: `orchestrator_llm.provider`, `orchestrator_llm.model`
- **CodeGraph LLM Setting**: `codegraph.provider`, `codegraph.model`
- **Settings UI**: Select provider from dropdown, linked to credentials

#### 2. Credential Resolution Flow
```
User selects "OpenAI" in OrchestratorLLM Settings Tab
  ↓
System looks up credential: "development_openai" or "production_openai"
  ↓
CredentialResolver resolves encrypted API key
  ↓
UnifiedLLMService creates OpenAI client
  ↓
Client ready for inference
```

#### 3. Service-Specific Configuration
- **Orchestrator**: Uses `orchestrator_llm.provider` + `orchestrator_llm.model` settings
- **CodeGraph**: Uses `codegraph.provider` + `codegraph.model` settings  
- **Other Services**: Default to Orchestrator settings

## Implementation Plan

### Phase 1: Refactor Existing LLM Provider Service

#### Step 1.1: Create Clients Directory Structure
```bash
mkdir -p orchestrator/services/llm_provider/clients
touch orchestrator/services/llm_provider/__init__.py
touch orchestrator/services/llm_provider/clients/__init__.py
```

#### Step 1.2: Move Existing Base Classes
**File**: `orchestrator/services/llm_provider/clients/base.py`
- Move `BaseLLMProvider` from `llm_provider.py`
- Move `LLMConfig`, `LLMResponse` dataclasses
- Keep existing interface (backward compatible)

#### Step 1.3: Split Provider Clients into Separate Files
**Files**:
1. `clients/openai_client.py` - Move `OpenAIProvider` from `llm_provider.py`
2. `clients/anthropic_client.py` - Move `AnthropicProvider` from `llm_provider.py`
3. `clients/google_client.py` - **NEW** - Google Gemini client (based on bagofwords)
4. `clients/azure_client.py` - **NEW** - Azure OpenAI client (based on bagofwords)
5. `clients/huggingface_client.py` - **NEW** - HuggingFace Inference API client

#### Step 1.4: Refactor Main LLM Service
**File**: `orchestrator/services/llm_provider/manager.py`
- Move `LLMManager` from `llm_provider.py`
- Add lazy loading from system settings
- Add per-service configuration support (orchestrator, codegraph, etc.)
- Keep backward compatibility with existing API

### Phase 2: Integrate with System Settings

#### Step 2.1: Update System Settings Model
**File**: `orchestrator/models/system_settings.py`
- Ensure `orchestrator_llm` category has `provider` and `model` settings
- Ensure `codegraph` category has `provider` and `model` settings

#### Step 2.2: Update OrchestratorLLMSettingsTab
**File**: `frontend/components/settings/OrchestratorLLMSettingsTab.tsx`
- Provider dropdown: OpenAI, Anthropic, Google, Azure, HuggingFace
- Model dropdown: Dynamic based on provider selection
- Link to credential management (show warning if credential not configured)
- Save settings to `orchestrator_llm.provider` and `orchestrator_llm.model`

#### Step 2.3: Update CodeGraphSettingsTab
**File**: `frontend/components/settings/CodeGraphSettingsTab.tsx`
- Add LLM Provider Configuration section (same as OrchestratorLLM)
- Provider dropdown: OpenAI, Anthropic, Google, Azure, HuggingFace
- Model dropdown: Dynamic based on provider
- Save to `codegraph.provider` and `codegraph.model`

### Phase 3: Migrate Existing Services

#### Step 3.1: Update LLM Provider Service
**File**: `orchestrator/services/llm_provider.py`
- **Option A**: Replace with `UnifiedLLMService` adapter (maintain backward compatibility)
- **Option B**: Refactor to use `UnifiedLLMService` internally
- Keep `LLMManager` interface for backward compatibility

#### Step 3.2: Update CodeGraph Service
**File**: `orchestrator/services/codegraph_service.py`
- Replace direct OpenAI client with `UnifiedLLMService`
- Use `codegraph.provider` setting for provider selection
- Lazy load on first use

#### Step 3.3: Update CodeGraph API
**File**: `orchestrator/api/codegraph.py`
- Remove `get_openai_key()` function
- Update `get_codegraph_service()` to use `UnifiedLLMService`

#### Step 3.4: Update Chatbot LLM API
**File**: `orchestrator/api/chatbot_llm.py`
- Replace direct Anthropic client with `UnifiedLLMService`
- Use `orchestrator_llm.provider` setting (or chatbot-specific setting)

### Phase 4: Update Other Services

#### Step 4.1: Document Processing
**File**: `orchestrator/api/document_processing.py`
- Migrate to use `UnifiedLLMService` with Orchestrator settings

#### Step 4.2: Embeddings Service
**File**: `orchestrator/context_engineering/embeddings.py`
- Migrate to use `UnifiedLLMService` for text embeddings
- Support provider selection via settings

#### Step 4.3: RAG Service
**File**: `orchestrator/services/rag_service.py`
- Migrate to use `UnifiedLLMService`

### Phase 5: Add HuggingFace Support

#### Step 5.1: Create HuggingFace Client
**File**: `orchestrator/services/unified_llm_service/clients/huggingface_client.py`
- Use `huggingface_hub` or `requests` library
- Support HuggingFace Inference API: `https://api-inference.huggingface.co/models/{model}`
- API token from credential: `development_huggingface.api_token`

#### Step 5.2: Update Credential Type
**Status**: ✅ Already exists in `credential_types_seed.json` as `huggingface_api`
- Verify credential type is seeded
- Ensure it has `api_token` field

### Phase 6: Testing & Validation

#### Step 6.1: Unit Tests
- Test each client implementation
- Test credential resolution
- Test lazy loading
- Test error handling

#### Step 6.2: Integration Tests
- Test OrchestratorLLM settings flow
- Test CodeGraph settings flow
- Test credential → service flow
- Test multi-provider scenarios

#### Step 6.3: User Journey Tests
1. User creates OpenAI credential
2. User selects OpenAI in OrchestratorLLM Settings
3. User tests LLM functionality
4. User switches to Anthropic
5. User tests CodeGraph with different provider

## File-by-File Migration Plan

### Files Requiring Migration

#### High Priority (Core Services)
1. ✅ `services/llm_provider.py` - Migrate to UnifiedLLMService
2. ✅ `services/codegraph_service.py` - Use UnifiedLLMService
3. ✅ `api/codegraph.py` - Remove direct API key usage
4. ✅ `api/chatbot_llm.py` - Use UnifiedLLMService

#### Medium Priority (Document Services)
5. `api/document_processing.py` - Migrate LLM calls
6. `api/documents.py` - Migrate LLM calls
7. `services/rag_service.py` - Migrate LLM calls
8. `context_engineering/embeddings.py` - Migrate LLM calls

#### Low Priority (Other Services)
9. `services/nl_to_sql_service.py` - Check LLM usage
10. `services/database_knowledge_service.py` - Check LLM usage
11. `core/llm/orchestrator_llm.py` - Check integration
12. `services/agent_factory.py` - Check LLM usage

### Migration Strategy per File

#### Pattern 1: Direct API Key Usage
```python
# BEFORE
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# AFTER
from services.unified_llm_service import get_unified_llm_service
llm_service = get_unified_llm_service(service_name="orchestrator")
response = llm_service.inference(prompt, model_id)
```

#### Pattern 2: Credential Resolver (Already Migrated)
```python
# BEFORE (Already using resolver)
resolver = get_credential_resolver()
api_key = resolver.get_openai_key()

# AFTER (Use UnifiedLLMService)
from services.unified_llm_service import get_unified_llm_service
llm_service = get_unified_llm_service(service_name="orchestrator")
```

#### Pattern 3: LLMManager Usage
```python
# BEFORE
llm_manager = LLMManager()
response = await llm_manager.generate_response(messages)

# AFTER (Maintain compatibility)
llm_manager = LLMManager()  # Now uses UnifiedLLMService internally
response = await llm_manager.generate_response(messages)
```

## Configuration Schema

### System Settings Required

#### Orchestrator LLM Settings
```json
{
  "category": "orchestrator_llm",
  "settings": [
    {
      "key": "provider",
      "default_value": "openai",
      "validation_rules": {
        "options": ["openai", "anthropic", "google", "azure", "huggingface"]
      }
    },
    {
      "key": "model",
      "default_value": "gpt-4",
      "validation_rules": {
        "depends_on": {"provider": "..."}
      }
    }
  ]
}
```

#### CodeGraph LLM Settings
```json
{
  "category": "codegraph",
  "settings": [
    {
      "key": "provider",
      "default_value": "openai"
    },
    {
      "key": "model",
      "default_value": "gpt-4"
    }
  ]
}
```

### Credential Naming Convention
- **Development**: `development_openai`, `development_anthropic`, etc.
- **Production**: `production_openai`, `production_anthropic`, etc.
- **Pattern**: `{environment}_{provider_type}`

## Error Handling Strategy

### Lazy Loading Errors
```python
# Service initialization doesn't fail
llm_service = get_unified_llm_service(service_name="orchestrator")
# Returns service instance even if credentials not configured

# First use fails gracefully
try:
    response = llm_service.inference("Hello", "gpt-4")
except CredentialNotFoundError as e:
    logger.warning(f"LLM service not configured: {e}")
    # Return user-friendly error
```

### Missing Credentials
- Show warning in UI: "OpenAI credential not configured. Add it in Settings > Credentials"
- Graceful degradation: Service returns error, doesn't crash application

### Missing Settings
- Default to `openai` provider, `gpt-4` model
- Log warning: "Using default LLM configuration"

## Testing Checklist

### Unit Tests
- [ ] Each client implementation (OpenAI, Anthropic, Google, Azure, HuggingFace)
- [ ] UnifiedLLMService initialization
- [ ] Credential resolution
- [ ] Settings retrieval
- [ ] Error handling

### Integration Tests
- [ ] OrchestratorLLM settings → service flow
- [ ] CodeGraph settings → service flow
- [ ] Credential creation → service usage
- [ ] Provider switching
- [ ] Multi-service configuration

### User Journey Tests
- [ ] Create credential → Select provider → Use service
- [ ] Switch provider → Verify service uses new provider
- [ ] Missing credential → See warning → Add credential → Service works
- [ ] CodeGraph with different provider than Orchestrator

## Deployment Checklist

### Pre-Deployment
- [ ] All unit tests passing
- [ ] Integration tests passing
- [ ] Credential types seeded (verify HuggingFace exists)
- [ ] System settings seeded (orchestrator_llm, codegraph)
- [ ] Frontend components updated

### Deployment Steps
1. Deploy new `unified_llm_service` module
2. Migrate existing services (one at a time)
3. Test each service after migration
4. Update frontend settings tabs
5. Verify credential flow
6. Test end-to-end user journey

### Post-Deployment
- [ ] Monitor error logs for credential resolution failures
- [ ] Verify lazy loading works (no startup failures)
- [ ] Check service performance (response times)
- [ ] User feedback on provider selection UI

## Success Criteria

### Functional
- ✅ All LLM services use UnifiedLLMService
- ✅ Orchestrator and CodeGraph can use different providers
- ✅ Credentials stored securely (PRD-18)
- ✅ Settings UI allows provider selection
- ✅ Lazy loading doesn't fail on startup
- ✅ HuggingFace support added

### Non-Functional
- ✅ No direct API key usage in codebase
- ✅ Graceful error handling
- ✅ Backward compatibility maintained (LLMManager still works)
- ✅ Clear user journey (credential → settings → usage)

## Timeline Estimate

- **Phase 1** (Unified Service): 2-3 days
- **Phase 2** (Settings Integration): 1-2 days
- **Phase 3** (Core Services Migration): 2-3 days
- **Phase 4** (Other Services): 1-2 days
- **Phase 5** (HuggingFace): 1 day
- **Phase 6** (Testing): 2-3 days

**Total**: ~10-14 days

## Risks & Mitigations

### Risk 1: Breaking Changes
- **Mitigation**: Maintain backward compatibility, gradual migration

### Risk 2: Credential Resolution Failures
- **Mitigation**: Fallback to environment variables during transition

### Risk 3: Performance Impact
- **Mitigation**: Lazy loading, caching, async operations

### Risk 4: Missing Provider Support
- **Mitigation**: Start with core providers (OpenAI, Anthropic), add others incrementally

## Next Steps

1. ✅ Review and approve plan
2. Create UnifiedLLMService structure
3. Implement provider clients (including HuggingFace)
4. Integrate with system settings
5. Migrate core services
6. Update frontend components
7. Test and validate
8. Deploy incrementally

---

**Document Version**: 1.0  
**Last Updated**: 2025-01-25  
**Status**: Ready for Implementation
