# PRD-27: Multi-Provider LLM Integration (AWS Bedrock & HuggingFace)

**Status:** ✅ Implementation Complete  
**Created:** 2025-11-08  
**Priority:** High  
**Estimated Effort:** 8 hours  
**Actual Effort:** 6 hours

---

## 📋 Overview

### Purpose
Integrate AWS Bedrock and HuggingFace as additional LLM providers to reduce costs and provide more model options for agents, while maintaining the existing OpenAI, Anthropic, and Google provider support.

### Business Value
- **Cost Reduction:** 80-95% cost savings using AWS Bedrock models
- **Model Diversity:** Access to 16+ models across 6 providers
- **Flexibility:** Per-agent model configuration for optimal cost/performance balance
- **Scalability:** Hybrid strategy (premium for orchestration, cost-effective for subtasks)

### Key Results (KRs)
- ✅ KR1: AWS Bedrock provider fully integrated with 4 models
- ✅ KR2: HuggingFace provider integrated with 3 models  
- ✅ KR3: Frontend UI updated to display all providers
- ✅ KR4: Database seeding updated with 16 total models
- ⏳ KR5: Cost tracking per provider implemented
- ⏳ KR6: Testing and validation complete

---

## 🎯 Goals

### Primary Goals
1. ✅ Add AWS Bedrock as a new LLM provider
2. ✅ Expand HuggingFace model offerings
3. ✅ Update frontend to support new providers
4. ✅ Maintain backward compatibility with existing agents
5. ⏳ Enable cost-effective model selection for agents

### Non-Goals
- Removing support for existing providers (OpenAI, Anthropic, Google)
- Auto-migration of existing agents to new providers
- Custom model hosting infrastructure

---

## 🏗️ Technical Architecture

### System Components Modified

#### 1. Backend LLM Provider System
**Files:**
- `orchestrator/services/llm_provider/clients/base.py` - Added `AWS_BEDROCK` enum
- `orchestrator/services/llm_provider/clients/bedrock_client.py` - **NEW** Bedrock implementation
- `orchestrator/services/llm_provider/clients/__init__.py` - Exported BedrockProvider
- `orchestrator/services/llm_provider/manager.py` - Registered Bedrock routing

#### 2. Database Models
**Files:**
- `orchestrator/seeds/seed_models.py` - Added 7 new models

**New Models:**
```
AWS Bedrock (4 models):
- Claude 3.5 Sonnet (via Bedrock) - $3.00/M tokens
- Claude 3 Haiku (via Bedrock) - $0.25/M tokens  
- Llama 3.1 70B (via Bedrock) - $0.99/M tokens
- Llama 3.1 8B (via Bedrock) - $0.22/M tokens

HuggingFace (3 models):
- Mistral 7B Instruct - FREE
- Llama 2 70B Chat - FREE
- Zephyr 7B Beta - FREE
```

#### 3. Frontend UI Components
**Files:**
- `frontend/components/agents/model-selector.tsx` - Provider color mappings
- `frontend/components/agents/agent-roster.tsx` - Provider badges
- All agent management components updated

---

## 🔧 Implementation Details

### AWS Bedrock Provider

#### Model Family Support
```python
# orchestrator/services/llm_provider/clients/bedrock_client.py

Supported Model Families:
1. Anthropic Claude 3 (Haiku, Sonnet, Opus)
   - Native Messages API
   - Function calling support
   - Tool use via toolSpec

2. Meta Llama 3.1 (8B, 70B, 405B)
   - Text generation API
   - Prompt-based function calling

3. Mistral AI (7B, 8x7B, Large)
   - Text generation API
   - Prompt-based function calling

4. Amazon Titan (Express, Lite, Embed)
   - Text generation API
   - No native function calling

5. Cohere Command (R, R+)
   - Text generation API
   - Limited function calling

6. AI21 Jurassic (Mid, Ultra)
   - Text generation API
   - No native function calling
```

#### Key Features
- **Unified API:** Single client for multiple model families
- **Cost Optimization:** Model ID mappings for easy switching
- **Function Calling:** Native support for Claude, prompt-based for others
- **Error Handling:** Retry logic and graceful fallbacks
- **Token Tracking:** Placeholder for usage tracking (TODO: implement)

### HuggingFace Provider

#### Current Implementation
- ✅ Already integrated via `huggingface_client.py`
- ✅ Supports Inference API
- ⚠️ **Limitation:** No native function calling
- 🔄 **Workaround:** Prompt-based tool calling (to be implemented)

#### Enhanced Model Support
Added 3 popular models to database:
1. **Mistral 7B Instruct** - General purpose, fast
2. **Llama 2 70B Chat** - High quality conversations
3. **Zephyr 7B Beta** - Optimized for chat

---

## 📝 Configuration Guide

### Step 1: Add AWS Credentials

#### Option A: Database (Recommended)
```bash
# Navigate to Settings > Credentials in UI
# Add new credential:
Name: development_aws_bedrock
Type: aws_api
AWS Access Key ID: <your-access-key>
AWS Secret Access Key: <your-secret-key>
AWS Region: us-east-1
```

#### Option B: Environment Variables
```bash
# Add to .env file
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
AWS_REGION=us-east-1
```

### Step 2: Install Dependencies
```bash
# Install boto3 for AWS Bedrock
pip install boto3

# Already installed on server (verified)
```

### Step 3: Seed New Models
```bash
# Run the seed script to add models to database
cd /root/automatos-ai/orchestrator
python3 -m seeds.seed_models

# Expected output:
# 🌱 Seeding/updating llm_models table...
# ✅ Successfully seeded 16 models into llm_models table!
```

### Step 4: Verify Installation
```bash
# Test imports
python3 -c "import boto3; from services.llm_provider.clients.bedrock_client import BedrockProvider; print('✅ Bedrock ready')"
```

### Step 5: Create Agent with Bedrock Model

#### Via UI:
1. Navigate to **Agents > Create Agent**
2. In the **Model Configuration** tab:
   - Select provider: `AWS Bedrock`
   - Select model: `Claude 3 Haiku (Bedrock)` or `Llama 3.1 8B (Bedrock)`
   - Adjust temperature, max_tokens as needed
3. Save and activate agent

#### Via API:
```bash
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Cost-Effective Assistant",
    "agent_type": "custom",
    "description": "Agent using cheap Bedrock models",
    "model_config": {
      "provider": "aws_bedrock",
      "model_id": "claude-3-haiku",
      "temperature": 0.7,
      "max_tokens": 2000
    }
  }'
```

---

## 🧪 Testing Procedures

### Test Plan Overview
1. **Unit Tests:** Provider initialization and API calls
2. **Integration Tests:** End-to-end workflow with Bedrock models
3. **Cost Validation:** Verify token tracking and cost calculations
4. **Performance Tests:** Response time comparison across providers
5. **UI Tests:** Frontend model selection and display

### Manual Testing Checklist

#### ✅ Phase 1: Provider Registration
- [x] Verify `AWS_BEDROCK` enum exists in `base.py`
- [x] Verify `BedrockProvider` imported in `__init__.py`
- [x] Verify provider routing in `manager.py`
- [x] Verify credential mapping for `aws_bedrock`

#### ⏳ Phase 2: Model Availability
- [ ] Verify 16 models in database: `SELECT COUNT(*) FROM llm_models;`
- [ ] Verify Bedrock models: `SELECT * FROM llm_models WHERE provider = 'aws_bedrock';`
- [ ] Verify HuggingFace models: `SELECT * FROM llm_models WHERE provider = 'huggingface';`

#### ⏳ Phase 3: API Endpoint Tests
```bash
# Test 1: List all models
curl http://localhost:8000/api/models/ | jq '.[] | {provider, model_id, display_name}'

# Test 2: Create agent with Bedrock model
curl -X POST http://localhost:8000/api/agents \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Bedrock Test Agent",
    "agent_type": "custom",
    "model_config": {
      "provider": "aws_bedrock",
      "model_id": "claude-3-haiku",
      "temperature": 0.7,
      "max_tokens": 1000
    }
  }'

# Test 3: Execute workflow with Bedrock agent
curl -X POST http://localhost:8000/api/workflows/execute \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "Test message using Bedrock",
    "agent_id": <agent_id_from_test_2>
  }'
```

#### ⏳ Phase 4: Frontend UI Tests
- [ ] Navigate to **Agents** page
- [ ] Click **Create Agent**
- [ ] Verify Model selector shows:
  - OpenAI models (green badge)
  - Anthropic models (purple badge)
  - Google models (blue badge)
  - AWS Bedrock models (orange badge)
  - HuggingFace models (yellow badge)
- [ ] Create agent with `Claude 3 Haiku (Bedrock)`
- [ ] Verify agent card shows orange Bedrock badge
- [ ] Open agent details and verify model config

#### ⏳ Phase 5: Cost Tracking
```python
# Test cost calculation
from services.llm_provider.manager import LLMManager
from services.credential_resolver import CredentialResolver

# Initialize
resolver = CredentialResolver(db)
config = {
    "provider": "aws_bedrock",
    "model_name": "claude-3-haiku",
    "api_key": resolver.get_credential("development_aws_bedrock")
}

# Create manager and test
manager = LLMManager(config)
response = await manager.generate("Test prompt", max_tokens=100)

# Verify response includes token_usage
assert "token_usage" in response
assert "total_tokens" in response["token_usage"]
```

---

## 📊 Cost Comparison

### Current vs. New Pricing
| Model | Provider | Input ($/M) | Output ($/M) | Use Case | Savings |
|-------|----------|-------------|--------------|----------|---------|
| **GPT-4** | OpenAI | $30.00 | $60.00 | Premium orchestration | Baseline |
| **Claude 3.5 Sonnet (Bedrock)** | AWS | $3.00 | $15.00 | Orchestration | 80-90% |
| **Claude 3 Haiku (Bedrock)** | AWS | $0.25 | $1.25 | High-volume tasks | 95-99% |
| **Llama 3.1 70B (Bedrock)** | AWS | $0.99 | $0.99 | Balanced tasks | 90-97% |
| **Llama 3.1 8B (Bedrock)** | AWS | $0.22 | $0.22 | Simple tasks | 99%+ |
| **Mistral 7B** | HuggingFace | FREE | FREE | Development/testing | 100% |

### Hybrid Strategy Example
```
Monthly Usage: 1M tokens/day = 30M tokens/month

Current Cost (all GPT-4):
- 30M tokens × $30/M = $900/month

New Hybrid Strategy:
- Orchestration (10%): 3M tokens × $3/M (Bedrock Sonnet) = $9
- Subtasks (90%): 27M tokens × $0.25/M (Bedrock Haiku) = $6.75
Total: $15.75/month

Monthly Savings: $884.25 (98.25% reduction)
Annual Savings: $10,611
```

---

## 🚀 Deployment Checklist

### Pre-Deployment
- [x] Code changes committed and reviewed
- [x] Database migration script prepared (`seed_models.py`)
- [x] Dependencies documented (`boto3`)
- [x] Configuration guide written
- [ ] Tests written and passing
- [ ] Documentation updated

### Deployment Steps
1. [x] Deploy backend code changes
   ```bash
   scp -i ~/.ssh/id_rsa orchestrator/services/llm_provider/clients/*.py root@206.81.0.227:/root/automatos-ai/orchestrator/services/llm_provider/clients/
   scp -i ~/.ssh/id_rsa orchestrator/services/llm_provider/manager.py root@206.81.0.227:/root/automatos-ai/orchestrator/services/llm_provider/
   scp -i ~/.ssh/id_rsa orchestrator/seeds/seed_models.py root@206.81.0.227:/root/automatos-ai/orchestrator/seeds/
   ```

2. [x] Install dependencies
   ```bash
   ssh root@206.81.0.227 "cd /root/automatos-ai/orchestrator && pip install boto3"
   ```

3. [x] Run database migrations
   ```bash
   ssh root@206.81.0.227 "cd /root/automatos-ai/orchestrator && python3 -m seeds.seed_models"
   ```

4. [x] Deploy frontend changes
   ```bash
   scp -i ~/.ssh/id_rsa frontend/components/agents/*.tsx root@206.81.0.227:/root/automatos-ai/frontend/components/agents/
   ```

5. [ ] Clear Python cache and restart services
   ```bash
   ssh root@206.81.0.227 "cd /root/automatos-ai/orchestrator && find . -type d -name '__pycache__' -exec rm -rf {} + 2>/dev/null; find . -name '*.pyc' -delete 2>/dev/null"
   ssh root@206.81.0.227 "systemctl restart automatos-backend automatos-frontend"
   ```

6. [ ] Verify deployment
   ```bash
   # Check backend is running
   curl http://206.81.0.227:8000/api/health
   
   # Check models are available
   curl http://206.81.0.227:8000/api/models/ | jq '. | length'
   # Should return: 16
   ```

### Post-Deployment
- [ ] Run smoke tests
- [ ] Monitor logs for errors
- [ ] Create test agent with Bedrock model
- [ ] Execute test workflow
- [ ] Verify cost tracking
- [ ] Update documentation

---

## 🔍 Monitoring & Observability

### Key Metrics to Track
1. **Provider Usage**
   - Requests per provider
   - Token consumption per provider
   - Cost per provider

2. **Performance**
   - Response time by provider
   - Error rate by provider
   - Timeout rate

3. **Cost Efficiency**
   - Daily/monthly spend by provider
   - Cost per workflow execution
   - Savings vs. baseline (GPT-4)

### Logging
```python
# Add logging in manager.py
logger.info(f"Using provider: {self.config.provider.value}, model: {self.config.model_name}")
logger.info(f"Token usage: {response.token_usage}, estimated cost: ${cost:.4f}")
```

### Alerts
- Cost exceeds $100/day
- Error rate > 5% for any provider
- Response time > 10s average

---

## 📚 API Reference

### New Endpoints
None - existing endpoints now support new providers

### Updated Models

#### `AgentModelConfig`
```python
{
  "provider": "aws_bedrock",  # NEW: Now supports aws_bedrock
  "model_id": "claude-3-haiku",  # Friendly name (mapped to full ID)
  "temperature": 0.7,
  "max_tokens": 2000,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "fallback_model_id": null
}
```

#### Model ID Mappings (Bedrock)
```python
MODEL_IDS = {
    # Friendly name -> Full Bedrock ID
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "llama-3-1-70b": "meta.llama3-1-70b-instruct-v1:0",
    "llama-3-1-8b": "meta.llama3-1-8b-instruct-v1:0",
}
```

---

## 🐛 Known Issues & Limitations

### Limitations
1. **Token Tracking:** Bedrock token counts are placeholders (TODO: implement)
2. **Function Calling:** Only Claude models on Bedrock have native support
3. **HuggingFace:** No native function calling support
4. **Rate Limits:** AWS Bedrock has model-specific rate limits
5. **Availability:** Some models (e.g., Llama 3.1 405B) are in preview

### Workarounds
1. **Token Tracking:** Estimate based on character count until API returns counts
2. **Function Calling:** Use prompt engineering for non-Claude models
3. **HuggingFace:** Use for non-tool tasks or implement prompt-based calling
4. **Rate Limits:** Implement exponential backoff (already in boto3 config)

---

## 🎯 Success Criteria

### Must Have (MVP)
- [x] AWS Bedrock provider integrated
- [x] 4+ Bedrock models available
- [x] Frontend UI supports new providers
- [ ] Basic testing complete
- [ ] Documentation complete

### Should Have
- [ ] Cost tracking per provider
- [ ] Performance benchmarking
- [ ] Automated tests
- [ ] Migration guide for existing agents

### Nice to Have
- [ ] Auto-model selection based on task complexity
- [ ] Cost optimization recommendations
- [ ] Provider health dashboard
- [ ] A/B testing framework

---

## 📅 Timeline

| Phase | Duration | Status | Completion Date |
|-------|----------|--------|-----------------|
| **Phase 1:** Backend Integration | 3 hours | ✅ Complete | 2025-11-08 |
| **Phase 2:** Database & Models | 1 hour | ✅ Complete | 2025-11-08 |
| **Phase 3:** Frontend UI | 2 hours | ✅ Complete | 2025-11-08 |
| **Phase 4:** Testing | 2 hours | ⏳ In Progress | - |
| **Phase 5:** Documentation | 1 hour | ⏳ In Progress | - |
| **Total** | **9 hours** | **67% Complete** | - |

---

## 🔗 Related PRDs

- **PRD-15:** Multi-Model Agent (foundation for this work)
- **PRD-18:** Credential Management (AWS creds storage)
- **PRD-20:** MCP Credential (tool credential management)
- **PRD-16:** LLM-Driven Orchestrator (uses these providers)

---

## 📞 Support & Contact

### Questions?
- Technical Issues: Check logs in `/root/automatos-ai/orchestrator/logs/`
- Configuration Help: Review `MULTI_PROVIDER_STRATEGY.md`
- Cost Questions: Review `BEDROCK_VS_DIRECT_COMPARISON.md`

### Resources
- AWS Bedrock Docs: https://docs.aws.amazon.com/bedrock/
- HuggingFace API: https://huggingface.co/docs/api-inference/
- boto3 Documentation: https://boto3.amazonaws.com/v1/documentation/api/latest/index.html

---

## 📝 Change Log

### v1.0.0 - 2025-11-08
- ✅ Initial implementation complete
- ✅ AWS Bedrock provider added
- ✅ 4 Bedrock models seeded
- ✅ 3 HuggingFace models seeded
- ✅ Frontend UI updated
- ⏳ Testing in progress

### v1.1.0 - TBD
- [ ] Token tracking implementation
- [ ] Cost analytics dashboard
- [ ] Automated tests
- [ ] Performance benchmarking

---

**Last Updated:** 2025-11-08  
**Document Owner:** Platform Team  
**Reviewers:** Backend Team, Frontend Team, DevOps

