# PRD 15: Multi-Model Agent Configuration

**Status:** Ready for Implementation  
**Priority:** P1 - High Priority Feature Enhancement  
**Effort:** 24-32 hours (3-4 days)  
**Dependencies:** PRD-02 (Agent Factory), PRD-10 (Workflow Orchestration)

---

## Executive Summary

Transform agents from defaulting to GPT-4 to supporting **user-configurable multi-model selection** across OpenAI, Claude (Anthropic), and future HuggingFace models. Each agent will have its own model configuration, allowing users to select the optimal model based on agent skills, task complexity, and cost considerations.

### Current State ❌
- ✅ Agents exist and execute tasks via LLM
- ✅ Basic model configuration in AgentMetadata (preferred_model)
- ✅ LLM Provider abstraction supports OpenAI and Anthropic
- ❌ All agents default to GPT-4 from environment variables
- ❌ No UI to select models per agent
- ❌ Limited model configuration options
- ❌ No model metadata or capabilities tracking
- ❌ Model configuration scattered across env vars and agent metadata

### Target State ✅
- ✅ Rich model configuration per agent
- ✅ UI to select from available models (OpenAI, Claude, future HuggingFace)
- ✅ Model metadata with capabilities, costs, and limits
- ✅ Database schema for model configurations
- ✅ Agent factory uses per-agent model configs
- ✅ Workflow execution respects agent model choices
- ✅ Model settings UI tab in agent configuration
- ✅ Default models with intelligent recommendations

---

## 1. Problem Statement

### Current Issues
1. **No Model Choice**: All agents use GPT-4 by default, regardless of task requirements
2. **Cost Inefficiency**: Simple tasks use expensive models unnecessarily
3. **No Optimization**: Can't match model capabilities to agent skills
4. **Limited Flexibility**: Changing models requires code/env changes
5. **Poor UX**: Users can't easily select models in UI
6. **No Model Tracking**: No visibility into which models agents use

### Business Impact
- **Cost Optimization**: Use cheaper models for simple tasks (GPT-3.5 vs GPT-4)
- **Performance Optimization**: Use Claude for reasoning-heavy tasks
- **Future-Proofing**: Ready for HuggingFace and custom models
- **User Control**: Empower users to optimize their workflows
- **Transparency**: Track model usage and costs per agent

---

## 2. Solution Overview

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   MULTI-MODEL AGENT SYSTEM                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. MODEL REGISTRY                                               │
│     └─> Available models (OpenAI, Claude, HuggingFace)          │
│     └─> Model metadata (capabilities, costs, limits)            │
│     └─> Model versions and deprecation tracking                 │
│                                                                  │
│  2. AGENT MODEL CONFIGURATION                                    │
│     └─> Per-agent model selection                               │
│     └─> Model-specific parameters (temp, max_tokens, etc)       │
│     └─> Provider credentials reference                          │
│     └─> Fallback model configuration                            │
│                                                                  │
│  3. UI COMPONENTS                                                │
│     └─> Model selection dropdown (with metadata)                │
│     └─> Model settings form (provider-specific)                 │
│     └─> Model comparison view                                   │
│     └─> Usage analytics per model                               │
│                                                                  │
│  4. AGENT FACTORY INTEGRATION                                    │
│     └─> Load model config from agent metadata                   │
│     └─> Initialize correct provider                             │
│     └─> Validate model availability                             │
│     └─> Handle fallbacks on errors                              │
│                                                                  │
│  5. WORKFLOW EXECUTION                                           │
│     └─> Respect agent model choices                             │
│     └─> Track model usage per execution                         │
│     └─> Cost estimation and tracking                            │
│     └─> Performance metrics per model                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Database Schema Updates

### 3.1 New Model Registry Table

```sql
-- Model registry for available LLM models
CREATE TABLE llm_models (
    id SERIAL PRIMARY KEY,
    provider VARCHAR(50) NOT NULL,  -- 'openai', 'anthropic', 'huggingface'
    model_id VARCHAR(255) NOT NULL UNIQUE,  -- 'gpt-4', 'claude-3-opus-20240229', etc.
    display_name VARCHAR(255) NOT NULL,  -- Human-readable name
    model_family VARCHAR(100),  -- 'gpt-4', 'claude-3', 'llama-2', etc.
    
    -- Capabilities
    capabilities JSONB DEFAULT '{}',  -- {reasoning: 'high', coding: 'excellent', ...}
    context_window INTEGER NOT NULL,  -- Max context tokens
    max_output_tokens INTEGER NOT NULL,  -- Max output tokens
    supports_functions BOOLEAN DEFAULT FALSE,
    supports_vision BOOLEAN DEFAULT FALSE,
    supports_streaming BOOLEAN DEFAULT TRUE,
    
    -- Cost information
    input_cost_per_1k_tokens DECIMAL(10, 6),  -- Cost per 1K input tokens
    output_cost_per_1k_tokens DECIMAL(10, 6),  -- Cost per 1K output tokens
    
    -- Metadata
    description TEXT,
    release_date DATE,
    deprecation_date DATE,
    status VARCHAR(50) DEFAULT 'active',  -- 'active', 'deprecated', 'beta'
    recommended_for JSONB DEFAULT '[]',  -- ['code_analysis', 'creative_writing', ...]
    
    -- Settings
    default_temperature DECIMAL(3, 2) DEFAULT 0.7,
    min_temperature DECIMAL(3, 2) DEFAULT 0.0,
    max_temperature DECIMAL(3, 2) DEFAULT 2.0,
    
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_llm_models_provider ON llm_models(provider);
CREATE INDEX idx_llm_models_status ON llm_models(status);
CREATE INDEX idx_llm_models_model_family ON llm_models(model_family);

-- Insert default models
INSERT INTO llm_models (provider, model_id, display_name, model_family, context_window, max_output_tokens, 
    input_cost_per_1k_tokens, output_cost_per_1k_tokens, capabilities, recommended_for, supports_functions) 
VALUES 
    -- OpenAI Models
    ('openai', 'gpt-4-turbo-preview', 'GPT-4 Turbo', 'gpt-4', 128000, 4096, 0.01, 0.03, 
        '{"reasoning": "excellent", "coding": "excellent", "analysis": "excellent"}', 
        '["code_analysis", "complex_reasoning", "system_design"]', TRUE),
    ('openai', 'gpt-4', 'GPT-4', 'gpt-4', 8192, 4096, 0.03, 0.06, 
        '{"reasoning": "excellent", "coding": "excellent", "analysis": "excellent"}', 
        '["code_review", "security_audit", "architecture"]', TRUE),
    ('openai', 'gpt-3.5-turbo', 'GPT-3.5 Turbo', 'gpt-3.5', 16385, 4096, 0.0005, 0.0015, 
        '{"reasoning": "good", "coding": "good", "speed": "fast"}', 
        '["simple_tasks", "data_processing", "quick_responses"]', TRUE),
    
    -- Anthropic Models
    ('anthropic', 'claude-3-opus-20240229', 'Claude 3 Opus', 'claude-3', 200000, 4096, 0.015, 0.075, 
        '{"reasoning": "excellent", "analysis": "excellent", "creativity": "excellent"}', 
        '["complex_analysis", "research", "planning"]', FALSE),
    ('anthropic', 'claude-3-sonnet-20240229', 'Claude 3 Sonnet', 'claude-3', 200000, 4096, 0.003, 0.015, 
        '{"reasoning": "excellent", "balance": "optimal", "speed": "fast"}', 
        '["balanced_tasks", "general_purpose", "workflows"]', FALSE),
    ('anthropic', 'claude-3-haiku-20240307', 'Claude 3 Haiku', 'claude-3', 200000, 4096, 0.00025, 0.00125, 
        '{"speed": "fastest", "cost": "lowest", "reasoning": "good"}', 
        '["high_volume", "simple_tasks", "cost_sensitive"]', FALSE);
```

### 3.2 Update Agents Table

```sql
-- Add model configuration to agents table
ALTER TABLE agents 
    ADD COLUMN model_config JSONB DEFAULT '{
        "provider": "openai",
        "model_id": "gpt-4",
        "temperature": 0.7,
        "max_tokens": 2000,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "fallback_model_id": "gpt-3.5-turbo"
    }';

-- Add model usage tracking
ALTER TABLE agents 
    ADD COLUMN model_usage_stats JSONB DEFAULT '{
        "total_tokens": 0,
        "total_cost": 0.0,
        "total_requests": 0,
        "avg_tokens_per_request": 0,
        "last_used_at": null
    }';

-- Create index for querying by model
CREATE INDEX idx_agents_model_config ON agents USING GIN (model_config);
```

### 3.3 Track Model Usage Per Execution

```sql
-- Add model tracking to workflow executions
ALTER TABLE workflow_executions 
    ADD COLUMN models_used JSONB DEFAULT '[]';  -- Array of {agent_id, model_id, tokens, cost}

-- Example structure:
-- [
--   {"agent_id": 5, "model_id": "gpt-4", "input_tokens": 1500, "output_tokens": 800, "cost": 0.051},
--   {"agent_id": 8, "model_id": "claude-3-sonnet", "input_tokens": 2000, "output_tokens": 1200, "cost": 0.024}
-- ]
```

---

## 4. Backend Implementation

### 4.1 Model Registry Service

**Location**: `orchestrator/services/model_registry.py`

```python
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum
from sqlalchemy.orm import Session
from database.models import LLMModel

class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    HUGGINGFACE = "huggingface"

@dataclass
class ModelInfo:
    """Model information with capabilities"""
    id: int
    provider: str
    model_id: str
    display_name: str
    model_family: str
    context_window: int
    max_output_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float
    capabilities: Dict[str, str]
    recommended_for: List[str]
    supports_functions: bool
    supports_vision: bool
    status: str
    
    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for token usage"""
        input_cost = (input_tokens / 1000) * self.input_cost_per_1k
        output_cost = (output_tokens / 1000) * self.output_cost_per_1k
        return round(input_cost + output_cost, 4)

class ModelRegistry:
    """
    Registry for managing available LLM models
    """
    
    def __init__(self, db_session: Session):
        self.db = db_session
        self._cache: Optional[Dict[str, ModelInfo]] = None
    
    def get_all_models(self, provider: Optional[str] = None, 
                       status: str = 'active') -> List[ModelInfo]:
        """Get all available models"""
        query = self.db.query(LLMModel).filter(LLMModel.status == status)
        
        if provider:
            query = query.filter(LLMModel.provider == provider)
        
        models = query.order_by(LLMModel.provider, LLMModel.display_name).all()
        return [self._model_to_info(m) for m in models]
    
    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """Get specific model by ID"""
        model = self.db.query(LLMModel).filter(
            LLMModel.model_id == model_id
        ).first()
        
        if model:
            return self._model_to_info(model)
        return None
    
    def get_models_by_provider(self, provider: str) -> List[ModelInfo]:
        """Get all models for a specific provider"""
        return self.get_all_models(provider=provider)
    
    def get_recommended_models(self, task_type: str) -> List[ModelInfo]:
        """Get models recommended for a specific task type"""
        models = self.get_all_models()
        return [
            m for m in models 
            if task_type in m.recommended_for
        ]
    
    def find_best_model(self, 
                       requirements: Dict[str, Any]) -> Optional[ModelInfo]:
        """
        Find best model based on requirements
        
        Args:
            requirements: {
                'task_type': 'code_analysis',
                'max_cost': 0.05,
                'min_context': 8000,
                'required_capabilities': ['reasoning', 'coding'],
                'prefer_provider': 'openai'
            }
        """
        models = self.get_all_models()
        
        # Filter by requirements
        candidates = []
        for model in models:
            # Check context window
            if requirements.get('min_context') and model.context_window < requirements['min_context']:
                continue
            
            # Check task type recommendation
            if requirements.get('task_type') and requirements['task_type'] not in model.recommended_for:
                continue
            
            # Check capabilities
            required_caps = requirements.get('required_capabilities', [])
            model_caps = list(model.capabilities.keys())
            if not all(cap in model_caps for cap in required_caps):
                continue
            
            # Calculate score
            score = 0.0
            
            # Prefer recommended provider
            if requirements.get('prefer_provider') and model.provider == requirements['prefer_provider']:
                score += 10
            
            # Prefer lower cost
            avg_cost = (model.input_cost_per_1k + model.output_cost_per_1k) / 2
            if avg_cost < 0.01:
                score += 5
            elif avg_cost < 0.05:
                score += 3
            
            # Prefer larger context
            if model.context_window >= 100000:
                score += 3
            elif model.context_window >= 32000:
                score += 2
            
            candidates.append((model, score))
        
        if not candidates:
            return None
        
        # Return best scoring model
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[0][0]
    
    def _model_to_info(self, model: LLMModel) -> ModelInfo:
        """Convert database model to ModelInfo"""
        return ModelInfo(
            id=model.id,
            provider=model.provider,
            model_id=model.model_id,
            display_name=model.display_name,
            model_family=model.model_family,
            context_window=model.context_window,
            max_output_tokens=model.max_output_tokens,
            input_cost_per_1k=float(model.input_cost_per_1k_tokens or 0),
            output_cost_per_1k=float(model.output_cost_per_1k_tokens or 0),
            capabilities=model.capabilities or {},
            recommended_for=model.recommended_for or [],
            supports_functions=model.supports_functions,
            supports_vision=model.supports_vision,
            status=model.status
        )
```

### 4.2 Enhanced AgentMetadata

**Location**: Update `orchestrator/services/agent_factory.py`

```python
@dataclass
class ModelConfiguration:
    """Complete model configuration for an agent"""
    provider: str  # 'openai', 'anthropic', 'huggingface'
    model_id: str  # 'gpt-4', 'claude-3-sonnet-20240229', etc.
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    fallback_model_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "fallback_model_id": self.fallback_model_id
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ModelConfiguration':
        return ModelConfiguration(
            provider=data.get("provider", "openai"),
            model_id=data.get("model_id", "gpt-4"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 2000),
            top_p=data.get("top_p", 1.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            fallback_model_id=data.get("fallback_model_id")
        )

@dataclass
class AgentMetadata:
    """Enhanced agent metadata with full model configuration"""
    name: str
    agent_type: str
    description: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    
    # New: Full model configuration
    model_config: Optional[ModelConfiguration] = None
    
    # Deprecated: Keep for backward compatibility
    preferred_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_model_config(self) -> ModelConfiguration:
        """Get model configuration with fallbacks"""
        # Use new model_config if available
        if self.model_config:
            return self.model_config
        
        # Fall back to deprecated fields
        if self.preferred_model:
            provider = "openai"
            if "claude" in self.preferred_model.lower():
                provider = "anthropic"
            
            return ModelConfiguration(
                provider=provider,
                model_id=self.preferred_model,
                temperature=self.temperature or 0.7,
                max_tokens=self.max_tokens or 2000
            )
        
        # Use default
        return ModelConfiguration(
            provider="openai",
            model_id="gpt-4",
            temperature=0.7,
            max_tokens=2000
        )
    
    def get_llm_config(self) -> Dict[str, Any]:
        """Get LLM configuration dict (backward compatible)"""
        model_config = self.get_model_config()
        return model_config.to_dict()
```

### 4.3 Update Agent Factory

**Location**: Update `orchestrator/services/agent_factory.py`

```python
class AgentFactory:
    """Enhanced agent factory with multi-model support"""
    
    async def create_agent(
        self,
        metadata: Union[AgentMetadata, Dict[str, Any]],
        auto_verify: bool = True
    ) -> AgentRuntime:
        """Create agent with model configuration"""
        start_time = time.time()
        
        # Convert dict to AgentMetadata if needed
        if isinstance(metadata, dict):
            # Handle model_config
            model_config = None
            if "model_config" in metadata:
                model_config = ModelConfiguration.from_dict(metadata["model_config"])
            
            metadata = AgentMetadata(
                name=metadata.get("name", "Unnamed Agent"),
                agent_type=metadata.get("type", "generic"),
                description=metadata.get("description"),
                skills=metadata.get("skills", []),
                model_config=model_config,
                # Deprecated fields for backward compatibility
                preferred_model=metadata.get("preferred_model"),
                temperature=metadata.get("temperature"),
                max_tokens=metadata.get("max_tokens"),
                context_window=metadata.get("context_window"),
                custom_metadata=metadata.get("metadata", {})
            )
        
        # Get model configuration
        model_config = metadata.get_model_config()
        
        # Create database record with model config
        db_agent = Agent(
            name=metadata.name,
            description=metadata.description or f"User-defined {metadata.agent_type} agent",
            agent_type=metadata.agent_type,
            status=AgentLifecycle.INITIALIZING.value,
            configuration={
                "skills": metadata.skills,
                "custom_metadata": metadata.custom_metadata
            },
            model_config=model_config.to_dict(),  # NEW: Store model config
            priority_level=PriorityLevel.MEDIUM.value,
            max_concurrent_tasks=5,
            auto_start=False,
            created_by="agent_factory"
        )
        
        self.db_session.add(db_agent)
        self.db_session.commit()
        
        # Initialize LLM connection with model config
        try:
            llm_manager = await self._create_llm_manager(model_config)
            
            # Verify LLM connection if requested
            if auto_verify:
                await self._verify_llm_connection(llm_manager, db_agent)
            
            # Create agent runtime
            agent_runtime = AgentRuntime(
                agent_id=db_agent.id,
                agent=db_agent,
                llm_manager=llm_manager,
                lifecycle_state=AgentLifecycle.ACTIVE,
                short_term_memory=deque(maxlen=10),
                metadata=metadata
            )
            
            # Store in active agents
            self.active_agents[db_agent.id] = agent_runtime
            
            # Update status
            db_agent.status = AgentLifecycle.ACTIVE.value
            self.db_session.commit()
            
            execution_time = time.time() - start_time
            self.logger.info(
                f"✅ Agent created successfully: id={db_agent.id}, "
                f"name={metadata.name}, model={model_config.model_id}, "
                f"provider={model_config.provider}, time={execution_time:.2f}s"
            )
            
            return agent_runtime
            
        except Exception as e:
            # Try fallback model if configured
            if model_config.fallback_model_id:
                self.logger.warning(
                    f"Primary model {model_config.model_id} failed, "
                    f"trying fallback {model_config.fallback_model_id}"
                )
                fallback_config = ModelConfiguration(
                    provider=model_config.provider,
                    model_id=model_config.fallback_model_id,
                    temperature=model_config.temperature,
                    max_tokens=model_config.max_tokens
                )
                llm_manager = await self._create_llm_manager(fallback_config)
                # ... continue with fallback
            else:
                # Mark as failed
                db_agent.status = AgentLifecycle.RETIRED.value
                self.db_session.commit()
                raise
    
    async def _create_llm_manager(self, model_config: ModelConfiguration) -> LLMManager:
        """Create LLM manager from model configuration"""
        from services.llm_provider import LLMConfig, LLMProvider
        
        # Map provider string to enum
        provider_map = {
            "openai": LLMProvider.OPENAI,
            "anthropic": LLMProvider.ANTHROPIC
        }
        
        provider = provider_map.get(model_config.provider, LLMProvider.OPENAI)
        
        # Get API key
        api_key = None
        if provider == LLMProvider.OPENAI:
            api_key = os.getenv("OPENAI_API_KEY")
        elif provider == LLMProvider.ANTHROPIC:
            api_key = os.getenv("ANTHROPIC_API_KEY")
        
        # Create LLM config
        llm_config = LLMConfig(
            provider=provider,
            model=model_config.model_id,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            api_key=api_key
        )
        
        return LLMManager(config=llm_config)
```

### 4.4 API Endpoints

**Location**: `orchestrator/api/models_endpoints.py` (NEW FILE)

```python
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from database.database import get_db
from services.model_registry import ModelRegistry, ModelInfo

router = APIRouter(prefix="/api/models", tags=["models"])

@router.get("/", response_model=List[Dict[str, Any]])
async def list_models(
    provider: Optional[str] = None,
    status: str = 'active',
    db: Session = Depends(get_db)
):
    """
    List all available LLM models
    
    Query Parameters:
    - provider: Filter by provider (openai, anthropic, huggingface)
    - status: Filter by status (active, deprecated, beta)
    """
    registry = ModelRegistry(db)
    models = registry.get_all_models(provider=provider, status=status)
    
    return [
        {
            "id": m.id,
            "provider": m.provider,
            "model_id": m.model_id,
            "display_name": m.display_name,
            "model_family": m.model_family,
            "context_window": m.context_window,
            "max_output_tokens": m.max_output_tokens,
            "input_cost_per_1k": m.input_cost_per_1k,
            "output_cost_per_1k": m.output_cost_per_1k,
            "capabilities": m.capabilities,
            "recommended_for": m.recommended_for,
            "supports_functions": m.supports_functions,
            "supports_vision": m.supports_vision,
            "status": m.status
        }
        for m in models
    ]

@router.get("/{model_id}")
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get specific model details"""
    registry = ModelRegistry(db)
    model = registry.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    return {
        "id": model.id,
        "provider": model.provider,
        "model_id": model.model_id,
        "display_name": model.display_name,
        "model_family": model.model_family,
        "context_window": model.context_window,
        "max_output_tokens": model.max_output_tokens,
        "input_cost_per_1k": model.input_cost_per_1k,
        "output_cost_per_1k": model.output_cost_per_1k,
        "capabilities": model.capabilities,
        "recommended_for": model.recommended_for,
        "supports_functions": model.supports_functions,
        "supports_vision": model.supports_vision,
        "status": model.status
    }

@router.get("/providers/")
async def list_providers(db: Session = Depends(get_db)):
    """List available providers"""
    registry = ModelRegistry(db)
    models = registry.get_all_models()
    
    providers = {}
    for model in models:
        if model.provider not in providers:
            providers[model.provider] = {
                "name": model.provider,
                "model_count": 0,
                "models": []
            }
        providers[model.provider]["model_count"] += 1
        providers[model.provider]["models"].append({
            "model_id": model.model_id,
            "display_name": model.display_name
        })
    
    return list(providers.values())

@router.post("/recommend")
async def recommend_model(
    requirements: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Recommend best model based on requirements
    
    Body:
    {
        "task_type": "code_analysis",
        "max_cost": 0.05,
        "min_context": 8000,
        "required_capabilities": ["reasoning", "coding"],
        "prefer_provider": "openai"
    }
    """
    registry = ModelRegistry(db)
    model = registry.find_best_model(requirements)
    
    if not model:
        raise HTTPException(
            status_code=404, 
            detail="No model found matching requirements"
        )
    
    return {
        "model_id": model.model_id,
        "display_name": model.display_name,
        "provider": model.provider,
        "reason": "Best match for requirements"
    }

@router.post("/estimate-cost")
async def estimate_cost(
    request: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """
    Estimate cost for model usage
    
    Body:
    {
        "model_id": "gpt-4",
        "input_tokens": 1500,
        "output_tokens": 800
    }
    """
    model_id = request.get("model_id")
    input_tokens = request.get("input_tokens", 0)
    output_tokens = request.get("output_tokens", 0)
    
    registry = ModelRegistry(db)
    model = registry.get_model(model_id)
    
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    cost = model.estimate_cost(input_tokens, output_tokens)
    
    return {
        "model_id": model_id,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost": cost,
        "currency": "USD"
    }
```

**Update** `orchestrator/api/agent_endpoints.py`:

```python
@router.get("/{agent_id}/model-usage")
async def get_agent_model_usage(agent_id: int, db: Session = Depends(get_db)):
    """Get model usage statistics for an agent"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return {
        "agent_id": agent_id,
        "current_model": agent.model_config,
        "usage_stats": agent.model_usage_stats or {},
        "configuration": {
            "provider": agent.model_config.get("provider"),
            "model_id": agent.model_config.get("model_id"),
            "temperature": agent.model_config.get("temperature"),
            "max_tokens": agent.model_config.get("max_tokens")
        }
    }

@router.put("/{agent_id}/model-config")
async def update_agent_model_config(
    agent_id: int,
    model_config: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Update agent's model configuration"""
    agent = db.query(Agent).filter(Agent.id == agent_id).first()
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    # Validate model exists
    registry = ModelRegistry(db)
    model = registry.get_model(model_config.get("model_id"))
    if not model:
        raise HTTPException(status_code=400, detail="Invalid model_id")
    
    # Update model config
    agent.model_config = model_config
    attributes.flag_modified(agent, "model_config")
    db.commit()
    
    return {
        "message": "Model configuration updated",
        "agent_id": agent_id,
        "model_config": agent.model_config
    }
```

---

## 5. Frontend Implementation

### 5.1 Model Selection Component

**Location**: `frontend/components/agents/model-selector.tsx` (NEW FILE)

```typescript
'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  Brain, 
  Zap, 
  DollarSign, 
  Info,
  Check,
  AlertCircle
} from 'lucide-react'
import {
  Select,
  SelectContent,
  SelectGroup,
  SelectItem,
  SelectLabel,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Card, CardContent } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'

interface Model {
  id: number
  provider: string
  model_id: string
  display_name: string
  model_family: string
  context_window: number
  max_output_tokens: number
  input_cost_per_1k: number
  output_cost_per_1k: number
  capabilities: Record<string, string>
  recommended_for: string[]
  supports_functions: boolean
  status: string
}

interface ModelSelectorProps {
  value: string  // model_id
  onChange: (modelId: string) => void
  agentType?: string
  provider?: string
}

export function ModelSelector({ value, onChange, agentType, provider }: ModelSelectorProps) {
  const [models, setModels] = useState<Model[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedModel, setSelectedModel] = useState<Model | null>(null)

  useEffect(() => {
    fetchModels()
  }, [provider])

  useEffect(() => {
    if (value && models.length > 0) {
      const model = models.find(m => m.model_id === value)
      setSelectedModel(model || null)
    }
  }, [value, models])

  const fetchModels = async () => {
    try {
      setLoading(true)
      const url = provider 
        ? `/api/models?provider=${provider}` 
        : '/api/models'
      const response = await fetch(url)
      const data = await response.json()
      setModels(data)
    } catch (error) {
      console.error('Failed to fetch models:', error)
    } finally {
      setLoading(false)
    }
  }

  const getProviderColor = (provider: string) => {
    const colors: Record<string, string> = {
      openai: 'bg-green-500/10 text-green-400 border-green-500/20',
      anthropic: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
      huggingface: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/20'
    }
    return colors[provider] || 'bg-gray-500/10 text-gray-400 border-gray-500/20'
  }

  const getCapabilityIcon = (capability: string, rating: string) => {
    const isExcellent = rating.toLowerCase().includes('excellent')
    const isGood = rating.toLowerCase().includes('good')
    
    return (
      <Badge 
        variant="outline" 
        className={`
          ${isExcellent ? 'border-green-500/30 text-green-400' : ''}
          ${isGood ? 'border-blue-500/30 text-blue-400' : ''}
        `}
      >
        {capability}: {rating}
      </Badge>
    )
  }

  // Group models by provider
  const groupedModels = models.reduce((acc, model) => {
    if (!acc[model.provider]) {
      acc[model.provider] = []
    }
    acc[model.provider].push(model)
    return acc
  }, {} as Record<string, Model[]>)

  return (
    <div className="space-y-4">
      {/* Model Selector */}
      <Select value={value} onValueChange={onChange} disabled={loading}>
        <SelectTrigger className="w-full bg-black/20 border-gray-800">
          <SelectValue placeholder="Select a model..." />
        </SelectTrigger>
        <SelectContent>
          {Object.entries(groupedModels).map(([provider, providerModels]) => (
            <SelectGroup key={provider}>
              <SelectLabel className="text-xs uppercase text-gray-400">
                {provider}
              </SelectLabel>
              {providerModels.map((model) => (
                <SelectItem key={model.model_id} value={model.model_id}>
                  <div className="flex items-center gap-2">
                    <span>{model.display_name}</span>
                    {model.status === 'beta' && (
                      <Badge variant="outline" className="text-xs">Beta</Badge>
                    )}
                    {agentType && model.recommended_for.includes(agentType) && (
                      <TooltipProvider>
                        <Tooltip>
                          <TooltipTrigger>
                            <Check className="h-3 w-3 text-green-400" />
                          </TooltipTrigger>
                          <TooltipContent>
                            Recommended for {agentType}
                          </TooltipContent>
                        </Tooltip>
                      </TooltipProvider>
                    )}
                  </div>
                </SelectItem>
              ))}
            </SelectGroup>
          ))}
        </SelectContent>
      </Select>

      {/* Model Details Card */}
      {selectedModel && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.2 }}
        >
          <Card className="bg-black/20 border-gray-800">
            <CardContent className="p-4 space-y-4">
              {/* Header */}
              <div className="flex items-start justify-between">
                <div>
                  <h4 className="font-semibold text-white">
                    {selectedModel.display_name}
                  </h4>
                  <p className="text-sm text-gray-400">
                    {selectedModel.model_family} by {selectedModel.provider}
                  </p>
                </div>
                <Badge className={getProviderColor(selectedModel.provider)}>
                  {selectedModel.provider}
                </Badge>
              </div>

              {/* Key Metrics */}
              <div className="grid grid-cols-3 gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <Brain className="h-3 w-3" />
                    <span>Context</span>
                  </div>
                  <p className="text-sm font-medium text-white">
                    {(selectedModel.context_window / 1000).toFixed(0)}K tokens
                  </p>
                </div>
                
                <div className="space-y-1">
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <Zap className="h-3 w-3" />
                    <span>Max Output</span>
                  </div>
                  <p className="text-sm font-medium text-white">
                    {(selectedModel.max_output_tokens / 1000).toFixed(1)}K tokens
                  </p>
                </div>
                
                <div className="space-y-1">
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <DollarSign className="h-3 w-3" />
                    <span>Cost</span>
                  </div>
                  <TooltipProvider>
                    <Tooltip>
                      <TooltipTrigger>
                        <p className="text-sm font-medium text-white">
                          ${selectedModel.input_cost_per_1k.toFixed(4)}/1K
                        </p>
                      </TooltipTrigger>
                      <TooltipContent>
                        <div className="text-xs">
                          <div>Input: ${selectedModel.input_cost_per_1k.toFixed(4)}/1K</div>
                          <div>Output: ${selectedModel.output_cost_per_1k.toFixed(4)}/1K</div>
                        </div>
                      </TooltipContent>
                    </Tooltip>
                  </TooltipProvider>
                </div>
              </div>

              {/* Capabilities */}
              {Object.keys(selectedModel.capabilities).length > 0 && (
                <div className="space-y-2">
                  <div className="flex items-center gap-1 text-xs text-gray-400">
                    <Info className="h-3 w-3" />
                    <span>Capabilities</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(selectedModel.capabilities).map(([cap, rating]) => (
                      <div key={cap}>
                        {getCapabilityIcon(cap, rating as string)}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Features */}
              <div className="flex flex-wrap gap-2">
                {selectedModel.supports_functions && (
                  <Badge variant="outline" className="border-blue-500/30 text-blue-400">
                    <Check className="h-3 w-3 mr-1" />
                    Function Calling
                  </Badge>
                )}
                {selectedModel.supports_vision && (
                  <Badge variant="outline" className="border-purple-500/30 text-purple-400">
                    <Check className="h-3 w-3 mr-1" />
                    Vision
                  </Badge>
                )}
              </div>

              {/* Recommended For */}
              {selectedModel.recommended_for.length > 0 && (
                <div className="space-y-2">
                  <div className="text-xs text-gray-400">Recommended for:</div>
                  <div className="flex flex-wrap gap-1">
                    {selectedModel.recommended_for.map((task) => (
                      <Badge 
                        key={task} 
                        variant="secondary"
                        className="text-xs"
                      >
                        {task.replace(/_/g, ' ')}
                      </Badge>
                    ))}
                  </div>
                </div>
              )}

              {/* Warning for deprecated */}
              {selectedModel.status === 'deprecated' && (
                <div className="flex items-start gap-2 p-2 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                  <AlertCircle className="h-4 w-4 text-yellow-400 mt-0.5" />
                  <div className="text-xs text-yellow-400">
                    This model is deprecated and may be removed in the future.
                  </div>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  )
}
```

### 5.2 Update Agent Configuration Modal

**Location**: Update `frontend/components/agents/agent-configuration-modal.tsx`

Add a new "Model" tab:

```typescript
// Add to imports
import { ModelSelector } from './model-selector'
import { Slider } from '@/components/ui/slider'

// Add to form state
const [modelConfig, setModelConfig] = useState({
  provider: 'openai',
  model_id: 'gpt-4',
  temperature: 0.7,
  max_tokens: 2000,
  top_p: 1.0,
  frequency_penalty: 0.0,
  presence_penalty: 0.0,
  fallback_model_id: ''
})

// Add new tab
<TabsContent value="model" className="space-y-6">
  <Card className="bg-black/20 border-gray-800">
    <CardHeader>
      <CardTitle className="text-lg font-semibold text-white flex items-center gap-2">
        <Brain className="h-5 w-5 text-purple-400" />
        Model Configuration
      </CardTitle>
      <p className="text-sm text-gray-400">
        Select and configure the LLM model for this agent
      </p>
    </CardHeader>
    <CardContent className="space-y-6">
      {/* Model Selection */}
      <div className="space-y-2">
        <Label>Model</Label>
        <ModelSelector
          value={modelConfig.model_id}
          onChange={(modelId) => setModelConfig({ ...modelConfig, model_id: modelId })}
          agentType={formData.agent_type}
        />
      </div>

      <Separator />

      {/* Temperature */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label>Temperature</Label>
          <span className="text-sm text-gray-400">{modelConfig.temperature.toFixed(2)}</span>
        </div>
        <Slider
          value={[modelConfig.temperature]}
          onValueChange={([value]) => setModelConfig({ ...modelConfig, temperature: value })}
          min={0}
          max={2}
          step={0.1}
          className="w-full"
        />
        <p className="text-xs text-gray-500">
          Controls randomness. Lower = more focused, Higher = more creative
        </p>
      </div>

      {/* Max Tokens */}
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <Label>Max Output Tokens</Label>
          <span className="text-sm text-gray-400">{modelConfig.max_tokens}</span>
        </div>
        <Slider
          value={[modelConfig.max_tokens]}
          onValueChange={([value]) => setModelConfig({ ...modelConfig, max_tokens: value })}
          min={100}
          max={4000}
          step={100}
          className="w-full"
        />
        <p className="text-xs text-gray-500">
          Maximum tokens in the model's response
        </p>
      </div>

      {/* Advanced Settings */}
      <div className="space-y-4 pt-4 border-t border-gray-800">
        <h4 className="text-sm font-medium text-white">Advanced Settings</h4>
        
        {/* Top P */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Top P (Nucleus Sampling)</Label>
            <span className="text-xs text-gray-400">{modelConfig.top_p.toFixed(2)}</span>
          </div>
          <Slider
            value={[modelConfig.top_p]}
            onValueChange={([value]) => setModelConfig({ ...modelConfig, top_p: value })}
            min={0}
            max={1}
            step={0.05}
            className="w-full"
          />
        </div>

        {/* Frequency Penalty */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Frequency Penalty</Label>
            <span className="text-xs text-gray-400">{modelConfig.frequency_penalty.toFixed(2)}</span>
          </div>
          <Slider
            value={[modelConfig.frequency_penalty]}
            onValueChange={([value]) => setModelConfig({ ...modelConfig, frequency_penalty: value })}
            min={0}
            max={2}
            step={0.1}
            className="w-full"
          />
        </div>

        {/* Presence Penalty */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <Label className="text-xs">Presence Penalty</Label>
            <span className="text-xs text-gray-400">{modelConfig.presence_penalty.toFixed(2)}</span>
          </div>
          <Slider
            value={[modelConfig.presence_penalty]}
            onValueChange={([value]) => setModelConfig({ ...modelConfig, presence_penalty: value })}
            min={0}
            max={2}
            step={0.1}
            className="w-full"
          />
        </div>
      </div>

      {/* Fallback Model */}
      <div className="space-y-2">
        <Label>Fallback Model (Optional)</Label>
        <Select 
          value={modelConfig.fallback_model_id || 'none'}
          onValueChange={(value) => setModelConfig({ 
            ...modelConfig, 
            fallback_model_id: value === 'none' ? '' : value 
          })}
        >
          <SelectTrigger className="bg-black/20 border-gray-800">
            <SelectValue placeholder="Select fallback model..." />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">No fallback</SelectItem>
            <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo</SelectItem>
            <SelectItem value="claude-3-haiku-20240307">Claude 3 Haiku</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-gray-500">
          Model to use if primary model fails or is unavailable
        </p>
      </div>
    </CardContent>
  </Card>
</TabsContent>
```

---

## 6. Implementation Timeline

### Week 1 (16h): Database & Backend Core
- **Day 1 (4h)**: Database schema
  - Create llm_models table with seed data
  - Add model_config column to agents table
  - Add model_usage_stats column
  - Create migrations
  - Test database changes

- **Day 2 (4h)**: Model Registry Service
  - Implement ModelRegistry class
  - Test model queries and recommendations
  - Seed database with OpenAI and Claude models

- **Day 3 (4h)**: Enhanced Agent Factory
  - Update AgentMetadata with ModelConfiguration
  - Update create_agent to use model configs
  - Implement fallback logic
  - Test agent creation with different models

- **Day 4 (4h)**: API Endpoints
  - Implement /api/models endpoints
  - Update /api/agents endpoints for model config
  - Test all endpoints
  - Documentation

### Week 2 (16h): Frontend & Integration
- **Day 1 (4h)**: Model Selector Component
  - Build ModelSelector component
  - Implement model details display
  - Add provider grouping
  - Test with real API data

- **Day 2 (4h)**: Agent Configuration UI
  - Add "Model" tab to configuration modal
  - Implement model settings controls
  - Add fallback model selection
  - Test user flows

- **Day 3 (4h)**: Workflow Integration
  - Ensure workflows respect agent model configs
  - Update execution manager to track model usage
  - Add model usage to execution reports
  - Test end-to-end

- **Day 4 (4h)**: Testing & Polish
  - End-to-end testing
  - Fix bugs
  - Performance optimization
  - Documentation updates

**Total**: 32 hours (4 days)

---

## 7. Success Criteria

### Functional Requirements ✅
- [ ] Database schema supports model registry and agent configs
- [ ] Model Registry service provides model discovery
- [ ] Agents can be created with custom model configs
- [ ] UI allows model selection with metadata display
- [ ] Model settings are configurable (temp, max_tokens, etc.)
- [ ] Fallback models work when primary fails
- [ ] Workflows use correct agent models
- [ ] Model usage is tracked per agent
- [ ] Cost estimation works accurately

### Quality Requirements ✅
- [ ] All endpoints have proper error handling
- [ ] UI is intuitive and informative
- [ ] Model recommendations are relevant
- [ ] Performance is not degraded
- [ ] Backward compatibility maintained
- [ ] Documentation is complete

### User Experience ✅
- [ ] Users can easily select models
- [ ] Model capabilities are clearly displayed
- [ ] Cost information is visible
- [ ] Recommendations help decision making
- [ ] Advanced settings are accessible but not overwhelming

---

## 8. Testing Strategy

### Unit Tests
```python
# Test model registry
async def test_model_registry():
    registry = ModelRegistry(db)
    
    # Test get all models
    models = registry.get_all_models()
    assert len(models) > 0
    
    # Test provider filter
    openai_models = registry.get_models_by_provider('openai')
    assert all(m.provider == 'openai' for m in openai_models)
    
    # Test recommendations
    recommended = registry.get_recommended_models('code_analysis')
    assert len(recommended) > 0
    
    # Test best model finding
    best = registry.find_best_model({
        'task_type': 'code_analysis',
        'max_cost': 0.05,
        'min_context': 8000
    })
    assert best is not None

# Test agent creation with model config
async def test_agent_with_model_config():
    factory = AgentFactory(db)
    
    model_config = ModelConfiguration(
        provider='anthropic',
        model_id='claude-3-sonnet-20240229',
        temperature=0.5,
        max_tokens=3000
    )
    
    metadata = AgentMetadata(
        name='Test Agent',
        agent_type='test',
        model_config=model_config
    )
    
    agent = await factory.create_agent(metadata)
    assert agent.llm_manager.config.provider.value == 'anthropic'
    assert agent.llm_manager.config.model == 'claude-3-sonnet-20240229'
```

### Integration Tests
```python
# Test end-to-end workflow with custom model
async def test_workflow_with_custom_models():
    # Create agents with different models
    gpt4_agent = await create_agent_with_model('gpt-4')
    claude_agent = await create_agent_with_model('claude-3-opus-20240229')
    gpt35_agent = await create_agent_with_model('gpt-3.5-turbo')
    
    # Execute workflow
    execution = await execute_workflow({
        'steps': [
            {'agent_id': gpt4_agent.id, 'task': 'analyze'},
            {'agent_id': claude_agent.id, 'task': 'research'},
            {'agent_id': gpt35_agent.id, 'task': 'summarize'}
        ]
    })
    
    # Verify correct models were used
    models_used = execution.output_data.get('models_used', [])
    assert len(models_used) == 3
    assert any(m['model_id'] == 'gpt-4' for m in models_used)
    assert any(m['model_id'] == 'claude-3-opus-20240229' for m in models_used)
    assert any(m['model_id'] == 'gpt-3.5-turbo' for m in models_used)
```

---

## 9. Future Enhancements (Post-MVP)

### Phase 2: HuggingFace Integration
- Add HuggingFace provider support
- Support for custom/fine-tuned models
- Model hosting configuration
- GPU/CPU resource management

### Phase 3: Advanced Features
- **Auto Model Selection**: AI-driven model selection based on task
- **A/B Testing**: Compare model performance
- **Cost Optimization**: Automatic model switching based on budget
- **Model Fine-Tuning**: Fine-tune models on agent's historical data
- **Model Analytics Dashboard**: Detailed model usage and performance metrics
- **Smart Fallbacks**: Intelligent fallback selection based on context

### Phase 4: Enterprise Features
- **Model Versioning**: Track model version changes
- **Custom Model Endpoints**: Support for private LLM endpoints
- **Multi-Region Support**: Different models per region
- **Compliance**: Model selection based on data compliance requirements

---

## 10. Dependencies

### Existing Components
- ✅ Agent Factory (PRD-02)
- ✅ LLM Provider Service
- ✅ Agent Configuration UI
- ✅ Workflow Execution (PRD-10)

### External Dependencies
- ✅ OpenAI API (already integrated)
- ✅ Anthropic API (already integrated)
- 🔄 HuggingFace API (future)

---

## 11. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Model API changes | High | Version tracking, deprecation warnings |
| Cost overruns | Medium | Usage limits, budget alerts |
| Model unavailability | High | Fallback models, retry logic |
| UI complexity | Medium | Progressive disclosure, good defaults |
| Migration issues | Medium | Backward compatibility, gradual rollout |

---

## Conclusion

PRD-15 transforms Automatos AI from a single-model platform to a **flexible, multi-model system** that empowers users to select the optimal model for each agent based on task requirements, cost considerations, and performance needs.

**Key Outcomes:**
- ✅ User-configurable model selection
- ✅ OpenAI and Claude support (HuggingFace ready)
- ✅ Rich model metadata and recommendations
- ✅ Cost tracking and optimization
- ✅ Fallback resilience
- ✅ Intuitive UI with model comparison

**Development Time:** 4 days (32 hours)  
**Business Impact:** Cost optimization, flexibility, future-proofing

---

**Next Steps:**
1. Review and approve PRD
2. Create database migrations
3. Begin Week 1 implementation
4. Daily progress updates
5. Weekly demo of working features

Let's make Automatos AI truly multi-model! 🚀

