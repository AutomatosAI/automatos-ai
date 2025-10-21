# PRD 13: Enterprise Workflows Enhancement - Integration & Testing

**Status:** Active Development  
**Priority:** P0 - Critical for November Event  
**Effort:** 3-4 days (testing & integration focused)  
**Target Date:** October 28, 2025

**⚠️ IMPORTANT**: This PRD focuses on **testing, fixing, and integrating** existing systems built in PRD-04 and PRD-05, NOT building new systems.

---

## 0. Quick Operations Guide

### 0.1 Deploy Code to Backend
```bash
# From local machine
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai
scp -i ~/.ssh/id_rsa -r orchestrator/ root@206.81.0.227:/root/automatos-ai/
```

### 0.2 Restart Backend
```bash
# Using restart script (recommended)
bash /Users/gkavanagh/Development/Automatos-AI-Platform/restart-backend.sh

# OR manually via SSH
ssh -i ~/.ssh/id_rsa root@206.81.0.227
cd /root/automatos-ai/orchestrator
pkill -9 -f 'uvicorn.*main:app'
sleep 2
nohup python3 -m uvicorn main:app --host 0.0.0.0 --port 8000 > ../backend.log 2>&1 < /dev/null &
disown
exit

# Check API health
curl -s https://api.automatos.app/api/v1/memory/stats/real | python3 -m json.tool
```

### 0.3 Access Database
```bash
# Via Docker (PostgreSQL)
ssh -i ~/.ssh/id_rsa root@206.81.0.227
docker exec -it Automatos_postgres psql -U postgres -d orchestrator_db

# Common queries
SELECT COUNT(*) FROM memory_items;
SELECT memory_level, COUNT(*) FROM memory_items GROUP BY memory_level;
SELECT * FROM memory_items ORDER BY created_at DESC LIMIT 5;

# Via Docker (Redis)
docker exec -it Automatos_redis redis-cli -a redis_password_123
KEYS working:*
KEYS agent:*
```

### 0.4 Restart Frontend (Local)
```bash
# Kill all running instances first
pkill -9 -f "next-server"
pkill -9 -f "npm.*dev"
lsof -ti:3000 | xargs kill -9 2>/dev/null

# Start fresh
cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai/frontend
rm -rf .next  # Clear cache
npm run dev
```

### 0.5 Environment Configuration

**Database Configuration:**
```env
POSTGRES_DB=orchestrator_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=secure_password_123
POSTGRES_HOST=127.0.0.1
POSTGRES_PORT=5432
DATABASE_URL=postgresql://postgres:secure_password_123@127.0.0.1:5432/orchestrator_db
```

**Redis Configuration:**
```env
REDIS_PASSWORD=redis_password_123
REDIS_PORT=6379
REDIS_HOST=127.0.0.1
```

**API Configuration:**
```env
API_PORT=8000
API_KEY=test_api_key_for_backend_validation_2025
API_URL=api.automatos.app
```

### 0.6 Check Backend Logs
```bash
ssh -i ~/.ssh/id_rsa root@206.81.0.227 "tail -100 /root/automatos-ai/backend.log"
```

---

## 1. Executive Summary

Transform workflows to **enterprise-grade** by properly integrating and testing the **already-built** systems:
- ✅ **Memory System** (PRD-05) - Built, needs integration testing
- ✅ **Agent Communication** (PRD-04) - Built, needs proper testing & fixes
- ✅ **Learning Engine** (PRD-05) - Built, needs research & activation
- 🔨 **UI Visibility** - Need dashboards for memory, communication, and learning

---

## 2. Current State (What's Already Built)

### ✅ 2.1 Memory System (PRD-05)
**Location**: `orchestrator/services/memory_knowledge_system.py`

**Components**:
- `HierarchicalMemorySystem` class
  - Working Memory (Redis, 5 min TTL)
  - Short-Term Memory (PostgreSQL, 24 hours)
  - Long-Term Memory (PostgreSQL + pgvector)
- `KnowledgeGraph` class
- `LearningEngine` class
- `WorkflowMemoryIntegrator` in `core/workflow_memory_integrator.py`

**Status**: ✅ Built, ⚠️ Needs testing & workflow integration

### ✅ 2.2 Agent Communication (PRD-04)
**Location**: `orchestrator/services/inter_agent_communication.py`

**Components**:
- `AgentCommunicationProtocol` (Redis Pub/Sub)
- `SharedContextManager` (collaborative workspace)
- `MessageType` enum (task_request, knowledge_share, etc.)
- Already integrated into `AgentExecutionManager` with `enable_communication` flag

**Status**: ✅ Built, ⚠️ Needs proper testing & fixes

### ✅ 2.3 Learning System (PRD-05)
**Location**: 
- `orchestrator/services/memory_knowledge_system.py` (`LearningEngine`)
- `orchestrator/context_engineering/learning_engine.py` (`AdaptiveLearningEngine`, `PatternRecognitionEngine`)

**Components**:
- `LearningEngine.learn_from_feedback()`
- `AdaptiveLearningEngine.learn_from_task_execution()`
- `PatternRecognitionEngine.analyze_task_patterns()`

**Status**: ✅ Built, ⚠️ Needs research, testing & activation

---

## 3. Implementation Tasks (Focus on Integration & Testing)

### 📋 Phase 1: Memory System Integration & Testing (Day 1)

**Task 1.1: Test HierarchicalMemorySystem Independently**
```python
# Test script: test_memory_system.py
async def test_memory_storage_and_retrieval():
    """Test memory system with real Redis & PostgreSQL"""
    
    memory_system = HierarchicalMemorySystem(
        redis_host=os.getenv("REDIS_HOST"),
        redis_port=int(os.getenv("REDIS_PORT")),
        redis_password=os.getenv("REDIS_PASSWORD"),
        postgres_url=os.getenv("DATABASE_URL"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # Test 1: Store experience
    experience_id = await memory_system.store_experience(
        agent_id=1,
        experience={
            "task_id": "test_001",
            "content": "Successfully analyzed code repository",
            "success": True,
            "tokens_used": 1500,
            "execution_time_ms": 2000
        }
    )
    
    # Test 2: Retrieve memories
    memories = await memory_system.retrieve_relevant_memories(
        agent_id=1,
        context="code analysis task",
        memory_types=["experience"],
        top_k=5
    )
    
    # Test 3: Consolidation
    await memory_system.consolidate_memories(agent_id=1)
    
    print(f"✅ Memory system test passed: {len(memories)} memories retrieved")
```

**Files to Check**:
- ✅ `services/memory_knowledge_system.py` - Main system
- ✅ `core/workflow_memory_integrator.py` - Integration layer

**Expected Output**: Memories stored in Redis (5 min TTL) and PostgreSQL, retrieval working via vector search.

---

**Task 1.2: Integrate Memory into Workflow Execution**

**File**: `api/workflows.py` or `core/workflow_orchestrator.py`

**Changes Needed**:
```python
# In workflow execution flow:

# 1. BEFORE execution: Retrieve memories
from core.workflow_memory_integrator import WorkflowMemoryIntegrator

memory_integrator = WorkflowMemoryIntegrator(
    memory_system=hierarchical_memory,
    knowledge_graph=knowledge_graph,
    learning_engine=learning_engine
)

# Retrieve agent memories
memories = await memory_integrator.retrieve_workflow_memories(
    workflow_id=workflow.id,
    workflow_description=workflow.description,
    agent_ids=[agent.id for agent in assigned_agents]
)

# 2. DURING execution: Inject memories into agent prompts
for subtask in subtasks:
    agent_memories = memories["agent_memories"].get(subtask.agent_id, {})
    
    # Add memory context to enhanced prompt
    memory_context = _format_agent_memories(agent_memories)
    enhanced_prompt = f"""
    ## Your Previous Experiences
    {memory_context}
    
    ## Current Task
    {subtask.description}
    
    Use your previous experiences to complete this task effectively.
    """

# 3. AFTER execution: Store experiences
storage_results = await memory_integrator.store_execution_experiences(
    workflow_id=workflow.id,
    execution_id=execution.id,
    subtask_executions=subtask_results,
    aggregated_results=aggregated_results
)

# 4. CONSOLIDATE learnings
consolidation_results = await memory_integrator.consolidate_workflow_learnings(
    workflow_id=workflow.id,
    execution_id=execution.id,
    aggregated_results=aggregated_results,
    decomposition_metadata=decomposition_metadata
)
```

**Estimated Effort**: 4-6 hours

---

### 📋 Phase 2: Agent Communication Testing & Fixes (Day 2)

**Task 2.1: Test Agent Communication Independently**

```python
# Test script: test_agent_communication.py
async def test_agent_messaging():
    """Test Redis Pub/Sub agent messaging"""
    
    comm = AgentCommunicationProtocol()
    await comm.connect()
    
    # Test 1: Send message
    result = await comm.send_message(
        from_agent=1,
        to_agent=2,
        message_type=MessageType.TASK_REQUEST,
        content={"request": "Help with code review", "urgent": True},
        priority=8
    )
    
    # Test 2: Broadcast
    await comm.broadcast(
        from_agent=1,
        execution_id=123,
        message_type=MessageType.RESULT_SHARE,
        content={"result": "Analysis complete"}
    )
    
    print(f"✅ Communication test passed: {result.status}")
```

**Files to Check**:
- ✅ `services/inter_agent_communication.py` - Main protocol
- ✅ `core/agent_execution_manager.py` - Integration (see line 109: `self.enable_communication`)

**Task 2.2: Enable Communication in Workflows**

**File**: `core/agent_execution_manager.py`

**Current Code** (Line 95):
```python
def __init__(
    self,
    db_session: Session,
    agent_factory: Optional[AgentFactory] = None,
    max_parallel_executions: int = 3,
    max_retries: int = 2,
    enable_communication: bool = True  # ← This exists!
):
```

**Action**: Ensure `enable_communication=True` is passed when creating `AgentExecutionManager` in workflows.

**Verify Integration** (Line 122+):
```python
async def execute_workflow_subtasks(
    self,
    subtasks: List[Dict[str, Any]],
    agent_assignments: Dict[str, Any],
    context_enhancements: Dict[str, Any],
    execution_id: int,
    workflow_id: int
) -> Dict[str, SubtaskExecution]:
    
    # VERIFY: Is SharedContext being created?
    if self.enable_communication:
        shared_context = await self.context_manager.create_shared_context(
            execution_id=execution_id,
            workflow_id=workflow_id,
            participants=[agent_id for agent_id in agent_assignments.values()]
        )
```

**Testing Focus**: Run a workflow and check logs for:
- `✅ Inter-agent communication ENABLED`
- `SharedContext created for execution {id}`
- Agent messages being published to Redis

**Estimated Effort**: 3-4 hours

---

### 📋 Phase 3: Learning System Activation & Testing (Day 2-3)

**Task 3.1: Research Existing Learning Engines**

**Files to Review**:
1. `services/memory_knowledge_system.py` (`LearningEngine` class)
2. `context_engineering/learning_engine.py` (`AdaptiveLearningEngine`, `PatternRecognitionEngine`)

**Key Methods**:
- `LearningEngine.learn_from_feedback()` - Updates success patterns
- `AdaptiveLearningEngine.learn_from_task_execution()` - Stores learning events
- `PatternRecognitionEngine.analyze_task_patterns()` - Clusters & identifies patterns

**Task 3.2: Activate Learning After Workflow Execution**

**File**: `api/workflows.py` or `core/workflow_orchestrator.py`

**Integration Point**: After workflow completes

```python
# After workflow execution completes:

if learning_engine:
    try:
        # Learn from each subtask
        for subtask_id, execution in subtask_results.items():
            await learning_engine.learn_from_feedback(
                agent_id=execution.agent_id,
                task_type=execution.agent_name,  # or categorize by type
                result={
                    "execution_time": execution.execution_time_ms,
                    "tokens_used": execution.tokens_used,
                    "quality_score": execution.quality_score
                },
                feedback={
                    "success": execution.status == SubtaskStatus.COMPLETED,
                    "context_quality": execution.context_quality,
                    "retry_count": execution.retry_count
                }
            )
        
        logger.info(f"✅ Learning complete for workflow {workflow.id}")
        
    except Exception as e:
        logger.error(f"Learning failed: {e}")
```

**Task 3.3: Pattern Extraction (AdaptiveLearningEngine)**

```python
# Use AdaptiveLearningEngine for more advanced learning:

adaptive_engine = AdaptiveLearningEngine(vector_store, embedding_generator)

await adaptive_engine.learn_from_task_execution(
    task_description=workflow.description,
    task_type=workflow.category or "general",
    context_used={
        "rag_sources": context_enhancements.get("num_sources", 0),
        "code_symbols": context_enhancements.get("num_code_symbols", 0),
        "semantic_matches": context_enhancements.get("semantic_matches", 0)
    },
    outcome=aggregated_results.final_output,
    success=aggregated_results.quality_scores.overall >= 0.7,
    execution_time=execution.execution_time_seconds,
    agent_used=lead_agent_name,
    user_feedback=user_feedback  # if available
)
```

**Estimated Effort**: 4-6 hours

---

### 📋 Phase 4: UI Dashboards for Visibility (Day 3-4)

**Task 4.1: Memory Dashboard**

**File**: `frontend/components/workflows/memory-dashboard.tsx` (new)

**Features**:
- Show agent memories (working, short-term, long-term counts)
- Display recent experiences per agent
- Show memory consolidation stats
- Visualize memory retrieval effectiveness

**API Endpoints** (new):
```python
# api/memory.py

@router.get("/memory/stats/{agent_id}")
async def get_agent_memory_stats(agent_id: int):
    """Get memory statistics for an agent"""
    # Query Redis for working memory count
    # Query PostgreSQL for short-term & long-term counts
    # Return stats

@router.get("/memory/{agent_id}/recent")
async def get_recent_memories(agent_id: int, limit: int = 10):
    """Get recent memories for an agent"""
    # Query from memory_items table
    # Return formatted memories
```

---

**Task 4.2: Communication Dashboard**

**File**: `frontend/components/workflows/communication-dashboard.tsx` (new)

**Features**:
- Show agent message history for a workflow execution
- Display message types (help_request, knowledge_share, etc.)
- Show shared workspace state
- Visualize agent collaboration graph

**API Endpoints** (new):
```python
# api/communication.py

@router.get("/communication/messages/{execution_id}")
async def get_execution_messages(execution_id: int):
    """Get all agent messages for an execution"""
    # Query agent_messages table or Redis history
    # Return message list with from/to/type/content

@router.get("/communication/workspace/{execution_id}")
async def get_shared_workspace(execution_id: int):
    """Get shared workspace state"""
    # Use SharedContextManager.get_shared_context()
    # Return workspace state
```

---

**Task 4.3: Learning Dashboard**

**File**: `frontend/components/workflows/learning-dashboard.tsx` (new)

**Features**:
- Show performance improvements over time
- Display learned patterns
- Show success rate trends per agent
- Visualize token/cost optimizations

**API Endpoints** (new):
```python
# api/learning.py

@router.get("/learning/outcomes/{agent_id}")
async def get_learning_outcomes(agent_id: int):
    """Get learning outcomes for an agent"""
    # Query learning_outcomes table
    # Return patterns & improvements

@router.get("/learning/performance/{workflow_id}")
async def get_workflow_performance_trend(workflow_id: int):
    """Get performance trend for a workflow type"""
    # Query workflow_executions
    # Calculate improvements over time
    # Return trend data
```

**Estimated Effort**: 6-8 hours

---

## 4. Testing Plan

### Test Workflow 1: PR Code Review (with Memory)

**Steps**:
1. Run PR review workflow (first time)
2. Check logs for:
   - `🧠 Retrieving memories for workflow` (should find 0 memories)
   - `💾 Storing execution experiences` (should store N experiences)
3. Run same PR review workflow (second time)
4. Check logs for:
   - `🧠 Retrieving memories for workflow` (should find >0 memories)
   - Memory context in agent prompts
5. Compare quality scores between first and second run (expect improvement)

### Test Workflow 2: Multi-Agent Collaboration

**Steps**:
1. Create workflow with 3+ agents
2. Enable communication in `AgentExecutionManager`
3. Check logs for:
   - `✅ Inter-agent communication ENABLED`
   - `SharedContext created for execution`
   - `Agent {id} → Agent {id}: {message_type}`
4. Verify Redis contains messages: `KEYS agent:*:messages`
5. Check database for stored messages in `agent_messages` table

### Test Workflow 3: Learning Over Time

**Steps**:
1. Run same workflow 5-10 times
2. After each run, check:
   - `learning_outcomes` table for new entries
   - Performance metrics (execution time, tokens)
3. Plot trends: expect improvements over time
4. Check for pattern extraction: `patterns_extracted > 0`

---

## 5. Database Verification

**Check Existing Tables** (should already exist from PRD-05):

```sql
-- Memory tables (from memory_knowledge_system.py Base.metadata)
SELECT table_name FROM information_schema.tables 
WHERE table_schema = 'public' 
AND table_name IN ('memory_items', 'knowledge_nodes', 'knowledge_edges', 'learning_outcomes');

-- Check if pgvector is enabled
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Check memory items count
SELECT memory_level, COUNT(*) FROM memory_items GROUP BY memory_level;

-- Check learning outcomes
SELECT agent_id, COUNT(*) as learning_count FROM learning_outcomes GROUP BY agent_id;
```

**If Tables Don't Exist**: Run `memory_system.initialize_database()` (see line 174 in `memory_knowledge_system.py`)

---

## 6. Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| **Agent Context Retention** | 0% | 100% | Memory retrieval success rate |
| **Collaboration Events** | 0 | 20+/workflow | Message count per execution |
| **Learning Improvements** | 0% | 10-15% | Performance improvement over 10 runs |
| **Memory Relevance** | N/A | >80% | Quality of retrieved memories |
| **Communication Latency** | N/A | <100ms | Message delivery time |

---

## 7. Implementation Priority

**Day 1**: Memory System Testing & Integration ✅ COMPLETE
- [x] Test `HierarchicalMemorySystem` independently (test_memory_system.py)
- [x] Integrate `WorkflowMemoryIntegrator` into workflows (api/workflows.py)
- [x] Verify Redis working memory (5 min TTL) ✅ Working
- [x] Verify PostgreSQL short-term & long-term storage ✅ 33 memories stored
- [x] **Bug Fixes:**
  - [x] Memory initialization in except block - FIXED
  - [x] JSON mutations not tracked (16 locations) - FIXED with `flag_modified`
  - [x] Datetime serialization errors - FIXED
  - [x] Mock dashboard data - FIXED (real API endpoint created)
  - [x] Hardcoded short_term levels - FIXED (dynamic 5-tier system)
  - [x] Importance thresholds too high - FIXED (adjusted to 0.72/0.78)
- [x] **API Tests Created:**
  - [x] test_api_memory_system.py
  - [x] test_api_communication.py
  - [x] test_api_learning_improvement.py
  - [x] test_api_repeated_execution.py
- [x] **Memory Dashboard:** Real data from `/api/v1/memory/stats/real`

**Day 2**: Communication Testing & Fixes 🔄 IN PROGRESS
- [ ] Test `AgentCommunicationProtocol` independently
- [ ] Verify `enable_communication=True` in workflows
- [ ] Test `SharedContextManager`
- [ ] Fix any Redis Pub/Sub issues

**Day 3**: Learning System Activation 🔄 PENDING
- [ ] Research existing `LearningEngine` & `AdaptiveLearningEngine`
- [ ] Integrate learning after workflow completion
- [ ] Test pattern extraction
- [ ] Verify `learning_outcomes` table population

**Day 4**: UI Dashboards & Polish 🔄 PENDING
- [x] Build Memory Dashboard (Execution Theater → Memory tab) ✅
- [ ] Build Communication Dashboard
- [ ] Build Learning Dashboard
- [ ] End-to-end testing with all systems enabled

---

## 8. Key Files Reference

**Memory System** (PRD-05):
- `orchestrator/services/memory_knowledge_system.py` - Main implementation
- `orchestrator/core/workflow_memory_integrator.py` - Workflow integration

**Agent Communication** (PRD-04):
- `orchestrator/services/inter_agent_communication.py` - Communication protocol
- `orchestrator/core/agent_execution_manager.py` - Integration point (line 109)

**Learning System** (PRD-05):
- `orchestrator/services/memory_knowledge_system.py` - `LearningEngine`
- `orchestrator/context_engineering/learning_engine.py` - `AdaptiveLearningEngine`, `PatternRecognitionEngine`

**Database Models**:
- Memory models defined in `memory_knowledge_system.py` (`Base.metadata`)
- Workflow models in `orchestrator/database/models.py`

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **Memory retrieval slow** | Use pgvector indexes, cache frequent queries |
| **Redis message loss** | Store all messages in database, use `require_ack=True` |
| **Learning false positives** | Set high confidence thresholds (>0.7), manual review |
| **Integration breaks existing** | Feature flags, comprehensive testing, rollback plan |

---

## 10. Demo Value for November Event

**What Investors Will See**:

1. **Agent Memory Demo** (1-2 min)
   - Run same workflow twice
   - Show: Second run uses memories from first
   - Proof: Logs show memory retrieval, improved quality score

2. **Agent Collaboration Demo** (1-2 min)
   - Multi-agent workflow with message passing
   - Show: Real-time message exchange in Communication Dashboard
   - Proof: Redis messages, shared workspace updates

3. **Self-Learning Demo** (1-2 min)
   - Show performance trend over 10 runs
   - Demonstrate: Execution time ↓, Quality ↑, Tokens optimized
   - Proof: Learning Dashboard with charts

**Total Demo Time**: 5-6 minutes  
**Investor Impact**: "AI that remembers, collaborates, and learns"

---

## Conclusion

PRD-13 is **NOT about building new systems** - it's about:
✅ Testing what's already built (PRD-04, PRD-05)  
✅ Integrating properly into workflows  
✅ Fixing issues discovered during testing  
✅ Building UI for visibility  
✅ Preparing for November demo  

**Next**: Execute Day 1 tasks (Memory System Testing & Integration)
