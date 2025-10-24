# PRD 10: Workflow Orchestration Engine - Complete Implementation

**Status:** Ready for Implementation  
**Priority:** CRITICAL - Core Platform Feature  
**Effort:** 40-60 hours  
**Dependencies:** PRD-01, PRD-02, PRD-03, PRD-08, PRD-09

---

## Executive Summary

The Workflow Orchestration Engine is the **brain** of Automatos AI. It takes a workflow request, intelligently breaks it down, selects optimal agents, engineers context-aware prompts, monitors execution, scores results, and feeds everything back into the learning system. This PRD transforms the platform from having "workflows that run" to having "intelligent, self-improving orchestration."

### Current State ✅
- ✅ 69 real workflows in database
- ✅ 98 workflow executions tracked
- ✅ Real agents created and managed
- ✅ Document RAG system operational (292 embeddings)
- ✅ Context engineering modules exist
- ✅ WebSocket real-time updates
- ✅ Task decomposer with LLM integration
- ✅ Agent factory with LLM connections

### Missing Critical Features ❌
- ❌ Complete orchestration flow end-to-end
- ❌ Real task decomposition in workflow execution
- ❌ Intelligent agent selection algorithm
- ❌ Context engineering integration in prompts
- ❌ Agent execution monitoring with live logs
- ❌ Result scoring and quality assessment
- ❌ Learning system updates from execution
- ❌ Execution completion logic (0% success rate issue)
- ❌ Missing CRUD endpoints (GET/PUT/DELETE individual workflow)

---

## 1. Vision & Architecture

### The Complete Orchestration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                   WORKFLOW ORCHESTRATION ENGINE                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  1. REQUEST ANALYSIS                                                 │
│     └─> Parse workflow definition                                   │
│     └─> Extract requirements & constraints                          │
│     └─> Validate inputs & resources                                 │
│                                                                      │
│  2. TASK DECOMPOSITION (Real LLM)                                    │
│     └─> RealTaskDecomposer.decompose_task()                         │
│     └─> Break into atomic subtasks                                  │
│     └─> Identify dependencies (DAG)                                 │
│     └─> Estimate complexity & duration                              │
│                                                                      │
│  3. AGENT SELECTION (Intelligent Matching)                           │
│     └─> Query agent capabilities from database                      │
│     └─> Match skills to subtask requirements                        │
│     └─> Consider agent load & availability                          │
│     └─> Create or reuse agents (Agent Factory)                      │
│                                                                      │
│  4. CONTEXT ENGINEERING (RAG + Semantic Search)                      │
│     └─> Retrieve relevant documents (semantic search)               │
│     └─> Build context-aware prompts (molecular level)               │
│     └─> Apply mathematical optimization (token budget)              │
│     └─> Include examples & patterns                                 │
│                                                                      │
│  5. EXECUTION & MONITORING                                           │
│     └─> Execute agents in parallel/sequential                       │
│     └─> Real-time progress tracking (WebSocket)                     │
│     └─> Live logging (orchestrator + agent logs)                    │
│     └─> Resource monitoring (CPU, memory, tokens)                   │
│     └─> Handle failures & retries                                   │
│                                                                      │
│  6. RESULT AGGREGATION & SCORING                                     │
│     └─> Collect all agent outputs                                   │
│     └─> Score quality (completeness, accuracy)                      │
│     └─> Identify patterns & insights                                │
│     └─> Generate final report                                       │
│                                                                      │
│  7. LEARNING & ANALYTICS                                             │
│     └─> Update agent performance metrics                            │
│     └─> Record context effectiveness                                │
│     └─> Update workflow success patterns                            │
│     └─> Feed into optimization algorithms                           │
│                                                                      │
│  8. COMPLETION & REPORTING                                           │
│     └─> Mark execution as COMPLETED/FAILED                          │
│     └─> Generate execution report                                   │
│     └─> Send WebSocket completion event                             │
│     └─> Update dashboard metrics                                    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Detailed Implementation Steps

### Step 1: Request Analysis & Validation (4h)

#### Component: `WorkflowRequestAnalyzer`

```python
# Location: orchestrator/core/workflow_request_analyzer.py

class WorkflowRequestAnalyzer:
    """
    Analyzes workflow requests and prepares for orchestration
    """
    
    async def analyze_workflow_request(
        self,
        workflow_id: int,
        input_data: Dict[str, Any],
        execution_options: Dict[str, Any],
        db: Session
    ) -> WorkflowAnalysis:
        """
        Analyze workflow request and extract requirements
        
        Returns:
            WorkflowAnalysis with:
            - workflow: Workflow object
            - requirements: extracted requirements
            - constraints: time, resource, quality constraints
            - validation_result: is request valid?
            - suggested_agents: preliminary agent suggestions
        """
```

**Features:**
- Load workflow definition from database
- Parse workflow configuration (steps, agents, config)
- Extract explicit requirements from input
- Infer implicit requirements from workflow type
- Validate resource availability
- Check for circular dependencies
- Estimate resource needs (tokens, time, agents)

**Logging Output:**
```
[ORCHESTRATOR] Workflow Request Received: workflow_id=42, name="Code Review Pipeline"
[ORCHESTRATOR] Input Data: 3 keys (repository, branch, files)
[ORCHESTRATOR] Validation: ✓ All required inputs present
[ORCHESTRATOR] Requirements Extracted: code_analysis, security_scan, documentation
[ORCHESTRATOR] Constraints: max_time=600s, max_cost=$0.50, min_quality=0.85
[ORCHESTRATOR] Resource Check: ✓ 3 agents available, 50K tokens budget
[ORCHESTRATOR] Estimated Duration: 4-7 minutes
```

---

### Step 2: Task Decomposition Integration (8h)

#### Component: `EnhancedTaskDecomposer`

**Already Exists:** `orchestrator/core/real_task_decomposer.py`

**Integration Needed:**
1. Wire into workflow execution
2. Add workflow-specific decomposition strategies
3. Enhance with dependency analysis
4. Add execution plan generation

```python
# Location: orchestrator/core/enhanced_task_decomposer.py

from orchestrator.core.real_task_decomposer import RealTaskDecomposer

class EnhancedTaskDecomposer:
    """
    Enhanced task decomposition for workflow orchestration
    """
    
    def __init__(self):
        self.real_decomposer = RealTaskDecomposer()
    
    async def decompose_for_workflow(
        self,
        workflow: Workflow,
        input_data: Dict[str, Any],
        requirements: List[str],
        db: Session
    ) -> WorkflowDecomposition:
        """
        Decompose workflow into executable subtasks
        
        Returns:
            WorkflowDecomposition with:
            - subtasks: list of Subtask objects
            - dependency_graph: networkx DAG
            - execution_plan: parallel/sequential strategy
            - estimated_time: total duration estimate
            - critical_path: longest dependency chain
        """
        
        # Step 1: Use RealTaskDecomposer for LLM-based breakdown
        description = self._build_decomposition_prompt(workflow, input_data)
        llm_decomposition = await self.real_decomposer.decompose_task(
            task_description=description,
            task_type=workflow.workflow_definition.get('category', 'general'),
            complexity=self._estimate_complexity(workflow),
            requirements=requirements
        )
        
        # Step 2: Enrich with workflow-specific context
        subtasks = self._enrich_subtasks(
            llm_subtasks=llm_decomposition['subtasks'],
            workflow_steps=workflow.workflow_definition.get('steps', []),
            workflow_config=workflow.workflow_definition.get('config', {})
        )
        
        # Step 3: Build dependency graph
        dependency_graph = self._build_dependency_graph(subtasks)
        
        # Step 4: Generate execution plan
        execution_plan = self._generate_execution_plan(
            dependency_graph,
            execution_strategy=workflow.workflow_definition.get('execution_strategy', 'auto')
        )
        
        return WorkflowDecomposition(
            subtasks=subtasks,
            dependency_graph=dependency_graph,
            execution_plan=execution_plan,
            estimated_time=self._calculate_critical_path(dependency_graph),
            metadata=llm_decomposition
        )
```

**Logging Output:**
```
[ORCHESTRATOR] Task Decomposition Started
[DECOMPOSER] Using LLM: gpt-4 for intelligent breakdown
[DECOMPOSER] Prompt Length: 1,247 tokens
[DECOMPOSER] LLM Response Time: 3.2s
[DECOMPOSER] Subtasks Identified: 6
  └─ [1] Code Analysis (priority: high, duration: 90s, deps: [])
  └─ [2] Security Scan (priority: high, duration: 120s, deps: [1])
  └─ [3] Performance Review (priority: medium, duration: 60s, deps: [1])
  └─ [4] Documentation Check (priority: low, duration: 45s, deps: [])
  └─ [5] Best Practices (priority: medium, duration: 75s, deps: [1,2,3])
  └─ [6] Generate Report (priority: high, duration: 30s, deps: [2,3,4,5])
[DECOMPOSER] Dependency Graph: 6 nodes, 7 edges, DAG ✓
[DECOMPOSER] Execution Strategy: MIXED (2 parallel, 4 sequential phases)
[DECOMPOSER] Critical Path: [1] → [2] → [5] → [6] = 315s (5.25min)
[DECOMPOSER] Parallelization Opportunity: Tasks [3,4] can run parallel to [2]
```

---

### Step 3: Intelligent Agent Selection (10h)

#### Component: `IntelligentAgentSelector`

```python
# Location: orchestrator/core/agent_selector.py

class IntelligentAgentSelector:
    """
    Selects optimal agents for workflow subtasks using intelligent matching
    """
    
    def __init__(self, agent_factory):
        self.agent_factory = agent_factory
        self.selection_cache = {}
        
    async def select_agents_for_workflow(
        self,
        subtasks: List[Subtask],
        workflow: Workflow,
        db: Session
    ) -> Dict[str, AgentAssignment]:
        """
        Select or create optimal agents for each subtask
        
        Algorithm:
        1. Query existing agents from database
        2. Calculate skill match scores (using cosine similarity)
        3. Consider agent load & availability
        4. Reuse agents when possible for efficiency
        5. Create new agents via Agent Factory if needed
        6. Assign agents to subtasks
        
        Returns:
            Dict mapping subtask_id to AgentAssignment(agent, confidence, reason)
        """
```

**Selection Algorithm:**

```python
async def _calculate_agent_match_score(
    self,
    subtask: Subtask,
    agent: Agent,
    current_load: float
) -> float:
    """
    Calculate how well an agent matches a subtask
    
    Score = 0.4 * skill_match + 0.3 * experience + 0.2 * (1 - load) + 0.1 * recency
    
    skill_match: cosine similarity between required skills and agent skills
    experience: agent's historical success rate on similar tasks
    load: current utilization (prefer less loaded agents)
    recency: favor recently active agents (warm LLM connections)
    """
    
    # Skill matching using embeddings
    required_skills = subtask.skills_required
    agent_skills = [skill.name for skill in agent.skills]
    
    skill_match = self._calculate_skill_similarity(
        required_skills,
        agent_skills
    )
    
    # Experience from performance history
    experience = await self._get_agent_experience_score(
        agent.id,
        subtask.agent_type,
        db
    )
    
    # Load factor (prefer available agents)
    load_factor = 1.0 - current_load
    
    # Recency (active agents have warm connections)
    recency = self._calculate_recency_score(agent.last_active_at)
    
    score = (
        0.4 * skill_match +
        0.3 * experience +
        0.2 * load_factor +
        0.1 * recency
    )
    
    return score
```

**Agent Creation Logic:**

```python
async def _create_agent_if_needed(
    self,
    subtask: Subtask,
    existing_agents: List[Agent],
    db: Session
) -> Agent:
    """
    Create new agent if no suitable agent exists
    
    Uses Agent Factory to create specialized agent:
    - Selects appropriate agent type
    - Configures with required skills
    - Initializes LLM connection
    - Registers in database
    """
    
    logger.info(f"[AGENT_SELECTOR] No suitable agent found for {subtask.agent_type}")
    logger.info(f"[AGENT_SELECTOR] Creating new agent via Agent Factory...")
    
    # Prepare agent metadata
    agent_metadata = {
        "name": f"{subtask.agent_type.title()} - {subtask.subtask_id}",
        "type": subtask.agent_type,
        "description": f"Specialized agent for {subtask.description}",
        "skills": subtask.skills_required,
        "preferred_model": self._select_model_for_task(subtask),
        "temperature": self._select_temperature(subtask),
        "max_tokens": 4000
    }
    
    # Create agent using factory
    agent_runtime = await self.agent_factory.create_agent(
        metadata=agent_metadata,
        auto_verify=True
    )
    
    logger.info(f"[AGENT_SELECTOR] ✓ Agent created: id={agent_runtime.agent_id}, name={agent_metadata['name']}")
    
    return agent_runtime
```

**Logging Output:**
```
[ORCHESTRATOR] Agent Selection Started: 6 subtasks
[AGENT_SELECTOR] Querying database for available agents...
[AGENT_SELECTOR] Found 12 active agents
[AGENT_SELECTOR] Calculating match scores for subtask #1 (Code Analysis)
  ├─ Agent "CodeArchitect-001" (id=5): score=0.92 (skills=0.95, exp=0.88, load=0.10, recent=1.0)
  ├─ Agent "SecurityExpert-003" (id=8): score=0.45 (skills=0.40, exp=0.50, load=0.20, recent=0.9)
  └─ SELECTED: CodeArchitect-001 (confidence: 0.92)
[AGENT_SELECTOR] Subtask #1 → Agent #5 (reused)
[AGENT_SELECTOR] Calculating match scores for subtask #2 (Security Scan)
  ├─ Agent "SecurityExpert-003" (id=8): score=0.89
  └─ SELECTED: SecurityExpert-003 (confidence: 0.89)
[AGENT_SELECTOR] Subtask #2 → Agent #8 (reused)
[AGENT_SELECTOR] Calculating match scores for subtask #3 (Performance Review)
  ├─ No suitable agent found (best score: 0.52 < threshold: 0.70)
  └─ CREATING NEW AGENT via Agent Factory...
[AGENT_FACTORY] Creating agent: PerformanceOptimizer
[AGENT_FACTORY] Initializing LLM: provider=openai, model=gpt-4
[AGENT_FACTORY] LLM verification: ✓ (response_time: 1.2s)
[AGENT_FACTORY] Agent created: id=45, status=ACTIVE
[AGENT_SELECTOR] Subtask #3 → Agent #45 (created)
[ORCHESTRATOR] Agent Selection Complete: 6 subtasks assigned to 5 agents (4 reused, 2 created)
```

---

### Step 4: Context Engineering Integration (12h)

#### Component: `WorkflowContextEngineer`

**Integrates:** RAG Service + Context Engineering Modules

```python
# Location: orchestrator/core/workflow_context_engineer.py

from orchestrator.services.rag_service import RAGService
from field_theory.field_manager import FieldContextManager

class WorkflowContextEngineer:
    """
    Builds context-aware prompts using RAG and Context Engineering
    """
    
    def __init__(self):
        self.rag_service = RAGService()
        self.field_manager = FieldContextManager()
        
    async def engineer_prompt_for_subtask(
        self,
        subtask: Subtask,
        agent: Agent,
        workflow_context: Dict[str, Any],
        db: Session
    ) -> EngineerPrompt:
        """
        Engineer optimal prompt using Context Engineering principles
        
        Follows Atoms → Molecules → Cells progression:
        - Atomic: Core instruction (clear, specific)
        - Molecular: + Examples + Patterns from RAG
        - Cellular: + Agent memory + Historical context
        
        Returns:
            EngineeredPrompt with:
            - system_prompt: Agent role and instructions
            - user_prompt: Specific task with context
            - retrieved_context: RAG results
            - token_count: Total tokens
            - optimization_metrics: Information density, etc.
        """
```

**Implementation Steps:**

```python
async def engineer_prompt_for_subtask(self, subtask, agent, workflow_context, db):
    logger.info(f"[CONTEXT_ENG] Engineering prompt for subtask: {subtask.description}")
    
    # STEP 1: ATOMIC PROMPT (Clear instruction)
    atomic_prompt = self._build_atomic_instruction(subtask)
    logger.info(f"[CONTEXT_ENG] Atomic prompt: {len(atomic_prompt)} chars")
    
    # STEP 2: RAG RETRIEVAL (Semantic search for relevant docs)
    logger.info(f"[CONTEXT_ENG] RAG retrieval started...")
    rag_results = await self.rag_service.retrieve_for_task(
        query=subtask.description,
        task_type=subtask.agent_type,
        max_chunks=5,
        max_tokens=2000,
        db=db
    )
    logger.info(f"[CONTEXT_ENG] RAG retrieved: {len(rag_results['chunks'])} chunks, {rag_results['total_tokens']} tokens")
    for i, chunk in enumerate(rag_results['chunks'], 1):
        logger.info(f"  └─ [{i}] {chunk['source']} (similarity: {chunk['similarity']:.3f})")
    
    # STEP 3: EXAMPLES & PATTERNS
    logger.info(f"[CONTEXT_ENG] Retrieving examples & patterns...")
    examples = await self._get_few_shot_examples(
        subtask.agent_type,
        similar_tasks=3,
        db=db
    )
    logger.info(f"[CONTEXT_ENG] Found {len(examples)} relevant examples")
    
    # STEP 4: AGENT MEMORY
    logger.info(f"[CONTEXT_ENG] Loading agent memory...")
    agent_memory = await self._get_agent_memory(
        agent.id,
        relevant_to=subtask.description,
        db=db
    )
    logger.info(f"[CONTEXT_ENG] Agent memory: {len(agent_memory)} relevant entries")
    
    # STEP 5: MATHEMATICAL OPTIMIZATION
    logger.info(f"[CONTEXT_ENG] Applying token budget optimization...")
    optimized_context = await self._optimize_token_budget(
        atomic_prompt=atomic_prompt,
        rag_context=rag_results,
        examples=examples,
        agent_memory=agent_memory,
        max_tokens=agent.metadata.context_window or 8000
    )
    logger.info(f"[CONTEXT_ENG] Optimization: {optimized_context['original_tokens']}→{optimized_context['final_tokens']} tokens (saved {optimized_context['saved_tokens']})")
    
    # STEP 6: ASSEMBLE FINAL PROMPT
    system_prompt = self._build_system_prompt(agent, subtask)
    user_prompt = self._build_user_prompt(
        subtask,
        optimized_context,
        workflow_context
    )
    
    logger.info(f"[CONTEXT_ENG] ✓ Prompt engineering complete")
    logger.info(f"[CONTEXT_ENG] System prompt: {len(system_prompt)} chars")
    logger.info(f"[CONTEXT_ENG] User prompt: {len(user_prompt)} chars")
    logger.info(f"[CONTEXT_ENG] Total tokens: {optimized_context['final_tokens']}")
    
    return EngineeredPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        retrieved_context=rag_results,
        examples=examples,
        token_count=optimized_context['final_tokens'],
        information_density=optimized_context['information_density']
    )
```

**Logging Output:**
```
[ORCHESTRATOR] Context Engineering Started for 6 subtasks
[CONTEXT_ENG] Engineering prompt for subtask: Code Analysis
[CONTEXT_ENG] Atomic prompt: 342 chars
[CONTEXT_ENG] RAG retrieval started...
[RAG_SERVICE] Searching for: "Analyze code for security vulnerabilities and best practices"
[RAG_SERVICE] Generating query embedding: text-embedding-ada-002
[RAG_SERVICE] Vector search: top_k=5, min_similarity=0.7
[RAG_SERVICE] Query results: 5 chunks found
[CONTEXT_ENG] RAG retrieved: 5 chunks, 1,847 tokens
  └─ [1] secure_coding_guide.pdf (similarity: 0.923)
  └─ [2] code_review_checklist.md (similarity: 0.891)
  └─ [3] python_security_patterns.txt (similarity: 0.867)
  └─ [4] owasp_top10.pdf (similarity: 0.845)
  └─ [5] static_analysis_guide.md (similarity: 0.812)
[CONTEXT_ENG] Retrieving examples & patterns...
[CONTEXT_ENG] Found 3 relevant examples
[CONTEXT_ENG] Loading agent memory...
[CONTEXT_ENG] Agent memory: 7 relevant entries
[CONTEXT_ENG] Applying token budget optimization...
[OPTIMIZER] Original content: 4,523 tokens
[OPTIMIZER] Token budget: 3,500 tokens (agent context window: 8,000)
[OPTIMIZER] Optimization strategy: MMR + knapsack
[OPTIMIZER] Information gain calculation...
[OPTIMIZER] Selected 3 chunks, 2 examples, 4 memory entries
[CONTEXT_ENG] Optimization: 4,523→3,287 tokens (saved 1,236 tokens, 27.3%)
[CONTEXT_ENG] ✓ Prompt engineering complete
[CONTEXT_ENG] System prompt: 512 chars
[CONTEXT_ENG] User prompt: 8,934 chars
[CONTEXT_ENG] Total tokens: 3,287 / 8,000 (41% utilization)
[CONTEXT_ENG] Information density: 0.87 (high)
```

---

### Step 5: Agent Execution & Monitoring (10h)

#### Component: `AgentExecutionManager`

```python
# Location: orchestrator/core/agent_execution_manager.py

class AgentExecutionManager:
    """
    Executes agents and monitors progress with real-time logging
    """
    
    async def execute_workflow_agents(
        self,
        execution_plan: ExecutionPlan,
        agent_assignments: Dict[str, AgentAssignment],
        engineered_prompts: Dict[str, EngineeredPrompt],
        workflow_execution: WorkflowExecution,
        db: Session
    ) -> WorkflowExecutionResult:
        """
        Execute all agents according to execution plan with real-time monitoring
        
        Features:
        - Parallel execution where possible
        - Sequential execution respecting dependencies
        - Real-time progress updates via WebSocket
        - Live logging (orchestrator + agent logs)
        - Resource monitoring (tokens, time, memory)
        - Automatic retries on failures
        - Graceful error handling
        """
        
        execution_context = {
            'workflow_execution_id': workflow_execution.id,
            'start_time': datetime.now(),
            'results': {},
            'errors': {},
            'logs': []
        }
        
        # Execute phases (parallel groups)
        for phase_idx, phase in enumerate(execution_plan.phases):
            logger.info(f"[EXECUTION] Phase {phase_idx+1}/{len(execution_plan.phases)}: {len(phase.subtasks)} parallel tasks")
            
            # Execute subtasks in parallel
            phase_results = await self._execute_phase(
                phase,
                agent_assignments,
                engineered_prompts,
                execution_context,
                workflow_execution,
                db
            )
            
            # Update context with results
            execution_context['results'].update(phase_results)
            
            # Send progress update
            await self._send_progress_update(
                workflow_execution.id,
                phase_idx + 1,
                len(execution_plan.phases),
                execution_context
            )
```

**Real-Time Monitoring:**

```python
async def _execute_subtask_with_monitoring(
    self,
    subtask: Subtask,
    agent_assignment: AgentAssignment,
    engineered_prompt: EngineeredPrompt,
    execution_context: Dict,
    workflow_execution: WorkflowExecution,
    db: Session
) -> SubtaskResult:
    """
    Execute single subtask with comprehensive monitoring
    """
    
    agent = agent_assignment.agent
    start_time = time.time()
    
    logger.info(f"[AGENT:{agent.name}] Starting subtask: {subtask.description}")
    logger.info(f"[AGENT:{agent.name}] Agent type: {agent.agent_type}, Model: {agent.metadata.preferred_model}")
    
    # Send start event via WebSocket
    await manager.broadcast({
        "type": "subtask_started",
        "data": {
            "workflow_execution_id": workflow_execution.id,
            "subtask_id": subtask.subtask_id,
            "agent_id": agent.id,
            "agent_name": agent.name,
            "description": subtask.description,
            "timestamp": datetime.now().isoformat()
        }
    })
    
    try:
        # Execute via LLM
        logger.info(f"[AGENT:{agent.name}] Calling LLM with {engineered_prompt.token_count} tokens...")
        
        response = await agent.llm_manager.generate(
            system_prompt=engineered_prompt.system_prompt,
            user_prompt=engineered_prompt.user_prompt,
            temperature=agent.metadata.temperature or 0.7,
            max_tokens=agent.metadata.max_tokens or 2000
        )
        
        execution_time = time.time() - start_time
        
        logger.info(f"[AGENT:{agent.name}] ✓ LLM response received: {len(response.content)} chars, {response.usage.total_tokens} tokens")
        logger.info(f"[AGENT:{agent.name}] Execution time: {execution_time:.2f}s")
        logger.info(f"[AGENT:{agent.name}] Token usage: prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}")
        
        # Process response
        result = SubtaskResult(
            subtask_id=subtask.subtask_id,
            agent_id=agent.id,
            status="completed",
            output=response.content,
            execution_time=execution_time,
            tokens_used=response.usage.total_tokens,
            metadata={
                "model": response.model,
                "finish_reason": response.finish_reason,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens
            }
        )
        
        # Send completion event
        await manager.broadcast({
            "type": "subtask_completed",
            "data": {
                "workflow_execution_id": workflow_execution.id,
                "subtask_id": subtask.subtask_id,
                "agent_id": agent.id,
                "status": "completed",
                "execution_time": execution_time,
                "tokens_used": response.usage.total_tokens,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        logger.info(f"[AGENT:{agent.name}] ✓ Subtask completed successfully")
        
        return result
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"[AGENT:{agent.name}] ✗ Subtask failed: {str(e)}")
        
        # Send error event
        await manager.broadcast({
            "type": "subtask_failed",
            "data": {
                "workflow_execution_id": workflow_execution.id,
                "subtask_id": subtask.subtask_id,
                "agent_id": agent.id,
                "error": str(e),
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat()
            }
        })
        
        return SubtaskResult(
            subtask_id=subtask.subtask_id,
            agent_id=agent.id,
            status="failed",
            error=str(e),
            execution_time=execution_time
        )
```

**Logging Output:**
```
[ORCHESTRATOR] Workflow Execution Started: workflow_id=42, execution_id=157
[EXECUTION] Execution plan: 3 phases, 6 total subtasks
[EXECUTION] Phase 1/3: 2 parallel tasks
  ├─ Subtask #1: Code Analysis (agent: CodeArchitect-001)
  └─ Subtask #4: Documentation Check (agent: DocumentationExpert-012)
[AGENT:CodeArchitect-001] Starting subtask: Code Analysis
[AGENT:CodeArchitect-001] Agent type: code_architect, Model: gpt-4
[AGENT:CodeArchitect-001] Calling LLM with 3,287 tokens...
[AGENT:DocumentationExpert-012] Starting subtask: Documentation Check
[AGENT:DocumentationExpert-012] Agent type: documentation_expert, Model: gpt-4
[AGENT:DocumentationExpert-012] Calling LLM with 2,145 tokens...
[AGENT:DocumentationExpert-012] ✓ LLM response received: 1,523 chars, 982 tokens
[AGENT:DocumentationExpert-012] Execution time: 4.7s
[AGENT:DocumentationExpert-012] Token usage: prompt=2145, completion=837
[AGENT:DocumentationExpert-012] ✓ Subtask completed successfully
[AGENT:CodeArchitect-001] ✓ LLM response received: 3,891 chars, 2,234 tokens
[AGENT:CodeArchitect-001] Execution time: 6.2s
[AGENT:CodeArchitect-001] Token usage: prompt=3287, completion=1947
[AGENT:CodeArchitect-001] ✓ Subtask completed successfully
[EXECUTION] Phase 1 complete: 2/2 subtasks successful (100%)
[WEBSOCKET] Progress update sent: phase=1/3, progress=33%
[EXECUTION] Phase 2/3: 3 parallel tasks
  ├─ Subtask #2: Security Scan (agent: SecurityExpert-003)
  ├─ Subtask #3: Performance Review (agent: PerformanceOptimizer-045)
  └─ Subtask #5: Best Practices (agent: CodeArchitect-001)
[AGENT:SecurityExpert-003] Starting subtask: Security Scan
[AGENT:PerformanceOptimizer-045] Starting subtask: Performance Review
[AGENT:CodeArchitect-001] Starting subtask: Best Practices
... (execution continues)
```

---

### Step 6: Result Aggregation & Scoring (6h)

#### Component: `ResultAggregator`

```python
# Location: orchestrator/core/result_aggregator.py

class ResultAggregator:
    """
    Aggregates agent results and scores quality
    """
    
    async def aggregate_and_score_results(
        self,
        subtask_results: List[SubtaskResult],
        workflow: Workflow,
        workflow_execution: WorkflowExecution,
        db: Session
    ) -> AggregatedResult:
        """
        Aggregate all subtask results and score quality
        
        Scoring Dimensions:
        1. Completeness (0-1): All subtasks completed?
        2. Accuracy (0-1): Results meet quality thresholds?
        3. Consistency (0-1): Results align with each other?
        4. Timeliness (0-1): Completed within time budget?
        5. Cost Efficiency (0-1): Token usage reasonable?
        
        Overall Score = weighted average of dimensions
        """
        
        logger.info(f"[SCORING] Aggregating {len(subtask_results)} subtask results")
        
        # Calculate scores
        completeness = self._calculate_completeness(subtask_results)
        accuracy = await self._calculate_accuracy(subtask_results, workflow, db)
        consistency = self._calculate_consistency(subtask_results)
        timeliness = self._calculate_timeliness(subtask_results, workflow_execution)
        cost_efficiency = self._calculate_cost_efficiency(subtask_results, workflow)
        
        # Weighted overall score
        overall_score = (
            0.30 * completeness +
            0.30 * accuracy +
            0.20 * consistency +
            0.10 * timeliness +
            0.10 * cost_efficiency
        )
        
        logger.info(f"[SCORING] Quality Scores:")
        logger.info(f"  ├─ Completeness: {completeness:.3f}")
        logger.info(f"  ├─ Accuracy: {accuracy:.3f}")
        logger.info(f"  ├─ Consistency: {consistency:.3f}")
        logger.info(f"  ├─ Timeliness: {timeliness:.3f}")
        logger.info(f"  ├─ Cost Efficiency: {cost_efficiency:.3f}")
        logger.info(f"  └─ Overall: {overall_score:.3f}")
        
        # Generate insights
        insights = self._generate_insights(subtask_results)
        
        # Create final report
        report = self._generate_execution_report(
            subtask_results,
            scores={
                "completeness": completeness,
                "accuracy": accuracy,
                "consistency": consistency,
                "timeliness": timeliness,
                "cost_efficiency": cost_efficiency,
                "overall": overall_score
            },
            insights=insights
        )
        
        return AggregatedResult(
            results=subtask_results,
            scores=scores,
            overall_score=overall_score,
            insights=insights,
            report=report
        )
```

---

### Step 7: Learning & Analytics Updates (6h)

#### Component: `LearningSystemUpdater`

```python
# Location: orchestrator/core/learning_system_updater.py

class LearningSystemUpdater:
    """
    Updates learning systems with execution feedback
    """
    
    async def update_learning_systems(
        self,
        workflow_execution: WorkflowExecution,
        aggregated_result: AggregatedResult,
        agent_assignments: Dict[str, AgentAssignment],
        engineered_prompts: Dict[str, EngineeredPrompt],
        db: Session
    ):
        """
        Update all learning and analytics systems
        
        Updates:
        1. Agent Performance Metrics
        2. Context Engineering Effectiveness
        3. Workflow Pattern Recognition
        4. Resource Optimization Models
        5. Quality Prediction Models
        """
        
        logger.info(f"[LEARNING] Updating learning systems...")
        
        # 1. Update Agent Performance
        await self._update_agent_performance(
            agent_assignments,
            aggregated_result.results,
            db
        )
        
        # 2. Update Context Effectiveness
        await self._update_context_effectiveness(
            engineered_prompts,
            aggregated_result.results,
            db
        )
        
        # 3. Update Workflow Patterns
        await self._update_workflow_patterns(
            workflow_execution,
            aggregated_result,
            db
        )
        
        # 4. Update Resource Models
        await self._update_resource_models(
            workflow_execution,
            aggregated_result,
            db
        )
        
        # 5. Store Execution Analytics
        await self._store_execution_analytics(
            workflow_execution,
            aggregated_result,
            db
        )
        
        logger.info(f"[LEARNING] ✓ All learning systems updated")
```

**Logging Output:**
```
[SCORING] Aggregating 6 subtask results
[SCORING] Quality Scores:
  ├─ Completeness: 1.000 (6/6 subtasks completed)
  ├─ Accuracy: 0.923 (high quality outputs)
  ├─ Consistency: 0.889 (results align well)
  ├─ Timeliness: 0.967 (312s / 320s budget = 97.5%)
  ├─ Cost Efficiency: 0.891 (12,453 tokens / 15,000 budget = 83%)
  └─ Overall: 0.938 (EXCELLENT)
[SCORING] Insights Generated:
  ├─ Security scan identified 3 vulnerabilities (high priority)
  ├─ Performance could improve by 23% with caching
  ├─ Documentation coverage: 87% (good)
  └─ Code quality: A- (best practices followed)
[LEARNING] Updating learning systems...
[LEARNING] Agent performance updated: 5 agents
  ├─ CodeArchitect-001: quality=0.95, speed=1.2x baseline
  ├─ SecurityExpert-003: quality=0.98, speed=0.9x baseline
  └─ ... (3 more agents)
[LEARNING] Context effectiveness recorded:
  ├─ RAG retrieval quality: 0.91 (excellent)
  ├─ Token optimization: 27% savings
  └─ Example relevance: 0.88 (high)
[LEARNING] Workflow pattern recorded:
  ├─ Type: code_review
  ├─ Success rate: 94.2% (historical)
  └─ Optimal agent combo: [code_architect, security_expert, perf_optimizer]
[LEARNING] ✓ All learning systems updated
```

---

### Step 8: Completion & Reporting (4h)

#### Component: `ExecutionCompleter`

```python
# Location: orchestrator/core/execution_completer.py

class ExecutionCompleter:
    """
    Completes workflow execution and generates final reports
    """
    
    async def complete_execution(
        self,
        workflow_execution: WorkflowExecution,
        aggregated_result: AggregatedResult,
        db: Session
    ):
        """
        Complete workflow execution
        
        Steps:
        1. Update execution status (COMPLETED/FAILED)
        2. Store final results
        3. Calculate final metrics
        4. Generate execution report
        5. Send WebSocket completion event
        6. Update workflow statistics
        7. Clean up resources
        """
        
        start_time = workflow_execution.started_at
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Determine final status
        final_status = (
            ExecutionStatus.COMPLETED
            if aggregated_result.overall_score >= 0.7
            else ExecutionStatus.FAILED
        )
        
        logger.info(f"[COMPLETION] Workflow execution {final_status.value}")
        logger.info(f"[COMPLETION] Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        logger.info(f"[COMPLETION] Overall Score: {aggregated_result.overall_score:.3f}")
        
        # Update database
        workflow_execution.status = final_status.value
        workflow_execution.completed_at = end_time
        workflow_execution.output_data = {
            "results": [r.to_dict() for r in aggregated_result.results],
            "scores": aggregated_result.scores,
            "insights": aggregated_result.insights,
            "execution_summary": {
                "total_subtasks": len(aggregated_result.results),
                "successful_subtasks": len([r for r in aggregated_result.results if r.status == "completed"]),
                "total_tokens": sum(r.tokens_used or 0 for r in aggregated_result.results),
                "total_duration": duration,
                "agents_used": len(set(r.agent_id for r in aggregated_result.results))
            }
        }
        workflow_execution.execution_log = aggregated_result.report
        
        db.commit()
        
        # Send completion event
        await manager.broadcast({
            "type": "execution_completed",
            "data": {
                "execution_id": workflow_execution.id,
                "workflow_id": workflow_execution.workflow_id,
                "status": final_status.value,
                "duration": duration,
                "overall_score": aggregated_result.overall_score,
                "summary": workflow_execution.output_data["execution_summary"],
                "timestamp": end_time.isoformat()
            }
        })
        
        logger.info(f"[COMPLETION] ✓ Execution complete, results stored")
        logger.info(f"[COMPLETION] Report: {len(aggregated_result.report)} chars")
```

---

## 3. Missing API Endpoints (2h)

### Fix Critical Missing Endpoints

Add to `/api/workflows.py`:

```python
@router.get("/{workflow_id}")
async def get_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """Get individual workflow by ID"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return {
        "id": workflow.id,
        "name": workflow.name,
        "description": workflow.description,
        "status": workflow.status,
        "workflow_definition": workflow.workflow_definition,
        "created_at": workflow.created_at.isoformat() if workflow.created_at else None,
        "updated_at": workflow.updated_at.isoformat() if workflow.updated_at else None
    }

@router.put("/{workflow_id}")
async def update_workflow(
    workflow_id: int,
    workflow_data: Dict[str, Any],
    db: Session = Depends(get_db)
):
    """Update workflow"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    # Update fields
    if "name" in workflow_data:
        workflow.name = workflow_data["name"]
    if "description" in workflow_data:
        workflow.description = workflow_data["description"]
    if "status" in workflow_data:
        workflow.status = workflow_data["status"]
    if "workflow_definition" in workflow_data:
        workflow.workflow_definition = workflow_data["workflow_definition"]
    
    workflow.updated_at = datetime.now()
    db.commit()
    
    return {"message": "Workflow updated", "id": workflow_id}

@router.delete("/{workflow_id}")
async def delete_workflow(workflow_id: int, db: Session = Depends(get_db)):
    """Delete workflow"""
    workflow = db.query(Workflow).filter(Workflow.id == workflow_id).first()
    if not workflow:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    db.delete(workflow)
    db.commit()
    
    return {"message": "Workflow deleted", "id": workflow_id}

@router.get("/executions/")
async def list_executions(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    workflow_id: Optional[int] = None,
    status: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List workflow executions"""
    query = db.query(WorkflowExecution)
    
    if workflow_id:
        query = query.filter(WorkflowExecution.workflow_id == workflow_id)
    if status:
        query = query.filter(WorkflowExecution.status == status)
    
    total = query.count()
    executions = query.order_by(desc(WorkflowExecution.started_at)).offset(skip).limit(limit).all()
    
    return {
        "items": [
            {
                "id": e.id,
                "workflow_id": e.workflow_id,
                "status": e.status,
                "started_at": e.started_at.isoformat() if e.started_at else None,
                "completed_at": e.completed_at.isoformat() if e.completed_at else None
            } for e in executions
        ],
        "total": total
    }
```

---

## 4. Implementation Order & Timeline

### Week 1 (16h): Foundation
- ✅ Day 1-2: Request Analysis & Validation (4h)
- ✅ Day 3-4: Task Decomposition Integration (8h)
- ✅ Day 5: Missing API Endpoints (2h)
- ✅ Day 5: Testing & Bug Fixes (2h)

### Week 2 (20h): Core Execution
- ✅ Day 1-2: Intelligent Agent Selection (10h)
- ✅ Day 3-4: Context Engineering Integration (10h)

### Week 3 (16h): Monitoring & Results
- ✅ Day 1-2: Agent Execution & Monitoring (10h)
- ✅ Day 3-4: Result Aggregation & Scoring (6h)

### Week 4 (8h): Learning & Polish
- ✅ Day 1: Learning System Updates (6h)
- ✅ Day 2: Completion & Reporting (4h)
- ✅ Day 3: End-to-End Testing (4h)
- ✅ Day 4: Performance Optimization (4h)
- ✅ Day 5: Documentation & Demos (4h)

**Total:** 60 hours (4 weeks)

---

## 5. Success Criteria

### Functional Requirements ✅
- [ ] Complete orchestration flow executes end-to-end
- [ ] Real task decomposition using LLM
- [ ] Intelligent agent selection with scoring
- [ ] Context engineering with RAG integration
- [ ] Real-time monitoring with WebSocket updates
- [ ] Comprehensive logging at every step
- [ ] Result scoring with quality metrics
- [ ] Learning systems update automatically
- [ ] Workflow execution completes successfully
- [ ] All API endpoints working (GET/PUT/DELETE)

### Performance Requirements ✅
- [ ] Workflow decomposition: < 5s
- [ ] Agent selection: < 3s
- [ ] Context engineering: < 10s per subtask
- [ ] Agent execution: varies by task (tracked)
- [ ] Result aggregation: < 2s
- [ ] Overall latency: < 60s + agent execution time

### Quality Requirements ✅
- [ ] Execution success rate: > 90%
- [ ] Overall quality score: > 0.85
- [ ] Context relevance: > 0.80
- [ ] Agent match accuracy: > 0.85
- [ ] Learning system updates: 100% success

---

## 6. Testing Strategy

### Unit Tests
- Request analyzer validation logic
- Task decomposition with mocked LLM
- Agent matching algorithm
- Context optimization
- Scoring calculations

### Integration Tests
- End-to-end workflow execution
- WebSocket events
- Database transactions
- Agent Factory integration
- RAG Service integration

### User Journey Tests
```python
# Test complete workflow execution
async def test_complete_workflow_orchestration():
    # 1. Create workflow
    workflow = await create_workflow({
        "name": "Test Code Review",
        "description": "Complete code review workflow",
        "category": "code_review"
    })
    
    # 2. Execute workflow
    execution = await execute_workflow(workflow.id, {
        "repository": "automatos-ai",
        "branch": "main",
        "files": ["orchestrator/core/workflow_orchestrator.py"]
    })
    
    # 3. Wait for completion
    result = await wait_for_execution(execution.id, timeout=300)
    
    # 4. Verify results
    assert result.status == "completed"
    assert result.overall_score >= 0.85
    assert len(result.subtask_results) == 6
    assert all(r.status == "completed" for r in result.subtask_results)
```

---

## 7. Monitoring & Observability

### Metrics to Track
- Workflow execution count (total, success, failed)
- Average execution duration
- Token usage per workflow
- Agent utilization
- Context engineering effectiveness
- Overall quality scores
- Error rates by component

### Logging Levels
- **INFO**: Major orchestration steps
- **DEBUG**: Detailed execution flow
- **WARN**: Non-fatal issues
- **ERROR**: Failures requiring attention

### Dashboard Integration
- Real-time workflow execution count
- Success/failure rate charts
- Agent performance leaderboard
- Token usage trends
- Quality score distribution

---

## 8. Risk Mitigation

### Technical Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM API failures | High | Retry logic + fallbacks |
| Agent creation delays | Medium | Agent pool + reuse strategy |
| Token budget exceeded | Medium | Strict optimization + alerts |
| Memory leaks | High | Resource cleanup + monitoring |
| Database deadlocks | Medium | Transaction optimization |

### Quality Risks
| Risk | Impact | Mitigation |
|------|--------|------------|
| Poor task decomposition | High | Quality scoring + human review |
| Agent mismatch | Medium | Confidence thresholds + fallbacks |
| Context irrelevance | Medium | RAG quality metrics + tuning |
| Execution timeouts | Medium | Adaptive timeouts + early termination |

---

## 9. Dependencies

### Existing Components (Already Built)
- ✅ `RealTaskDecomposer` - Task decomposition
- ✅ `AgentFactory` - Agent creation
- ✅ `RAGService` - Document retrieval
- ✅ `FieldContextManager` - Field theory
- ✅ `CoordinationManager` - Multi-agent coordination
- ✅ `WebSocketManager` - Real-time updates

### External Dependencies
- ✅ OpenAI API (GPT-4, text-embedding-ada-002)
- ✅ PostgreSQL with pgvector
- ✅ Redis (for caching & coordination)

---

## 10. Future Enhancements (Post-MVP)

### Phase 2 Features
- Workflow templates library
- Interactive human-in-the-loop
- Conditional branching logic
- Parallel workflow execution
- Advanced failure recovery
- Cost optimization AI
- A/B testing for strategies
- Workflow versioning

### Phase 3 Features
- Visual workflow designer
- Workflow marketplace
- Multi-tenant isolation
- Cross-workflow communication
- Automated workflow generation
- Predictive optimization

---

## Conclusion

PRD-10 completes the Automatos AI platform by implementing **intelligent, production-ready workflow orchestration**. By integrating task decomposition, agent selection, context engineering, real-time monitoring, result scoring, and learning systems, we transform the platform from a collection of components into a **cohesive, self-improving AI orchestration system**.

**Key Outcomes:**
- ✅ Real task decomposition (not mock)
- ✅ Intelligent agent matching
- ✅ Context-aware prompts with RAG
- ✅ Comprehensive real-time logging
- ✅ Quality scoring & learning
- ✅ Production-ready monitoring
- ✅ 0% → 90%+ success rate

**Development Time:** 4 weeks (60 hours)  
**Business Impact:** Platform becomes production-ready for real customer workflows

---

**Next Steps:**
1. Review this PRD for completeness
2. Get stakeholder approval
3. Begin implementation Week 1
4. Daily standups to track progress
5. Weekly demos to show working features

Let's build the most intelligent orchestration system! 🚀

