# PRD 01: Core Orchestration Engine

## 1. Overview

### Purpose
The Core Orchestration Engine is the brain of Automatos AI, responsible for task decomposition, agent assignment, and workflow coordination using Context Engineering principles.

### Vision Alignment
Following the Context Engineering paradigm:
- Receives complex tasks (organisms)
- Breaks them into subtasks (organs)
- Assigns to specialized agents (cells)
- Manages context flow (molecules)
- Generates optimized prompts (atoms)

## 2. Problem Statement

Currently, the orchestrator returns mock data and doesn't actually:
- Break down tasks intelligently
- Assign agents based on capabilities
- Generate context-aware prompts
- Coordinate multi-agent workflows
- Learn from execution results

## 3. Success Criteria

- [ ] Real task decomposition using LLM reasoning
- [ ] Dynamic agent selection based on skills
- [ ] Context-aware prompt generation
- [ ] Parallel and sequential task execution
- [ ] Performance tracking and optimization

## 4. Functional Requirements

### 4.1 Task Analysis & Decomposition

```python
class TaskDecomposer:
    """
    Analyzes complex tasks and breaks them into atomic operations
    """
    
    async def analyze_task(self, task_description: str) -> TaskAnalysis:
        # Use LLM to understand task requirements
        # Identify required skills and resources
        # Determine task complexity and dependencies
        
    async def decompose_task(self, task: Task) -> List[Subtask]:
        # Break complex task into subtasks
        # Identify dependencies between subtasks
        # Determine execution order (parallel/sequential)
        # Assign priority levels
```

### 4.2 Agent Selection & Assignment

```python
class AgentSelector:
    """
    Selects optimal agents for each subtask
    """
    
    async def match_agents_to_tasks(
        self, 
        subtasks: List[Subtask],
        available_agents: List[Agent]
    ) -> List[TaskAssignment]:
        # Match subtask requirements to agent skills
        # Consider agent availability and workload
        # Optimize for performance and efficiency
        # Handle skill gaps and fallbacks
```

### 4.3 Prompt Engineering & Context Management

```python
class PromptOrchestrator:
    """
    Generates optimized prompts using Context Engineering
    """
    
    async def generate_prompt(
        self,
        task: Subtask,
        agent: Agent,
        context: OrchestratorContext
    ) -> ContextualPrompt:
        # Apply atomic prompt principles
        # Add molecular context (examples, patterns)
        # Include cellular memory (agent's past experience)
        # Incorporate organ-level coordination instructions
```

### 4.4 Execution Coordination

```python
class ExecutionCoordinator:
    """
    Manages workflow execution and agent coordination
    """
    
    async def execute_workflow(
        self,
        workflow: Workflow,
        assignments: List[TaskAssignment]
    ) -> WorkflowResult:
        # Initialize execution context
        # Start parallel task groups
        # Manage sequential dependencies
        # Handle inter-agent communication
        # Aggregate results
```

## 5. Technical Architecture

### 5.1 Core Components

```
┌─────────────────────────────────────────────────────────┐
│                  Orchestration Engine                    │
├───────────────────┬─────────────────┬──────────────────┤
│  Task Analyzer    │  Agent Matcher  │  Prompt Builder  │
├───────────────────┼─────────────────┼──────────────────┤
│  Decomposer       │  Skill Matcher  │  Context Manager │
├───────────────────┼─────────────────┼──────────────────┤
│  Dependency Graph │  Load Balancer  │  Template Engine │
└───────────────────┴─────────────────┴──────────────────┘
```

### 5.2 Integration Points

- **LLM Provider Service**: For task analysis and decomposition
- **Agent Registry**: For capability matching
- **Context Engineering**: For prompt optimization
- **Memory Service**: For historical context
- **Field Theory**: For context propagation

## 6. Implementation Details

### 6.1 Task Decomposition Algorithm

1. **Analyze task intent** using LLM
2. **Identify key operations** (CRUD, analysis, generation, etc.)
3. **Determine dependencies** between operations
4. **Create execution graph** with parallel/sequential paths
5. **Assign complexity scores** to each subtask

### 6.2 Agent Matching Algorithm

1. **Extract required skills** from subtask
2. **Query agent capabilities** from database
3. **Calculate skill match scores**
4. **Consider agent availability** and current load
5. **Optimize assignment** for overall efficiency

### 6.3 Context Engineering Integration

```python
# Example: Generating a context-aware prompt
async def generate_contextual_prompt(self, subtask, agent, global_context):
    # Atomic level - core instruction
    atomic_prompt = self.prompt_builder.create_atomic_instruction(subtask)
    
    # Molecular level - add examples and patterns
    molecular_context = await self.context_retriever.get_relevant_examples(
        task_type=subtask.type,
        similarity_threshold=0.8
    )
    
    # Cellular level - add agent's memory
    agent_memory = await self.memory_service.get_agent_memory(
        agent_id=agent.id,
        relevant_to=subtask
    )
    
    # Organ level - add coordination context
    coordination_context = self.get_coordination_instructions(
        subtask.dependencies,
        subtask.downstream_tasks
    )
    
    # Combine using mathematical optimization
    optimized_prompt = self.context_optimizer.optimize(
        atomic=atomic_prompt,
        molecular=molecular_context,
        cellular=agent_memory,
        organ=coordination_context,
        max_tokens=agent.context_window
    )
    
    return optimized_prompt
```

## 7. Database Schema Updates

```sql
-- Add tables for orchestration
CREATE TABLE task_decompositions (
    id SERIAL PRIMARY KEY,
    parent_task_id INTEGER REFERENCES tasks(id),
    subtask_id INTEGER REFERENCES tasks(id),
    dependency_type VARCHAR(50),
    execution_order INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE task_assignments (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(id),
    agent_id INTEGER REFERENCES agents(id),
    assignment_score FLOAT,
    assignment_reason TEXT,
    status VARCHAR(50),
    assigned_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE execution_contexts (
    id SERIAL PRIMARY KEY,
    workflow_id INTEGER REFERENCES workflows(id),
    context_data JSONB,
    field_state JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 8. API Endpoints

```python
# Task decomposition
POST /api/orchestrator/decompose
{
    "task_description": "...",
    "complexity_limit": 10,
    "decomposition_strategy": "hierarchical"
}

# Agent assignment
POST /api/orchestrator/assign
{
    "task_id": "...",
    "optimization_criteria": ["speed", "quality"],
    "fallback_strategy": "create_new_agent"
}

# Execute orchestrated workflow
POST /api/orchestrator/execute
{
    "workflow_id": "...",
    "execution_mode": "parallel_optimized",
    "monitoring": true
}

# Get orchestration status
GET /api/orchestrator/status/{execution_id}
```

## 9. Files to Modify

1. **Create new file**: `orchestrator/core/task_decomposer.py`
2. **Create new file**: `orchestrator/core/agent_selector.py`
3. **Create new file**: `orchestrator/core/prompt_orchestrator.py`
4. **Update**: `orchestrator/services/orchestrator_service.py` (replace mock)
5. **Update**: `orchestrator/api/orchestrator.py` (add real endpoints)
6. **Create migration**: `alembic/versions/xxx_add_orchestration_tables.py`

## 10. Testing Strategy

### Unit Tests
- Task decomposition accuracy
- Agent matching logic
- Prompt generation quality
- Context optimization

### Integration Tests
- End-to-end workflow execution
- Multi-agent coordination
- Memory integration
- Performance benchmarks

### User Acceptance Criteria
- Complex task successfully decomposed
- Agents assigned based on real capabilities
- Prompts include relevant context
- Execution completes with real results
- Performance improves over time

## 11. Dependencies

- **PRD 02**: Agent Factory (for agent creation)
- **PRD 03**: Context Engineering (for prompt optimization)
- **PRD 05**: Memory Systems (for historical context)

## 12. Timeline

- Week 1: Core decomposition logic
- Week 2: Agent selection algorithm
- Week 3: Context integration
- Week 4: Testing and optimization

## 13. Success Metrics

- Task decomposition accuracy > 85%
- Agent assignment optimization > 90%
- Prompt quality score > 8/10
- Execution success rate > 95%
- Average task completion time < baseline - 30%
