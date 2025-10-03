# Orchestrator-Agent Integration Guide

## Architecture Overview

The system now follows a clear separation of concerns:

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   User/UI/API       │────▶│   Orchestrator   │────▶│  Agent Factory  │
│                     │     │                  │     │                 │
│ - Define agents     │     │ - Decompose task │     │ - Execute LLM   │
│ - Submit tasks      │     │ - Build prompts  │     │ - Track metrics │
│                     │     │ - Route to agent │     │ - Manage state  │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
                                    │                          │
                                    ▼                          ▼
                            ┌──────────────────┐     ┌─────────────────┐
                            │  Vector Store    │     │   LLM APIs      │
                            │                  │     │                 │
                            │ - Context docs   │     │ - Claude        │
                            │ - Examples       │     │ - GPT-4         │
                            │ - Domain data    │     │ - HuggingFace   │
                            └──────────────────┘     └─────────────────┘
```

## Key Components

### 1. User-Defined Agents

Users create agents via UI or API with flexible metadata:

```python
{
    "name": "RiskAnalyzer-v2",
    "type": "financial_risk",  # User-defined type
    "skills": ["risk_analysis", "statistical_modeling", "python"],
    "preferred_model": "claude-3-opus-20240229",  # Optional
    "temperature": 0.3,  # Optional - lower for consistency
    "metadata": {
        "compliance": "SOC2",
        "specializations": ["market_risk", "credit_risk"]
    }
}
```

### 2. Task Decomposition (Orchestrator)

The orchestrator decomposes tasks and identifies required skills:

```python
# real_task_decomposer.py output
{
    "subtasks": [
        {
            "subtask_id": "task_123_risk",
            "description": "Analyze portfolio risk metrics",
            "skills_required": ["risk_analysis", "statistical_modeling"],
            "priority": "high"
        }
    ]
}
```

### 3. Agent Selection

The orchestrator finds the best agent for each subtask:

```python
# Integration point
from services.agent_factory_v2 import AgentFactory

factory = AgentFactory()

# Find agent matching required skills
agent = await factory.get_agent_by_skills(
    required_skills=subtask["skills_required"],
    agent_type=None  # Or specific type if needed
)
```

### 4. Prompt Engineering (Context Engineering)

The orchestrator builds prompts using the atomic → molecular → cellular approach:

```python
async def build_prompt_for_subtask(subtask, context_store):
    """
    Build engineered prompt using Context Engineering principles
    """
    
    # 1. ATOMIC - Basic instruction
    atomic = subtask["description"]
    
    # 2. MOLECULAR - Add examples and format
    examples = await context_store.get_examples(subtask["skills_required"])
    molecular = f"""
Task: {atomic}

Examples:
{examples}

Expected format:
- Analysis
- Findings
- Recommendations
"""
    
    # 3. CELLULAR - Add context and memory
    context = await context_store.get_context(subtask["task_id"])
    system_prompt = f"""
You are performing: {subtask["description"]}
Required skills: {', '.join(subtask["skills_required"])}

Context:
{context}

Previous related tasks:
{await get_related_task_history(subtask)}
"""
    
    return molecular, system_prompt
```

### 5. Execution

The agent factory executes with orchestrator-provided prompts:

```python
# Agent executes with orchestrator's prompt
result = await factory.execute_with_prompt(
    agent=agent,
    prompt=molecular_prompt,      # From orchestrator
    system_prompt=cellular_prompt, # From orchestrator
    use_memory=True
)
```

## Integration Code Example

Here's how to integrate the orchestrator with the new agent factory:

```python
# In orchestrator service
from services.agent_factory_v2 import AgentFactory
from core.real_task_decomposer import RealTaskDecomposer
from services.vector_store import VectorStore  # Your context store

class EnhancedOrchestrator:
    def __init__(self):
        self.decomposer = RealTaskDecomposer()
        self.agent_factory = AgentFactory()
        self.vector_store = VectorStore()
    
    async def process_task(self, task_description: str):
        """
        Complete task processing with Context Engineering
        """
        
        # Step 1: Decompose task
        decomposition = await self.decomposer.decompose_task(task_description)
        
        results = []
        for subtask in decomposition["subtasks"]:
            
            # Step 2: Find best agent
            agent = await self.agent_factory.get_agent_by_skills(
                required_skills=subtask.get("skills_required", [])
            )
            
            if not agent:
                # Create agent on-demand if none exists
                agent = await self.agent_factory.create_agent({
                    "name": f"Agent-{subtask['subtask_id']}",
                    "type": subtask.get("agent_type", "general"),
                    "skills": subtask.get("skills_required", [])
                })
            
            # Step 3: Build prompt using Context Engineering
            
            # Query vector store for relevant context
            context_docs = await self.vector_store.query(
                query=subtask["description"],
                k=5
            )
            
            # Build molecular prompt
            molecular_prompt = self._build_molecular_prompt(
                task=subtask["description"],
                examples=context_docs.get("examples", [])
            )
            
            # Build cellular system prompt
            cellular_prompt = self._build_cellular_prompt(
                task=subtask["description"],
                context=context_docs.get("context", ""),
                skills=subtask.get("skills_required", [])
            )
            
            # Step 4: Execute with agent
            result = await self.agent_factory.execute_with_prompt(
                agent=agent,
                prompt=molecular_prompt,
                system_prompt=cellular_prompt,
                context={
                    "subtask_id": subtask["subtask_id"],
                    "dependencies": subtask.get("dependencies", [])
                },
                use_memory=True
            )
            
            results.append({
                "subtask_id": subtask["subtask_id"],
                "agent_used": agent.metadata.name,
                "result": result
            })
        
        return {
            "task": task_description,
            "decomposition": decomposition,
            "results": results
        }
    
    def _build_molecular_prompt(self, task: str, examples: list) -> str:
        """Build molecular prompt with examples"""
        prompt = f"Task: {task}\n\n"
        
        if examples:
            prompt += "Examples:\n"
            for ex in examples[:3]:  # Limit to 3 examples
                prompt += f"- {ex}\n"
            prompt += "\n"
        
        prompt += "Provide a comprehensive solution."
        return prompt
    
    def _build_cellular_prompt(self, task: str, context: str, skills: list) -> str:
        """Build cellular system prompt with full context"""
        return f"""You are executing: {task}

Your capabilities include: {', '.join(skills)}

Relevant context:
{context}

Follow best practices and provide detailed analysis."""
```

## Skill Matching Strategy

### MVP (Current)
Simple tag matching:
```python
def match_skills(required: List[str], agent_skills: List[str]) -> float:
    """Simple intersection-based matching"""
    if not required:
        return 1.0
    common = set(required) & set(agent_skills)
    return len(common) / len(required)
```

### Future (Vector Embeddings)
Semantic similarity matching:
```python
async def match_skills_semantic(required: List[str], agent_skills: List[str]) -> float:
    """Semantic similarity using embeddings"""
    required_embeddings = await embed_skills(required)
    agent_embeddings = await embed_skills(agent_skills)
    
    # Cosine similarity
    similarity = cosine_similarity(required_embeddings, agent_embeddings)
    return similarity.max(axis=1).mean()
```

## API Flow Example

```python
# 1. User creates agent
POST /api/v2/agents/create
{
    "name": "DataAnalyst-Alpha",
    "type": "data_analysis",
    "skills": ["python", "sql", "statistics", "visualization"]
}

# 2. User submits task
POST /api/orchestrator/process
{
    "task": "Analyze Q4 sales data and create visualizations"
}

# 3. Orchestrator decomposes and routes
# - Finds DataAnalyst-Alpha via skill matching
# - Builds Context Engineering prompts
# - Executes via agent factory

# 4. Results returned
{
    "task": "Analyze Q4 sales data...",
    "agents_used": ["DataAnalyst-Alpha"],
    "results": [...]
}
```

## Configuration for MVP

### Environment Variables
```bash
# Default LLM for agents
LLM_PROVIDER=anthropic
LLM_MODEL=claude-3-sonnet-20240229
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=2000

# API Keys
ANTHROPIC_API_KEY=your-key
OPENAI_API_KEY=your-key  # If users want GPT models
```

### Database Schema
```sql
-- Agents table (existing)
CREATE TABLE agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    agent_type VARCHAR(100),  -- User-defined type
    configuration JSON,        -- Stores all metadata
    status VARCHAR(50),
    ...
);

-- Skills table (existing)
CREATE TABLE skills (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    category VARCHAR(100),
    ...
);

-- Agent-Skills relation (existing)
CREATE TABLE agent_skills (
    agent_id INT REFERENCES agents(id),
    skill_id INT REFERENCES skills(id)
);
```

## Next Steps

1. **Immediate**: Update orchestrator to use `agent_factory_v2.py`
2. **Soon**: Implement vector store integration for context retrieval
3. **Future**: Add skill embeddings for semantic matching
4. **Later**: Fine-tuning integration based on skills

## Testing the Integration

```python
# Test script
async def test_integration():
    orchestrator = EnhancedOrchestrator()
    
    # Create test agent
    await orchestrator.agent_factory.create_agent({
        "name": "TestAnalyst",
        "type": "analyst",
        "skills": ["analysis", "reporting"]
    })
    
    # Process task
    result = await orchestrator.process_task(
        "Analyze website traffic and create a report"
    )
    
    print(f"Task processed using {len(result['results'])} agents")
    print(f"Results: {result}")

asyncio.run(test_integration())
```

## Summary

The new architecture provides:
- ✅ User-defined agents (no hard-coding)
- ✅ Orchestrator handles prompt engineering
- ✅ Context Engineering integration ready
- ✅ Skill-based agent matching
- ✅ Clean separation of concerns
- ✅ Future-proof for embeddings and fine-tuning
