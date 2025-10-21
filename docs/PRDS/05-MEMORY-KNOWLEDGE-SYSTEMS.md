# PRD 05: Memory & Knowledge Systems

## 1. Overview

### Purpose
Implement hierarchical memory systems that enable agents to learn, remember, and improve over time, transforming them from stateless to truly intelligent entities.

### Vision Alignment
Memory transforms agents from simple "molecules" to living "cells" that:
- Remember past interactions
- Learn from experience
- Build domain knowledge
- Share collective intelligence

## 2. Problem Statement

Current system lacks:
- Persistent agent memory
- Knowledge accumulation
- Experience-based learning
- Memory consolidation
- Collective intelligence

## 3. Success Criteria

- [ ] Agents retain memory across sessions
- [ ] Performance improves with experience
- [ ] Knowledge transfers between agents
- [ ] Memory retrieval is fast and relevant
- [ ] System learns from all interactions

## 4. Functional Requirements

### 4.1 Memory Hierarchy

```python
class MemoryHierarchy:
    """
    Multi-level memory system inspired by human cognition
    """
    
    WORKING_MEMORY = {
        "capacity": 7,  # Miller's Law
        "duration": "5 minutes",
        "purpose": "Active task context"
    }
    
    SHORT_TERM_MEMORY = {
        "capacity": 100,
        "duration": "24 hours",
        "purpose": "Recent interactions"
    }
    
    LONG_TERM_MEMORY = {
        "capacity": "unlimited",
        "duration": "permanent",
        "purpose": "Consolidated knowledge"
    }
    
    COLLECTIVE_MEMORY = {
        "capacity": "unlimited",
        "duration": "permanent",
        "purpose": "Shared organizational knowledge"
    }
```

### 4.2 Memory Operations

```python
class MemoryManager:
    """
    Manages all memory operations for agents
    """
    
    async def store_memory(
        self,
        agent_id: str,
        memory_item: MemoryItem,
        importance: float
    ) -> MemoryId:
        # Calculate memory strength
        # Determine storage level
        # Create associations
        # Update indices
        
    async def retrieve_memory(
        self,
        agent_id: str,
        query: str,
        memory_type: MemoryType = None,
        top_k: int = 5
    ) -> List[Memory]:
        # Search across memory levels
        # Rank by relevance
        # Apply forgetting curve
        # Return memories with context
        
    async def consolidate_memory(
        self,
        agent_id: str,
        consolidation_strategy: str = "sleep"
    ) -> ConsolidationResult:
        # Move important memories to long-term
        # Extract patterns and abstractions
        # Create knowledge structures
        # Prune redundant memories
```

### 4.3 Knowledge Graphs

```python
class KnowledgeGraph:
    """
    Structured knowledge representation
    """
    
    async def add_knowledge(
        self,
        subject: str,
        predicate: str,
        object: str,
        confidence: float
    ) -> KnowledgeNode:
        # Create nodes and edges
        # Update graph structure
        # Calculate centrality
        # Propagate changes
        
    async def query_knowledge(
        self,
        query: str,
        inference_depth: int = 2
    ) -> KnowledgeResult:
        # Parse query
        # Traverse graph
        # Apply inference rules
        # Return structured knowledge
```

### 4.4 Learning Mechanisms

```python
class LearningEngine:
    """
    Enables agents to learn from experience
    """
    
    async def learn_from_feedback(
        self,
        agent_id: str,
        task: Task,
        result: Result,
        feedback: Feedback
    ) -> LearningOutcome:
        # Analyze performance
        # Update success patterns
        # Adjust strategies
        # Store learned knowledge
        
    async def transfer_learning(
        self,
        from_agent: Agent,
        to_agent: Agent,
        knowledge_domain: str
    ) -> TransferResult:
        # Extract transferable knowledge
        # Adapt to target agent
        # Integrate with existing knowledge
        # Test understanding
```

## 5. Technical Architecture

### 5.1 Memory System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Memory System                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Working Memory (Redis)                                  │
│  ┌────────────────────┐                                  │
│  │ Current Task       │ ← 5 min TTL                      │
│  │ Active Context     │                                  │
│  └────────────────────┘                                  │
│           ↓                                              │
│  Short-term Memory (PostgreSQL)                          │
│  ┌────────────────────┐                                  │
│  │ Recent Sessions    │ ← 24 hour window                 │
│  │ Task History       │                                  │
│  └────────────────────┘                                  │
│           ↓ Consolidation                                │
│  Long-term Memory (PostgreSQL + pgvector)                │
│  ┌────────────────────┐                                  │
│  │ Knowledge Base     │ ← Permanent                      │
│  │ Learned Patterns   │                                  │
│  │ Success Strategies │                                  │
│  └────────────────────┘                                  │
│           ↓                                              │
│  Collective Memory (Graph DB / PostgreSQL)               │
│  ┌────────────────────┐                                  │
│  │ Organizational     │ ← Shared across agents           │
│  │ Knowledge Graph    │                                  │
│  └────────────────────┘                                  │
└─────────────────────────────────────────────────────────┘
```

## 6. Implementation Details

### 6.1 Memory Storage & Retrieval

```python
class HierarchicalMemorySystem:
    """
    Complete memory implementation
    """
    
    async def store_experience(
        self,
        agent_id: str,
        experience: Experience
    ):
        # Step 1: Store in working memory (Redis)
        await self.redis_client.setex(
            f"working:{agent_id}:{experience.id}",
            300,  # 5 minutes
            json.dumps(experience.to_dict())
        )
        
        # Step 2: Store in short-term (PostgreSQL)
        db_memory = MemoryItem(
            agent_id=agent_id,
            content=experience.content,
            memory_type="experience",
            importance=self.calculate_importance(experience),
            embedding=self.generate_embedding(experience.content),
            metadata={
                "task_id": experience.task_id,
                "success": experience.success,
                "timestamp": datetime.now()
            }
        )
        self.db.add(db_memory)
        
        # Step 3: Schedule consolidation
        if db_memory.importance > 0.7:
            await self.schedule_consolidation(agent_id, db_memory.id)
    
    async def retrieve_relevant_memories(
        self,
        agent_id: str,
        context: str,
        memory_types: List[str] = None
    ) -> List[Memory]:
        """
        Multi-level memory retrieval with ranking
        """
        memories = []
        
        # Check working memory
        working_memories = await self.get_working_memories(agent_id)
        memories.extend(working_memories)
        
        # Query short-term memory
        query_embedding = self.generate_embedding(context)
        
        # Vector similarity search using pgvector
        similar_memories = self.db.query(MemoryItem).filter(
            MemoryItem.agent_id == agent_id,
            MemoryItem.created_at > datetime.now() - timedelta(days=1)
        ).order_by(
            MemoryItem.embedding.l2_distance(query_embedding)
        ).limit(10).all()
        
        # Query long-term memory with forgetting curve
        long_term_memories = self.db.query(MemoryItem).filter(
            MemoryItem.agent_id == agent_id,
            MemoryItem.memory_level == "long_term"
        ).all()
        
        # Apply forgetting curve
        for memory in long_term_memories:
            memory.relevance = self.apply_forgetting_curve(
                memory,
                query_embedding
            )
        
        # Combine and rank
        all_memories = memories + similar_memories + long_term_memories
        ranked_memories = self.rank_memories(all_memories, context)
        
        return ranked_memories[:10]
```

### 6.2 Knowledge Consolidation

```python
async def consolidate_memories(
    self,
    agent_id: str
):
    """
    Consolidate short-term memories into long-term knowledge
    """
    # Get recent high-importance memories
    recent_memories = self.db.query(MemoryItem).filter(
        MemoryItem.agent_id == agent_id,
        MemoryItem.importance > 0.6,
        MemoryItem.memory_level == "short_term",
        MemoryItem.created_at > datetime.now() - timedelta(hours=24)
    ).all()
    
    # Group by patterns
    pattern_groups = self.identify_patterns(recent_memories)
    
    for pattern in pattern_groups:
        # Create abstracted knowledge
        knowledge = self.abstract_pattern(pattern)
        
        # Store as long-term memory
        long_term_memory = MemoryItem(
            agent_id=agent_id,
            content=knowledge.description,
            memory_type="knowledge",
            memory_level="long_term",
            importance=knowledge.importance,
            embedding=knowledge.embedding,
            metadata={
                "pattern_type": knowledge.pattern_type,
                "confidence": knowledge.confidence,
                "source_memories": [m.id for m in pattern.memories]
            }
        )
        self.db.add(long_term_memory)
        
        # Add to knowledge graph
        await self.add_to_knowledge_graph(knowledge)
    
    # Prune redundant short-term memories
    await self.prune_memories(agent_id)
```

## 7. Database Schema

```sql
-- Enhanced memory items table
CREATE TABLE memory_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id),
    content TEXT,
    memory_type VARCHAR(50),
    memory_level VARCHAR(50), -- working, short_term, long_term
    importance FLOAT,
    embedding VECTOR(1536),
    access_count INTEGER DEFAULT 0,
    last_access TIMESTAMP,
    decay_rate FLOAT DEFAULT 0.1,
    associations UUID[], -- Related memory IDs
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Knowledge graph nodes
CREATE TABLE knowledge_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id INTEGER REFERENCES agents(id),
    concept VARCHAR(255),
    description TEXT,
    node_type VARCHAR(50),
    embedding VECTOR(1536),
    importance FLOAT,
    confidence FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Knowledge graph edges
CREATE TABLE knowledge_edges (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_node_id UUID REFERENCES knowledge_nodes(id),
    to_node_id UUID REFERENCES knowledge_nodes(id),
    relationship VARCHAR(100),
    strength FLOAT,
    evidence_count INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Learning outcomes
CREATE TABLE learning_outcomes (
    id SERIAL PRIMARY KEY,
    agent_id INTEGER REFERENCES agents(id),
    task_id INTEGER REFERENCES tasks(id),
    learned_pattern TEXT,
    success_rate FLOAT,
    confidence FLOAT,
    application_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 8. API Endpoints

```python
# Store memory
POST /api/memory/store
{
    "agent_id": "...",
    "content": "...",
    "importance": 0.8,
    "associations": [...]
}

# Retrieve memories
POST /api/memory/retrieve
{
    "agent_id": "...",
    "query": "...",
    "memory_types": ["experience", "knowledge"],
    "top_k": 5
}

# Consolidate memories
POST /api/memory/consolidate
{
    "agent_id": "...",
    "strategy": "pattern_extraction"
}

# Query knowledge graph
POST /api/knowledge/query
{
    "query": "What do I know about API design?",
    "inference_depth": 2,
    "confidence_threshold": 0.7
}
```

## 9. Success Metrics

- Memory retrieval latency: < 100ms
- Relevance accuracy: > 85%
- Knowledge retention rate: > 90%
- Learning improvement: > 30% over baseline
- Memory utilization: < 80% capacity
