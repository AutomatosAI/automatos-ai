# PRD 03: Context Engineering Layer

## 1. Overview

### Purpose
The Context Engineering Layer implements the mathematical and algorithmic foundations for optimal prompt construction, following the atoms → molecules → cells → organs progression from simple prompts to complex, context-aware instructions.

### Vision Alignment
- **Atoms**: Basic instructions optimized for clarity
- **Molecules**: Instructions + examples + patterns
- **Cells**: Memory-augmented contextual prompts
- **Organs**: Multi-agent coordinated contexts
- **Mathematical Selection**: Information theory for context optimization

## 2. Problem Statement

Current system lacks:
- Mathematical context selection
- Dynamic example retrieval
- Pattern recognition and reuse
- Context window optimization
- Semantic chunking
- Information density maximization

## 3. Success Criteria

- [ ] Prompts follow Context Engineering principles
- [ ] Mathematical optimization of context selection
- [ ] Dynamic example retrieval based on similarity
- [ ] Context window utilization > 80%
- [ ] Improved task success rates via better prompts

## 4. Functional Requirements

### 4.1 Atomic Prompt Engineering

```python
class AtomicPromptBuilder:
    """
    Creates optimized single-instruction prompts
    """
    
    def create_atomic_prompt(
        self,
        task: str,
        constraints: List[str],
        output_format: str
    ) -> AtomicPrompt:
        # Optimize instruction clarity
        # Remove ambiguity
        # Specify precise outputs
        # Minimize token usage
        
    def measure_prompt_entropy(self, prompt: str) -> float:
        """
        Calculate information entropy of prompt
        Using Shannon entropy: H(X) = -Σ p(x) log p(x)
        """
```

### 4.2 Molecular Context Construction

```python
class MolecularContextBuilder:
    """
    Combines prompts with examples and patterns
    """
    
    async def build_molecular_context(
        self,
        atomic_prompt: AtomicPrompt,
        example_count: int = 3,
        pattern_type: str = "few_shot"
    ) -> MolecularContext:
        # Retrieve relevant examples
        # Select optimal patterns
        # Structure for few-shot learning
        # Calculate information gain
        
    async def select_optimal_examples(
        self,
        query: str,
        candidate_pool: List[Example],
        k: int = 3
    ) -> List[Example]:
        """
        Select examples using:
        - Cosine similarity in embedding space
        - Coverage of edge cases
        - Diversity maximization
        - Recency weighting
        """
```

### 4.3 Mathematical Context Optimization

```python
class ContextOptimizer:
    """
    Mathematically optimizes context selection
    """
    
    def optimize_context(
        self,
        available_context: List[ContextItem],
        max_tokens: int,
        objective: str = "maximize_information"
    ) -> OptimizedContext:
        """
        Optimization using:
        - Information gain: IG(D,A) = H(D) - H(D|A)
        - Mutual information: I(X;Y) = H(X) - H(X|Y)
        - KL divergence for relevance
        - Knapsack algorithm for token budget
        """
        
    def calculate_information_density(
        self,
        context: str
    ) -> InformationMetrics:
        """
        Returns:
        - Shannon entropy
        - Compression ratio
        - Semantic density
        - Redundancy score
        """
```

### 4.4 Semantic Chunking & Retrieval

```python
class SemanticChunker:
    """
    Intelligently chunks and retrieves context
    """
    
    async def chunk_document(
        self,
        document: str,
        chunk_strategy: str = "semantic"
    ) -> List[Chunk]:
        # Identify semantic boundaries
        # Maintain context coherence
        # Optimize chunk size for retrieval
        # Preserve relationships
        
    async def retrieve_chunks(
        self,
        query: str,
        top_k: int = 5,
        rerank: bool = True
    ) -> List[RankedChunk]:
        # Embed query
        # Vector similarity search
        # Rerank using cross-encoder
        # Apply MMR for diversity
```

## 5. Technical Architecture

### 5.1 Context Engineering Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                 Context Engineering Layer                │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input Task                                              │
│      ↓                                                   │
│  ┌────────────────┐                                      │
│  │ Atomic Builder │ → Basic Instruction                  │
│  └────────────────┘                                      │
│      ↓                                                   │
│  ┌────────────────┐                                      │
│  │ Example Retriever│ → Relevant Examples                │
│  └────────────────┘                                      │
│      ↓                                                   │
│  ┌────────────────┐                                      │
│  │ Pattern Matcher │ → Applicable Patterns               │
│  └────────────────┘                                      │
│      ↓                                                   │
│  ┌────────────────┐                                      │
│  │ Memory Augmenter│ → Historical Context                │
│  └────────────────┘                                      │
│      ↓                                                   │
│  ┌────────────────┐                                      │
│  │ Math Optimizer  │ → Optimized Selection               │
│  └────────────────┘                                      │
│      ↓                                                   │
│  Final Context                                           │
└─────────────────────────────────────────────────────────┘
```

### 5.2 Mathematical Foundations Integration

```python
# From context_engineering/mathematical_foundations/
from mathematical_foundations.information_theory import (
    calculate_entropy,
    calculate_mutual_information,
    calculate_kl_divergence
)
from mathematical_foundations.vector_operations import (
    cosine_similarity,
    euclidean_distance,
    manhattan_distance
)
from mathematical_foundations.optimization_algorithms import (
    knapsack_optimization,
    gradient_descent,
    simulated_annealing
)
```

## 6. Implementation Details

### 6.1 Context Selection Algorithm

```python
async def select_optimal_context(
    self,
    query: str,
    available_contexts: List[ContextItem],
    token_budget: int
) -> OptimalContext:
    """
    Multi-objective optimization for context selection
    """
    # Step 1: Calculate relevance scores
    relevance_scores = []
    for context in available_contexts:
        embedding_similarity = cosine_similarity(
            query_embedding,
            context.embedding
        )
        semantic_overlap = calculate_semantic_overlap(
            query,
            context.text
        )
        relevance = 0.7 * embedding_similarity + 0.3 * semantic_overlap
        relevance_scores.append(relevance)
    
    # Step 2: Calculate information gain
    information_gains = []
    for context in available_contexts:
        entropy_before = calculate_entropy(query)
        entropy_after = calculate_entropy(query + context.text)
        info_gain = entropy_before - entropy_after
        information_gains.append(info_gain)
    
    # Step 3: Apply knapsack optimization
    selected_indices = knapsack_optimization(
        values=[(r * i) for r, i in zip(relevance_scores, information_gains)],
        weights=[context.token_count for context in available_contexts],
        capacity=token_budget
    )
    
    # Step 4: Order by dependency and coherence
    selected_contexts = [available_contexts[i] for i in selected_indices]
    ordered_contexts = order_by_coherence(selected_contexts)
    
    return OptimalContext(
        contexts=ordered_contexts,
        total_tokens=sum(c.token_count for c in ordered_contexts),
        expected_information_gain=sum(information_gains[i] for i in selected_indices)
    )
```

### 6.2 Few-Shot Example Selection

```python
def select_few_shot_examples(
    self,
    task: Task,
    example_pool: List[Example],
    k: int = 3
) -> List[Example]:
    """
    Select diverse, relevant examples for few-shot learning
    """
    selected = []
    remaining = example_pool.copy()
    
    # Select first example (most similar)
    similarities = [
        cosine_similarity(task.embedding, ex.embedding)
        for ex in remaining
    ]
    best_idx = np.argmax(similarities)
    selected.append(remaining.pop(best_idx))
    
    # Select remaining examples (MMR - Maximum Marginal Relevance)
    while len(selected) < k and remaining:
        mmr_scores = []
        for candidate in remaining:
            # Relevance to query
            relevance = cosine_similarity(task.embedding, candidate.embedding)
            
            # Maximum similarity to already selected
            max_sim = max([
                cosine_similarity(candidate.embedding, s.embedding)
                for s in selected
            ])
            
            # MMR score
            mmr = 0.7 * relevance - 0.3 * max_sim
            mmr_scores.append(mmr)
        
        best_idx = np.argmax(mmr_scores)
        selected.append(remaining.pop(best_idx))
    
    return selected
```

## 7. Database Schema Updates

```sql
-- Context templates
CREATE TABLE context_templates (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255),
    template_type VARCHAR(50), -- atomic, molecular, cellular
    template_text TEXT,
    parameters JSONB,
    usage_count INTEGER DEFAULT 0,
    success_rate FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Example library
CREATE TABLE context_examples (
    id SERIAL PRIMARY KEY,
    category VARCHAR(100),
    input_text TEXT,
    output_text TEXT,
    embedding VECTOR(1536), -- For pgvector
    metadata JSONB,
    quality_score FLOAT,
    usage_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Pattern library
CREATE TABLE context_patterns (
    id SERIAL PRIMARY KEY,
    pattern_name VARCHAR(255),
    pattern_type VARCHAR(50), -- few_shot, chain_of_thought, etc
    pattern_structure JSONB,
    applicable_tasks JSONB,
    effectiveness_score FLOAT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Context optimization logs
CREATE TABLE context_optimizations (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(id),
    original_tokens INTEGER,
    optimized_tokens INTEGER,
    information_gain FLOAT,
    optimization_strategy VARCHAR(100),
    metrics JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

## 8. API Endpoints

```python
# Optimize context for task
POST /api/context/optimize
{
    "task_description": "...",
    "available_context": [...],
    "token_budget": 4000,
    "optimization_objective": "maximize_information"
}

# Get few-shot examples
POST /api/context/examples/select
{
    "task": "...",
    "example_count": 3,
    "selection_strategy": "mmr",
    "diversity_weight": 0.3
}

# Analyze prompt quality
POST /api/context/analyze
{
    "prompt": "...",
    "metrics": ["entropy", "clarity", "specificity", "ambiguity"]
}

# Chunk document
POST /api/context/chunk
{
    "document": "...",
    "chunk_strategy": "semantic",
    "max_chunk_size": 500,
    "overlap": 50
}
```

## 9. Integration Points

### Use Existing Modules

```python
# In context_manager.py
from context_engineering.prompt_builder import ContextAwarePromptBuilder
from context_engineering.mathematical_foundations.information_theory import *
from context_engineering.mathematical_foundations.vector_operations import *
from context_engineering.retrieval.context_retrieval_engine import ContextRetrievalEngine
from context_engineering.chunking.semantic_chunker import SemanticChunker
```

## 10. Testing Strategy

### Unit Tests
- Atomic prompt generation
- Example selection algorithm
- Mathematical optimization
- Chunking accuracy

### Integration Tests
- End-to-end context generation
- Retrieval accuracy
- Token budget adherence
- Information gain measurement

### Quality Metrics
- Prompt clarity score
- Information density
- Task success correlation
- Token efficiency

## 11. Dependencies

- **Existing**: All `context_engineering/` modules
- **PRD 01**: Orchestration (provides tasks)
- **PRD 02**: Agent Factory (consumes contexts)
- **PRD 05**: Memory Systems (historical context)

## 12. Timeline

- Week 1: Atomic/molecular builders
- Week 2: Mathematical optimization
- Week 3: Retrieval integration
- Week 4: Testing and tuning

## 13. Success Metrics

- Information density improvement: > 40%
- Token usage reduction: > 30%
- Task success rate improvement: > 25%
- Example relevance score: > 0.85
- Context coherence score: > 0.90
