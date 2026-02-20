"""
Context Optimizer - Mathematical Optimization for Prompt Engineering
====================================================================

Implements PRD-03: Context Engineering Layer with REAL mathematical optimization.
Uses information theory, embeddings, and optimization algorithms for optimal prompt construction.

This module provides:
- Atomic → Molecular → Cellular prompt progression
- Information-theoretic context selection
- Few-shot example selection using cosine similarity
- Token budget optimization using knapsack algorithm
- Real entropy and mutual information calculations

Author: Automatos AI Platform
Version: 1.0.0

Migrated to: modules/search/optimization/context_optimizer.py
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
from collections import Counter
import math
import tiktoken

# OpenAI for embeddings
from openai import OpenAI  # For sync LLM calls only (not embeddings)

# Import mathematical foundations from shared
from core.math import InformationTheory, VectorOperations, OptimizationAlgorithms

# Use centralized embedding manager



def _get_system_dimension() -> int:
    """Get embedding dimension from system settings"""
    try:
        from core.database.database import SessionLocal
        from core.models.system_settings import SystemSetting
        db = SessionLocal()
        try:
            setting = db.query(SystemSetting).filter(
                SystemSetting.key == "vector_store_dimensions"
            ).first()
            if setting and setting.value:
                return int(setting.value)
        finally:
            db.close()
    except Exception:
        pass
    return 4096  # Fallback if DB unavailable

# PromptType and PromptTemplate defined locally (only consumer)
from enum import Enum

class PromptType(Enum):
    """Types of prompts for different tasks"""
    CODE_GENERATION = "code_generation"
    CODE_ANALYSIS = "code_analysis"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    API_DEVELOPMENT = "api_development"
    TESTING = "testing"
    ARCHITECTURE = "architecture"
    OPTIMIZATION = "optimization"
    GENERAL = "general"

@dataclass
class PromptTemplate:
    """Template for prompt generation"""
    name: str
    prompt_type: PromptType
    system_prompt: str
    user_template: str
    context_template: str
    max_context_length: int = 4000
    required_context_types: List[str] = None
    success_rate: float = 0.0
    usage_count: int = 0

logger = logging.getLogger(__name__)

# Token counter for OpenAI models
try:
    encoding = tiktoken.encoding_for_model("gpt-4")
except:
    encoding = tiktoken.get_encoding("cl100k_base")

@dataclass
class ContextItem:
    """Represents a piece of context with metadata"""
    content: str
    source: str
    context_type: str  # code, documentation, example, etc.
    relevance_score: float = 0.0
    token_count: int = 0
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Calculate token count on initialization"""
        if not self.token_count:
            self.token_count = len(encoding.encode(self.content))

@dataclass
class Example:
    """Represents a few-shot example"""
    input_text: str
    output_text: str
    category: str
    quality_score: float = 1.0
    embedding: Optional[List[float]] = None
    token_count: int = 0
    
    def __post_init__(self):
        """Calculate total token count"""
        if not self.token_count:
            full_text = f"Input: {self.input_text}\nOutput: {self.output_text}"
            self.token_count = len(encoding.encode(full_text))

@dataclass
class OptimizedContext:
    """Result of context optimization"""
    contexts: List[ContextItem]
    total_tokens: int
    information_gain: float
    entropy_reduction: float
    diversity_score: float
    optimization_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AtomicPrompt:
    """Atomic level prompt - basic instruction"""
    instruction: str
    constraints: List[str]
    output_format: str
    token_count: int = 0
    clarity_score: float = 0.0
    
    def __post_init__(self):
        if not self.token_count:
            full_prompt = self._build_prompt()
            self.token_count = len(encoding.encode(full_prompt))
    
    def _build_prompt(self) -> str:
        """Build the full atomic prompt"""
        prompt = f"{self.instruction}\n"
        if self.constraints:
            prompt += "\nConstraints:\n" + "\n".join(f"- {c}" for c in self.constraints)
        if self.output_format:
            prompt += f"\n\nOutput Format: {self.output_format}"
        return prompt

@dataclass
class EnhancedPrompt:
    """Molecular/Cellular level prompt with examples and context"""
    atomic_prompt: AtomicPrompt
    examples: List[Example]
    context_items: List[ContextItem]
    total_tokens: int = 0
    expected_performance: float = 0.0
    
    def __post_init__(self):
        if not self.total_tokens:
            self.total_tokens = (
                self.atomic_prompt.token_count +
                sum(ex.token_count for ex in self.examples) +
                sum(ctx.token_count for ctx in self.context_items)
            )

class ContextOptimizer:
    """
    Mathematical optimizer for context selection and prompt engineering.
    Implements real information theory and optimization algorithms.
    """
    
    def optimize_agent_resources(self, num_agents: int, resource_limit: float) -> Dict[str, float]:
        """
        Calculates optimal agent allocation and efficiency gains.
        Restored logic from original implementation.
        
        Args:
            num_agents: Number of available agents
            resource_limit: Resource constraint factor (0.0 - 1.0)
            
        Returns:
            Dictionary with optimization metrics
        """
        # Ensure inputs are safe
        num_agents = max(1, num_agents)
        resource_limit = max(0.1, resource_limit)
        
        optimal_agents = min(num_agents + 2, int(resource_limit * 10))
        
        if num_agents > 0:
            performance_boost = (optimal_agents / num_agents) * 0.2
        else:
            performance_boost = 0.3
            
        # Avoid division by zero in efficiency gain
        denom = resource_limit * 10
        efficiency_gain = 1.0 - (optimal_agents / denom) if denom > 0 else 0.0
        
        return {
            "optimal_agents": optimal_agents,
            "performance_boost": performance_boost,
            "efficiency_gain": efficiency_gain,
            "resource_utilization": optimal_agents / denom if denom > 0 else 1.0
        }

    def __init__(self):
        """
        Initialize the context optimizer with centralized embedding manager.
        """
        # Use centralized embedding manager (reads from General Settings)
        from core.llm import create_embedding_manager
        self.embedding_manager = create_embedding_manager()
        self.embedding_dim = self.embedding_manager.get_dimension()
        provider_info = self.embedding_manager.get_provider_info()
        
        logger.info(
            f"Context Optimizer using {provider_info['provider']} embeddings "
            f"(model: {provider_info.get('model', 'N/A')}, dimension: {self.embedding_dim})"
        )
        
        # Initialize mathematical foundations
        self.info_theory = InformationTheory()
        self.vector_ops = VectorOperations()
        self.optimization = OptimizationAlgorithms()
        
        # Cache for embeddings
        self._embedding_cache = {}
    
    async def optimize_context(
        self,
        available_context: List[ContextItem],
        max_tokens: int,
        objective: str = "maximize_information"
    ) -> OptimizedContext:
        """
        Optimize context selection using mathematical algorithms.
        
        Uses:
        - Information gain calculation: IG(D,A) = H(D) - H(D|A)
        - Knapsack optimization for token budget
        - Diversity maximization for coverage
        
        Args:
            available_context: List of available context items
            max_tokens: Maximum token budget
            objective: Optimization objective
            
        Returns:
            OptimizedContext with selected items and metrics
        """
        if not available_context:
            return OptimizedContext(
                contexts=[],
                total_tokens=0,
                information_gain=0.0,
                entropy_reduction=0.0,
                diversity_score=0.0
            )
        
        # Step 1: Calculate information metrics for each context
        info_metrics = []
        for context in available_context:
            # Calculate real Shannon entropy
            entropy = self._calculate_shannon_entropy(context.content)
            
            # Calculate information density (entropy per token)
            density = entropy / max(context.token_count, 1)
            
            # Calculate semantic uniqueness (requires embeddings)
            if not context.embedding:
                context.embedding = await self._get_embedding(context.content)
            
            info_metrics.append({
                'context': context,
                'entropy': entropy,
                'density': density,
                'embedding': context.embedding
            })
        
        # Step 2: Calculate mutual information between contexts
        # This helps avoid redundant information
        mutual_info_matrix = self._calculate_mutual_information_matrix(info_metrics)
        
        # Step 3: Apply knapsack optimization with information value
        if objective == "maximize_information":
            selected_indices = self._knapsack_optimization(
                values=[m['density'] for m in info_metrics],
                weights=[c.token_count for c in available_context],
                capacity=max_tokens,
                mutual_info_matrix=mutual_info_matrix
            )
        elif objective == "maximize_diversity":
            selected_indices = self._diversity_optimization(
                info_metrics, max_tokens
            )
        else:
            # Default: balanced approach
            selected_indices = self._balanced_optimization(
                info_metrics, max_tokens, mutual_info_matrix
            )
        
        # Step 4: Order selected contexts by dependency and coherence
        selected_contexts = [available_context[i] for i in selected_indices]
        ordered_contexts = self._order_by_coherence(selected_contexts)
        
        # Step 5: Calculate final metrics
        total_tokens = sum(c.token_count for c in ordered_contexts)
        total_entropy = sum(info_metrics[i]['entropy'] for i in selected_indices)
        
        # Calculate diversity score using embeddings
        diversity_score = self._calculate_diversity_score(
            [info_metrics[i]['embedding'] for i in selected_indices]
        )
        
        # Calculate information gain
        baseline_entropy = np.mean([m['entropy'] for m in info_metrics])
        information_gain = total_entropy - baseline_entropy * len(selected_indices)
        
        return OptimizedContext(
            contexts=ordered_contexts,
            total_tokens=total_tokens,
            information_gain=max(0, information_gain),
            entropy_reduction=total_entropy,
            diversity_score=diversity_score,
            optimization_metadata={
                'objective': objective,
                'selected_count': len(selected_indices),
                'total_available': len(available_context),
                'token_utilization': total_tokens / max_tokens if max_tokens > 0 else 0
            }
        )
    
    async def select_few_shot_examples(
        self,
        task: str,
        examples: List[Example],
        k: int = 3
    ) -> List[Example]:
        """
        Select diverse, relevant examples for few-shot learning.
        Uses Maximum Marginal Relevance (MMR) with real embeddings.
        
        MMR = λ * relevance - (1-λ) * max_similarity_to_selected
        
        Args:
            task: Task description to match examples against
            examples: Pool of available examples
            k: Number of examples to select
            
        Returns:
            List of selected examples optimized for diversity and relevance
        """
        if not examples or k <= 0:
            return []
        
        if len(examples) <= k:
            return examples
        
        # Get task embedding
        task_embedding = await self._get_embedding(task)
        
        # Get embeddings for all examples if not present
        for example in examples:
            if not example.embedding:
                example_text = f"{example.input_text} {example.output_text}"
                example.embedding = await self._get_embedding(example_text)
        
        selected = []
        remaining = examples.copy()
        
        # Select first example (most similar to task)
        similarities = []
        for ex in remaining:
            sim = VectorOperations.cosine_similarity(task_embedding, ex.embedding)
            # Weight by quality score
            weighted_sim = sim * ex.quality_score
            similarities.append(weighted_sim)
        
        best_idx = np.argmax(similarities)
        selected.append(remaining.pop(best_idx))
        
        # Select remaining examples using MMR
        lambda_param = 0.7  # Balance between relevance and diversity
        
        while len(selected) < k and remaining:
            mmr_scores = []
            
            for candidate in remaining:
                # Relevance to task
                relevance = VectorOperations.cosine_similarity(task_embedding, candidate.embedding)
                relevance *= candidate.quality_score
                
                # Maximum similarity to already selected examples
                max_sim = max([
                    VectorOperations.cosine_similarity(candidate.embedding, s.embedding)
                    for s in selected
                ])
                
                # MMR score
                mmr = lambda_param * relevance - (1 - lambda_param) * max_sim
                mmr_scores.append(mmr)
            
            # Select example with highest MMR score
            best_idx = np.argmax(mmr_scores)
            selected.append(remaining.pop(best_idx))
        
        # Sort selected examples by relevance for better prompt flow
        selected.sort(
            key=lambda ex: VectorOperations.cosine_similarity(task_embedding, ex.embedding),
            reverse=True
        )
        
        return selected
    
    def calculate_information_density(self, text: str) -> float:
        """
        Calculate information density of text using real entropy.
        
        Information density = Shannon entropy / token count
        
        Args:
            text: Input text
            
        Returns:
            Information density score (bits per token)
        """
        if not text:
            return 0.0
        
        # Calculate Shannon entropy
        entropy = self._calculate_shannon_entropy(text)
        
        # Get token count
        token_count = len(encoding.encode(text))
        
        if token_count == 0:
            return 0.0
        
        # Information density = entropy per token
        density = entropy / token_count
        
        # Also calculate compression ratio as additional metric
        compressed_size = len(text.encode('utf-8'))
        original_size = len(text) * 8  # bits
        compression_ratio = compressed_size / original_size if original_size > 0 else 1
        
        # Combine metrics (weighted average)
        final_density = 0.7 * density + 0.3 * (1 - compression_ratio)
        
        return final_density
    
    async def build_molecular_context(
        self,
        atomic_prompt: AtomicPrompt,
        examples: List[Example],
        context_items: Optional[List[ContextItem]] = None,
        max_tokens: int = 4000
    ) -> EnhancedPrompt:
        """
        Build molecular-level context by combining atomic prompt with examples.
        
        Progression: Atomic (instruction) → Molecular (instruction + examples + context)
        
        Args:
            atomic_prompt: Base atomic prompt
            examples: Few-shot examples to include
            context_items: Optional additional context
            max_tokens: Maximum token budget
            
        Returns:
            Enhanced prompt with examples and optimized context
        """
        # Start with atomic prompt tokens
        remaining_tokens = max_tokens - atomic_prompt.token_count
        
        if remaining_tokens <= 0:
            # Atomic prompt exceeds budget
            return EnhancedPrompt(
                atomic_prompt=atomic_prompt,
                examples=[],
                context_items=[],
                total_tokens=atomic_prompt.token_count
            )
        
        # Allocate tokens: 60% for examples, 40% for context
        example_budget = int(remaining_tokens * 0.6)
        context_budget = remaining_tokens - example_budget
        
        # Select examples within budget
        selected_examples = []
        example_tokens = 0
        
        for example in examples:
            if example_tokens + example.token_count <= example_budget:
                selected_examples.append(example)
                example_tokens += example.token_count
        
        # Select context items if provided
        selected_context = []
        if context_items and context_budget > 0:
            # Optimize context selection
            optimized = await self.optimize_context(
                context_items,
                context_budget,
                objective="maximize_information"
            )
            selected_context = optimized.contexts
        
        # Calculate expected performance based on example quality and relevance
        avg_quality = np.mean([ex.quality_score for ex in selected_examples]) if selected_examples else 0.5
        context_boost = min(0.2, len(selected_context) * 0.05)  # Up to 20% boost from context
        expected_performance = min(1.0, avg_quality + context_boost)
        
        return EnhancedPrompt(
            atomic_prompt=atomic_prompt,
            examples=selected_examples,
            context_items=selected_context,
            total_tokens=atomic_prompt.token_count + example_tokens + sum(c.token_count for c in selected_context),
            expected_performance=expected_performance
        )
    
    # ============= Private Helper Methods =============
    
    def _calculate_shannon_entropy(self, text: str) -> float:
        """
        Calculate real Shannon entropy of text.
        H(X) = -Σ p(x) * log2(p(x))
        
        Args:
            text: Input text
            
        Returns:
            Shannon entropy in bits
        """
        if not text:
            return 0.0
        
        # Calculate character frequency distribution
        char_counts = Counter(text)
        total_chars = len(text)
        
        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            if count > 0:
                probability = count / total_chars
                entropy -= probability * math.log2(probability)
        
        return entropy
    
    def _calculate_mutual_information_matrix(
        self,
        info_metrics: List[Dict[str, Any]]
    ) -> np.ndarray:
        """
        Calculate mutual information between all pairs of contexts.
        I(X;Y) = H(X) + H(Y) - H(X,Y)
        
        Args:
            info_metrics: List of context metrics with embeddings
            
        Returns:
            Matrix of mutual information scores
        """
        n = len(info_metrics)
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Use embedding similarity as proxy for mutual information
                # (Real MI calculation would require joint distribution)
                if info_metrics[i]['embedding'] and info_metrics[j]['embedding']:
                    similarity = VectorOperations.cosine_similarity(
                        info_metrics[i]['embedding'],
                        info_metrics[j]['embedding']
                    )
                    
                    # Approximate mutual information
                    h_i = info_metrics[i]['entropy']
                    h_j = info_metrics[j]['entropy']
                    
                    # MI approximation: higher similarity = more mutual information
                    mi = similarity * min(h_i, h_j)
                    
                    matrix[i][j] = mi
                    matrix[j][i] = mi
        
        return matrix
    
    def _knapsack_optimization(
        self,
        values: List[float],
        weights: List[int],
        capacity: int,
        mutual_info_matrix: np.ndarray
    ) -> List[int]:
        """
        0/1 Knapsack optimization with mutual information penalty.
        
        Args:
            values: Information value of each item
            weights: Token count of each item
            capacity: Maximum token budget
            mutual_info_matrix: Mutual information between items
            
        Returns:
            Indices of selected items
        """
        n = len(values)
        if n == 0:
            return []
        
        # Dynamic programming table
        dp = [[0.0 for _ in range(capacity + 1)] for _ in range(n + 1)]
        
        # Build table
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                # Don't include item i-1
                dp[i][w] = dp[i-1][w]
                
                # Include item i-1 if it fits
                if weights[i-1] <= w:
                    # Value minus redundancy penalty
                    redundancy_penalty = 0
                    for j in range(i-1):
                        if dp[j][w - weights[i-1]] > 0:  # Item j is selected
                            redundancy_penalty += mutual_info_matrix[i-1][j] * 0.3
                    
                    value_with_item = (
                        dp[i-1][w - weights[i-1]] + 
                        values[i-1] - 
                        redundancy_penalty
                    )
                    
                    if value_with_item > dp[i][w]:
                        dp[i][w] = value_with_item
        
        # Backtrack to find selected items
        selected = []
        w = capacity
        for i in range(n, 0, -1):
            if dp[i][w] != dp[i-1][w]:
                selected.append(i-1)
                w -= weights[i-1]
        
        return list(reversed(selected))
    
    def _diversity_optimization(
        self,
        info_metrics: List[Dict[str, Any]],
        max_tokens: int
    ) -> List[int]:
        """
        Optimize for maximum diversity in selected contexts.
        
        Args:
            info_metrics: Context metrics with embeddings
            max_tokens: Token budget
            
        Returns:
            Indices of selected items
        """
        if not info_metrics:
            return []
        
        selected = []
        remaining = list(range(len(info_metrics)))
        total_tokens = 0
        
        # Start with highest entropy item
        entropies = [m['entropy'] for m in info_metrics]
        first_idx = np.argmax(entropies)
        
        context = info_metrics[first_idx]['context']
        if context.token_count <= max_tokens:
            selected.append(first_idx)
            remaining.remove(first_idx)
            total_tokens += context.token_count
        
        # Greedily add most diverse items
        while remaining and total_tokens < max_tokens:
            max_min_distance = -1
            best_idx = None
            
            for idx in remaining:
                context = info_metrics[idx]['context']
                if total_tokens + context.token_count > max_tokens:
                    continue
                
                # Find minimum distance to selected items
                min_distance = float('inf')
                for sel_idx in selected:
                    distance = 1 - VectorOperations.cosine_similarity(
                        info_metrics[idx]['embedding'],
                        info_metrics[sel_idx]['embedding']
                    )
                    min_distance = min(min_distance, distance)
                
                # Select item with maximum minimum distance (most diverse)
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx
            
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                total_tokens += info_metrics[best_idx]['context'].token_count
            else:
                break
        
        return selected
    
    def _balanced_optimization(
        self,
        info_metrics: List[Dict[str, Any]],
        max_tokens: int,
        mutual_info_matrix: np.ndarray
    ) -> List[int]:
        """
        Balanced optimization combining information value and diversity.
        
        Args:
            info_metrics: Context metrics
            max_tokens: Token budget
            mutual_info_matrix: Mutual information matrix
            
        Returns:
            Indices of selected items
        """
        # Combine knapsack and diversity approaches
        knapsack_indices = self._knapsack_optimization(
            values=[m['density'] for m in info_metrics],
            weights=[m['context'].token_count for m in info_metrics],
            capacity=max_tokens,
            mutual_info_matrix=mutual_info_matrix
        )
        
        diversity_indices = self._diversity_optimization(
            info_metrics, max_tokens
        )
        
        # Merge results, preferring items that appear in both
        all_indices = set(knapsack_indices) | set(diversity_indices)
        scores = {}
        
        for idx in all_indices:
            score = 0
            if idx in knapsack_indices:
                score += 1
            if idx in diversity_indices:
                score += 1
            scores[idx] = score
        
        # Sort by score and token constraints
        sorted_indices = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        selected = []
        total_tokens = 0
        
        for idx in sorted_indices:
            context = info_metrics[idx]['context']
            if total_tokens + context.token_count <= max_tokens:
                selected.append(idx)
                total_tokens += context.token_count
        
        return selected
    
    def _order_by_coherence(self, contexts: List[ContextItem]) -> List[ContextItem]:
        """
        Order contexts for maximum coherence and flow.
        
        Args:
            contexts: Selected context items
            
        Returns:
            Ordered list of contexts
        """
        if len(contexts) <= 1:
            return contexts
        
        # Group by type for logical flow
        type_groups = {}
        for context in contexts:
            if context.context_type not in type_groups:
                type_groups[context.context_type] = []
            type_groups[context.context_type].append(context)
        
        # Order types logically
        type_order = ['documentation', 'api', 'code', 'example', 'tutorial', 'error', 'config']
        
        ordered = []
        for ctx_type in type_order:
            if ctx_type in type_groups:
                # Sort within type by relevance
                type_groups[ctx_type].sort(key=lambda x: x.relevance_score, reverse=True)
                ordered.extend(type_groups[ctx_type])
        
        # Add any remaining types
        for ctx_type, items in type_groups.items():
            if ctx_type not in type_order:
                items.sort(key=lambda x: x.relevance_score, reverse=True)
                ordered.extend(items)
        
        return ordered
    
    def _calculate_diversity_score(self, embeddings: List[List[float]]) -> float:
        """
        Calculate diversity score of selected embeddings.
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(embeddings) <= 1:
            return 1.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                # Use cosine distance (1 - similarity)
                distance = 1 - VectorOperations.cosine_similarity(embeddings[i], embeddings[j])
                distances.append(distance)
        
        if not distances:
            return 1.0
        
        # Average distance as diversity score
        avg_distance = np.mean(distances)
        
        # Normalize to 0-1 range (assuming max distance is 2)
        diversity_score = min(1.0, avg_distance / 2)
        
        return diversity_score
    
    # REMOVED: Duplicate _cosine_similarity - now using VectorOperations.cosine_similarity()
    # See core/math/vector_operations.py for centralized implementation
    
    async def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for text using OpenAI API.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        # Check cache
        text_hash = hash(text)
        if text_hash in self._embedding_cache:
            return self._embedding_cache[text_hash]
        
        # Use centralized embedding manager
        try:
            embedding = await self.embedding_manager.generate_embedding(text[:8000])
            self._embedding_cache[text_hash] = embedding
            return embedding
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Fallback handled by embedding manager
            return await self.embedding_manager.generate_embedding(text[:8000])

# ============= Factory Functions =============

def create_context_optimizer() -> ContextOptimizer:
    """
    Factory function to create context optimizer.
    Uses centralized embedding manager from General Settings.
    
    Returns:
        Initialized context optimizer
    """
    return ContextOptimizer()

async def optimize_prompt_context(
    task: str,
    available_context: List[Dict[str, Any]],
    max_tokens: int = 4000,
    include_examples: bool = True,
    example_count: int = 3
) -> Dict[str, Any]:
    """
    High-level function to optimize prompt context.
    
    Args:
        task: Task description
        available_context: List of context dictionaries
        max_tokens: Token budget
        include_examples: Whether to include few-shot examples
        example_count: Number of examples to include
        
    Returns:
        Optimized prompt components
    """
    optimizer = create_context_optimizer()
    
    # Convert dicts to ContextItems
    context_items = []
    for ctx in available_context:
        item = ContextItem(
            content=ctx.get('content', ''),
            source=ctx.get('source', 'unknown'),
            context_type=ctx.get('type', 'general'),
            relevance_score=ctx.get('relevance', 0.5)
        )
        context_items.append(item)
    
    # Optimize context selection
    optimized = await optimizer.optimize_context(
        context_items,
        max_tokens,
        objective="maximize_information"
    )
    
    return {
        'task': task,
        'selected_contexts': [
            {
                'content': c.content,
                'source': c.source,
                'type': c.context_type,
                'tokens': c.token_count
            }
            for c in optimized.contexts
        ],
        'metrics': {
            'total_tokens': optimized.total_tokens,
            'information_gain': optimized.information_gain,
            'diversity_score': optimized.diversity_score,
            'token_utilization': optimized.optimization_metadata.get('token_utilization', 0)
        }
    }

