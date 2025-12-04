"""
Context Level Decision Module
=============================

Implements the "Five Levels of Context" decision framework.
Both chatbot and workflow orchestrator use this for consistent context decisions.

Five Levels:
1. ATOMIC: Simple instructions, no context needed
2. MOLECULAR: Instructions + examples/patterns
3. CELLULAR: Memory-augmented context
4. ORGANIC: Multi-agent coordinated context
5. ORGANISMIC: Full workflow orchestration
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class ContextLevel(Enum):
    """Five Levels of Context Complexity"""
    ATOMIC = 1      # Simple instructions only
    MOLECULAR = 2   # + Examples and patterns
    CELLULAR = 3    # + Memory and history
    ORGANIC = 4     # + Multi-agent coordination
    ORGANISMIC = 5  # + Full workflow orchestration


@dataclass
class ContextDecision:
    """Decision result from context level analysis"""
    level: ContextLevel
    confidence: float
    reasoning: str
    recommended_tools: List[str]
    context_sources: List[str]
    estimated_tokens: int
    requires_workflow: bool


class ContextLevelDecision:
    """
    Central decision engine for context complexity analysis.
    Used by both chatbot and workflow orchestrator.
    """

    def __init__(self):
        # Complexity indicators for each level
        self.level_indicators = {
            ContextLevel.ATOMIC: {
                'max_words': 50,
                'max_complexity_score': 2,
                'keywords': ['what', 'how', 'simple', 'basic', 'quick']
            },
            ContextLevel.MOLECULAR: {
                'max_words': 200,
                'max_complexity_score': 5,
                'keywords': ['example', 'like', 'similar', 'pattern', 'template']
            },
            ContextLevel.CELLULAR: {
                'max_words': 500,
                'max_complexity_score': 8,
                'keywords': ['remember', 'previously', 'before', 'history', 'experience']
            },
            ContextLevel.ORGANIC: {
                'max_words': 1000,
                'max_complexity_score': 12,
                'keywords': ['coordinate', 'multiple', 'together', 'collaborate', 'team']
            },
            ContextLevel.ORGANISMIC: {
                'max_words': float('inf'),
                'max_complexity_score': float('inf'),
                'keywords': ['workflow', 'process', 'automate', 'complex', 'project']
            }
        }

        # Tool recommendations by level
        self.level_tools = {
            ContextLevel.ATOMIC: ['search_knowledge'],
            ContextLevel.MOLECULAR: ['search_knowledge', 'semantic_search'],
            ContextLevel.CELLULAR: ['search_knowledge', 'semantic_search', 'query_database'],
            ContextLevel.ORGANIC: ['search_knowledge', 'semantic_search', 'query_database', 'search_codebase'],
            ContextLevel.ORGANISMIC: ['search_knowledge', 'semantic_search', 'query_database', 'search_codebase', 'search_multimodal']
        }

        # Context sources by level
        self.level_context_sources = {
            ContextLevel.ATOMIC: ['instructions'],
            ContextLevel.MOLECULAR: ['instructions', 'examples', 'patterns'],
            ContextLevel.CELLULAR: ['instructions', 'examples', 'patterns', 'memory'],
            ContextLevel.ORGANIC: ['instructions', 'examples', 'patterns', 'memory', 'inter_agent'],
            ContextLevel.ORGANISMIC: ['instructions', 'examples', 'patterns', 'memory', 'inter_agent', 'workflow_history']
        }

    def analyze_prompt(self, prompt: str, metadata: Optional[Dict[str, Any]] = None) -> ContextDecision:
        """
        Analyze a prompt and determine appropriate context level.

        Args:
            prompt: The user prompt/query
            metadata: Additional context (urgency, domain, user_history, etc.)

        Returns:
            ContextDecision with level, confidence, and recommendations
        """
        metadata = metadata or {}

        # Calculate complexity metrics
        complexity_score = self._calculate_complexity_score(prompt, metadata)
        word_count = len(prompt.split())

        # Safety/criticality analysis
        safety_score = self._analyze_safety_criticality(prompt)
        domain_complexity = self._analyze_domain_complexity(prompt, metadata)

        # Historical patterns (if available)
        historical_level = metadata.get('historical_context_level', 3)

        # Combine factors for final decision
        final_level, confidence, reasoning = self._make_decision(
            complexity_score, word_count, safety_score, domain_complexity, historical_level
        )

        # Get recommendations for this level
        recommended_tools = self.level_tools[final_level]
        context_sources = self.level_context_sources[final_level]
        estimated_tokens = self._estimate_token_usage(final_level, word_count)
        requires_workflow = final_level.value >= 4  # ORGANIC and above

        return ContextDecision(
            level=final_level,
            confidence=confidence,
            reasoning=reasoning,
            recommended_tools=recommended_tools,
            context_sources=context_sources,
            estimated_tokens=estimated_tokens,
            requires_workflow=requires_workflow
        )

    def _calculate_complexity_score(self, prompt: str, metadata: Dict[str, Any]) -> float:
        """Calculate complexity score based on multiple factors"""
        score = 0.0

        # Length-based complexity
        word_count = len(prompt.split())
        if word_count > 100:
            score += 3
        elif word_count > 50:
            score += 2
        elif word_count > 20:
            score += 1

        # Question complexity
        question_words = ['how', 'why', 'what', 'when', 'where', 'which', 'explain']
        questions = sum(1 for word in prompt.lower().split() if word in question_words)
        score += questions * 0.5

        # Technical indicators
        technical_terms = [
            'implement', 'architecture', 'algorithm', 'optimization', 'integration',
            'deployment', 'configuration', 'database', 'api', 'security', 'performance'
        ]
        tech_count = sum(1 for term in technical_terms if term in prompt.lower())
        score += tech_count * 0.8

        # Coordination indicators
        coordination_terms = [
            'coordinate', 'multiple', 'together', 'collaborate', 'workflow',
            'process', 'automate', 'schedule', 'batch', 'parallel'
        ]
        coord_count = sum(1 for term in coordination_terms if term in prompt.lower())
        score += coord_count * 1.5

        # Metadata factors
        if metadata.get('urgency') == 'high':
            score += 1
        if metadata.get('domain') in ['security', 'infrastructure', 'finance']:
            score += 1
        if metadata.get('requires_research', False):
            score += 1.5

        return score

    def _analyze_safety_criticality(self, prompt: str) -> float:
        """Analyze if prompt involves safety-critical operations"""
        safety_keywords = [
            'security', 'vulnerability', 'exploit', 'hack', 'breach', 'confidential',
            'sensitive', 'critical', 'production', 'live', 'emergency'
        ]

        safety_score = 0.0
        for keyword in safety_keywords:
            if keyword in prompt.lower():
                safety_score += 1.0

        return min(safety_score, 3.0)  # Cap at 3

    def _analyze_domain_complexity(self, prompt: str, metadata: Dict[str, Any]) -> float:
        """Analyze domain-specific complexity"""
        complexity = 0.0

        # Programming domains
        if any(term in prompt.lower() for term in ['code', 'programming', 'software', 'development']):
            complexity += 1.5

        # Data/analysis domains
        if any(term in prompt.lower() for term in ['data', 'analytics', 'database', 'query']):
            complexity += 1.2

        # System/infrastructure domains
        if any(term in prompt.lower() for term in ['infrastructure', 'deployment', 'monitoring']):
            complexity += 1.3

        # Multi-domain coordination
        domain_indicators = ['frontend', 'backend', 'database', 'api', 'ui', 'ux', 'testing']
        domains = sum(1 for term in domain_indicators if term in prompt.lower())
        if domains >= 2:
            complexity += 1.0

        return complexity

    def _make_decision(
        self,
        complexity_score: float,
        word_count: int,
        safety_score: float,
        domain_complexity: float,
        historical_level: int
    ) -> Tuple[ContextLevel, float, str]:
        """Make final context level decision"""

        # Base level from complexity
        if complexity_score <= 2 and word_count <= 50:
            base_level = ContextLevel.ATOMIC
        elif complexity_score <= 5 and word_count <= 200:
            base_level = ContextLevel.MOLECULAR
        elif complexity_score <= 8 and word_count <= 500:
            base_level = ContextLevel.CELLULAR
        elif complexity_score <= 12 or word_count <= 1000:
            base_level = ContextLevel.ORGANIC
        else:
            base_level = ContextLevel.ORGANISMIC

        # Adjust for safety criticality
        if safety_score >= 2:
            base_level = max(base_level, ContextLevel.CELLULAR)

        # Adjust for domain complexity
        if domain_complexity >= 2.5:
            base_level = max(base_level, ContextLevel.ORGANIC)

        # Consider historical patterns (slight bias toward consistency)
        historical_enum = ContextLevel(min(max(historical_level, 1), 5))
        if abs(base_level.value - historical_enum.value) <= 1:
            # Keep consistent with history
            pass
        elif historical_enum.value > base_level.value:
            # History suggests higher complexity
            base_level = historical_enum

        # Calculate confidence based on factors
        confidence_factors = [
            1.0 if complexity_score < 10 else 0.9,  # Clear complexity
            1.0 if word_count < 1000 else 0.8,     # Reasonable length
            0.9 if safety_score < 2 else 1.0,      # Safety considerations
            0.9 if domain_complexity < 3 else 1.0  # Domain expertise
        ]
        confidence = sum(confidence_factors) / len(confidence_factors)

        # Generate reasoning
        reasoning_parts = []
        if complexity_score > 5:
            reasoning_parts.append(f"High complexity score ({complexity_score:.1f})")
        if safety_score > 1:
            reasoning_parts.append(f"Safety-critical content (score: {safety_score:.1f})")
        if domain_complexity > 1.5:
            reasoning_parts.append(f"Domain complexity (score: {domain_complexity:.1f})")
        if word_count > 200:
            reasoning_parts.append(f"Long prompt ({word_count} words)")

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "Standard complexity analysis"

        return base_level, confidence, reasoning

    def _estimate_token_usage(self, level: ContextLevel, prompt_words: int) -> int:
        """Estimate token usage for this context level"""
        base_tokens = prompt_words * 1.3  # Rough word to token conversion

        level_multipliers = {
            ContextLevel.ATOMIC: 1.0,
            ContextLevel.MOLECULAR: 1.5,
            ContextLevel.CELLULAR: 2.5,
            ContextLevel.ORGANIC: 4.0,
            ContextLevel.ORGANISMIC: 6.0
        }

        return int(base_tokens * level_multipliers[level])

    def get_level_requirements(self, level: ContextLevel) -> Dict[str, Any]:
        """Get requirements for a specific context level"""
        return {
            'indicators': self.level_indicators[level],
            'tools': self.level_tools[level],
            'context_sources': self.level_context_sources[level]
        }


# Global instance for shared access
_context_decision_engine = None

def get_context_decision_engine() -> ContextLevelDecision:
    """Get or create the global context decision engine instance"""
    global _context_decision_engine
    if _context_decision_engine is None:
        _context_decision_engine = ContextLevelDecision()
        logger.info("✅ Context Level Decision Engine initialized")
    return _context_decision_engine
