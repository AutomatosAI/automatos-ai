"""
Semantic Skill Matching for Agent Selection
Uses embeddings to intelligently match skills instead of dumb string matching

Uses the centralized EmbeddingManager from services/llm_provider.
"""

import asyncio
from typing import List, Dict, Tuple, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class SemanticSkillMatcher:
    """
    Matches skills using semantic similarity instead of substring matching.
    
    Uses the centralized EmbeddingManager which reads from General Settings.
    
    Examples:
    - "writing" matches "writer", "author", "documentation", "technical_writing"
    - "coding" matches "programming", "development", "software_engineering"
    - "analysis" matches "analyst", "analytical", "data_analysis"
    """
    
    def __init__(self, similarity_threshold: float = 0.65):
        """
        Initialize with centralized embedding manager.
        
        Args:
            similarity_threshold: Minimum cosine similarity for a match (0-1)
        """
        self.similarity_threshold = similarity_threshold
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Use centralized embedding manager
        from shared.llm import create_embedding_manager
        self.embedding_manager = create_embedding_manager()
        
        provider_info = self.embedding_manager.get_provider_info()
        logger.info(
            f"SemanticSkillMatcher: Using {provider_info['provider']} embeddings "
            f"(model: {provider_info.get('model', 'N/A')}, "
            f"dimension: {provider_info.get('dimension', 'N/A')})"
        )
        
    async def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using centralized manager (with caching)"""
        # Normalize text
        text = text.lower().strip()
        
        # Check cache
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        # Use embedding manager
        embedding = await self.embedding_manager.generate_embedding(text)
        self.embedding_cache[text] = embedding
        return embedding
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Compute cosine similarity between two vectors"""
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Handle zero vectors
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    async def match_skill(
        self,
        required_skill: str,
        agent_skills: List[str],
        agent_tools: List[str] = None,
        agent_tags: List[str] = None
    ) -> Tuple[bool, float, str, str]:
        """
        Check if required skill matches any agent skill/tool
        
        Args:
            required_skill: The skill being searched for
            agent_skills: List of agent's skills
            agent_tools: List of agent's tools (optional)
            agent_tags: List of agent tags (optional)
            
        Returns:
            (matched, best_similarity, best_match_name, match_origin)
        """
        agent_tools = agent_tools or []
        agent_tags = agent_tags or []
        
        all_agent_capabilities: List[str] = []
        capability_origins: List[str] = []
        
        for capability in agent_skills:
            all_agent_capabilities.append(capability)
            capability_origins.append("skill")
        
        for capability in agent_tools:
            all_agent_capabilities.append(capability)
            capability_origins.append("tool")
        
        for capability in agent_tags:
            all_agent_capabilities.append(capability)
            capability_origins.append("tag")
        
        if not all_agent_capabilities:
            return False, 0.0, "", ""
        
        # Get embeddings
        required_embedding = await self.get_embedding(required_skill)
        agent_embeddings = await asyncio.gather(*[
            self.get_embedding(cap) for cap in all_agent_capabilities
        ])
        
        # Find best match
        best_similarity = 0.0
        best_match = ""
        best_origin = ""
        
        for capability, embedding, origin in zip(all_agent_capabilities, agent_embeddings, capability_origins):
            similarity = self.cosine_similarity(required_embedding, embedding)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = capability
                best_origin = origin
        
        # Check if it passes threshold
        matched = best_similarity >= self.similarity_threshold
        
        logger.debug(
            f"  Skill match: '{required_skill}' → '{best_match}' "
            f"(similarity={best_similarity:.3f}, matched={matched}, origin={best_origin or 'unknown'})"
        )
        
        return matched, best_similarity, best_match, best_origin
    
    async def compute_skill_coverage(
        self,
        required_skills: List[str],
        agent_skills: List[str],
        agent_tools: List[str] = None,
        agent_tags: List[str] = None
    ) -> Tuple[float, List[Dict[str, any]]]:
        """
        Compute semantic skill coverage for an agent
        
        Args:
            required_skills: Skills needed for the task
            agent_skills: Agent's skills
            agent_tools: Agent's tools (optional)
            agent_tags: Agent's lightweight tags (optional)
            
        Returns:
            (coverage_score, match_details)
            - coverage_score: 0.0 to 1.0 (% of required skills matched)
            - match_details: List of match info for each required skill
        """
        if not required_skills:
            return 1.0, []
        
        agent_tools = agent_tools or []
        agent_tags = agent_tags or []
        matches = []
        total_matched = 0
        
        # Check each required skill
        for required_skill in required_skills:
            matched, similarity, best_match, origin = await self.match_skill(
                required_skill,
                agent_skills,
                agent_tools,
                agent_tags
            )
            
            matches.append({
                "required": required_skill,
                "matched": matched,
                "similarity": similarity,
                "agent_capability": best_match,
                "match_type": origin or ("skill" if best_match in agent_skills else "tool")
            })
            
            if matched:
                total_matched += 1
        
        coverage = total_matched / len(required_skills)
        return coverage, matches


# Factory function - uses centralized embedding manager
def get_skill_matcher() -> SemanticSkillMatcher:
    """
    Get or create the global skill matcher.
    Uses centralized embedding manager (no API key needed).
    """
    return SemanticSkillMatcher()
