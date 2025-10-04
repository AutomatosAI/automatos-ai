"""
PRD-05: Memory & Knowledge Systems
===================================

Hierarchical memory system with real Redis TTL, PostgreSQL storage,
pgvector embeddings, and learning capabilities.

NO MOCK DATA - All operations use real services and produce real results. Test
"""

import json
import math
import redis
import asyncio
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import uuid4
from enum import Enum

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import Column, String, Float, Integer, DateTime, JSON, ForeignKey, Text, Boolean, select, func, text
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector

from openai import AsyncOpenAI
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Database models for memory system
Base = declarative_base()


class MemoryLevel(str, Enum):
    WORKING = "working"
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    COLLECTIVE = "collective"


class MemoryType(str, Enum):
    EXPERIENCE = "experience"
    KNOWLEDGE = "knowledge"
    SKILL = "skill"
    PATTERN = "pattern"
    FEEDBACK = "feedback"

class MemoryItem(Base):
    """Enhanced memory items with vector embeddings"""
    __tablename__ = 'memory_items'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    memory_type = Column(String(50), nullable=False)
    memory_level = Column(String(50), nullable=False, default=MemoryLevel.SHORT_TERM)
    importance = Column(Float, default=0.5)
    embedding = Column(Vector(384))  # OpenAI text-embedding-3-small dimension
    access_count = Column(Integer, default=0)
    last_access = Column(DateTime, default=datetime.now)
    decay_rate = Column(Float, default=0.1)
    associations = Column(ARRAY(String), default=[])  # Related memory IDs
    meta_data = Column('metadata', JSON, default={})
    created_at = Column(DateTime, default=datetime.now)
    
    # Performance tracking
    success_rate = Column(Float, default=0.0)
    usage_in_solutions = Column(Integer, default=0)
    average_retrieval_time = Column(Float, default=0.0)

class KnowledgeNode(Base):
    """Knowledge graph nodes"""
    __tablename__ = 'knowledge_nodes'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    agent_id = Column(Integer, nullable=True)  # Null for collective knowledge
    concept = Column(String(255), nullable=False)
    description = Column(Text)
    node_type = Column(String(50), nullable=False)
    embedding = Column(Vector(384))
    importance = Column(Float, default=0.5)
    confidence = Column(Float, default=0.5)
    meta_data = Column('metadata', JSON, default={})
    created_at = Column(DateTime, default=datetime.now)


class KnowledgeEdge(Base):
    """Knowledge graph edges"""
    __tablename__ = 'knowledge_edges'
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid4)
    from_node_id = Column(UUID(as_uuid=True), ForeignKey('knowledge_nodes.id'))
    to_node_id = Column(UUID(as_uuid=True), ForeignKey('knowledge_nodes.id'))
    relationship = Column(String(100), nullable=False)
    strength = Column(Float, default=0.5)
    evidence_count = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.now)


class LearningOutcome(Base):
    """Tracks learning and performance improvements"""
    __tablename__ = 'learning_outcomes'
    
    id = Column(Integer, primary_key=True)
    agent_id = Column(Integer, nullable=False)
    task_type = Column(String(255))
    learned_pattern = Column(Text)
    success_rate_before = Column(Float)
    success_rate_after = Column(Float)
    execution_time_before = Column(Float)
    execution_time_after = Column(Float)
    confidence = Column(Float, default=0.5)
    application_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)


class HierarchicalMemorySystem:
    """
    Complete hierarchical memory system with real services.
    Uses Redis for working memory, PostgreSQL for persistence, and OpenAI for embeddings.
    """
    
    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        redis_password: str = None,
        postgres_url: str = None,
        openai_api_key: str = None
    ):
        # Redis for working memory with TTL
        redis_kwargs = {
            "host": redis_host,
            "port": redis_port,
            "decode_responses": True
        }
        if redis_password:
            redis_kwargs["password"] = redis_password
        
        self.redis_client = redis.Redis(**redis_kwargs)
        
        # PostgreSQL with pgvector for persistent memory
        if postgres_url:
            # Convert postgresql:// to postgresql+asyncpg:// for async support
            if postgres_url.startswith("postgresql://"):
                postgres_url = postgres_url.replace("postgresql://", "postgresql+asyncpg://")
            self.engine = create_async_engine(postgres_url, echo=False)
            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
        else:
            raise ValueError("PostgreSQL URL required for memory system")
        
        # OpenAI for embeddings
        if openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=openai_api_key)
        else:
            raise ValueError("OpenAI API key required for embeddings")
        
        # Memory constraints based on cognitive science
        self.WORKING_MEMORY_CAPACITY = 7  # Miller's Law
        self.WORKING_MEMORY_TTL = 300  # 5 minutes in seconds
        self.SHORT_TERM_CAPACITY = 100
        self.SHORT_TERM_WINDOW = timedelta(hours=24)
        
        logger.info("HierarchicalMemorySystem initialized with real services")
    
    async def initialize_database(self):
        """Create database tables if they don't exist"""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            # Enable pgvector extension
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            logger.info("Database initialized with pgvector support")
    
    async def generate_embedding(self, text: str) -> np.ndarray:
        """Generate real embeddings using OpenAI text-embedding-3-small"""
        try:
            response = await self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=text,
                dimensions=384  # Explicitly request 384 dimensions
            )
            embedding = response.data[0].embedding
            return np.array(embedding, dtype=np.float32)
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            raise
    
    def calculate_importance(self, experience: Dict[str, Any]) -> float:
        """
        Calculate importance score based on multiple factors.
        Higher importance = more likely to be consolidated to long-term memory.
        """
        importance = 0.5  # Base importance
        
        # Factor 1: Success/failure impacts importance
        if experience.get("success"):
            importance += 0.2
        else:
            importance += 0.3  # Failures are important to remember
        
        # Factor 2: Novelty (is this different from existing patterns?)
        if experience.get("is_novel"):
            importance += 0.2
        
        # Factor 3: Emotional salience (errors, breakthroughs)
        if experience.get("error_occurred"):
            importance += 0.15
        
        # Factor 4: Relevance to current goals
        if experience.get("goal_relevant"):
            importance += 0.1
        
        return min(importance, 1.0)
    
    async def store_experience(
        self,
        agent_id: int,
        experience: Dict[str, Any]
    ) -> str:
        """
        Store an experience in the hierarchical memory system.
        Starts in working memory, then moves to short-term based on importance.
        """
        experience_id = str(uuid4())
        importance = self.calculate_importance(experience)
        
        # Step 1: Store in working memory (Redis with TTL)
        working_key = f"working:{agent_id}:{experience_id}"
        redis_data = {
            "id": experience_id,
            "agent_id": agent_id,
            "content": json.dumps(experience),
            "importance": importance,
            "timestamp": datetime.now().isoformat()
        }
        
        # Store with TTL and check capacity
        self.redis_client.setex(
            working_key,
            self.WORKING_MEMORY_TTL,
            json.dumps(redis_data)
        )
        
        # Enforce working memory capacity (remove least important if over limit)
        await self._enforce_working_memory_capacity(agent_id)
        
        # Step 2: If important enough, also store in short-term memory (PostgreSQL)
        if importance > 0.6:
            embedding = await self.generate_embedding(str(experience))
            
            async with self.async_session() as session:
                memory_item = MemoryItem(
                    id=experience_id,
                    agent_id=agent_id,
                    content=json.dumps(experience),
                    memory_type=MemoryType.EXPERIENCE,
                    memory_level=MemoryLevel.SHORT_TERM,
                    importance=importance,
                    embedding=embedding.tolist(),
                    metadata={
                        "task_id": experience.get("task_id"),
                        "success": experience.get("success"),
                        "execution_time": experience.get("execution_time"),
                        "tokens_used": experience.get("tokens_used")
                    }
                )
                session.add(memory_item)
                await session.commit()
                
            logger.info(f"Stored experience {experience_id} with importance {importance:.2f}")
        
        # Step 3: Schedule consolidation for very important items
        if importance > 0.8:
            await self._schedule_consolidation(agent_id, experience_id)
        
        return experience_id
    
    async def _enforce_working_memory_capacity(self, agent_id: int):
        """Enforce Miller's Law - keep only 7 items in working memory"""
        pattern = f"working:{agent_id}:*"
        keys = self.redis_client.keys(pattern)
        
        if len(keys) > self.WORKING_MEMORY_CAPACITY:
            # Get all items with their importance
            items = []
            for key in keys:
                data = self.redis_client.get(key)
                if data:
                    item = json.loads(data)
                    items.append((key, item.get("importance", 0)))
            
            # Sort by importance and keep only top 7
            items.sort(key=lambda x: x[1], reverse=True)
            to_remove = items[self.WORKING_MEMORY_CAPACITY:]
            
            for key, _ in to_remove:
                self.redis_client.delete(key)
                logger.debug(f"Removed {key} from working memory (capacity limit)")
    
    async def retrieve_relevant_memories(
        self,
        agent_id: int,
        context: str,
        memory_types: Optional[List[str]] = None,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the given context using semantic similarity.
        Searches across all memory levels with proper ranking.
        """
        memories = []
        query_embedding = await self.generate_embedding(context)
        
        # Step 1: Check working memory (Redis)
        working_memories = await self._get_working_memories(agent_id)
        for mem in working_memories:
            mem["level"] = "working"
            mem["relevance"] = 1.0  # Working memory is always highly relevant
            memories.append(mem)
        
        # Step 2: Query short-term and long-term memory with vector similarity
        async with self.async_session() as session:
            # Build query with pgvector distance operator
            query = select(MemoryItem).filter(
                MemoryItem.agent_id == agent_id
            )
            
            if memory_types:
                query = query.filter(MemoryItem.memory_type.in_(memory_types))
            
            # Order by cosine similarity (using <=> operator for cosine distance)
            query = query.order_by(
                MemoryItem.embedding.cosine_distance(query_embedding.tolist())
            ).limit(top_k * 2)  # Get more than needed for filtering
            
            result = await session.execute(query)
            db_memories = result.scalars().all()
            
            # Process and apply forgetting curve
            for memory in db_memories:
                relevance = await self._apply_forgetting_curve(memory, query_embedding)
                
                # Update access count and last access
                memory.access_count += 1
                memory.last_access = datetime.now()
                
                memories.append({
                    "id": str(memory.id),
                    "content": json.loads(memory.content) if isinstance(memory.content, str) else memory.content,
                    "level": memory.memory_level,
                    "type": memory.memory_type,
                    "importance": memory.importance,
                    "relevance": relevance,
                    "created_at": memory.created_at,
                    "access_count": memory.access_count
                })
            
            await session.commit()
        
        # Step 3: Rank and filter memories
        memories = self._rank_memories(memories, context)
        
        return memories[:top_k]
    
    async def _get_working_memories(self, agent_id: int) -> List[Dict[str, Any]]:
        """Get all working memories from Redis"""
        pattern = f"working:{agent_id}:*"
        keys = self.redis_client.keys(pattern)
        memories = []
        
        for key in keys:
            data = self.redis_client.get(key)
            if data:
                memory = json.loads(data)
                content = json.loads(memory["content"]) if isinstance(memory["content"], str) else memory["content"]
                memories.append({
                    "id": memory["id"],
                    "content": content,
                    "importance": memory.get("importance", 0.5),
                    "timestamp": memory.get("timestamp")
                })
        
        return memories
    
    async def _apply_forgetting_curve(
        self,
        memory: MemoryItem,
        query_embedding: np.ndarray
    ) -> float:
        """
        Apply Ebbinghaus forgetting curve to calculate memory strength.
        Retention = e^(-time/strength) * similarity
        """
        time_since_access = (datetime.now() - memory.last_access).total_seconds() / 3600  # Hours
        time_since_creation = (datetime.now() - memory.created_at).total_seconds() / 3600
        
        # Memory strength based on importance and access frequency
        strength = memory.importance * (1 + math.log(1 + memory.access_count))
        
        # Apply forgetting curve
        retention = math.exp(-time_since_access / (strength * 24))  # Decay over days
        
        # Calculate similarity
        memory_embedding = np.array(memory.embedding)
        similarity = 1 - np.dot(query_embedding, memory_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(memory_embedding)
        )
        
        # Combine retention and similarity
        relevance = retention * similarity * memory.importance
        
        return relevance
    
    def _rank_memories(self, memories: List[Dict], context: str) -> List[Dict]:
        """Rank memories by combined relevance score"""
        for memory in memories:
            # Combine multiple factors for final ranking
            recency_weight = 0.2
            importance_weight = 0.3
            relevance_weight = 0.5
            
            # Recency score (exponential decay)
            if "created_at" in memory:
                hours_old = (datetime.now() - memory["created_at"]).total_seconds() / 3600
                recency_score = math.exp(-hours_old / 24)  # Decay over days
            else:
                recency_score = 1.0  # Working memory is most recent
            
            memory["final_score"] = (
                recency_weight * recency_score +
                importance_weight * memory.get("importance", 0.5) +
                relevance_weight * memory.get("relevance", 0.5)
            )
        
        return sorted(memories, key=lambda x: x["final_score"], reverse=True)
    
    async def consolidate_memories(self, agent_id: int):
        """
        Consolidate short-term memories into long-term knowledge.
        Identifies patterns and creates abstracted knowledge.
        """
        async with self.async_session() as session:
            # Get recent high-importance short-term memories
            cutoff_time = datetime.now() - self.SHORT_TERM_WINDOW
            query = select(MemoryItem).filter(
                MemoryItem.agent_id == agent_id,
                MemoryItem.memory_level == MemoryLevel.SHORT_TERM,
                MemoryItem.importance > 0.6,
                MemoryItem.created_at > cutoff_time
            )
            
            result = await session.execute(query)
            recent_memories = result.scalars().all()
            
            if not recent_memories:
                return
            
            # Group memories by patterns
            pattern_groups = await self._identify_patterns(recent_memories)
            
            for pattern in pattern_groups:
                # Create abstracted knowledge from pattern
                knowledge = await self._abstract_pattern(pattern)
                
                if knowledge:
                    # Store as long-term memory
                    embedding = await self.generate_embedding(knowledge["description"])
                    
                    long_term_memory = MemoryItem(
                        agent_id=agent_id,
                        content=json.dumps(knowledge),
                        memory_type=MemoryType.KNOWLEDGE,
                        memory_level=MemoryLevel.LONG_TERM,
                        importance=knowledge["importance"],
                        embedding=embedding.tolist(),
                        metadata={
                            "pattern_type": knowledge["pattern_type"],
                            "confidence": knowledge["confidence"],
                            "source_memories": [str(m.id) for m in pattern["memories"]],
                            "consolidation_date": datetime.now().isoformat()
                        }
                    )
                    session.add(long_term_memory)
                    
                    # Add to knowledge graph
                    await self._add_to_knowledge_graph(
                        session,
                        agent_id,
                        knowledge
                    )
                    
                    # Mark source memories as consolidated
                    for memory in pattern["memories"]:
                        memory.memory_level = MemoryLevel.LONG_TERM
                        memory.meta_data["consolidated"] = True
                    
                    logger.info(f"Consolidated {len(pattern['memories'])} memories into knowledge pattern")
            
            await session.commit()
            
            # Prune redundant short-term memories
            await self._prune_memories(session, agent_id)
    
    async def _identify_patterns(
        self,
        memories: List[MemoryItem]
    ) -> List[Dict[str, Any]]:
        """Identify patterns in memories using clustering and similarity"""
        if not memories:
            return []
        
        # Extract embeddings
        embeddings = np.array([m.embedding for m in memories])
        
        # Simple clustering based on similarity threshold
        patterns = []
        used_indices = set()
        similarity_threshold = 0.8
        
        for i, memory in enumerate(memories):
            if i in used_indices:
                continue
            
            pattern = {
                "memories": [memory],
                "indices": [i]
            }
            
            # Find similar memories
            for j, other_memory in enumerate(memories):
                if j <= i or j in used_indices:
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
                )
                
                if similarity > similarity_threshold:
                    pattern["memories"].append(other_memory)
                    pattern["indices"].append(j)
                    used_indices.add(j)
            
            if len(pattern["memories"]) >= 2:  # Only keep patterns with multiple memories
                patterns.append(pattern)
                used_indices.add(i)
        
        return patterns
    
    async def _abstract_pattern(self, pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create abstracted knowledge from a pattern of memories"""
        memories = pattern["memories"]
        
        if not memories:
            return None
        
        # Extract common elements
        contents = [json.loads(m.content) if isinstance(m.content, str) else m.content for m in memories]
        
        # Calculate pattern statistics
        success_count = sum(1 for c in contents if c.get("success"))
        success_rate = success_count / len(contents)
        avg_importance = np.mean([m.importance for m in memories])
        
        # Determine pattern type
        if all(c.get("error_occurred") for c in contents):
            pattern_type = "error_pattern"
        elif success_rate > 0.8:
            pattern_type = "success_pattern"
        else:
            pattern_type = "mixed_pattern"
        
        # Create knowledge description
        description = f"Pattern identified from {len(memories)} experiences. "
        description += f"Success rate: {success_rate:.1%}. "
        
        if pattern_type == "error_pattern":
            description += "Common failure pattern detected."
        elif pattern_type == "success_pattern":
            description += "Successful strategy identified."
        
        return {
            "description": description,
            "pattern_type": pattern_type,
            "importance": min(avg_importance * 1.2, 1.0),  # Patterns are more important
            "confidence": min(len(memories) / 10, 1.0),  # More examples = higher confidence
            "success_rate": success_rate,
            "example_count": len(memories)
        }
    
    async def _add_to_knowledge_graph(
        self,
        session: AsyncSession,
        agent_id: int,
        knowledge: Dict[str, Any]
    ):
        """Add knowledge to the knowledge graph"""
        # Create knowledge node
        embedding = await self.generate_embedding(knowledge["description"])
        
        node = KnowledgeNode(
            agent_id=agent_id,
            concept=knowledge["pattern_type"],
            description=knowledge["description"],
            node_type="pattern",
            embedding=embedding.tolist(),
            importance=knowledge["importance"],
            confidence=knowledge["confidence"],
            metadata=knowledge
        )
        session.add(node)
    
    async def _prune_memories(self, session: AsyncSession, agent_id: int):
        """Remove redundant or low-value memories"""
        # Get short-term memories older than 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        query = select(MemoryItem).filter(
            MemoryItem.agent_id == agent_id,
            MemoryItem.memory_level == MemoryLevel.SHORT_TERM,
            MemoryItem.created_at < cutoff_time,
            MemoryItem.importance < 0.4,  # Only prune low-importance memories
            MemoryItem.access_count < 2  # Rarely accessed
        )
        
        result = await session.execute(query)
        to_prune = result.scalars().all()
        
        for memory in to_prune:
            await session.delete(memory)
        
        if to_prune:
            logger.info(f"Pruned {len(to_prune)} low-value memories")
    
    async def _schedule_consolidation(self, agent_id: int, memory_id: str):
        """Schedule a memory for consolidation"""
        # In production, this would use a task queue like Celery
        # For now, we'll mark it for consolidation
        consolidation_key = f"consolidate:{agent_id}:{memory_id}"
        self.redis_client.setex(consolidation_key, 3600, "pending")  # 1 hour TTL
    
    async def get_memory_stats(self, agent_id: int) -> Dict[str, Any]:
        """Get statistics about an agent's memory system"""
        stats = {
            "working_memory": {
                "count": 0,
                "capacity": self.WORKING_MEMORY_CAPACITY
            },
            "short_term_memory": {
                "count": 0,
                "capacity": self.SHORT_TERM_CAPACITY
            },
            "long_term_memory": {
                "count": 0,
                "total_knowledge": 0
            },
            "performance": {
                "avg_retrieval_time": 0,
                "total_accesses": 0
            }
        }
        
        # Count working memories
        pattern = f"working:{agent_id}:*"
        stats["working_memory"]["count"] = len(self.redis_client.keys(pattern))
        
        # Count database memories
        async with self.async_session() as session:
            # Short-term count
            st_query = select(func.count(MemoryItem.id)).filter(
                MemoryItem.agent_id == agent_id,
                MemoryItem.memory_level == MemoryLevel.SHORT_TERM
            )
            st_result = await session.execute(st_query)
            stats["short_term_memory"]["count"] = st_result.scalar()
            
            # Long-term count
            lt_query = select(func.count(MemoryItem.id)).filter(
                MemoryItem.agent_id == agent_id,
                MemoryItem.memory_level == MemoryLevel.LONG_TERM
            )
            lt_result = await session.execute(lt_query)
            stats["long_term_memory"]["count"] = lt_result.scalar()
            
            # Performance stats
            perf_query = select(
                func.avg(MemoryItem.average_retrieval_time),
                func.sum(MemoryItem.access_count)
            ).filter(MemoryItem.agent_id == agent_id)
            perf_result = await session.execute(perf_query)
            avg_time, total_accesses = perf_result.first()
            
            stats["performance"]["avg_retrieval_time"] = avg_time or 0
            stats["performance"]["total_accesses"] = total_accesses or 0
        
        return stats


class KnowledgeGraph:
    """
    Knowledge graph implementation using PostgreSQL.
    Manages relationships between concepts and enables inference.
    """
    
    def __init__(self, memory_system: HierarchicalMemorySystem):
        self.memory_system = memory_system
        self.session_maker = memory_system.async_session
        logger.info("KnowledgeGraph initialized")
    
    async def add_knowledge(
        self,
        agent_id: Optional[int],
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.5
    ) -> Dict[str, Any]:
        """Add a knowledge triple to the graph"""
        async with self.session_maker() as session:
            # Create or find subject node
            subject_node = await self._get_or_create_node(
                session, agent_id, subject, "entity"
            )
            
            # Create or find object node
            object_node = await self._get_or_create_node(
                session, agent_id, object, "entity"
            )
            
            # Create edge
            edge = KnowledgeEdge(
                from_node_id=subject_node.id,
                to_node_id=object_node.id,
                relationship=predicate,
                strength=confidence
            )
            session.add(edge)
            
            await session.commit()
            
            return {
                "subject": subject,
                "predicate": predicate,
                "object": object,
                "confidence": confidence,
                "edge_id": str(edge.id)
            }
    
    async def _get_or_create_node(
        self,
        session: AsyncSession,
        agent_id: Optional[int],
        concept: str,
        node_type: str
    ) -> KnowledgeNode:
        """Get existing node or create new one"""
        # Check if node exists
        query = select(KnowledgeNode).filter(
            KnowledgeNode.concept == concept,
            KnowledgeNode.agent_id == agent_id
        )
        result = await session.execute(query)
        node = result.scalar_one_or_none()
        
        if not node:
            # Create new node with embedding
            embedding = await self.memory_system.generate_embedding(concept)
            node = KnowledgeNode(
                agent_id=agent_id,
                concept=concept,
                node_type=node_type,
                embedding=embedding.tolist()
            )
            session.add(node)
            await session.flush()
        
        return node
    
    async def query_knowledge(
        self,
        query: str,
        agent_id: Optional[int] = None,
        inference_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Query the knowledge graph with inference.
        Traverses relationships up to inference_depth hops.
        """
        async with self.session_maker() as session:
            # Find relevant starting nodes using embedding similarity
            query_embedding = await self.memory_system.generate_embedding(query)
            
            # Get similar nodes
            node_query = select(KnowledgeNode)
            if agent_id:
                node_query = node_query.filter(KnowledgeNode.agent_id == agent_id)
            
            node_query = node_query.order_by(
                KnowledgeNode.embedding.cosine_distance(query_embedding.tolist())
            ).limit(5)
            
            result = await session.execute(node_query)
            start_nodes = result.scalars().all()
            
            if not start_nodes:
                return {"nodes": [], "edges": [], "insights": []}
            
            # Traverse graph from starting nodes
            visited_nodes = set()
            visited_edges = []
            insights = []
            
            for start_node in start_nodes:
                await self._traverse_graph(
                    session,
                    start_node,
                    inference_depth,
                    visited_nodes,
                    visited_edges,
                    insights
                )
            
            return {
                "query": query,
                "nodes": [
                    {
                        "id": str(node_id),
                        "concept": await self._get_node_concept(session, node_id)
                    }
                    for node_id in visited_nodes
                ],
                "edges": visited_edges,
                "insights": insights[:10]  # Top 10 insights
            }
    
    async def _traverse_graph(
        self,
        session: AsyncSession,
        node: KnowledgeNode,
        depth: int,
        visited_nodes: set,
        visited_edges: list,
        insights: list
    ):
        """Recursive graph traversal for inference"""
        if depth <= 0 or node.id in visited_nodes:
            return
        
        visited_nodes.add(node.id)
        
        # Find edges from this node
        edge_query = select(KnowledgeEdge).filter(
            KnowledgeEdge.from_node_id == node.id
        )
        edge_result = await session.execute(edge_query)
        edges = edge_result.scalars().all()
        
        for edge in edges:
            # Record edge
            visited_edges.append({
                "from": str(edge.from_node_id),
                "to": str(edge.to_node_id),
                "relationship": edge.relationship,
                "strength": edge.strength
            })
            
            # Generate insight
            to_node_query = select(KnowledgeNode).filter(
                KnowledgeNode.id == edge.to_node_id
            )
            to_node_result = await session.execute(to_node_query)
            to_node = to_node_result.scalar_one_or_none()
            
            if to_node:
                insight = {
                    "pattern": f"{node.concept} {edge.relationship} {to_node.concept}",
                    "confidence": edge.strength * node.confidence * to_node.confidence,
                    "evidence_count": edge.evidence_count
                }
                insights.append(insight)
                
                # Recurse
                await self._traverse_graph(
                    session,
                    to_node,
                    depth - 1,
                    visited_nodes,
                    visited_edges,
                    insights
                )
    
    async def _get_node_concept(self, session: AsyncSession, node_id: UUID) -> str:
        """Get concept for a node ID"""
        query = select(KnowledgeNode.concept).filter(KnowledgeNode.id == node_id)
        result = await session.execute(query)
        return result.scalar_one_or_none() or "Unknown"


class LearningEngine:
    """
    Enables agents to learn from experience and improve performance.
    Tracks real metrics and enables knowledge transfer.
    """
    
    def __init__(self, memory_system: HierarchicalMemorySystem):
        self.memory_system = memory_system
        self.session_maker = memory_system.async_session
        logger.info("LearningEngine initialized")
    
    async def learn_from_feedback(
        self,
        agent_id: int,
        task_type: str,
        result: Dict[str, Any],
        feedback: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Learn from task execution feedback.
        Updates success patterns and adjusts strategies.
        """
        async with self.session_maker() as session:
            # Get baseline performance for this task type
            baseline = await self._get_baseline_performance(session, agent_id, task_type)
            
            # Calculate current performance
            current_success = feedback.get("success", False)
            current_time = result.get("execution_time", 0)
            
            # Analyze what worked or didn't
            learned_pattern = await self._analyze_execution(result, feedback)
            
            # Store learning outcome
            outcome = LearningOutcome(
                agent_id=agent_id,
                task_type=task_type,
                learned_pattern=learned_pattern,
                success_rate_before=baseline["success_rate"],
                success_rate_after=baseline["updated_success_rate"],
                execution_time_before=baseline["avg_time"],
                execution_time_after=current_time,
                confidence=min(baseline["sample_size"] / 10, 1.0)
            )
            session.add(outcome)
            
            # Store as memory for future retrieval
            experience = {
                "task_type": task_type,
                "result": result,
                "feedback": feedback,
                "learned": learned_pattern,
                "improvement": baseline["updated_success_rate"] - baseline["success_rate"]
            }
            
            memory_id = await self.memory_system.store_experience(agent_id, experience)
            
            await session.commit()
            
            return {
                "learned_pattern": learned_pattern,
                "performance_improvement": baseline["updated_success_rate"] - baseline["success_rate"],
                "time_improvement": baseline["avg_time"] - current_time,
                "confidence": outcome.confidence,
                "memory_id": memory_id
            }
    
    async def _get_baseline_performance(
        self,
        session: AsyncSession,
        agent_id: int,
        task_type: str
    ) -> Dict[str, Any]:
        """Get baseline performance metrics for a task type"""
        # Query previous outcomes for this task type
        query = select(LearningOutcome).filter(
            LearningOutcome.agent_id == agent_id,
            LearningOutcome.task_type == task_type
        ).order_by(LearningOutcome.created_at.desc()).limit(10)
        
        result = await session.execute(query)
        outcomes = result.scalars().all()
        
        if not outcomes:
            return {
                "success_rate": 0.5,
                "updated_success_rate": 0.5,
                "avg_time": 0,
                "sample_size": 0
            }
        
        # Calculate metrics
        success_rates = [o.success_rate_after for o in outcomes if o.success_rate_after is not None]
        exec_times = [o.execution_time_after for o in outcomes if o.execution_time_after is not None]
        
        baseline = {
            "success_rate": np.mean(success_rates) if success_rates else 0.5,
            "updated_success_rate": np.mean(success_rates) if success_rates else 0.5,
            "avg_time": np.mean(exec_times) if exec_times else 0,
            "sample_size": len(outcomes)
        }
        
        return baseline
    
    async def _analyze_execution(
        self,
        result: Dict[str, Any],
        feedback: Dict[str, Any]
    ) -> str:
        """Analyze execution to identify what was learned"""
        patterns = []
        
        if feedback.get("success"):
            patterns.append("Successful approach identified")
            if result.get("execution_time", 0) < feedback.get("expected_time", float('inf')):
                patterns.append("Efficient execution path found")
        else:
            patterns.append("Failure pattern detected")
            if feedback.get("error"):
                patterns.append(f"Error type: {feedback['error'][:100]}")
        
        if result.get("tokens_used", 0) > 0:
            efficiency = result.get("tokens_used", 0) / max(len(str(result.get("output", ""))), 1)
            if efficiency < 10:
                patterns.append("Token-efficient approach")
        
        return "; ".join(patterns) if patterns else "No clear pattern identified"
    
    async def transfer_learning(
        self,
        from_agent_id: int,
        to_agent_id: int,
        knowledge_domain: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Transfer learned knowledge from one agent to another.
        Includes success patterns, strategies, and domain knowledge.
        """
        transferred = {
            "memories": 0,
            "patterns": 0,
            "strategies": 0,
            "success": True
        }
        
        async with self.session_maker() as session:
            # Step 1: Transfer successful patterns
            pattern_query = select(MemoryItem).filter(
                MemoryItem.agent_id == from_agent_id,
                MemoryItem.memory_type == MemoryType.PATTERN,
                MemoryItem.memory_level == MemoryLevel.LONG_TERM,
                MemoryItem.importance > 0.7
            )
            
            if knowledge_domain:
                pattern_query = pattern_query.filter(
                    MemoryItem.meta_data["pattern_type"].astext == knowledge_domain
                )
            
            result = await session.execute(pattern_query)
            patterns = result.scalars().all()
            
            for pattern in patterns:
                # Create copy for target agent
                new_pattern = MemoryItem(
                    agent_id=to_agent_id,
                    content=pattern.content,
                    memory_type=pattern.memory_type,
                    memory_level=MemoryLevel.LONG_TERM,
                    importance=pattern.importance * 0.9,  # Slightly lower for transferred knowledge
                    embedding=pattern.embedding,
                    metadata={
                        **pattern.meta_data,
                        "transferred_from": from_agent_id,
                        "transfer_date": datetime.now().isoformat()
                    }
                )
                session.add(new_pattern)
                transferred["patterns"] += 1
            
            # Step 2: Transfer high-value memories
            memory_query = select(MemoryItem).filter(
                MemoryItem.agent_id == from_agent_id,
                MemoryItem.memory_level == MemoryLevel.LONG_TERM,
                MemoryItem.success_rate > 0.8
            ).limit(50)
            
            result = await session.execute(memory_query)
            memories = result.scalars().all()
            
            for memory in memories:
                new_memory = MemoryItem(
                    agent_id=to_agent_id,
                    content=memory.content,
                    memory_type=memory.memory_type,
                    memory_level=MemoryLevel.LONG_TERM,
                    importance=memory.importance * 0.85,
                    embedding=memory.embedding,
                    metadata={
                        **memory.meta_data,
                        "transferred_from": from_agent_id
                    }
                )
                session.add(new_memory)
                transferred["memories"] += 1
            
            # Step 3: Transfer learning outcomes
            outcome_query = select(LearningOutcome).filter(
                LearningOutcome.agent_id == from_agent_id,
                LearningOutcome.confidence > 0.7
            )
            
            result = await session.execute(outcome_query)
            outcomes = result.scalars().all()
            
            for outcome in outcomes:
                new_outcome = LearningOutcome(
                    agent_id=to_agent_id,
                    task_type=outcome.task_type,
                    learned_pattern=outcome.learned_pattern,
                    success_rate_before=0.5,  # Reset baseline for new agent
                    success_rate_after=outcome.success_rate_after * 0.9,
                    execution_time_before=0,
                    execution_time_after=outcome.execution_time_after,
                    confidence=outcome.confidence * 0.8
                )
                session.add(new_outcome)
                transferred["strategies"] += 1
            
            await session.commit()
            
            logger.info(f"Transferred knowledge from agent {from_agent_id} to {to_agent_id}: "
                       f"{transferred['patterns']} patterns, {transferred['memories']} memories, "
                       f"{transferred['strategies']} strategies")
        
        return transferred
    
    async def get_performance_metrics(
        self,
        agent_id: int,
        task_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get detailed performance metrics showing learning progress"""
        async with self.session_maker() as session:
            query = select(LearningOutcome).filter(
                LearningOutcome.agent_id == agent_id
            )
            
            if task_type:
                query = query.filter(LearningOutcome.task_type == task_type)
            
            query = query.order_by(LearningOutcome.created_at)
            
            result = await session.execute(query)
            outcomes = result.scalars().all()
            
            if not outcomes:
                return {
                    "agent_id": agent_id,
                    "task_type": task_type,
                    "learning_curve": [],
                    "improvement": 0,
                    "current_success_rate": 0.5,
                    "average_success_rate": 0.0,
                    "average_execution_time": 0.0,
                    "total_learning_events": 0,
                    "confidence": 0.0
                }
            
            # Build learning curve
            learning_curve = []
            for outcome in outcomes:
                learning_curve.append({
                    "timestamp": outcome.created_at.isoformat(),
                    "success_rate": outcome.success_rate_after,
                    "execution_time": outcome.execution_time_after,
                    "confidence": outcome.confidence
                })
            
            # Calculate overall improvement
            first_rate = outcomes[0].success_rate_before or 0.5
            current_rate = outcomes[-1].success_rate_after or 0.5
            improvement = current_rate - first_rate
            
            # Calculate average metrics
            recent_outcomes = outcomes[-10:]  # Last 10 outcomes
            avg_success = np.mean([o.success_rate_after for o in recent_outcomes if o.success_rate_after])
            avg_time = np.mean([o.execution_time_after for o in recent_outcomes if o.execution_time_after])
            
            return {
                "agent_id": agent_id,
                "task_type": task_type,
                "learning_curve": learning_curve,
                "improvement": improvement,
                "current_success_rate": current_rate,
                "average_success_rate": avg_success,
                "average_execution_time": avg_time,
                "total_learning_events": len(outcomes),
                "confidence": outcomes[-1].confidence if outcomes else 0
            }
    
    async def transfer_learning(
        self,
        from_agent_id: int,
        to_agent_id: int,
        knowledge_domain: str = None
    ) -> Dict[str, Any]:
        """
        Transfer learning outcomes and knowledge from one agent to another.
        
        Args:
            from_agent_id: Source agent ID
            to_agent_id: Target agent ID
            knowledge_domain: Optional domain filter
            
        Returns:
            Dictionary with transfer results
        """
        async with self.session_maker() as session:
            transferred = {
                "patterns": 0,
                "memories": 0,
                "strategies": 0
            }
            
            # Step 1: Transfer learning outcomes
            query = select(LearningOutcome).filter(
                LearningOutcome.agent_id == from_agent_id
            )
            
            if knowledge_domain:
                query = query.filter(LearningOutcome.task_type == knowledge_domain)
            
            query = query.filter(LearningOutcome.confidence > 0.7)
            
            result = await session.execute(query)
            outcomes = result.scalars().all()
            
            for outcome in outcomes:
                new_outcome = LearningOutcome(
                    agent_id=to_agent_id,
                    task_type=outcome.task_type,
                    learned_pattern=outcome.learned_pattern,
                    success_rate_before=0.5,  # Reset baseline for new agent
                    success_rate_after=outcome.success_rate_after * 0.9,
                    execution_time_before=0,
                    execution_time_after=outcome.execution_time_after,
                    confidence=outcome.confidence * 0.8
                )
                session.add(new_outcome)
                transferred["strategies"] += 1
            
            await session.commit()
            
            logger.info(f"Transferred knowledge from agent {from_agent_id} to {to_agent_id}: "
                       f"{transferred['strategies']} strategies")
        
        return transferred


# Integration with existing systems
async def integrate_with_agent_factory(
    memory_system: HierarchicalMemorySystem,
    agent_factory,
    shared_context_manager
):
    """
    Hook memory system into existing AgentFactory and SharedContextManager.
    This function patches the existing systems to use the new memory system.
    """
    
    # Store original execute_with_prompt
    original_execute = agent_factory.execute_with_prompt
    
    async def enhanced_execute_with_prompt(agent, prompt, **kwargs):
        """Enhanced execution that stores experiences in memory"""
        agent_id = agent.id if hasattr(agent, 'id') else agent
        
        # Retrieve relevant memories for context
        memories = await memory_system.retrieve_relevant_memories(
            agent_id,
            prompt,
            top_k=5
        )
        
        # Add memories to context
        if memories:
            memory_context = "\n\nRelevant memories:\n"
            for mem in memories:
                memory_context += f"- {mem['content']}\n"
            enhanced_prompt = prompt + memory_context
        else:
            enhanced_prompt = prompt
        
        # Execute with enhanced prompt
        start_time = datetime.now()
        result = await original_execute(agent, enhanced_prompt, **kwargs)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Store experience in memory
        experience = {
            "prompt": prompt,
            "result": result,
            "execution_time": execution_time,
            "success": result.get("success", True) if isinstance(result, dict) else True,
            "tokens_used": result.get("tokens_used", 0) if isinstance(result, dict) else 0,
            "timestamp": datetime.now().isoformat()
        }
        
        await memory_system.store_experience(agent_id, experience)
        
        return result
    
    # Patch the agent factory
    agent_factory.execute_with_prompt = enhanced_execute_with_prompt
    
    # Also integrate with shared context manager
    if shared_context_manager:
        original_store_context = shared_context_manager.store_context
        
        async def enhanced_store_context(team_id, context_data):
            """Enhanced context storage that also updates collective memory"""
            # Store in original system
            result = await original_store_context(team_id, context_data)
            
            # Store in collective memory
            for agent_id in context_data.get("agent_ids", []):
                experience = {
                    "team_id": team_id,
                    "shared_context": context_data,
                    "is_collaborative": True
                }
                await memory_system.store_experience(agent_id, experience)
            
            return result
        
        shared_context_manager.store_context = enhanced_store_context
    
    logger.info("Memory system integrated with AgentFactory and SharedContextManager")
    return True

