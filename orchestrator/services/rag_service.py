"""
RAG (Retrieval-Augmented Generation) Service
=============================================

Provides vector-based context retrieval for enhanced agent prompts.
Uses in-memory vector store for simplicity, can be upgraded to Qdrant/Weaviate later.

Features:
- Document chunking and embedding
- Semantic search with similarity scoring
- Context caching for performance
- Multiple retrieval strategies
"""

import logging
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import hashlib
# from sklearn.metrics.pairwise import cosine_similarity  # Removed sklearn dependency
import openai
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Simple cosine similarity implementation to avoid sklearn dependency.
    Works for 2D arrays where each row is a vector.
    """
    # Handle 1D to 2D conversion
    if len(a.shape) == 1:
        a = a.reshape(1, -1)
    if len(b.shape) == 1:
        b = b.reshape(1, -1)
    
    # Normalize vectors
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    
    # Compute cosine similarity
    return np.dot(a_norm, b_norm.T)


@dataclass
class Document:
    """Represents a document in the vector store"""
    id: str
    content: str
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    source: str = "unknown"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SearchResult:
    """Represents a search result from the vector store"""
    document: Document
    score: float
    relevance: str  # 'high', 'medium', 'low'


class VectorStore:
    """Simple in-memory vector store with persistence"""
    
    def __init__(self, persist_path: str = "/tmp/automatos_vectors.pkl"):
        self.documents: Dict[str, Document] = {}
        self.embeddings: Dict[str, np.ndarray] = {}
        self.persist_path = persist_path
        self.load_from_disk()
    
    def add_document(self, doc: Document):
        """Add a document to the vector store"""
        self.documents[doc.id] = doc
        if doc.embedding is not None:
            self.embeddings[doc.id] = doc.embedding
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, float]]:
        """Search for similar documents using cosine similarity"""
        if not self.embeddings:
            return []
        
        # Convert embeddings to matrix
        doc_ids = list(self.embeddings.keys())
        doc_embeddings = np.array([self.embeddings[doc_id] for doc_id in doc_ids])
        
        # Calculate similarities
        similarities = cosine_similarity(query_embedding, doc_embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [(doc_ids[i], float(similarities[i])) for i in top_indices]
        
        return results
    
    def save_to_disk(self):
        """Persist vector store to disk"""
        try:
            with open(self.persist_path, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings
                }, f)
            logger.info(f"Saved {len(self.documents)} documents to {self.persist_path}")
        except Exception as e:
            logger.error(f"Failed to save vector store: {e}")
    
    def load_from_disk(self):
        """Load vector store from disk if exists"""
        if os.path.exists(self.persist_path):
            try:
                with open(self.persist_path, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', {})
                    self.embeddings = data.get('embeddings', {})
                logger.info(f"Loaded {len(self.documents)} documents from {self.persist_path}")
            except Exception as e:
                logger.error(f"Failed to load vector store: {e}")


class RAGService:
    """
    Main RAG service for context retrieval and enhancement
    """
    
    def __init__(self, openai_api_key: Optional[str] = None):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if self.api_key:
            openai.api_key = self.api_key
        else:
            logger.warning("No OpenAI API key found - embeddings will be simulated")
        
        self.vector_store = VectorStore()
        self.cache = {}  # Simple query cache
        self.cache_ttl = 300  # 5 minutes
        
        # Pre-load some domain knowledge
        self._load_default_knowledge()
    
    def _load_default_knowledge(self):
        """Load default knowledge base for common tasks"""
        default_docs = [
            {
                "content": "When performing code review, check for: security vulnerabilities, performance issues, code style compliance, proper error handling, test coverage, and documentation completeness.",
                "metadata": {"category": "code_review", "type": "guideline"}
            },
            {
                "content": "For API design, follow RESTful principles: use proper HTTP methods, implement versioning, provide clear error messages, use consistent naming conventions, implement pagination for lists, and include proper authentication.",
                "metadata": {"category": "api_design", "type": "best_practice"}
            },
            {
                "content": "Python best practices: Use type hints, follow PEP 8, write docstrings, handle exceptions properly, use context managers for resources, prefer list comprehensions over loops when appropriate.",
                "metadata": {"category": "python", "type": "best_practice"}
            },
            {
                "content": "Security considerations: Validate all inputs, use parameterized queries to prevent SQL injection, implement rate limiting, use HTTPS everywhere, store passwords using bcrypt or similar, implement CORS properly.",
                "metadata": {"category": "security", "type": "checklist"}
            },
            {
                "content": "Performance optimization strategies: Profile before optimizing, cache expensive operations, use database indexes, implement pagination, optimize images and assets, use CDN for static files, minimize API calls.",
                "metadata": {"category": "performance", "type": "strategy"}
            },
            {
                "content": "Testing best practices: Write unit tests first (TDD), aim for 80%+ coverage, test edge cases, use mocking for external dependencies, write integration tests for critical paths, automate testing in CI/CD.",
                "metadata": {"category": "testing", "type": "best_practice"}
            },
            {
                "content": "Documentation standards: Write clear README files, document API endpoints with examples, include installation instructions, provide usage examples, document environment variables, maintain a changelog.",
                "metadata": {"category": "documentation", "type": "standard"}
            },
            {
                "content": "Git workflow: Use feature branches, write descriptive commit messages, squash commits before merging, use pull requests for code review, tag releases, maintain a clean commit history.",
                "metadata": {"category": "git", "type": "workflow"}
            },
            {
                "content": "Docker best practices: Use multi-stage builds, minimize layers, use specific base image tags, don't run as root, use .dockerignore, cache dependencies separately, health checks are important.",
                "metadata": {"category": "docker", "type": "best_practice"}
            },
            {
                "content": "Microservices patterns: Use service discovery, implement circuit breakers, centralized logging, distributed tracing, API gateway pattern, event-driven architecture where appropriate, handle partial failures gracefully.",
                "metadata": {"category": "microservices", "type": "pattern"}
            }
        ]
        
        for i, doc_data in enumerate(default_docs):
            doc = Document(
                id=f"default_{i}",
                content=doc_data["content"],
                metadata=doc_data["metadata"],
                source="default_knowledge"
            )
            doc.embedding = self._generate_embedding(doc.content)
            self.vector_store.add_document(doc)
        
        self.vector_store.save_to_disk()
        logger.info(f"Loaded {len(default_docs)} default knowledge documents")
    
    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using OpenAI or fallback to simple method"""
        if self.api_key:
            try:
                import openai
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return np.array(response['data'][0]['embedding'])
            except Exception as e:
                logger.warning(f"Failed to generate OpenAI embedding: {e}, using fallback")
        
        # Fallback: Simple hash-based pseudo-embedding
        # This is just for testing - real embeddings are much better
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        # Convert hash to 384-dimensional vector (simplified)
        embedding = []
        for i in range(0, len(text_hash), 2):
            val = int(text_hash[i:i+2], 16) / 255.0
            embedding.extend([val] * 12)  # Expand to match ada-002 dimension (1536/128)
        
        # Pad or truncate to 1536 dimensions
        embedding = embedding[:1536] if len(embedding) > 1536 else embedding + [0] * (1536 - len(embedding))
        return np.array(embedding)
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None, source: str = "user") -> str:
        """Add a new document to the RAG system"""
        doc_id = hashlib.md5(content.encode()).hexdigest()
        
        doc = Document(
            id=doc_id,
            content=content,
            metadata=metadata or {},
            source=source
        )
        
        doc.embedding = self._generate_embedding(content)
        self.vector_store.add_document(doc)
        self.vector_store.save_to_disk()
        
        logger.info(f"Added document {doc_id} to RAG system")
        return doc_id
    
    def retrieve_context(
        self, 
        query: str, 
        top_k: int = 5,
        min_similarity: float = 0.5,
        category_filter: Optional[str] = None
    ) -> List[SearchResult]:
        """
        Retrieve relevant context for a query
        
        Args:
            query: The search query
            top_k: Number of results to return
            min_similarity: Minimum similarity threshold
            category_filter: Optional category to filter results
        
        Returns:
            List of SearchResult objects
        """
        
        # Check cache
        cache_key = f"{query}_{top_k}_{min_similarity}_{category_filter}"
        if cache_key in self.cache:
            cached_time, cached_results = self.cache[cache_key]
            if datetime.now() - cached_time < timedelta(seconds=self.cache_ttl):
                logger.info(f"Using cached results for query: {query[:50]}")
                return cached_results
        
        # Generate query embedding
        query_embedding = self._generate_embedding(query)
        
        # Search vector store
        search_results = self.vector_store.search(query_embedding, top_k * 2)  # Get extra for filtering
        
        # Convert to SearchResult objects
        results = []
        for doc_id, score in search_results:
            if score < min_similarity:
                continue
            
            doc = self.vector_store.documents.get(doc_id)
            if not doc:
                continue
            
            # Apply category filter if specified
            if category_filter and doc.metadata.get('category') != category_filter:
                continue
            
            # Determine relevance level
            relevance = 'high' if score > 0.8 else 'medium' if score > 0.6 else 'low'
            
            results.append(SearchResult(
                document=doc,
                score=score,
                relevance=relevance
            ))
            
            if len(results) >= top_k:
                break
        
        # Cache results
        self.cache[cache_key] = (datetime.now(), results)
        
        logger.info(f"Retrieved {len(results)} results for query: {query[:50]}")
        return results
    
    def enhance_prompt_with_context(
        self,
        original_prompt: str,
        task_type: Optional[str] = None,
        max_context_tokens: int = 2000
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Enhance a prompt with relevant context from the RAG system
        
        Args:
            original_prompt: The original prompt
            task_type: Optional task type for better context selection
            max_context_tokens: Maximum tokens for context
        
        Returns:
            Enhanced prompt and metadata about the enhancement
        """
        
        # Retrieve relevant context
        results = self.retrieve_context(
            query=original_prompt,
            top_k=5,
            category_filter=task_type
        )
        
        if not results:
            return original_prompt, {"enhanced": False, "reason": "No relevant context found"}
        
        # Build context section
        context_parts = []
        total_tokens = 0
        used_documents = []
        
        for result in results:
            # Estimate tokens (rough approximation)
            doc_tokens = len(result.document.content.split()) * 1.3
            if total_tokens + doc_tokens > max_context_tokens:
                break
            
            context_parts.append(f"[Context {result.relevance} relevance - {result.score:.2f}]:\n{result.document.content}")
            total_tokens += doc_tokens
            used_documents.append({
                "id": result.document.id,
                "source": result.document.source,
                "relevance": result.relevance,
                "score": result.score
            })
        
        if not context_parts:
            return original_prompt, {"enhanced": False, "reason": "Context too large"}
        
        # Build enhanced prompt
        enhanced_prompt = f"""## Relevant Context from Knowledge Base

{chr(10).join(context_parts)}

## Original Task

{original_prompt}

Please complete the task using the provided context as guidance where relevant."""
        
        metadata = {
            "enhanced": True,
            "documents_used": len(used_documents),
            "total_context_tokens": int(total_tokens),
            "sources": used_documents,
            "enhancement_timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Enhanced prompt with {len(used_documents)} context documents")
        return enhanced_prompt, metadata
    
    def clear_cache(self):
        """Clear the query cache"""
        self.cache.clear()
        logger.info("Cleared RAG cache")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the RAG system"""
        return {
            "total_documents": len(self.vector_store.documents),
            "total_embeddings": len(self.vector_store.embeddings),
            "cache_size": len(self.cache),
            "persist_path": self.vector_store.persist_path,
            "categories": list(set(
                doc.metadata.get('category', 'uncategorized') 
                for doc in self.vector_store.documents.values()
            ))
        }


# Singleton instance
_rag_service_instance = None

def get_rag_service() -> RAGService:
    """Get or create the RAG service singleton"""
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService()
    return _rag_service_instance