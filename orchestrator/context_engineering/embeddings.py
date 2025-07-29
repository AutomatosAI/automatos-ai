
"""
Embeddings Generation System
============================

Advanced embeddings generation with multiple model support and optimization.
"""

import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
import openai
from openai import AsyncOpenAI
import os

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "all-MiniLM-L6-v2"
    model_type: str = "sentence_transformer"  # 'sentence_transformer', 'openai'
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    cache_embeddings: bool = True

class EmbeddingGenerator:
    """Advanced embedding generation with multiple model support"""
    
    def __init__(self, config: EmbeddingConfig = None):
        self.config = config or EmbeddingConfig()
        self.model = None
        self.openai_client = None
        self._embedding_cache = {}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model"""
        try:
            if self.config.model_type == "sentence_transformer":
                self.model = SentenceTransformer(self.config.model_name)
                self.config.dimension = self.model.get_sentence_embedding_dimension()
                logger.info(f"Initialized SentenceTransformer: {self.config.model_name}")
                
            elif self.config.model_type == "openai":
                self.openai_client = AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
                # OpenAI text-embedding-ada-002 has 1536 dimensions
                self.config.dimension = 1536
                logger.info("Initialized OpenAI embeddings")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            # Fallback to default model
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.config.model_type = "sentence_transformer"
            self.config.dimension = 384
    
    async def generate_embeddings(self, texts: List[str], 
                                metadata: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings
            metadata: Optional metadata for each text
            
        Returns:
            List of embedding dictionaries with metadata
        """
        if not texts:
            return []
        
        try:
            if self.config.model_type == "sentence_transformer":
                embeddings = await self._generate_sentence_transformer_embeddings(texts)
            elif self.config.model_type == "openai":
                embeddings = await self._generate_openai_embeddings(texts)
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Combine embeddings with metadata
            results = []
            for i, embedding in enumerate(embeddings):
                result = {
                    'text': texts[i],
                    'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
                    'dimension': len(embedding),
                    'model': self.config.model_name,
                    'model_type': self.config.model_type
                }
                
                if metadata and i < len(metadata):
                    result.update(metadata[i])
                
                results.append(result)
            
            logger.info(f"Generated {len(results)} embeddings")
            return results
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return []
    
    async def _generate_sentence_transformer_embeddings(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings using SentenceTransformer"""
        
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if self.config.cache_embeddings:
            for i, text in enumerate(texts):
                text_hash = hash(text)
                if text_hash in self._embedding_cache:
                    cached_embeddings.append((i, self._embedding_cache[text_hash]))
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        new_embeddings = []
        if uncached_texts:
            # Process in batches to avoid memory issues
            for i in range(0, len(uncached_texts), self.config.batch_size):
                batch = uncached_texts[i:i + self.config.batch_size]
                batch_embeddings = self.model.encode(
                    batch,
                    normalize_embeddings=self.config.normalize,
                    show_progress_bar=False
                )
                new_embeddings.extend(batch_embeddings)
            
            # Cache new embeddings
            if self.config.cache_embeddings:
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self._embedding_cache[hash(text)] = embedding
        
        # Combine cached and new embeddings in correct order
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        for idx, embedding in cached_embeddings:
            all_embeddings[idx] = embedding
        
        # Place new embeddings
        for i, embedding in enumerate(new_embeddings):
            original_idx = uncached_indices[i]
            all_embeddings[original_idx] = embedding
        
        return all_embeddings
    
    async def _generate_openai_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using OpenAI API"""
        
        embeddings = []
        
        # Process in batches (OpenAI has rate limits)
        batch_size = min(self.config.batch_size, 100)  # OpenAI limit
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                response = await self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Rate limiting
                if len(texts) > batch_size:
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"OpenAI embedding error for batch {i//batch_size}: {str(e)}")
                # Fallback to zero embeddings
                embeddings.extend([[0.0] * self.config.dimension] * len(batch))
        
        return embeddings
    
    def generate_embedding_sync(self, text: str) -> Optional[List[float]]:
        """Generate single embedding synchronously"""
        try:
            if self.config.model_type == "sentence_transformer":
                embedding = self.model.encode([text], normalize_embeddings=self.config.normalize)[0]
                return embedding.tolist()
            else:
                # For OpenAI, we need async, so return None for sync calls
                logger.warning("Sync embedding generation not supported for OpenAI model")
                return None
        except Exception as e:
            logger.error(f"Error generating sync embedding: {str(e)}")
            return None
    
    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings"""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            return 0.0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            'model_name': self.config.model_name,
            'model_type': self.config.model_type,
            'dimension': self.config.dimension,
            'batch_size': self.config.batch_size,
            'normalize': self.config.normalize,
            'cache_size': len(self._embedding_cache)
        }

class ContextAwareEmbedding:
    """Context-aware embedding generation with domain adaptation"""
    
    def __init__(self, base_generator: EmbeddingGenerator):
        self.base_generator = base_generator
        self.domain_weights = {}
        self.context_templates = {
            'code': "This is a code snippet: {text}",
            'documentation': "This is documentation: {text}",
            'api': "This is API documentation: {text}",
            'tutorial': "This is a tutorial: {text}",
            'error': "This is an error message: {text}",
            'config': "This is configuration: {text}"
        }
    
    async def generate_context_aware_embeddings(self, 
                                              texts: List[str],
                                              contexts: List[str],
                                              metadata: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate embeddings with context awareness
        
        Args:
            texts: Original texts
            contexts: Context types for each text
            metadata: Additional metadata
            
        Returns:
            Context-enhanced embeddings
        """
        
        # Enhance texts with context templates
        enhanced_texts = []
        for text, context in zip(texts, contexts):
            template = self.context_templates.get(context, "{text}")
            enhanced_text = template.format(text=text)
            enhanced_texts.append(enhanced_text)
        
        # Generate embeddings for enhanced texts
        embeddings = await self.base_generator.generate_embeddings(enhanced_texts, metadata)
        
        # Add context information to results
        for i, embedding in enumerate(embeddings):
            embedding['context_type'] = contexts[i] if i < len(contexts) else 'general'
            embedding['original_text'] = texts[i] if i < len(texts) else ''
            embedding['enhanced_text'] = enhanced_texts[i] if i < len(enhanced_texts) else ''
        
        return embeddings
    
    def set_domain_weights(self, domain: str, weights: Dict[str, float]):
        """Set domain-specific weights for embedding adjustment"""
        self.domain_weights[domain] = weights
    
    def adjust_embedding_for_domain(self, embedding: List[float], domain: str) -> List[float]:
        """Adjust embedding based on domain-specific weights"""
        if domain not in self.domain_weights:
            return embedding
        
        weights = self.domain_weights[domain]
        adjusted = np.array(embedding)
        
        # Apply domain-specific transformations
        if 'scale' in weights:
            adjusted *= weights['scale']
        
        if 'bias' in weights:
            adjusted += weights['bias']
        
        # Normalize if needed
        if self.base_generator.config.normalize:
            norm = np.linalg.norm(adjusted)
            if norm > 0:
                adjusted = adjusted / norm
        
        return adjusted.tolist()

# Factory function for easy initialization
def create_embedding_generator(model_type: str = "sentence_transformer",
                             model_name: str = None) -> EmbeddingGenerator:
    """
    Factory function to create embedding generator
    
    Args:
        model_type: 'sentence_transformer' or 'openai'
        model_name: Specific model name (optional)
        
    Returns:
        Configured EmbeddingGenerator
    """
    
    if model_name is None:
        if model_type == "sentence_transformer":
            model_name = "all-MiniLM-L6-v2"  # Fast and good quality
        elif model_type == "openai":
            model_name = "text-embedding-ada-002"
    
    config = EmbeddingConfig(
        model_name=model_name,
        model_type=model_type
    )
    
    return EmbeddingGenerator(config)
