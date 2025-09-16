
"""
Context Engineering System
==========================

Advanced context engineering system with RAG capabilities, knowledge base integration,
context-aware prompts, historical pattern recognition, and advanced agent collaboration.
"""

from .chunking import DocumentChunker, MultiModalChunker, ChunkMetadata
from .embeddings import EmbeddingGenerator, ContextAwareEmbedding, create_embedding_generator
from .vector_store import PgVectorStore, ContextAwareVectorStore, VectorSearchResult
from .context_retriever import ContextRetriever, SmartContextRetriever, ContextResult, RetrievalConfig
from .prompt_builder import ContextAwarePromptBuilder, AdaptivePromptBuilder, PromptType
from .learning_engine import AdaptiveLearningEngine, PatternRecognitionEngine, TaskPattern, LearningEvent
from .agent_collaboration import CollaborationOrchestrator, SmartAgentRegistry, AgentProfile, create_collaboration_system

__version__ = "2.0.0"

__all__ = [
    # Chunking
    'DocumentChunker',
    'MultiModalChunker', 
    'ChunkMetadata',
    
    # Embeddings
    'EmbeddingGenerator',
    'ContextAwareEmbedding',
    'create_embedding_generator',
    
    # Vector Store
    'PgVectorStore',
    'ContextAwareVectorStore',
    'VectorSearchResult',
    
    # Context Retrieval
    'ContextRetriever',
    'SmartContextRetriever',
    'ContextResult',
    'RetrievalConfig',
    
    # Prompt Building
    'ContextAwarePromptBuilder',
    'AdaptivePromptBuilder',
    'PromptType',
    
    # Learning Engine
    'AdaptiveLearningEngine',
    'PatternRecognitionEngine',
    'TaskPattern',
    'LearningEvent',
    
    # Agent Collaboration
    'CollaborationOrchestrator',
    'SmartAgentRegistry',
    'AgentProfile',
    'create_collaboration_system'
]

# Convenience function to create a complete context engineering system
async def create_context_engineering_system(database_url: str, 
                                           embedding_model_type: str = "sentence_transformer",
                                           embedding_model_name: str = None) -> dict:
    """
    Create a complete context engineering system with all components
    
    Args:
        database_url: PostgreSQL database URL with pgvector extension
        embedding_model_type: Type of embedding model ('sentence_transformer' or 'openai')
        embedding_model_name: Specific model name (optional)
        
    Returns:
        Dictionary containing all initialized components
    """
    
    # Initialize embedding generator
    embedding_generator = create_embedding_generator(
        model_type=embedding_model_type,
        model_name=embedding_model_name
    )
    
    # Initialize vector store
    vector_store = ContextAwareVectorStore(database_url)
    await vector_store.initialize(dimension=embedding_generator.config.dimension)
    
    # Initialize document chunker
    chunker = MultiModalChunker()
    
    # Initialize context retriever
    context_retriever = SmartContextRetriever(vector_store, embedding_generator)
    
    # Initialize prompt builder
    prompt_builder = AdaptivePromptBuilder()
    
    # Initialize learning engine
    learning_engine = AdaptiveLearningEngine(vector_store, embedding_generator)
    
    # Initialize collaboration system
    collaboration_system = create_collaboration_system(vector_store, embedding_generator)
    
    return {
        'embedding_generator': embedding_generator,
        'vector_store': vector_store,
        'chunker': chunker,
        'context_retriever': context_retriever,
        'prompt_builder': prompt_builder,
        'learning_engine': learning_engine,
        'collaboration_system': collaboration_system
    }
