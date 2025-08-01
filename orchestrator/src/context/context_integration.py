"""
Context Engineering Integration
===============================

Integration of context engineering system with the enhanced orchestrator.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import os

# Import context engineering components
from src.context.chunking import MultiModalChunker
from src.context.vector_store import PgVectorStore
from src.context.context_retriever import ContextRetriever, RetrievalConfig
from src.context.prompt_builder import ContextAwarePromptBuilder, PromptType
from src.context.learning_engine import AdaptiveLearningEngine
from src.context.agent_collaboration import create_collaboration_system

logger = logging.getLogger(__name__)

class ContextEngineeringManager:
    """Manager for all context engineering components"""
    
    def __init__(self, database_url: str = None):
        self.database_url = database_url or os.getenv('DATABASE_URL')
        
        # Components
        self.vector_store = None
        self.embedding_generator = None
        self.chunker = None
        self.context_retriever = None
        self.prompt_builder = None
        self.learning_engine = None
        self.collaboration_system = None
        
        self.initialized = False
    
    async def initialize(self):
        """Initialize all context engineering components"""
        try:
            logger.info("Initializing Context Engineering System...")
            
            # Initialize vector store
            self.vector_store = PgVectorStore(self.database_url)
            await self.vector_store.initialize(dimension=384)  # Using sentence transformer dimension
            logger.info("Vector store initialized")
            
            # Create simple embedding generator (mock for now)
            self.embedding_generator = SimpleEmbeddingGenerator()
            logger.info("Embedding generator initialized")
            
            # Initialize chunker
            self.chunker = MultiModalChunker()
            logger.info("Document chunker initialized")
            
            # Initialize context retriever
            retrieval_config = RetrievalConfig(
                max_results=10,
                similarity_threshold=0.3,
                enable_reranking=True
            )
            self.context_retriever = ContextRetriever(
                self.vector_store, 
                self.embedding_generator, 
                retrieval_config
            )
            logger.info("Context retriever initialized")
            
            # Initialize prompt builder
            self.prompt_builder = ContextAwarePromptBuilder()
            logger.info("Prompt builder initialized")
            
            # Initialize learning engine
            self.learning_engine = AdaptiveLearningEngine(
                self.vector_store, 
                self.embedding_generator
            )
            logger.info("Learning engine initialized")
            
            # Initialize collaboration system
            self.collaboration_system = create_collaboration_system(
                self.vector_store, 
                self.embedding_generator
            )
            logger.info("Collaboration system initialized")
            
            self.initialized = True
            logger.info("Context Engineering System fully initialized!")
            
        except Exception as e:
            logger.error(f"Failed to initialize context engineering system: {str(e)}")
            raise
    
    async def retrieve_context_for_task(self, task_description: str, task_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve relevant context for a task"""
        if not self.initialized:
            await self.initialize()
        
        try:
            context_results = await self.context_retriever.retrieve_context(
                query=task_description,
                task_type=task_type
            )
            
            return [
                {
                    'content': result.content,
                    'source': result.source,
                    'relevance_score': result.relevance_score,
                    'context_type': result.context_type,
                    'metadata': result.metadata
                }
                for result in context_results
            ]
            
        except Exception as e:
            logger.error(f"Error retrieving context: {str(e)}")
            return []
    
    async def build_context_aware_prompt(self, task_description: str, 
                                       prompt_type: str = 'general',
                                       context_results: List[Dict[str, Any]] = None) -> Dict[str, str]:
        """Build a context-aware prompt"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Convert string prompt type to enum
            prompt_type_enum = PromptType.GENERAL
            if prompt_type == 'code_generation':
                prompt_type_enum = PromptType.CODE_GENERATION
            elif prompt_type == 'api_development':
                prompt_type_enum = PromptType.API_DEVELOPMENT
            elif prompt_type == 'debugging':
                prompt_type_enum = PromptType.DEBUGGING
            
            # Convert context results to ContextResult objects
            from src.context.context_retriever import ContextResult
            context_objects = []
            
            if context_results:
                for ctx in context_results:
                    context_obj = ContextResult(
                        content=ctx.get('content', ''),
                        source=ctx.get('source', ''),
                        relevance_score=ctx.get('relevance_score', 0.0),
                        context_type=ctx.get('context_type', 'general'),
                        metadata=ctx.get('metadata', {}),
                        chunk_id=ctx.get('chunk_id', '')
                    )
                    context_objects.append(context_obj)
            
            # Build prompt
            prompt_result = await self.prompt_builder.build_prompt(
                task_description=task_description,
                prompt_type=prompt_type_enum,
                context_results=context_objects
            )
            
            return prompt_result
            
        except Exception as e:
            logger.error(f"Error building context-aware prompt: {str(e)}")
            return {
                'system_prompt': "You are a helpful AI assistant.",
                'user_prompt': task_description,
                'template_name': 'fallback',
                'context_count': 0
            }
    
    async def learn_from_task_execution(self, task_description: str, task_type: str,
                                      context_used: Dict[str, Any], outcome: str,
                                      success: bool, execution_time: float,
                                      agent_used: str = 'default'):
        """Learn from task execution"""
        if not self.initialized:
            await self.initialize()
        
        try:
            await self.learning_engine.learn_from_task_execution(
                task_description=task_description,
                task_type=task_type,
                context_used=context_used,
                outcome=outcome,
                success=success,
                execution_time=execution_time,
                agent_used=agent_used
            )
            
        except Exception as e:
            logger.error(f"Error learning from task execution: {str(e)}")
    
    async def ingest_document(self, file_path: str, content: str = None) -> Dict[str, Any]:
        """Ingest a document into the knowledge base"""
        if not self.initialized:
            await self.initialize()
        
        try:
            # Read content if not provided
            if content is None:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Determine content type
            if file_path.endswith('.py'):
                content_type = 'code'
            elif file_path.endswith('.md'):
                content_type = 'markdown'
            else:
                content_type = 'text'
            
            # Chunk the document
            chunks = self.chunker.chunk_document(content, file_path, content_type)
            
            if not chunks:
                return {'success': False, 'message': 'No chunks generated'}
            
            # Generate embeddings (mock for now)
            for chunk in chunks:
                chunk['embedding'] = [0.1] * 384  # Mock embedding
            
            # Store in vector database
            await self.vector_store.add_embeddings(chunks)
            
            # Update document record
            doc_id = await self.vector_store.add_document_record(file_path, {
                'content_type': content_type,
                'chunk_count': len(chunks)
            })
            
            return {
                'success': True,
                'chunks_created': len(chunks),
                'document_id': doc_id,
                'content_type': content_type
            }
            
        except Exception as e:
            logger.error(f"Error ingesting document: {str(e)}")
            return {'success': False, 'message': str(e)}
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get context engineering system statistics"""
        if not self.initialized:
            return {'initialized': False}
        
        try:
            # Get vector store stats
            vector_stats = await self.vector_store.get_document_stats()
            
            # Get learning stats
            learning_stats = self.learning_engine.get_learning_stats()
            
            # Get collaboration stats
            collab_stats = self.collaboration_system.get_collaboration_stats()
            
            return {
                'initialized': True,
                'vector_store': vector_stats,
                'learning_engine': learning_stats,
                'collaboration_system': collab_stats,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting system stats: {str(e)}")
            return {'initialized': True, 'error': str(e)}
    
    async def close(self):
        """Close all connections"""
        if self.vector_store:
            await self.vector_store.close()

class SimpleEmbeddingGenerator:
    """Simple mock embedding generator for testing"""
    
    def __init__(self):
        self.config = type('Config', (), {
            'model_name': 'mock_model',
            'dimension': 384
        })()
    
    async def generate_embeddings(self, texts: List[str], metadata: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Generate mock embeddings"""
        results = []
        for i, text in enumerate(texts):
            # Simple hash-based mock embedding
            import hashlib
            text_hash = hashlib.md5(text.encode()).hexdigest()
            
            # Convert hash to numbers and normalize
            embedding = []
            for j in range(0, len(text_hash), 2):
                val = int(text_hash[j:j+2], 16) / 255.0
                embedding.append(val)
            
            # Pad or truncate to 384 dimensions
            while len(embedding) < 384:
                embedding.extend(embedding[:384-len(embedding)])
            embedding = embedding[:384]
            
            result = {
                'text': text,
                'embedding': embedding,
                'dimension': 384,
                'model': 'mock_model'
            }
            
            if metadata and i < len(metadata):
                result.update(metadata[i])
            
            results.append(result)
        
        return results

# Global instance
context_manager = None

async def get_context_manager() -> ContextEngineeringManager:
    """Get or create the global context manager"""
    global context_manager
    
    if context_manager is None:
        context_manager = ContextEngineeringManager()
        await context_manager.initialize()
    
    return context_manager

async def initialize_context_engineering():
    """Initialize the context engineering system"""
    return await get_context_manager()
