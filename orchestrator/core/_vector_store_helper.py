"""
Helper method for ContextEngineeringIntegrator to perform semantic search using PgVectorStore.
This should be added as a method to the ContextEngineeringIntegrator class.
"""

async def _retrieve_from_vector_store(
    self,
    query: str,
    limit: int = 5,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Retrieve context from PgVectorStore using semantic search.
    
    Args:
        query: Search query  
        limit: Maximum chunks to retrieve
        similarity_threshold: Minimum similarity score
        
    Returns:
        List of context chunks
    """
    if not self.use_vector_store:
        return []
        
    try:
        # Lazy initialization of vector store
        if not self.vector_store:
            from database.database import get_database_url as get_db_session_string
            db_string = get_db_session_string()
            self.vector_store = PgVectorStore(connection_string=db_string)
            await self.vector_store.initialize(dimension=1536)
            self.logger.info("✅ Vector Store initialized")
            
        # Generate query embedding
        query_embedding = await self.embedding_generator.generate_embedding(query)
        
        # Perform semantic search
        results = await self.vector_store.similarity_search(
            query_embedding=query_embedding.tolist() if hasattr(query_embedding, 'tolist') else query_embedding,
            limit=limit,
            similarity_threshold=similarity_threshold
        )
        
        # Convert VectorSearchResult to dict format
        chunks = []
        for result in results:
            chunks.append({
                "content": result.content,
                "source_file": result.source_file,
                "chunk_id": result.id,
                "similarity_score": result.similarity_score,
                "similarity": result.similarity_score, # Alias for compatibility
                "metadata": result.metadata,
                "chunk_index": result.chunk_index,
                "content_type": result.content_type
            })
            
        self.logger.info(f"📊 Vector Store returned {len(chunks)} chunks for query: '{query[:50]}...'")
        return chunks
        
    except Exception as e:
        self.logger.error(f"Error retrieving from vector store: {e}")
        return []
