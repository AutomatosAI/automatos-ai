
"""
Document Ingestion CLI
======================

Command-line interface for ingesting documents into the context engineering system.
"""

import asyncio
import argparse
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import logging
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from context_engineering.chunking import DocumentChunker, MultiModalChunker
from context_engineering.embeddings import create_embedding_generator
from context_engineering.vector_store import PgVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentIngestionPipeline:
    """Complete document ingestion pipeline"""
    
    def __init__(self, vector_store: PgVectorStore, embedding_generator, chunker: DocumentChunker):
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.chunker = chunker
        
        # Supported file types
        self.supported_extensions = {
            '.txt': 'text',
            '.md': 'markdown',
            '.py': 'code',
            '.js': 'code',
            '.ts': 'code',
            '.java': 'code',
            '.cpp': 'code',
            '.c': 'code',
            '.go': 'code',
            '.rs': 'code',
            '.php': 'code',
            '.rb': 'code',
            '.json': 'json',
            '.yaml': 'config',
            '.yml': 'config',
            '.toml': 'config',
            '.ini': 'config',
            '.html': 'markup',
            '.xml': 'markup',
            '.css': 'code',
            '.sql': 'code'
        }
    
    async def ingest_directory(self, directory_path: str, recursive: bool = True) -> Dict[str, Any]:
        """Ingest all supported files from a directory"""
        
        directory = Path(directory_path)
        if not directory.exists():
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        # Find all supported files
        files_to_process = []
        
        if recursive:
            for ext in self.supported_extensions.keys():
                files_to_process.extend(directory.rglob(f"*{ext}"))
        else:
            for ext in self.supported_extensions.keys():
                files_to_process.extend(directory.glob(f"*{ext}"))
        
        logger.info(f"Found {len(files_to_process)} files to process")
        
        # Process files
        results = {
            'processed_files': 0,
            'total_chunks': 0,
            'failed_files': [],
            'processing_time': 0
        }
        
        start_time = datetime.now()
        
        for file_path in files_to_process:
            try:
                file_results = await self.ingest_file(str(file_path))
                results['processed_files'] += 1
                results['total_chunks'] += file_results['chunk_count']
                
                logger.info(f"Processed {file_path.name}: {file_results['chunk_count']} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {str(e)}")
                results['failed_files'].append(str(file_path))
        
        results['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        return results
    
    async def ingest_file(self, file_path: str) -> Dict[str, Any]:
        """Ingest a single file"""
        
        file_path_obj = Path(file_path)
        
        if not file_path_obj.exists():
            raise ValueError(f"File does not exist: {file_path}")
        
        # Determine content type
        extension = file_path_obj.suffix.lower()
        content_type = self.supported_extensions.get(extension, 'text')
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
        
        if not content.strip():
            logger.warning(f"File is empty: {file_path}")
            return {'chunk_count': 0, 'embeddings_count': 0}
        
        # Chunk the document
        if content_type == 'code':
            language = self._detect_programming_language(file_path_obj.suffix)
            chunks = self.chunker.chunk_code_file(content, file_path, language)
        else:
            chunks = self.chunker.chunk_document(content, file_path, content_type)
        
        if not chunks:
            logger.warning(f"No chunks generated for: {file_path}")
            return {'chunk_count': 0, 'embeddings_count': 0}
        
        # Generate embeddings
        chunk_texts = [chunk['content'] for chunk in chunks]
        chunk_metadata = [chunk['metadata'] for chunk in chunks]
        
        embeddings = await self.embedding_generator.generate_embeddings(
            texts=chunk_texts,
            metadata=chunk_metadata
        )
        
        # Combine chunks with embeddings
        for i, embedding in enumerate(embeddings):
            if i < len(chunks):
                chunks[i].update(embedding)
        
        # Store in vector database
        await self.vector_store.add_embeddings(chunks)
        
        # Update document record
        doc_id = await self.vector_store.add_document_record(file_path, {
            'content_type': content_type,
            'file_size': file_path_obj.stat().st_size,
            'chunk_count': len(chunks)
        })
        
        if doc_id:
            await self.vector_store.update_document_chunk_count(file_path, len(chunks))
        
        return {
            'chunk_count': len(chunks),
            'embeddings_count': len(embeddings),
            'content_type': content_type
        }
    
    def _detect_programming_language(self, extension: str) -> str:
        """Detect programming language from file extension"""
        
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust',
            '.php': 'php',
            '.rb': 'ruby',
            '.css': 'css',
            '.sql': 'sql'
        }
        
        return lang_map.get(extension.lower(), 'text')

async def main():
    """Main CLI function"""
    
    parser = argparse.ArgumentParser(description='Ingest documents into the context engineering system')
    parser.add_argument('--path', required=True, help='Path to file or directory to ingest')
    parser.add_argument('--recursive', action='store_true', help='Process directories recursively')
    parser.add_argument('--database-url', help='Database URL (defaults to environment variable)')
    parser.add_argument('--embedding-model', default='sentence_transformer', 
                       choices=['sentence_transformer', 'openai'],
                       help='Embedding model to use')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size for document splitting')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap size')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize components
    try:
        # Vector store
        database_url = args.database_url or os.getenv('DATABASE_URL')
        if not database_url:
            logger.error("Database URL not provided. Set DATABASE_URL environment variable or use --database-url")
            return 1
        
        vector_store = PgVectorStore(database_url)
        
        # Embedding generator
        embedding_generator = create_embedding_generator(
            model_type=args.embedding_model
        )
        
        # Document chunker
        chunker = MultiModalChunker(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Initialize vector store
        await vector_store.initialize(dimension=embedding_generator.config.dimension)
        
        # Create ingestion pipeline
        pipeline = DocumentIngestionPipeline(vector_store, embedding_generator, chunker)
        
        # Process input path
        input_path = Path(args.path)
        
        if input_path.is_file():
            logger.info(f"Processing single file: {input_path}")
            results = await pipeline.ingest_file(str(input_path))
            
            print(f"\nFile Processing Results:")
            print(f"  Chunks created: {results['chunk_count']}")
            print(f"  Embeddings generated: {results['embeddings_count']}")
            print(f"  Content type: {results['content_type']}")
            
        elif input_path.is_dir():
            logger.info(f"Processing directory: {input_path}")
            results = await pipeline.ingest_directory(str(input_path), args.recursive)
            
            print(f"\nDirectory Processing Results:")
            print(f"  Files processed: {results['processed_files']}")
            print(f"  Total chunks created: {results['total_chunks']}")
            print(f"  Processing time: {results['processing_time']:.2f} seconds")
            
            if results['failed_files']:
                print(f"  Failed files: {len(results['failed_files'])}")
                for failed_file in results['failed_files']:
                    print(f"    - {failed_file}")
        
        else:
            logger.error(f"Path does not exist: {input_path}")
            return 1
        
        # Show database statistics
        stats = await vector_store.get_document_stats()
        print(f"\nDatabase Statistics:")
        print(f"  Total documents: {stats.get('total_documents', 0)}")
        print(f"  Total chunks: {stats.get('total_chunks', 0)}")
        print(f"  Average chunk length: {stats.get('avg_chunk_length', 0):.0f} characters")
        
        content_types = stats.get('content_type_distribution', {})
        if content_types:
            print(f"  Content type distribution:")
            for content_type, count in content_types.items():
                print(f"    {content_type}: {count}")
        
        # Close connections
        await vector_store.close()
        
        logger.info("Ingestion completed successfully")
        return 0
        
    except Exception as e:
        logger.error(f"Ingestion failed: {str(e)}")
        return 1

if __name__ == '__main__':
    sys.exit(asyncio.run(main()))
