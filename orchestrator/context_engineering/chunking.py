
"""
Document Chunking and Preprocessing System
==========================================

Advanced document chunking with semantic awareness and multi-modal support.
"""

import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging
# Simple Document class
class Document:
    def __init__(self, page_content: str, metadata: dict = None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Simple text splitter implementation
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200, 
                 length_function=len, separators: List[str] = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = []
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            for chunk_text in text_chunks:
                chunks.append(Document(page_content=chunk_text, metadata=doc.metadata.copy()))
        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks using recursive approach"""
        if self.length_function(text) <= self.chunk_size:
            return [text]
        
        # Try each separator in order
        for separator in self.separators:
            if separator in text:
                splits = text.split(separator)
                return self._merge_splits(splits, separator)
        
        # If no separator works, split by character count
        return self._split_by_length(text)
    
    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks with overlap"""
        chunks = []
        current_chunk = ""
        
        for split in splits:
            # Check if adding this split would exceed chunk size
            test_chunk = current_chunk + separator + split if current_chunk else split
            
            if self.length_function(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk)
                
                # Handle overlap
                if self.chunk_overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_text + separator + split if overlap_text else split
                else:
                    current_chunk = split
        
        # Add final chunk
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _split_by_length(self, text: str) -> List[str]:
        """Split text by character length with overlap"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - self.chunk_overlap if self.chunk_overlap > 0 else end
            
            if start >= len(text):
                break
        
        return chunks

logger = logging.getLogger(__name__)

@dataclass
class ChunkMetadata:
    """Metadata for document chunks"""
    source_file: str
    chunk_index: int
    chunk_type: str  # 'text', 'code', 'markdown', 'json'
    language: Optional[str] = None
    section_title: Optional[str] = None
    parent_chunk_id: Optional[str] = None
    semantic_tags: List[str] = None
    
    def __post_init__(self):
        if self.semantic_tags is None:
            self.semantic_tags = []

class DocumentChunker:
    """Advanced document chunking with context awareness"""
    
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 min_chunk_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        # Different splitters for different content types
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        self.code_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\nclass ", "\n\ndef ", "\n\n", "\n", " ", ""]
        )
        
        self.markdown_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""]
        )
    
    def chunk_document(self, content: str, source_file: str, 
                      content_type: str = 'text') -> List[Dict[str, Any]]:
        """
        Chunk a document with appropriate strategy based on content type
        
        Args:
            content: Document content
            source_file: Source file path
            content_type: Type of content ('text', 'code', 'markdown', 'json')
            
        Returns:
            List of chunk dictionaries with metadata
        """
        try:
            # Select appropriate splitter
            splitter = self._get_splitter(content_type)
            
            # Create document object
            doc = Document(page_content=content, metadata={"source": source_file})
            
            # Split document
            chunks = splitter.split_documents([doc])
            
            # Process chunks with metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                if len(chunk.page_content.strip()) < self.min_chunk_size:
                    continue
                
                chunk_id = self._generate_chunk_id(source_file, i, chunk.page_content)
                
                metadata = ChunkMetadata(
                    source_file=source_file,
                    chunk_index=i,
                    chunk_type=content_type,
                    language=self._detect_language(chunk.page_content, content_type),
                    section_title=self._extract_section_title(chunk.page_content, content_type),
                    semantic_tags=self._extract_semantic_tags(chunk.page_content, content_type)
                )
                
                processed_chunks.append({
                    'id': chunk_id,
                    'content': chunk.page_content,
                    'metadata': metadata,
                    'source_file': source_file,
                    'chunk_index': i,
                    'content_type': content_type
                })
            
            logger.info(f"Chunked {source_file}: {len(processed_chunks)} chunks")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"Error chunking document {source_file}: {str(e)}")
            return []
    
    def chunk_code_file(self, content: str, source_file: str, 
                       language: str = None) -> List[Dict[str, Any]]:
        """Specialized chunking for code files"""
        
        # Extract functions, classes, and methods
        code_blocks = self._extract_code_blocks(content, language)
        
        chunks = []
        for i, block in enumerate(code_blocks):
            chunk_id = self._generate_chunk_id(source_file, i, block['content'])
            
            metadata = ChunkMetadata(
                source_file=source_file,
                chunk_index=i,
                chunk_type='code',
                language=language or self._detect_programming_language(source_file),
                section_title=block.get('name'),
                semantic_tags=block.get('tags', [])
            )
            
            chunks.append({
                'id': chunk_id,
                'content': block['content'],
                'metadata': metadata,
                'source_file': source_file,
                'chunk_index': i,
                'content_type': 'code',
                'code_type': block.get('type')  # 'function', 'class', 'method'
            })
        
        return chunks
    
    def _get_splitter(self, content_type: str):
        """Get appropriate splitter for content type"""
        splitters = {
            'code': self.code_splitter,
            'markdown': self.markdown_splitter,
            'text': self.text_splitter
        }
        return splitters.get(content_type, self.text_splitter)
    
    def _generate_chunk_id(self, source_file: str, index: int, content: str) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{Path(source_file).stem}_{index}_{content_hash}"
    
    def _detect_language(self, content: str, content_type: str) -> Optional[str]:
        """Detect content language"""
        if content_type == 'code':
            return self._detect_programming_language_from_content(content)
        return None
    
    def _detect_programming_language(self, filename: str) -> Optional[str]:
        """Detect programming language from file extension"""
        ext_map = {
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
            '.swift': 'swift',
            '.kt': 'kotlin'
        }
        ext = Path(filename).suffix.lower()
        return ext_map.get(ext)
    
    def _detect_programming_language_from_content(self, content: str) -> Optional[str]:
        """Detect programming language from content patterns"""
        patterns = {
            'python': [r'def\s+\w+\(', r'import\s+\w+', r'from\s+\w+\s+import', r'class\s+\w+:'],
            'javascript': [r'function\s+\w+\(', r'const\s+\w+\s*=', r'let\s+\w+\s*=', r'=>'],
            'java': [r'public\s+class\s+\w+', r'public\s+static\s+void\s+main', r'import\s+java\.'],
            'cpp': [r'#include\s*<', r'int\s+main\s*\(', r'std::'],
            'go': [r'package\s+\w+', r'func\s+\w+\(', r'import\s+\(']
        }
        
        for lang, lang_patterns in patterns.items():
            if any(re.search(pattern, content) for pattern in lang_patterns):
                return lang
        
        return None
    
    def _extract_section_title(self, content: str, content_type: str) -> Optional[str]:
        """Extract section title from content"""
        if content_type == 'markdown':
            # Look for markdown headers
            lines = content.split('\n')
            for line in lines[:3]:  # Check first few lines
                if line.startswith('#'):
                    return line.strip('#').strip()
        
        elif content_type == 'code':
            # Look for function/class definitions
            lines = content.split('\n')
            for line in lines[:5]:
                if 'def ' in line or 'class ' in line or 'function ' in line:
                    return line.strip()
        
        return None
    
    def _extract_semantic_tags(self, content: str, content_type: str) -> List[str]:
        """Extract semantic tags from content"""
        tags = []
        
        if content_type == 'code':
            # Programming concepts
            if 'async' in content or 'await' in content:
                tags.append('async')
            if 'class' in content:
                tags.append('oop')
            if 'test' in content.lower() or 'assert' in content:
                tags.append('testing')
            if 'api' in content.lower() or 'endpoint' in content.lower():
                tags.append('api')
            if 'database' in content.lower() or 'db' in content.lower():
                tags.append('database')
        
        elif content_type == 'text' or content_type == 'markdown':
            # Documentation concepts
            if 'install' in content.lower() or 'setup' in content.lower():
                tags.append('installation')
            if 'config' in content.lower() or 'configuration' in content.lower():
                tags.append('configuration')
            if 'example' in content.lower() or 'tutorial' in content.lower():
                tags.append('example')
            if 'api' in content.lower():
                tags.append('api-docs')
        
        return tags
    
    def _extract_code_blocks(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Extract meaningful code blocks (functions, classes, etc.)"""
        blocks = []
        
        if language == 'python':
            # Extract Python functions and classes
            class_pattern = r'class\s+(\w+).*?(?=\nclass|\ndef|\Z)'
            function_pattern = r'def\s+(\w+)\(.*?\).*?(?=\ndef|\nclass|\Z)'
            
            for match in re.finditer(class_pattern, content, re.DOTALL):
                blocks.append({
                    'type': 'class',
                    'name': match.group(1),
                    'content': match.group(0),
                    'tags': ['class', 'oop']
                })
            
            for match in re.finditer(function_pattern, content, re.DOTALL):
                blocks.append({
                    'type': 'function',
                    'name': match.group(1),
                    'content': match.group(0),
                    'tags': ['function']
                })
        
        # If no specific blocks found, use regular chunking
        if not blocks:
            chunks = self.code_splitter.split_text(content)
            for i, chunk in enumerate(chunks):
                blocks.append({
                    'type': 'code_block',
                    'name': f'block_{i}',
                    'content': chunk,
                    'tags': ['code']
                })
        
        return blocks

class MultiModalChunker(DocumentChunker):
    """Extended chunker for multi-modal content"""
    
    def chunk_mixed_content(self, content: str, source_file: str) -> List[Dict[str, Any]]:
        """Chunk content with mixed types (e.g., markdown with code blocks)"""
        
        # Split by code blocks first
        code_block_pattern = r'```(\w+)?\n(.*?)\n```'
        parts = []
        last_end = 0
        
        for match in re.finditer(code_block_pattern, content, re.DOTALL):
            # Add text before code block
            if match.start() > last_end:
                text_content = content[last_end:match.start()].strip()
                if text_content:
                    parts.append({
                        'content': text_content,
                        'type': 'text',
                        'language': None
                    })
            
            # Add code block
            parts.append({
                'content': match.group(2),
                'type': 'code',
                'language': match.group(1)
            })
            
            last_end = match.end()
        
        # Add remaining text
        if last_end < len(content):
            remaining = content[last_end:].strip()
            if remaining:
                parts.append({
                    'content': remaining,
                    'type': 'text',
                    'language': None
                })
        
        # Chunk each part appropriately
        all_chunks = []
        for i, part in enumerate(parts):
            if part['type'] == 'code':
                chunks = self.chunk_code_file(part['content'], f"{source_file}#code_{i}", part['language'])
            else:
                chunks = self.chunk_document(part['content'], f"{source_file}#text_{i}", 'text')
            
            all_chunks.extend(chunks)
        
        return all_chunks
