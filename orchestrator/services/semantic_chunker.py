"""
Semantic Chunker - Intelligent Document Chunking for RAG
=========================================================

Chunks documents respecting semantic boundaries:
- Sentence boundaries (never split mid-sentence)
- Paragraph boundaries (prefer natural breaks)
- Section headers (keep with their content)
- Code blocks (keep complete)
- Lists (keep items together)

This is critical for high-quality RAG retrieval.
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Content type for specialized chunking"""
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    JSON = "json"
    HTML = "html"


@dataclass
class SemanticChunk:
    """A semantically coherent chunk of content"""
    content: str
    chunk_index: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any]
    
    @property
    def token_estimate(self) -> int:
        """Rough token estimate (1 token ≈ 4 chars)"""
        return len(self.content) // 4


class SemanticChunker:
    """
    Intelligent chunker that respects semantic boundaries.
    
    Key principles:
    1. Never split mid-sentence
    2. Prefer paragraph boundaries
    3. Keep headers with their content
    4. Keep code blocks intact
    5. Maintain configurable overlap for context
    """
    
    def __init__(
        self,
        target_chunk_size: int = 512,      # Target tokens per chunk
        max_chunk_size: int = 1024,         # Hard max tokens
        min_chunk_size: int = 100,          # Min tokens (skip smaller)
        overlap_tokens: int = 50,           # Context overlap
        respect_code_blocks: bool = True,
        respect_headers: bool = True
    ):
        self.target_chunk_size = target_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_tokens = overlap_tokens
        self.respect_code_blocks = respect_code_blocks
        self.respect_headers = respect_headers
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')
        
        # Markdown patterns
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        self.code_block_pattern = re.compile(r'```[\s\S]*?```', re.MULTILINE)
        self.list_item_pattern = re.compile(r'^[\s]*[-*+]\s+', re.MULTILINE)
    
    def chunk_document(
        self,
        content: str,
        source_file: str,
        content_type: ContentType = ContentType.TEXT,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> List[SemanticChunk]:
        """
        Chunk a document into semantic units.
        
        Args:
            content: Full document content
            source_file: Source filename for metadata
            content_type: Type of content for specialized handling
            extra_metadata: Additional metadata to include
            
        Returns:
            List of SemanticChunk objects
        """
        if not content or not content.strip():
            return []
        
        # Clean content
        content = self._clean_content(content)
        
        # Detect content type if not specified
        if content_type == ContentType.TEXT:
            content_type = self._detect_content_type(content, source_file)
        
        # Choose chunking strategy based on content type
        if content_type == ContentType.MARKDOWN:
            chunks = self._chunk_markdown(content)
        elif content_type == ContentType.CODE:
            chunks = self._chunk_code(content)
        else:
            chunks = self._chunk_text(content)
        
        # Post-process chunks
        final_chunks = []
        for idx, (chunk_content, start_char, end_char) in enumerate(chunks):
            # Skip chunks that are too small or empty
            if len(chunk_content.strip()) < self.min_chunk_size * 4:  # chars
                continue
            
            # Skip chunks that are just separators
            if self._is_separator_only(chunk_content):
                continue
            
            metadata = {
                "source_file": source_file,
                "content_type": content_type.value,
                "chunk_index": idx,
                "char_start": start_char,
                "char_end": end_char,
                **(extra_metadata or {})
            }
            
            # Extract section title if markdown
            if content_type == ContentType.MARKDOWN:
                section = self._extract_section_title(chunk_content)
                if section:
                    metadata["section_title"] = section
            
            final_chunks.append(SemanticChunk(
                content=chunk_content.strip(),
                chunk_index=len(final_chunks),
                start_char=start_char,
                end_char=end_char,
                metadata=metadata
            ))
        
        logger.info(f"Chunked {source_file}: {len(final_chunks)} semantic chunks from {len(content)} chars")
        return final_chunks
    
    def _clean_content(self, content: str) -> str:
        """Clean content before chunking"""
        # Normalize line endings
        content = content.replace('\r\n', '\n').replace('\r', '\n')
        # Remove excessive blank lines (keep max 2)
        content = re.sub(r'\n{4,}', '\n\n\n', content)
        return content
    
    def _detect_content_type(self, content: str, filename: str) -> ContentType:
        """Detect content type from content and filename"""
        ext = filename.lower().split('.')[-1] if '.' in filename else ''
        
        if ext in ['md', 'markdown']:
            return ContentType.MARKDOWN
        elif ext in ['py', 'js', 'ts', 'tsx', 'jsx', 'java', 'go', 'rs', 'c', 'cpp', 'h']:
            return ContentType.CODE
        elif ext == 'json':
            return ContentType.JSON
        elif ext in ['html', 'htm']:
            return ContentType.HTML
        
        # Content-based detection
        if self.header_pattern.search(content) or '```' in content:
            return ContentType.MARKDOWN
        
        return ContentType.TEXT
    
    def _chunk_markdown(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk markdown content respecting structure"""
        chunks = []
        
        # First, protect code blocks
        code_blocks = []
        def save_code_block(match):
            code_blocks.append(match.group(0))
            return f"__CODE_BLOCK_{len(code_blocks) - 1}__"
        
        protected = self.code_block_pattern.sub(save_code_block, content)
        
        # Split by headers (keep header with content)
        sections = self._split_by_headers(protected)
        
        current_chunk = ""
        current_start = 0
        
        for section in sections:
            section_tokens = len(section) // 4
            
            if section_tokens <= self.target_chunk_size:
                # Section fits in target size
                if len(current_chunk) // 4 + section_tokens <= self.max_chunk_size:
                    current_chunk += section
                else:
                    # Save current chunk and start new one
                    if current_chunk.strip():
                        chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                    current_chunk = section
                    current_start = current_start + len(current_chunk)
            else:
                # Section too large, need to split by paragraphs
                if current_chunk.strip():
                    chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                    current_start = current_start + len(current_chunk)
                    current_chunk = ""
                
                # Split large section by paragraphs
                para_chunks = self._split_by_paragraphs(section)
                for para in para_chunks:
                    if len(current_chunk) // 4 + len(para) // 4 <= self.max_chunk_size:
                        current_chunk += para
                    else:
                        if current_chunk.strip():
                            chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                        current_chunk = para
                        current_start = current_start + len(current_chunk)
        
        # Don't forget the last chunk
        if current_chunk.strip():
            chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
        
        # Restore code blocks
        restored_chunks = []
        for chunk_content, start, end in chunks:
            for i, code_block in enumerate(code_blocks):
                chunk_content = chunk_content.replace(f"__CODE_BLOCK_{i}__", code_block)
            restored_chunks.append((chunk_content, start, end))
        
        return restored_chunks
    
    def _split_by_headers(self, content: str) -> List[str]:
        """Split content by markdown headers, keeping header with content"""
        # Find all header positions
        headers = list(self.header_pattern.finditer(content))
        
        if not headers:
            return [content]
        
        sections = []
        prev_end = 0
        
        for header in headers:
            # Content before this header
            if header.start() > prev_end:
                before = content[prev_end:header.start()].strip()
                if before:
                    sections.append(before + "\n\n")
            prev_end = header.start()
        
        # Last section (from last header to end)
        if prev_end < len(content):
            sections.append(content[prev_end:])
        
        return sections if sections else [content]
    
    def _split_by_paragraphs(self, content: str) -> List[str]:
        """Split content by paragraphs (double newlines)"""
        paragraphs = re.split(r'\n\n+', content)
        return [p + "\n\n" for p in paragraphs if p.strip()]
    
    def _chunk_text(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk plain text respecting sentence boundaries"""
        chunks = []
        
        # Split into paragraphs first
        paragraphs = self._split_by_paragraphs(content)
        
        current_chunk = ""
        current_start = 0
        
        for para in paragraphs:
            para_tokens = len(para) // 4
            
            if para_tokens > self.max_chunk_size:
                # Paragraph too large, split by sentences
                if current_chunk.strip():
                    chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                    current_start = current_start + len(current_chunk)
                    current_chunk = ""
                
                sentences = self._split_by_sentences(para)
                for sentence in sentences:
                    if len(current_chunk) // 4 + len(sentence) // 4 <= self.max_chunk_size:
                        current_chunk += sentence
                    else:
                        if current_chunk.strip():
                            chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                        current_chunk = sentence
                        current_start = current_start + len(current_chunk)
            else:
                # Check if adding paragraph exceeds target
                if len(current_chunk) // 4 + para_tokens <= self.target_chunk_size:
                    current_chunk += para
                elif len(current_chunk) // 4 + para_tokens <= self.max_chunk_size:
                    # Exceeds target but under max - add it
                    current_chunk += para
                else:
                    # Would exceed max, start new chunk
                    if current_chunk.strip():
                        chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
                    current_chunk = para
                    current_start = current_start + len(current_chunk)
        
        if current_chunk.strip():
            chunks.append((current_chunk, current_start, current_start + len(current_chunk)))
        
        return chunks
    
    def _split_by_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = self.sentence_endings.split(text)
        return [s.strip() + " " for s in sentences if s.strip()]
    
    def _chunk_code(self, content: str) -> List[Tuple[str, int, int]]:
        """Chunk code respecting function/class boundaries"""
        chunks = []
        
        # Split by function/class definitions (Python-style)
        # This is a simplified approach - could be enhanced with AST parsing
        definition_pattern = re.compile(
            r'^((?:async\s+)?(?:def|class)\s+\w+.*?:)',
            re.MULTILINE
        )
        
        definitions = list(definition_pattern.finditer(content))
        
        if not definitions:
            # No clear structure, fall back to text chunking
            return self._chunk_text(content)
        
        current_start = 0
        for i, match in enumerate(definitions):
            # Get content up to next definition or end
            if i + 1 < len(definitions):
                end = definitions[i + 1].start()
            else:
                end = len(content)
            
            chunk_content = content[match.start():end]
            
            # If chunk is too large, split by blank lines
            if len(chunk_content) // 4 > self.max_chunk_size:
                sub_chunks = self._split_by_paragraphs(chunk_content)
                for sub in sub_chunks:
                    if sub.strip():
                        chunks.append((sub, match.start(), match.start() + len(sub)))
            else:
                chunks.append((chunk_content, match.start(), end))
        
        return chunks
    
    def _is_separator_only(self, content: str) -> bool:
        """Check if content is just separators/whitespace"""
        stripped = content.strip()
        if not stripped:
            return True
        
        # Remove common separators
        cleaned = stripped.replace('-', '').replace('=', '').replace('_', '')
        cleaned = cleaned.replace('#', '').replace('*', '').replace('`', '')
        cleaned = cleaned.replace('\n', '').replace(' ', '')
        
        return len(cleaned) < 20
    
    def _extract_section_title(self, content: str) -> Optional[str]:
        """Extract section title from markdown content"""
        match = self.header_pattern.search(content)
        if match:
            return match.group(2).strip()
        return None


# Global instance
_semantic_chunker = None

def get_semantic_chunker(**kwargs) -> SemanticChunker:
    """Get or create semantic chunker instance"""
    global _semantic_chunker
    if _semantic_chunker is None:
        _semantic_chunker = SemanticChunker(**kwargs)
    return _semantic_chunker

