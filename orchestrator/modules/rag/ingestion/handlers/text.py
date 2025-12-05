"""
Text File Handler
=================

Handler for plain text files (.txt, .md, .py, .js, etc.)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TextContent:
    """Parsed text content"""
    content: str
    metadata: Dict[str, Any]
    line_count: int
    char_count: int
    word_count: int


class TextHandler:
    """Handler for text-based files"""
    
    SUPPORTED_EXTENSIONS = {
        '.txt': 'text',
        '.md': 'markdown',
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.h': 'c',
        '.hpp': 'cpp',
        '.go': 'go',
        '.rs': 'rust',
        '.php': 'php',
        '.rb': 'ruby',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.r': 'r',
        '.sql': 'sql',
        '.sh': 'shell',
        '.bash': 'shell',
        '.zsh': 'shell',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.json': 'json',
        '.xml': 'xml',
        '.html': 'html',
        '.htm': 'html',
        '.css': 'css',
        '.scss': 'scss',
        '.less': 'less',
        '.ini': 'ini',
        '.toml': 'toml',
        '.cfg': 'config',
        '.conf': 'config',
        '.env': 'env',
        '.gitignore': 'gitignore',
        '.dockerfile': 'dockerfile',
    }
    
    def __init__(self):
        self.encodings_to_try = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']
    
    def can_handle(self, file_path: str) -> bool:
        """Check if this handler can process the file"""
        ext = Path(file_path).suffix.lower()
        name = Path(file_path).name.lower()
        
        # Check extension
        if ext in self.SUPPORTED_EXTENSIONS:
            return True
        
        # Check special filenames
        if name in ['dockerfile', 'makefile', 'readme', 'license', 'changelog']:
            return True
        
        return False
    
    def get_content_type(self, file_path: str) -> str:
        """Get the content type for a file"""
        ext = Path(file_path).suffix.lower()
        name = Path(file_path).name.lower()
        
        if ext in self.SUPPORTED_EXTENSIONS:
            return self.SUPPORTED_EXTENSIONS[ext]
        
        if name == 'dockerfile':
            return 'dockerfile'
        if name == 'makefile':
            return 'makefile'
        
        return 'text'
    
    def read_file(self, file_path: str) -> Optional[TextContent]:
        """Read and parse a text file"""
        path = Path(file_path)
        
        if not path.exists():
            logger.error(f"File not found: {file_path}")
            return None
        
        if not path.is_file():
            logger.error(f"Not a file: {file_path}")
            return None
        
        # Try different encodings
        content = None
        used_encoding = None
        
        for encoding in self.encodings_to_try:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()
                used_encoding = encoding
                break
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                return None
        
        if content is None:
            logger.error(f"Could not decode {file_path} with any supported encoding")
            return None
        
        # Calculate stats
        lines = content.split('\n')
        words = content.split()
        
        metadata = {
            'file_path': str(path.absolute()),
            'file_name': path.name,
            'extension': path.suffix.lower(),
            'content_type': self.get_content_type(file_path),
            'encoding': used_encoding,
            'size_bytes': path.stat().st_size,
        }
        
        return TextContent(
            content=content,
            metadata=metadata,
            line_count=len(lines),
            char_count=len(content),
            word_count=len(words)
        )
    
    def extract_sections(self, content: str, content_type: str) -> list:
        """Extract logical sections from content based on type"""
        
        if content_type == 'markdown':
            return self._extract_markdown_sections(content)
        elif content_type in ['python', 'javascript', 'typescript', 'java', 'cpp', 'c', 'go', 'rust']:
            return self._extract_code_sections(content, content_type)
        else:
            return self._extract_paragraph_sections(content)
    
    def _extract_markdown_sections(self, content: str) -> list:
        """Extract sections from markdown using headers"""
        sections = []
        current_section = []
        current_header = None
        
        for line in content.split('\n'):
            if line.startswith('#'):
                if current_section:
                    sections.append({
                        'header': current_header,
                        'content': '\n'.join(current_section)
                    })
                current_header = line.lstrip('#').strip()
                current_section = [line]
            else:
                current_section.append(line)
        
        if current_section:
            sections.append({
                'header': current_header,
                'content': '\n'.join(current_section)
            })
        
        return sections
    
    def _extract_code_sections(self, content: str, content_type: str) -> list:
        """Extract sections from code (functions, classes)"""
        # Simple extraction - split on blank lines and function/class definitions
        sections = []
        current_section = []
        
        for line in content.split('\n'):
            # Detect function/class boundaries
            is_boundary = False
            if content_type == 'python':
                is_boundary = line.startswith('def ') or line.startswith('class ') or line.startswith('async def ')
            elif content_type in ['javascript', 'typescript']:
                is_boundary = 'function ' in line or line.strip().startswith('class ') or '=>' in line
            elif content_type in ['java', 'cpp', 'c']:
                is_boundary = '{' in line and ('(' in line or 'class ' in line)
            
            if is_boundary and current_section:
                sections.append({
                    'header': None,
                    'content': '\n'.join(current_section)
                })
                current_section = []
            
            current_section.append(line)
        
        if current_section:
            sections.append({
                'header': None,
                'content': '\n'.join(current_section)
            })
        
        return sections
    
    def _extract_paragraph_sections(self, content: str) -> list:
        """Extract sections based on paragraphs (blank line separated)"""
        sections = []
        current_paragraph = []
        
        for line in content.split('\n'):
            if line.strip() == '':
                if current_paragraph:
                    sections.append({
                        'header': None,
                        'content': '\n'.join(current_paragraph)
                    })
                    current_paragraph = []
            else:
                current_paragraph.append(line)
        
        if current_paragraph:
            sections.append({
                'header': None,
                'content': '\n'.join(current_paragraph)
            })
        
        return sections

