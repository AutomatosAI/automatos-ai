
"""
Multi-Modal Context Processor
=============================

Handles processing of multiple modalities including text, code, structured data,
and metadata for comprehensive context understanding.
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib

# Import mathematical foundations
from ..mathematical_foundations.information_theory import InformationTheory
from ..mathematical_foundations.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class ContentModality(Enum):
    """Types of content modalities"""
    TEXT = "text"
    CODE = "code"
    STRUCTURED_DATA = "structured_data"  # JSON, XML, etc.
    TABULAR = "tabular"  # CSV, tables
    METADATA = "metadata"
    MIXED = "mixed"

class CodeLanguage(Enum):
    """Supported programming languages"""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    JAVA = "java"
    CPP = "cpp"
    CSHARP = "csharp"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    SHELL = "shell"
    YAML = "yaml"
    JSON = "json"
    XML = "xml"
    HTML = "html"
    CSS = "css"
    MARKDOWN = "markdown"
    UNKNOWN = "unknown"

@dataclass
class ProcessedContent:
    """Processed content with modality-specific information"""
    original_content: str
    modality: ContentModality
    processed_text: str  # Cleaned and normalized for embedding
    searchable_text: str  # Optimized for search
    metadata: Dict[str, Any]
    structure: Dict[str, Any]  # Structural information
    entities: List[str]  # Extracted entities
    keywords: List[str]  # Important keywords
    complexity_score: float
    quality_score: float
    embedding_ready: bool

@dataclass
class CodeAnalysis:
    """Code-specific analysis results"""
    language: CodeLanguage
    functions: List[str]
    classes: List[str]
    imports: List[str]
    variables: List[str]
    comments: List[str]
    docstrings: List[str]
    complexity_metrics: Dict[str, float]
    api_calls: List[str]
    patterns: List[str]  # Design patterns, common structures

@dataclass
class StructuredDataAnalysis:
    """Structured data analysis results"""
    data_type: str  # json, xml, yaml, etc.
    schema: Dict[str, Any]
    fields: List[str]
    nested_levels: int
    data_types: Dict[str, str]
    sample_values: Dict[str, Any]
    relationships: List[Tuple[str, str]]  # Field relationships

class MultiModalProcessor:
    """Advanced multi-modal content processor"""
    
    def __init__(self):
        # Mathematical components
        self.info_theory = InformationTheory()
        self.stats = StatisticalAnalysis()
        
        # Language detection patterns
        self.language_patterns = self._load_language_patterns()
        self.code_patterns = self._load_code_patterns()
        
        # Processing statistics
        self.processing_stats = {
            'total_processed': 0,
            'modality_distribution': {},
            'language_distribution': {},
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0}
        }
    
    def process_content(
        self, 
        content: str, 
        source_info: Optional[Dict[str, Any]] = None,
        force_modality: Optional[ContentModality] = None
    ) -> ProcessedContent:
        """Main content processing method"""
        
        self.processing_stats['total_processed'] += 1
        
        # Step 1: Detect content modality
        modality = force_modality or self._detect_modality(content, source_info)
        
        # Step 2: Process based on modality
        if modality == ContentModality.CODE:
            processed = self._process_code_content(content, source_info)
        elif modality == ContentModality.STRUCTURED_DATA:
            processed = self._process_structured_data(content, source_info)
        elif modality == ContentModality.TABULAR:
            processed = self._process_tabular_data(content, source_info)
        elif modality == ContentModality.METADATA:
            processed = self._process_metadata(content, source_info)
        elif modality == ContentModality.MIXED:
            processed = self._process_mixed_content(content, source_info)
        else:  # TEXT
            processed = self._process_text_content(content, source_info)
        
        # Step 3: Calculate quality metrics
        processed.quality_score = self._calculate_quality_score(processed)
        
        # Step 4: Update statistics
        self._update_statistics(modality, processed.quality_score)
        
        return processed
    
    def _detect_modality(self, content: str, source_info: Optional[Dict[str, Any]]) -> ContentModality:
        """Detect the modality of content"""
        
        # Check source info first
        if source_info:
            file_ext = source_info.get('file_extension', '').lower()
            mime_type = source_info.get('mime_type', '')
            
            # File extension based detection
            code_extensions = {
                '.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs', '.php', '.rb', '.swift'
            }
            structured_extensions = {
                '.json', '.xml', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.config'
            }
            tabular_extensions = {
                '.csv', '.tsv', '.xls', '.xlsx'
            }
            
            if file_ext in code_extensions:
                return ContentModality.CODE
            elif file_ext in structured_extensions:
                return ContentModality.STRUCTURED_DATA
            elif file_ext in tabular_extensions:
                return ContentModality.TABULAR
        
        # Content-based detection
        content_trimmed = content.strip()
        
        # Check for structured data patterns
        if self._is_structured_data(content_trimmed):
            return ContentModality.STRUCTURED_DATA
        
        # Check for code patterns
        if self._is_code_content(content_trimmed):
            return ContentModality.CODE
        
        # Check for tabular data
        if self._is_tabular_data(content_trimmed):
            return ContentModality.TABULAR
        
        # Check for mixed content
        if self._is_mixed_content(content_trimmed):
            return ContentModality.MIXED
        
        # Default to text
        return ContentModality.TEXT
    
    def _process_text_content(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process pure text content"""
        
        # Clean and normalize
        processed_text = self._clean_text(content)
        
        # Extract entities and keywords
        entities = self._extract_text_entities(processed_text)
        keywords = self._extract_text_keywords(processed_text)
        
        # Create searchable text
        searchable_text = self._create_searchable_text(processed_text, keywords)
        
        # Calculate complexity
        complexity = self._calculate_text_complexity(processed_text)
        
        # Build metadata
        metadata = {
            'word_count': len(processed_text.split()),
            'char_count': len(processed_text),
            'sentence_count': len(re.split(r'[.!?]+', processed_text)),
            'paragraph_count': len(processed_text.split('\n\n')),
            'entropy': self.info_theory.calculate_entropy(processed_text),
            'language': self._detect_text_language(processed_text)
        }
        
        if source_info:
            metadata.update(source_info)
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.TEXT,
            processed_text=processed_text,
            searchable_text=searchable_text,
            metadata=metadata,
            structure={'type': 'plain_text'},
            entities=entities,
            keywords=keywords,
            complexity_score=complexity,
            quality_score=0.0,  # Will be calculated later
            embedding_ready=True
        )
    
    def _process_code_content(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process code content"""
        
        # Detect programming language
        language = self._detect_code_language(content, source_info)
        
        # Analyze code structure
        code_analysis = self._analyze_code_structure(content, language)
        
        # Create processed text (code + comments + docstrings)
        processed_text = self._create_code_processed_text(content, code_analysis)
        
        # Extract entities (functions, classes, variables)
        entities = (code_analysis.functions + code_analysis.classes + 
                   code_analysis.variables + code_analysis.imports)
        
        # Extract keywords (including language keywords and domain terms)
        keywords = self._extract_code_keywords(content, code_analysis)
        
        # Create searchable text
        searchable_text = self._create_code_searchable_text(processed_text, code_analysis)
        
        # Calculate complexity
        complexity = self._calculate_code_complexity(code_analysis)
        
        # Build metadata
        metadata = {
            'language': language.value,
            'function_count': len(code_analysis.functions),
            'class_count': len(code_analysis.classes),
            'import_count': len(code_analysis.imports),
            'comment_ratio': len(' '.join(code_analysis.comments)) / max(len(content), 1),
            'lines_of_code': len([line for line in content.split('\n') if line.strip()]),
            'cyclomatic_complexity': code_analysis.complexity_metrics.get('cyclomatic', 0)
        }
        
        if source_info:
            metadata.update(source_info)
        
        # Structure information
        structure = {
            'type': 'code',
            'language': language.value,
            'functions': code_analysis.functions,
            'classes': code_analysis.classes,
            'imports': code_analysis.imports,
            'patterns': code_analysis.patterns
        }
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.CODE,
            processed_text=processed_text,
            searchable_text=searchable_text,
            metadata=metadata,
            structure=structure,
            entities=entities,
            keywords=keywords,
            complexity_score=complexity,
            quality_score=0.0,
            embedding_ready=True
        )
    
    def _process_structured_data(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process structured data (JSON, XML, YAML, etc.)"""
        
        # Analyze structure
        structure_analysis = self._analyze_structured_data(content, source_info)
        
        # Extract text content from structured data
        extracted_text = self._extract_text_from_structured_data(content, structure_analysis)
        
        # Process the extracted text
        processed_text = self._clean_text(extracted_text)
        
        # Extract entities (field names, values)
        entities = structure_analysis.fields + list(structure_analysis.sample_values.keys())
        
        # Extract keywords from field names and text values
        keywords = self._extract_structured_keywords(structure_analysis, extracted_text)
        
        # Create searchable text
        searchable_text = self._create_structured_searchable_text(processed_text, structure_analysis)
        
        # Calculate complexity based on nesting and field count
        complexity = self._calculate_structured_complexity(structure_analysis)
        
        # Build metadata
        metadata = {
            'data_type': structure_analysis.data_type,
            'field_count': len(structure_analysis.fields),
            'nested_levels': structure_analysis.nested_levels,
            'total_values': len(structure_analysis.sample_values),
            'schema_complexity': complexity
        }
        
        if source_info:
            metadata.update(source_info)
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.STRUCTURED_DATA,
            processed_text=processed_text,
            searchable_text=searchable_text,
            metadata=metadata,
            structure={
                'type': 'structured_data',
                'data_type': structure_analysis.data_type,
                'schema': structure_analysis.schema,
                'fields': structure_analysis.fields,
                'relationships': structure_analysis.relationships
            },
            entities=entities,
            keywords=keywords,
            complexity_score=complexity,
            quality_score=0.0,
            embedding_ready=True
        )
    
    def _process_tabular_data(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process tabular data (CSV, TSV, etc.)"""
        
        # Parse tabular structure
        rows, headers = self._parse_tabular_data(content)
        
        # Extract text from cells
        all_text = []
        for row in rows:
            all_text.extend([str(cell) for cell in row if str(cell).strip()])
        
        extracted_text = ' '.join(all_text)
        processed_text = self._clean_text(extracted_text)
        
        # Entities are column headers and unique values
        entities = headers.copy()
        
        # Add sample unique values from each column
        for i, header in enumerate(headers):
            column_values = set()
            for row in rows:
                if i < len(row) and str(row[i]).strip():
                    column_values.add(str(row[i]).strip())
                if len(column_values) >= 5:  # Limit to 5 samples per column
                    break
            entities.extend(list(column_values))
        
        # Extract keywords
        keywords = self._extract_tabular_keywords(headers, rows)
        
        # Create searchable text
        searchable_text = f"{' '.join(headers)} {processed_text}"
        
        # Calculate complexity based on dimensions and data variety
        complexity = self._calculate_tabular_complexity(rows, headers)
        
        # Build metadata
        metadata = {
            'row_count': len(rows),
            'column_count': len(headers),
            'total_cells': len(rows) * len(headers),
            'headers': headers,
            'data_types': self._infer_column_types(rows, headers)
        }
        
        if source_info:
            metadata.update(source_info)
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.TABULAR,
            processed_text=processed_text,
            searchable_text=searchable_text,
            metadata=metadata,
            structure={
                'type': 'tabular',
                'headers': headers,
                'row_count': len(rows),
                'column_count': len(headers)
            },
            entities=entities,
            keywords=keywords,
            complexity_score=complexity,
            quality_score=0.0,
            embedding_ready=True
        )
    
    def _process_metadata(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process metadata content"""
        
        # Parse metadata (assume key-value format)
        metadata_dict = self._parse_metadata(content)
        
        # Extract text from metadata values
        text_values = []
        for key, value in metadata_dict.items():
            if isinstance(value, str) and len(value.strip()) > 0:
                text_values.append(f"{key}: {value}")
        
        extracted_text = ' '.join(text_values)
        processed_text = self._clean_text(extracted_text)
        
        # Entities are metadata keys and important values
        entities = list(metadata_dict.keys())
        
        # Extract keywords from keys and values
        keywords = self._extract_metadata_keywords(metadata_dict)
        
        # Create searchable text
        searchable_text = processed_text
        
        # Simple complexity based on number of fields
        complexity = min(len(metadata_dict) / 20, 1.0)
        
        # Build metadata
        metadata = {
            'field_count': len(metadata_dict),
            'metadata_fields': list(metadata_dict.keys())
        }
        
        if source_info:
            metadata.update(source_info)
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.METADATA,
            processed_text=processed_text,
            searchable_text=searchable_text,
            metadata=metadata,
            structure={
                'type': 'metadata',
                'fields': list(metadata_dict.keys())
            },
            entities=entities,
            keywords=keywords,
            complexity_score=complexity,
            quality_score=0.0,
            embedding_ready=True
        )
    
    def _process_mixed_content(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process mixed content (combination of different modalities)"""
        
        # Split content into different modalities
        content_segments = self._segment_mixed_content(content)
        
        # Process each segment
        processed_segments = []
        all_entities = []
        all_keywords = []
        combined_text = []
        
        for segment_content, segment_modality in content_segments:
            if len(segment_content.strip()) < 10:  # Skip very short segments
                continue
                
            segment_processed = self.process_content(segment_content, source_info, segment_modality)
            processed_segments.append(segment_processed)
            
            all_entities.extend(segment_processed.entities)
            all_keywords.extend(segment_processed.keywords)
            combined_text.append(segment_processed.processed_text)
        
        # Combine results
        processed_text = ' '.join(combined_text)
        
        # Deduplicate entities and keywords
        entities = list(set(all_entities))
        keywords = list(set(all_keywords))
        
        # Create searchable text
        searchable_text = processed_text
        
        # Calculate overall complexity
        segment_complexities = [seg.complexity_score for seg in processed_segments]
        complexity = sum(segment_complexities) / max(len(segment_complexities), 1)
        
        # Build metadata
        modality_distribution = {}
        for segment in processed_segments:
            modality = segment.modality.value
            modality_distribution[modality] = modality_distribution.get(modality, 0) + 1
        
        metadata = {
            'segment_count': len(processed_segments),
            'modality_distribution': modality_distribution,
            'total_length': len(content)
        }
        
        if source_info:
            metadata.update(source_info)
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.MIXED,
            processed_text=processed_text,
            searchable_text=searchable_text,
            metadata=metadata,
            structure={
                'type': 'mixed',
                'segments': len(processed_segments),
                'modalities': list(modality_distribution.keys())
            },
            entities=entities,
            keywords=keywords,
            complexity_score=complexity,
            quality_score=0.0,
            embedding_ready=True
        )
    
    # Detection helper methods
    
    def _is_structured_data(self, content: str) -> bool:
        """Check if content is structured data"""
        stripped = content.strip()
        
        # JSON detection
        if (stripped.startswith('{') and stripped.endswith('}')) or \
           (stripped.startswith('[') and stripped.endswith(']')):
            try:
                json.loads(content)
                return True
            except:
                pass
        
        # XML detection
        if stripped.startswith('<') and stripped.endswith('>'):
            return True
        
        # YAML detection (simple heuristic)
        if re.search(r'^\s*[\w-]+\s*:', content, re.MULTILINE):
            return True
        
        return False
    
    def _is_code_content(self, content: str) -> bool:
        """Check if content is code"""
        
        # Check for common code patterns
        code_indicators = [
            r'def\s+\w+\s*\(',  # Python function
            r'function\s+\w+\s*\(',  # JavaScript function
            r'class\s+\w+',  # Class definition
            r'import\s+\w+',  # Import statement
            r'#include\s*<',  # C++ include
            r'public\s+class\s+\w+',  # Java class
            r'^\s*if\s*\(',  # If statement
            r'^\s*for\s*\(',  # For loop
            r'^\s*while\s*\(',  # While loop
            r'=>',  # Arrow function
            r'\{[\s\S]*\}',  # Code blocks
        ]
        
        code_score = 0
        for pattern in code_indicators:
            if re.search(pattern, content, re.MULTILINE | re.IGNORECASE):
                code_score += 1
        
        # Also check for high ratio of special characters typical in code
        special_chars = len(re.findall(r'[{}();=<>!&|]', content))
        total_chars = len(content)
        special_ratio = special_chars / max(total_chars, 1)
        
        return code_score >= 2 or special_ratio > 0.1
    
    def _is_tabular_data(self, content: str) -> bool:
        """Check if content is tabular data"""
        
        lines = content.split('\n')
        if len(lines) < 2:
            return False
        
        # Check for consistent delimiters
        delimiters = [',', '\t', ';', '|']
        
        for delimiter in delimiters:
            delimiter_counts = [line.count(delimiter) for line in lines if line.strip()]
            if len(set(delimiter_counts)) == 1 and delimiter_counts[0] > 0:
                return True
        
        return False
    
    def _is_mixed_content(self, content: str) -> bool:
        """Check if content contains mixed modalities"""
        
        # Simple heuristic: if content has multiple modality indicators
        indicators = 0
        
        if self._is_code_content(content):
            indicators += 1
        if self._is_structured_data(content):
            indicators += 1
        if self._is_tabular_data(content):
            indicators += 1
        
        # Also check for markdown-like mixed content
        if '```' in content or re.search(r'^\s*#', content, re.MULTILINE):
            indicators += 1
        
        return indicators >= 2
    
    # Additional helper methods would go here...
    # (Due to length limits, I'm including key methods but would implement all helpers in production)
    
    def _load_language_patterns(self) -> Dict[str, List[str]]:
        """Load programming language detection patterns"""
        return {
            'python': [r'def\s+\w+', r'import\s+\w+', r'from\s+\w+\s+import', r'if\s+__name__\s*=='],
            'javascript': [r'function\s+\w+', r'var\s+\w+', r'const\s+\w+', r'=>', r'require\('],
            'java': [r'public\s+class', r'public\s+static\s+void\s+main', r'import\s+java'],
            'cpp': [r'#include\s*<', r'int\s+main\s*\(', r'std::', r'using\s+namespace'],
            'sql': [r'SELECT\s+', r'FROM\s+\w+', r'WHERE\s+', r'INSERT\s+INTO']
        }
    
    def _load_code_patterns(self) -> Dict[str, Dict[str, str]]:
        """Load code analysis patterns"""
        return {
            'function_patterns': {
                'python': r'def\s+(\w+)\s*\(',
                'javascript': r'function\s+(\w+)\s*\(',
                'java': r'(?:public|private|protected)?\s*(?:static)?\s*\w+\s+(\w+)\s*\('
            },
            'class_patterns': {
                'python': r'class\s+(\w+)',
                'javascript': r'class\s+(\w+)',
                'java': r'(?:public|private|protected)?\s*class\s+(\w+)'
            }
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        cleaned = re.sub(r'\s+', ' ', text.strip())
        # Remove very long sequences of repeated characters
        cleaned = re.sub(r'(.)\1{10,}', r'\1\1\1', cleaned)
        return cleaned
    
    def _calculate_quality_score(self, processed: ProcessedContent) -> float:
        """Calculate content quality score"""
        
        score = 0.5  # Base score
        
        # Content length factor
        content_length = len(processed.processed_text)
        if content_length > 100:
            score += 0.1
        if content_length > 500:
            score += 0.1
        
        # Entity and keyword richness
        if len(processed.entities) > 3:
            score += 0.1
        if len(processed.keywords) > 5:
            score += 0.1
        
        # Complexity factor (moderate complexity is good)
        if 0.3 <= processed.complexity_score <= 0.7:
            score += 0.2
        
        return min(score, 1.0)
    
    def _update_statistics(self, modality: ContentModality, quality_score: float):
        """Update processing statistics"""
        
        # Update modality distribution
        modality_key = modality.value
        self.processing_stats['modality_distribution'][modality_key] = (
            self.processing_stats['modality_distribution'].get(modality_key, 0) + 1
        )
        
        # Update quality distribution
        if quality_score >= 0.7:
            self.processing_stats['quality_distribution']['high'] += 1
        elif quality_score >= 0.4:
            self.processing_stats['quality_distribution']['medium'] += 1
        else:
            self.processing_stats['quality_distribution']['low'] += 1
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()
    
    # Placeholder methods for complex operations
    # (These would be fully implemented in production)
    
    def _detect_code_language(self, content: str, source_info: Optional[Dict[str, Any]]) -> CodeLanguage:
        """Detect programming language"""
        # Implementation would use pattern matching and heuristics
        return CodeLanguage.PYTHON  # Placeholder
    
    def _analyze_code_structure(self, content: str, language: CodeLanguage) -> CodeAnalysis:
        """Analyze code structure"""
        # Implementation would parse code and extract structural information
        return CodeAnalysis(
            language=language,
            functions=[],
            classes=[],
            imports=[],
            variables=[],
            comments=[],
            docstrings=[],
            complexity_metrics={},
            api_calls=[],
            patterns=[]
        )
    
    def _analyze_structured_data(self, content: str, source_info: Optional[Dict[str, Any]]) -> StructuredDataAnalysis:
        """Analyze structured data"""
        # Implementation would parse and analyze structured data
        return StructuredDataAnalysis(
            data_type="json",
            schema={},
            fields=[],
            nested_levels=0,
            data_types={},
            sample_values={},
            relationships=[]
        )
    
    # Additional placeholder methods...
    def _extract_text_entities(self, text: str) -> List[str]:
        return []
    
    def _extract_text_keywords(self, text: str) -> List[str]:
        return []
    
    def _calculate_text_complexity(self, text: str) -> float:
        return 0.5
    
    def _detect_text_language(self, text: str) -> str:
        return "english"
    
    # ... other placeholder methods
