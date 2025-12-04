
"""
Multi-Modal Context Processor
=============================

Handles processing of multiple modalities including text, code, structured data,
and metadata for comprehensive context understanding.

Migrated to: modules/search/retrieval/multimodal_processor.py
"""

import json
import logging
import re
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import hashlib

# Import mathematical foundations from search module
from modules.search.optimization.information_theory import InformationTheory
from modules.search.optimization.statistical_analysis import StatisticalAnalysis

logger = logging.getLogger(__name__)

class ContentModality(Enum):
    """Types of content modalities"""
    TEXT = "text"
    CODE = "code"
    STRUCTURED_DATA = "structured_data"
    TABULAR = "tabular"
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
    processed_text: str
    searchable_text: str
    metadata: Dict[str, Any]
    structure: Dict[str, Any]
    entities: List[str]
    keywords: List[str]
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
    patterns: List[str]

@dataclass
class StructuredDataAnalysis:
    """Structured data analysis results"""
    data_type: str
    schema: Dict[str, Any]
    fields: List[str]
    nested_levels: int
    data_types: Dict[str, str]
    sample_values: Dict[str, Any]
    relationships: List[Tuple[str, str]]

class MultiModalProcessor:
    """Advanced multi-modal content processor"""
    
    def __init__(self):
        self.info_theory = InformationTheory()
        self.stats = StatisticalAnalysis()
        self.language_patterns = self._load_language_patterns()
        self.code_patterns = self._load_code_patterns()
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
        modality = force_modality or self._detect_modality(content, source_info)
        
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
        else:
            processed = self._process_text_content(content, source_info)
        
        processed.quality_score = self._calculate_quality_score(processed)
        self._update_statistics(modality, processed.quality_score)
        return processed
    
    def _detect_modality(self, content: str, source_info: Optional[Dict[str, Any]]) -> ContentModality:
        """Detect the modality of content"""
        if source_info:
            file_ext = source_info.get('file_extension', '').lower()
            code_extensions = {'.py', '.js', '.java', '.cpp', '.c', '.h', '.cs', '.go', '.rs'}
            structured_extensions = {'.json', '.xml', '.yaml', '.yml', '.toml'}
            tabular_extensions = {'.csv', '.tsv', '.xls', '.xlsx'}
            
            if file_ext in code_extensions:
                return ContentModality.CODE
            elif file_ext in structured_extensions:
                return ContentModality.STRUCTURED_DATA
            elif file_ext in tabular_extensions:
                return ContentModality.TABULAR
        
        content_trimmed = content.strip()
        if self._is_structured_data(content_trimmed):
            return ContentModality.STRUCTURED_DATA
        if self._is_code_content(content_trimmed):
            return ContentModality.CODE
        if self._is_tabular_data(content_trimmed):
            return ContentModality.TABULAR
        if self._is_mixed_content(content_trimmed):
            return ContentModality.MIXED
        return ContentModality.TEXT
    
    def _process_text_content(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process pure text content"""
        processed_text = self._clean_text(content)
        entities = self._extract_text_entities(processed_text)
        keywords = self._extract_text_keywords(processed_text)
        searchable_text = self._create_searchable_text(processed_text, keywords)
        complexity = self._calculate_text_complexity(processed_text)
        
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
            quality_score=0.0,
            embedding_ready=True
        )
    
    def _process_code_content(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process code content"""
        language = self._detect_code_language(content, source_info)
        code_analysis = self._analyze_code_structure(content, language)
        processed_text = self._create_code_processed_text(content, code_analysis)
        entities = (code_analysis.functions + code_analysis.classes + 
                   code_analysis.variables + code_analysis.imports)
        keywords = self._extract_code_keywords(content, code_analysis)
        searchable_text = self._create_code_searchable_text(processed_text, code_analysis)
        complexity = self._calculate_code_complexity(code_analysis)
        
        metadata = {
            'language': language.value,
            'function_count': len(code_analysis.functions),
            'class_count': len(code_analysis.classes),
            'import_count': len(code_analysis.imports),
            'lines_of_code': len([line for line in content.split('\n') if line.strip()]),
        }
        if source_info:
            metadata.update(source_info)
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.CODE,
            processed_text=processed_text,
            searchable_text=searchable_text,
            metadata=metadata,
            structure={'type': 'code', 'language': language.value},
            entities=entities,
            keywords=keywords,
            complexity_score=complexity,
            quality_score=0.0,
            embedding_ready=True
        )
    
    def _process_structured_data(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process structured data"""
        structure_analysis = self._analyze_structured_data(content, source_info)
        extracted_text = self._extract_text_from_structured_data(content, structure_analysis)
        processed_text = self._clean_text(extracted_text)
        entities = structure_analysis.fields + list(structure_analysis.sample_values.keys())
        keywords = self._extract_structured_keywords(structure_analysis, extracted_text)
        searchable_text = self._create_structured_searchable_text(processed_text, structure_analysis)
        complexity = self._calculate_structured_complexity(structure_analysis)
        
        metadata = {
            'data_type': structure_analysis.data_type,
            'field_count': len(structure_analysis.fields),
            'nested_levels': structure_analysis.nested_levels,
        }
        if source_info:
            metadata.update(source_info)
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.STRUCTURED_DATA,
            processed_text=processed_text,
            searchable_text=searchable_text,
            metadata=metadata,
            structure={'type': 'structured_data', 'data_type': structure_analysis.data_type},
            entities=entities,
            keywords=keywords,
            complexity_score=complexity,
            quality_score=0.0,
            embedding_ready=True
        )
    
    def _process_tabular_data(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process tabular data"""
        rows, headers = self._parse_tabular_data(content)
        all_text = [str(cell) for row in rows for cell in row if str(cell).strip()]
        extracted_text = ' '.join(all_text)
        processed_text = self._clean_text(extracted_text)
        entities = headers.copy()
        keywords = self._extract_tabular_keywords(headers, rows)
        searchable_text = f"{' '.join(headers)} {processed_text}"
        complexity = self._calculate_tabular_complexity(rows, headers)
        
        metadata = {
            'row_count': len(rows),
            'column_count': len(headers),
            'headers': headers,
        }
        if source_info:
            metadata.update(source_info)
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.TABULAR,
            processed_text=processed_text,
            searchable_text=searchable_text,
            metadata=metadata,
            structure={'type': 'tabular', 'headers': headers, 'row_count': len(rows)},
            entities=entities,
            keywords=keywords,
            complexity_score=complexity,
            quality_score=0.0,
            embedding_ready=True
        )
    
    def _process_metadata(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process metadata content"""
        metadata_dict = self._parse_metadata(content)
        text_values = [f"{k}: {v}" for k, v in metadata_dict.items() if isinstance(v, str)]
        extracted_text = ' '.join(text_values)
        processed_text = self._clean_text(extracted_text)
        entities = list(metadata_dict.keys())
        keywords = self._extract_metadata_keywords(metadata_dict)
        complexity = min(len(metadata_dict) / 20, 1.0)
        
        metadata = {'field_count': len(metadata_dict), 'metadata_fields': list(metadata_dict.keys())}
        if source_info:
            metadata.update(source_info)
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.METADATA,
            processed_text=processed_text,
            searchable_text=processed_text,
            metadata=metadata,
            structure={'type': 'metadata', 'fields': list(metadata_dict.keys())},
            entities=entities,
            keywords=keywords,
            complexity_score=complexity,
            quality_score=0.0,
            embedding_ready=True
        )
    
    def _process_mixed_content(self, content: str, source_info: Optional[Dict[str, Any]]) -> ProcessedContent:
        """Process mixed content"""
        processed_text = self._clean_text(content)
        entities = self._extract_text_entities(processed_text)
        keywords = self._extract_text_keywords(processed_text)
        
        metadata = {'total_length': len(content), 'is_mixed': True}
        if source_info:
            metadata.update(source_info)
        
        return ProcessedContent(
            original_content=content,
            modality=ContentModality.MIXED,
            processed_text=processed_text,
            searchable_text=processed_text,
            metadata=metadata,
            structure={'type': 'mixed'},
            entities=entities,
            keywords=keywords,
            complexity_score=0.5,
            quality_score=0.0,
            embedding_ready=True
        )
    
    # Detection helpers
    def _is_structured_data(self, content: str) -> bool:
        stripped = content.strip()
        if (stripped.startswith('{') and stripped.endswith('}')) or \
           (stripped.startswith('[') and stripped.endswith(']')):
            try:
                json.loads(content)
                return True
            except:
                pass
        if stripped.startswith('<') and stripped.endswith('>'):
            return True
        return False
    
    def _is_code_content(self, content: str) -> bool:
        code_indicators = [
            r'def\s+\w+\s*\(', r'function\s+\w+\s*\(', r'class\s+\w+',
            r'import\s+\w+', r'#include\s*<', r'public\s+class\s+\w+'
        ]
        code_score = sum(1 for p in code_indicators if re.search(p, content, re.MULTILINE | re.IGNORECASE))
        return code_score >= 2
    
    def _is_tabular_data(self, content: str) -> bool:
        lines = content.split('\n')
        if len(lines) < 2:
            return False
        for delimiter in [',', '\t', ';', '|']:
            counts = [line.count(delimiter) for line in lines if line.strip()]
            if len(set(counts)) == 1 and counts[0] > 0:
                return True
        return False
    
    def _is_mixed_content(self, content: str) -> bool:
        return '```' in content or re.search(r'^\s*#', content, re.MULTILINE)
    
    # Helper methods
    def _load_language_patterns(self) -> Dict[str, List[str]]:
        return {
            'python': [r'def\s+\w+', r'import\s+\w+'],
            'javascript': [r'function\s+\w+', r'const\s+\w+', r'=>'],
            'java': [r'public\s+class', r'import\s+java'],
        }
    
    def _load_code_patterns(self) -> Dict[str, Dict[str, str]]:
        return {
            'function_patterns': {'python': r'def\s+(\w+)\s*\('},
            'class_patterns': {'python': r'class\s+(\w+)'}
        }
    
    def _clean_text(self, text: str) -> str:
        cleaned = re.sub(r'\s+', ' ', text.strip())
        return re.sub(r'(.)\1{10,}', r'\1\1\1', cleaned)
    
    def _calculate_quality_score(self, processed: ProcessedContent) -> float:
        score = 0.5
        if len(processed.processed_text) > 100: score += 0.1
        if len(processed.processed_text) > 500: score += 0.1
        if len(processed.entities) > 3: score += 0.1
        if len(processed.keywords) > 5: score += 0.1
        if 0.3 <= processed.complexity_score <= 0.7: score += 0.2
        return min(score, 1.0)
    
    def _update_statistics(self, modality: ContentModality, quality_score: float):
        modality_key = modality.value
        self.processing_stats['modality_distribution'][modality_key] = \
            self.processing_stats['modality_distribution'].get(modality_key, 0) + 1
        if quality_score >= 0.7:
            self.processing_stats['quality_distribution']['high'] += 1
        elif quality_score >= 0.4:
            self.processing_stats['quality_distribution']['medium'] += 1
        else:
            self.processing_stats['quality_distribution']['low'] += 1
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        return self.processing_stats.copy()
    
    # Placeholder implementations
    def _detect_code_language(self, content: str, source_info: Optional[Dict[str, Any]]) -> CodeLanguage:
        return CodeLanguage.PYTHON
    
    def _analyze_code_structure(self, content: str, language: CodeLanguage) -> CodeAnalysis:
        return CodeAnalysis(language=language, functions=[], classes=[], imports=[],
                          variables=[], comments=[], docstrings=[], 
                          complexity_metrics={}, api_calls=[], patterns=[])
    
    def _analyze_structured_data(self, content: str, source_info: Optional[Dict[str, Any]]) -> StructuredDataAnalysis:
        return StructuredDataAnalysis(data_type="json", schema={}, fields=[],
                                     nested_levels=0, data_types={}, sample_values={}, relationships=[])
    
    def _create_code_processed_text(self, content: str, analysis: CodeAnalysis) -> str:
        return content
    
    def _create_code_searchable_text(self, text: str, analysis: CodeAnalysis) -> str:
        return text
    
    def _extract_code_keywords(self, content: str, analysis: CodeAnalysis) -> List[str]:
        return []
    
    def _calculate_code_complexity(self, analysis: CodeAnalysis) -> float:
        return 0.5
    
    def _extract_text_from_structured_data(self, content: str, analysis: StructuredDataAnalysis) -> str:
        return content
    
    def _extract_structured_keywords(self, analysis: StructuredDataAnalysis, text: str) -> List[str]:
        return []
    
    def _create_structured_searchable_text(self, text: str, analysis: StructuredDataAnalysis) -> str:
        return text
    
    def _calculate_structured_complexity(self, analysis: StructuredDataAnalysis) -> float:
        return 0.5
    
    def _parse_tabular_data(self, content: str) -> Tuple[List[List[str]], List[str]]:
        lines = content.strip().split('\n')
        if not lines:
            return [], []
        headers = lines[0].split(',')
        rows = [line.split(',') for line in lines[1:]]
        return rows, headers
    
    def _extract_tabular_keywords(self, headers: List[str], rows: List[List[str]]) -> List[str]:
        return headers
    
    def _calculate_tabular_complexity(self, rows: List[List[str]], headers: List[str]) -> float:
        return min(len(rows) * len(headers) / 1000, 1.0)
    
    def _infer_column_types(self, rows: List[List[str]], headers: List[str]) -> Dict[str, str]:
        return {h: 'string' for h in headers}
    
    def _parse_metadata(self, content: str) -> Dict[str, Any]:
        return {}
    
    def _extract_metadata_keywords(self, metadata: Dict[str, Any]) -> List[str]:
        return list(metadata.keys())
    
    def _extract_text_entities(self, text: str) -> List[str]:
        return []
    
    def _extract_text_keywords(self, text: str) -> List[str]:
        return []
    
    def _create_searchable_text(self, text: str, keywords: List[str]) -> str:
        return text
    
    def _calculate_text_complexity(self, text: str) -> float:
        return 0.5
    
    def _detect_text_language(self, text: str) -> str:
        return "english"

