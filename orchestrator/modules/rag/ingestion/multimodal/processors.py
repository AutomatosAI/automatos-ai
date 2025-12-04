"""
Multimodal Content Processors
==============================

Implements specialized processors for different content modalities:
- Tables: Extract and structure tabular data
- Images: Extract, describe, and analyze visual content
- Formulas: Parse and understand mathematical expressions
- Diagrams: Analyze flowcharts and diagrams

Research-Informed Development:
Multimodal processing concepts informed by RAG-Anything (MIT License, HKUDS 2024).
Implementation is original Automatos code built on our Context Engineering framework

Integrates with Automatos Context Engineering framework.
"""

import logging
import re
import hashlib
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import base64
from io import BytesIO

# PDF and image processing
import pdfplumber
from PIL import Image
import pytesseract

# Table extraction
import camelot  # For table extraction from PDFs
import pandas as pd

logger = logging.getLogger(__name__)


class ContentModality(Enum):
    """Types of content modalities"""
    TEXT = "text"
    TABLE = "table"
    IMAGE = "image"
    FORMULA = "formula"
    DIAGRAM = "diagram"
    CHART = "chart"
    CODE = "code"
    MIXED = "mixed"


@dataclass
class ProcessedContent:
    """Processed content with modality-specific information"""
    modality: ContentModality
    original_content: Any  # Original data
    processed_text: str  # Text representation for embedding
    searchable_text: str  # Optimized for search
    metadata: Dict[str, Any]
    quality_score: float  # 0.0-1.0
    confidence_score: float  # 0.0-1.0
    embeddings_ready: bool = True


@dataclass
class TableExtraction:
    """Extracted table with multiple representations"""
    headers: List[str]
    rows: List[List[str]]
    data_types: Dict[str, str]
    markdown: str
    csv: str
    json: List[Dict[str, Any]]
    row_count: int
    column_count: int
    has_header: bool
    caption: Optional[str] = None
    page_number: Optional[int] = None
    position: Optional[Dict[str, float]] = None  # {x, y, width, height}
    confidence: float = 0.0


@dataclass
class ImageExtraction:
    """Extracted image with analysis"""
    format: str
    width: int
    height: int
    file_size_bytes: int
    description: str  # AI-generated description
    alt_text: Optional[str] = None
    caption: Optional[str] = None
    detected_text: Optional[str] = None  # OCR result
    detected_objects: List[str] = None
    image_data: bytes = None
    thumbnail_data: bytes = None
    page_number: Optional[int] = None
    position: Optional[Dict[str, float]] = None
    confidence: float = 0.0


@dataclass
class FormulaExtraction:
    """Extracted mathematical formula"""
    latex: str
    ascii_math: str
    variables: List[str]
    operators: List[str]
    formula_type: str  # 'equation', 'expression', 'inequality'
    domain: str  # 'algebra', 'calculus', 'statistics', etc.
    complexity: str  # 'basic', 'intermediate', 'advanced'
    page_number: Optional[int] = None
    confidence: float = 0.0


class TableProcessor:
    """
    Extract and process tables from documents.
    Supports multiple extraction methods for different table types.
    """
    
    def __init__(self):
        self.extraction_methods = ['lattice', 'stream']  # Camelot methods
        
    def extract_tables_from_pdf(
        self, 
        pdf_path: str, 
        pages: Optional[str] = 'all'
    ) -> List[TableExtraction]:
        """
        Extract tables from PDF using Camelot.
        
        Args:
            pdf_path: Path to PDF file
            pages: Pages to extract from ('all', '1,2,3', '1-5')
        
        Returns:
            List of extracted tables
        """
        extracted_tables = []
        
        for method in self.extraction_methods:
            try:
                # Extract using Camelot
                tables = camelot.read_pdf(
                    pdf_path, 
                    pages=pages, 
                    flavor=method,
                    suppress_stdout=True
                )
                
                for i, table in enumerate(tables):
                    try:
                        extraction = self._process_camelot_table(table, i)
                        extracted_tables.append(extraction)
                        logger.info(f"Extracted table {i+1} with {method} method: {extraction.row_count}x{extraction.column_count}")
                    except Exception as e:
                        logger.warning(f"Failed to process table {i} with {method}: {e}")
                
                # If we got tables with lattice, skip stream
                if tables and method == 'lattice':
                    break
                    
            except Exception as e:
                logger.warning(f"Table extraction with {method} failed: {e}")
        
        return extracted_tables
    
    def _process_camelot_table(
        self, 
        table: Any,  # Camelot Table object
        index: int
    ) -> TableExtraction:
        """Process a Camelot table into our format"""
        
        df = table.df
        
        # Detect if first row is header
        has_header = self._detect_header_row(df)
        
        if has_header:
            headers = df.iloc[0].tolist()
            data_rows = df.iloc[1:].values.tolist()
        else:
            headers = [f"Column_{i+1}" for i in range(len(df.columns))]
            data_rows = df.values.tolist()
        
        # Infer data types
        data_types = self._infer_column_types(headers, data_rows)
        
        # Generate multiple representations
        markdown = self._to_markdown(headers, data_rows)
        csv = self._to_csv(headers, data_rows)
        json_data = self._to_json(headers, data_rows)
        
        # Get position from table.parsing_report
        position = None
        if hasattr(table, '_bbox'):
            bbox = table._bbox
            position = {
                'x': bbox[0],
                'y': bbox[1],
                'width': bbox[2] - bbox[0],
                'height': bbox[3] - bbox[1]
            }
        
        return TableExtraction(
            headers=headers,
            rows=data_rows,
            data_types=data_types,
            markdown=markdown,
            csv=csv,
            json=json_data,
            row_count=len(data_rows),
            column_count=len(headers),
            has_header=has_header,
            page_number=table.page,
            position=position,
            confidence=table.accuracy / 100.0 if hasattr(table, 'accuracy') else 0.8
        )
    
    def _detect_header_row(self, df: pd.DataFrame) -> bool:
        """Detect if first row is likely a header"""
        if df.empty or len(df) < 2:
            return False
        
        first_row = df.iloc[0]
        second_row = df.iloc[1]
        
        # Check if first row has more non-numeric values
        first_non_numeric = sum(not str(v).replace('.','').isdigit() for v in first_row)
        second_non_numeric = sum(not str(v).replace('.','').isdigit() for v in second_row)
        
        return first_non_numeric > second_non_numeric
    
    def _infer_column_types(
        self, 
        headers: List[str], 
        rows: List[List[str]]
    ) -> Dict[str, str]:
        """Infer data type for each column"""
        types = {}
        
        for i, header in enumerate(headers):
            column_values = [row[i] for row in rows if i < len(row)]
            types[header] = self._infer_type_from_values(column_values)
        
        return types
    
    def _infer_type_from_values(self, values: List[str]) -> str:
        """Infer type from column values"""
        if not values:
            return 'text'
        
        # Check if all values are numeric
        numeric_count = sum(1 for v in values if str(v).replace('.','').replace('-','').isdigit())
        
        if numeric_count / len(values) > 0.8:
            # Check if float or int
            float_count = sum(1 for v in values if '.' in str(v))
            return 'float' if float_count > 0 else 'integer'
        
        # Check for dates
        date_count = sum(1 for v in values if self._is_date(str(v)))
        if date_count / len(values) > 0.8:
            return 'date'
        
        return 'text'
    
    def _is_date(self, value: str) -> bool:
        """Simple date detection"""
        date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',
            r'\d{4}-\d{2}-\d{2}',
            r'\d{1,2}-\d{1,2}-\d{2,4}'
        ]
        return any(re.match(pattern, value) for pattern in date_patterns)
    
    def _to_markdown(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert table to Markdown format"""
        md = '| ' + ' | '.join(headers) + ' |\n'
        md += '| ' + ' | '.join(['---'] * len(headers)) + ' |\n'
        
        for row in rows:
            md += '| ' + ' | '.join(str(v) for v in row) + ' |\n'
        
        return md
    
    def _to_csv(self, headers: List[str], rows: List[List[str]]) -> str:
        """Convert table to CSV format"""
        lines = [','.join(f'"{h}"' for h in headers)]
        for row in rows:
            lines.append(','.join(f'"{str(v)}"' for v in row))
        return '\n'.join(lines)
    
    def _to_json(self, headers: List[str], rows: List[List[str]]) -> List[Dict[str, Any]]:
        """Convert table to JSON array of objects"""
        return [
            {headers[i]: row[i] for i in range(min(len(headers), len(row)))}
            for row in rows
        ]


class ImageProcessor:
    """
    Extract and process images from documents.
    Generates AI descriptions and performs OCR text extraction.
    """
    
    def __init__(self, openai_client=None):
        """
        Initialize image processor.
        
        Args:
            openai_client: OpenAI client for image description (GPT-4V)
        """
        self.openai_client = openai_client
        self.thumbnail_size = (200, 200)
    
    def extract_images_from_pdf(
        self, 
        pdf_path: str, 
        pages: Optional[List[int]] = None
    ) -> List[ImageExtraction]:
        """
        Extract images from PDF.
        
        Args:
            pdf_path: Path to PDF file
            pages: Specific pages to extract from (None = all pages)
        
        Returns:
            List of extracted images
        """
        extracted_images = []
        
        with pdfplumber.open(pdf_path) as pdf:
            pages_to_process = pages or range(len(pdf.pages))
            
            for page_num in pages_to_process:
                if page_num >= len(pdf.pages):
                    continue
                
                page = pdf.pages[page_num]
                
                # Extract images from page
                if hasattr(page, 'images'):
                    for i, img_info in enumerate(page.images):
                        try:
                            extraction = self._process_pdf_image(
                                page, img_info, page_num, i
                            )
                            extracted_images.append(extraction)
                            logger.info(f"Extracted image from page {page_num+1}, index {i}")
                        except Exception as e:
                            logger.warning(f"Failed to extract image: {e}")
        
        return extracted_images
    
    def _process_pdf_image(
        self, 
        page: Any, 
        img_info: Dict, 
        page_num: int, 
        index: int
    ) -> ImageExtraction:
        """Process a single PDF image"""
        
        # Get image object from page
        # Note: This is simplified - actual implementation depends on PDF structure
        x0, y0, x1, y1 = img_info['x0'], img_info['top'], img_info['x1'], img_info['bottom']
        
        # Crop image from page (simplified)
        img = page.within_bbox((x0, y0, x1, y1)).to_image()
        pil_image = img.original
        
        # Get image properties
        width, height = pil_image.size
        format_name = pil_image.format or 'PNG'
        
        # Convert to bytes
        img_buffer = BytesIO()
        pil_image.save(img_buffer, format=format_name)
        image_data = img_buffer.getvalue()
        file_size = len(image_data)
        
        # Create thumbnail
        thumbnail = pil_image.copy()
        thumbnail.thumbnail(self.thumbnail_size, Image.Resampling.LANCZOS)
        thumb_buffer = BytesIO()
        thumbnail.save(thumb_buffer, format='PNG')
        thumbnail_data = thumb_buffer.getvalue()
        
        # OCR text extraction
        detected_text = self._extract_text_with_ocr(pil_image)
        
        # Generate description (if OpenAI client available)
        description = self._generate_image_description(pil_image)
        
        # Position
        position = {
            'x': x0,
            'y': y0,
            'width': x1 - x0,
            'height': y1 - y0
        }
        
        return ImageExtraction(
            format=format_name,
            width=width,
            height=height,
            file_size_bytes=file_size,
            description=description,
            detected_text=detected_text,
            image_data=image_data,
            thumbnail_data=thumbnail_data,
            page_number=page_num + 1,
            position=position,
            confidence=0.9 if description else 0.7
        )
    
    def _extract_text_with_ocr(self, image: Image.Image) -> Optional[str]:
        """Extract text from image using OCR"""
        try:
            text = pytesseract.image_to_string(image)
            return text.strip() if text.strip() else None
        except Exception as e:
            logger.warning(f"OCR extraction failed: {e}")
            return None
    
    def _generate_image_description(self, image: Image.Image) -> str:
        """
        Generate AI description of image using GPT-4V.
        Falls back to basic description if OpenAI not available.
        """
        if not self.openai_client:
            return f"Image {image.size[0]}x{image.size[1]} pixels"
        
        try:
            # Convert image to base64
            buffer = BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            # Call GPT-4V
            response = self.openai_client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Describe this image in 2-3 sentences. Focus on what's important for understanding the document context."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{img_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=200
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"Image description generation failed: {e}")
            return f"Image {image.size[0]}x{image.size[1]} pixels"


class FormulaProcessor:
    """
    Extract and process mathematical formulas.
    Parses LaTeX and extracts mathematical structure.
    """
    
    def __init__(self):
        self.formula_patterns = {
            'inline': r'\$[^\$]+\$',
            'display': r'\$\$[^\$]+\$\$',
            'equation': r'\\begin\{equation\}.*?\\end\{equation\}',
        }
    
    def extract_formulas_from_text(self, text: str) -> List[FormulaExtraction]:
        """
        Extract LaTeX formulas from text.
        
        Args:
            text: Text containing LaTeX formulas
        
        Returns:
            List of extracted formulas
        """
        # Clean text first - fix double characters from PDF extraction
        import re
        cleaned_text = re.sub(r'(.)\1+', r'\1', text)  # Remove duplicate characters
        
        formulas = []
        
        for formula_type, pattern in self.formula_patterns.items():
            matches = re.finditer(pattern, cleaned_text, re.DOTALL)
            for match in matches:
                latex = match.group(0)
                formula = self._process_formula(latex, formula_type)
                formulas.append(formula)
        
        return formulas
    
    def _process_formula(self, latex: str, formula_type: str) -> FormulaExtraction:
        """Process a single formula"""
        
        # Clean LaTeX
        cleaned = latex.strip('$').strip()
        
        # Extract variables
        variables = self._extract_variables(cleaned)
        
        # Extract operators
        operators = self._extract_operators(cleaned)
        
        # Convert to ASCII math (simplified)
        ascii_math = self._latex_to_ascii(cleaned)
        
        # Determine domain
        domain = self._determine_domain(cleaned)
        
        # Determine complexity
        complexity = self._assess_complexity(cleaned)
        
        return FormulaExtraction(
            latex=cleaned,
            ascii_math=ascii_math,
            variables=variables,
            operators=operators,
            formula_type=formula_type,
            domain=domain,
            complexity=complexity,
            confidence=0.85
        )
    
    def _extract_variables(self, latex: str) -> List[str]:
        """Extract variables from LaTeX"""
        # Simple pattern for single-letter variables
        variables = re.findall(r'\b[a-zA-Z]\b', latex)
        return list(set(variables))
    
    def _extract_operators(self, latex: str) -> List[str]:
        """Extract mathematical operators"""
        operators = []
        operator_map = {
            '+': 'plus',
            '-': 'minus',
            '*': 'multiply',
            '/': 'divide',
            '=': 'equals',
            '\\frac': 'fraction',
            '\\int': 'integral',
            '\\sum': 'summation',
            '\\prod': 'product',
            '^': 'exponent',
            '_': 'subscript'
        }
        
        for symbol, name in operator_map.items():
            if symbol in latex:
                operators.append(name)
        
        return operators
    
    def _latex_to_ascii(self, latex: str) -> str:
        """Convert LaTeX to ASCII representation (simplified)"""
        # This is a very basic conversion
        ascii_str = latex.replace('\\frac', '/')
        ascii_str = ascii_str.replace('^', '**')
        ascii_str = ascii_str.replace('\\times', '*')
        ascii_str = ascii_str.replace('\\cdot', '*')
        return ascii_str
    
    def _determine_domain(self, latex: str) -> str:
        """Determine mathematical domain"""
        if any(op in latex for op in ['\\int', '\\diff', '\\partial']):
            return 'calculus'
        elif any(op in latex for op in ['\\sum', '\\prod']):
            return 'algebra'
        elif any(op in latex for op in ['\\mu', '\\sigma', 'P(']):
            return 'statistics'
        else:
            return 'general'
    
    def _assess_complexity(self, latex: str) -> str:
        """Assess formula complexity"""
        # Count nesting levels
        nesting = max(latex.count('{'), latex.count('['))
        
        if nesting <= 2:
            return 'basic'
        elif nesting <= 5:
            return 'intermediate'
        else:
            return 'advanced'


class MultimodalDocumentProcessor:
    """
    Unified processor that orchestrates all modality processors.
    Main interface for multimodal document processing.
    """
    
    def __init__(self, openai_client=None):
        """
        Initialize multimodal processor.
        
        Args:
            openai_client: OpenAI client for AI-powered features
        """
        self.table_processor = TableProcessor()
        self.image_processor = ImageProcessor(openai_client)
        self.formula_processor = FormulaProcessor()
    
    def process_pdf_multimodal(
        self, 
        pdf_path: str, 
        extract_tables: bool = True,
        extract_images: bool = True,
        extract_formulas: bool = True
    ) -> Dict[str, List[Any]]:
        """
        Extract all multimodal content from PDF.
        
        Args:
            pdf_path: Path to PDF file
            extract_tables: Whether to extract tables
            extract_images: Whether to extract images
            extract_formulas: Whether to extract formulas
        
        Returns:
            Dictionary with extracted content by modality
        """
        results = {
            'tables': [],
            'images': [],
            'formulas': [],
            'text': ''
        }
        
        # Extract text (for formula detection)
        with pdfplumber.open(pdf_path) as pdf:
            full_text = '\n'.join([page.extract_text() or '' for page in pdf.pages])
            results['text'] = full_text
        
        # Extract tables
        if extract_tables:
            try:
                tables = self.table_processor.extract_tables_from_pdf(pdf_path)
                results['tables'] = tables
                logger.info(f"Extracted {len(tables)} tables from PDF")
            except Exception as e:
                logger.error(f"Table extraction failed: {e}")
        
        # Extract images
        if extract_images:
            try:
                images = self.image_processor.extract_images_from_pdf(pdf_path)
                results['images'] = images
                logger.info(f"Extracted {len(images)} images from PDF")
            except Exception as e:
                logger.error(f"Image extraction failed: {e}")
        
        # Extract formulas
        if extract_formulas:
            try:
                formulas = self.formula_processor.extract_formulas_from_text(full_text)
                results['formulas'] = formulas
                logger.info(f"Extracted {len(formulas)} formulas from PDF")
            except Exception as e:
                logger.error(f"Formula extraction failed: {e}")
        
        return results
    
    def detect_modality(self, content: Any, context: Dict[str, Any] = None) -> ContentModality:
        """
        Detect content modality from content and context.
        
        Args:
            content: Content to analyze
            context: Additional context (file_type, metadata, etc.)
        
        Returns:
            Detected modality
        """
        if context and 'modality' in context:
            return ContentModality(context['modality'])
        
        # Simple heuristics
        if isinstance(content, dict):
            if 'headers' in content and 'rows' in content:
                return ContentModality.TABLE
            elif 'image_data' in content or 'format' in content:
                return ContentModality.IMAGE
            elif 'latex' in content:
                return ContentModality.FORMULA
        
        if isinstance(content, str):
            if '$' in content or '\\begin{equation}' in content:
                return ContentModality.FORMULA
            elif '|' in content and '---' in content:
                return ContentModality.TABLE
        
        return ContentModality.TEXT


# Convenience function for easy importing
def create_multimodal_processor():
    """
    Factory function to create multimodal processor
    
    Returns:
        Multi modal processor instance
    """
    # Uses centralized LLM manager - no API key needed
    from core.llm import create_llm_manager
    
    return MultimodalDocumentProcessor()
