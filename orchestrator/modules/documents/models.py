"""
Data models for the Document Generation module (PRD-63).
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GeneratedDocument:
    """Result of a document generation operation."""
    path: str
    format: str  # pdf, docx, xlsx
    filename: str
    size: int  # bytes
    download_url: Optional[str] = None
    preview_url: Optional[str] = None
