"""
Data models for the Document Generation module (PRD-63).
"""

from dataclasses import dataclass, field
from typing import List, Optional


class UnresolvedDeliverableError(Exception):
    """A generated document still contains unresolved/unknown template variables.

    Raised by ``DocumentGenerationService.generate`` at finalisation (P2-09 S3):
    a client-facing Deliverable with visible ``[[variable]]`` markers must be
    BLOCKED, never delivered. ``unresolved`` are known catalog paths that
    resolved empty (e.g. no ``company.address`` on file); ``unknown`` are
    authoring errors — paths that are not in the variable catalog at all.
    """

    def __init__(self, unresolved: Optional[List[str]] = None, unknown: Optional[List[str]] = None):
        self.unresolved = list(unresolved or [])
        self.unknown = list(unknown or [])
        parts = []
        if self.unresolved:
            parts.append(f"unresolved (no value on file): {', '.join(self.unresolved)}")
        if self.unknown:
            parts.append(f"unknown (not in the variable catalog): {', '.join(self.unknown)}")
        super().__init__(
            "Document blocked at finalisation — template variables did not resolve: "
            + "; ".join(parts)
            + ". Supply the missing values (brand kit / business profile / data.*) "
            "or fix the template's variable paths, then regenerate."
        )


@dataclass
class GeneratedDocument:
    """Result of a document generation operation."""
    path: str
    format: str  # pdf, docx, xlsx
    filename: str
    size: int  # bytes
    download_url: Optional[str] = None
    preview_url: Optional[str] = None
    content: Optional[str] = None  # Markdown content for live widget display
    # P2-09 S3: render honesty, captured off the block render instead of being
    # discarded behind a log line. Non-empty lists block finalisation in
    # ``DocumentGenerationService.generate`` (UnresolvedDeliverableError).
    unresolved: List[str] = field(default_factory=list)  # known paths, empty value
    unknown: List[str] = field(default_factory=list)     # paths not in the catalog
    # P2-09 S4: which render lane produced the file — "block" (canonical block
    # renderer, incl. the no-template brand-aware fallback) or "legacy" (Jinja
    # HTML / uploaded-docx). None when no template render applies (xlsx).
    # Persisted on the Deliverable ``extra`` so lane coverage is a tracked number.
    template_lane: Optional[str] = None
