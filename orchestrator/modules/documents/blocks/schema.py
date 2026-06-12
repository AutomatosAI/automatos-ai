"""Canonical block schema for document templates (PRD-167 S2).

This is the **storage + render contract** for a template body. The Plate editor (S5)
serialises to/from this schema via a thin adapter; the renderers (``html_renderer``,
``docx_renderer``) consume it. It is deliberately **editor-independent** — swap the
editor, keep the schema and renderers.

Block types (PRD-167 S2): ``heading``, ``text``, ``table``, ``image`` (incl. logo via
``source="brand_logo"``), ``variable``, ``page_break``, ``section``.

Inline content is a list of *runs*: ``text`` runs (with marks) and ``variable`` runs —
the chips that resolve from profiles/workspace/brand/date at render time.

``extra="forbid"`` on every model means a malformed block fails with a *field-level*
Pydantic error rather than being silently coerced or dropped (PRD-167 S2: "no silent
swallow").
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Bump when the schema changes shape in a non-additive way; stored alongside blocks
# so a future migration can detect and upgrade old documents.
SCHEMA_VERSION = 1

# Text decoration marks permitted on an inline text run.
Mark = Literal["bold", "italic", "underline", "strike", "code"]


class _Base(BaseModel):
    # Reject unknown keys with a field-level error instead of swallowing them.
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# Inline content (lives inside headings, text blocks and table cells)
# ---------------------------------------------------------------------------


class TextRun(_Base):
    """A run of literal text with optional formatting marks."""

    type: Literal["text"] = "text"
    text: str
    marks: List[Mark] = Field(default_factory=list)


class VariableRun(_Base):
    """An inline variable chip, e.g. ``{{user.name}}``, resolved at render time.

    ``path`` is a dotted address into the variable catalog (``user.*``,
    ``company.*``, ``brand.*``, ``date.*``). ``fallback`` is used only when the
    template author explicitly allows a default; otherwise an unresolved path is an
    error surfaced to the caller (PRD-167 S3 policy — never a silent blank).
    """

    type: Literal["variable"] = "variable"
    path: str
    fallback: Optional[str] = None


Inline = Annotated[Union[TextRun, VariableRun], Field(discriminator="type")]


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------


class HeadingBlock(_Base):
    type: Literal["heading"] = "heading"
    id: str
    level: int = Field(ge=1, le=6)
    content: List[Inline] = Field(default_factory=list)


class TextBlock(_Base):
    type: Literal["text"] = "text"
    id: str
    content: List[Inline] = Field(default_factory=list)


class TableBlock(_Base):
    """A table. ``rows`` is row → cell → inline-content. When ``header`` is true the
    first row is styled as the header row."""

    type: Literal["table"] = "table"
    id: str
    header: bool = True
    rows: List[List[List[Inline]]] = Field(default_factory=list)


class ImageBlock(_Base):
    """An image block.

    ``source``:
      - ``url``        — ``src`` is an http(s) URL (subject to the WeasyPrint SSRF
                         allowlist from PRD-156 S4 at render time).
      - ``upload``     — ``src`` is a workspace storage path/key.
      - ``brand_logo`` — resolved from the workspace brand kit; ``src`` ignored.
    """

    type: Literal["image"] = "image"
    id: str
    source: Literal["url", "upload", "brand_logo"] = "url"
    src: Optional[str] = None
    alt: str = ""
    width_mm: Optional[float] = Field(default=None, gt=0, le=500)

    @model_validator(mode="after")
    def _src_required_unless_brand_logo(self) -> "ImageBlock":
        if self.source != "brand_logo" and not self.src:
            raise ValueError("image block requires 'src' unless source is 'brand_logo'")
        return self


class VariableBlock(_Base):
    """A block-level variable insert (whole resolved value occupies its own block).

    Inline chips use :class:`VariableRun`; this is for cases where an entire
    paragraph/value is a single variable.
    """

    type: Literal["variable"] = "variable"
    id: str
    path: str
    fallback: Optional[str] = None


class PageBreakBlock(_Base):
    type: Literal["page_break"] = "page_break"
    id: str


class SectionBlock(_Base):
    """A titled grouping container. Renders its (optional) title as a heading and then
    its child blocks. Sections may nest."""

    type: Literal["section"] = "section"
    id: str
    title: Optional[str] = None
    children: List["Block"] = Field(default_factory=list)


Block = Annotated[
    Union[
        HeadingBlock,
        TextBlock,
        TableBlock,
        ImageBlock,
        VariableBlock,
        PageBreakBlock,
        SectionBlock,
    ],
    Field(discriminator="type"),
]


class BlockDocument(_Base):
    """The full template body as stored in ``document_templates.blocks`` JSONB."""

    version: int = SCHEMA_VERSION
    blocks: List[Block] = Field(default_factory=list)


# Resolve the forward reference in SectionBlock.children -> Block.
SectionBlock.model_rebuild()
BlockDocument.model_rebuild()


__all__ = [
    "SCHEMA_VERSION",
    "Mark",
    "TextRun",
    "VariableRun",
    "Inline",
    "HeadingBlock",
    "TextBlock",
    "TableBlock",
    "ImageBlock",
    "VariableBlock",
    "PageBreakBlock",
    "SectionBlock",
    "Block",
    "BlockDocument",
]
