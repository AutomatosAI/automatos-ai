"""Block document validation with field-level errors (PRD-167 S2).

Wraps Pydantic validation so callers (API, generation service, agent tools) get a
structured list of field-level errors instead of a raw exception or a silently
coerced document. PRD-167 S2 acceptance: "malformed blocks rejected with field-level
errors — no silent swallow".
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Set, Tuple, Union

from pydantic import ValidationError

from .schema import (
    BlockDocument,
    BrandBlock,
    DataTableBlock,
    SectionBlock,
    TableBlock,
    VariableBlock,
    VariableRun,
)


class BlockValidationError(Exception):
    """Raised when a block document fails schema validation.

    ``errors`` is a list of ``{"loc": "blocks.0.level", "msg": "...", "type": "..."}``
    suitable for returning to an editor for inline display.
    """

    def __init__(self, errors: List[Dict[str, str]]):
        self.errors = errors
        super().__init__(f"{len(errors)} block validation error(s)")


# Discriminator tags Pydantic injects into error locations (the Literal `type` values).
# Stripped from `loc` so an editor sees `blocks.0.level`, not `blocks.0.heading.level`.
_DISCRIMINATOR_TAGS = frozenset(
    {"heading", "text", "table", "image", "variable", "data_table", "page_break", "section", "brand"}
)


def _format_errors(exc: ValidationError) -> List[Dict[str, str]]:
    formatted: List[Dict[str, str]] = []
    for err in exc.errors():
        # loc is a tuple like ("blocks", 0, "heading", "level"); join into a dotted
        # path, dropping the discriminated-union tags Pydantic injects.
        loc_parts = [str(p) for p in err.get("loc", ()) if str(p) not in _DISCRIMINATOR_TAGS]
        formatted.append(
            {
                "loc": ".".join(loc_parts) or "(root)",
                "msg": err.get("msg", "invalid"),
                "type": err.get("type", "value_error"),
            }
        )
    return formatted


def validate_blocks(raw: Union[Dict[str, Any], List[Any], None]) -> BlockDocument:
    """Parse and validate raw block JSON into a :class:`BlockDocument`.

    Accepts either the full ``{"version": .., "blocks": [..]}`` envelope or a bare
    list of blocks (which is wrapped). Raises :class:`BlockValidationError` with
    field-level detail on failure.
    """
    if raw is None:
        return BlockDocument()
    payload: Dict[str, Any]
    if isinstance(raw, list):
        payload = {"blocks": raw}
    elif isinstance(raw, dict):
        payload = raw
    else:
        raise BlockValidationError(
            [{"loc": "(root)", "msg": "blocks must be an object or array", "type": "type_error"}]
        )

    try:
        doc = BlockDocument.model_validate(payload)
    except ValidationError as exc:
        raise BlockValidationError(_format_errors(exc)) from exc
    repeated = repeated_brand_parts(doc)
    if repeated:
        raise BlockValidationError(repeated)
    return doc


def _brand_parts(blocks: Sequence[Any], where: str) -> List[Tuple[str, str]]:
    """``(loc, part)`` of every ``brand`` block, sections walked."""
    found: List[Tuple[str, str]] = []
    for index, block in enumerate(blocks):
        if isinstance(block, BrandBlock):
            found.append((f"{where}.{index}.part", block.part))
        elif isinstance(block, SectionBlock):
            found.extend(_brand_parts(block.children, f"{where}.{index}.children"))
    return found


def repeated_brand_parts(doc: BlockDocument) -> List[Dict[str, str]]:
    """A field-level error for each brand board part after its first (PRD-255 US-009).

    Each part is drawn from the kit, and the applications part prints two
    starters to draw them: one of each per document keeps a render's cost bounded."""
    seen: Set[str] = set()
    errors: List[Dict[str, str]] = []
    for loc, part in _brand_parts(doc.blocks, "blocks"):
        if part in seen:
            errors.append({"loc": loc, "msg": f"the brand board's {part} part appears once per document",
                           "type": "value_error"})
        seen.add(part)
    return errors


def collect_variable_paths(doc: BlockDocument) -> Set[str]:
    """Return every variable path referenced anywhere in the document.

    Used to (a) drive the editor's "variables in use" panel and (b) let the resolver
    pre-flight which paths a template needs before rendering.
    """
    paths: Set[str] = set()

    def walk_inline(content: list) -> None:
        for run in content:
            if isinstance(run, VariableRun):
                paths.add(run.path)

    def walk_block(block) -> None:
        if isinstance(block, (VariableBlock, DataTableBlock)):
            paths.add(block.path)
        elif isinstance(block, TableBlock):
            for row in block.rows:
                for cell in row:
                    walk_inline(cell)
        elif isinstance(block, SectionBlock):
            for child in block.children:
                walk_block(child)
        elif hasattr(block, "content"):
            walk_inline(block.content)

    for block in doc.blocks:
        walk_block(block)
    return paths


def collect_list_fields(doc: BlockDocument) -> List[Dict[str, Any]]:
    """The ``data.*`` LIST fields the document expects (PRD-243): one entry per
    ``data_table`` block — ``{"field": "line_items", "columns": ["description", ...]}``
    — so the Studio form and the agent tool schema can say "a list of rows, each with
    these keys" instead of treating it like a scalar chip."""
    found: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    def walk(block) -> None:
        if isinstance(block, DataTableBlock):
            field = block.path[len("data."):]
            if field not in seen:
                seen.add(field)
                found.append({"field": field, "columns": [c.key for c in block.columns]})
        elif isinstance(block, SectionBlock):
            for child in block.children:
                walk(child)

    for block in doc.blocks:
        walk(block)
    return found


__all__ = ["BlockValidationError", "validate_blocks", "collect_variable_paths", "collect_list_fields"]
