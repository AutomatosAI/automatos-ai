"""Block document validation with field-level errors (PRD-167 S2).

Wraps Pydantic validation so callers (API, generation service, agent tools) get a
structured list of field-level errors instead of a raw exception or a silently
coerced document. PRD-167 S2 acceptance: "malformed blocks rejected with field-level
errors — no silent swallow".
"""

from __future__ import annotations

from typing import Any, Dict, List, Set, Union

from pydantic import ValidationError

from .schema import (
    BlockDocument,
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
    {"heading", "text", "table", "image", "variable", "page_break", "section"}
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
        return BlockDocument.model_validate(payload)
    except ValidationError as exc:
        raise BlockValidationError(_format_errors(exc)) from exc


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
        if isinstance(block, VariableBlock):
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


__all__ = ["BlockValidationError", "validate_blocks", "collect_variable_paths"]
