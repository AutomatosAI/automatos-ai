"""What a template asks for and what fills itself, said once for the Studio card and the agents (F345).

A template's fields are of two kinds: the ones the guard will block on when they are
empty (``required_fields``), and the ones with a fallback the template prints instead
(``fallback_fields``, path → the text printed). A ``data_table`` is listed in
``tables`` with its columns, the columns that may stay empty, and whether an empty
table is allowed. The template summary (the Studio gallery) and
``platform_get_template_schema`` (the agents) both read it from here, so the card
and the agent say the same thing about one template.

* A block template: every chip and table in its blocks. A path used both with and
  without a fallback is required (the place without one blocks).
* A legacy template: its ``data_schema``'s required fields (inside objects and list
  rows too) and its array-of-object fields as tables; its fallbacks are the
  ``default('…')`` its Jinja source prints for a field.
* A social template: its variables; one with a default fills itself.

Pure: no IO.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from core.social_templates import is_social_format
from modules.documents.blocks import BlockValidationError, validate_blocks
from modules.documents.blocks.schema import DataTableBlock, SectionBlock, TableBlock, VariableBlock, VariableRun
from modules.documents.blocks.table_cells import QUANTITY_KEY, UNIT_KEY, UNIT_NOTE
from modules.documents.variables.catalog import DYNAMIC_PREFIX, SIGN_OFF_PATH, SIGNER_FALLBACK_TEXT, SIGNER_KEY

DATA = "data"
# ``{{ client.address | default('') }}``: the field and the text printed in its place
# (also ``default('', true)``, which prints it for a blank value too, as F356's starters do).
_JINJA_DEFAULT = re.compile(
    r"""\{\{\s*([A-Za-z_][\w.]*)\s*\|\s*default\(\s*(['"])(.*?)\2\s*(?:,\s*(?:boolean\s*=\s*)?(?:true|True)\s*)?\)"""
)
# A legacy default that reads the brand kit is the platform's, not a data field.
_PLATFORM_ROOTS = ("brand.",)


def _answer(required: set, fallbacks: Dict[str, str], tables: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "required_fields": sorted(required),
        "fallback_fields": {path: text for path, text in sorted(fallbacks.items()) if path not in required},
        "tables": tables,
    }


def _table(block: DataTableBlock) -> Dict[str, Any]:
    """A table's columns, the ones that may stay empty, and (F369) the row keys it reads beside its columns:
    a table with a quantity column prints a row's ``unit`` after it."""
    keys = [column.key for column in block.columns]
    table: Dict[str, Any] = {
        "field": block.path[len(DYNAMIC_PREFIX):],
        "columns": keys,
        "optional_columns": [column.key for column in block.columns if column.optional],
        "required": block.empty_text is None,
    }
    return {**table, "also_reads": {UNIT_KEY: UNIT_NOTE}} if QUANTITY_KEY in keys else table


def _chips(block: Any) -> List[Any]:
    """The variable chips one block holds (sections are walked by the caller)."""
    if isinstance(block, VariableBlock):
        return [block]
    if isinstance(block, TableBlock):
        return [run for row in block.rows for cell in row for run in cell if isinstance(run, VariableRun)]
    return [run for run in getattr(block, "content", None) or [] if isinstance(run, VariableRun)]


def block_requirements(blocks: Any) -> Dict[str, Any]:
    """A block template's required fields, fallbacks and tables ({} lists when it does not validate)."""
    try:
        doc = validate_blocks(blocks)
    except BlockValidationError:
        return _answer(set(), {}, [])
    required: set = set()
    fallbacks: Dict[str, str] = {}
    tables: List[Dict[str, Any]] = []
    pending = list(doc.blocks)
    while pending:
        block = pending.pop(0)
        if isinstance(block, SectionBlock):
            pending = [*block.children, *pending]
        elif isinstance(block, DataTableBlock):
            tables.append(_table(block))
            if block.empty_text is None:
                required.add(block.path)
        for chip in _chips(block):
            if chip.path == SIGN_OFF_PATH:  # F364: a named signer (data.signer) signs in its place
                fallbacks.setdefault(f"{DATA}.{SIGNER_KEY}", SIGNER_FALLBACK_TEXT)
            if chip.fallback is None:
                required.add(chip.path)
            else:
                fallbacks.setdefault(chip.path, chip.fallback)
    return _answer(required, fallbacks, tables)


def _schema_required(schema: Any, prefix: str) -> List[str]:
    """Every required path of a JSON schema, objects' own required keys included."""
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties") or {}
    found: List[str] = []
    for key in schema.get("required") or []:
        found.append(f"{prefix}.{key}")
        found.extend(_schema_required(properties.get(key), f"{prefix}.{key}"))
    return found


def _schema_tables(schema: Any) -> List[Dict[str, Any]]:
    """A legacy schema's lists of objects, as tables."""
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties") or {}
    required = set(schema.get("required") or [])
    tables: List[Dict[str, Any]] = []
    for key, spec in properties.items():
        items = spec.get("items") if isinstance(spec, dict) and spec.get("type") == "array" else None
        if not isinstance(items, dict) or not items.get("properties"):
            continue
        columns = list(items["properties"])
        needed = set(items.get("required") or [])
        tables.append({"field": key, "columns": columns,
                       "optional_columns": [c for c in columns if c not in needed], "required": key in required})
    return tables


def legacy_requirements(schema: Any, source: Optional[str]) -> Dict[str, Any]:
    """A legacy template's required fields (from its schema), fallbacks (from its Jinja) and tables."""
    fallbacks = {
        f"{DATA}.{name}": text
        for name, _quote, text in _JINJA_DEFAULT.findall(source or "")
        if not name.startswith(_PLATFORM_ROOTS)
    }
    return _answer(set(_schema_required(schema, DATA)), fallbacks, _schema_tables(schema))


def social_requirements(blocks: Any) -> Dict[str, Any]:
    """A social template's variables: one without a default is required."""
    schema = blocks.get("variables_schema") if isinstance(blocks, dict) else None
    specs = {name: spec for name, spec in (schema or {}).items() if isinstance(spec, dict)}
    required = {f"{DATA}.{name}" for name, spec in specs.items() if spec.get("default") is None}
    fallbacks = {f"{DATA}.{name}": str(spec["default"]) for name, spec in specs.items() if spec.get("default") is not None}
    return _answer(required, fallbacks, [])


def requirements_of(template: Any) -> Dict[str, Any]:
    """What ``template`` asks for and what fills itself. Pure."""
    blocks = getattr(template, "blocks", None)
    if is_social_format(getattr(template, "format", None)):
        return social_requirements(blocks)
    if blocks:
        return block_requirements(blocks)
    return legacy_requirements(getattr(template, "data_schema", None), getattr(template, "template_content", None))


__all__ = ["block_requirements", "legacy_requirements", "requirements_of", "social_requirements"]
