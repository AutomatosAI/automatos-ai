"""PRD-261 — the reader refuses what the browser would silently leave out of a package's look files.

The browser takes a theme's ``boot/prepaint.json`` through its own gate before it paints anything
(``frontend/lib/theme-host/boot-sanitize.ts`` ``sanitizePrepaint``; ``mobile_tokens.boot_table`` is the
server's port), and its look runtime builds the theme's look spec from the kept table alone
(``frontend/lib/theme-host/looks/spec.ts`` ``lookSpecOf``): one choice knob for each table under
``knobs``, one number knob for each ``numeric`` rule's knob. Two things the import grammar accepts
are lost there with no error, so :func:`first_paint_errors` refuses them:

* a value the gate drops (``css_value.prepaint_drop_errors``, ``BAD_CSS_VALUE``): a gradient, another
  function it never runs, a character it refuses. Every value built on it with ``var()`` is then
  invalid at computed-value time, so the theme paints less and nothing says why;
* a ``looks/schema.json`` knob the kept table never paints (``KNOB_NOT_PAINTED``): the look spec leaves
  it out, so every look that has it is refused as carrying a field it doesn't know, the presets
  included, and the look editor changes nothing.

Both look only at a table the grammar accepted (``css_value.prepaint_errors`` refuses the rest on its
own). A table whose ``defaults`` don't fit what the gate keeps is dropped whole by the browser, so it
has no look spec to hold the schema to, and no ``KNOB_NOT_PAINTED`` is given for it.
"""

from __future__ import annotations

from typing import Any, FrozenSet, List, Mapping, Optional

from services.themes.contract import ThemeError, echo, parse_json, theme_error
from services.themes.css_value import PREPAINT_PATH, prepaint_drop_errors, prepaint_errors
from services.themes.look_schema import SCHEMA_PATH, LookSchema
from services.themes.mobile_tokens import BootTable, boot_table

KNOB_NOT_PAINTED = "KNOB_NOT_PAINTED"


def painted_knobs(table: BootTable) -> FrozenSet[str]:
    """``lookSpecOf``'s knobs: a table's under ``knobs``, then each of its numeric rules' knob."""
    return frozenset(table.knobs) | frozenset(rule.knob for rule in table.numeric)


def _not_painted(name: str) -> ThemeError:
    knob = echo(name)
    where = f"{SCHEMA_PATH}#knobs.{knob}"
    return theme_error(KNOB_NOT_PAINTED, where, (
        f'{where}: "{knob}" is a knob the look table never paints: the browser builds a theme\'s looks from '
        f'{PREPAINT_PATH} alone (a choice knob for each table under "knobs", a number knob for each "numeric" '
        f'rule), so its look spec leaves "{knob}" out and every look fails, the presets included. Paint it '
        'there, or take it out of the schema.'
    ))


def unpainted_knob_errors(schema: LookSchema, table: Any) -> List[ThemeError]:
    """A ``KNOB_NOT_PAINTED`` for each knob of ``schema`` the table, as the browser keeps it, never paints."""
    kept = boot_table(table)
    if kept is None:
        return []
    painted = painted_knobs(kept)
    return [_not_painted(name) for name in schema.knobs if name not in painted]


def first_paint_errors(schema: Optional[LookSchema], files: Mapping[str, bytes]) -> List[ThemeError]:
    """What the browser would leave out of the package's first paint and looks: dropped values, unpainted knobs.

    ``schema`` is the package's checked look schema (``look_schema.look_file_errors``), None when it has
    none that passes; ``files`` the package's bytes by path. Empty without a table, or with one the
    grammar refuses.
    """
    raw = files.get(PREPAINT_PATH)
    if raw is None:
        return []
    table, error = parse_json(raw, PREPAINT_PATH)
    if error is not None or not isinstance(table, dict) or prepaint_errors(table):
        return []
    knobs = [] if schema is None else unpainted_knob_errors(schema, table)
    return prepaint_drop_errors(table) + knobs
