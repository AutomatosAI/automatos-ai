"""PRD-261 — the reader refuses what the browser would silently leave out of a package's look files.

The browser takes a theme's ``boot/prepaint.json`` through its own gate before it paints anything
(``frontend/lib/theme-host/boot-sanitize.ts`` ``sanitizePrepaint``; ``mobile_tokens.boot_table`` is the
server's port), and its look runtime builds the theme's look spec from the kept table alone
(``frontend/lib/theme-host/looks/spec.ts`` ``lookSpecOf``): one choice knob for each table under
``knobs``, one number knob for each ``numeric`` rule's knob. What the import grammar accepts but is lost
there with no error, :func:`first_paint_errors` refuses:

* what the gate drops of the table (``css_value.prepaint_drop_errors``): a table, a pair table or a
  numeric rule whose knob name it refuses (``BAD_PREPAINT``), and a cell's value it refuses
  (``BAD_CSS_VALUE``): a gradient, another function it never runs, a character it refuses. Every value
  built on a dropped one with ``var()`` is invalid at computed-value time, so the theme paints less and
  nothing says why;
* ``defaults`` that don't fit what the gate keeps (``PREPAINT_DEFAULTS_UNFIT``): the browser then drops the
  whole table, so nothing is painted from it and the theme has no looks;
* a ``looks/schema.json`` knob the kept table never paints (``KNOB_NOT_PAINTED``): the look spec leaves it
  out, so every look that has it is refused as carrying a field it doesn't know, the presets included;
* a knob the kept table paints that the schema doesn't declare (``KNOB_NOT_IN_SCHEMA``): the look spec
  needs it in every look, and no look may carry it, so every look fails the same way.

All of them look only at a table the grammar accepted (``css_value.prepaint_errors`` refuses the rest on
its own). The boot read (``serve.theme_prepaint``) never runs them, so an installed package keeps booting.
``frontend/lib/theme-host/first-paint.ts`` is the TypeScript twin, over the browser's own
``sanitizePrepaint`` and ``lookSpecOf``; the corpus's ``firstPaint`` cases hold both to the same sentences.
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Tuple

from services.themes.contract import ThemeError, echo, parse_json, theme_error
from services.themes.css_value import PREPAINT_PATH, prepaint_drop_errors, prepaint_errors
from services.themes.look_schema import SCHEMA_PATH, LookSchema
from services.themes.mobile_tokens import BootTable, boot_table

KNOB_NOT_PAINTED = "KNOB_NOT_PAINTED"
KNOB_NOT_IN_SCHEMA = "KNOB_NOT_IN_SCHEMA"
PREPAINT_DEFAULTS_UNFIT = "PREPAINT_DEFAULTS_UNFIT"
FIRST_PAINT_ERROR_CODES: Tuple[str, ...] = (
    "BAD_PREPAINT", "BAD_CSS_VALUE", PREPAINT_DEFAULTS_UNFIT, KNOB_NOT_PAINTED, KNOB_NOT_IN_SCHEMA,
)
SPEC = (
    f'the browser builds a theme\'s looks from {PREPAINT_PATH} alone (a choice knob for each table under "knobs", '
    'a number knob for each "numeric" rule)'
)


def spec_knobs(table: BootTable) -> List[str]:
    """``lookSpecOf``'s knobs, in its order: each table's under ``knobs``, then each numeric rule's knob once."""
    numbers = [rule.knob for rule in table.numeric]
    return list(table.knobs) + [knob for at, knob in enumerate(numbers) if knob not in table.knobs and knob not in numbers[:at]]


def _not_painted(name: str) -> ThemeError:
    knob = echo(name)
    where = f"{SCHEMA_PATH}#knobs.{knob}"
    return theme_error(KNOB_NOT_PAINTED, where, (
        f'{where}: "{knob}" is a knob the look table never paints: {SPEC}, so its look spec leaves "{knob}" out '
        "and every look fails, the presets included. Paint it there, or take it out of the schema."
    ))


def _not_in_schema(name: str, table: BootTable, document: Mapping[str, Any]) -> ThemeError:
    knob = echo(name)
    rules = document.get("numeric") if isinstance(document.get("numeric"), list) else []
    at = next((index for index, rule in enumerate(rules) if isinstance(rule, dict) and rule.get("knob") == name), 0)
    where = f"{PREPAINT_PATH}#knobs.{knob}" if name in table.knobs else f"{PREPAINT_PATH}#numeric[{at}].knob"
    return theme_error(KNOB_NOT_IN_SCHEMA, where, (
        f'{where}: "{knob}" is a knob the table paints that {SCHEMA_PATH} doesn\'t declare: {SPEC}, so its look spec '
        f'needs "{knob}" in every look and no look may carry it, and every look fails, the presets included. Declare '
        "it in the schema, or take it out of the table."
    ))


def _unfit() -> ThemeError:
    where = f"{PREPAINT_PATH}#defaults"
    return theme_error(PREPAINT_DEFAULTS_UNFIT, where, (
        f'{where}: "defaults" don\'t fit what the browser\'s first paint keeps of the table (an option of every knob '
        "table, a cell of every pair table, and for every numeric rule a value in range and a colour it keeps), so it "
        "drops the whole table: nothing is painted from it and the theme has no looks."
    ))


def unpainted_knob_errors(schema: LookSchema, table: Any) -> List[ThemeError]:
    """A ``KNOB_NOT_PAINTED`` for each knob of ``schema`` the table, as the browser keeps it, never paints."""
    kept = boot_table(table)
    if kept is None:
        return []
    painted = spec_knobs(kept)
    return [_not_painted(name) for name in schema.knobs if name not in painted]


def unknown_knob_errors(schema: LookSchema, table: Any) -> List[ThemeError]:
    """A ``KNOB_NOT_IN_SCHEMA`` for each knob the table, as the browser keeps it, paints and ``schema`` lacks."""
    kept = boot_table(table)
    if kept is None:
        return []
    return [_not_in_schema(name, kept, table) for name in spec_knobs(kept) if name not in schema.knobs]


def first_paint_table_errors(schema: Optional[LookSchema], table: Any) -> List[ThemeError]:
    """Every refusal above for one table and the package's checked look schema (None: no knob is held to it)."""
    if not isinstance(table, dict) or prepaint_errors(table):
        return []
    dropped = prepaint_drop_errors(table)
    if boot_table(table) is None:
        return dropped + [_unfit()]
    knobs = [] if schema is None else unpainted_knob_errors(schema, table) + unknown_knob_errors(schema, table)
    return dropped + knobs


def first_paint_errors(schema: Optional[LookSchema], files: Mapping[str, bytes]) -> List[ThemeError]:
    """What the browser would leave out of the package's first paint and looks.

    ``schema`` is the package's checked look schema (``look_schema.look_file_errors``), None when it has
    none that passes; ``files`` the package's bytes by path. Empty without a table, or with one that
    isn't JSON (the reader refuses that on its own).
    """
    raw = files.get(PREPAINT_PATH)
    if raw is None:
        return []
    table, error = parse_json(raw, PREPAINT_PATH)
    return [] if error is not None else first_paint_table_errors(schema, table)
