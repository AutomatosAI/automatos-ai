"""SQL validator — sqlglot AST based (PRD-160 S2).

Replaces the previous regex heuristics with a real parse. Regex could not
reliably see into subqueries/CTEs/UNIONs (the old code blocked them wholesale
as a workaround); the AST validates every SELECT, table and column in the tree.

Security contract (unchanged, now enforced structurally):
  * SELECT-only — any DML/DDL node (INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/
    TRUNCATE/MERGE/GRANT/REVOKE/CALL/COPY/… and ``SELECT … INTO``) is rejected.
  * Single statement only — stacked statements are rejected.
  * Table allowlist — every base table must be in the connection's schema
    metadata (PRD-160 S2: "table allowlist from connection scope").
  * Cross-schema references (``other_schema.table``) are rejected; only the
    default schema (public/dbo/main) is allowed.
  * LIMIT is injected when missing and capped when above ``max_limit``.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

# Schema qualifiers that are NOT treated as a cross-schema escape.
_DEFAULT_SCHEMAS = {"", "public", "dbo", "main"}

# AST node types that must never appear in a read-only query.
_FORBIDDEN_NODES = (
    exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter,
    exp.TruncateTable, exp.Merge, exp.Grant, exp.Command,
)

# Roots that are read-only result expressions.
_READ_ONLY_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


class SQLValidationError(Exception):
    pass


class SQLValidator:
    """Validate (and lightly rewrite) a single read-only SQL statement.

    The public surface is unchanged: ``validate_and_rewrite(sql, schema_metadata)``
    returns ``(safe_sql, warnings)`` or raises :class:`SQLValidationError`.
    """

    def __init__(
        self,
        max_limit: int = 1000,
        strict_column_validation: bool = True,
        dialect: Optional[str] = None,
    ):
        self.max_limit = max_limit
        self.strict_column_validation = strict_column_validation
        self.dialect = dialect

    # -- helpers ----------------------------------------------------------------

    def _build_column_map(self, schema_metadata: Dict[str, Any]) -> Dict[str, Set[str]]:
        """table_name -> set(column_names), lower-cased, from schema metadata."""
        column_map: Dict[str, Set[str]] = {}
        for table in (schema_metadata.get("tables") or []):
            name = (table.get("name") or "").lower()
            cols = {(c.get("name") or "").lower() for c in (table.get("columns") or [])}
            column_map[name] = {c for c in cols if c}
        return column_map

    @staticmethod
    def _cte_names(tree: exp.Expression) -> Set[str]:
        return {(c.alias_or_name or "").lower() for c in tree.find_all(exp.CTE)}

    def _check_tables(
        self,
        tree: exp.Expression,
        schema_metadata: Dict[str, Any],
    ) -> None:
        """Enforce the table allowlist + reject cross-schema references."""
        allowed = {
            (t.get("name") or "").lower() for t in (schema_metadata.get("tables") or [])
        }
        cte_names = self._cte_names(tree)

        for tbl in tree.find_all(exp.Table):
            name = (tbl.name or "").lower()
            schema_qual = (tbl.db or "").lower()
            if not name or name in cte_names:
                continue  # CTE reference, not a base table
            if name not in allowed:
                raise SQLValidationError(
                    f"Reference to unknown table: '{name}'. "
                    f"Available tables: {', '.join(sorted(list(allowed))[:10])}"
                )
            if schema_qual and schema_qual not in _DEFAULT_SCHEMAS:
                raise SQLValidationError(
                    f"Cross-schema reference not allowed: '{schema_qual}.{name}'"
                )

    def _check_columns(
        self,
        tree: exp.Expression,
        schema_metadata: Dict[str, Any],
    ) -> None:
        """Conservatively reject hallucinated *qualified* columns.

        Only flags ``alias.col`` where the alias resolves to a base table that
        declares columns and ``col`` is not among them. Unqualified columns and
        tables with no declared columns are left alone — we never reject on
        absence of metadata (that was the old regex validator's false-positive).
        """
        column_map = self._build_column_map(schema_metadata)
        if not any(column_map.values()):
            return  # no column metadata anywhere → nothing to validate

        # alias / table-name -> base table name
        alias_to_table: Dict[str, str] = {}
        for tbl in tree.find_all(exp.Table):
            base = (tbl.name or "").lower()
            alias_to_table[base] = base
            if tbl.alias:
                alias_to_table[tbl.alias.lower()] = base

        invalid: List[str] = []
        for col in tree.find_all(exp.Column):
            tref = (col.table or "").lower()
            cname = (col.name or "").lower()
            if not tref or cname == "*":
                continue
            base = alias_to_table.get(tref)
            if not base:
                continue  # subquery/CTE alias — not a base table
            known = column_map.get(base)
            if known and cname not in known:
                invalid.append(f"{tref}.{cname}")

        if invalid:
            sample: List[str] = []
            for cols in column_map.values():
                sample.extend(sorted(cols)[:5])
                if len(sample) >= 10:
                    break
            raise SQLValidationError(
                f"Reference to unknown column(s): {', '.join(sorted(set(invalid)))}. "
                f"Valid columns include: {', '.join(sample[:10])}"
            )

    def _apply_limit(self, tree: exp.Expression, reasons: List[str]) -> str:
        """Inject LIMIT when missing, cap it when above ``max_limit``."""
        limit = tree.args.get("limit")
        if limit is None:
            tree = tree.limit(self.max_limit)
            reasons.append(f"LIMIT {self.max_limit} injected")
            return tree.sql(dialect=self.dialect)

        current: Optional[int] = None
        try:
            current = int(limit.expression.name)
        except (AttributeError, ValueError, TypeError):
            current = None
        if current is not None and current > self.max_limit:
            reasons.append(f"LIMIT capped from {current} to {self.max_limit}")
            tree = tree.limit(self.max_limit)
        return tree.sql(dialect=self.dialect)

    # -- public -----------------------------------------------------------------

    def validate_and_rewrite(
        self,
        sql: str,
        schema_metadata: Dict[str, Any] | None = None,
    ) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        raw = (sql or "").strip()
        if not raw:
            raise SQLValidationError("Empty query: no SQL provided")
        stripped = raw.rstrip(";").strip()
        if not stripped:
            raise SQLValidationError("Empty query: no SQL provided")

        # Stacked-statement guard (string-literal aware): after the single
        # trailing ';' is removed, any remaining ';' means multiple statements.
        literal_free = re.sub(r"'[^']*'", "''", stripped)
        literal_free = re.sub(r'"[^"]*"', '""', literal_free)
        if ";" in literal_free:
            raise SQLValidationError("Multiple statements not allowed")

        # Parse to an AST. A parse failure on a non-SELECT is reported as the
        # SELECT-only violation it is, not as an opaque parser error.
        try:
            expressions = sqlglot.parse(stripped, read=self.dialect)
        except Exception as e:  # noqa: BLE001
            if not re.match(r"^\s*SELECT\b", stripped, re.IGNORECASE):
                raise SQLValidationError("Only SELECT statements are allowed")
            raise SQLValidationError(f"Could not parse SQL: {e}")

        expressions = [e for e in expressions if e is not None]
        if len(expressions) > 1:
            raise SQLValidationError("Multiple statements not allowed")
        if not expressions:
            raise SQLValidationError("Only SELECT statements are allowed")
        tree = expressions[0]

        # Root must be a read-only result expression …
        if not isinstance(tree, _READ_ONLY_ROOTS):
            raise SQLValidationError("Only SELECT statements are allowed")
        # … 'SELECT … INTO new_table' writes despite the SELECT root …
        if tree.args.get("into") is not None:
            raise SQLValidationError("Only SELECT statements are allowed")
        # … and no write/DDL/command node may hide anywhere in the tree.
        for node in tree.walk():
            if isinstance(node, _FORBIDDEN_NODES):
                raise SQLValidationError("Only SELECT statements are allowed")

        if schema_metadata:
            self._check_tables(tree, schema_metadata)
            if self.strict_column_validation:
                self._check_columns(tree, schema_metadata)

        safe_sql = self._apply_limit(tree, reasons)
        return safe_sql, reasons
