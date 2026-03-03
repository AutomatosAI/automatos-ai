import re
import logging
from typing import Dict, Any, List, Tuple, Set, Optional

logger = logging.getLogger(__name__)


class SQLValidationError(Exception):
    pass


class SQLValidator:
    """
    SQL validator with schema validation:
    - SELECT-only (reject INSERT/UPDATE/DELETE/ALTER/TRUNCATE/DROP)
    - Single statement only
    - LIMIT injection/enforcement (max_limit)
    - Table allowlist validation from schema metadata
    - Column validation to prevent hallucinated columns
    """

    DENY_KEYWORDS = [
        r"\bINSERT\b", r"\bUPDATE\b", r"\bDELETE\b", r"\bALTER\b",
        r"\bTRUNCATE\b", r"\bDROP\b", r"\bCREATE\b", r"\bMERGE\b",
        r"\bGRANT\b", r"\bREVOKE\b", r"\bVACUUM\b",
        # PRD-70 FIX-04: Additional deny keywords
        r"\bRETURNING\b",  # Prevents mutation disguised as SELECT
        r"\bCOPY\b",       # Prevents COPY TO/FROM
        r"\bEXECUTE\b",    # Prevents dynamic SQL execution
    ]

    # SQL functions/keywords that look like columns but aren't
    SQL_FUNCTIONS_AND_KEYWORDS = {
        'count', 'sum', 'avg', 'max', 'min', 'distinct', 'as', 'and', 'or',
        'not', 'in', 'is', 'null', 'true', 'false', 'like', 'ilike', 'between',
        'case', 'when', 'then', 'else', 'end', 'coalesce', 'nullif', 'cast',
        'extract', 'date', 'timestamp', 'interval', 'now', 'current_date',
        'current_timestamp', 'upper', 'lower', 'trim', 'substring', 'length',
        'concat', 'replace', 'to_char', 'to_date', 'to_timestamp', 'asc', 'desc',
        'limit', 'offset', 'group', 'by', 'order', 'having', 'where', 'from',
        'join', 'left', 'right', 'inner', 'outer', 'on', 'select', 'all'
    }

    def __init__(self, max_limit: int = 1000, strict_column_validation: bool = True):
        self.max_limit = max_limit
        self.strict_column_validation = strict_column_validation

    def _build_column_map(self, schema_metadata: Dict[str, Any]) -> Dict[str, Set[str]]:
        """Build a map of table_name -> set of column_names from schema metadata."""
        column_map: Dict[str, Set[str]] = {}
        for table in (schema_metadata.get("tables") or []):
            table_name = table.get("name", "").lower()
            columns = {col.get("name", "").lower() for col in (table.get("columns") or [])}
            column_map[table_name] = columns
        return column_map

    def _extract_referenced_tables(self, sql: str) -> List[Tuple[str, Optional[str]]]:
        """
        Extract tables referenced in SQL with their aliases.
        Returns list of (table_name, alias_or_none).
        """
        tables = []
        # Match FROM table [AS alias] or FROM table alias
        from_pattern = r"\bFROM\s+([\w\.\"]+)(?:\s+(?:AS\s+)?(\w+))?"
        join_pattern = r"\bJOIN\s+([\w\.\"]+)(?:\s+(?:AS\s+)?(\w+))?"

        for m in re.finditer(from_pattern, sql, flags=re.IGNORECASE):
            table = (m.group(1) or '').replace('"', '').lower()
            if '.' in table:
                table = table.split('.')[-1]
            alias = (m.group(2) or '').lower() if m.group(2) else None
            if table:
                tables.append((table, alias))

        for m in re.finditer(join_pattern, sql, flags=re.IGNORECASE):
            table = (m.group(1) or '').replace('"', '').lower()
            if '.' in table:
                table = table.split('.')[-1]
            alias = (m.group(2) or '').lower() if m.group(2) else None
            if table:
                tables.append((table, alias))

        return tables

    def _extract_column_references(self, sql: str) -> List[Tuple[Optional[str], str]]:
        """
        Extract column references from SQL.
        Returns list of (table_prefix_or_none, column_name).
        """
        columns = []

        # Pattern for column references: table.column or just column
        # This is a simplified pattern that catches most cases
        col_pattern = r'(?:(\w+)\.)?(\w+)'

        # Remove string literals to avoid false matches
        sql_clean = re.sub(r"'[^']*'", "''", sql)

        # Extract from SELECT clause (between SELECT and FROM)
        select_match = re.search(r'\bSELECT\b(.+?)\bFROM\b', sql_clean, flags=re.IGNORECASE | re.DOTALL)
        if select_match:
            select_clause = select_match.group(1)
            for m in re.finditer(col_pattern, select_clause):
                prefix = m.group(1).lower() if m.group(1) else None
                col = m.group(2).lower()
                if col not in self.SQL_FUNCTIONS_AND_KEYWORDS:
                    columns.append((prefix, col))

        # Extract from WHERE clause
        where_match = re.search(r'\bWHERE\b(.+?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|$)', sql_clean, flags=re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            for m in re.finditer(col_pattern, where_clause):
                prefix = m.group(1).lower() if m.group(1) else None
                col = m.group(2).lower()
                if col not in self.SQL_FUNCTIONS_AND_KEYWORDS:
                    columns.append((prefix, col))

        # Extract from GROUP BY clause
        group_match = re.search(r'\bGROUP\s+BY\b(.+?)(?:\bHAVING\b|\bORDER\b|\bLIMIT\b|$)', sql_clean, flags=re.IGNORECASE | re.DOTALL)
        if group_match:
            group_clause = group_match.group(1)
            for m in re.finditer(col_pattern, group_clause):
                prefix = m.group(1).lower() if m.group(1) else None
                col = m.group(2).lower()
                if col not in self.SQL_FUNCTIONS_AND_KEYWORDS:
                    columns.append((prefix, col))

        # Extract from ORDER BY clause
        order_match = re.search(r'\bORDER\s+BY\b(.+?)(?:\bLIMIT\b|$)', sql_clean, flags=re.IGNORECASE | re.DOTALL)
        if order_match:
            order_clause = order_match.group(1)
            for m in re.finditer(col_pattern, order_clause):
                prefix = m.group(1).lower() if m.group(1) else None
                col = m.group(2).lower()
                if col not in self.SQL_FUNCTIONS_AND_KEYWORDS:
                    columns.append((prefix, col))

        return list(set(columns))

    def _validate_columns(
        self,
        sql: str,
        column_map: Dict[str, Set[str]],
        tables: List[Tuple[str, Optional[str]]]
    ) -> List[str]:
        """
        Validate that all column references exist in the schema.
        Returns list of invalid column references.
        """
        invalid_columns = []

        # Build alias -> table mapping
        alias_map: Dict[str, str] = {}
        for table, alias in tables:
            if alias:
                alias_map[alias] = table
            alias_map[table] = table  # Table name can also be used as reference

        # Get all valid columns from referenced tables
        all_valid_columns: Set[str] = set()
        for table, _ in tables:
            if table in column_map:
                all_valid_columns.update(column_map[table])

        # Extract and validate column references
        column_refs = self._extract_column_references(sql)

        for prefix, col in column_refs:
            # Skip if column is "*"
            if col == '*':
                continue

            # If prefixed, validate against specific table
            if prefix:
                table_name = alias_map.get(prefix)
                if table_name and table_name in column_map:
                    if col not in column_map[table_name]:
                        invalid_columns.append(f"{prefix}.{col}")
                # If prefix not found in aliases, might be a subquery alias - skip validation
            else:
                # Unprefixed column must exist in at least one referenced table
                if col not in all_valid_columns:
                    # Check if it might be an alias defined in SELECT
                    # Simple heuristic: if it appears right after "AS", it's an alias
                    if not re.search(rf'\bAS\s+{col}\b', sql, flags=re.IGNORECASE):
                        invalid_columns.append(col)

        return invalid_columns

    def validate_and_rewrite(
        self,
        sql: str,
        schema_metadata: Dict[str, Any] | None = None
    ) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        sql_stripped = sql.strip().rstrip(';')

        # Single statement check (very basic)
        if ';' in sql_stripped:
            raise SQLValidationError("Multiple statements not allowed")

        # Must start with SELECT
        if not re.match(r"^\s*SELECT\b", sql_stripped, flags=re.IGNORECASE):
            raise SQLValidationError("Only SELECT statements are allowed")

        # PRD-61 Bug Fix (US-009): Check for mutations across entire SQL including
        # subqueries. Strip string literals first to avoid false positives on
        # quoted keywords like WHERE name = 'DELETE FROM'.
        clean_sql = re.sub(r"'[^']*'", "''", sql_stripped)
        clean_sql = re.sub(r'"[^"]*"', '""', clean_sql)
        for kw in self.DENY_KEYWORDS:
            if re.search(kw, clean_sql, flags=re.IGNORECASE):
                raise SQLValidationError("Statement contains forbidden keyword")

        # PRD-70 FIX-04: Block CTEs (WITH clauses) — they can hide mutations
        # and complicate static analysis. Regex can't reliably parse nested SQL.
        if re.search(r"\bWITH\b", clean_sql, flags=re.IGNORECASE):
            raise SQLValidationError("CTEs (WITH clauses) are not allowed")

        # PRD-70 FIX-04: Block UNION-based cross-workspace data exfiltration.
        # UNION allows combining results from different WHERE clauses, bypassing
        # workspace isolation if the attacker crafts a second SELECT.
        if re.search(r"\bUNION\b", clean_sql, flags=re.IGNORECASE):
            raise SQLValidationError("UNION queries are not allowed")

        # Schema-based validation
        if schema_metadata:
            allowed_tables = {t.get("name", "").lower() for t in (schema_metadata.get("tables") or [])}

            # Extract and validate table references
            tables = self._extract_referenced_tables(sql_stripped)
            for table, _ in tables:
                if table and table not in allowed_tables:
                    raise SQLValidationError(
                        f"Reference to unknown table: '{table}'. "
                        f"Available tables: {', '.join(sorted(list(allowed_tables)[:10]))}"
                    )

            # Validate columns if strict validation is enabled
            if self.strict_column_validation and tables:
                column_map = self._build_column_map(schema_metadata)
                invalid_cols = self._validate_columns(sql_stripped, column_map, tables)

                if invalid_cols:
                    # Provide helpful error with valid columns
                    table_names = [t for t, _ in tables]
                    valid_cols_sample = []
                    for tname in table_names[:2]:  # Show columns from first 2 tables
                        if tname in column_map:
                            valid_cols_sample.extend(list(column_map[tname])[:5])

                    raise SQLValidationError(
                        f"Reference to unknown column(s): {', '.join(invalid_cols)}. "
                        f"Valid columns include: {', '.join(valid_cols_sample[:10])}"
                    )

        # LIMIT enforcement
        if re.search(r"\bLIMIT\s+\d+\b", sql_stripped, flags=re.IGNORECASE):
            # cap existing LIMIT
            def _cap_limit(match):
                current = int(re.findall(r"\d+", match.group(0))[0])
                if current > self.max_limit:
                    reasons.append(f"LIMIT capped from {current} to {self.max_limit}")
                    return f"LIMIT {self.max_limit}"
                return match.group(0)
            sql_capped = re.sub(r"\bLIMIT\s+\d+\b", _cap_limit, sql_stripped, flags=re.IGNORECASE)
            return sql_capped, reasons
        else:
            reasons.append(f"LIMIT {self.max_limit} injected")
            return f"{sql_stripped} LIMIT {self.max_limit}", reasons

