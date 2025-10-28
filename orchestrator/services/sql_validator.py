import re
from typing import Dict, Any, List, Tuple


class SQLValidationError(Exception):
    pass


class SQLValidator:
    """
    Minimal SQL validator (MVP):
    - SELECT-only (reject INSERT/UPDATE/DELETE/ALTER/TRUNCATE/DROP)
    - Single statement only
    - LIMIT injection/enforcement (max_limit)
    - Optional allowlist by schema metadata (tables/columns)
    """

    DENY_KEYWORDS = [
        r"\\bINSERT\\b", r"\\bUPDATE\\b", r"\\bDELETE\\b", r"\\bALTER\\b",
        r"\\bTRUNCATE\\b", r"\\bDROP\\b", r"\\bCREATE\\b", r"\\bMERGE\\b",
        r"\\bGRANT\\b", r"\\bREVOKE\\b", r"\\bVACUUM\\b"
    ]

    def __init__(self, max_limit: int = 1000):
        self.max_limit = max_limit

    def validate_and_rewrite(self, sql: str, schema_metadata: Dict[str, Any] | None = None) -> Tuple[str, List[str]]:
        reasons: List[str] = []
        sql_stripped = sql.strip().rstrip(';')

        # Single statement check (very basic)
        if ';' in sql_stripped:
            raise SQLValidationError("Multiple statements not allowed")

        # Must start with SELECT
        if not re.match(r"^\s*SELECT\b", sql_stripped, flags=re.IGNORECASE):
            raise SQLValidationError("Only SELECT statements are allowed")

        # Denylist
        for kw in self.DENY_KEYWORDS:
            if re.search(kw, sql_stripped, flags=re.IGNORECASE):
                raise SQLValidationError("Statement contains forbidden keyword")

        # Optional allowlist based on schema metadata
        if schema_metadata:
            allowed_tables = {t.get("name") for t in (schema_metadata.get("tables") or [])}
            # naive table reference checks (FROM, JOIN)
            for m in re.finditer(r"\bFROM\s+([\w\.\"]+)|\bJOIN\s+([\w\.\"]+)", sql_stripped, flags=re.IGNORECASE):
                table = (m.group(1) or m.group(2) or '').replace('"', '')
                # strip schema qualifier if present
                if '.' in table:
                    table = table.split('.')[-1]
                if table and table not in allowed_tables:
                    raise SQLValidationError(f"Reference to unknown table: {table}")

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
