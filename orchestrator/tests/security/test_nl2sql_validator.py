"""
Adversarial tests for NL2SQL SQLValidator — PRD-70 FIX-04,
reconciled with the PRD-160 S2 sqlglot AST validator.

Covers:
  - Mutation detection (INSERT/UPDATE/DELETE/DROP anywhere in the tree)
  - Deny keywords (RETURNING, COPY, EXECUTE) — rejected structurally
  - CTE / UNION scoping: the AST allowlist walks every table in the tree,
    including inside ``WITH`` clauses and every UNION branch, so an escape to
    an unlisted table is blocked while a safe CTE/UNION over allowlisted
    tables is allowed (the old regex blocked them wholesale, which also
    rejected legitimate queries).
  - String literal bypass resistance
"""

import pytest

import importlib.util
import sys
from pathlib import Path

# Import validator directly to avoid pulling in the full nl2sql service
# which requires sqlparse and other heavy deps not needed for unit tests.
_validator_path = Path(__file__).resolve().parents[2] / "modules" / "nl2sql" / "query" / "validator.py"
_spec = importlib.util.spec_from_file_location("nl2sql_validator", _validator_path)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["nl2sql_validator"] = _mod
_spec.loader.exec_module(_mod)
SQLValidator = _mod.SQLValidator
SQLValidationError = _mod.SQLValidationError


@pytest.fixture
def validator():
    return SQLValidator(max_limit=100)


# ============================================================================
# Existing mutation detection (PRD-61 — regression tests)
# ============================================================================

class TestMutationDetection:
    """Verify existing mutation denial still works."""

    def test_rejects_plain_insert(self, validator):
        # Caught by "must start with SELECT" before deny keyword check
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("INSERT INTO users (name) VALUES ('x')")

    def test_rejects_plain_delete(self, validator):
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("DELETE FROM users WHERE id = 1")

    def test_rejects_plain_update(self, validator):
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("UPDATE users SET name = 'x' WHERE id = 1")

    def test_rejects_drop_table(self, validator):
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("DROP TABLE users")

    def test_rejects_insert_in_subquery(self, validator):
        """INSERT hidden in a subquery — rejected by the AST walk.

        PRD-160 S2: the sqlglot validator refuses any non-SELECT node anywhere
        in the tree (and unparseable SQL), so the regex-era "forbidden keyword"
        message no longer applies — what matters is the write never executes.
        """
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite(
                "SELECT * FROM (INSERT INTO users (name) VALUES ('x') RETURNING *) AS t"
            )

    def test_rejects_truncate(self, validator):
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("TRUNCATE TABLE users")

    def test_accepts_valid_select(self, validator):
        sql, reasons = validator.validate_and_rewrite(
            "SELECT id, name FROM users WHERE active = true"
        )
        assert "id" in sql
        assert "LIMIT" in sql

    def test_keyword_in_string_literal_is_safe(self, validator):
        """Keywords inside string literals should NOT trigger denial."""
        sql, _ = validator.validate_and_rewrite(
            "SELECT * FROM users WHERE name = 'DELETE FROM'"
        )
        assert "users" in sql


# ============================================================================
# PRD-70 FIX-04: New deny keywords
# ============================================================================

class TestNewDenyKeywords:
    """Test RETURNING, COPY, EXECUTE are now blocked."""

    def test_rejects_returning_clause(self, validator):
        # RETURNING is not valid on a SELECT; the AST parse fails closed and
        # the validator raises rather than letting the statement through.
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite(
                "SELECT * FROM users RETURNING *"
            )

    def test_rejects_copy_to(self, validator):
        # COPY doesn't start with SELECT, caught by structure check
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("COPY users TO '/tmp/dump.csv'")

    def test_rejects_copy_from(self, validator):
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("COPY users FROM '/tmp/data.csv'")

    def test_rejects_execute(self, validator):
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("EXECUTE my_plan")

    def test_rejects_execute_in_select(self, validator):
        """EXECUTE embedded in a SELECT — rejected (the AST fails closed)."""
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite(
                "SELECT EXECUTE('SELECT 1') FROM users"
            )

    def test_returning_in_string_literal_is_safe(self, validator):
        """'RETURNING' inside a string literal should not be blocked."""
        sql, _ = validator.validate_and_rewrite(
            "SELECT * FROM orders WHERE status = 'RETURNING'"
        )
        assert "orders" in sql


# ============================================================================
# PRD-160 S2: CTE scoping (the AST allowlist sees into WITH clauses)
# ============================================================================

class TestCTEScoping:
    """CTEs are no longer banned wholesale. The sqlglot validator walks every
    table in the tree — including inside ``WITH`` clauses — so a CTE that
    reaches a table outside the connection's allowlist is blocked, while a CTE
    over allowlisted tables is allowed. Hidden mutations inside a CTE are still
    caught by the non-SELECT-node walk (see TestMutationDetection)."""

    _SCHEMA = {"tables": [{"name": "users"}]}

    def test_blocks_cte_escape_to_unlisted_table(self, validator):
        """A CTE reaching a table outside the allowlist is rejected."""
        with pytest.raises(SQLValidationError, match="unknown table"):
            validator.validate_and_rewrite(
                "WITH x AS (SELECT * FROM secrets) SELECT * FROM x",
                schema_metadata=self._SCHEMA,
            )

    def test_allows_safe_cte_over_allowlisted_tables(self, validator):
        """A CTE that only touches allowlisted tables is allowed."""
        sql, _ = validator.validate_and_rewrite(
            "WITH active_users AS (SELECT * FROM users WHERE active = true) "
            "SELECT * FROM active_users",
            schema_metadata=self._SCHEMA,
        )
        assert "active_users" in sql

    def test_with_in_string_literal_is_safe(self, validator):
        """'WITH' inside a string literal should not be blocked."""
        sql, _ = validator.validate_and_rewrite(
            "SELECT * FROM orders WHERE note = 'WITH love'"
        )
        assert "orders" in sql


# ============================================================================
# PRD-160 S2: UNION scoping (the AST allowlist sees into every UNION branch)
# ============================================================================

class TestUnionScoping:
    """UNION queries are no longer banned wholesale. The validator allowlists
    every table in every branch, so a UNION that reaches a table outside the
    connection's schema (the cross-table exfiltration vector) is blocked, while
    a UNION over allowlisted tables is allowed. Cross-*workspace* reach is
    additionally prevented at the source layer — a query only ever runs against
    the caller's own resolved connection (see the _get_source workspace pins in
    test_nl2sql_validation_path.py)."""

    _SCHEMA = {"tables": [{"name": "users"}]}

    def test_blocks_union_escape_to_unlisted_table(self, validator):
        """A UNION branch reaching an unlisted table is rejected."""
        with pytest.raises(SQLValidationError, match="unknown table"):
            validator.validate_and_rewrite(
                "SELECT * FROM users UNION SELECT * FROM secrets",
                schema_metadata=self._SCHEMA,
            )

    def test_allows_safe_union_over_allowlisted_tables(self, validator):
        """A UNION over only allowlisted tables is allowed."""
        sql, _ = validator.validate_and_rewrite(
            "SELECT id FROM users UNION ALL SELECT id FROM users",
            schema_metadata=self._SCHEMA,
        )
        assert "users" in sql

    def test_union_in_string_literal_is_safe(self, validator):
        """'UNION' inside a string literal should not be blocked."""
        sql, _ = validator.validate_and_rewrite(
            "SELECT * FROM users WHERE organization = 'European UNION'"
        )
        assert "users" in sql


# ============================================================================
# LIMIT enforcement (regression)
# ============================================================================

class TestLimitEnforcement:
    """Verify LIMIT injection/capping still works."""

    def test_injects_limit_when_missing(self, validator):
        sql, reasons = validator.validate_and_rewrite("SELECT * FROM users")
        assert "LIMIT 100" in sql
        assert any("injected" in r for r in reasons)

    def test_caps_excessive_limit(self, validator):
        sql, reasons = validator.validate_and_rewrite(
            "SELECT * FROM users LIMIT 99999"
        )
        assert "LIMIT 100" in sql

    def test_preserves_small_limit(self, validator):
        sql, reasons = validator.validate_and_rewrite(
            "SELECT * FROM users LIMIT 10"
        )
        assert "LIMIT 10" in sql


# ============================================================================
# Statement structure (regression)
# ============================================================================

class TestStatementStructure:
    """Verify basic structural checks."""

    def test_rejects_multiple_statements(self, validator):
        with pytest.raises(SQLValidationError, match="Multiple statements"):
            validator.validate_and_rewrite("SELECT 1; SELECT 2")

    def test_rejects_non_select(self, validator):
        with pytest.raises(SQLValidationError, match="SELECT"):
            validator.validate_and_rewrite("SHOW TABLES")


# ============================================================================
# PRD-172 F019 — side-effecting functions inside a nominal SELECT
# ============================================================================

class TestSideEffectingFunctionsF019:
    """The SELECT-only validator must reject side-effecting SQL functions.

    Before PRD-172 the AST walk only looked for DML/DDL *statement* nodes, so a
    payload that wraps a mutation in a function call
    (``SELECT query_to_xml('UPDATE …', …)``) had a SELECT root, no INTO, and no
    Insert/Update/Delete node — it passed and still mutated. These tests pin the
    function denylist. The exact same schema_metadata allowlist is provided so
    the rejection is on the FUNCTION, not on an unlisted table.
    """

    _META = {"tables": [{"name": "users", "columns": [{"name": "id"}, {"name": "email"}]}]}

    @pytest.mark.parametrize(
        "sql",
        [
            # The canonical review payload: a write smuggled through query_to_xml.
            "SELECT query_to_xml('UPDATE users SET email=''x''', true, true, '')",
            # dblink executes arbitrary statements on a (possibly the same) DB.
            "SELECT dblink_exec('DELETE FROM users')",
            "SELECT * FROM users WHERE id IN (SELECT dblink('host=x', 'DROP TABLE users'))",
            # Server-side filesystem read/write.
            "SELECT pg_read_file('/etc/passwd')",
            "SELECT lo_export(1234, '/tmp/x')",
            # Availability / timing side effect.
            "SELECT pg_sleep(10)",
            # Session/config mutation.
            "SELECT set_config('search_path', 'evil', false)",
            # Hidden deep in the tree: inside a CTE.
            "WITH t AS (SELECT query_to_xml('UPDATE users SET id=1', true, true, '')) "
            "SELECT * FROM t",
            # Hidden in a UNION branch.
            "SELECT id FROM users UNION SELECT pg_sleep(5)",
        ],
    )
    def test_rejects_side_effecting_functions(self, validator, sql):
        with pytest.raises(SQLValidationError, match="Side-effecting function not allowed"):
            validator.validate_and_rewrite(sql, schema_metadata=self._META)

    @pytest.mark.parametrize(
        "sql",
        [
            # Legitimate read-only scalar/aggregate functions must still pass.
            "SELECT upper(email) FROM users",
            "SELECT count(*) FROM users",
            "SELECT id, lower(email) AS e FROM users WHERE email LIKE '%@x.com'",
            "SELECT coalesce(email, 'none') FROM users",
        ],
    )
    def test_allows_safe_read_only_functions(self, validator, sql):
        safe_sql, _ = validator.validate_and_rewrite(sql, schema_metadata=self._META)
        assert "SELECT" in safe_sql.upper()
        # LIMIT is injected by the validator — proves it reached the rewrite path
        # (i.e. it did NOT reject the query).
        assert "LIMIT" in safe_sql.upper()

    def test_denylist_is_case_insensitive(self, validator):
        with pytest.raises(SQLValidationError, match="Side-effecting function not allowed"):
            validator.validate_and_rewrite(
                "SELECT QUERY_TO_XML('UPDATE users SET id=1', true, true, '')",
                schema_metadata=self._META,
            )
