"""
Adversarial tests for NL2SQL SQLValidator — PRD-70 FIX-04.

Covers:
  - Existing mutation detection (INSERT/UPDATE/DELETE in subqueries)
  - New deny keywords (RETURNING, COPY, EXECUTE)
  - CTE (WITH clause) blocking
  - UNION cross-workspace blocking
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
        """INSERT hidden in a subquery — caught by deny keyword scan."""
        with pytest.raises(SQLValidationError, match="forbidden keyword"):
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
        with pytest.raises(SQLValidationError, match="forbidden keyword"):
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
        """EXECUTE keyword embedded in a SELECT — caught by deny keyword scan."""
        with pytest.raises(SQLValidationError, match="forbidden keyword"):
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
# PRD-70 FIX-04: CTE blocking
# ============================================================================

class TestCTEBlocking:
    """WITH clauses (CTEs) are blocked to prevent hidden mutations."""

    def test_rejects_simple_cte(self, validator):
        with pytest.raises(SQLValidationError, match="CTE"):
            validator.validate_and_rewrite(
                "SELECT * FROM (WITH x AS (SELECT 1) SELECT * FROM x) AS t"
            )

    def test_rejects_cte_at_start(self, validator):
        """CTEs at statement start are caught before the SELECT check."""
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite(
                "WITH active_users AS (SELECT * FROM users WHERE active = true) "
                "SELECT * FROM active_users"
            )

    def test_with_in_string_literal_is_safe(self, validator):
        """'WITH' inside a string literal should not be blocked."""
        sql, _ = validator.validate_and_rewrite(
            "SELECT * FROM orders WHERE note = 'WITH love'"
        )
        assert "orders" in sql


# ============================================================================
# PRD-70 FIX-04: UNION cross-workspace blocking
# ============================================================================

class TestUnionBlocking:
    """UNION queries are blocked to prevent cross-workspace data exfiltration."""

    def test_rejects_union(self, validator):
        with pytest.raises(SQLValidationError, match="UNION"):
            validator.validate_and_rewrite(
                "SELECT * FROM users WHERE workspace_id = 'mine' "
                "UNION SELECT * FROM users WHERE workspace_id = 'theirs'"
            )

    def test_rejects_union_all(self, validator):
        with pytest.raises(SQLValidationError, match="UNION"):
            validator.validate_and_rewrite(
                "SELECT id FROM users UNION ALL SELECT id FROM users"
            )

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
