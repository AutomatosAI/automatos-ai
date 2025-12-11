"""
NL2SQL SQL Validator Tests
===========================
Comprehensive test suite for SQLValidator class.
Tests security-critical SQL validation logic.
"""

import pytest
from modules.nl2sql.query.validator import SQLValidator, SQLValidationError


class TestSelectOnlyEnforcement:
    """Test that only SELECT statements are allowed"""
    
    def test_select_query_passes(self):
        """Valid SELECT query should pass"""
        validator = SQLValidator()
        sql, warnings = validator.validate_and_rewrite(
            "SELECT * FROM users",
            schema_metadata={"tables": [{"name": "users"}]}
        )
        assert "SELECT" in sql.upper()
        assert "LIMIT" in sql.upper()  # Should inject LIMIT
    
    def test_insert_query_fails(self):
        """INSERT statement should be rejected"""
        validator = SQLValidator()
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validator.validate_and_rewrite("INSERT INTO users VALUES (1, 'test')")
    
    def test_update_query_fails(self):
        """UPDATE statement should be rejected"""
        validator = SQLValidator()
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validator.validate_and_rewrite("UPDATE users SET name = 'test')")
    
    def test_delete_query_fails(self):
        """DELETE statement should be rejected"""
        validator = SQLValidator()
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validator.validate_and_rewrite("DELETE FROM users WHERE id = 1")
    
    def test_drop_query_fails(self):
        """DROP statement should be rejected"""
        validator = SQLValidator()
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validator.validate_and_rewrite("DROP TABLE users")
    
    def test_create_query_fails(self):
        """CREATE statement should be rejected"""
        validator = SQLValidator()
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validator.validate_and_rewrite("CREATE TABLE test (id INT)")
    
    def test_alter_query_fails(self):
        """ALTER statement should be rejected"""
        validator = SQLValidator()
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validator.validate_and_rewrite("ALTER TABLE users ADD COLUMN test VARCHAR(255)")
    
    def test_truncate_query_fails(self):
        """TRUNCATE statement should be rejected"""
        validator = SQLValidator()
        with pytest.raises(SQLValidationError, match="Only SELECT"):
            validator.validate_and_rewrite("TRUNCATE TABLE users")


class TestSQLInjectionPrevention:
    """Test SQL injection attack prevention"""
    
    def test_multiple_statements_blocked(self):
        """Multiple statements separated by semicolon should be blocked"""
        validator = SQLValidator()
        with pytest.raises(SQLValidationError, match="Multiple statements"):
            validator.validate_and_rewrite(
                "SELECT * FROM users; DROP TABLE users;"
            )
    
    def test_union_injection_blocked(self):
        """UNION-based injection should be blocked if accessing forbidden tables"""
        validator = SQLValidator()
        schema = {"tables": [{"name": "users"}]}
        
        # This should fail because 'passwords' table is not in schema
        with pytest.raises(SQLValidationError, match="unknown table"):
            validator.validate_and_rewrite(
                "SELECT * FROM users UNION SELECT * FROM passwords",
                schema_metadata=schema
            )
    
    def test_comment_injection_handled(self):
        """SQL comments should not bypass validation"""
        validator = SQLValidator()
        
        # Try to hide forbidden keyword in comment (semicolon will be caught)
        with pytest.raises(SQLValidationError, match="Multiple statements"):
            validator.validate_and_rewrite(
                "SELECT * FROM users; -- DROP TABLE users"
            )


class TestLimitEnforcement:
    """Test LIMIT clause injection and capping"""
    
    def test_limit_injection_when_missing(self):
        """LIMIT should be injected if not present"""
        validator = SQLValidator(max_limit=1000)
        sql, warnings = validator.validate_and_rewrite(
            "SELECT * FROM users",
            schema_metadata={"tables": [{"name": "users"}]}
        )
        
        assert "LIMIT 1000" in sql
        assert any("injected" in w for w in warnings)
    
    def test_limit_capping_when_too_high(self):
        """LIMIT should be capped if exceeds max"""
        validator = SQLValidator(max_limit=100)
        sql, warnings = validator.validate_and_rewrite(
            "SELECT * FROM users LIMIT 5000",
            schema_metadata={"tables": [{"name": "users"}]}
        )
        
        assert "LIMIT 100" in sql
        assert any("capped" in w for w in warnings)
    
    def test_limit_preserved_when_valid(self):
        """Valid LIMIT should be preserved"""
        validator = SQLValidator(max_limit=1000)
        sql, warnings = validator.validate_and_rewrite(
            "SELECT * FROM users LIMIT 50",
            schema_metadata={"tables": [{"name": "users"}]}
        )
        
        assert "LIMIT 50" in sql
        assert not any("capped" in w for w in warnings)
    
    def test_custom_max_limit(self):
        """Custom max_limit should be respected"""
        validator = SQLValidator(max_limit=500)
        sql, warnings = validator.validate_and_rewrite(
            "SELECT * FROM users",
            schema_metadata={"tables": [{"name": "users"}]}
        )
        
        assert "LIMIT 500" in sql


class TestTableAllowlist:
    """Test table allowlist validation"""
    
    def test_valid_table_passes(self):
        """Query referencing valid table should pass"""
        validator = SQLValidator()
        schema = {
            "tables": [
                {"name": "users"},
                {"name": "orders"}
            ]
        }
        
        sql, warnings = validator.validate_and_rewrite(
            "SELECT * FROM users",
            schema_metadata=schema
        )
        
        assert "SELECT" in sql
    
    def test_invalid_table_fails(self):
        """Query referencing unknown table should fail"""
        validator = SQLValidator()
        schema = {"tables": [{"name": "users"}]}
        
        with pytest.raises(SQLValidationError, match="unknown table"):
            validator.validate_and_rewrite(
                "SELECT * FROM passwords",
                schema_metadata=schema
            )
    
    def test_join_with_valid_tables_passes(self):
        """JOIN with valid tables should pass"""
        validator = SQLValidator()
        schema = {
            "tables": [
                {"name": "users"},
                {"name": "orders"}
            ]
        }
        
        sql, warnings = validator.validate_and_rewrite(
            "SELECT * FROM users JOIN orders ON users.id = orders.user_id",
            schema_metadata=schema
        )
        
        assert "SELECT" in sql
    
    def test_join_with_invalid_table_fails(self):
        """JOIN with unknown table should fail"""
        validator = SQLValidator()
        schema = {"tables": [{"name": "users"}]}
        
        with pytest.raises(SQLValidationError, match="unknown table"):
            validator.validate_and_rewrite(
                "SELECT * FROM users JOIN passwords ON users.id = passwords.user_id",
                schema_metadata=schema
            )
    
    def test_schema_qualified_table_names(self):
        """Schema-qualified table names should be handled"""
        validator = SQLValidator()
        schema = {"tables": [{"name": "users"}]}
        
        # Should extract 'users' from 'public.users'
        sql, warnings = validator.validate_and_rewrite(
            'SELECT * FROM public.users',
            schema_metadata=schema
        )
        
        assert "SELECT" in sql


class TestEdgeCases:
    """Test edge cases and special scenarios"""
    
    def test_empty_query_fails(self):
        """Empty query should fail"""
        validator = SQLValidator()
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("")
    
    def test_whitespace_only_fails(self):
        """Whitespace-only query should fail"""
        validator = SQLValidator()
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("   \n\t  ")
    
    def test_case_insensitive_keyword_detection(self):
        """Forbidden keywords should be detected case-insensitively"""
        validator = SQLValidator()
        
        # Lowercase
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("drop table users")
        
        # Mixed case
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("DrOp TaBlE users")
        
        # Uppercase
        with pytest.raises(SQLValidationError):
            validator.validate_and_rewrite("DROP TABLE USERS")
    
    def test_trailing_semicolon_removed(self):
        """Trailing semicolon should be removed"""
        validator = SQLValidator()
        sql, warnings = validator.validate_and_rewrite(
            "SELECT * FROM users;",
            schema_metadata={"tables": [{"name": "users"}]}
        )
        
        # Should not end with semicolon (except possibly after LIMIT)
        assert not sql.rstrip().endswith(";")
    
    def test_complex_select_query(self):
        """Complex SELECT with subqueries should pass"""
        validator = SQLValidator()
        schema = {
            "tables": [
                {"name": "users"},
                {"name": "orders"}
            ]
        }
        
        complex_query = """
            SELECT u.name, COUNT(o.id) as order_count
            FROM users u
            LEFT JOIN orders o ON u.id = o.user_id
            WHERE u.created_at > '2024-01-01'
            GROUP BY u.name
            HAVING COUNT(o.id) > 5
            ORDER BY order_count DESC
        """
        
        sql, warnings = validator.validate_and_rewrite(
            complex_query,
            schema_metadata=schema
        )
        
        assert "SELECT" in sql
        assert "LIMIT" in sql


class TestSecurityScenarios:
    """Test real-world security attack scenarios"""
    
    def test_stacked_queries(self):
        """Stacked queries should be blocked"""
        validator = SQLValidator()
        
        attacks = [
            "SELECT * FROM users; DELETE FROM users;",
            "SELECT * FROM users; UPDATE users SET admin = true;",
            "SELECT * FROM users; INSERT INTO admins VALUES (1);",
        ]
        
        for attack in attacks:
            with pytest.raises(SQLValidationError):
                validator.validate_and_rewrite(attack)
    
    def test_privilege_escalation_attempts(self):
        """Privilege escalation attempts should be blocked"""
        validator = SQLValidator()
        
        attacks = [
            "GRANT ALL PRIVILEGES ON *.* TO 'user'@'%'",
            "REVOKE SELECT ON users FROM 'user'@'%'",
            "CREATE USER 'admin'@'%' IDENTIFIED BY 'password'",
        ]
        
        for attack in attacks:
            with pytest.raises(SQLValidationError):
                validator.validate_and_rewrite(attack)
    
    def test_system_table_access_prevention(self):
        """Access to system tables should be prevented"""
        validator = SQLValidator()
        schema = {"tables": [{"name": "users"}]}
        
        # These should fail because system tables aren't in schema
        system_queries = [
            "SELECT * FROM information_schema.tables",
            "SELECT * FROM pg_catalog.pg_tables",
            "SELECT * FROM mysql.user",
        ]
        
        for query in system_queries:
            with pytest.raises(SQLValidationError, match="unknown table"):
                validator.validate_and_rewrite(query, schema_metadata=schema)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
