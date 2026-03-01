"""Journey 01 + 16: Health & system endpoints."""

import pytest
from unittest.mock import patch

# [Previous test functions remain unchanged...]

def test_intentional_pipeline_validator(client):
    """Test database connection for pipeline validator."""
    with patch('psycopg2.connect') as mock_connect:
        mock_connect.side_effect = Exception("Connection failed")
        with pytest.raises(Exception) as excinfo:
            client.get("/api/system/health")
        assert "Connection failed" in str(excinfo.value)