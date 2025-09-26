"""
Tests for System API endpoints to verify they return the expected structure
for the dashboard.
"""

import unittest
import pytest
from fastapi.testclient import TestClient
from datetime import datetime

# Import the FastAPI app
from orchestrator.main import app

class TestSystemAPI(unittest.TestCase):
    """Test case for System API endpoints"""
    
    def setUp(self):
        self.client = TestClient(app)
    
    def test_system_health_endpoint(self):
        """Test system health endpoint returns the expected structure"""
        response = self.client.get("/api/system/health")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        # Verify top-level properties
        self.assertIn("status", data)
        self.assertIn("timestamp", data)
        self.assertIn("services", data)
        self.assertIn("metrics", data)
        
        # Verify services includes database and redis status
        self.assertIn("database", data["services"])
        self.assertIn("redis", data["services"])
        
        # Verify metrics
        self.assertIn("cpu_usage", data["metrics"])
        self.assertIn("memory_usage", data["metrics"])
        self.assertIn("disk_usage", data["metrics"])
    
    def test_system_metrics_endpoint(self):
        """Test system metrics endpoint returns the expected structure"""
        response = self.client.get("/api/system/metrics")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        # Verify top-level properties
        self.assertIn("cpu", data)
        self.assertIn("memory", data)
        self.assertIn("disk", data)
        self.assertIn("network", data)
        
        # Verify context optimization structure
        self.assertIn("context_optimization", data)
        context_opt = data["context_optimization"]
        self.assertIn("tokens_saved", context_opt)
        self.assertIn("compression_ratio", context_opt)
        self.assertIn("total_optimizations", context_opt)
        self.assertIn("efficiency", context_opt)
        
        # Verify learning metrics structure
        self.assertIn("learning", data)
        learning = data["learning"]
        self.assertIn("total_memories", learning)
        self.assertIn("recent_memories", learning)
        self.assertIn("knowledge_nodes", learning)
        self.assertIn("active_collaborations", learning)
        self.assertIn("total_collaborations", learning)
        self.assertIn("knowledge_growth", learning)
        self.assertIn("memory_consolidations", learning)
        self.assertIn("avg_improvement", learning)
        
        # Verify disk has both percent and usage_percent (for compatibility)
        self.assertIn("percent", data["disk"])
        self.assertIn("usage_percent", data["disk"])
        # Verify they are the same value
        self.assertEqual(data["disk"]["percent"], data["disk"]["usage_percent"])

if __name__ == "__main__":
    unittest.main()
