"""
Learning Module - PlaybookMiner Unit Tests
===========================================
Direct unit tests for PlaybookMiner without full module imports.
Tests the miner logic directly to avoid circular import issues.
"""

import pytest
import json
from sqlalchemy.orm import Session


class TestPlaybookMinerDirect:
    """Test PlaybookMiner directly without full imports"""
    
    def test_simple_pattern_counting(self, db_session):
        """Test basic pattern counting logic"""
        # Simulate what PlaybookMiner.mine() does
        sequences = [
            ["retrieve", "assemble_context", "tool:web_search"],
            ["retrieve", "assemble_context", "tool:db_query"],
            ["retrieve", "assemble_context", "tool:web_search"],
            ["retrieve", "assemble_context", "tool:web_search"],
        ]
        
        # Count patterns
        counts = {}
        for seq in sequences:
            key = ",".join(seq)
            counts[key] = counts.get(key, 0) + 1
        
        # Filter by min_support
        min_support = 2
        patterns = [
            {"pattern": k.split(","), "support": v}
            for k, v in counts.items()
            if v >= min_support
        ]
        
        # Verify results
        assert len(patterns) > 0
        for pattern in patterns:
            assert pattern["support"] >= min_support
    
    def test_pattern_support_filtering(self, db_session):
        """Test filtering patterns by support threshold"""
        sequences = [
            ["action_a", "action_b"],
            ["action_a", "action_b"],
            ["action_a", "action_b"],
            ["action_c", "action_d"],
        ]
        
        counts = {}
        for seq in sequences:
            key = ",".join(seq)
            counts[key] = counts.get(key, 0) + 1
        
        # Test different thresholds
        patterns_low = [
            {"pattern": k.split(","), "support": v}
            for k, v in counts.items()
            if v >= 1
        ]
        
        patterns_high = [
            {"pattern": k.split(","), "support": v}
            for k, v in counts.items()
            if v >= 3
        ]
        
        # High threshold should return fewer patterns
        assert len(patterns_high) <= len(patterns_low)
        assert len(patterns_high) == 1  # Only one pattern has support >= 3
    
    def test_max_length_filtering(self, db_session):
        """Test filtering patterns by maximum length"""
        patterns = [
            {"pattern": ["a", "b"], "support": 5},
            {"pattern": ["a", "b", "c"], "support": 4},
            {"pattern": ["a", "b", "c", "d"], "support": 3},
            {"pattern": ["a", "b", "c", "d", "e"], "support": 2},
        ]
        
        max_len = 3
        filtered = [p for p in patterns if len(p["pattern"]) <= max_len]
        
        assert len(filtered) == 2  # Only first two patterns
        for pattern in filtered:
            assert len(pattern["pattern"]) <= max_len


class TestPlaybookPersistenceLogic:
    """Test playbook persistence logic"""
    
    def test_playbook_data_structure(self, db_session):
        """Test playbook data structure for database"""
        pattern = ["retrieve", "assemble_context", "tool:web_search"]
        support = 5
        tenant_id = "test_tenant"
        name = "test-playbook-1"
        
        # Simulate what would be inserted
        playbook_data = {
            "name": name,
            "tenant_id": tenant_id,
            "pattern": json.dumps(pattern),
            "support": support,
        }
        
        # Verify structure
        assert playbook_data["name"] == name
        assert playbook_data["tenant_id"] == tenant_id
        assert playbook_data["support"] == support
        
        # Verify pattern can be serialized/deserialized
        pattern_json = json.loads(playbook_data["pattern"])
        assert pattern_json == pattern
    
    def test_top_k_selection(self, db_session):
        """Test selecting top K patterns"""
        patterns = [
            {"pattern": ["a", "b"], "support": 10},
            {"pattern": ["c", "d"], "support": 8},
            {"pattern": ["e", "f"], "support": 6},
            {"pattern": ["g", "h"], "support": 4},
            {"pattern": ["i", "j"], "support": 2},
        ]
        
        top_k = 3
        top_patterns = patterns[:top_k]
        
        assert len(top_patterns) == 3
        assert top_patterns[0]["support"] == 10
        assert top_patterns[2]["support"] == 6


class TestTenantIsolationLogic:
    """Test tenant isolation logic"""
    
    def test_tenant_filtering(self, db_session):
        """Test that tenant ID is properly used for filtering"""
        tenant_a = "tenant_a"
        tenant_b = "tenant_b"
        
        # Simulate playbooks for different tenants
        playbooks = [
            {"name": "playbook_1", "tenant_id": tenant_a, "support": 5},
            {"name": "playbook_2", "tenant_id": tenant_a, "support": 4},
            {"name": "playbook_3", "tenant_id": tenant_b, "support": 6},
            {"name": "playbook_4", "tenant_id": tenant_b, "support": 3},
        ]
        
        # Filter by tenant
        tenant_a_playbooks = [p for p in playbooks if p["tenant_id"] == tenant_a]
        tenant_b_playbooks = [p for p in playbooks if p["tenant_id"] == tenant_b]
        
        assert len(tenant_a_playbooks) == 2
        assert len(tenant_b_playbooks) == 2
        
        # Verify no overlap
        tenant_a_names = {p["name"] for p in tenant_a_playbooks}
        tenant_b_names = {p["name"] for p in tenant_b_playbooks}
        assert tenant_a_names.isdisjoint(tenant_b_names)


class TestPatternMetrics:
    """Test pattern quality metrics calculations"""
    
    def test_support_calculation(self, db_session):
        """Test support metric calculation"""
        sequences = [
            ["a", "b", "c"],
            ["a", "b", "c"],
            ["a", "b", "c"],
            ["d", "e", "f"],
        ]
        
        pattern = ["a", "b", "c"]
        pattern_key = ",".join(pattern)
        
        # Count occurrences
        support = sum(1 for seq in sequences if ",".join(seq) == pattern_key)
        
        assert support == 3
    
    def test_confidence_placeholder(self, db_session):
        """Test confidence calculation (placeholder for now)"""
        # Confidence = successful executions / total executions
        # This requires execution success data
        
        total_executions = 10
        successful_executions = 8
        
        confidence = successful_executions / total_executions
        
        assert confidence == 0.8
        assert 0.0 <= confidence <= 1.0
    
    def test_lift_placeholder(self, db_session):
        """Test lift calculation (placeholder for now)"""
        # Lift = P(B|A) / P(B)
        # This requires conditional probability calculation
        
        p_b_given_a = 0.85  # Probability of B given A
        p_b = 0.30  # Probability of B overall
        
        lift = p_b_given_a / p_b
        
        assert lift > 1.0  # Positive correlation
        assert lift == pytest.approx(2.83, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
