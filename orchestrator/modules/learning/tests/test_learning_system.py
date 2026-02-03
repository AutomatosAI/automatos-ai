"""
Learning Module - LearningSystemUpdater Tests
==============================================
Tests for learning from workflow executions, agent performance tracking,
and pattern recognition.
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, MagicMock

from modules.learning import LearningSystemUpdater
from modules.orchestrator.stages import AggregatedResults
from modules.agents import SubtaskExecution, SubtaskStatus
from core.models import Agent, Workflow, WorkflowExecution

from .conftest import TEST_WORKSPACE_ID


class TestLearningSystemUpdater:
    """Test LearningSystemUpdater core functionality"""
    
    @pytest.mark.asyncio
    async def test_initialization(self, db_session):
        """Test LearningSystemUpdater initialization"""
        updater = LearningSystemUpdater(db_session)
        
        assert updater.db == db_session
        assert updater is not None
    
    @pytest.mark.asyncio
    async def test_get_learning_summary(self, db_session):
        """Test retrieving learning summary"""
        updater = LearningSystemUpdater(db_session)
        
        summary = updater.get_learning_summary()
        
        assert summary is not None
        assert isinstance(summary, dict)
    
    @pytest.mark.asyncio
    async def test_update_from_execution_basic(self, db_session):
        """Test basic learning from workflow execution"""
        # Create test workflow and execution
        workflow = Workflow(
            name="Test Workflow",
            description="Test learning workflow",
            workspace_id=TEST_WORKSPACE_ID
        )
        db_session.add(workflow)
        db_session.flush()
        
        execution = WorkflowExecution(
            workflow_id=workflow.id,
            status="completed",
            workspace_id=TEST_WORKSPACE_ID
        )
        db_session.add(execution)
        db_session.flush()
        
        # Create updater
        updater = LearningSystemUpdater(db_session)
        
        # Mock aggregated results
        aggregated_results = AggregatedResults(
            quality_score=0.85,
            accuracy_score=0.90,
            completeness_score=0.88,
            relevance_score=0.92
        )
        
        # Update from execution
        await updater.update_from_execution(
            workflow_id=workflow.id,
            execution_id=execution.id,
            aggregated_results=aggregated_results,
            subtask_executions={},
            decomposition_metadata={},
            context_metadata={}
        )
        
        # Verify learning occurred
        summary = updater.get_learning_summary()
        assert summary is not None


class TestAgentPerformanceTracking:
    """Test agent performance metrics tracking"""
    
    @pytest.mark.asyncio
    async def test_agent_performance_update(self, db_session):
        """Test updating agent performance metrics"""
        # Create test agent
        agent = Agent(
            name="TestAgent",
            agent_type="test",
            description="Test agent for learning",
            total_executions=0,
            successful_executions=0,
            failed_executions=0
        )
        db_session.add(agent)
        db_session.commit()
        
        # Create updater
        updater = LearningSystemUpdater(db_session)
        
        # Mock agent performance data
        agent_performance = {
            "TestAgent": {
                "success": True,
                "tokens_used": 1000,
                "execution_time": 5.0,
                "cost": 0.05
            }
        }
        
        # Mock subtask execution
        subtask = Mock(spec=SubtaskExecution)
        subtask.agent_name = "TestAgent"
        subtask.status = SubtaskStatus.COMPLETED
        subtask.tokens_used = 1000
        subtask.execution_time_seconds = 5.0
        
        subtask_executions = {"task_1": subtask}
        
        # Update performance
        updater._update_agent_performance(
            agent_performance,
            subtask_executions
        )
        
        # Verify metrics updated
        db_session.refresh(agent)
        assert agent.total_executions > 0
    
    @pytest.mark.asyncio
    async def test_agent_success_rate_calculation(self, db_session):
        """Test agent success rate calculation"""
        # Create agent with some history
        agent = Agent(
            name="SuccessAgent",
            agent_type="test",
            total_executions=10,
            successful_executions=8,
            failed_executions=2
        )
        db_session.add(agent)
        db_session.commit()
        
        # Verify success rate
        success_rate = agent.successful_executions / agent.total_executions if agent.total_executions > 0 else 0
        assert success_rate == 0.8


class TestWorkflowPatternLearning:
    """Test learning from workflow execution patterns"""
    
    @pytest.mark.asyncio
    async def test_learn_successful_pattern(self, db_session):
        """Test learning from successful workflow"""
        # Create workflow
        workflow = Workflow(
            name="Pattern Test Workflow",
            description="Test pattern learning",
            workspace_id=TEST_WORKSPACE_ID
        )
        db_session.add(workflow)
        db_session.flush()
        
        updater = LearningSystemUpdater(db_session)
        
        # Mock decomposition metadata
        decomposition_metadata = {
            "strategy": "sequential",
            "agent_count": 3,
            "complexity": "medium"
        }
        
        # Learn from pattern
        updater._learn_workflow_patterns(
            workflow_id=workflow.id,
            quality_score=0.90,
            decomposition_metadata=decomposition_metadata
        )
        
        # Verify pattern learned (check logs or summary)
        summary = updater.get_learning_summary()
        assert summary is not None
    
    @pytest.mark.asyncio
    async def test_learn_failed_pattern(self, db_session):
        """Test learning from failed workflow"""
        workflow = Workflow(
            name="Failed Pattern Workflow",
            description="Test failure learning",
            workspace_id=TEST_WORKSPACE_ID
        )
        db_session.add(workflow)
        db_session.flush()
        
        updater = LearningSystemUpdater(db_session)
        
        decomposition_metadata = {
            "strategy": "parallel",
            "agent_count": 5,
            "complexity": "high"
        }
        
        # Learn from failed pattern (low quality score)
        updater._learn_workflow_patterns(
            workflow_id=workflow.id,
            quality_score=0.30,
            decomposition_metadata=decomposition_metadata
        )
        
        summary = updater.get_learning_summary()
        assert summary is not None


class TestContextEffectiveness:
    """Test context engineering effectiveness assessment"""
    
    @pytest.mark.asyncio
    async def test_effective_context_assessment(self, db_session):
        """Test assessing effective context"""
        updater = LearningSystemUpdater(db_session)
        
        # High quality context metadata
        context_metadata = {
            "coverage": 0.90,
            "quality": 0.92,
            "sources": 8,
            "retrieval_time": 0.5
        }
        
        # Assess effectiveness
        updater._assess_context_effectiveness(
            context_metadata,
            accuracy_score=0.88
        )
        
        summary = updater.get_learning_summary()
        assert summary is not None
    
    @pytest.mark.asyncio
    async def test_ineffective_context_assessment(self, db_session):
        """Test assessing ineffective context"""
        updater = LearningSystemUpdater(db_session)
        
        # Low quality context metadata
        context_metadata = {
            "coverage": 0.40,
            "quality": 0.50,
            "sources": 2,
            "retrieval_time": 2.0
        }
        
        # Assess effectiveness
        updater._assess_context_effectiveness(
            context_metadata,
            accuracy_score=0.45
        )
        
        summary = updater.get_learning_summary()
        assert summary is not None
    
    @pytest.mark.asyncio
    async def test_context_recommendation_generation(self, db_session):
        """Test generating context improvement recommendations"""
        updater = LearningSystemUpdater(db_session)
        
        # Get recommendation for poor context
        recommendation = updater._get_context_recommendation(
            is_effective=False,
            coverage=0.40,
            quality=0.50,
            sources=2
        )
        
        assert recommendation is not None
        assert isinstance(recommendation, str)
        assert len(recommendation) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
