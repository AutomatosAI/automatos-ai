#!/usr/bin/env python3
"""
Test LLM-Driven Orchestration
==============================

PRD-16: Test script to demonstrate and validate the LLM-driven orchestration system.
Runs A/B testing between algorithmic and LLM-driven agent selection.

Usage:
    python test_llm_orchestration.py [--full-test] [--execute]
"""

import asyncio
import argparse
import logging
import json
from datetime import datetime
from typing import List, Dict, Any

# Setup path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Import database and configuration
from database.database import get_db_session
from config import config

# Import LLM orchestration components
from core.llm import OrchestratorLLM, FunctionRegistry, FunctionExecutor
from core.llm.llm_agent_selector import LLMAgentSelector
from core.llm.selection_ab_test import SelectionABTest, ABTestResults

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Sample test workflows and subtasks
TEST_WORKFLOWS = {
    "research_analysis": {
        "name": "Market Research and Analysis",
        "description": "Comprehensive market research and competitive analysis",
        "subtasks": [
            {
                "subtask_id": "subtask_0",
                "description": "Research current market trends in AI orchestration platforms",
                "skills_required": ["research", "analysis", "data"],
                "agent_type": "researcher",
                "priority": "high",
                "estimated_duration": 180,
                "dependencies": []
            },
            {
                "subtask_id": "subtask_1",
                "description": "Analyze competitor features and pricing strategies",
                "skills_required": ["analysis", "research", "comparison"],
                "agent_type": "analyst",
                "priority": "high",
                "estimated_duration": 240,
                "dependencies": ["subtask_0"]
            },
            {
                "subtask_id": "subtask_2",
                "description": "Create comprehensive market analysis report with recommendations",
                "skills_required": ["writing", "analysis", "communication"],
                "agent_type": "writer",
                "priority": "medium",
                "estimated_duration": 300,
                "dependencies": ["subtask_0", "subtask_1"]
            }
        ]
    },
    "code_development": {
        "name": "Feature Development Workflow",
        "description": "Develop new feature with testing and documentation",
        "subtasks": [
            {
                "subtask_id": "subtask_0",
                "description": "Design system architecture for new authentication module",
                "skills_required": ["design", "architecture", "planning"],
                "agent_type": "architect",
                "priority": "high",
                "estimated_duration": 240,
                "dependencies": []
            },
            {
                "subtask_id": "subtask_1",
                "description": "Implement OAuth2 authentication with JWT tokens",
                "skills_required": ["coding", "implementation", "security"],
                "agent_type": "developer",
                "priority": "high",
                "estimated_duration": 480,
                "dependencies": ["subtask_0"]
            },
            {
                "subtask_id": "subtask_2",
                "description": "Write comprehensive unit tests and integration tests",
                "skills_required": ["testing", "coding", "validation"],
                "agent_type": "tester",
                "priority": "high",
                "estimated_duration": 300,
                "dependencies": ["subtask_1"]
            },
            {
                "subtask_id": "subtask_3",
                "description": "Create API documentation and usage examples",
                "skills_required": ["writing", "documentation", "communication"],
                "agent_type": "documenter",
                "priority": "medium",
                "estimated_duration": 180,
                "dependencies": ["subtask_1"]
            }
        ]
    },
    "data_pipeline": {
        "name": "Data Processing Pipeline",
        "description": "Build and optimize data processing pipeline",
        "subtasks": [
            {
                "subtask_id": "subtask_0",
                "description": "Extract data from multiple sources including APIs and databases",
                "skills_required": ["data", "extraction", "api"],
                "agent_type": "data_engineer",
                "priority": "high",
                "estimated_duration": 180,
                "dependencies": []
            },
            {
                "subtask_id": "subtask_1",
                "description": "Transform and clean data with validation rules",
                "skills_required": ["data", "transformation", "validation"],
                "agent_type": "data_engineer",
                "priority": "high",
                "estimated_duration": 240,
                "dependencies": ["subtask_0"]
            },
            {
                "subtask_id": "subtask_2",
                "description": "Optimize query performance and database indexes",
                "skills_required": ["optimization", "database", "performance"],
                "agent_type": "dba",
                "priority": "medium",
                "estimated_duration": 180,
                "dependencies": ["subtask_1"]
            }
        ]
    }
}


class LLMOrchestrationTester:
    """Test harness for LLM-driven orchestration"""
    
    def __init__(self):
        self.db_session = None
        self.llm = None
        self.ab_tester = None
        
    async def setup(self):
        """Initialize test components"""
        logger.info("🚀 Initializing LLM Orchestration Test Environment")
        
        # Get database session
        self.db_session = next(get_db_session())
        
        # Initialize LLM
        self.llm = OrchestratorLLM(
            model=config.LLM_MODEL,
            provider=config.LLM_PROVIDER,
            temperature=0.7
        )
        
        # Initialize A/B tester
        self.ab_tester = SelectionABTest(
            db_session=self.db_session,
            enable_execution=False,  # Don't actually execute for testing
            parallel_execution=True  # Run both methods in parallel
        )
        
        logger.info("✅ Test environment ready")
    
    async def test_llm_reasoning(self):
        """Test basic LLM reasoning capabilities"""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: LLM Reasoning Capabilities")
        logger.info("="*60)
        
        # Test reasoning
        prompt = """
        Analyze this situation and provide your reasoning:
        
        We need to select an agent for a complex data analysis task.
        The task requires strong analytical skills and experience with Python.
        We have 3 agents available:
        - Agent A: Expert in Python, moderate analysis skills
        - Agent B: Expert in analysis, basic Python skills
        - Agent C: Good at both Python and analysis
        
        Which agent would you select and why?
        """
        
        response = await self.llm.generate_with_reasoning(
            prompt=prompt,
            require_confidence=True
        )
        
        logger.info(f"\n📝 LLM Response:")
        logger.info(f"Decision: {response.content}")
        logger.info(f"Reasoning: {response.reasoning}")
        logger.info(f"Confidence: {response.confidence:.2%}")
        logger.info(f"Processing Time: {response.processing_time:.2f}s")
        logger.info(f"Tokens Used: {response.tokens_used}")
        logger.info(f"Cost: ${response.cost:.4f}")
        
        return response.confidence > 0.7
    
    async def test_function_calling(self):
        """Test LLM function calling capabilities"""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: Function Calling")
        logger.info("="*60)
        
        # Create a simple function registry
        registry = FunctionRegistry()
        executor = FunctionExecutor(registry)
        
        # Register test function
        @registry.register_decorator(
            name="calculate_score",
            description="Calculate agent score based on skills",
            timeout=1.0
        )
        async def calculate_score(skill_match: float, experience: float) -> Dict[str, Any]:
            score = (skill_match * 0.6 + experience * 0.4)
            return {"score": score, "recommendation": "good" if score > 0.7 else "poor"}
        
        # Test function calling
        prompt = """
        Calculate the agent score for an agent with:
        - Skill match: 0.85
        - Experience: 0.70
        
        Use the calculate_score function to get the result.
        """
        
        functions = registry.get_for_prompt()
        
        response = await self.llm.generate_with_functions(
            prompt=prompt,
            functions=functions,
            function_executor=lambda name, params: executor.execute(name, params).then(lambda r: r.result),
            max_function_calls=3
        )
        
        logger.info(f"\n🔧 Function Calls Made: {len(response.function_calls)}")
        for fc in response.function_calls:
            logger.info(f"  - {fc['name']}({fc['parameters']})")
            logger.info(f"    Result: {fc.get('result')}")
        
        logger.info(f"\n📝 Final Response: {response.content}")
        
        return len(response.function_calls) > 0
    
    async def test_agent_selection(self, workflow_key: str = "research_analysis"):
        """Test LLM agent selection"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: LLM Agent Selection")
        logger.info("="*60)
        
        workflow = TEST_WORKFLOWS[workflow_key]
        subtask = workflow["subtasks"][0]
        
        # Initialize LLM selector
        selector = LLMAgentSelector(self.db_session, self.llm)
        
        # Test selection
        logger.info(f"\n📋 Subtask: {subtask['description']}")
        logger.info(f"   Skills Required: {subtask['skills_required']}")
        
        result = await selector.select_agent_with_reasoning(
            subtask=subtask,
            workflow_context={
                "workflow_id": 1,
                "total_subtasks": len(workflow["subtasks"]),
                "completed_subtasks": [],
                "selected_agents": []
            }
        )
        
        logger.info(f"\n🤖 Selected Agent:")
        logger.info(f"  ID: {result.agent_id}")
        logger.info(f"  Name: {result.agent_name}")
        logger.info(f"  Confidence: {result.confidence:.2%}")
        logger.info(f"  Functions Used: {', '.join(result.function_calls_made)}")
        logger.info(f"\n  Reasoning: {result.reasoning[:300]}...")
        
        if result.risk_factors:
            logger.info(f"\n  ⚠️ Risks: {', '.join(result.risk_factors)}")
        
        return result.confidence > 0.5
    
    async def test_ab_comparison(self, workflow_key: str = "code_development"):
        """Run A/B test comparison"""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: A/B Testing - Algorithmic vs LLM Selection")
        logger.info("="*60)
        
        workflow = TEST_WORKFLOWS[workflow_key]
        
        logger.info(f"\n🔬 Testing Workflow: {workflow['name']}")
        logger.info(f"   Subtasks: {len(workflow['subtasks'])}")
        
        # Run A/B test
        results = await self.ab_tester.run_comparison(
            workflow_id=1,
            subtasks=workflow["subtasks"],
            execute_tasks=True  # Simulate execution
        )
        
        # Check success criteria
        success_criteria = {
            "Success Rate Improvement": results.llm_success_rate - results.algo_success_rate > 0.15,
            "Quality Improvement": results.llm_avg_quality - results.algo_avg_quality > 0.1,
            "Reasoning Quality": results.reasoning_quality_score > 0.7,
            "Selection Time": results.llm_avg_selection_time < 5.0
        }
        
        logger.info("\n✅ Success Criteria Check:")
        for criterion, passed in success_criteria.items():
            logger.info(f"  {criterion}: {'PASS ✅' if passed else 'FAIL ❌'}")
        
        # Export results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"ab_test_results_{timestamp}.json"
        self.ab_tester.export_results(results, output_file)
        logger.info(f"\n📊 Detailed results exported to: {output_file}")
        
        return sum(success_criteria.values()) >= 3  # Pass if 3/4 criteria met
    
    async def run_full_test(self):
        """Run complete test suite"""
        logger.info("\n" + "🎯"*30)
        logger.info("AUTOMATOS AI - LLM ORCHESTRATION TEST SUITE")
        logger.info("PRD-16: Software 3.0 Transformation")
        logger.info("🎯"*30)
        
        await self.setup()
        
        test_results = {}
        
        # Test 1: Basic Reasoning
        try:
            test_results["reasoning"] = await self.test_llm_reasoning()
        except Exception as e:
            logger.error(f"Reasoning test failed: {e}")
            test_results["reasoning"] = False
        
        # Test 2: Function Calling
        try:
            test_results["function_calling"] = await self.test_function_calling()
        except Exception as e:
            logger.error(f"Function calling test failed: {e}")
            test_results["function_calling"] = False
        
        # Test 3: Agent Selection
        try:
            test_results["agent_selection"] = await self.test_agent_selection()
        except Exception as e:
            logger.error(f"Agent selection test failed: {e}")
            test_results["agent_selection"] = False
        
        # Test 4: A/B Comparison
        try:
            test_results["ab_testing"] = await self.test_ab_comparison()
        except Exception as e:
            logger.error(f"A/B testing failed: {e}")
            test_results["ab_testing"] = False
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUITE SUMMARY")
        logger.info("="*60)
        
        for test_name, passed in test_results.items():
            status = "PASS ✅" if passed else "FAIL ❌"
            logger.info(f"  {test_name.replace('_', ' ').title()}: {status}")
        
        total_passed = sum(test_results.values())
        total_tests = len(test_results)
        
        logger.info(f"\nOverall: {total_passed}/{total_tests} tests passed")
        
        if total_passed == total_tests:
            logger.info("🎉 ALL TESTS PASSED! LLM Orchestration is ready!")
        elif total_passed >= total_tests * 0.75:
            logger.info("⚠️ Most tests passed, but some issues need attention")
        else:
            logger.info("❌ Significant issues detected, further development needed")
        
        # Cleanup
        if self.db_session:
            self.db_session.close()
        
        return total_passed == total_tests
    
    async def run_single_workflow_test(self, workflow_key: str):
        """Test a single workflow"""
        await self.setup()
        
        if workflow_key not in TEST_WORKFLOWS:
            logger.error(f"Unknown workflow: {workflow_key}")
            logger.info(f"Available workflows: {list(TEST_WORKFLOWS.keys())}")
            return False
        
        result = await self.test_ab_comparison(workflow_key)
        
        if self.db_session:
            self.db_session.close()
        
        return result


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Test LLM-Driven Orchestration")
    parser.add_argument(
        "--full-test",
        action="store_true",
        help="Run full test suite"
    )
    parser.add_argument(
        "--workflow",
        type=str,
        choices=list(TEST_WORKFLOWS.keys()),
        help="Test specific workflow"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually execute tasks (requires agents in database)"
    )
    
    args = parser.parse_args()
    
    tester = LLMOrchestrationTester()
    
    try:
        if args.full_test:
            success = await tester.run_full_test()
        elif args.workflow:
            success = await tester.run_single_workflow_test(args.workflow)
        else:
            # Default: run quick test
            success = await tester.run_full_test()
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())




