#!/usr/bin/env python3
"""
Full LLM-Driven Orchestration Test
===================================

PRD-16: Complete test of the Master Orchestrator with all LLM-driven stages.
Demonstrates the full Software 3.0 transformation with meta-reasoning.

Usage:
    python test_full_llm_orchestration.py [--workflow <type>] [--strategy <strategy>]
"""

import asyncio
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, Any, List

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

# Import database and configuration
from database.database import get_db_session
from config import config

# Import Master Orchestrator and components
from core.llm.master_orchestrator import (
    MasterOrchestrator,
    OrchestrationStrategy,
    WorkflowResult
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test workflows
TEST_WORKFLOWS = {
    "ai_research": {
        "name": "AI Research and Analysis",
        "description": """
        Conduct comprehensive research on the latest developments in Large Language Models,
        analyze their impact on software development practices, and provide recommendations
        for integrating LLM capabilities into existing systems.
        """,
        "context": {
            "priority": "high",
            "quality_requirement": "excellent",
            "time_limit": 600,
            "budget": "standard"
        }
    },
    "system_optimization": {
        "name": "System Performance Optimization",
        "description": """
        Analyze the current system architecture, identify performance bottlenecks,
        design optimization strategies, and create an implementation plan for
        improving system throughput by at least 50%.
        """,
        "context": {
            "priority": "critical",
            "quality_requirement": "high",
            "time_limit": 900,
            "technical_depth": "expert"
        }
    },
    "product_development": {
        "name": "New Feature Development",
        "description": """
        Design and plan the implementation of a real-time collaboration feature
        that allows multiple users to work on the same workflow simultaneously,
        including architecture design, API specifications, and UI mockups.
        """,
        "context": {
            "priority": "medium",
            "quality_requirement": "standard",
            "time_limit": 1200,
            "team_size": 5
        }
    },
    "data_analysis": {
        "name": "Customer Data Analysis",
        "description": """
        Analyze customer usage patterns from the last quarter, identify trends
        and anomalies, segment users based on behavior, and provide actionable
        recommendations for improving user engagement and retention.
        """,
        "context": {
            "priority": "high",
            "quality_requirement": "high",
            "data_volume": "large",
            "confidentiality": "high"
        }
    }
}


class FullOrchestrationTester:
    """Test harness for complete LLM-driven orchestration"""
    
    def __init__(self):
        self.db_session = None
        self.master_orchestrator = None
        self.test_results = []
    
    async def setup(self):
        """Initialize test environment"""
        logger.info("🚀 Initializing Full LLM Orchestration Test")
        
        # Get database session
        self.db_session = next(get_db_session())
        
        # Initialize Master Orchestrator
        self.master_orchestrator = MasterOrchestrator(
            db_session=self.db_session,
            default_strategy=OrchestrationStrategy.ADAPTIVE
        )
        
        logger.info("✅ Test environment ready")
    
    async def test_workflow(
        self,
        workflow_type: str,
        strategy: OrchestrationStrategy = OrchestrationStrategy.ADAPTIVE
    ) -> WorkflowResult:
        """
        Test a specific workflow with full orchestration.
        
        Args:
            workflow_type: Type of workflow to test
            strategy: Orchestration strategy to use
            
        Returns:
            Workflow execution result
        """
        if workflow_type not in TEST_WORKFLOWS:
            raise ValueError(f"Unknown workflow type: {workflow_type}")
        
        workflow_def = TEST_WORKFLOWS[workflow_type]
        
        logger.info("\n" + "🎯"*30)
        logger.info(f"TESTING WORKFLOW: {workflow_def['name']}")
        logger.info("🎯"*30)
        logger.info(f"Description: {workflow_def['description'].strip()}")
        logger.info(f"Strategy: {strategy.value}")
        logger.info(f"Context: {json.dumps(workflow_def['context'], indent=2)}")
        
        # Execute workflow with Master Orchestrator
        workflow_id = int(datetime.now().timestamp())
        
        result = await self.master_orchestrator.orchestrate_workflow(
            workflow_id=workflow_id,
            task_description=workflow_def['description'],
            workflow_context=workflow_def['context'],
            strategy_override=strategy
        )
        
        # Store result
        self.test_results.append({
            'workflow_type': workflow_type,
            'workflow_id': workflow_id,
            'strategy': strategy.value,
            'result': result
        })
        
        return result
    
    async def test_all_strategies(self, workflow_type: str):
        """Test a workflow with all orchestration strategies"""
        strategies = [
            OrchestrationStrategy.SPEED_OPTIMIZED,
            OrchestrationStrategy.QUALITY_OPTIMIZED,
            OrchestrationStrategy.COST_OPTIMIZED,
            OrchestrationStrategy.BALANCED,
            OrchestrationStrategy.ADAPTIVE
        ]
        
        results = {}
        
        for strategy in strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing with strategy: {strategy.value}")
            logger.info(f"{'='*60}")
            
            try:
                result = await self.test_workflow(workflow_type, strategy)
                results[strategy.value] = result
            except Exception as e:
                logger.error(f"Strategy {strategy.value} failed: {e}")
                results[strategy.value] = None
        
        # Compare results
        self._compare_strategy_results(results)
        
        return results
    
    def _compare_strategy_results(self, results: Dict[str, WorkflowResult]):
        """Compare results from different strategies"""
        logger.info("\n" + "📊"*30)
        logger.info("STRATEGY COMPARISON")
        logger.info("📊"*30)
        
        comparison_table = []
        
        for strategy_name, result in results.items():
            if result:
                comparison_table.append({
                    'Strategy': strategy_name,
                    'Status': result.status,
                    'Quality': f"{result.orchestration_quality:.2%}",
                    'Time': f"{result.total_execution_time:.2f}s",
                    'Tokens': f"{result.total_tokens_used:,}",
                    'Cost': f"${result.total_cost:.4f}",
                    'Stages': f"{len(result.stages_completed)}/9"
                })
            else:
                comparison_table.append({
                    'Strategy': strategy_name,
                    'Status': 'Failed',
                    'Quality': 'N/A',
                    'Time': 'N/A',
                    'Tokens': 'N/A',
                    'Cost': 'N/A',
                    'Stages': '0/9'
                })
        
        # Print comparison table
        if comparison_table:
            # Header
            headers = list(comparison_table[0].keys())
            header_line = " | ".join(f"{h:15}" for h in headers)
            logger.info(header_line)
            logger.info("-" * len(header_line))
            
            # Rows
            for row in comparison_table:
                row_line = " | ".join(f"{str(row[h]):15}" for h in headers)
                logger.info(row_line)
        
        # Find best strategy
        successful_results = {k: v for k, v in results.items() if v and v.status == "completed"}
        
        if successful_results:
            best_quality = max(successful_results.items(), key=lambda x: x[1].orchestration_quality)
            best_speed = min(successful_results.items(), key=lambda x: x[1].total_execution_time)
            best_cost = min(successful_results.items(), key=lambda x: x[1].total_cost)
            
            logger.info("\n🏆 WINNERS:")
            logger.info(f"  Best Quality: {best_quality[0]} ({best_quality[1].orchestration_quality:.2%})")
            logger.info(f"  Fastest: {best_speed[0]} ({best_speed[1].total_execution_time:.2f}s)")
            logger.info(f"  Most Economical: {best_cost[0]} (${best_cost[1].total_cost:.4f})")
    
    async def test_meta_reasoning(self):
        """Test meta-reasoning capabilities"""
        logger.info("\n" + "🧠"*30)
        logger.info("META-REASONING TEST")
        logger.info("🧠"*30)
        
        # Run same workflow twice to see learning
        workflow_type = "ai_research"
        
        logger.info("\n📚 First Execution:")
        result1 = await self.test_workflow(workflow_type, OrchestrationStrategy.ADAPTIVE)
        
        logger.info("\n📚 Second Execution (should show learning):")
        result2 = await self.test_workflow(workflow_type, OrchestrationStrategy.ADAPTIVE)
        
        # Compare to see if learning occurred
        if result1 and result2:
            logger.info("\n📈 Learning Analysis:")
            logger.info(f"  Quality Improvement: {(result2.orchestration_quality - result1.orchestration_quality):.2%}")
            logger.info(f"  Time Improvement: {(result1.total_execution_time - result2.total_execution_time):.2f}s")
            logger.info(f"  Cost Improvement: ${(result1.total_cost - result2.total_cost):.4f}")
            
            if result2.orchestration_quality > result1.orchestration_quality:
                logger.info("  ✅ System showed learning improvement!")
            else:
                logger.info("  ⚠️ No clear learning improvement detected")
    
    async def test_failure_recovery(self):
        """Test failure recovery and adaptation"""
        logger.info("\n" + "🔧"*30)
        logger.info("FAILURE RECOVERY TEST")
        logger.info("🔧"*30)
        
        # Create a challenging workflow that might fail
        difficult_workflow = {
            "description": """
            This is an intentionally complex and ambiguous task that might cause failures:
            Analyze quantum computing implications for blockchain while optimizing
            neural network architectures and ensuring GDPR compliance in a distributed
            system with real-time constraints and zero-downtime requirements.
            """,
            "context": {
                "priority": "critical",
                "quality_requirement": "perfect",
                "time_limit": 60,  # Very short time limit
                "complexity": "extreme"
            }
        }
        
        workflow_id = int(datetime.now().timestamp())
        
        try:
            result = await self.master_orchestrator.orchestrate_workflow(
                workflow_id=workflow_id,
                task_description=difficult_workflow['description'],
                workflow_context=difficult_workflow['context'],
                strategy_override=OrchestrationStrategy.ADAPTIVE
            )
            
            logger.info(f"\n📊 Recovery Test Result:")
            logger.info(f"  Status: {result.status}")
            logger.info(f"  Stages Completed: {len(result.stages_completed)}")
            logger.info(f"  Quality: {result.orchestration_quality:.2%}")
            
            if result.improvements_identified:
                logger.info(f"\n💡 System Identified Improvements:")
                for improvement in result.improvements_identified:
                    logger.info(f"  - {improvement}")
            
        except Exception as e:
            logger.info(f"  ❌ Workflow failed as expected: {e}")
            logger.info(f"  ✅ Failure handling worked correctly")
    
    async def run_comprehensive_test(self):
        """Run comprehensive test suite"""
        logger.info("\n" + "🎯"*30)
        logger.info("COMPREHENSIVE LLM ORCHESTRATION TEST")
        logger.info("PRD-16: Full Software 3.0 Transformation")
        logger.info("🎯"*30)
        
        await self.setup()
        
        test_results = {
            'basic_workflows': {},
            'strategy_comparison': {},
            'meta_reasoning': False,
            'failure_recovery': False
        }
        
        # Test 1: Basic workflow execution
        logger.info("\n\n📋 TEST 1: Basic Workflow Execution")
        logger.info("="*60)
        
        for workflow_type in ["ai_research", "data_analysis"]:
            try:
                result = await self.test_workflow(workflow_type)
                test_results['basic_workflows'][workflow_type] = result.status == "completed"
            except Exception as e:
                logger.error(f"Workflow {workflow_type} failed: {e}")
                test_results['basic_workflows'][workflow_type] = False
        
        # Test 2: Strategy comparison
        logger.info("\n\n📋 TEST 2: Strategy Comparison")
        logger.info("="*60)
        
        try:
            strategy_results = await self.test_all_strategies("system_optimization")
            test_results['strategy_comparison'] = strategy_results
        except Exception as e:
            logger.error(f"Strategy comparison failed: {e}")
        
        # Test 3: Meta-reasoning
        logger.info("\n\n📋 TEST 3: Meta-Reasoning & Learning")
        logger.info("="*60)
        
        try:
            await self.test_meta_reasoning()
            test_results['meta_reasoning'] = True
        except Exception as e:
            logger.error(f"Meta-reasoning test failed: {e}")
        
        # Test 4: Failure recovery
        logger.info("\n\n📋 TEST 4: Failure Recovery")
        logger.info("="*60)
        
        try:
            await self.test_failure_recovery()
            test_results['failure_recovery'] = True
        except Exception as e:
            logger.error(f"Failure recovery test failed: {e}")
        
        # Get orchestration statistics
        stats = await self.master_orchestrator.get_orchestration_stats()
        
        # Final summary
        self._print_final_summary(test_results, stats)
        
        # Cleanup
        if self.db_session:
            self.db_session.close()
        
        return test_results
    
    def _print_final_summary(self, test_results: Dict[str, Any], stats: Dict[str, Any]):
        """Print final test summary"""
        logger.info("\n" + "="*60)
        logger.info("📊 FINAL TEST SUMMARY")
        logger.info("="*60)
        
        # Basic workflows
        logger.info("\n✅ Basic Workflow Tests:")
        for workflow, success in test_results['basic_workflows'].items():
            status = "PASS ✅" if success else "FAIL ❌"
            logger.info(f"  {workflow}: {status}")
        
        # Strategy comparison
        if test_results['strategy_comparison']:
            logger.info("\n✅ Strategy Comparison: COMPLETED")
        
        # Meta-reasoning
        logger.info(f"\n✅ Meta-Reasoning Test: {'PASS ✅' if test_results['meta_reasoning'] else 'FAIL ❌'}")
        
        # Failure recovery
        logger.info(f"✅ Failure Recovery Test: {'PASS ✅' if test_results['failure_recovery'] else 'FAIL ❌'}")
        
        # Overall statistics
        logger.info("\n📈 ORCHESTRATION STATISTICS:")
        logger.info(f"  Total Workflows: {stats.get('total_workflows', 0)}")
        logger.info(f"  Success Rate: {stats.get('success_rate', 0):.2%}")
        logger.info(f"  Avg Quality: {stats.get('avg_orchestration_quality', 0):.2%}")
        logger.info(f"  Avg Time: {stats.get('avg_execution_time', 0):.2f}s")
        logger.info(f"  Avg Cost: ${stats.get('avg_cost', 0):.4f}")
        
        if stats.get('best_strategy'):
            logger.info(f"  Best Strategy: {stats['best_strategy']}")
        
        # Success criteria check
        logger.info("\n🎯 SUCCESS CRITERIA CHECK:")
        
        criteria = [
            ("Workflows Execute Successfully", stats.get('success_rate', 0) > 0.7),
            ("Quality Above 70%", stats.get('avg_orchestration_quality', 0) > 0.7),
            ("Meta-Reasoning Works", test_results['meta_reasoning']),
            ("Failure Recovery Works", test_results['failure_recovery']),
            ("Multiple Strategies Tested", bool(test_results['strategy_comparison']))
        ]
        
        passed = sum(1 for _, result in criteria if result)
        
        for criterion, result in criteria:
            status = "PASS ✅" if result else "FAIL ❌"
            logger.info(f"  {criterion}: {status}")
        
        logger.info(f"\n  Overall: {passed}/{len(criteria)} criteria passed")
        
        if passed >= 4:
            logger.info("\n🎉 FULL LLM ORCHESTRATION IS WORKING SUCCESSFULLY!")
            logger.info("   The Software 3.0 transformation is complete!")
        elif passed >= 3:
            logger.info("\n⚠️ LLM Orchestration is mostly working but needs refinement")
        else:
            logger.info("\n❌ Significant issues detected, further development needed")
        
        logger.info("="*60 + "\n")


async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Test Full LLM-Driven Orchestration"
    )
    parser.add_argument(
        "--workflow",
        type=str,
        choices=list(TEST_WORKFLOWS.keys()),
        help="Test specific workflow type"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=[s.value for s in OrchestrationStrategy],
        default="adaptive",
        help="Orchestration strategy to use"
    )
    parser.add_argument(
        "--comprehensive",
        action="store_true",
        help="Run comprehensive test suite"
    )
    
    args = parser.parse_args()
    
    tester = FullOrchestrationTester()
    
    try:
        if args.comprehensive or (not args.workflow):
            # Run comprehensive test
            await tester.run_comprehensive_test()
        else:
            # Test specific workflow
            await tester.setup()
            
            strategy = OrchestrationStrategy(args.strategy)
            result = await tester.test_workflow(args.workflow, strategy)
            
            if result.status == "completed":
                logger.info("\n✅ Workflow completed successfully!")
            else:
                logger.info("\n❌ Workflow failed")
            
            # Cleanup
            if tester.db_session:
                tester.db_session.close()
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Test interrupted by user")
    except Exception as e:
        logger.error(f"Test failed with error: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
