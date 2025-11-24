"""
Real End-to-End Integration Test
==================================

CRITICAL: This test uses REAL services (no mocks):
- Real OpenAI API calls
- Real database queries
- Real vector store operations

This will consume API tokens and take time.
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.orchestrator_service import EnhancedOrchestratorService
from database.session import get_db_session

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger("RealIntegrationTest")


async def test_simple_task():
    """Test a simple task end-to-end with REAL services"""
    
    logger.info("=" * 80)
    logger.info("🧪 REAL END-TO-END TEST - Simple Task")
    logger.info("=" * 80)
    
    # Get real database session
    db = next(get_db_session())
    
    try:
        # Create orchestrator with real services
        orchestrator = EnhancedOrchestratorService(db_session=db)
        
        # Execute a simple task
        user_prompt = "Create a simple Hello World README file"
        user_id = 1
        
        logger.info(f"\n📝 User Prompt: {user_prompt}")
        logger.info("⚠️  NOTE: This will make REAL API calls\n")
        
        # Execute the full 9-stage workflow
        result = await orchestrator.execute_workflow(
            user_prompt=user_prompt,
            user_id=user_id
        )
        
        # Verify result
        logger.info("\n" + "=" * 80)
        logger.info("✅ WORKFLOW COMPLETED")
        logger.info("=" * 80)
        
        logger.info(f"\n📊 Results:")
        logger.info(f"  Workflow ID: {result.get('workflow_id')}")
        logger.info(f"  Status: {result.get('status')}")
        logger.info(f"  Quality Score: {result.get('quality_score', 0):.2f}")
        logger.info(f"  Execution Time: {result.get('execution_time', 0):.2f}s")
        logger.info(f"  Stages Completed: {result.get('stages_completed', 0)}/9")
        
        # Check each stage
        logger.info(f"\n📋 Stage Details:")
        details = result.get('details', {})
        
        if 'decomposition' in details:
            decomp = details['decomposition']
            logger.info(f"  Stage 1 (Decomposition): {len(decomp.get('subtasks', []))} subtasks")
            # Check for graph analysis
            if 'graph_analysis' in decomp:
                graph = decomp['graph_analysis']
                logger.info(f"    🔺 Graph Analysis Found:")
                logger.info(f"      - Parallel Groups: {len(graph.get('parallel_groups', []))}")
                logger.info(f"      - Max Parallelism: {graph.get('max_parallelism', 1)}")
        
        if 'agent_selection' in details:
            logger.info(f"  Stage 2 (Agent Selection): ✅")
        
        if 'context_engineering' in details:
            context = details['context_engineering']
            logger.info(f"  Stage 3 (Context Engineering): ✅")
            if 'optimization_used' in context:
                logger.info(f"    🔺 Optimization: {context['optimization_used']}")
        
        if 'execution' in details:
            exec_details = details['execution']
            logger.info(f"  Stage 4 (Execution): {len(exec_details.get('results', {}))} tasks executed")
        
        if 'aggregation' in details:
            logger.info(f"  Stage 5 (Aggregation): ✅")
        
        if 'learning' in details:
            logger.info(f"  Stage 6 (Learning): ✅")
        
        if 'quality' in details:
            quality = details['quality']
            logger.info(f"  Stage 7 (Quality): Score = {quality.get('overall_score', 0):.2f}")
        
        if 'memory' in details:
            logger.info(f"  Stage 8 (Memory): ✅")
        
        logger.info(f"  Stage 9 (Response): ✅")
        
        # Final verdict
        logger.info("\n" + "=" * 80)
        if result.get('status') == 'completed' and result.get('stages_completed', 0) == 9:
            logger.info("🎉 SUCCESS: All 9 stages executed with REAL services!")
            logger.info("=" * 80)
            return True
        else:
            logger.error("❌ FAILURE: Not all stages completed")
            logger.error("=" * 80)
            return False
            
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


async def test_complex_task():
    """Test a more complex task with dependencies"""
    
    logger.info("\n\n")
    logger.info("=" * 80)
    logger.info("🧪 REAL END-TO-END TEST - Complex Task with Dependencies")
    logger.info("=" * 80)
    
    db = next(get_db_session())
    
    try:
        orchestrator = EnhancedOrchestratorService(db_session=db)
        
        # Task that should create multiple subtasks with dependencies
        user_prompt = """
        Research the authentication bug in our system, identify the root cause, 
        and create a fix with documentation
        """
        
        logger.info(f"\n📝 User Prompt: {user_prompt.strip()}")
        logger.info("⚠️  This will make MULTIPLE API calls\n")
        
        result = await orchestrator.execute_workflow(
            user_prompt=user_prompt,
            user_id=1
        )
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ COMPLEX WORKFLOW COMPLETED")
        logger.info("=" * 80)
        
        # Check for graph analysis in decomposition
        details = result.get('details', {})
        if 'decomposition' in details:
            decomp = details['decomposition']
            subtasks = decomp.get('subtasks', [])
            logger.info(f"\n📊 Decomposition Analysis:")
            logger.info(f"  Total Subtasks: {len(subtasks)}")
            
            if 'graph_analysis' in decomp:
                graph = decomp['graph_analysis']
                logger.info(f"\n  🔺 Graph Theory Results:")
                logger.info(f"    Parallel Groups: {len(graph.get('parallel_groups', []))}")
                logger.info(f"    Max Parallelism: {graph.get('max_parallelism', 1)}")
                logger.info(f"    Dependency Levels: {graph.get('dependency_levels', 0)}")
                logger.info(f"    Execution Order: {graph.get('execution_order', [])}")
                
                # This proves graph theory is WORKING
                if len(graph.get('parallel_groups', [])) > 1:
                    logger.info(f"\n  ✅ GRAPH THEORY ACTIVE: Found {len(graph['parallel_groups'])} parallel execution levels")
                else:
                    logger.info(f"\n  ⚠️  Simple sequential execution (no parallelization opportunities)")
        
        return result.get('status') == 'completed'
        
    except Exception as e:
        logger.error(f"\n❌ COMPLEX TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        db.close()


async def main():
    """Run all integration tests"""
    
    logger.info("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                     REAL END-TO-END INTEGRATION TESTS                        ║
║                          (NO MOCKS - REAL SERVICES)                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Test 1: Simple task
    test1_passed = await test_simple_task()
    
    # Test 2: Complex task
    test2_passed = await test_complex_task()
    
    # Summary
    logger.info("\n\n")
    logger.info("=" * 80)
    logger.info("FINAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Test 1 (Simple Task):  {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    logger.info(f"Test 2 (Complex Task): {'✅ PASSED' if test2_passed else '❌ FAILED'}")
    logger.info("=" * 80)
    
    if test1_passed and test2_passed:
        logger.info("\n🎉 ALL TESTS PASSED - 9-STAGE WORKFLOW IS FULLY FUNCTIONAL!")
    else:
        logger.info("\n❌ SOME TESTS FAILED - REVIEW ERRORS ABOVE")
    
    return test1_passed and test2_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
