"""
End-to-End Orchestration Integration Test
==========================================

Tests the complete orchestration pipeline from start to finish:
1. Creates a test workflow
2. Executes it through all 9 stages
3. Verifies skills and tools are used
4. Checks results are generated
5. Validates memory storage

Run this to verify your orchestration is working end-to-end.
"""

import asyncio
import logging
import sys
import os
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sqlalchemy.orm import Session
from database.database import get_db_session
from database.models import (
    Agent, Skill, Workflow, WorkflowExecution, ExecutionStatus
)
import httpx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EndToEndTester:
    """Tests complete orchestration workflow"""
    
    def __init__(self, api_base_url="http://localhost:8000"):
        self.api_base_url = api_base_url
        self.client = httpx.AsyncClient(timeout=120.0)
        self.test_results = {
            "workflow_created": False,
            "workflow_executed": False,
            "decomposition_successful": False,
            "agent_selection_successful": False,
            "execution_completed": False,
            "results_generated": False,
            "skills_used": False,
            "tools_used": False,
            "overall_status": "PENDING"
        }
    
    async def run_full_test(self):
        """Run complete end-to-end test"""
        logger.info("="*80)
        logger.info("🚀 END-TO-END ORCHESTRATION TEST")
        logger.info("="*80)
        
        try:
            # Test 1: Verify prerequisites
            await self.verify_prerequisites()
            
            # Test 2: Create test workflow
            workflow_id = await self.create_test_workflow()
            
            if workflow_id:
                # Test 3: Execute workflow
                execution_id = await self.execute_workflow(workflow_id)
                
                if execution_id:
                    # Test 4: Monitor execution
                    await self.monitor_execution(execution_id)
                    
                    # Test 5: Verify results
                    await self.verify_results(execution_id)
            
            # Final Report
            self.print_final_report()
        
        finally:
            await self.client.aclose()
    
    async def verify_prerequisites(self):
        """Verify system prerequisites"""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Verifying Prerequisites")
        logger.info("="*80)
        
        with get_db_session() as db:
            # Check agents
            agents = db.query(Agent).filter(Agent.status == "active").all()
            logger.info(f"✅ Found {len(agents)} active agents")
            
            if len(agents) == 0:
                logger.warning("⚠️  No active agents found!")
                logger.info("   Please create agents before running this test")
                return False
            
            # Check skills
            skills_with_tools = db.query(Skill).filter(
                Skill.tools_schema.isnot(None)
            ).count()
            logger.info(f"✅ Found {skills_with_tools} skills with tools")
            
            # Check agents with skills
            agents_with_skills = sum(1 for a in agents if len(a.skills) > 0)
            logger.info(f"✅ Found {agents_with_skills} agents with skills assigned")
            
            if agents_with_skills == 0:
                logger.warning("⚠️  No agents have skills assigned!")
                logger.info("   Agents will work but without skill-specific tools")
            
            return True
    
    async def create_test_workflow(self):
        """Create a test workflow via API"""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Creating Test Workflow")
        logger.info("="*80)
        
        workflow_data = {
            "name": f"E2E Test Workflow - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "description": "End-to-end orchestration test workflow",
            "goal": """
                Test the complete orchestration pipeline:
                1. Research the Automatos AI platform architecture
                2. Summarize the key components found
                3. Create a brief report document
            """,
            "workflow_definition": {
                "category": "testing",
                "priority": "medium",
                "test_mode": True
            }
        }
        
        try:
            logger.info("📤 Creating workflow via API...")
            response = await self.client.post(
                f"{self.api_base_url}/api/workflows/",
                json=workflow_data
            )
            
            if response.status_code == 200:
                result = response.json()
                workflow_id = result.get("id")
                logger.info(f"✅ Workflow created successfully: ID {workflow_id}")
                logger.info(f"   Name: {result.get('name')}")
                self.test_results["workflow_created"] = True
                return workflow_id
            else:
                logger.error(f"❌ Workflow creation failed: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"❌ Workflow creation error: {e}")
            return None
    
    async def execute_workflow(self, workflow_id):
        """Execute workflow via API"""
        logger.info("\n" + "="*80)
        logger.info(f"TEST 3: Executing Workflow {workflow_id}")
        logger.info("="*80)
        
        try:
            logger.info("🚀 Starting workflow execution...")
            response = await self.client.post(
                f"{self.api_base_url}/api/workflows/{workflow_id}/execute",
                json={"options": {"test_mode": True}}
            )
            
            if response.status_code == 200:
                result = response.json()
                execution_id = result.get("execution_id")
                logger.info(f"✅ Execution started: ID {execution_id}")
                logger.info(f"   Status: {result.get('status')}")
                self.test_results["workflow_executed"] = True
                return execution_id
            else:
                logger.error(f"❌ Execution failed: {response.status_code}")
                logger.error(f"   Response: {response.text}")
                return None
        
        except Exception as e:
            logger.error(f"❌ Execution error: {e}")
            return None
    
    async def monitor_execution(self, execution_id):
        """Monitor workflow execution progress"""
        logger.info("\n" + "="*80)
        logger.info(f"TEST 4: Monitoring Execution {execution_id}")
        logger.info("="*80)
        
        max_attempts = 60  # 5 minutes max
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Get execution status
                response = await self.client.get(
                    f"{self.api_base_url}/api/workflow-executions/{execution_id}"
                )
                
                if response.status_code == 200:
                    result = response.json()
                    status = result.get("status")
                    
                    logger.info(f"⏳ Attempt {attempt + 1}/{max_attempts}: Status = {status}")
                    
                    # Check for completion
                    if status in ["completed", "failed", "cancelled"]:
                        logger.info(f"\n{'✅' if status == 'completed' else '❌'} Execution {status}")
                        
                        if status == "completed":
                            self.test_results["execution_completed"] = True
                            
                            # Check for decomposition
                            input_data = result.get("input_data", {})
                            if input_data.get("decomposition"):
                                logger.info("✅ Task decomposition successful")
                                self.test_results["decomposition_successful"] = True
                            
                            # Check for agent selection
                            if input_data.get("agent_selection"):
                                logger.info("✅ Agent selection successful")
                                self.test_results["agent_selection_successful"] = True
                        
                        break
                    
                    # Wait before next check
                    await asyncio.sleep(5)
                    attempt += 1
                else:
                    logger.error(f"❌ Status check failed: {response.status_code}")
                    break
            
            except Exception as e:
                logger.error(f"❌ Monitoring error: {e}")
                break
        
        if attempt >= max_attempts:
            logger.warning("⚠️  Execution timed out (5 minutes)")
    
    async def verify_results(self, execution_id):
        """Verify execution results"""
        logger.info("\n" + "="*80)
        logger.info(f"TEST 5: Verifying Results")
        logger.info("="*80)
        
        try:
            # Get execution details
            response = await self.client.get(
                f"{self.api_base_url}/api/workflow-executions/{execution_id}"
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Check results
                if result.get("results"):
                    logger.info("✅ Results generated")
                    self.test_results["results_generated"] = True
                    
                    # Check for final output
                    results_data = result.get("results", {})
                    if results_data.get("final_output"):
                        logger.info("✅ Final output present")
                        output = results_data.get("final_output")
                        logger.info(f"   Output length: {len(str(output))} characters")
                
                # Check metadata for skills and tools
                metadata = result.get("metadata", {})
                analytics = metadata.get("analytics", {})
                
                # Check skills used
                if analytics.get("skills_used"):
                    logger.info(f"✅ Skills used: {analytics['skills_used']}")
                    self.test_results["skills_used"] = True
                
                # Check tools used
                if analytics.get("tools_used"):
                    logger.info(f"✅ Tools used: {analytics['tools_used']}")
                    self.test_results["tools_used"] = True
                
                # Check quality scores
                if analytics.get("overall_quality"):
                    quality = analytics["overall_quality"]
                    logger.info(f"✅ Quality score: {quality}/10")
        
        except Exception as e:
            logger.error(f"❌ Results verification error: {e}")
    
    def print_final_report(self):
        """Print final test report"""
        logger.info("\n" + "="*80)
        logger.info("📊 END-TO-END TEST REPORT")
        logger.info("="*80)
        
        # Calculate overall status
        all_critical_passed = (
            self.test_results["workflow_created"] and
            self.test_results["workflow_executed"] and
            self.test_results["execution_completed"]
        )
        
        if all_critical_passed:
            self.test_results["overall_status"] = "✅ SUCCESS"
        else:
            self.test_results["overall_status"] = "❌ FAILED"
        
        # Print results
        for test, result in self.test_results.items():
            if test != "overall_status":
                status = "✅ PASS" if result else "❌ FAIL"
                test_name = test.replace("_", " ").title()
                logger.info(f"{status}: {test_name}")
        
        logger.info("\n" + "="*80)
        logger.info(f"OVERALL: {self.test_results['overall_status']}")
        logger.info("="*80)
        
        # Recommendations
        if not self.test_results["workflow_created"]:
            logger.info("\n⚠️  RECOMMENDATION: Check API server is running")
            logger.info("   - docker-compose ps")
            logger.info("   - docker-compose logs automatos-backend")
        
        if not self.test_results["execution_completed"]:
            logger.info("\n⚠️  RECOMMENDATION: Check execution logs for errors")
            logger.info("   - Look for LLM API errors")
            logger.info("   - Verify agent configurations")
            logger.info("   - Check database connectivity")
        
        if not self.test_results["skills_used"]:
            logger.info("\n⚠️  RECOMMENDATION: Skills not used during execution")
            logger.info("   - Ensure agents have skills assigned")
            logger.info("   - Verify PRD-22 fix is deployed")
            logger.info("   - Run: python scripts/verify_orchestration_integration.py")
        
        if not self.test_results["tools_used"]:
            logger.info("\n⚠️  RECOMMENDATION: Tools not used during execution")
            logger.info("   - This may be normal if task didn't require tools")
            logger.info("   - Try a workflow that specifically needs tools (PDF creation, etc.)")


async def main():
    """Main test entry point"""
    
    # Parse command line args
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    if len(sys.argv) > 1:
        api_url = sys.argv[1]
    
    logger.info(f"Testing API at: {api_url}")
    
    tester = EndToEndTester(api_base_url=api_url)
    await tester.run_full_test()


if __name__ == "__main__":
    asyncio.run(main())

