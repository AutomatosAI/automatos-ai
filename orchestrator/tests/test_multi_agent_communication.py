#!/usr/bin/env python3
"""
Test Multi-Agent Communication in Workflows
============================================

This script tests agent communication during workflow execution.
Part of Week 1 Day 5: Agent Communication testing.
"""

import asyncio
import logging
import os
import sys
import json
import requests
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API configuration
API_BASE_URL = "https://api.automatos.app/api"
API_KEY = os.getenv("API_KEY", "test_api_key")


class MultiAgentCommunicationTester:
    """Test multi-agent workflows with communication"""
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "Content-Type": "application/json",
            "X-API-Key": API_KEY
        })
        self.created_agents = []
        self.created_workflows = []
        self.test_results = []
    
    def get_existing_agents(self) -> List[Dict]:
        """Get existing agents from the system"""
        try:
            response = self.session.get(f"{API_BASE_URL}/agents")
            if response.status_code == 200:
                agents = response.json()
                logger.info(f"Found {len(agents)} existing agents")
                return agents
            else:
                logger.warning(f"Failed to get agents: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting agents: {e}")
            return []
    
    def create_multi_agent_workflow(self) -> Dict[str, Any]:
        """Create a workflow that requires multiple agents to collaborate"""
        
        workflow_data = {
            "name": f"Multi-Agent Code Review - {datetime.now().strftime('%H%M%S')}",
            "description": "Comprehensive code review requiring collaboration between security, performance, and documentation experts",
            "workflow_definition": {
                "category": "code_review",
                "execution_strategy": "collaborative",
                "steps": [
                    {
                        "id": "security_analysis",
                        "name": "Security Analysis",
                        "description": "Analyze code for security vulnerabilities and best practices",
                        "agent_type": "security_expert",
                        "skills_required": ["security_audit", "vulnerability_scanning"],
                        "priority": "high",
                        "dependencies": []
                    },
                    {
                        "id": "performance_review",
                        "name": "Performance Review",
                        "description": "Review code for performance optimizations and bottlenecks",
                        "agent_type": "performance_optimizer",
                        "skills_required": ["performance_analysis", "optimization"],
                        "priority": "high",
                        "dependencies": []
                    },
                    {
                        "id": "documentation_check",
                        "name": "Documentation Check",
                        "description": "Verify code documentation and generate missing docs",
                        "agent_type": "documentation_writer",
                        "skills_required": ["technical_writing", "api_documentation"],
                        "priority": "medium",
                        "dependencies": []
                    },
                    {
                        "id": "final_report",
                        "name": "Final Report Generation",
                        "description": "Compile findings from all agents into comprehensive report",
                        "agent_type": "general",
                        "skills_required": ["analysis", "report_generation"],
                        "priority": "high",
                        "dependencies": ["security_analysis", "performance_review", "documentation_check"]
                    }
                ],
                "config": {
                    "enable_communication": True,
                    "enable_memory": True,
                    "max_parallel": 3,
                    "timeout": 600
                }
            },
            "status": "active"
        }
        
        try:
            response = self.session.post(f"{API_BASE_URL}/workflows", json=workflow_data)
            if response.status_code in [200, 201]:
                workflow = response.json()
                self.created_workflows.append(workflow["id"])
                logger.info(f"✅ Created multi-agent workflow: {workflow['name']} (ID: {workflow['id']})")
                return workflow
            else:
                logger.error(f"Failed to create workflow: {response.status_code}")
                logger.error(f"Response: {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error creating workflow: {e}")
            return None
    
    def execute_workflow(self, workflow_id: int) -> Dict[str, Any]:
        """Execute workflow and monitor communication"""
        
        execution_data = {
            "workflow_id": workflow_id,
            "input_data": {
                "code_to_review": """
                def process_user_input(user_data):
                    # Process user data without validation
                    query = f"SELECT * FROM users WHERE id = {user_data['id']}"
                    result = db.execute(query)
                    return result
                """,
                "context": "Python web application code",
                "requirements": [
                    "Check for security vulnerabilities",
                    "Identify performance issues",
                    "Verify documentation completeness"
                ]
            }
        }
        
        try:
            # Start workflow execution
            response = self.session.post(
                f"{API_BASE_URL}/workflows/execute",
                json=execution_data
            )
            
            if response.status_code in [200, 201, 202]:
                execution = response.json()
                execution_id = execution.get("execution_id", execution.get("id"))
                logger.info(f"✅ Workflow execution started: ID {execution_id}")
                
                # Poll for completion
                return self.poll_execution_status(execution_id)
            else:
                logger.error(f"Failed to execute workflow: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return None
    
    def poll_execution_status(self, execution_id: int, max_wait: int = 120) -> Dict[str, Any]:
        """Poll execution status until complete"""
        
        import time
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = self.session.get(f"{API_BASE_URL}/workflows/executions/{execution_id}")
                if response.status_code == 200:
                    execution = response.json()
                    status = execution.get("status", "unknown")
                    
                    if status in ["completed", "failed"]:
                        logger.info(f"Execution {status}: {execution_id}")
                        return execution
                    
                    logger.info(f"Execution status: {status}... waiting")
                    time.sleep(5)
                else:
                    logger.warning(f"Failed to get execution status: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error polling execution: {e}")
            
            time.sleep(5)
        
        logger.error(f"Execution timed out after {max_wait} seconds")
        return None
    
    def analyze_communication(self, execution_result: Dict[str, Any]):
        """Analyze agent communication from execution results"""
        
        logger.info("\n" + "=" * 60)
        logger.info("AGENT COMMUNICATION ANALYSIS")
        logger.info("=" * 60)
        
        # Check if communication was enabled
        input_data = execution_result.get("input_data", {})
        
        # Look for communication logs
        if "agent_communications" in input_data:
            communications = input_data["agent_communications"]
            logger.info(f"Found {len(communications)} agent communications")
            
            for comm in communications[:5]:  # Show first 5
                logger.info(f"  📨 {comm.get('from_agent')} → {comm.get('to_agent')}")
                logger.info(f"     Type: {comm.get('message_type')}")
                logger.info(f"     Content: {comm.get('content', '')[:100]}...")
        
        # Check shared context
        if "shared_context" in input_data:
            context = input_data["shared_context"]
            logger.info(f"\n📋 Shared Context:")
            logger.info(f"  Team size: {context.get('team_size', 0)} agents")
            logger.info(f"  Updates: {context.get('update_count', 0)}")
        
        # Check execution logs for communication
        exec_log = execution_result.get("execution_log", "")
        if "communication" in exec_log.lower() or "message" in exec_log.lower():
            logger.info("\n✅ Communication detected in execution logs")
        
        return True
    
    async def run_all_tests(self):
        """Run all multi-agent communication tests"""
        
        logger.info("=" * 60)
        logger.info("MULTI-AGENT COMMUNICATION TEST SUITE")
        logger.info("=" * 60)
        
        # Test 1: Get existing agents
        logger.info("\n📦 TEST 1: Check existing agents...")
        agents = self.get_existing_agents()
        if agents:
            logger.info(f"✅ Found {len(agents)} existing agents:")
            for agent in agents[:5]:  # Show first 5
                logger.info(f"  - {agent.get('name')} (ID: {agent.get('id')})")
            self.test_results.append(("Get Agents", True, f"Found {len(agents)} agents"))
        else:
            logger.warning("⚠️ No existing agents found")
            self.test_results.append(("Get Agents", False, "No agents found"))
        
        # Test 2: Create multi-agent workflow
        logger.info("\n🔧 TEST 2: Create multi-agent workflow...")
        workflow = self.create_multi_agent_workflow()
        if workflow:
            self.test_results.append(("Create Workflow", True, f"Created workflow {workflow['id']}"))
        else:
            self.test_results.append(("Create Workflow", False, "Failed to create workflow"))
            return False
        
        # Test 3: Execute workflow with communication
        logger.info("\n🚀 TEST 3: Execute workflow with agent communication...")
        execution_result = self.execute_workflow(workflow["id"])
        if execution_result:
            self.test_results.append(("Execute Workflow", True, "Workflow executed"))
            
            # Analyze communication
            self.analyze_communication(execution_result)
        else:
            self.test_results.append(("Execute Workflow", False, "Execution failed"))
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = sum(1 for _, success, _ in self.test_results if not success)
        
        for test_name, success, message in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status} - {test_name}: {message}")
        
        logger.info(f"\nTotal: {len(self.test_results)} tests")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {passed/len(self.test_results)*100:.1f}%")
        
        return failed == 0


async def main():
    """Main test runner"""
    tester = MultiAgentCommunicationTester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
