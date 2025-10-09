#!/usr/bin/env python3
"""
Comprehensive Workflow Agent Selection Test Suite
=================================================

Tests intelligent agent selection based on:
- Skills matching
- Agent types
- Availability
- Performance history

This script creates test agents with specific skills via API,
then runs workflows to validate correct agent selection.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# API Configuration
API_BASE_URL = "https://api.automatos.app/api/v1"
API_KEY = "test_api_key_for_backend_validation_2025"

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TestAgent:
    """Test agent configuration"""
    name: str
    agent_type: str
    skills: List[str]
    description: str
    model_config: Dict[str, Any]
    
    def to_api_payload(self) -> Dict:
        """Convert to API payload format"""
        return {
            "name": self.name,
            "agent_type": self.agent_type,
            "description": self.description,
            "skills": self.skills,
            "model_config": self.model_config,
            "configuration": {
                "skills": {skill: {"proficiency": 0.8 + (i * 0.05)} for i, skill in enumerate(self.skills)},
                "max_concurrent_tasks": 3,
                "preferred_task_types": [self.agent_type]
            },
            "status": "active"
        }


@dataclass 
class TestWorkflow:
    """Test workflow configuration"""
    name: str
    description: str
    category: str
    workflow_definition: Dict[str, Any]
    
    def to_api_payload(self) -> Dict:
        """Convert to API payload format"""
        return {
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "workflow_definition": self.workflow_definition,
            "status": "active"
        }


class WorkflowAgentSelectionTester:
    """Main test orchestrator"""
    
    def __init__(self):
        self.api_base = API_BASE_URL
        self.api_key = API_KEY
        self.headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json"
        }
        self.created_agents = []
        self.created_workflows = []
        self.test_results = []
    
    def api_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make API request with error handling"""
        url = f"{self.api_base}/{endpoint}"
        
        try:
            if method == "GET":
                response = requests.get(url, headers=self.headers)
            elif method == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            elif method == "PUT":
                response = requests.put(url, headers=self.headers, json=data)
            elif method == "DELETE":
                response = requests.delete(url, headers=self.headers)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            response.raise_for_status()
            return response.json() if response.content else {}
            
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            if hasattr(e.response, 'text'):
                logger.error(f"Response: {e.response.text}")
            raise
    
    async def create_test_agents(self) -> List[int]:
        """Create diverse test agents with specific skills"""
        
        test_agents = [
            # Code-related agents
            TestAgent(
                name="Python Expert",
                agent_type="code_architect",
                skills=["python", "code_review", "api_design", "testing"],
                description="Expert in Python development and code architecture",
                model_config={"model_id": "gpt-4", "temperature": 0.7}
            ),
            TestAgent(
                name="Security Specialist",
                agent_type="security_expert",
                skills=["security_audit", "vulnerability_scanning", "penetration_testing", "compliance"],
                description="Cybersecurity and vulnerability assessment specialist",
                model_config={"model_id": "gpt-4", "temperature": 0.3}
            ),
            TestAgent(
                name="Performance Optimizer",
                agent_type="performance_analyst",
                skills=["performance_analysis", "optimization", "profiling", "benchmarking"],
                description="System performance analysis and optimization expert",
                model_config={"model_id": "gpt-4", "temperature": 0.5}
            ),
            
            # Data-related agents
            TestAgent(
                name="Data Analyst",
                agent_type="data_analyst",
                skills=["data_processing", "statistics", "visualization", "sql"],
                description="Data analysis and visualization specialist",
                model_config={"model_id": "gpt-4", "temperature": 0.4}
            ),
            TestAgent(
                name="ML Engineer",
                agent_type="ml_engineer",
                skills=["machine_learning", "deep_learning", "model_training", "feature_engineering"],
                description="Machine learning model development specialist",
                model_config={"model_id": "gpt-4", "temperature": 0.6}
            ),
            
            # Documentation agents
            TestAgent(
                name="Documentation Writer",
                agent_type="documentation_expert",
                skills=["technical_writing", "api_documentation", "user_guides", "markdown"],
                description="Technical documentation specialist",
                model_config={"model_id": "gpt-4", "temperature": 0.7}
            ),
            
            # DevOps agents
            TestAgent(
                name="DevOps Engineer",
                agent_type="devops_expert",
                skills=["docker", "kubernetes", "ci_cd", "infrastructure"],
                description="DevOps and infrastructure automation specialist",
                model_config={"model_id": "gpt-4", "temperature": 0.5}
            ),
            
            # Generalist agent
            TestAgent(
                name="AI Assistant",
                agent_type="general_assistant",
                skills=["general_tasks", "research", "analysis", "communication"],
                description="General-purpose AI assistant",
                model_config={"model_id": "gpt-4", "temperature": 0.7}
            )
        ]
        
        agent_ids = []
        
        for agent in test_agents:
            logger.info(f"Creating agent: {agent.name} with skills: {agent.skills}")
            
            try:
                response = self.api_request("POST", "agents", agent.to_api_payload())
                agent_id = response.get("id")
                agent_ids.append(agent_id)
                self.created_agents.append(agent_id)
                logger.info(f"✅ Created agent: {agent.name} (ID: {agent_id})")
                
            except Exception as e:
                logger.error(f"❌ Failed to create agent {agent.name}: {e}")
        
        return agent_ids
    
    async def create_test_workflows(self) -> List[int]:
        """Create test workflows that require specific skills"""
        
        test_workflows = [
            # Code Review Workflow - Should select Python Expert + Security Specialist
            TestWorkflow(
                name="Code Security Review",
                description="Review Python code for security vulnerabilities and best practices",
                category="code_review",
                workflow_definition={
                    "steps": [
                        {
                            "name": "Code Analysis",
                            "description": "Analyze Python code structure and patterns",
                            "required_skills": ["python", "code_review"],
                            "agent_type": "code_architect",
                            "priority": "high"
                        },
                        {
                            "name": "Security Scan",
                            "description": "Scan for security vulnerabilities",
                            "required_skills": ["security_audit", "vulnerability_scanning"],
                            "agent_type": "security_expert",
                            "priority": "high"
                        }
                    ],
                    "execution_strategy": "sequential"
                }
            ),
            
            # Performance Analysis - Should select Performance Optimizer + Data Analyst
            TestWorkflow(
                name="System Performance Analysis",
                description="Analyze system performance metrics and create visualizations",
                category="performance_analysis",
                workflow_definition={
                    "steps": [
                        {
                            "name": "Performance Profiling",
                            "description": "Profile system performance and identify bottlenecks",
                            "required_skills": ["performance_analysis", "profiling"],
                            "agent_type": "performance_analyst",
                            "priority": "high"
                        },
                        {
                            "name": "Data Visualization",
                            "description": "Create performance metric visualizations",
                            "required_skills": ["data_processing", "visualization"],
                            "agent_type": "data_analyst",
                            "priority": "medium"
                        }
                    ],
                    "execution_strategy": "parallel"
                }
            ),
            
            # ML Pipeline - Should select ML Engineer + DevOps Engineer
            TestWorkflow(
                name="ML Model Deployment",
                description="Train ML model and deploy to production",
                category="ml_deployment",
                workflow_definition={
                    "steps": [
                        {
                            "name": "Model Training",
                            "description": "Train and optimize ML model",
                            "required_skills": ["machine_learning", "model_training"],
                            "agent_type": "ml_engineer",
                            "priority": "high"
                        },
                        {
                            "name": "Container Deployment",
                            "description": "Deploy model in Docker container",
                            "required_skills": ["docker", "kubernetes"],
                            "agent_type": "devops_expert",
                            "priority": "high"
                        }
                    ],
                    "execution_strategy": "sequential"
                }
            ),
            
            # Documentation Workflow - Should select Documentation Writer
            TestWorkflow(
                name="API Documentation",
                description="Generate comprehensive API documentation",
                category="documentation",
                workflow_definition={
                    "steps": [
                        {
                            "name": "Generate API Docs",
                            "description": "Create API documentation in markdown",
                            "required_skills": ["api_documentation", "markdown"],
                            "agent_type": "documentation_expert",
                            "priority": "medium"
                        }
                    ],
                    "execution_strategy": "sequential"
                }
            ),
            
            # Complex Multi-Agent Workflow
            TestWorkflow(
                name="Full Stack Application Review",
                description="Complete review of full-stack application",
                category="comprehensive_review",
                workflow_definition={
                    "steps": [
                        {
                            "name": "Backend Code Review",
                            "description": "Review backend Python code",
                            "required_skills": ["python", "code_review", "api_design"],
                            "agent_type": "code_architect",
                            "priority": "high"
                        },
                        {
                            "name": "Security Audit",
                            "description": "Comprehensive security audit",
                            "required_skills": ["security_audit", "penetration_testing"],
                            "agent_type": "security_expert",
                            "priority": "high"
                        },
                        {
                            "name": "Performance Testing",
                            "description": "Load testing and performance analysis",
                            "required_skills": ["performance_analysis", "benchmarking"],
                            "agent_type": "performance_analyst",
                            "priority": "medium"
                        },
                        {
                            "name": "Documentation Review",
                            "description": "Review and update documentation",
                            "required_skills": ["technical_writing", "api_documentation"],
                            "agent_type": "documentation_expert",
                            "priority": "low"
                        }
                    ],
                    "execution_strategy": "mixed"
                }
            )
        ]
        
        workflow_ids = []
        
        for workflow in test_workflows:
            logger.info(f"Creating workflow: {workflow.name}")
            
            try:
                response = self.api_request("POST", "workflows", workflow.to_api_payload())
                workflow_id = response.get("id")
                workflow_ids.append(workflow_id)
                self.created_workflows.append(workflow_id)
                logger.info(f"✅ Created workflow: {workflow.name} (ID: {workflow_id})")
                
            except Exception as e:
                logger.error(f"❌ Failed to create workflow {workflow.name}: {e}")
        
        return workflow_ids
    
    async def execute_workflow_test(self, workflow_id: int) -> Dict:
        """Execute workflow and monitor agent selection"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Executing workflow ID: {workflow_id}")
        logger.info(f"{'='*60}")
        
        # Start workflow execution
        execution_data = {
            "workflow_id": workflow_id,
            "input_data": {
                "test_mode": True,
                "monitor_agent_selection": True
            }
        }
        
        try:
            response = self.api_request("POST", f"workflows/{workflow_id}/execute", execution_data)
            execution_id = response.get("execution_id")
            logger.info(f"Started execution ID: {execution_id}")
            
            # Poll for completion
            max_attempts = 60  # 5 minutes max
            for attempt in range(max_attempts):
                await asyncio.sleep(5)  # Check every 5 seconds
                
                status_response = self.api_request("GET", f"workflows/executions/{execution_id}")
                status = status_response.get("status")
                progress = status_response.get("progress", 0)
                
                logger.info(f"Execution status: {status} (Progress: {progress}%)")
                
                if status in ["completed", "failed"]:
                    # Get detailed results
                    if status == "completed":
                        return self.analyze_execution_results(execution_id, workflow_id, status_response)
                    else:
                        logger.error(f"❌ Workflow execution failed: {status_response.get('error')}")
                        return {"success": False, "error": status_response.get("error")}
            
            logger.warning("⚠️ Workflow execution timed out")
            return {"success": False, "error": "Execution timed out"}
            
        except Exception as e:
            logger.error(f"❌ Error executing workflow: {e}")
            return {"success": False, "error": str(e)}
    
    def analyze_execution_results(self, execution_id: int, workflow_id: int, execution_data: Dict) -> Dict:
        """Analyze execution results focusing on agent selection"""
        
        logger.info(f"\n📊 ANALYZING EXECUTION RESULTS")
        logger.info(f"{'='*60}")
        
        results = {
            "execution_id": execution_id,
            "workflow_id": workflow_id,
            "success": True,
            "agent_selection_analysis": {},
            "performance_metrics": {},
            "cost_analysis": {}
        }
        
        # Extract agent selection data
        input_data = execution_data.get("input_data", {})
        agent_selection = input_data.get("agent_selection", {})
        
        if agent_selection.get("is_real"):
            summary = agent_selection.get("summary", {})
            assignments = agent_selection.get("assignments", {})
            
            logger.info(f"\n🤖 AGENT SELECTION SUMMARY:")
            logger.info(f"  Total subtasks: {summary.get('total_subtasks', 0)}")
            logger.info(f"  Subtasks with agents: {summary.get('subtasks_with_agents', 0)}")
            logger.info(f"  Coverage: {summary.get('coverage', 0):.1%}")
            logger.info(f"  Avg match score: {summary.get('avg_match_score', 0):.1%}")
            
            results["agent_selection_analysis"] = {
                "coverage": summary.get("coverage", 0),
                "avg_match_score": summary.get("avg_match_score", 0),
                "assignments": {}
            }
            
            # Analyze each subtask assignment
            for subtask_id, matches in assignments.items():
                if matches:
                    best_match = matches[0]  # First is best
                    logger.info(f"\n  📌 {subtask_id}:")
                    logger.info(f"    Selected: {best_match['agent_name']} (ID: {best_match['agent_id']})")
                    logger.info(f"    Match score: {best_match['match_score']:.1%}")
                    logger.info(f"    Skill coverage: {best_match['skill_coverage']:.1%}")
                    logger.info(f"    Matched skills: {best_match['matched_skills']}")
                    logger.info(f"    Missing skills: {best_match['missing_skills']}")
                    
                    results["agent_selection_analysis"]["assignments"][subtask_id] = {
                        "agent_name": best_match["agent_name"],
                        "match_score": best_match["match_score"],
                        "skill_coverage": best_match["skill_coverage"],
                        "matched_skills": best_match["matched_skills"],
                        "missing_skills": best_match["missing_skills"]
                    }
        
        # Extract execution metrics
        agent_execution = input_data.get("agent_execution", {})
        if agent_execution.get("is_real"):
            exec_summary = agent_execution.get("summary", {})
            
            logger.info(f"\n⚡ EXECUTION METRICS:")
            logger.info(f"  Success rate: {exec_summary.get('success_rate', 0):.1%}")
            logger.info(f"  Total duration: {exec_summary.get('total_duration_seconds', 0):.1f}s")
            logger.info(f"  Total tokens: {exec_summary.get('total_tokens_used', 0):,}")
            logger.info(f"  Avg tokens/task: {exec_summary.get('avg_tokens_per_subtask', 0):.0f}")
            
            results["performance_metrics"] = {
                "success_rate": exec_summary.get("success_rate", 0),
                "duration_seconds": exec_summary.get("total_duration_seconds", 0),
                "total_tokens": exec_summary.get("total_tokens_used", 0)
            }
        
        # Extract quality scores
        result_aggregation = input_data.get("result_aggregation", {})
        if result_aggregation.get("is_real"):
            quality_scores = result_aggregation.get("quality_scores", {})
            
            logger.info(f"\n✨ QUALITY SCORES:")
            logger.info(f"  Overall: {quality_scores.get('overall', 0):.1%}")
            logger.info(f"  Completeness: {quality_scores.get('completeness', 0):.1%}")
            logger.info(f"  Accuracy: {quality_scores.get('accuracy', 0):.1%}")
            logger.info(f"  Consistency: {quality_scores.get('consistency', 0):.1%}")
            
            results["quality_scores"] = quality_scores
        
        # Cost analysis
        models_used = execution_data.get("models_used", [])
        if models_used:
            total_cost = sum(model.get("cost", 0) for model in models_used)
            
            logger.info(f"\n💰 COST ANALYSIS:")
            logger.info(f"  Total cost: ${total_cost:.6f}")
            logger.info(f"  Models used: {len(models_used)}")
            
            results["cost_analysis"] = {
                "total_cost": total_cost,
                "models_used": len(models_used)
            }
        
        return results
    
    async def run_all_tests(self):
        """Run complete test suite"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"STARTING WORKFLOW AGENT SELECTION TEST SUITE")
        logger.info(f"{'='*60}\n")
        
        # Step 1: Create test agents
        logger.info("📦 STEP 1: Creating test agents...")
        agent_ids = await self.create_test_agents()
        logger.info(f"Created {len(agent_ids)} test agents\n")
        
        # Step 2: Create test workflows
        logger.info("📋 STEP 2: Creating test workflows...")
        workflow_ids = await self.create_test_workflows()
        logger.info(f"Created {len(workflow_ids)} test workflows\n")
        
        # Step 3: Execute workflows and analyze results
        logger.info("🚀 STEP 3: Executing workflows and analyzing agent selection...")
        
        for workflow_id in workflow_ids:
            result = await self.execute_workflow_test(workflow_id)
            self.test_results.append(result)
            
            # Small delay between workflows
            await asyncio.sleep(2)
        
        # Step 4: Generate summary report
        self.generate_summary_report()
    
    def generate_summary_report(self):
        """Generate comprehensive test summary"""
        
        logger.info(f"\n{'='*60}")
        logger.info(f"TEST SUITE SUMMARY REPORT")
        logger.info(f"{'='*60}\n")
        
        successful_tests = [r for r in self.test_results if r.get("success")]
        failed_tests = [r for r in self.test_results if not r.get("success")]
        
        logger.info(f"📊 OVERALL RESULTS:")
        logger.info(f"  Total tests: {len(self.test_results)}")
        logger.info(f"  Successful: {len(successful_tests)}")
        logger.info(f"  Failed: {len(failed_tests)}")
        logger.info(f"  Success rate: {len(successful_tests)/len(self.test_results)*100:.1f}%")
        
        if successful_tests:
            # Calculate averages
            avg_match_score = sum(
                r["agent_selection_analysis"]["avg_match_score"] 
                for r in successful_tests
            ) / len(successful_tests)
            
            avg_coverage = sum(
                r["agent_selection_analysis"]["coverage"] 
                for r in successful_tests
            ) / len(successful_tests)
            
            total_cost = sum(
                r.get("cost_analysis", {}).get("total_cost", 0) 
                for r in successful_tests
            )
            
            logger.info(f"\n🎯 AGENT SELECTION METRICS:")
            logger.info(f"  Average match score: {avg_match_score:.1%}")
            logger.info(f"  Average coverage: {avg_coverage:.1%}")
            
            logger.info(f"\n💰 COST METRICS:")
            logger.info(f"  Total cost: ${total_cost:.6f}")
            logger.info(f"  Avg cost/workflow: ${total_cost/len(successful_tests):.6f}")
        
        # Save detailed report to file
        report_file = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump({
                "summary": {
                    "total_tests": len(self.test_results),
                    "successful": len(successful_tests),
                    "failed": len(failed_tests),
                    "created_agents": self.created_agents,
                    "created_workflows": self.created_workflows
                },
                "detailed_results": self.test_results
            }, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed report saved to: {report_file}")
    
    async def cleanup(self):
        """Clean up test data (optional)"""
        
        logger.info(f"\n🧹 Cleaning up test data...")
        
        # Note: You may want to keep test data for analysis
        # Uncomment to delete:
        
        # for workflow_id in self.created_workflows:
        #     try:
        #         self.api_request("DELETE", f"workflows/{workflow_id}")
        #         logger.info(f"Deleted workflow {workflow_id}")
        #     except:
        #         pass
        
        # for agent_id in self.created_agents:
        #     try:
        #         self.api_request("DELETE", f"agents/{agent_id}")
        #         logger.info(f"Deleted agent {agent_id}")
        #     except:
        #         pass


async def main():
    """Main entry point"""
    tester = WorkflowAgentSelectionTester()
    
    try:
        await tester.run_all_tests()
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Optional cleanup
        # await tester.cleanup()
        pass


if __name__ == "__main__":
    asyncio.run(main())
