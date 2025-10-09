#!/usr/bin/env python3
"""
Comprehensive Workflow Testing Suite
=====================================
Creates and executes diverse workflows to generate real learning data
and prove system intelligence improvements.
"""

import asyncio
import json
import time
import random
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class WorkflowTestSuite:
    """Comprehensive workflow testing for learning and benchmarking"""
    
    def __init__(self, base_url: str = "https://api.automatos.app"):
        self.base_url = base_url
        self.created_agents = []
        self.created_workflows = []
        self.execution_results = []
        
    async def setup_test_agents(self) -> List[Dict[str, Any]]:
        """Create diverse agents for testing"""
        logger.info("🤖 Creating test agents...")
        
        agent_specs = [
            {
                "name": "Python Developer",
                "type": "developer",
                "skills": ["python", "django", "flask", "fastapi", "testing", "debugging"],
                "proficiency_level": 0.9,
                "description": "Expert Python developer for backend development"
            },
            {
                "name": "Frontend Developer",
                "type": "developer",
                "skills": ["javascript", "react", "vue", "typescript", "css", "html"],
                "proficiency_level": 0.85,
                "description": "Frontend specialist for UI development"
            },
            {
                "name": "DevOps Engineer",
                "type": "infrastructure",
                "skills": ["docker", "kubernetes", "ci/cd", "aws", "monitoring", "deployment"],
                "proficiency_level": 0.88,
                "description": "Infrastructure and deployment expert"
            },
            {
                "name": "Security Analyst",
                "type": "security",
                "skills": ["vulnerability_assessment", "penetration_testing", "security_audit", "compliance"],
                "proficiency_level": 0.92,
                "description": "Security expert for vulnerability analysis"
            },
            {
                "name": "Data Scientist",
                "type": "analyst",
                "skills": ["machine_learning", "data_analysis", "statistics", "visualization", "python", "sql"],
                "proficiency_level": 0.87,
                "description": "Data analysis and ML expert"
            },
            {
                "name": "Technical Writer",
                "type": "documentation",
                "skills": ["technical_writing", "api_documentation", "user_guides", "markdown", "diagrams"],
                "proficiency_level": 0.83,
                "description": "Documentation specialist"
            },
            {
                "name": "QA Engineer",
                "type": "testing",
                "skills": ["test_automation", "unit_testing", "integration_testing", "performance_testing", "bug_tracking"],
                "proficiency_level": 0.86,
                "description": "Quality assurance and testing expert"
            },
            {
                "name": "Project Manager",
                "type": "management",
                "skills": ["project_planning", "task_decomposition", "resource_allocation", "risk_management", "agile"],
                "proficiency_level": 0.84,
                "description": "Project coordination and management"
            },
            {
                "name": "Database Administrator",
                "type": "database",
                "skills": ["postgresql", "mysql", "mongodb", "optimization", "backup", "migration"],
                "proficiency_level": 0.89,
                "description": "Database management and optimization"
            },
            {
                "name": "Research Analyst",
                "type": "research",
                "skills": ["market_research", "competitive_analysis", "trend_analysis", "report_writing"],
                "proficiency_level": 0.81,
                "description": "Research and analysis specialist"
            }
        ]
        
        for spec in agent_specs:
            try:
                response = requests.post(
                    f"{self.base_url}/v1/agents",
                    json=spec
                )
                if response.status_code == 200:
                    agent = response.json()
                    self.created_agents.append(agent)
                    logger.info(f"✅ Created agent: {spec['name']} (ID: {agent.get('id')})")
                else:
                    logger.error(f"❌ Failed to create agent {spec['name']}: {response.status_code}")
            except Exception as e:
                logger.error(f"❌ Error creating agent {spec['name']}: {e}")
        
        return self.created_agents
    
    def get_test_workflows(self) -> List[Dict[str, Any]]:
        """Define diverse test workflows"""
        return [
            # 1. Hello World App Development
            {
                "name": "Build Hello World Web App",
                "description": "Complete development of a Hello World web application",
                "task_description": """
                    Create a complete Hello World web application with the following requirements:
                    1. Design the application architecture (frontend/backend separation)
                    2. Implement a Python Flask backend with REST API
                    3. Create a React frontend with modern UI
                    4. Write comprehensive unit tests
                    5. Setup Docker containerization
                    6. Create deployment configuration for AWS
                    7. Write API documentation
                    8. Perform security audit
                    9. Optimize performance
                    10. Create user documentation
                """,
                "type": "development",
                "priority": "high",
                "complexity": "complex",
                "expected_agents": ["Python Developer", "Frontend Developer", "DevOps Engineer", "QA Engineer", "Technical Writer"],
                "repeat_count": 5  # Run 5 times to show learning
            },
            
            # 2. Market Research Project
            {
                "name": "AI Market Research & Analysis",
                "description": "Comprehensive market research on AI industry",
                "task_description": """
                    Conduct thorough market research on the AI industry:
                    1. Analyze current AI market size and growth projections
                    2. Identify top 10 competitors and their strategies
                    3. Research emerging AI technologies and trends
                    4. Analyze customer needs and pain points
                    5. Evaluate regulatory landscape and compliance requirements
                    6. Assess market opportunities and threats
                    7. Create financial projections and ROI analysis
                    8. Develop go-to-market strategy recommendations
                    9. Create executive presentation with visualizations
                    10. Write detailed research report with citations
                """,
                "type": "research",
                "priority": "high",
                "complexity": "complex",
                "expected_agents": ["Research Analyst", "Data Scientist", "Technical Writer", "Project Manager"],
                "repeat_count": 4
            },
            
            # 3. System Performance Optimization
            {
                "name": "Database Performance Optimization",
                "description": "Optimize database performance for high-load system",
                "task_description": """
                    Optimize a high-traffic database system:
                    1. Analyze current database performance metrics
                    2. Identify slow queries and bottlenecks
                    3. Design and implement indexing strategy
                    4. Optimize query execution plans
                    5. Implement caching strategy
                    6. Setup read replicas and load balancing
                    7. Configure connection pooling
                    8. Implement monitoring and alerting
                    9. Create backup and recovery procedures
                    10. Document optimization changes and results
                """,
                "type": "optimization",
                "priority": "critical",
                "complexity": "complex",
                "expected_agents": ["Database Administrator", "DevOps Engineer", "Python Developer", "Technical Writer"],
                "repeat_count": 6  # More repetitions for optimization tasks
            },
            
            # 4. Security Audit & Remediation
            {
                "name": "Comprehensive Security Audit",
                "description": "Full security audit and vulnerability remediation",
                "task_description": """
                    Perform complete security audit:
                    1. Conduct vulnerability assessment of all systems
                    2. Perform penetration testing on APIs
                    3. Review authentication and authorization mechanisms
                    4. Analyze data encryption and storage security
                    5. Check for OWASP Top 10 vulnerabilities
                    6. Review network security configurations
                    7. Assess compliance with security standards
                    8. Create remediation plan with priorities
                    9. Implement critical security fixes
                    10. Generate security audit report
                """,
                "type": "security",
                "priority": "critical",
                "complexity": "complex",
                "expected_agents": ["Security Analyst", "DevOps Engineer", "Python Developer", "Technical Writer"],
                "repeat_count": 4
            },
            
            # 5. Data Pipeline Development
            {
                "name": "Build ETL Data Pipeline",
                "description": "Create end-to-end data processing pipeline",
                "task_description": """
                    Build a production-ready ETL pipeline:
                    1. Design data pipeline architecture
                    2. Implement data extraction from multiple sources
                    3. Create data transformation logic
                    4. Setup data validation and quality checks
                    5. Implement error handling and retry logic
                    6. Create data loading to warehouse
                    7. Setup scheduling and orchestration
                    8. Implement monitoring and logging
                    9. Create data lineage documentation
                    10. Write operational runbook
                """,
                "type": "data_engineering",
                "priority": "high",
                "complexity": "complex",
                "expected_agents": ["Data Scientist", "Python Developer", "Database Administrator", "DevOps Engineer"],
                "repeat_count": 5
            },
            
            # 6. API Development & Documentation
            {
                "name": "Design & Build REST API",
                "description": "Create comprehensive REST API with documentation",
                "task_description": """
                    Design and implement a REST API:
                    1. Design API architecture and endpoints
                    2. Implement authentication and authorization
                    3. Create CRUD operations for all resources
                    4. Implement input validation and error handling
                    5. Add rate limiting and throttling
                    6. Create comprehensive API tests
                    7. Generate OpenAPI/Swagger documentation
                    8. Implement versioning strategy
                    9. Setup API monitoring and analytics
                    10. Create developer guide and examples
                """,
                "type": "development",
                "priority": "high",
                "complexity": "moderate",
                "expected_agents": ["Python Developer", "QA Engineer", "Technical Writer", "Security Analyst"],
                "repeat_count": 7  # More repetitions for API tasks
            },
            
            # 7. Machine Learning Model Development
            {
                "name": "Build ML Prediction Model",
                "description": "Develop and deploy machine learning model",
                "task_description": """
                    Create a production ML model:
                    1. Analyze business requirements and data
                    2. Perform exploratory data analysis
                    3. Engineer relevant features
                    4. Select and train multiple models
                    5. Perform hyperparameter tuning
                    6. Validate model performance
                    7. Create model API endpoint
                    8. Implement model monitoring
                    9. Setup A/B testing framework
                    10. Document model methodology
                """,
                "type": "machine_learning",
                "priority": "high",
                "complexity": "complex",
                "expected_agents": ["Data Scientist", "Python Developer", "DevOps Engineer", "Technical Writer"],
                "repeat_count": 4
            },
            
            # 8. Microservices Architecture
            {
                "name": "Design Microservices System",
                "description": "Architecture and implement microservices",
                "task_description": """
                    Design microservices architecture:
                    1. Decompose monolith into microservices
                    2. Design service communication patterns
                    3. Implement service discovery
                    4. Setup API gateway
                    5. Implement circuit breakers
                    6. Create distributed tracing
                    7. Setup centralized logging
                    8. Implement saga pattern for transactions
                    9. Create deployment pipelines
                    10. Write architecture documentation
                """,
                "type": "architecture",
                "priority": "high",
                "complexity": "complex",
                "expected_agents": ["Python Developer", "DevOps Engineer", "Database Administrator", "Technical Writer"],
                "repeat_count": 3
            },
            
            # 9. Testing Automation Suite
            {
                "name": "Build Test Automation Framework",
                "description": "Create comprehensive test automation",
                "task_description": """
                    Build test automation framework:
                    1. Design test automation architecture
                    2. Create unit test suite
                    3. Implement integration tests
                    4. Build end-to-end test scenarios
                    5. Create performance test suite
                    6. Implement security testing
                    7. Setup continuous testing pipeline
                    8. Create test data management
                    9. Implement test reporting
                    10. Write test documentation
                """,
                "type": "testing",
                "priority": "high",
                "complexity": "moderate",
                "expected_agents": ["QA Engineer", "Python Developer", "DevOps Engineer", "Technical Writer"],
                "repeat_count": 5
            },
            
            # 10. Real-time Monitoring System
            {
                "name": "Implement Monitoring & Alerting",
                "description": "Setup comprehensive monitoring system",
                "task_description": """
                    Create monitoring and alerting system:
                    1. Define monitoring requirements and SLIs
                    2. Setup infrastructure monitoring
                    3. Implement application monitoring
                    4. Create custom metrics and dashboards
                    5. Configure alerting rules
                    6. Setup log aggregation
                    7. Implement distributed tracing
                    8. Create runbooks for alerts
                    9. Setup on-call rotation
                    10. Document monitoring strategy
                """,
                "type": "operations",
                "priority": "high",
                "complexity": "moderate",
                "expected_agents": ["DevOps Engineer", "Python Developer", "Database Administrator", "Technical Writer"],
                "repeat_count": 4
            }
        ]
    
    async def create_and_execute_workflow(self, workflow_spec: Dict[str, Any]) -> Dict[str, Any]:
        """Create and execute a single workflow"""
        try:
            # Create workflow
            create_data = {
                "name": workflow_spec["name"],
                "description": workflow_spec["description"],
                "goal": workflow_spec["task_description"],
                "workflow_definition": {
                    "type": workflow_spec["type"],
                    "priority": workflow_spec["priority"],
                    "complexity": workflow_spec["complexity"]
                },
                "context": {
                    "expected_agents": workflow_spec.get("expected_agents", []),
                    "test_run": True
                }
            }
            
            response = requests.post(
                f"{self.base_url}/v1/workflows",
                json=create_data
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to create workflow: {response.status_code}")
                return None
            
            workflow = response.json()
            workflow_id = workflow["id"]
            logger.info(f"📝 Created workflow: {workflow_spec['name']} (ID: {workflow_id})")
            
            # Execute workflow
            exec_response = requests.post(
                f"{self.base_url}/v1/workflows/{workflow_id}/execute"
            )
            
            if exec_response.status_code != 200:
                logger.error(f"Failed to execute workflow: {exec_response.status_code}")
                return None
            
            execution = exec_response.json()
            execution_id = execution.get("execution_id")
            logger.info(f"🚀 Started execution: {execution_id}")
            
            # Wait for completion (with timeout)
            start_time = time.time()
            timeout = 300  # 5 minutes timeout
            
            while time.time() - start_time < timeout:
                status_response = requests.get(
                    f"{self.base_url}/v1/workflows/{workflow_id}/status"
                )
                
                if status_response.status_code == 200:
                    status = status_response.json()
                    if status.get("status") in ["completed", "failed"]:
                        execution_time = time.time() - start_time
                        logger.info(f"✅ Execution completed in {execution_time:.2f}s: {status.get('status')}")
                        
                        # Get execution details for analysis
                        details_response = requests.get(
                            f"{self.base_url}/v1/workflows/executions/{execution_id}"
                        )
                        
                        if details_response.status_code == 200:
                            details = details_response.json()
                            return {
                                "workflow_id": workflow_id,
                                "execution_id": execution_id,
                                "name": workflow_spec["name"],
                                "status": status.get("status"),
                                "execution_time": execution_time,
                                "details": details
                            }
                        break
                
                await asyncio.sleep(5)  # Check every 5 seconds
            
            logger.warning(f"⏱️ Workflow execution timed out after {timeout}s")
            return None
            
        except Exception as e:
            logger.error(f"Error executing workflow: {e}")
            return None
    
    async def run_workflow_iterations(self, workflow_spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run a workflow multiple times to generate learning data"""
        repeat_count = workflow_spec.get("repeat_count", 3)
        results = []
        
        logger.info(f"\n🔄 Running '{workflow_spec['name']}' {repeat_count} times...")
        
        for i in range(repeat_count):
            logger.info(f"\n  Iteration {i+1}/{repeat_count}")
            result = await self.create_and_execute_workflow(workflow_spec)
            
            if result:
                results.append(result)
                
                # Add delay between iterations to simulate real usage
                if i < repeat_count - 1:
                    delay = random.uniform(5, 15)
                    logger.info(f"  ⏸️ Waiting {delay:.1f}s before next iteration...")
                    await asyncio.sleep(delay)
            else:
                logger.error(f"  ❌ Iteration {i+1} failed")
        
        return results
    
    async def analyze_results(self, all_results: List[List[Dict[str, Any]]]):
        """Analyze results to show learning and improvements"""
        logger.info("\n" + "="*60)
        logger.info("📊 ANALYSIS OF WORKFLOW EXECUTIONS")
        logger.info("="*60)
        
        for workflow_results in all_results:
            if not workflow_results:
                continue
            
            workflow_name = workflow_results[0]["name"]
            execution_times = [r["execution_time"] for r in workflow_results]
            
            logger.info(f"\n📈 {workflow_name}:")
            logger.info(f"  Executions: {len(workflow_results)}")
            logger.info(f"  Execution times: {[f'{t:.2f}s' for t in execution_times]}")
            
            if len(execution_times) > 1:
                improvement = (execution_times[0] - execution_times[-1]) / execution_times[0] * 100
                avg_time = sum(execution_times) / len(execution_times)
                
                logger.info(f"  Average time: {avg_time:.2f}s")
                logger.info(f"  First execution: {execution_times[0]:.2f}s")
                logger.info(f"  Last execution: {execution_times[-1]:.2f}s")
                logger.info(f"  Improvement: {improvement:.1f}% {'faster ✅' if improvement > 0 else 'slower ❌'}")
                
                # Check for learning patterns
                if improvement > 10:
                    logger.info(f"  🧠 LEARNING DETECTED: Significant improvement over iterations!")
                elif improvement > 0:
                    logger.info(f"  📚 Learning in progress: Moderate improvement observed")
    
    async def get_benchmarking_report(self):
        """Get final benchmarking report from API"""
        logger.info("\n" + "="*60)
        logger.info("🎯 FETCHING BENCHMARKING REPORT")
        logger.info("="*60)
        
        try:
            # Get performance summary
            perf_response = requests.get(f"{self.base_url}/v1/benchmarking/performance-summary")
            if perf_response.status_code == 200:
                perf_data = perf_response.json()
                
                logger.info("\n📊 Performance Summary:")
                logger.info(f"  Overall Improvement: {perf_data.get('overall_improvement', 0):.1f}%")
                logger.info(f"  Status: {perf_data.get('status', 'unknown')}")
                
                improvements = perf_data.get('improvements', {})
                if improvements:
                    logger.info("\n  Detailed Improvements:")
                    for key, value in improvements.items():
                        if isinstance(value, dict) and 'improvement_percentage' in value:
                            logger.info(f"    {key}: {value['improvement_percentage']:.1f}%")
                
                insights = perf_data.get('key_insights', [])
                if insights:
                    logger.info("\n  Key Insights:")
                    for insight in insights:
                        logger.info(f"    • {insight}")
            
            # Get learning effectiveness
            learning_response = requests.get(f"{self.base_url}/v1/benchmarking/learning-effectiveness")
            if learning_response.status_code == 200:
                learning_data = learning_response.json()
                
                logger.info("\n🧠 Learning Effectiveness:")
                summary = learning_data.get('summary', {})
                logger.info(f"  Patterns Learned: {summary.get('total_patterns_learned', 0)}")
                logger.info(f"  Memory Hit Rate: {summary.get('avg_memory_hit_rate', 0):.1f}%")
                logger.info(f"  Learning Velocity: {summary.get('learning_velocity', 0)*100:.1f}% per execution")
                logger.info(f"  Effectiveness Score: {learning_data.get('effectiveness_score', 0):.1f}%")
                
        except Exception as e:
            logger.error(f"Error fetching benchmarking report: {e}")
    
    async def run_comprehensive_test(self):
        """Run the complete test suite"""
        logger.info("="*60)
        logger.info("🚀 STARTING COMPREHENSIVE WORKFLOW TEST SUITE")
        logger.info("="*60)
        
        # Setup agents
        await self.setup_test_agents()
        
        if not self.created_agents:
            logger.error("❌ No agents created, cannot proceed with testing")
            return
        
        logger.info(f"\n✅ Created {len(self.created_agents)} test agents")
        
        # Get test workflows
        test_workflows = self.get_test_workflows()
        logger.info(f"\n📋 Prepared {len(test_workflows)} test workflows")
        
        # Run each workflow multiple times
        all_results = []
        for i, workflow_spec in enumerate(test_workflows, 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Workflow {i}/{len(test_workflows)}: {workflow_spec['name']}")
            logger.info(f"{'='*60}")
            
            results = await self.run_workflow_iterations(workflow_spec)
            all_results.append(results)
            
            # Small delay between different workflows
            if i < len(test_workflows):
                await asyncio.sleep(10)
        
        # Analyze results
        await self.analyze_results(all_results)
        
        # Get benchmarking report
        await self.get_benchmarking_report()
        
        # Summary
        total_executions = sum(len(results) for results in all_results)
        successful_executions = sum(
            sum(1 for r in results if r and r.get("status") == "completed")
            for results in all_results
        )
        
        logger.info("\n" + "="*60)
        logger.info("✅ TEST SUITE COMPLETE")
        logger.info("="*60)
        logger.info(f"Total Workflows Tested: {len(test_workflows)}")
        logger.info(f"Total Executions: {total_executions}")
        logger.info(f"Successful Executions: {successful_executions}")
        logger.info(f"Success Rate: {(successful_executions/total_executions*100) if total_executions > 0 else 0:.1f}%")
        logger.info("="*60)

async def main():
    """Main entry point"""
    test_suite = WorkflowTestSuite()
    await test_suite.run_comprehensive_test()

if __name__ == "__main__":
    asyncio.run(main())
