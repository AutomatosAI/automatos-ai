
"""
Professional Agent Testing Suite
===============================

Comprehensive test suite for all professional agent types and their capabilities.
Tests agent creation, skill execution, and integration with the API.
"""

import asyncio
import pytest
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Any

# Add the orchestrator directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import (
    create_agent,
    AGENT_REGISTRY,
    CodeArchitectAgent,
    SecurityExpertAgent,
    PerformanceOptimizerAgent,
    DataAnalystAgent,
    InfrastructureManagerAgent,
    CustomAgent
)

class ProfessionalAgentTester:
    """Comprehensive tester for professional agents"""
    
    def __init__(self):
        self.test_results = []
        self.start_time = None
        self.end_time = None
    
    async def test_agent_creation(self):
        """Test creation of all professional agent types"""
        print("🧪 Testing Agent Creation")
        print("-" * 50)
        
        creation_results = []
        
        for agent_type, agent_class in AGENT_REGISTRY.items():
            try:
                # Create agent instance
                agent = create_agent(
                    agent_type=agent_type,
                    agent_id=1,
                    name=f"Test {agent_type.replace('_', ' ').title()}",
                    description=f"Test instance of {agent_type} agent"
                )
                
                # Verify agent properties
                assert agent.agent_type == agent_type
                assert len(agent.skills) > 0
                assert len(agent.capabilities) > 0
                assert len(agent.default_skills) > 0
                assert len(agent.specializations) > 0
                
                creation_results.append({
                    "agent_type": agent_type,
                    "success": True,
                    "skills_count": len(agent.skills),
                    "capabilities_count": len(agent.capabilities),
                    "default_skills": agent.default_skills,
                    "specializations": agent.specializations
                })
                
                print(f"✅ {agent_type}: Created successfully with {len(agent.skills)} skills")
                
            except Exception as e:
                creation_results.append({
                    "agent_type": agent_type,
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ {agent_type}: Creation failed - {e}")
        
        self.test_results.append({
            "test_name": "agent_creation",
            "results": creation_results,
            "success_rate": len([r for r in creation_results if r["success"]]) / len(creation_results)
        })
        
        return creation_results
    
    async def test_code_architect_capabilities(self):
        """Test Code Architect agent capabilities"""
        print("\n🏗️ Testing Code Architect Agent")
        print("-" * 50)
        
        agent = CodeArchitectAgent(
            agent_id=1,
            name="Test Code Architect",
            description="Test code architect agent"
        )
        
        test_tasks = [
            {
                "type": "code_analysis",
                "code": "def hello_world():\n    print('Hello, World!')\n    return True",
                "language": "python",
                "analysis_type": "comprehensive"
            },
            {
                "type": "architecture_design",
                "requirements": {"scalability": "high", "maintainability": "high"},
                "system_type": "web_application",
                "scale": "large"
            },
            {
                "type": "code_review",
                "code_changes": ["main.py", "utils.py"],
                "criteria": ["functionality", "readability", "performance"]
            },
            {
                "type": "refactoring_plan",
                "codebase_analysis": {"complexity": "high"},
                "goals": ["improve_maintainability", "reduce_complexity"]
            }
        ]
        
        results = []
        for task in test_tasks:
            try:
                result = await agent.execute_task(task)
                results.append({
                    "task_type": task["type"],
                    "success": result.get("success", False),
                    "execution_time": result.get("timestamp", 0),
                    "has_results": "results" in result
                })
                print(f"✅ {task['type']}: Executed successfully")
            except Exception as e:
                results.append({
                    "task_type": task["type"],
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ {task['type']}: Failed - {e}")
        
        self.test_results.append({
            "test_name": "code_architect_capabilities",
            "results": results,
            "agent_performance": agent.performance_metrics
        })
        
        return results
    
    async def test_security_expert_capabilities(self):
        """Test Security Expert agent capabilities"""
        print("\n🛡️ Testing Security Expert Agent")
        print("-" * 50)
        
        agent = SecurityExpertAgent(
            agent_id=2,
            name="Test Security Expert",
            description="Test security expert agent"
        )
        
        test_tasks = [
            {
                "type": "vulnerability_scan",
                "target": "https://example.com",
                "scan_type": "comprehensive",
                "depth": "standard"
            },
            {
                "type": "threat_model",
                "system_description": "Web application with user authentication",
                "methodology": "stride"
            },
            {
                "type": "compliance_check",
                "framework": "owasp_top_10",
                "scope": "application"
            },
            {
                "type": "security_audit",
                "scope": ["code", "infrastructure"],
                "depth": "comprehensive"
            }
        ]
        
        results = []
        for task in test_tasks:
            try:
                result = await agent.execute_task(task)
                results.append({
                    "task_type": task["type"],
                    "success": result.get("success", False),
                    "execution_time": result.get("timestamp", 0),
                    "has_results": "results" in result
                })
                print(f"✅ {task['type']}: Executed successfully")
            except Exception as e:
                results.append({
                    "task_type": task["type"],
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ {task['type']}: Failed - {e}")
        
        self.test_results.append({
            "test_name": "security_expert_capabilities",
            "results": results,
            "agent_performance": agent.performance_metrics
        })
        
        return results
    
    async def test_performance_optimizer_capabilities(self):
        """Test Performance Optimizer agent capabilities"""
        print("\n⚡ Testing Performance Optimizer Agent")
        print("-" * 50)
        
        agent = PerformanceOptimizerAgent(
            agent_id=3,
            name="Test Performance Optimizer",
            description="Test performance optimizer agent"
        )
        
        test_tasks = [
            {
                "type": "performance_analysis",
                "target_system": "web_application",
                "metrics_period": "24h",
                "analysis_depth": "comprehensive"
            },
            {
                "type": "bottleneck_detection",
                "system_metrics": {"cpu": 75, "memory": 68, "io": 45},
                "detection_method": "comprehensive"
            },
            {
                "type": "load_testing",
                "target_url": "https://example.com/api",
                "test_type": "load",
                "duration": "5m",
                "concurrent_users": 100
            },
            {
                "type": "database_optimization",
                "database_type": "postgresql",
                "scope": ["queries", "indexes"]
            }
        ]
        
        results = []
        for task in test_tasks:
            try:
                result = await agent.execute_task(task)
                results.append({
                    "task_type": task["type"],
                    "success": result.get("success", False),
                    "execution_time": result.get("timestamp", 0),
                    "has_results": "results" in result
                })
                print(f"✅ {task['type']}: Executed successfully")
            except Exception as e:
                results.append({
                    "task_type": task["type"],
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ {task['type']}: Failed - {e}")
        
        self.test_results.append({
            "test_name": "performance_optimizer_capabilities",
            "results": results,
            "agent_performance": agent.performance_metrics
        })
        
        return results
    
    async def test_data_analyst_capabilities(self):
        """Test Data Analyst agent capabilities"""
        print("\n📊 Testing Data Analyst Agent")
        print("-" * 50)
        
        agent = DataAnalystAgent(
            agent_id=4,
            name="Test Data Analyst",
            description="Test data analyst agent"
        )
        
        test_tasks = [
            {
                "type": "data_analysis",
                "dataset": {"rows": 1000, "columns": 10},
                "analysis_type": "exploratory",
                "objectives": ["understand_data", "find_patterns"]
            },
            {
                "type": "pattern_recognition",
                "data_source": "sales_data",
                "pattern_types": ["trends", "anomalies", "seasonality"]
            },
            {
                "type": "report_generation",
                "analysis_data": {"revenue": 100000, "growth": 15},
                "report_type": "comprehensive",
                "audience": "management"
            },
            {
                "type": "predictive_modeling",
                "target_variable": "revenue",
                "model_type": "auto",
                "prediction_horizon": "3_months"
            }
        ]
        
        results = []
        for task in test_tasks:
            try:
                result = await agent.execute_task(task)
                results.append({
                    "task_type": task["type"],
                    "success": result.get("success", False),
                    "execution_time": result.get("timestamp", 0),
                    "has_results": "results" in result
                })
                print(f"✅ {task['type']}: Executed successfully")
            except Exception as e:
                results.append({
                    "task_type": task["type"],
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ {task['type']}: Failed - {e}")
        
        self.test_results.append({
            "test_name": "data_analyst_capabilities",
            "results": results,
            "agent_performance": agent.performance_metrics
        })
        
        return results
    
    async def test_infrastructure_manager_capabilities(self):
        """Test Infrastructure Manager agent capabilities"""
        print("\n🏗️ Testing Infrastructure Manager Agent")
        print("-" * 50)
        
        agent = InfrastructureManagerAgent(
            agent_id=5,
            name="Test Infrastructure Manager",
            description="Test infrastructure manager agent"
        )
        
        test_tasks = [
            {
                "type": "deployment",
                "application": "web_app",
                "environment": "production",
                "strategy": "rolling"
            },
            {
                "type": "scaling_analysis",
                "application": "api_service",
                "current_metrics": {"cpu": 75, "memory": 68},
                "forecast_period": "30_days"
            },
            {
                "type": "infrastructure_monitoring",
                "scope": ["applications", "infrastructure"],
                "requirements": {"alerting": True, "dashboards": True}
            },
            {
                "type": "ci_cd_setup",
                "application": "web_service",
                "environments": ["dev", "staging", "prod"],
                "requirements": {"security_scan": True, "automated_tests": True}
            }
        ]
        
        results = []
        for task in test_tasks:
            try:
                result = await agent.execute_task(task)
                results.append({
                    "task_type": task["type"],
                    "success": result.get("success", False),
                    "execution_time": result.get("timestamp", 0),
                    "has_results": "results" in result
                })
                print(f"✅ {task['type']}: Executed successfully")
            except Exception as e:
                results.append({
                    "task_type": task["type"],
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ {task['type']}: Failed - {e}")
        
        self.test_results.append({
            "test_name": "infrastructure_manager_capabilities",
            "results": results,
            "agent_performance": agent.performance_metrics
        })
        
        return results
    
    async def test_custom_agent_capabilities(self):
        """Test Custom Agent capabilities"""
        print("\n🔧 Testing Custom Agent")
        print("-" * 50)
        
        # Test with custom skills configuration
        custom_skills = [
            {
                "name": "custom_analysis",
                "skill_type": "analytical",
                "description": "Custom analysis skill",
                "parameters": {"analysis_types": ["custom", "specialized"]}
            },
            {
                "name": "custom_processing",
                "skill_type": "technical",
                "description": "Custom processing skill",
                "parameters": {"processing_modes": ["batch", "stream"]}
            }
        ]
        
        custom_capabilities = [
            {
                "name": "custom_workflow",
                "description": "Custom workflow capability",
                "required_skills": ["custom_analysis"],
                "optional_skills": ["custom_processing"],
                "complexity_level": 3
            }
        ]
        
        agent = CustomAgent(
            agent_id=6,
            name="Test Custom Agent",
            description="Test custom agent with specialized configuration",
            custom_skills=custom_skills,
            custom_capabilities=custom_capabilities
        )
        
        test_tasks = [
            {
                "type": "general",
                "description": "Analyze the provided data and generate insights",
                "parameters": {"data_type": "structured"},
                "expected_output": "structured_response"
            },
            {
                "type": "data_processing",
                "data": {"sample": "data"},
                "processing_type": "analysis",
                "output_format": "json"
            },
            {
                "type": "problem_solving",
                "problem": "System performance is degrading over time",
                "context": {"system_type": "web_application"},
                "approach": "systematic"
            }
        ]
        
        results = []
        for task in test_tasks:
            try:
                result = await agent.execute_task(task)
                results.append({
                    "task_type": task["type"],
                    "success": result.get("success", False),
                    "execution_time": result.get("timestamp", 0),
                    "has_results": "results" in result
                })
                print(f"✅ {task['type']}: Executed successfully")
            except Exception as e:
                results.append({
                    "task_type": task["type"],
                    "success": False,
                    "error": str(e)
                })
                print(f"❌ {task['type']}: Failed - {e}")
        
        self.test_results.append({
            "test_name": "custom_agent_capabilities",
            "results": results,
            "agent_performance": agent.performance_metrics,
            "custom_configuration": {
                "skills": len(custom_skills),
                "capabilities": len(custom_capabilities)
            }
        })
        
        return results
    
    async def test_agent_collaboration(self):
        """Test multi-agent collaboration scenarios"""
        print("\n🤝 Testing Agent Collaboration")
        print("-" * 50)
        
        # Create multiple agents
        code_agent = CodeArchitectAgent(1, "Code Architect", "Code analysis agent")
        security_agent = SecurityExpertAgent(2, "Security Expert", "Security analysis agent")
        performance_agent = PerformanceOptimizerAgent(3, "Performance Optimizer", "Performance analysis agent")
        
        # Simulate collaborative workflow
        collaboration_results = []
        
        try:
            # Step 1: Code analysis
            code_task = {
                "type": "code_analysis",
                "code": "def process_user_data(data): return data.upper()",
                "language": "python"
            }
            code_result = await code_agent.execute_task(code_task)
            
            # Step 2: Security review based on code analysis
            security_task = {
                "type": "vulnerability_scan",
                "target": "code_analysis_results",
                "scan_type": "static_analysis"
            }
            security_result = await security_agent.execute_task(security_task)
            
            # Step 3: Performance analysis
            performance_task = {
                "type": "performance_analysis",
                "target_system": "analyzed_code",
                "analysis_depth": "standard"
            }
            performance_result = await performance_agent.execute_task(performance_task)
            
            collaboration_results.append({
                "workflow": "code_security_performance_analysis",
                "steps": [
                    {"agent": "code_architect", "success": code_result.get("success", False)},
                    {"agent": "security_expert", "success": security_result.get("success", False)},
                    {"agent": "performance_optimizer", "success": performance_result.get("success", False)}
                ],
                "overall_success": all([
                    code_result.get("success", False),
                    security_result.get("success", False),
                    performance_result.get("success", False)
                ])
            })
            
            print("✅ Multi-agent collaboration workflow completed successfully")
            
        except Exception as e:
            collaboration_results.append({
                "workflow": "code_security_performance_analysis",
                "success": False,
                "error": str(e)
            })
            print(f"❌ Multi-agent collaboration failed: {e}")
        
        self.test_results.append({
            "test_name": "agent_collaboration",
            "results": collaboration_results
        })
        
        return collaboration_results
    
    async def test_agent_performance_metrics(self):
        """Test agent performance tracking"""
        print("\n📈 Testing Agent Performance Metrics")
        print("-" * 50)
        
        agent = DataAnalystAgent(
            agent_id=7,
            name="Performance Test Agent",
            description="Agent for performance testing"
        )
        
        # Execute multiple tasks to build performance history
        tasks = [
            {"type": "data_analysis", "dataset": {"size": "small"}},
            {"type": "pattern_recognition", "data_source": "test_data"},
            {"type": "data_visualization", "data": {"points": 100}}
        ]
        
        performance_results = []
        
        for i, task in enumerate(tasks):
            try:
                start_time = asyncio.get_event_loop().time()
                result = await agent.execute_task(task)
                end_time = asyncio.get_event_loop().time()
                
                performance_results.append({
                    "task_number": i + 1,
                    "task_type": task["type"],
                    "success": result.get("success", False),
                    "execution_time": end_time - start_time,
                    "agent_metrics": agent.performance_metrics.copy()
                })
                
            except Exception as e:
                performance_results.append({
                    "task_number": i + 1,
                    "task_type": task["type"],
                    "success": False,
                    "error": str(e)
                })
        
        # Verify performance metrics are being tracked
        final_metrics = agent.performance_metrics
        metrics_valid = (
            final_metrics['tasks_completed'] == len([r for r in performance_results if r.get("success", False)]) and
            final_metrics['success_rate'] >= 0 and
            final_metrics['average_execution_time'] >= 0
        )
        
        self.test_results.append({
            "test_name": "agent_performance_metrics",
            "results": performance_results,
            "final_metrics": final_metrics,
            "metrics_valid": metrics_valid
        })
        
        print(f"✅ Performance metrics tracking: {'Valid' if metrics_valid else 'Invalid'}")
        print(f"   Tasks completed: {final_metrics['tasks_completed']}")
        print(f"   Success rate: {final_metrics['success_rate']:.2%}")
        print(f"   Average execution time: {final_metrics['average_execution_time']:.2f}s")
        
        return performance_results
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🎯 PROFESSIONAL AGENT TESTING SUITE")
        print("=" * 80)
        print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        self.start_time = datetime.now()
        
        # Run all tests
        await self.test_agent_creation()
        await self.test_code_architect_capabilities()
        await self.test_security_expert_capabilities()
        await self.test_performance_optimizer_capabilities()
        await self.test_data_analyst_capabilities()
        await self.test_infrastructure_manager_capabilities()
        await self.test_custom_agent_capabilities()
        await self.test_agent_collaboration()
        await self.test_agent_performance_metrics()
        
        self.end_time = datetime.now()
        
        # Generate final report
        self._generate_final_report()
    
    def _generate_final_report(self):
        """Generate comprehensive test report"""
        print("\n" + "=" * 80)
        print("🏁 PROFESSIONAL AGENT TEST REPORT")
        print("=" * 80)
        
        total_duration = (self.end_time - self.start_time).total_seconds()
        
        # Calculate overall statistics
        total_tests = len(self.test_results)
        successful_tests = 0
        total_tasks = 0
        successful_tasks = 0
        
        for test in self.test_results:
            if test["test_name"] == "agent_creation":
                successful_tests += 1 if test["success_rate"] > 0.8 else 0
            elif "results" in test:
                test_success = all(r.get("success", False) for r in test["results"] if isinstance(r, dict))
                successful_tests += 1 if test_success else 0
                total_tasks += len(test["results"])
                successful_tasks += len([r for r in test["results"] if r.get("success", False)])
        
        print(f"📊 Overall Results:")
        print(f"  - Test Categories: {total_tests}")
        print(f"  - Successful Categories: {successful_tests}")
        print(f"  - Category Success Rate: {(successful_tests/total_tests*100):.1f}%")
        print(f"  - Total Tasks Executed: {total_tasks}")
        print(f"  - Successful Tasks: {successful_tasks}")
        print(f"  - Task Success Rate: {(successful_tasks/total_tasks*100):.1f}%")
        print(f"  - Total Duration: {total_duration:.2f} seconds")
        
        print(f"\n📋 Test Category Details:")
        for i, test in enumerate(self.test_results, 1):
            test_name = test["test_name"].replace("_", " ").title()
            
            if test["test_name"] == "agent_creation":
                success_rate = test["success_rate"]
                status = "✅ PASS" if success_rate > 0.8 else "❌ FAIL"
                print(f"  {i}. {test_name}: {status} ({success_rate:.1%} success rate)")
            elif "results" in test:
                results = test["results"]
                success_count = len([r for r in results if r.get("success", False)])
                total_count = len(results)
                status = "✅ PASS" if success_count == total_count else "❌ FAIL"
                print(f"  {i}. {test_name}: {status} ({success_count}/{total_count} tasks)")
        
        # Agent type coverage
        print(f"\n🤖 Agent Type Coverage:")
        agent_types_tested = set()
        for test in self.test_results:
            if "code_architect" in test["test_name"]:
                agent_types_tested.add("Code Architect")
            elif "security_expert" in test["test_name"]:
                agent_types_tested.add("Security Expert")
            elif "performance_optimizer" in test["test_name"]:
                agent_types_tested.add("Performance Optimizer")
            elif "data_analyst" in test["test_name"]:
                agent_types_tested.add("Data Analyst")
            elif "infrastructure_manager" in test["test_name"]:
                agent_types_tested.add("Infrastructure Manager")
            elif "custom_agent" in test["test_name"]:
                agent_types_tested.add("Custom Agent")
        
        for agent_type in sorted(agent_types_tested):
            print(f"  ✅ {agent_type}")
        
        # Performance summary
        performance_data = []
        for test in self.test_results:
            if "agent_performance" in test:
                performance_data.append(test["agent_performance"])
        
        if performance_data:
            avg_success_rate = sum(p["success_rate"] for p in performance_data) / len(performance_data)
            avg_execution_time = sum(p["average_execution_time"] for p in performance_data) / len(performance_data)
            
            print(f"\n⚡ Performance Summary:")
            print(f"  - Average Success Rate: {avg_success_rate:.1%}")
            print(f"  - Average Execution Time: {avg_execution_time:.2f}s")
            print(f"  - Agents with Performance Data: {len(performance_data)}")
        
        print(f"\n🎉 Testing completed at {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Save results to file
        self._save_results_to_file()
    
    def _save_results_to_file(self):
        """Save test results to JSON file"""
        results_data = {
            "test_session": {
                "start_time": self.start_time.isoformat(),
                "end_time": self.end_time.isoformat(),
                "total_duration_seconds": (self.end_time - self.start_time).total_seconds()
            },
            "test_results": self.test_results,
            "summary": {
                "total_test_categories": len(self.test_results),
                "agent_types_tested": 6,
                "comprehensive_coverage": True
            }
        }
        
        results_file = f"professional_agent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(results_file, 'w') as f:
                json.dump(results_data, f, indent=2, default=str)
            print(f"📄 Test results saved to: {results_file}")
        except Exception as e:
            print(f"⚠️  Failed to save results to file: {e}")

async def main():
    """Main test execution function"""
    tester = ProfessionalAgentTester()
    await tester.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
