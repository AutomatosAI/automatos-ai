#!/usr/bin/env python3
"""
Comprehensive Benchmark Suite for Automatos AI Platform
========================================================
Tests all major system capabilities with real-world scenarios.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from datetime import datetime
import requests
import statistics
from dataclasses import dataclass, asdict
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('benchmark_results.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class BenchmarkResult:
    """Represents a single benchmark test result"""
    test_name: str
    category: str
    success: bool
    execution_time: float
    metrics: Dict[str, Any]
    errors: List[str]
    timestamp: datetime

@dataclass
class BenchmarkSuite:
    """Overall benchmark suite results"""
    total_tests: int
    passed: int
    failed: int
    total_time: float
    categories: Dict[str, Dict[str, Any]]
    detailed_results: List[BenchmarkResult]
    summary: Dict[str, Any]

class AutomatosBenchmark:
    """Main benchmark runner for Automatos AI Platform"""
    
    def __init__(self, base_url: str = "https://api.automatos.app"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []
        self.start_time = None
        self.auth_token = None
        
    async def setup(self):
        """Setup benchmark environment"""
        logger.info("🚀 Setting up benchmark environment...")
        
        # Check API health
        try:
            response = requests.get(f"{self.base_url}/health")
            if response.status_code != 200:
                raise Exception(f"API health check failed: {response.status_code}")
            logger.info("✅ API is healthy")
        except Exception as e:
            logger.error(f"❌ API health check failed: {e}")
            raise
            
        # Get list of existing agents for testing
        try:
            response = requests.get(f"{self.base_url}/v1/agents")
            self.available_agents = response.json() if response.status_code == 200 else []
            logger.info(f"📊 Found {len(self.available_agents)} available agents")
        except Exception as e:
            logger.warning(f"⚠️ Could not fetch agents: {e}")
            self.available_agents = []
    
    async def benchmark_agent_selection(self) -> BenchmarkResult:
        """Benchmark intelligent agent selection"""
        logger.info("\n📋 Testing: Intelligent Agent Selection")
        start_time = time.time()
        errors = []
        metrics = {}
        
        try:
            # Test 1: Skill-based selection
            workflow_data = {
                "name": "Benchmark: Multi-Skill Workflow",
                "description": "Test workflow requiring diverse skills",
                "task_description": """
                    1. Analyze security vulnerabilities in a web application
                    2. Optimize database queries for performance
                    3. Generate API documentation
                    4. Create unit tests for critical functions
                """,
                "type": "technical",
                "priority": "high",
                "max_agents": 4
            }
            
            response = requests.post(
                f"{self.base_url}/v1/workflows",
                json=workflow_data
            )
            
            if response.status_code == 200:
                result = response.json()
                workflow_id = result.get("id")
                
                # Execute workflow
                exec_response = requests.post(
                    f"{self.base_url}/v1/workflows/{workflow_id}/execute"
                )
                
                if exec_response.status_code == 200:
                    exec_result = exec_response.json()
                    
                    # Analyze agent selection
                    if "selected_agents" in exec_result:
                        agents = exec_result["selected_agents"]
                        metrics["agents_selected"] = len(agents)
                        metrics["skill_coverage"] = self._calculate_skill_coverage(agents)
                        metrics["selection_time"] = exec_result.get("selection_time", 0)
                        
                        logger.info(f"✅ Selected {len(agents)} agents with {metrics['skill_coverage']}% skill coverage")
                    else:
                        errors.append("No agent selection data in response")
                else:
                    errors.append(f"Workflow execution failed: {exec_response.status_code}")
            else:
                errors.append(f"Workflow creation failed: {response.status_code}")
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"❌ Agent selection benchmark failed: {e}")
        
        execution_time = time.time() - start_time
        return BenchmarkResult(
            test_name="Agent Selection",
            category="Core Functionality",
            success=len(errors) == 0,
            execution_time=execution_time,
            metrics=metrics,
            errors=errors,
            timestamp=datetime.now()
        )
    
    async def benchmark_memory_system(self) -> BenchmarkResult:
        """Benchmark memory storage and retrieval"""
        logger.info("\n📋 Testing: Memory System Performance")
        start_time = time.time()
        errors = []
        metrics = {}
        
        try:
            # Use an existing agent ID (assuming at least one exists)
            if not self.available_agents:
                errors.append("No agents available for memory testing")
                return BenchmarkResult(
                    test_name="Memory System",
                    category="Memory & Learning",
                    success=False,
                    execution_time=time.time() - start_time,
                    metrics=metrics,
                    errors=errors,
                    timestamp=datetime.now()
                )
            
            agent_id = self.available_agents[0]["id"]
            
            # Test memory storage
            memory_items = []
            storage_times = []
            
            for i in range(10):
                memory_data = {
                    "agent_id": agent_id,
                    "memory_type": "working_memory",
                    "content": f"Benchmark memory item {i}: Testing memory storage performance",
                    "context": {"test_id": i, "timestamp": datetime.now().isoformat()}
                }
                
                store_start = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/memory/store",
                    json=memory_data
                )
                storage_times.append(time.time() - store_start)
                
                if response.status_code == 200:
                    memory_items.append(response.json())
                else:
                    errors.append(f"Memory storage {i} failed: {response.status_code}")
            
            metrics["avg_storage_time"] = statistics.mean(storage_times) if storage_times else 0
            metrics["items_stored"] = len(memory_items)
            
            # Test memory retrieval
            retrieval_start = time.time()
            response = requests.get(
                f"{self.base_url}/v1/memory/retrieve",
                params={"agent_id": agent_id, "limit": 10}
            )
            metrics["retrieval_time"] = time.time() - retrieval_start
            
            if response.status_code == 200:
                retrieved = response.json()
                metrics["items_retrieved"] = len(retrieved)
                logger.info(f"✅ Stored {metrics['items_stored']} items, retrieved {metrics['items_retrieved']}")
            else:
                errors.append(f"Memory retrieval failed: {response.status_code}")
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"❌ Memory system benchmark failed: {e}")
        
        execution_time = time.time() - start_time
        return BenchmarkResult(
            test_name="Memory System",
            category="Memory & Learning",
            success=len(errors) == 0,
            execution_time=execution_time,
            metrics=metrics,
            errors=errors,
            timestamp=datetime.now()
        )
    
    async def benchmark_communication(self) -> BenchmarkResult:
        """Benchmark inter-agent communication"""
        logger.info("\n📋 Testing: Inter-Agent Communication")
        start_time = time.time()
        errors = []
        metrics = {}
        
        try:
            # Create a collaborative workflow
            workflow_data = {
                "name": "Benchmark: Collaborative Task",
                "description": "Test workflow requiring agent collaboration",
                "task_description": "Design and implement a microservices architecture with security review",
                "type": "collaborative",
                "priority": "high",
                "enable_communication": True
            }
            
            response = requests.post(
                f"{self.base_url}/v1/workflows",
                json=workflow_data
            )
            
            if response.status_code == 200:
                workflow_id = response.json()["id"]
                
                # Execute and monitor communication
                exec_response = requests.post(
                    f"{self.base_url}/v1/workflows/{workflow_id}/execute"
                )
                
                if exec_response.status_code == 200:
                    # Get communication logs
                    logs_response = requests.get(
                        f"{self.base_url}/v1/workflows/{workflow_id}/communications"
                    )
                    
                    if logs_response.status_code == 200:
                        communications = logs_response.json()
                        metrics["total_messages"] = len(communications)
                        metrics["message_types"] = self._analyze_message_types(communications)
                        metrics["avg_response_time"] = self._calculate_avg_response_time(communications)
                        
                        logger.info(f"✅ {metrics['total_messages']} messages exchanged")
                    else:
                        errors.append(f"Failed to get communication logs: {logs_response.status_code}")
                else:
                    errors.append(f"Workflow execution failed: {exec_response.status_code}")
            else:
                errors.append(f"Workflow creation failed: {response.status_code}")
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"❌ Communication benchmark failed: {e}")
        
        execution_time = time.time() - start_time
        return BenchmarkResult(
            test_name="Inter-Agent Communication",
            category="Collaboration",
            success=len(errors) == 0,
            execution_time=execution_time,
            metrics=metrics,
            errors=errors,
            timestamp=datetime.now()
        )
    
    async def benchmark_context_engineering(self) -> BenchmarkResult:
        """Benchmark context retrieval and RAG"""
        logger.info("\n📋 Testing: Context Engineering & RAG")
        start_time = time.time()
        errors = []
        metrics = {}
        
        try:
            # Test document upload and indexing
            test_content = "This is a benchmark test document for RAG system evaluation."
            
            upload_start = time.time()
            response = requests.post(
                f"{self.base_url}/v1/documents/upload",
                files={"file": ("benchmark.txt", test_content, "text/plain")}
            )
            metrics["upload_time"] = time.time() - upload_start
            
            if response.status_code == 200:
                doc_id = response.json()["id"]
                
                # Test semantic search
                search_start = time.time()
                search_response = requests.post(
                    f"{self.base_url}/v1/search",
                    json={"query": "benchmark evaluation", "limit": 5}
                )
                metrics["search_time"] = time.time() - search_start
                
                if search_response.status_code == 200:
                    results = search_response.json()
                    metrics["search_results"] = len(results)
                    metrics["avg_relevance_score"] = statistics.mean(
                        [r.get("score", 0) for r in results]
                    ) if results else 0
                    
                    logger.info(f"✅ Search returned {metrics['search_results']} results")
                else:
                    errors.append(f"Search failed: {search_response.status_code}")
            else:
                errors.append(f"Document upload failed: {response.status_code}")
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"❌ Context engineering benchmark failed: {e}")
        
        execution_time = time.time() - start_time
        return BenchmarkResult(
            test_name="Context Engineering",
            category="Knowledge Management",
            success=len(errors) == 0,
            execution_time=execution_time,
            metrics=metrics,
            errors=errors,
            timestamp=datetime.now()
        )
    
    async def benchmark_workflow_performance(self) -> BenchmarkResult:
        """Benchmark overall workflow performance"""
        logger.info("\n📋 Testing: Workflow Performance & Scalability")
        start_time = time.time()
        errors = []
        metrics = {}
        
        try:
            # Test different workflow complexities
            complexities = ["simple", "moderate", "complex"]
            performance_data = {}
            
            for complexity in complexities:
                task_count = {"simple": 3, "moderate": 7, "complex": 15}[complexity]
                
                workflow_data = {
                    "name": f"Benchmark: {complexity.title()} Workflow",
                    "description": f"Performance test with {task_count} subtasks",
                    "task_description": "\n".join([
                        f"{i+1}. Subtask {i+1}: Process data segment {i+1}"
                        for i in range(task_count)
                    ]),
                    "type": "batch_processing",
                    "priority": "normal"
                }
                
                # Create and execute workflow
                create_start = time.time()
                response = requests.post(
                    f"{self.base_url}/v1/workflows",
                    json=workflow_data
                )
                
                if response.status_code == 200:
                    workflow_id = response.json()["id"]
                    
                    exec_start = time.time()
                    exec_response = requests.post(
                        f"{self.base_url}/v1/workflows/{workflow_id}/execute"
                    )
                    
                    if exec_response.status_code == 200:
                        # Wait for completion (with timeout)
                        completed = await self._wait_for_completion(workflow_id, timeout=300)
                        
                        if completed:
                            total_time = time.time() - exec_start
                            performance_data[complexity] = {
                                "subtasks": task_count,
                                "execution_time": total_time,
                                "time_per_task": total_time / task_count
                            }
                            
                            logger.info(f"✅ {complexity.title()}: {total_time:.2f}s for {task_count} tasks")
                        else:
                            errors.append(f"{complexity} workflow timeout")
                    else:
                        errors.append(f"{complexity} execution failed: {exec_response.status_code}")
                else:
                    errors.append(f"{complexity} creation failed: {response.status_code}")
            
            metrics["performance_by_complexity"] = performance_data
            
            # Calculate scalability metrics
            if len(performance_data) >= 2:
                times = [v["time_per_task"] for v in performance_data.values()]
                metrics["scalability_factor"] = max(times) / min(times)
                metrics["avg_time_per_task"] = statistics.mean(times)
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"❌ Workflow performance benchmark failed: {e}")
        
        execution_time = time.time() - start_time
        return BenchmarkResult(
            test_name="Workflow Performance",
            category="Performance",
            success=len(errors) == 0,
            execution_time=execution_time,
            metrics=metrics,
            errors=errors,
            timestamp=datetime.now()
        )
    
    async def benchmark_learning_system(self) -> BenchmarkResult:
        """Benchmark learning and adaptation capabilities"""
        logger.info("\n📋 Testing: Learning System & Adaptation")
        start_time = time.time()
        errors = []
        metrics = {}
        
        try:
            # Execute similar workflows to test learning
            base_task = "Optimize a Python function for better performance"
            iterations = 3
            execution_times = []
            quality_scores = []
            
            for i in range(iterations):
                workflow_data = {
                    "name": f"Benchmark: Learning Test {i+1}",
                    "description": "Test learning from repeated similar tasks",
                    "task_description": f"{base_task} - Iteration {i+1}",
                    "type": "optimization",
                    "priority": "normal"
                }
                
                response = requests.post(
                    f"{self.base_url}/v1/workflows",
                    json=workflow_data
                )
                
                if response.status_code == 200:
                    workflow_id = response.json()["id"]
                    
                    exec_start = time.time()
                    exec_response = requests.post(
                        f"{self.base_url}/v1/workflows/{workflow_id}/execute"
                    )
                    
                    if exec_response.status_code == 200:
                        # Wait for completion
                        completed = await self._wait_for_completion(workflow_id, timeout=120)
                        
                        if completed:
                            exec_time = time.time() - exec_start
                            execution_times.append(exec_time)
                            
                            # Get quality metrics
                            analytics_response = requests.get(
                                f"{self.base_url}/v1/analytics/workflow/{workflow_id}"
                            )
                            
                            if analytics_response.status_code == 200:
                                analytics = analytics_response.json()
                                quality = analytics.get("quality_score", 0.5)
                                quality_scores.append(quality)
                                
                                logger.info(f"✅ Iteration {i+1}: {exec_time:.2f}s, Quality: {quality:.2f}")
                            else:
                                errors.append(f"Failed to get analytics for iteration {i+1}")
                        else:
                            errors.append(f"Iteration {i+1} timeout")
                    else:
                        errors.append(f"Iteration {i+1} execution failed")
                else:
                    errors.append(f"Iteration {i+1} creation failed")
            
            # Analyze learning improvement
            if len(execution_times) >= 2:
                metrics["time_improvement"] = (execution_times[0] - execution_times[-1]) / execution_times[0]
                metrics["quality_improvement"] = (quality_scores[-1] - quality_scores[0]) if quality_scores else 0
                metrics["learning_rate"] = metrics["time_improvement"] / iterations
                
                logger.info(f"📈 Learning improvement: {metrics['time_improvement']*100:.1f}% faster")
                
        except Exception as e:
            errors.append(str(e))
            logger.error(f"❌ Learning system benchmark failed: {e}")
        
        execution_time = time.time() - start_time
        return BenchmarkResult(
            test_name="Learning System",
            category="Learning & Adaptation",
            success=len(errors) == 0,
            execution_time=execution_time,
            metrics=metrics,
            errors=errors,
            timestamp=datetime.now()
        )
    
    async def run_full_benchmark(self) -> BenchmarkSuite:
        """Run the complete benchmark suite"""
        logger.info("=" * 60)
        logger.info("🏁 STARTING AUTOMATOS AI PLATFORM BENCHMARK SUITE")
        logger.info("=" * 60)
        
        self.start_time = time.time()
        await self.setup()
        
        # Run all benchmarks
        benchmarks = [
            self.benchmark_agent_selection(),
            self.benchmark_memory_system(),
            self.benchmark_communication(),
            self.benchmark_context_engineering(),
            self.benchmark_workflow_performance(),
            self.benchmark_learning_system()
        ]
        
        # Execute benchmarks concurrently where possible
        results = await asyncio.gather(*benchmarks, return_exceptions=True)
        
        # Process results
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Benchmark failed with exception: {result}")
                self.results.append(BenchmarkResult(
                    test_name="Unknown",
                    category="Error",
                    success=False,
                    execution_time=0,
                    metrics={},
                    errors=[str(result)],
                    timestamp=datetime.now()
                ))
            else:
                self.results.append(result)
        
        # Generate summary
        total_time = time.time() - self.start_time
        passed = sum(1 for r in self.results if r.success)
        failed = len(self.results) - passed
        
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {
                    "total": 0,
                    "passed": 0,
                    "failed": 0,
                    "avg_time": []
                }
            
            categories[result.category]["total"] += 1
            if result.success:
                categories[result.category]["passed"] += 1
            else:
                categories[result.category]["failed"] += 1
            categories[result.category]["avg_time"].append(result.execution_time)
        
        # Calculate averages
        for cat in categories.values():
            cat["avg_time"] = statistics.mean(cat["avg_time"]) if cat["avg_time"] else 0
        
        suite = BenchmarkSuite(
            total_tests=len(self.results),
            passed=passed,
            failed=failed,
            total_time=total_time,
            categories=categories,
            detailed_results=self.results,
            summary={
                "success_rate": (passed / len(self.results) * 100) if self.results else 0,
                "avg_execution_time": statistics.mean([r.execution_time for r in self.results]) if self.results else 0,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Save results
        self._save_results(suite)
        
        # Print summary
        self._print_summary(suite)
        
        return suite
    
    def _calculate_skill_coverage(self, agents: List[Dict]) -> float:
        """Calculate skill coverage percentage"""
        required_skills = {"security", "performance", "documentation", "testing"}
        covered_skills = set()
        
        for agent in agents:
            for skill in agent.get("skills", []):
                if any(req in skill.lower() for req in required_skills):
                    covered_skills.add(skill)
        
        return (len(covered_skills) / len(required_skills)) * 100 if required_skills else 0
    
    def _analyze_message_types(self, communications: List[Dict]) -> Dict[str, int]:
        """Analyze types of messages exchanged"""
        types = {}
        for msg in communications:
            msg_type = msg.get("type", "unknown")
            types[msg_type] = types.get(msg_type, 0) + 1
        return types
    
    def _calculate_avg_response_time(self, communications: List[Dict]) -> float:
        """Calculate average response time between messages"""
        if len(communications) < 2:
            return 0
        
        response_times = []
        for i in range(1, len(communications)):
            if communications[i].get("in_reply_to") == communications[i-1].get("id"):
                time_diff = (
                    datetime.fromisoformat(communications[i]["timestamp"]) -
                    datetime.fromisoformat(communications[i-1]["timestamp"])
                ).total_seconds()
                response_times.append(time_diff)
        
        return statistics.mean(response_times) if response_times else 0
    
    async def _wait_for_completion(self, workflow_id: int, timeout: int = 300) -> bool:
        """Wait for workflow to complete with timeout"""
        start = time.time()
        
        while time.time() - start < timeout:
            try:
                response = requests.get(f"{self.base_url}/v1/workflows/{workflow_id}/status")
                if response.status_code == 200:
                    status = response.json().get("status")
                    if status in ["completed", "failed"]:
                        return status == "completed"
                
                await asyncio.sleep(2)
            except Exception:
                pass
        
        return False
    
    def _save_results(self, suite: BenchmarkSuite):
        """Save benchmark results to file"""
        output_dir = Path("benchmark_results")
        output_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON
        with open(output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", "w") as f:
            json.dump(asdict(suite), f, indent=2, default=str)
        
        # Save summary markdown
        with open(output_dir / "benchmark_summary.md", "w") as f:
            f.write("# Automatos AI Platform Benchmark Results\n\n")
            f.write(f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Total Tests**: {suite.total_tests}\n")
            f.write(f"**Passed**: {suite.passed}\n")
            f.write(f"**Failed**: {suite.failed}\n")
            f.write(f"**Success Rate**: {suite.summary['success_rate']:.1f}%\n")
            f.write(f"**Total Time**: {suite.total_time:.2f}s\n\n")
            
            f.write("## Results by Category\n\n")
            for cat_name, cat_data in suite.categories.items():
                f.write(f"### {cat_name}\n")
                f.write(f"- Tests: {cat_data['total']}\n")
                f.write(f"- Passed: {cat_data['passed']}\n")
                f.write(f"- Failed: {cat_data['failed']}\n")
                f.write(f"- Avg Time: {cat_data['avg_time']:.2f}s\n\n")
            
            f.write("## Detailed Results\n\n")
            for result in suite.detailed_results:
                status = "✅" if result.success else "❌"
                f.write(f"### {status} {result.test_name}\n")
                f.write(f"- Time: {result.execution_time:.2f}s\n")
                if result.metrics:
                    f.write("- Metrics:\n")
                    for key, value in result.metrics.items():
                        f.write(f"  - {key}: {value}\n")
                if result.errors:
                    f.write("- Errors:\n")
                    for error in result.errors:
                        f.write(f"  - {error}\n")
                f.write("\n")
        
        logger.info(f"📁 Results saved to {output_dir}")
    
    def _print_summary(self, suite: BenchmarkSuite):
        """Print benchmark summary to console"""
        logger.info("\n" + "=" * 60)
        logger.info("📊 BENCHMARK COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {suite.total_tests}")
        logger.info(f"Passed: {suite.passed} ✅")
        logger.info(f"Failed: {suite.failed} ❌")
        logger.info(f"Success Rate: {suite.summary['success_rate']:.1f}%")
        logger.info(f"Total Time: {suite.total_time:.2f}s")
        logger.info(f"Avg Time per Test: {suite.summary['avg_execution_time']:.2f}s")
        logger.info("=" * 60)

async def main():
    """Main entry point"""
    benchmark = AutomatosBenchmark()
    suite = await benchmark.run_full_benchmark()
    
    # Return exit code based on success
    sys.exit(0 if suite.failed == 0 else 1)

if __name__ == "__main__":
    asyncio.run(main())
