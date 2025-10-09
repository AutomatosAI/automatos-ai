#!/usr/bin/env python3
"""
Simple Workflow Test - Tests the core workflow functionality
"""

import requests
import json
import time
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
API_BASE = "https://api.automatos.app"
HEADERS = {"Content-Type": "application/json"}

def test_workflow_execution():
    """Test basic workflow execution with existing data"""
    
    logger.info("=" * 60)
    logger.info("🚀 SIMPLE WORKFLOW TEST")
    logger.info("=" * 60)
    
    # 1. Create a simple test workflow
    workflow_data = {
        "name": f"Test Workflow {datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "description": "Simple test to verify system functionality",
        "workflow_definition": {
            "goal": "Analyze the benefits of AI automation and create a brief summary",
            "context": "We need to understand how AI can improve business processes",
            "constraints": ["Keep it concise", "Focus on practical benefits"],
            "workflow_type": "research"
        },
        "is_active": True
    }
    
    logger.info("📝 Creating test workflow...")
    try:
        # Try different endpoint patterns
        endpoints = [
            f"{API_BASE}/workflows",
            f"{API_BASE}/v1/workflows",
            f"{API_BASE}/api/v1/workflows",
            f"{API_BASE}/api/workflows"
        ]
        
        workflow_id = None
        for endpoint in endpoints:
            response = requests.post(endpoint, headers=HEADERS, json=workflow_data)
            if response.status_code in [200, 201]:
                result = response.json()
                workflow_id = result.get("id") or result.get("workflow_id")
                logger.info(f"✅ Workflow created with ID: {workflow_id} at {endpoint}")
                break
            elif response.status_code != 404:
                logger.warning(f"Endpoint {endpoint} returned: {response.status_code}")
        
        if not workflow_id:
            logger.error("❌ Could not create workflow - trying to list existing workflows")
            
            # Try to list existing workflows
            for endpoint in endpoints:
                response = requests.get(endpoint, headers=HEADERS)
                if response.status_code == 200:
                    workflows = response.json()
                    if workflows and len(workflows) > 0:
                        workflow_id = workflows[0].get("id")
                        logger.info(f"📋 Using existing workflow ID: {workflow_id}")
                        break
        
        if not workflow_id:
            logger.error("❌ No workflows available for testing")
            return
        
        # 2. Execute the workflow
        logger.info(f"▶️  Executing workflow {workflow_id}...")
        
        execution_endpoints = [
            f"{API_BASE}/workflows/{workflow_id}/execute",
            f"{API_BASE}/v1/workflows/{workflow_id}/execute",
            f"{API_BASE}/api/v1/workflows/{workflow_id}/execute",
            f"{API_BASE}/api/workflows/{workflow_id}/execute"
        ]
        
        execution_id = None
        for endpoint in execution_endpoints:
            response = requests.post(
                endpoint,
                headers=HEADERS,
                json={"input_data": {"test": True}}
            )
            if response.status_code in [200, 201, 202]:
                result = response.json()
                execution_id = result.get("execution_id") or result.get("id")
                logger.info(f"✅ Workflow execution started: {execution_id}")
                break
        
        if not execution_id:
            logger.error("❌ Could not execute workflow")
            return
        
        # 3. Monitor execution progress
        logger.info("📊 Monitoring execution progress...")
        
        status_endpoints = [
            f"{API_BASE}/workflows/{workflow_id}/executions/{execution_id}",
            f"{API_BASE}/v1/workflows/{workflow_id}/executions/{execution_id}",
            f"{API_BASE}/api/v1/workflows/{workflow_id}/executions/{execution_id}",
            f"{API_BASE}/api/workflows/{workflow_id}/executions/{execution_id}"
        ]
        
        max_attempts = 30  # Wait up to 30 seconds
        for attempt in range(max_attempts):
            time.sleep(1)
            
            for endpoint in status_endpoints:
                response = requests.get(endpoint, headers=HEADERS)
                if response.status_code == 200:
                    execution = response.json()
                    status = execution.get("status", "UNKNOWN")
                    logger.info(f"  Status: {status}")
                    
                    if status in ["COMPLETED", "FAILED"]:
                        logger.info(f"🏁 Execution finished with status: {status}")
                        
                        # Get analytics data
                        logger.info("📈 Fetching analytics data...")
                        analytics_endpoints = [
                            f"{API_BASE}/analytics/workflows/{workflow_id}",
                            f"{API_BASE}/v1/analytics/workflows/{workflow_id}",
                            f"{API_BASE}/api/v1/analytics/workflows/{workflow_id}"
                        ]
                        
                        for analytics_endpoint in analytics_endpoints:
                            response = requests.get(analytics_endpoint, headers=HEADERS)
                            if response.status_code == 200:
                                analytics = response.json()
                                logger.info(f"📊 Analytics: {json.dumps(analytics, indent=2)}")
                                break
                        
                        # Get benchmarking data
                        logger.info("🎯 Fetching benchmarking data...")
                        benchmark_endpoints = [
                            f"{API_BASE}/benchmarking/performance-summary",
                            f"{API_BASE}/v1/benchmarking/performance-summary",
                            f"{API_BASE}/api/v1/benchmarking/performance-summary"
                        ]
                        
                        for benchmark_endpoint in benchmark_endpoints:
                            response = requests.get(benchmark_endpoint, headers=HEADERS)
                            if response.status_code == 200:
                                benchmarks = response.json()
                                logger.info(f"📈 Benchmarks: {json.dumps(benchmarks, indent=2)}")
                                break
                        
                        return
                    break
        
        logger.warning("⏱️  Execution timed out")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}")

def test_analytics_endpoints():
    """Test the analytics and benchmarking endpoints"""
    
    logger.info("=" * 60)
    logger.info("📊 TESTING ANALYTICS ENDPOINTS")
    logger.info("=" * 60)
    
    endpoints = [
        ("/analytics/system", "System Analytics"),
        ("/benchmarking/performance-summary", "Performance Summary"),
        ("/benchmarking/learning-effectiveness", "Learning Effectiveness"),
        ("/benchmarking/agent-performance", "Agent Performance"),
        ("/benchmarking/cost-roi", "Cost ROI Analysis")
    ]
    
    for path, name in endpoints:
        logger.info(f"\n📈 Testing {name}...")
        
        # Try different URL patterns
        urls = [
            f"{API_BASE}{path}",
            f"{API_BASE}/v1{path}",
            f"{API_BASE}/api/v1{path}",
            f"{API_BASE}/api{path}"
        ]
        
        for url in urls:
            response = requests.get(url, headers=HEADERS)
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ {name} working at {url}")
                logger.info(f"   Data preview: {str(data)[:200]}...")
                break
            elif response.status_code != 404:
                logger.warning(f"   {url} returned: {response.status_code}")

if __name__ == "__main__":
    # Run tests
    test_workflow_execution()
    test_analytics_endpoints()
    
    logger.info("=" * 60)
    logger.info("✅ TEST SUITE COMPLETE")
    logger.info("=" * 60)
