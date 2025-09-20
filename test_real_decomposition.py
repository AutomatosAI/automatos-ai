#!/usr/bin/env python3
"""
Test REAL Task Decomposition
=============================

Run this on your dev environment to verify LLM-based task decomposition
"""

import asyncio
import json
import time
from datetime import datetime
import sys
from pathlib import Path

# Add to path
sys.path.append(str(Path(__file__).parent / "orchestrator"))

async def test_decomposition():
    """Test the REAL task decomposer"""
    
    print("🚀 TESTING REAL TASK DECOMPOSITION")
    print("=" * 60)
    print("This will make actual API calls to OpenAI GPT-4")
    print()
    
    from orchestrator.core.real_task_decomposer import RealTaskDecomposer
    
    # Initialize decomposer
    decomposer = RealTaskDecomposer()
    
    # Test cases
    test_tasks = [
        {
            "description": "Build a user authentication system with email verification, password reset, and 2FA support",
            "type": "development",
            "complexity": "high"
        },
        {
            "description": "Analyze customer feedback data to identify top 3 pain points and create action plan",
            "type": "analysis", 
            "complexity": "medium"
        },
        {
            "description": "Write a blog post about the benefits of AI in healthcare",
            "type": "content",
            "complexity": "low"
        }
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\n📋 Test {i}/{len(test_tasks)}")
        print(f"Task: {task['description'][:80]}...")
        print(f"Type: {task['type']}, Complexity: {task['complexity']}")
        print("-" * 60)
        
        try:
            # Measure time
            start_time = time.time()
            
            # REAL decomposition
            result = await decomposer.decompose_task(
                task_description=task["description"],
                task_type=task["type"],
                complexity=task["complexity"]
            )
            
            elapsed = time.time() - start_time
            
            # Display results
            print(f"\n✅ Decomposition completed in {elapsed:.2f} seconds")
            print(f"🤖 Model used: {result.get('llm_model', 'unknown')}")
            print(f"📊 Tokens used: {result.get('tokens_used', 'N/A')}")
            print(f"🔍 Is real decomposition: {result.get('is_real_decomposition', False)}")
            
            # Show subtasks
            subtasks = result.get("subtasks", [])
            print(f"\n📝 Generated {len(subtasks)} subtasks:")
            
            for j, subtask in enumerate(subtasks, 1):
                deps = subtask.get("dependencies", [])
                dep_str = f" (depends on: {', '.join(deps)})" if deps else ""
                print(f"  {j}. [{subtask['priority']}] {subtask['description'][:60]}{dep_str}")
                print(f"     Agent: {subtask['agent_type']}, Duration: {subtask['estimated_duration']}")
            
            # Show execution strategy
            print(f"\n🎯 Execution Strategy: {result.get('execution_strategy', 'unknown')}")
            print(f"⏱️  Total Estimated Time: {result.get('total_estimated_time', 'unknown')}")
            
            # Show complexity assessment
            if "complexity_assessment" in result:
                ca = result["complexity_assessment"]
                print(f"\n📈 Complexity Assessment:")
                print(f"  - Technical: {ca.get('technical_complexity', 'N/A')}")
                print(f"  - Coordination: {ca.get('coordination_complexity', 'N/A')}")
                print(f"  - Resources: {ca.get('resource_requirements', 'N/A')}")
            
            # Verify it's not mock
            if elapsed < 0.5:
                print("\n⚠️  WARNING: Response was very fast - might be cached or mock!")
            
            if "mock" in json.dumps(result).lower():
                print("\n❌ ERROR: Found 'mock' in response - this might be fake data!")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Make sure OPENAI_API_KEY is set in your environment")
    
    print("\n" + "=" * 60)
    print("✅ Testing complete!")
    print("\nVerification checklist:")
    print("- [ ] Each decomposition took 1-3 seconds (not instant)")
    print("- [ ] Different subtasks for different inputs")  
    print("- [ ] Logical dependencies between subtasks")
    print("- [ ] Real agent types and durations")
    print("- [ ] No mock data indicators")

async def test_api_endpoint():
    """Test the API endpoint directly"""
    
    print("\n\n🌐 TESTING API ENDPOINT")
    print("=" * 60)
    
    import aiohttp
    import os
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get API configuration from environment
    api_host = os.getenv("API_URL", "localhost")
    api_port = os.getenv("API_PORT", "8000")
    api_key = os.getenv("API_KEY", "")
    
    # Construct URL based on environment
    if api_host == "localhost" or "127.0.0.1" in api_host:
        api_url = f"http://{api_host}:{api_port}/api/orchestrator/task/analyze"
    else:
        # For production/staging, use HTTPS and no port
        api_url = f"https://{api_host}/api/orchestrator/task/analyze"
    
    test_payload = {
        "task_id": f"test_{int(datetime.now().timestamp())}",
        "description": "Create a machine learning model to predict customer churn",
        "type": "data_science",
        "complexity": "high",
        "requirements": [
            "Use customer behavior data",
            "Include feature engineering",
            "Provide model explainability"
        ]
    }
    
    print(f"📡 Calling: {api_url}")
    print(f"🔑 API Key: {'Set' if api_key else 'Not set'}")
    print(f"📦 Payload: {json.dumps(test_payload, indent=2)}")
    
    # Prepare headers with API key if available
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    
    try:
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            
            async with session.post(api_url, json=test_payload, headers=headers) as response:
                elapsed = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    
                    print(f"\n✅ API call successful in {elapsed:.2f} seconds")
                    print(f"📊 Response:")
                    print(json.dumps(result, indent=2))
                    
                    # Verify it's real
                    if result.get("is_real_decomposition"):
                        print("\n✅ Confirmed: Using REAL LLM decomposition!")
                    else:
                        print("\n⚠️  Warning: May not be using real decomposition")
                else:
                    error = await response.text()
                    print(f"\n❌ API Error ({response.status}): {error}")
                    
    except aiohttp.ClientConnectorError:
        print("\n❌ Could not connect to API")
        print("Make sure the FastAPI server is running:")
        print("cd orchestrator && python main.py")
    except Exception as e:
        print(f"\n❌ Error: {e}")

async def main():
    """Run all tests"""
    
    # Test direct decomposer
    await test_decomposition()
    
    # Test API endpoint (if server is running)
    await test_api_endpoint()

if __name__ == "__main__":
    print("🎯 AUTOMATOS AI - REAL IMPLEMENTATION VERIFICATION")
    print("=" * 60)
    print("NO MOCK DATA ALLOWED")
    print()
    
    import os
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  WARNING: OPENAI_API_KEY not set in environment")
        print("The tests will fail without it!")
        print("\nSet it with:")
        print("export OPENAI_API_KEY='your-key-here'")
    
    asyncio.run(main())
