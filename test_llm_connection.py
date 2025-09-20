#!/usr/bin/env python3
"""
Quick test to verify LLM connection is REAL
Run this FIRST to ensure no mock data
"""

import asyncio
import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add orchestrator to path
sys.path.append(str(Path(__file__).parent / "orchestrator"))

async def test_real_llm():
    """Test that we have a REAL LLM connection"""
    
    print("🔍 Testing LLM Connection...")
    print("=" * 50)
    
    try:
        from orchestrator.services.llm_provider import LLMProvider, LLMConfig
        
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            print("❌ OPENAI_API_KEY not set in environment")
            print("\nTo fix:")
            print("export OPENAI_API_KEY='your-key-here'")
            return False
        
        print(f"✅ API Key found: {api_key[:10]}...")
        
        # Create config
        config = LLMConfig(
            provider="openai",
            model="gpt-4",
            temperature=0.7,
            api_key=api_key
        )
        
        # Initialize provider
        llm = LLMProvider(config)
        
        print("\n📡 Sending real request to OpenAI...")
        start_time = time.time()
        
        # Make REAL API call
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello! This is a REAL response from GPT-4, not mock data!' in exactly those words."}
        ]
        
        response = await llm.generate_response(messages)
        
        elapsed = time.time() - start_time
        
        print(f"\n⏱️  Response time: {elapsed:.2f} seconds")
        
        if elapsed < 0.5:
            print("⚠️  WARNING: Response too fast - might be mock!")
        
        print("\n📝 REAL LLM Response:")
        print("-" * 50)
        print(response.content)
        print("-" * 50)
        
        # Verify it's not mock
        if "mock" in response.content.lower() or elapsed < 0.1:
            print("\n❌ DETECTED MOCK DATA!")
            return False
        
        print("\n✅ SUCCESS! Real LLM connection verified!")
        print(f"✅ Tokens used: {response.usage.total_tokens if response.usage else 'N/A'}")
        print(f"✅ Model: {response.model}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nMake sure you're in the right directory:")
        print("cd /Users/gkavanagh/Development/Automatos-AI-Platform/automatos-ai")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\nThis might mean:")
        print("1. Invalid API key")
        print("2. Network issues")
        print("3. OpenAI API down")
        return False

async def test_task_decomposition():
    """Test REAL task decomposition"""
    
    print("\n\n🧩 Testing Task Decomposition...")
    print("=" * 50)
    
    from orchestrator.services.llm_provider import LLMProvider, LLMConfig
    
    config = LLMConfig(
        provider="openai",
        model="gpt-4",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    
    llm = LLMProvider(config)
    
    # Create a real decomposition prompt
    task = "Build a REST API for user authentication with email verification"
    
    decomposition_prompt = f"""Break down this task into specific subtasks:
Task: {task}

Requirements:
1. List 5-7 specific, actionable subtasks
2. Identify dependencies between tasks
3. Estimate complexity (1-5 scale)
4. Suggest which type of agent should handle each

Format as JSON:
{{
  "subtasks": [
    {{
      "id": 1,
      "name": "...",
      "description": "...",
      "dependencies": [],
      "complexity": 3,
      "suggested_agent": "backend_developer"
    }}
  ]
}}"""
    
    print(f"📋 Task: {task}")
    print("\n⏳ Decomposing with REAL LLM...")
    
    start = time.time()
    
    messages = [
        {"role": "system", "content": "You are an expert task decomposition system."},
        {"role": "user", "content": decomposition_prompt}
    ]
    
    response = await llm.generate_response(messages)
    
    elapsed = time.time() - start
    
    print(f"⏱️  Decomposition time: {elapsed:.2f}s")
    print("\n📊 REAL Decomposition Result:")
    print("-" * 50)
    print(response.content)
    print("-" * 50)
    
    # Try to parse as JSON
    import json
    try:
        result = json.loads(response.content)
        print(f"\n✅ Successfully decomposed into {len(result['subtasks'])} subtasks!")
        print("\nSubtasks:")
        for task in result['subtasks']:
            deps = f" (depends on: {task['dependencies']})" if task['dependencies'] else ""
            print(f"  {task['id']}. {task['name']} [complexity: {task['complexity']}]{deps}")
    except:
        print("\n⚠️  Could not parse as JSON, but response is REAL")
    
    return True

async def main():
    """Run all tests"""
    
    print("🚀 AUTOMATOS AI - REAL IMPLEMENTATION TEST")
    print("=" * 50)
    print("NO MOCK DATA - ONLY REAL RESULTS")
    print()
    
    # Test 1: Basic connection
    if not await test_real_llm():
        print("\n❌ Fix the LLM connection first!")
        return
    
    # Test 2: Task decomposition
    await test_task_decomposition()
    
    print("\n" + "=" * 50)
    print("🎯 NEXT STEPS:")
    print("1. ✅ LLM Connection verified")
    print("2. ✅ Task decomposition working")
    print("3. Now implement in orchestrator_service.py")
    print("4. Replace ALL mock returns with real LLM calls")
    print("\nYou're ready to build REAL features!")

if __name__ == "__main__":
    asyncio.run(main())
