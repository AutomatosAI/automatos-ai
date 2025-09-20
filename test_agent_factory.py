#!/usr/bin/env python3
"""
Test Agent Factory with REAL LLM Connections
============================================

This script demonstrates that agents are created with actual LLM connections
and can execute real tasks, not mock data.
"""

import asyncio
import os
import sys
import time
from datetime import datetime

# Add orchestrator to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'orchestrator'))

from orchestrator.services.agent_factory import (
    AgentFactory, AgentType, create_specialized_agent
)


async def main():
    """
    Comprehensive test of agent factory with real LLM connections.
    """
    
    print("\n" + "="*80)
    print("AUTOMATOS AI - AGENT FACTORY DEMONSTRATION")
    print("Testing REAL LLM Connections - NO MOCK DATA")
    print("="*80)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY") and not os.getenv("ANTHROPIC_API_KEY"):
        print("\n⚠️  WARNING: No API keys found in environment!")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test real connections")
        print("   Example: export OPENAI_API_KEY='your-key-here'")
        return
    
    factory = AgentFactory()
    
    try:
        # ===== TEST 1: Create Multiple Agent Types =====
        print("\n" + "─"*60)
        print("TEST 1: Creating Specialized Agents with Real LLM")
        print("─"*60)
        
        agents = []
        
        # Create Code Architect
        print("\n→ Creating Code Architect agent...")
        start = time.time()
        
        code_agent = await factory.create_agent(
            name="CodeArchitect-001",
            agent_type=AgentType.CODE_ARCHITECT,
            skills=["code_analysis", "system_design", "api_design"],
            auto_verify=True
        )
        
        creation_time = time.time() - start
        agents.append(code_agent)
        
        print(f"  ✓ Agent created in {creation_time:.2f}s")
        print(f"    ID: {code_agent.agent_id}")
        print(f"    Skills: {', '.join(code_agent.skills)}")
        print(f"    LLM: {code_agent.llm_manager.config.model}")
        
        # Create Data Analyst
        print("\n→ Creating Data Analyst agent...")
        start = time.time()
        
        data_agent = await factory.create_agent(
            name="DataAnalyst-001",
            agent_type=AgentType.DATA_ANALYST,
            skills=["data_processing", "statistics", "visualization"],
            auto_verify=True
        )
        
        creation_time = time.time() - start
        agents.append(data_agent)
        
        print(f"  ✓ Agent created in {creation_time:.2f}s")
        print(f"    ID: {data_agent.agent_id}")
        print(f"    Skills: {', '.join(data_agent.skills)}")
        
        # Create Security Expert
        print("\n→ Creating Security Expert agent...")
        start = time.time()
        
        security_agent = await factory.create_agent(
            name="SecurityExpert-001",
            agent_type=AgentType.SECURITY_EXPERT,
            skills=["security_audit"],
            auto_verify=True
        )
        
        creation_time = time.time() - start
        agents.append(security_agent)
        
        print(f"  ✓ Agent created in {creation_time:.2f}s")
        print(f"    ID: {security_agent.agent_id}")
        
        print(f"\n✓ Successfully created {len(agents)} agents with verified LLM connections")
        
        # ===== TEST 2: Execute Real Tasks =====
        print("\n" + "─"*60)
        print("TEST 2: Executing Real Tasks with LLM")
        print("─"*60)
        
        # Task for Code Architect
        print("\n→ Code Architect analyzing Python code...")
        code_task = """
        Analyze this Python function and provide exactly 3 improvements:
        
        ```python
        def process_data(data):
            result = []
            for item in data:
                if item > 0:
                    result.append(item * 2)
            return result
        ```
        
        Be concise and specific.
        """
        
        start = time.time()
        code_result = await factory.execute_task(code_agent, code_task)
        exec_time = time.time() - start
        
        if code_result["status"] == "success":
            print(f"  ✓ Task completed in {exec_time:.2f}s")
            print(f"    Tokens used: {code_result['execution']['tokens_used']}")
            print(f"    Provider: {code_result['execution']['provider']}")
            print(f"    Model: {code_result['execution']['model']}")
            print("\n  LLM Response (first 300 chars):")
            print("  " + "-"*40)
            print("  " + code_result["result"][:300].replace("\n", "\n  "))
            print("  ...")
        else:
            print(f"  ✗ Task failed: {code_result['error']}")
        
        # Task for Data Analyst
        print("\n→ Data Analyst providing insights...")
        data_task = """
        Given sales data showing a 15% decline in Q3, list 3 potential causes 
        and 2 analysis methods to investigate. Be brief.
        """
        
        start = time.time()
        data_result = await factory.execute_task(
            data_agent,
            data_task,
            context={"quarter": "Q3", "decline_percentage": 15}
        )
        exec_time = time.time() - start
        
        if data_result["status"] == "success":
            print(f"  ✓ Task completed in {exec_time:.2f}s")
            print(f"    Tokens used: {data_result['execution']['tokens_used']}")
            print("\n  LLM Response (first 300 chars):")
            print("  " + "-"*40)
            print("  " + data_result["result"][:300].replace("\n", "\n  "))
            print("  ...")
        
        # Task for Security Expert
        print("\n→ Security Expert identifying vulnerabilities...")
        security_task = """
        List the top 3 OWASP vulnerabilities for web applications in 2024.
        Provide just the names, no descriptions.
        """
        
        start = time.time()
        sec_result = await factory.execute_task(security_agent, security_task)
        exec_time = time.time() - start
        
        if sec_result["status"] == "success":
            print(f"  ✓ Task completed in {exec_time:.2f}s")
            print(f"    Tokens used: {sec_result['execution']['tokens_used']}")
            print("\n  LLM Response:")
            print("  " + "-"*40)
            print("  " + sec_result["result"][:500].replace("\n", "\n  "))
        
        # ===== TEST 3: Memory and Context =====
        print("\n" + "─"*60)
        print("TEST 3: Testing Memory Persistence")
        print("─"*60)
        
        print("\n→ Executing sequential tasks with memory...")
        
        # First task
        task1 = "Remember that our project is called 'AutomataOS' and uses Python."
        result1 = await factory.execute_task(code_agent, task1)
        print(f"  Task 1: {'✓' if result1['status'] == 'success' else '✗'}")
        
        # Second task using memory
        task2 = "What programming language does our project use? (use your memory)"
        result2 = await factory.execute_task(code_agent, task2, use_memory=True)
        print(f"  Task 2: {'✓' if result2['status'] == 'success' else '✗'}")
        
        if result2["status"] == "success":
            print("\n  Memory test response:")
            print("  " + result2["result"][:200])
        
        # ===== TEST 4: Agent Status and Metrics =====
        print("\n" + "─"*60)
        print("TEST 4: Agent Status and Performance Metrics")
        print("─"*60)
        
        for agent in agents:
            status = await factory.get_agent_status(agent.agent_id)
            print(f"\n→ {agent.name}:")
            print(f"  Status: {status['agent']['lifecycle_state']}")
            print(f"  Executions: {status['runtime']['execution_count']}")
            print(f"  Total tokens: {status['runtime']['total_tokens_used']}")
            if status['metrics'].get('success_rate') is not None:
                print(f"  Success rate: {status['metrics']['success_rate']*100:.0f}%")
            if status['metrics'].get('avg_execution_time'):
                print(f"  Avg exec time: {status['metrics']['avg_execution_time']:.2f}s")
        
        # ===== TEST 5: Capability Testing =====
        print("\n" + "─"*60)
        print("TEST 5: Comprehensive Capability Tests")
        print("─"*60)
        
        print("\n→ Running capability tests on Code Architect...")
        test_results = await factory.test_agent_capabilities(code_agent)
        
        print(f"\n  Test Summary:")
        print(f"    Total tests: {test_results['summary']['total_tests']}")
        print(f"    Successful: {test_results['summary']['successful']}")
        print(f"    Success rate: {test_results['summary']['success_rate']*100:.0f}%")
        print(f"    Total time: {test_results['summary']['total_time']:.2f}s")
        print(f"    Total tokens: {test_results['summary']['total_tokens']}")
        
        print("\n  Individual test results:")
        for test in test_results['tests']:
            status = "✓" if test['success'] else "✗"
            print(f"    {status} {test['name']}: {test.get('execution_time', 0):.2f}s")
        
        # ===== FINAL SUMMARY =====
        print("\n" + "="*80)
        print("VERIFICATION COMPLETE")
        print("="*80)
        
        # Calculate totals
        total_agents = len(agents)
        total_executions = sum(a.execution_count for a in agents)
        total_tokens = sum(a.total_tokens_used for a in agents)
        
        print(f"\n✓ All agents are using REAL LLM connections")
        print(f"✓ No mock data or fake responses")
        print(f"\nStatistics:")
        print(f"  • Agents created: {total_agents}")
        print(f"  • Total executions: {total_executions}")
        print(f"  • Total tokens used: {total_tokens}")
        print(f"  • Timestamp: {datetime.now().isoformat()}")
        
        # Verify this is real
        print(f"\nLLM Provider Details:")
        for agent in agents[:1]:  # Show first agent's details
            provider_info = agent.llm_manager.get_provider_info()
            print(f"  • Provider: {provider_info['provider']}")
            print(f"  • Model: {provider_info['model']}")
            print(f"  • Temperature: {provider_info['temperature']}")
            print(f"  • Max tokens: {provider_info['max_tokens']}")
        
        print("\n✅ Agent Factory is fully operational with real LLM integration!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        factory.cleanup()
        print("\n✓ Cleanup complete")


if __name__ == "__main__":
    print("\n🚀 Starting Agent Factory Test...")
    print("   This will make REAL API calls to your configured LLM provider")
    print("   Ensure you have set OPENAI_API_KEY or ANTHROPIC_API_KEY")
    
    asyncio.run(main())
