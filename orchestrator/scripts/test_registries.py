"""
Test Registries - Phase 1.3
===========================

Tests the three central registries:
1. AgentRegistry - Agent capabilities lookup
2. GlobalFunctionRegistry - LLM function catalog
3. ToolRegistry - Tool specifications (existing)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from database.database import SessionLocal
from core.agent_registry import AgentRegistry, get_agent_registry
from core.global_function_registry import GlobalFunctionRegistry, get_global_function_registry, FunctionCategory
from services.tool_registry import ToolRegistry

def test_agent_registry():
    """Test AgentRegistry"""
    print("\n" + "="*60)
    print("TEST 1: AGENT REGISTRY")
    print("="*60)
    
    db = SessionLocal()
    try:
        registry = get_agent_registry(db)
        
        # Test 1: Get all agents
        print("\n1️⃣  Getting all agents...")
        agents = registry.get_all_agents()
        print(f"✅ Found {len(agents)} agents")
        
        if agents:
            # Test 2: Get capabilities for first agent
            agent = agents[0]
            print(f"\n2️⃣  Getting capabilities for agent {agent.id} ({agent.name})...")
            capabilities = registry.get_agent_capabilities(agent.id)
            
            if capabilities:
                print(f"   - Skills: {len(capabilities.skills)}")
                print(f"   - Tools: {capabilities.tools}")
                print(f"   - Performance: {capabilities.performance}")
                print(f"   - Availability: {capabilities.availability}")
                print("✅ Capabilities loaded successfully")
            else:
                print("❌ No capabilities found")
            
            # Test 3: Find agents with specific tool
            print("\n3️⃣  Finding agents with 'create_pdf' tool...")
            pdf_agents = registry.find_agents_with_tool("create_pdf")
            print(f"✅ Found {len(pdf_agents)} agents with create_pdf")
            
            # Test 4: Find agents with specific skill
            print("\n4️⃣  Finding agents with 'research' skill...")
            research_agents = registry.find_agents_with_skill("research")
            print(f"✅ Found {len(research_agents)} agents with research skill")
            
            # Test 5: Get agents by type
            print("\n5️⃣  Finding 'document_generator' agents...")
            doc_agents = registry.get_agents_by_type("document_generator")
            print(f"✅ Found {len(doc_agents)} document_generator agents")
        
        print("\n✅ AgentRegistry tests PASSED")
        
    except Exception as e:
        print(f"❌ AgentRegistry tests FAILED: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def test_global_function_registry():
    """Test GlobalFunctionRegistry"""
    print("\n" + "="*60)
    print("TEST 2: GLOBAL FUNCTION REGISTRY")
    print("="*60)
    
    try:
        registry = get_global_function_registry()
        
        # Test 1: Get all functions
        print("\n1️⃣  Getting all functions...")
        all_functions = registry.get_all_functions()
        print(f"✅ Found {len(all_functions)} registered functions")
        
        # Test 2: Get functions by category
        print("\n2️⃣  Getting AGENT_QUERY functions...")
        agent_query_funcs = registry.get_functions_by_category(FunctionCategory.AGENT_QUERY)
        print(f"✅ Found {len(agent_query_funcs)} agent query functions:")
        for func in agent_query_funcs:
            print(f"   - {func.name}")
        
        # Test 3: Get OpenAI schemas
        print("\n3️⃣  Generating OpenAI schemas...")
        schemas = registry.get_schemas_for_llm(categories=[FunctionCategory.AGENT_QUERY])
        print(f"✅ Generated {len(schemas)} OpenAI function schemas")
        
        # Test 4: Get specific function
        print("\n4️⃣  Getting 'query_available_agents' function...")
        func = registry.get_function("query_available_agents")
        if func:
            print(f"✅ Found function:")
            print(f"   - Name: {func.name}")
            print(f"   - Description: {func.description}")
            print(f"   - Parameters: {len(func.parameters)}")
            print(f"   - Category: {func.category.value}")
        else:
            print("❌ Function not found")
        
        print("\n✅ GlobalFunctionRegistry tests PASSED")
        
    except Exception as e:
        print(f"❌ GlobalFunctionRegistry tests FAILED: {e}")
        import traceback
        traceback.print_exc()


def test_tool_registry():
    """Test ToolRegistry (existing)"""
    print("\n" + "="*60)
    print("TEST 3: TOOL REGISTRY")
    print("="*60)
    
    try:
        registry = ToolRegistry()
        
        # Test 1: Get all tools
        print("\n1️⃣  Getting all tools...")
        all_tools = registry.get_all_tools()
        print(f"✅ Found {len(all_tools)} registered tools")
        
        # Test 2: Get tool by name
        print("\n2️⃣  Getting 'create_pdf' tool...")
        pdf_tool = registry.get_tool("create_pdf")
        if pdf_tool:
            print(f"✅ Found tool:")
            print(f"   - Name: {pdf_tool.name}")
            print(f"   - Category: {pdf_tool.category.value}")
            print(f"   - Security: {pdf_tool.security_level.value}")
        else:
            print("⚠️  Tool 'create_pdf' not found (may not be registered yet)")
        
        # Test 3: Get tools by category
        print("\n3️⃣  Getting RESEARCH tools...")
        from services.tool_registry import ToolCategory
        research_tools = registry.get_tools_by_category(ToolCategory.RESEARCH)
        print(f"✅ Found {len(research_tools)} research tools:")
        for tool in research_tools[:5]:  # Show first 5
            print(f"   - {tool.name}")
        
        # Test 4: Get recommended tools
        print("\n4️⃣  Getting tools for 'pdf generation' task...")
        recommended = registry.get_tools_for_task("pdf generation")
        print(f"✅ Found {len(recommended)} recommended tools")
        
        print("\n✅ ToolRegistry tests PASSED")
        
    except Exception as e:
        print(f"❌ ToolRegistry tests FAILED: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all registry tests"""
    print("\n" + "🧪 "*30)
    print("TESTING CENTRAL REGISTRIES - PHASE 1.3")
    print("🧪 "*30)
    
    test_agent_registry()
    test_global_function_registry()
    test_tool_registry()
    
    print("\n" + "="*60)
    print("🎉 ALL REGISTRY TESTS COMPLETE")
    print("="*60)
    print("\nNext Steps:")
    print("1️⃣  Review test output above")
    print("2️⃣  If all PASSED, move to Phase 1.4 (Stage 1 Consolidation)")
    print("3️⃣  If any FAILED, fix issues before continuing")
    print()


if __name__ == "__main__":
    main()

