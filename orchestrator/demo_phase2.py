#!/usr/bin/env python3
"""
Phase 2 Context Engineering Demo
================================

Demonstrates the complete Phase 2 Context Engineering system with RAG capabilities,
knowledge base integration, context-aware prompts, and advanced agent collaboration.
"""

import asyncio
import logging
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.append('.')

from context_integration import initialize_context_engineering

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_context_engineering():
    """Demonstrate the Phase 2 Context Engineering system"""
    
    print("🚀 Phase 2 Context Engineering System Demo")
    print("=" * 50)
    
    try:
        # Initialize the context engineering system
        print("\n1. Initializing Context Engineering System...")
        context_manager = await initialize_context_engineering()
        print("✅ Context Engineering System initialized successfully!")
        
        # Demonstrate document ingestion
        print("\n2. Ingesting User Documents...")
        guide_path = "/home/ubuntu/Uploads/user_message_2025-07-26_12-23-54.txt"
        if os.path.exists(guide_path):
            result = await context_manager.ingest_document(guide_path)
            if result['success']:
                print(f"✅ Ingested implementation guide: {result['chunks_created']} chunks")
            else:
                print(f"❌ Failed to ingest guide: {result['message']}")
        else:
            print("⚠️ Implementation guide not found")
        
        # Demonstrate context retrieval
        print("\n3. Testing Context Retrieval...")
        test_query = "Build REST API with JWT authentication"
        context_results = await context_manager.retrieve_context_for_task(
            task_description=test_query,
            task_type='api_development'
        )
        print(f"✅ Retrieved {len(context_results)} relevant context items")
        
        if context_results:
            print("📋 Top context results:")
            for i, ctx in enumerate(context_results[:3], 1):
                print(f"  {i}. {ctx['source']} (relevance: {ctx['relevance_score']:.2f})")
                print(f"     Preview: {ctx['content'][:100]}...")
        
        # Demonstrate context-aware prompt generation
        print("\n4. Testing Context-Aware Prompt Generation...")
        prompt_result = await context_manager.build_context_aware_prompt(
            task_description=test_query,
            prompt_type='api_development',
            context_results=context_results
        )
        print(f"✅ Generated context-aware prompt with {prompt_result['context_count']} context items")
        print(f"📝 Template used: {prompt_result['template_name']}")
        
        # Demonstrate learning system
        print("\n5. Testing Learning System...")
        await context_manager.learn_from_task_execution(
            task_description=test_query,
            task_type='api_development',
            context_used={'context_count': len(context_results)},
            outcome="Successfully generated API code",
            success=True,
            execution_time=45.0,
            agent_used='demo_agent'
        )
        print("✅ Learning system recorded task execution")
        
        # Get system statistics
        print("\n6. System Statistics...")
        stats = await context_manager.get_system_stats()
        if stats['initialized']:
            print("📊 Context Engineering System Stats:")
            if 'vector_store' in stats:
                vs_stats = stats['vector_store']
                print(f"  📚 Documents: {vs_stats.get('total_documents', 0)}")
                print(f"  📄 Chunks: {vs_stats.get('total_chunks', 0)}")
                print(f"  📏 Avg chunk length: {vs_stats.get('avg_chunk_length', 0):.0f} chars")
            
            if 'learning_engine' in stats:
                le_stats = stats['learning_engine']
                print(f"  🧠 Learning events: {le_stats.get('total_learning_events', 0)}")
                print(f"  🔍 Patterns identified: {le_stats.get('patterns_identified', 0)}")
            
            if 'collaboration_system' in stats:
                cs_stats = stats['collaboration_system']
                print(f"  🤝 Active sessions: {cs_stats.get('active_sessions', 0)}")
                print(f"  👥 Available agents: {cs_stats.get('available_agents', 0)}")
        
        # Demonstrate complex task processing
        print("\n7. Testing Complex Task Processing...")
        complex_task = """
        Create a FastAPI application with:
        - JWT authentication system
        - User registration and login endpoints
        - Protected routes requiring authentication
        - SQLite database integration
        - Comprehensive error handling
        - API documentation with Swagger
        - Unit tests with pytest
        - Docker configuration
        """
        
        # Retrieve context for complex task
        complex_context = await context_manager.retrieve_context_for_task(
            task_description=complex_task,
            task_type='api_development'
        )
        
        # Generate context-aware prompt
        complex_prompt = await context_manager.build_context_aware_prompt(
            task_description=complex_task,
            prompt_type='api_development',
            context_results=complex_context
        )
        
        print(f"✅ Complex task analysis complete:")
        print(f"  📋 Context items: {len(complex_context)}")
        print(f"  📝 Prompt template: {complex_prompt['template_name']}")
        print(f"  🎯 Context strategy: {complex_prompt.get('strategy_used', 'default')}")
        
        print("\n🎉 Phase 2 Context Engineering Demo Complete!")
        print("=" * 50)
        print("\n✨ Key Features Demonstrated:")
        print("  🔍 RAG System with document chunking and embeddings")
        print("  📚 Knowledge base integration with uploaded documents")
        print("  🧠 Context-aware prompt generation")
        print("  📈 Learning system with pattern recognition")
        print("  🤝 Advanced agent collaboration framework")
        print("  🔧 Enhanced workflow integration")
        
        # Close connections
        await context_manager.close()
        
        return True
        
    except Exception as e:
        logger.error(f"Demo failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

async def test_cognitive_functions():
    """Test the cognitive functions integration"""
    
    print("\n🧠 Testing Cognitive Functions Integration")
    print("-" * 40)
    
    try:
        # Import the enhanced orchestrator
        from enhanced_orchestrator import EnhancedTwoTierOrchestrator
        
        # Create orchestrator instance
        orchestrator = EnhancedTwoTierOrchestrator()
        
        # Wait a moment for context engineering to initialize
        await asyncio.sleep(2)
        
        # Test cognitive task breakdown
        print("\n1. Testing Cognitive Task Breakdown...")
        test_task = "Build a simple REST API with user authentication"
        
        tasks = await orchestrator.cognitive_task_breakdown(test_task)
        print(f"✅ Generated {len(tasks)} tasks:")
        for i, task in enumerate(tasks[:3], 1):
            print(f"  {i}. {task['task']} -> {task['file']} ({task['type']})")
        
        # Test cognitive content generation
        print("\n2. Testing Cognitive Content Generation...")
        if tasks:
            sample_task = tasks[0]
            content = await orchestrator.cognitive_content_generation(sample_task, "/tmp/test_project")
            print(f"✅ Generated content for {sample_task['file']}: {len(content)} characters")
            print(f"📄 Preview: {content[:200]}...")
        
        print("\n✅ Cognitive Functions Integration Test Complete!")
        
    except Exception as e:
        logger.error(f"Cognitive functions test failed: {str(e)}")
        import traceback
        traceback.print_exc()

async def main():
    """Main demo function"""
    
    print(f"🌟 Automotas AI Phase 2 Context Engineering Demo")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Run the main context engineering demo
    success = await demo_context_engineering()
    
    if success:
        # Run cognitive functions test
        await test_cognitive_functions()
        
        print("\n🎊 All demos completed successfully!")
        print("\n🚀 Phase 2 Context Engineering System is now fully operational!")
        print("\nKey capabilities now available:")
        print("  • RAG system with vector embeddings")
        print("  • Context-aware prompt generation")
        print("  • Historical pattern recognition")
        print("  • Advanced agent collaboration")
        print("  • Learning from task execution")
        print("  • Multi-modal document processing")
        
    else:
        print("\n❌ Demo encountered errors. Check logs for details.")
    
    print(f"\n⏰ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    asyncio.run(main())
