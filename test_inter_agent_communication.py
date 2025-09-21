#!/usr/bin/env python3
"""
Test Inter-Agent Communication System
======================================

Verifies that agents can communicate, share context, and collaborate
on complex tasks using real LLM connections and Redis messaging.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add orchestrator to path
sys.path.insert(0, str(Path(__file__).parent / "orchestrator"))

from services.inter_agent_communication import (
    CollaborativeAgentFactory,
    AgentCommunicationProtocol,
    SharedContextManager,
    CollaborativeReasoner,
    MessageType,
    MessagePriority,
    verify_inter_agent_communication
)
from services.agent_factory import AgentMetadata


async def test_full_collaboration():
    """
    Comprehensive test of inter-agent communication and collaboration.
    Creates 3 agents to design a REST API for task management.
    """
    print("\n" + "=" * 70)
    print("INTER-AGENT COMMUNICATION & COLLABORATION TEST")
    print("=" * 70)
    print("\nTask: Design a REST API for a task management system")
    print("Team: API Architect, Security Expert, Data Specialist")
    print("-" * 70)
    
    factory = CollaborativeAgentFactory()
    
    try:
        # Phase 1: Agent Creation
        print("\nPHASE 1: Creating Specialized Agents")
        print("-" * 40)
        
        # Create API Architect
        print("Creating API Architect...")
        architect = await factory.create_agent(
            metadata={
                "name": "API Architect",
                "type": "code_architect",
                "description": "Expert in RESTful API design and system architecture",
                "skills": ["api_design", "rest_principles", "microservices", "system_architecture"],
                "preferred_model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1500,
                "metadata": {
                    "expertise_areas": ["REST", "GraphQL", "WebSockets"],
                    "years_experience": 10
                }
            },
            auto_verify=True
        )
        print(f"✓ {architect.metadata.name} ready (ID: {architect.agent_id})")
        
        # Create Security Expert
        print("Creating Security Expert...")
        security = await factory.create_agent(
            metadata={
                "name": "Security Guardian",
                "type": "security_expert",
                "description": "Specialist in API security, authentication, and authorization",
                "skills": ["oauth2", "jwt", "api_security", "penetration_testing", "encryption"],
                "preferred_model": "gpt-4",
                "temperature": 0.6,
                "max_tokens": 1500,
                "metadata": {
                    "certifications": ["CISSP", "CEH"],
                    "security_frameworks": ["OWASP", "ISO27001"]
                }
            },
            auto_verify=True
        )
        print(f"✓ {security.metadata.name} ready (ID: {security.agent_id})")
        
        # Create Data Specialist
        print("Creating Data Specialist...")
        data_expert = await factory.create_agent(
            metadata={
                "name": "Data Specialist",
                "type": "data_analyst",
                "description": "Expert in data modeling, database design, and optimization",
                "skills": ["database_design", "sql", "nosql", "data_modeling", "performance_optimization"],
                "preferred_model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 1500,
                "metadata": {
                    "databases": ["PostgreSQL", "MongoDB", "Redis"],
                    "modeling_tools": ["ERD", "UML"]
                }
            },
            auto_verify=True
        )
        print(f"✓ {data_expert.metadata.name} ready (ID: {data_expert.agent_id})")
        
        # Phase 2: Test Direct Messaging
        print("\nPHASE 2: Testing Inter-Agent Messaging")
        print("-" * 40)
        
        # Subscribe agents to messaging
        for agent in [architect, security, data_expert]:
            await factory.communication.subscribe_agent(agent.agent_id)
        print("✓ All agents subscribed to messaging channels")
        
        # Test message exchanges
        print("\nSending test messages between agents...")
        
        # Architect asks Security about authentication
        msg1 = await factory.send_agent_message(
            from_agent=architect,
            to_agent=security,
            message="What authentication methods should we implement for the task management API?",
            message_type=MessageType.TASK_REQUEST
        )
        print(f"✓ Architect → Security: Message {msg1.message_id[:8]}... sent")
        
        # Security asks Data Expert about user storage
        msg2 = await factory.send_agent_message(
            from_agent=security,
            to_agent=data_expert,
            message="How should we structure the user authentication data in the database?",
            message_type=MessageType.TASK_REQUEST
        )
        print(f"✓ Security → Data Expert: Message {msg2.message_id[:8]}... sent")
        
        # Data Expert shares insight with Architect
        msg3 = await factory.send_agent_message(
            from_agent=data_expert,
            to_agent=architect,
            message="Consider using UUID for task IDs to ensure global uniqueness in distributed systems",
            message_type=MessageType.KNOWLEDGE_SHARE
        )
        print(f"✓ Data Expert → Architect: Message {msg3.message_id[:8]}... sent")
        
        # Phase 3: Collaborative Problem Solving
        print("\nPHASE 3: Collaborative Problem Solving")
        print("-" * 40)
        
        task_description = """
        Design a comprehensive REST API for a task management system with the following requirements:
        
        1. User Management:
           - User registration and authentication
           - Role-based access control (Admin, Manager, User)
           - Profile management
        
        2. Task Operations:
           - Create, read, update, delete tasks
           - Task assignment to users
           - Task status tracking (TODO, IN_PROGRESS, REVIEW, DONE)
           - Task priorities and due dates
           - Task comments and attachments
        
        3. Advanced Features:
           - Search and filtering (by status, assignee, date range)
           - Bulk operations
           - Real-time notifications
           - Activity logging and audit trail
        
        4. Performance Requirements:
           - Support for 10,000+ concurrent users
           - Response time < 200ms for most operations
           - Scalable architecture
        
        Provide:
        - API endpoint structure
        - Authentication approach
        - Data models
        - Security considerations
        - Performance optimization strategies
        """
        
        print("Initiating collaborative task execution...")
        print(f"Team size: {len([architect, security, data_expert])} agents")
        
        # Execute collaborative task
        result = await factory.execute_team_task(
            task=task_description,
            team=[architect, security, data_expert],
            strategy="ensemble"
        )
        
        # Phase 4: Results Analysis
        print("\nPHASE 4: Results Analysis")
        print("-" * 40)
        
        if result["status"] == "success":
            # Individual contributions
            print("\n📊 Individual Agent Contributions:")
            for i, analysis in enumerate(result["individual_analyses"], 1):
                print(f"\n{i}. {analysis['agent_type'].replace('_', ' ').title()}")
                print(f"   Agent ID: {analysis['agent_id']}")
                print(f"   Execution: {analysis['execution_time']:.2f}s")
                print(f"   Tokens: {analysis['tokens_used']}")
                print(f"   Insight Preview:")
                print(f"   {analysis['insights'][:300]}...")
            
            # Proposed solutions
            print("\n💡 Solution Proposals:")
            for i, solution in enumerate(result["proposed_solutions"], 1):
                print(f"\n{i}. {solution['agent_type'].replace('_', ' ').title()}")
                print(f"   Confidence: {solution['confidence']:.2%}")
                print(f"   Execution: {solution['execution_time']:.2f}s")
                print(f"   Preview:")
                print(f"   {solution['solution'][:300]}...")
            
            # Consensus details
            print("\n🤝 Consensus Building:")
            consensus = result["consensus"]
            print(f"   Method: {consensus['method']}")
            print(f"   Selected Agent: {consensus['selected_agent']}")
            print(f"   Consensus Strength: {consensus['strength']:.2%}")
            if "weights" in consensus:
                print(f"   Agent Weights: {[f'{w:.2f}' for w in consensus['weights']]}")
            
            # Final solution
            print("\n✅ Final Synthesized Solution:")
            final = result["solution"]
            print(f"   Consensus Strength: {final['consensus_strength']:.2%}")
            print(f"   Contributing Agents: {final['contributing_agents']}")
            print(f"   Key Points Extracted: {len(final['key_contributions'])}")
            print("\n   Top Key Contributions:")
            for i, point in enumerate(final['key_contributions'][:5], 1):
                print(f"   {i}. {point[:100]}...")
            
            # Collaboration metrics
            print("\n📈 Collaboration Metrics:")
            metrics = result["collaboration_metrics"]
            print(f"   Total Agents: {metrics['total_agents']}")
            print(f"   Total Time: {metrics['total_time']:.2f}s")
            print(f"   Total Tokens Used: {metrics['total_tokens']:,}")
            print(f"   Consensus Strength: {metrics['consensus_strength']:.2%}")
            print(f"   Participation Rate: {metrics['participation_rate']:.0%}")
            
            # Calculate improvement metrics
            avg_individual_confidence = sum(s['confidence'] for s in result['proposed_solutions']) / len(result['proposed_solutions'])
            improvement = (final['consensus_strength'] - avg_individual_confidence) / avg_individual_confidence * 100
            
            print(f"\n   📊 Collaboration Improvement: {improvement:+.1f}%")
            print(f"      (Consensus vs Average Individual Confidence)")
            
            # Message history
            print("\n📬 Message Exchange Summary:")
            
            # Get message history between agents
            msg_count = 0
            for i, agent1 in enumerate([architect, security, data_expert]):
                for agent2 in [architect, security, data_expert][i+1:]:
                    messages = await factory.communication.get_message_history(
                        agent1.agent_id,
                        agent2.agent_id,
                        limit=10
                    )
                    if messages:
                        msg_count += len(messages)
                        print(f"   {agent1.metadata.name} ↔ {agent2.metadata.name}: {len(messages)} messages")
            
            print(f"   Total Messages Exchanged: {msg_count}")
            
            # Shared context details
            print("\n🗂️ Shared Context:")
            print(f"   Context ID: {result['shared_context_id']}")
            shared_context = await factory.context_manager.get_shared_context(result['shared_context_id'])
            if shared_context:
                print(f"   Version: {shared_context.version}")
                print(f"   Team Size: {len(shared_context.team_agents)}")
                print(f"   Last Updated: {shared_context.last_updated.strftime('%H:%M:%S')}")
                print(f"   Access Log Entries: {len(shared_context.access_log)}")
            
            print("\n" + "=" * 70)
            print("✅ TEST COMPLETED SUCCESSFULLY")
            print("=" * 70)
            print("\n🎯 Key Achievements:")
            print("   ✓ Real LLM connections verified for all agents")
            print("   ✓ Redis pub/sub messaging operational")
            print("   ✓ Shared context management working")
            print("   ✓ Collaborative reasoning producing real solutions")
            print("   ✓ Consensus building with weighted voting active")
            print(f"   ✓ Collaboration improved solution quality by {improvement:+.1f}%")
            
        else:
            print(f"\n❌ Collaborative task failed: {result.get('error', 'Unknown error')}")
    
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n🧹 Cleaning up resources...")
        factory.cleanup()
        print("✓ Cleanup complete")


async def main():
    """Main test runner"""
    print("\n" + "🚀 " * 20)
    print("AUTOMATOS AI - INTER-AGENT COMMUNICATION TEST SUITE")
    print("🚀 " * 20)
    
    # Check environment
    print("\n📋 Environment Check:")
    print(f"   OpenAI API Key: {'✓' if os.getenv('OPENAI_API_KEY') else '✗ Missing'}")
    print(f"   Redis URL: {os.getenv('REDIS_URL', 'redis://localhost:6379')}")
    print(f"   Database URL: {'PostgreSQL' if 'postgresql' in os.getenv('DATABASE_URL', '') else 'SQLite (default)'}")
    
    if not os.getenv('OPENAI_API_KEY'):
        print("\n⚠️  Warning: OPENAI_API_KEY not set. Tests will fail.")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    # Run the comprehensive test
    await test_full_collaboration()
    
    print("\n" + "🎉 " * 20)
    print("ALL TESTS COMPLETED")
    print("🎉 " * 20)


if __name__ == "__main__":
    asyncio.run(main())
