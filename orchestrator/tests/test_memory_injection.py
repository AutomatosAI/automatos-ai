#!/usr/bin/env python3
"""
Test Memory Injection in Workflows
===================================

This script tests that memories are properly injected into agent prompts during workflow execution.
Part of Week 1 Day 3: Memory Integration testing.
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.memory_knowledge_system import HierarchicalMemorySystem
from core.memory_prompt_injector import MemoryPromptInjector
from core.workflow_memory_integrator import WorkflowMemoryIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MemoryInjectionTester:
    """Test memory injection into agent prompts"""
    
    def __init__(self):
        self.memory_system = None
        self.memory_injector = MemoryPromptInjector()
        self.test_results = []
    
    async def setup(self):
        """Initialize memory system"""
        try:
            self.memory_system = HierarchicalMemorySystem(
                redis_host=os.getenv("REDIS_HOST", "127.0.0.1"),
                redis_port=int(os.getenv("REDIS_PORT", 6379)),
                redis_password=os.getenv("REDIS_PASSWORD"),
                postgres_url=os.getenv("DATABASE_URL"),
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )
            logger.info("✅ Memory system initialized")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize memory system: {e}")
            return False
    
    async def test_memory_storage(self):
        """Test storing experiences in memory"""
        logger.info("\n📝 TEST 1: Memory Storage")
        logger.info("-" * 50)
        
        try:
            # Store test experience
            test_experience = {
                "task_id": f"test_{datetime.now().timestamp()}",
                "content": "Successfully analyzed Python code for security vulnerabilities",
                "success": True,
                "tokens_used": 1500,
                "execution_time_ms": 2000,
                "timestamp": datetime.now().isoformat()
            }
            
            # Store for test agent ID 999
            await self.memory_system.store_experience(
                agent_id=999,
                experience=test_experience
            )
            
            logger.info("✅ Experience stored successfully")
            self.test_results.append(("Memory Storage", True, "Experience stored"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory storage failed: {e}")
            self.test_results.append(("Memory Storage", False, str(e)))
            return False
    
    async def test_memory_retrieval(self):
        """Test retrieving memories"""
        logger.info("\n🔍 TEST 2: Memory Retrieval")
        logger.info("-" * 50)
        
        try:
            # Retrieve memories for test agent
            memories = await self.memory_system.retrieve_relevant_memories(
                agent_id=999,
                context="code security analysis",
                memory_types=None,
                top_k=5
            )
            
            logger.info(f"✅ Retrieved {len(memories)} memories")
            for i, mem in enumerate(memories, 1):
                logger.info(f"  Memory {i}: {mem.get('content', '')[:100]}...")
            
            self.test_results.append(("Memory Retrieval", True, f"Retrieved {len(memories)} memories"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Memory retrieval failed: {e}")
            self.test_results.append(("Memory Retrieval", False, str(e)))
            return False
    
    async def test_prompt_injection(self):
        """Test injecting memories into prompts"""
        logger.info("\n💉 TEST 3: Prompt Injection")
        logger.info("-" * 50)
        
        try:
            # Create mock memory retrieval results
            mock_memories = {
                "agent_memories": {
                    "999": {
                        "working_memory": [
                            {"content": "Recent code review found SQL injection vulnerability", "timestamp": datetime.now().isoformat()},
                        ],
                        "short_term": [
                            {"content": "Successfully fixed XSS vulnerability in user input", "success": True},
                            {"content": "Failed to detect race condition in async code", "success": False}
                        ],
                        "long_term": [
                            {"content": "Always validate user input before database queries", "confidence": 0.95}
                        ]
                    }
                },
                "collective_memories": [
                    {"content": "Team best practice: Use parameterized queries for all database operations", "source": "security team"}
                ]
            }
            
            # Original prompt
            original_prompt = "Review this Python code for security vulnerabilities"
            
            # Inject memories
            enhanced_prompt = self.memory_injector.inject_memory_into_prompt(
                original_prompt=original_prompt,
                agent_id=999,
                memory_retrieval_results=mock_memories,
                subtask_id="test_subtask_001"
            )
            
            # Check if memories were injected
            if len(enhanced_prompt) > len(original_prompt):
                logger.info("✅ Memories successfully injected into prompt")
                logger.info(f"Original length: {len(original_prompt)} chars")
                logger.info(f"Enhanced length: {len(enhanced_prompt)} chars")
                logger.info("\nEnhanced prompt preview:")
                logger.info("-" * 30)
                logger.info(enhanced_prompt[:500] + "...")
                logger.info("-" * 30)
                
                self.test_results.append(("Prompt Injection", True, f"Prompt enhanced from {len(original_prompt)} to {len(enhanced_prompt)} chars"))
                return True
            else:
                logger.error("❌ No memories were injected")
                self.test_results.append(("Prompt Injection", False, "No enhancement detected"))
                return False
                
        except Exception as e:
            logger.error(f"❌ Prompt injection failed: {e}")
            self.test_results.append(("Prompt Injection", False, str(e)))
            return False
    
    async def test_workflow_integration(self):
        """Test memory integration with workflow"""
        logger.info("\n🔄 TEST 4: Workflow Integration")
        logger.info("-" * 50)
        
        try:
            # Create memory integrator
            integrator = WorkflowMemoryIntegrator(self.memory_system)
            
            # Test memory retrieval for workflow
            memories = await integrator.retrieve_workflow_memories(
                workflow_id=1,
                workflow_description="Code security review workflow",
                agent_ids=[999]
            )
            
            logger.info(f"✅ Workflow memory retrieval successful")
            logger.info(f"  Total memories retrieved: {memories.get('total_memories_retrieved', 0)}")
            logger.info(f"  Agents with memories: {len(memories.get('agent_memories', {}))}")
            logger.info(f"  Collective memories: {len(memories.get('collective_memories', []))}")
            
            self.test_results.append(("Workflow Integration", True, "Memory integrator working"))
            return True
            
        except Exception as e:
            logger.error(f"❌ Workflow integration failed: {e}")
            self.test_results.append(("Workflow Integration", False, str(e)))
            return False
    
    async def run_all_tests(self):
        """Run all memory injection tests"""
        logger.info("=" * 60)
        logger.info("MEMORY INJECTION TEST SUITE")
        logger.info("=" * 60)
        
        # Setup
        if not await self.setup():
            logger.error("Setup failed, aborting tests")
            return
        
        # Run tests
        await self.test_memory_storage()
        await asyncio.sleep(1)  # Give time for storage
        
        await self.test_memory_retrieval()
        await self.test_prompt_injection()
        await self.test_workflow_integration()
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        passed = sum(1 for _, success, _ in self.test_results if success)
        failed = sum(1 for _, success, _ in self.test_results if not success)
        
        for test_name, success, message in self.test_results:
            status = "✅ PASS" if success else "❌ FAIL"
            logger.info(f"{status} - {test_name}: {message}")
        
        logger.info(f"\nTotal: {len(self.test_results)} tests")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {passed/len(self.test_results)*100:.1f}%")
        
        return failed == 0


async def main():
    """Main test runner"""
    tester = MemoryInjectionTester()
    success = await tester.run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
