"""
Memory System Integration Module
=================================

Integrates the hierarchical memory system with existing services.
Provides hooks and patches for AgentFactory and SharedContextManager.
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from .memory_knowledge_system import (
    HierarchicalMemorySystem,
    KnowledgeGraph,
    LearningEngine
)

logger = logging.getLogger(__name__)


class MemoryEnabledAgentFactory:
    """
    Enhanced AgentFactory with integrated memory system.
    Wraps the existing AgentFactory to add memory capabilities.
    """
    
    def __init__(
        self,
        agent_factory,
        memory_system: HierarchicalMemorySystem,
        knowledge_graph: KnowledgeGraph,
        learning_engine: LearningEngine
    ):
        self.agent_factory = agent_factory
        self.memory_system = memory_system
        self.knowledge_graph = knowledge_graph
        self.learning_engine = learning_engine
        
        # Store original methods
        self._original_execute = agent_factory.execute_with_prompt
        self._original_create = agent_factory.create_agent
        
        # Patch methods
        self._patch_methods()
        
        logger.info("MemoryEnabledAgentFactory initialized")
    
    def _patch_methods(self):
        """Patch AgentFactory methods to add memory capabilities"""
        self.agent_factory.execute_with_prompt = self.execute_with_memory
        self.agent_factory.create_agent = self.create_agent_with_memory
    
    async def execute_with_memory(
        self,
        agent,
        prompt: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Enhanced execution that uses memory for context and stores experiences.
        """
        agent_id = agent.id if hasattr(agent, 'id') else agent
        start_time = datetime.now()
        
        # Step 1: Retrieve relevant memories for context
        relevant_memories = await self.memory_system.retrieve_relevant_memories(
            agent_id,
            prompt,
            top_k=5
        )
        
        # Step 2: Enhance prompt with memory context
        enhanced_prompt = self._enhance_prompt_with_memories(prompt, relevant_memories)
        
        # Log memory usage
        if relevant_memories:
            logger.info(f"Agent {agent_id}: Using {len(relevant_memories)} relevant memories")
        
        # Step 3: Execute with enhanced prompt
        try:
            result = await self._original_execute(agent, enhanced_prompt, **kwargs)
            success = True
            error = None
        except Exception as e:
            logger.error(f"Execution failed: {e}")
            result = {"error": str(e)}
            success = False
            error = str(e)
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Step 4: Store experience in memory
        experience = {
            "prompt": prompt,
            "enhanced_prompt": enhanced_prompt,
            "result": result,
            "success": success,
            "error_occurred": error is not None,
            "error": error,
            "execution_time": execution_time,
            "tokens_used": result.get("tokens_used", 0) if isinstance(result, dict) else 0,
            "timestamp": datetime.now().isoformat(),
            "used_memories": len(relevant_memories)
        }
        
        # Calculate importance based on execution
        if error:
            experience["importance"] = 0.8  # Failures are important to learn from
        elif execution_time > 5:
            experience["importance"] = 0.7  # Long executions are notable
        else:
            experience["importance"] = 0.5
        
        memory_id = await self.memory_system.store_experience(agent_id, experience)
        
        # Step 5: Learn from feedback if available
        if kwargs.get("feedback"):
            learning_result = await self.learning_engine.learn_from_feedback(
                agent_id,
                kwargs.get("task_type", "general"),
                result,
                kwargs["feedback"]
            )
            logger.info(f"Agent {agent_id} learned: {learning_result['learned_pattern'][:50]}...")
        
        # Step 6: Update result with memory info
        if isinstance(result, dict):
            result["memory_id"] = memory_id
            result["memories_used"] = len(relevant_memories)
        
        return result
    
    def _enhance_prompt_with_memories(
        self,
        prompt: str,
        memories: list
    ) -> str:
        """Enhance prompt with relevant memory context"""
        if not memories:
            return prompt
        
        memory_context = "\n\n[Relevant Context from Memory]:\n"
        
        for i, memory in enumerate(memories[:3], 1):  # Use top 3 memories
            content = memory.get("content", {})
            
            # Extract useful information from memory
            if isinstance(content, dict):
                # Handle different memory content structures
                if "result" in content:
                    result = content["result"]
                    if isinstance(result, dict) and "output" in result:
                        memory_text = f"{i}. Previous result: {result['output'][:100]}..."
                    else:
                        memory_text = f"{i}. Previous experience: {str(result)[:100]}..."
                elif "learned" in content:
                    memory_text = f"{i}. Learned pattern: {content['learned']}"
                elif "content" in content:
                    memory_text = f"{i}. Context: {content['content'][:100]}..."
                else:
                    memory_text = f"{i}. Memory: {str(content)[:100]}..."
            else:
                memory_text = f"{i}. {str(content)[:100]}..."
            
            memory_context += f"{memory_text}\n"
            
            # Add relevance info
            if "relevance" in memory:
                memory_context += f"   (Relevance: {memory['relevance']:.2f})\n"
        
        memory_context += "\n[Current Task]:\n"
        
        return memory_context + prompt
    
    async def create_agent_with_memory(self, agent_config: Dict[str, Any]) -> Any:
        """Create agent with memory initialization"""
        # Create agent using original method
        agent = await self._original_create(agent_config)
        
        # Initialize memory for new agent
        if hasattr(agent, 'id'):
            initial_experience = {
                "event": "agent_created",
                "config": agent_config,
                "timestamp": datetime.now().isoformat(),
                "importance": 0.5
            }
            await self.memory_system.store_experience(agent.id, initial_experience)
            
            logger.info(f"Initialized memory for new agent {agent.id}")
        
        return agent
    
    async def share_team_knowledge(
        self,
        team_agents: list,
        knowledge: Dict[str, Any]
    ):
        """Share knowledge among team members"""
        for agent_id in team_agents:
            # Store as collective memory
            experience = {
                "source": "team_knowledge",
                "shared_by": team_agents,
                "knowledge": knowledge,
                "is_collective": True,
                "importance": 0.7
            }
            await self.memory_system.store_experience(agent_id, experience)
        
        logger.info(f"Shared knowledge among {len(team_agents)} team members")
    
    async def consolidate_team_learning(self, team_agents: list):
        """Consolidate learning from all team members"""
        consolidated_patterns = []
        
        for agent_id in team_agents:
            # Get high-value memories from each agent
            memories = await self.memory_system.retrieve_relevant_memories(
                agent_id,
                "successful patterns learned insights",
                memory_types=["pattern", "knowledge"],
                top_k=10
            )
            consolidated_patterns.extend(memories)
        
        # Share consolidated knowledge with all team members
        if consolidated_patterns:
            knowledge = {
                "type": "consolidated_team_learning",
                "patterns": consolidated_patterns,
                "timestamp": datetime.now().isoformat()
            }
            await self.share_team_knowledge(team_agents, knowledge)
            
            logger.info(f"Consolidated {len(consolidated_patterns)} patterns from team")
        
        return consolidated_patterns


class MemorySystemInitializer:
    """Initialize and configure the memory system for the platform"""
    
    @staticmethod
    async def initialize(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize all memory system components.
        
        Returns dict with:
        - memory_system: HierarchicalMemorySystem instance
        - knowledge_graph: KnowledgeGraph instance
        - learning_engine: LearningEngine instance
        - success: bool
        """
        try:
            # Create memory system using centralized clients
            # No hardcoded defaults - will use centralized get_redis_client()
            memory_system = HierarchicalMemorySystem(
                openai_api_key=config.get("OPENAI_API_KEY")
                # postgres_url and redis_client will use centralized defaults
            )
            
            # Initialize database
            await memory_system.initialize_database()
            
            # Create knowledge graph
            knowledge_graph = KnowledgeGraph(memory_system)
            
            # Create learning engine
            learning_engine = LearningEngine(memory_system)
            
            logger.info("✅ Memory system initialized successfully")
            
            return {
                "memory_system": memory_system,
                "knowledge_graph": knowledge_graph,
                "learning_engine": learning_engine,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            return {
                "memory_system": None,
                "knowledge_graph": None,
                "learning_engine": None,
                "success": False,
                "error": str(e)
            }
    
    @staticmethod
    async def integrate_with_platform(
        memory_components: Dict[str, Any],
        agent_factory,
        shared_context_manager
    ) -> bool:
        """
        Integrate memory system with existing platform components.
        """
        if not memory_components["success"]:
            logger.error("Cannot integrate - memory system not initialized")
            return False
        
        try:
            # Create enhanced agent factory
            memory_enabled_factory = MemoryEnabledAgentFactory(
                agent_factory,
                memory_components["memory_system"],
                memory_components["knowledge_graph"],
                memory_components["learning_engine"]
            )
            
            # Patch shared context manager
            if shared_context_manager:
                original_store = shared_context_manager.store_context
                
                async def store_with_memory(team_id, context_data):
                    # Store in original system
                    result = await original_store(team_id, context_data)
                    
                    # Also store in collective memory
                    if "agent_ids" in context_data:
                        for agent_id in context_data["agent_ids"]:
                            experience = {
                                "source": "shared_context",
                                "team_id": team_id,
                                "context": context_data,
                                "is_collective": True,
                                "importance": 0.6
                            }
                            await memory_components["memory_system"].store_experience(
                                agent_id,
                                experience
                            )
                    
                    return result
                
                shared_context_manager.store_context = store_with_memory
            
            logger.info("✅ Memory system integrated with platform")
            return True
            
        except Exception as e:
            logger.error(f"Integration failed: {e}")
            return False


# Convenience function for quick setup
async def setup_memory_system(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    One-line setup for memory system.
    
    Usage:
        from orchestrator.services.memory_integration import setup_memory_system
        memory = await setup_memory_system(config)
    """
    return await MemorySystemInitializer.initialize(config)
