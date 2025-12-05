"""
Memory Prompt Injector for Workflow Execution
==============================================

This module properly injects retrieved memories into agent prompts during workflow execution.
Part of Week 1 Day 3: Memory Integration fixes.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class MemoryPromptInjector:
    """
    Injects retrieved memories into agent prompts during workflow execution.
    
    This class bridges the gap between memory retrieval and actual prompt execution,
    ensuring that agents receive context from their previous experiences.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def inject_memory_into_prompt(
        self,
        original_prompt: str,
        agent_id: int,
        memory_retrieval_results: Dict[str, Any],
        subtask_id: str = None
    ) -> str:
        """
        Inject relevant memories into the agent's prompt.
        
        Args:
            original_prompt: The base prompt for the agent
            agent_id: The agent's ID
            memory_retrieval_results: Results from memory retrieval
            subtask_id: Optional subtask ID for specific context
            
        Returns:
            Enhanced prompt with memory context
        """
        
        # Check if we have memories for this agent
        agent_memories = memory_retrieval_results.get("agent_memories", {}).get(str(agent_id), {})
        
        if not agent_memories:
            self.logger.info(f"No memories found for agent {agent_id}")
            return original_prompt
        
        # Build memory context sections
        memory_sections = []
        
        # 1. Add working memory (most recent, most relevant)
        working_memory = agent_memories.get("working_memory", [])
        if working_memory:
            memory_sections.append(self._format_working_memory(working_memory))
        
        # 2. Add short-term memory (recent experiences)
        short_term = agent_memories.get("short_term", [])
        if short_term:
            memory_sections.append(self._format_short_term_memory(short_term))
        
        # 3. Add long-term memory (consolidated knowledge)
        long_term = agent_memories.get("long_term", [])
        if long_term:
            memory_sections.append(self._format_long_term_memory(long_term))
        
        # 4. Add collective memory (shared knowledge)
        collective = memory_retrieval_results.get("collective_memories", [])
        if collective:
            memory_sections.append(self._format_collective_memory(collective))
        
        # If no memory sections, return original
        if not memory_sections:
            return original_prompt
        
        # Combine all memory sections
        memory_context = "\n\n".join(memory_sections)
        
        # Create enhanced prompt with memory context
        enhanced_prompt = f"""## Your Previous Experience & Knowledge

{memory_context}

## Current Task

{original_prompt}

**Important**: Use your previous experiences to complete this task more effectively. Apply lessons learned and avoid past mistakes."""
        
        total_memories = (
            len(working_memory) + 
            len(short_term) + 
            len(long_term) + 
            len(collective)
        )
        
        self.logger.info(
            f"✅ Injected {total_memories} memories into prompt for agent {agent_id} "
            f"(working: {len(working_memory)}, short: {len(short_term)}, "
            f"long: {len(long_term)}, collective: {len(collective)})"
        )
        
        # Detailed logging for UI visibility
        if working_memory:
            self.logger.info(f"  🧠 Working Memory: {working_memory[0].get('content', '')[:100]}...")
        if short_term:
            successes = [m for m in short_term if m.get("success", False)]
            failures = [m for m in short_term if not m.get("success", True)]
            if successes:
                self.logger.info(f"  ✅ Recent Success: {successes[0].get('content', '')[:100]}...")
            if failures:
                self.logger.info(f"  ⚠️ Lesson Learned: {failures[0].get('content', '')[:100]}...")
        if long_term:
            self.logger.info(f"  💡 Knowledge: {long_term[0].get('content', '')[:100]}...")
        if collective:
            self.logger.info(f"  🌐 Team Knowledge: {collective[0].get('content', '')[:100]}...")
        
        return enhanced_prompt
    
    def _format_working_memory(self, memories: List[Dict]) -> str:
        """Format working memory (most recent, most relevant)"""
        if not memories:
            return ""
        
        lines = ["### 🧠 Recent Context (Last 5 minutes)"]
        for mem in memories[:3]:  # Limit to top 3 most recent
            content = mem.get("content", "")
            timestamp = mem.get("timestamp", "")
            lines.append(f"- {content[:200]}...")  # Truncate long content
        
        return "\n".join(lines)
    
    def _format_short_term_memory(self, memories: List[Dict]) -> str:
        """Format short-term memory (recent experiences)"""
        if not memories:
            return ""
        
        lines = ["### 📝 Recent Experiences (Last 24 hours)"]
        
        # Group by success/failure
        successes = [m for m in memories if m.get("success", False)]
        failures = [m for m in memories if not m.get("success", True)]
        
        if successes:
            lines.append("\n**Successful Approaches:**")
            for mem in successes[:3]:
                content = mem.get("content", "")
                lines.append(f"✅ {content[:150]}...")
        
        if failures:
            lines.append("\n**Lessons Learned (Avoid These):**")
            for mem in failures[:2]:
                content = mem.get("content", "")
                lines.append(f"⚠️ {content[:150]}...")
        
        return "\n".join(lines)
    
    def _format_long_term_memory(self, memories: List[Dict]) -> str:
        """Format long-term memory (consolidated knowledge)"""
        if not memories:
            return ""
        
        lines = ["### 💡 Consolidated Knowledge"]
        
        for mem in memories[:5]:  # Top 5 most relevant
            content = mem.get("content", "")
            confidence = mem.get("confidence", 0.0)
            if confidence > 0.7:  # Only show high-confidence knowledge
                lines.append(f"• {content[:200]}... (confidence: {confidence:.0%})")
        
        return "\n".join(lines)
    
    def _format_collective_memory(self, memories: List[Dict]) -> str:
        """Format collective memory (shared organizational knowledge)"""
        if not memories:
            return ""
        
        lines = ["### 🌐 Organizational Knowledge"]
        
        for mem in memories[:3]:  # Top 3 most relevant
            content = mem.get("content", "")
            source = mem.get("source", "team")
            lines.append(f"• [{source}] {content[:200]}...")
        
        return "\n".join(lines)
    
    def inject_memory_into_system_prompt(
        self,
        system_prompt: str,
        agent_type: str,
        agent_memories: Dict[str, Any]
    ) -> str:
        """
        Enhance the system prompt with agent's identity and learned behaviors.
        
        Args:
            system_prompt: Original system prompt
            agent_type: Type of agent
            agent_memories: Agent's memory statistics
            
        Returns:
            Enhanced system prompt
        """
        
        # Extract memory stats
        total_tasks = agent_memories.get("total_tasks_completed", 0)
        success_rate = agent_memories.get("success_rate", 0.0)
        expertise_areas = agent_memories.get("expertise_areas", [])
        
        if total_tasks == 0:
            # New agent, use default prompt
            return system_prompt
        
        # Build experience summary
        experience_summary = f"""You are an experienced {agent_type} agent with:
- {total_tasks} completed tasks
- {success_rate:.0%} success rate
- Expertise in: {', '.join(expertise_areas[:5]) if expertise_areas else 'general tasks'}

Apply your accumulated knowledge and experience to complete tasks efficiently."""
        
        # Combine with original system prompt
        enhanced_system_prompt = f"{experience_summary}\n\n{system_prompt}"
        
        return enhanced_system_prompt
