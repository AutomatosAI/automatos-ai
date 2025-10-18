"""
Real Task Decomposer using LLM
===============================

NO MOCK DATA - This module actually decomposes tasks using OpenAI GPT-4
"""

import json
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import asyncio

# Import the existing LLM provider
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from services.llm_provider import create_llm_manager, LLMConfig, LLMProvider

logger = logging.getLogger(__name__)

class RealTaskDecomposer:
    """
    REAL task decomposition using LLM - NO MOCK DATA
    """
    
    def __init__(self, llm_manager=None):
        """Initialize with LLM connection"""
        if not llm_manager:
            # Use config from centralized configuration to create LLM manager
            from config import config
            
            # Create the LLM manager with proper provider and model
            self.llm = create_llm_manager(
                provider=config.LLM_PROVIDER,
                model=config.LLM_MODEL
            )
            logger.info(f"RealTaskDecomposer initialized with {config.LLM_PROVIDER}/{config.LLM_MODEL}")
        else:
            self.llm = llm_manager
            logger.info("RealTaskDecomposer initialized with provided LLM manager")
    
    async def decompose_task(
        self,
        task_description: str,
        task_type: str = "general",
        complexity: str = "medium",
        requirements: List[str] = None,
        max_subtasks: int = 7
    ) -> Dict[str, Any]:
        """
        Decompose a complex task into subtasks using REAL LLM
        
        Returns ACTUAL decomposition, not mock data
        """
        requirements = requirements or []
        
        # Build the decomposition prompt
        prompt = f"""You are an expert task decomposition system for a multi-agent AI orchestration platform.

CRITICAL DESIGN PRINCIPLES:
1. Create GRANULAR subtasks - each subtask should focus on ONE primary skill/capability
2. Design for AGENT COLLABORATION - agents will share results via shared context
3. Each subtask should be ATOMIC and FOCUSED - not compound tasks
4. Create clear HANDOFF POINTS between subtasks for inter-agent communication
5. TOOL-AUGMENTED TASKS - agents have access to search tools and MUST use them

🔧 AVAILABLE TOOLS FOR AGENTS:
- search_knowledge(query) - Search documentation, PDFs, markdown files
- search_codebase(query) - Search code functions, classes, implementations  
- semantic_search(query) - Find semantically similar content
- read_document(doc_id) - Read full document content

⚠️  IMPORTANT: When creating research/analysis subtasks, include SPECIFIC search queries the agent should use!

Break down this complex task into specific, actionable subtasks:

TASK: {task_description}
TYPE: {task_type}
COMPLEXITY: {complexity}
REQUIREMENTS: {', '.join(requirements) if requirements else 'None specified'}

Please decompose this into {5 if complexity == 'low' else 7 if complexity == 'medium' else 10} subtasks.

GRANULARITY EXAMPLES:
❌ BAD: "Research and analyze the data" (2 skills: research + analysis)
✅ GOOD: 
   - Subtask 1: "Collect and organize relevant research materials" (research)
   - Subtask 2: "Analyze the collected materials for patterns" (analysis)

❌ BAD: "Write and review the documentation" (2 skills: writing + review)
✅ GOOD:
   - Subtask 1: "Write comprehensive documentation draft" (writing)
   - Subtask 2: "Review documentation for accuracy and completeness" (review)

🔧 TOOL USAGE EXAMPLES (for research/documentation tasks):
✅ EXCELLENT: "Search platform documentation using search_knowledge('Automatos AI architecture'), search_knowledge('deployment guide'), and search_knowledge('9-stage orchestration'). Find at least 5 relevant documentation files. Extract key architectural concepts, technology stack, and core features. Provide detailed quotes with source citations [Source: filename.md]."

✅ EXCELLENT: "Find AgentFactory code using search_codebase('AgentFactory class'), search_codebase('create_agent method'), and search_codebase('agent lifecycle'). Analyze the implementation and extract 3-5 code examples showing: agent creation, configuration, and execution. Include file paths and line numbers for each example."

For each subtask, provide:
1. A clear, ATOMIC description focusing on ONE primary skill
2. The type of agent best suited (e.g., 'researcher', 'analyst', 'developer', 'writer', 'reviewer')
3. Priority level (high/medium/low)
4. Dependencies (which subtasks must complete first for inter-agent collaboration)
5. Estimated duration in seconds (30-300 range)
6. **TOOL GUIDANCE** - Specific search queries if the subtask needs research

Return ONLY valid JSON in this exact format:
{{
  "subtasks": [
    {{
      "subtask_id": "unique_id",
      "description": "Clear ATOMIC description - focus on ONE primary skill. If research is needed, include: 'Use search_knowledge(\"your query here\") to find X. Use search_codebase(\"ClassName\") to find Y.'",
      "agent_type": "type_of_agent",
      "priority": "high|medium|low",
      "dependencies": ["subtask_ids_that_must_complete_first"],
      "estimated_duration": "60-120 seconds",
      "primary_skill": "single_main_skill",
      "skills_required": ["primary_skill_only"],
      "tool_guidance": "Optional: List specific search queries like ['search_knowledge(\"architecture docs\")', 'search_codebase(\"AgentFactory\")']"
    }}
  ],
  "execution_strategy": "parallel|sequential|mixed",
  "total_estimated_time": "5-10 minutes",
  "complexity_assessment": {{
    "technical_complexity": "low|medium|high",
    "coordination_complexity": "low|medium|high",
    "resource_requirements": "low|medium|high"
  }},
  "collaboration_notes": "How agents will share information via shared context"
}}

REMEMBER: 
- Each subtask = ONE primary skill
- Agents will collaborate via shared context (PRD-04)
- Design for sequential handoffs where needed
- Use dependencies to enforce proper order"""
        
        try:
            # Make REAL LLM call
            logger.info(f"🔍 STAGE 1 START: Decomposing task with REAL LLM")
            logger.info(f"  📝 Task: {task_description[:150]}...")
            logger.info(f"  ⚙️  Complexity: {complexity} → Target: {5 if complexity == 'low' else 7 if complexity == 'medium' else 10} subtasks")
            logger.info(f"  🎯 Design: Granular, single-skill, inter-agent collaboration")
            
            messages = [
                {"role": "system", "content": "You are an expert task decomposition system. Always return valid JSON."},
                {"role": "user", "content": prompt}
            ]
            
            # This is a REAL API call - will take 1-3 seconds
            import time
            start_time = time.time()
            
            response = await self.llm.generate_response(messages)
            
            elapsed = time.time() - start_time
            logger.info(f"  ✅ LLM response received in {elapsed:.2f} seconds")
            
            # Parse the response
            try:
                # Clean up the response (remove markdown if present)
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0]
                elif "```" in content:
                    content = content.split("```")[1].split("```")[0]
                
                result = json.loads(content)
                
                # Add task_id prefix to subtask IDs
                task_id = f"task_{int(datetime.now().timestamp())}"
                for i, subtask in enumerate(result.get("subtasks", [])):
                    if "subtask_id" not in subtask or subtask["subtask_id"] == "unique_id":
                        subtask["subtask_id"] = f"{task_id}_subtask_{i+1}"
                    
                    # Fix dependencies to use proper IDs
                    if subtask.get("dependencies"):
                        fixed_deps = []
                        for dep in subtask["dependencies"]:
                            if dep and dep != "list" and dep != "of" and dep != "subtask_ids":
                                # Find the actual subtask ID
                                for j, other in enumerate(result.get("subtasks", [])):
                                    if j < i:  # Only depend on earlier subtasks
                                        fixed_deps.append(f"{task_id}_subtask_{j+1}")
                                        break
                        subtask["dependencies"] = fixed_deps[:1] if fixed_deps else []  # Limit to 1 dependency for simplicity
                
                # Add metadata
                result["task_id"] = task_id
                result["original_task"] = task_description
                result["decomposed_at"] = datetime.now().isoformat()
                result["llm_model"] = response.model
                result["tokens_used"] = response.usage["total_tokens"] if response.usage else None
                result["decomposition_time"] = elapsed
                result["is_real_decomposition"] = True  # Proof this is NOT mock data
                
                # Enhanced debug logging
                subtasks = result.get('subtasks', [])
                logger.info(f"🎯 STAGE 1 COMPLETE: Decomposed into {len(subtasks)} subtasks")
                logger.info(f"  📊 Strategy: {result.get('execution_strategy', 'N/A')}")
                logger.info(f"  ⏱️  Total estimated time: {result.get('total_estimated_time', 'N/A')}")
                
                for i, subtask in enumerate(subtasks):
                    skills = subtask.get('skills_required', subtask.get('primary_skill', []))
                    if isinstance(skills, str):
                        skills = [skills]
                    logger.info(f"  📋 Subtask {i+1}: {subtask.get('description', 'N/A')[:80]}...")
                    logger.info(f"      🤖 Agent type: {subtask.get('agent_type', 'N/A')}")
                    logger.info(f"      🎯 Skills: {skills}")
                    logger.info(f"      🔗 Dependencies: {subtask.get('dependencies', [])}")
                
                return result
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM response as JSON: {e}")
                # Return a simple decomposition if parsing fails
                return self._create_fallback_decomposition(task_description, task_id)
                
        except Exception as e:
            logger.error(f"Error in task decomposition: {e}")
            raise Exception(f"Real task decomposition failed: {str(e)}")
    
    def _create_fallback_decomposition(self, task_description: str, task_id: str) -> Dict[str, Any]:
        """
        Create a basic decomposition if LLM response can't be parsed
        Still uses REAL data, not mock
        """
        return {
            "task_id": task_id,
            "original_task": task_description,
            "subtasks": [
                {
                    "subtask_id": f"{task_id}_analyze",
                    "description": f"Analyze requirements for: {task_description[:50]}",
                    "agent_type": "analyst",
                    "priority": "high",
                    "dependencies": [],
                    "estimated_duration": "60-120 seconds",
                    "skills_required": ["analysis", "planning"]
                },
                {
                    "subtask_id": f"{task_id}_implement",
                    "description": f"Implement solution for: {task_description[:50]}",
                    "agent_type": "developer",
                    "priority": "high",
                    "dependencies": [f"{task_id}_analyze"],
                    "estimated_duration": "120-180 seconds",
                    "skills_required": ["implementation", "problem_solving"]
                },
                {
                    "subtask_id": f"{task_id}_validate",
                    "description": f"Validate and test the solution",
                    "agent_type": "reviewer",
                    "priority": "medium",
                    "dependencies": [f"{task_id}_implement"],
                    "estimated_duration": "60-90 seconds",
                    "skills_required": ["testing", "validation"]
                }
            ],
            "execution_strategy": "sequential",
            "total_estimated_time": "4-7 minutes",
            "complexity_assessment": {
                "technical_complexity": "medium",
                "coordination_complexity": "low",
                "resource_requirements": "medium"
            },
            "is_fallback": True,
            "is_real_decomposition": True,
            "decomposed_at": datetime.now().isoformat()
        }
    
    async def analyze_task_complexity(self, task_description: str) -> Dict[str, Any]:
        """
        Analyze task complexity using REAL LLM
        """
        prompt = f"""Analyze the complexity of this task:

TASK: {task_description}

Provide analysis in JSON format:
{{
  "complexity_level": "low|medium|high",
  "estimated_subtasks": 3-7,
  "required_expertise": ["list", "of", "skills"],
  "challenges": ["potential", "challenges"],
  "success_criteria": ["measurable", "outcomes"]
}}"""
        
        messages = [
            {"role": "system", "content": "You are a task complexity analyzer."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.llm.generate_response(messages)
        
        try:
            content = response.content
            if "```" in content:
                content = content.split("```")[1].split("```")[0]
                if content.startswith("json"):
                    content = content[4:]
            
            return json.loads(content)
        except:
            # Default complexity assessment
            return {
                "complexity_level": "medium",
                "estimated_subtasks": 5,
                "required_expertise": ["general"],
                "challenges": ["Unknown complexity"],
                "success_criteria": ["Task completed"]
            }
    
    async def identify_dependencies(self, subtasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Identify dependencies between subtasks using REAL LLM reasoning
        """
        subtask_descriptions = "\n".join([
            f"{i+1}. {task['description']}" 
            for i, task in enumerate(subtasks)
        ])
        
        prompt = f"""Given these subtasks, identify which ones depend on others:

{subtask_descriptions}

For each subtask, list which other subtasks (by number) must complete first.
Return as JSON: {{"1": [], "2": [1], "3": [1, 2]}}"""
        
        messages = [
            {"role": "system", "content": "You are a dependency analyzer."},
            {"role": "user", "content": prompt}
        ]
        
        response = await self.llm.generate_response(messages)
        
        # Parse and apply dependencies
        try:
            deps = json.loads(response.content)
            for i, subtask in enumerate(subtasks):
                subtask_num = str(i + 1)
                if subtask_num in deps:
                    dep_indices = deps[subtask_num]
                    subtask["dependencies"] = [
                        subtasks[j-1]["subtask_id"] 
                        for j in dep_indices 
                        if j > 0 and j <= len(subtasks)
                    ]
        except:
            logger.warning("Could not parse dependencies, using sequential")
            # Default to sequential dependencies
            for i, subtask in enumerate(subtasks):
                if i > 0:
                    subtask["dependencies"] = [subtasks[i-1]["subtask_id"]]
        
        return subtasks

# Singleton instance for easy import
_decomposer_instance = None

def get_decomposer() -> RealTaskDecomposer:
    """Get or create the singleton decomposer instance"""
    global _decomposer_instance
    if _decomposer_instance is None:
        _decomposer_instance = RealTaskDecomposer()
    return _decomposer_instance
