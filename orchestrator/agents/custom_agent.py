
"""
Custom Agent Implementation
==========================

Flexible agent implementation that can be configured with custom skills and capabilities.
Allows users to create specialized agents for specific use cases.
"""

from typing import Dict, List, Any, Optional
import asyncio
import json
from .base_agent import BaseAgent, AgentSkill, AgentCapability, SkillType, AgentStatus

class CustomAgent(BaseAgent):
    """Flexible agent that can be customized with specific skills and capabilities"""
    
    def __init__(self, 
                 agent_id: int,
                 name: str,
                 description: str,
                 configuration: Dict[str, Any] = None,
                 custom_skills: List[Dict[str, Any]] = None,
                 custom_capabilities: List[Dict[str, Any]] = None):
        
        # Store custom configuration before calling parent init
        self.custom_skills_config = custom_skills or []
        self.custom_capabilities_config = custom_capabilities or []
        
        super().__init__(agent_id, name, description, configuration)
    
    @property
    def agent_type(self) -> str:
        return "custom"
    
    @property
    def default_skills(self) -> List[str]:
        # Return skills from configuration or empty list
        return [skill.get("name", "") for skill in self.custom_skills_config]
    
    @property
    def specializations(self) -> List[str]:
        # Extract specializations from configuration
        config_specializations = self.configuration.get("specializations", [])
        return config_specializations if config_specializations else ["general_purpose"]
    
    def _initialize_skills(self):
        """Initialize custom skills based on configuration"""
        
        # If no custom skills provided, add some default general-purpose skills
        if not self.custom_skills_config:
            self._add_default_skills()
        else:
            self._add_custom_skills()
    
    def _add_default_skills(self):
        """Add default general-purpose skills"""
        
        # General Task Processing Skill
        self.add_skill(AgentSkill(
            name="general_task_processing",
            skill_type=SkillType.COGNITIVE,
            description="Process general tasks and requests",
            parameters={
                "task_types": ["analysis", "processing", "generation"],
                "complexity_handling": "adaptive",
                "output_formats": ["text", "json", "structured"]
            }
        ))
        
        # Data Processing Skill
        self.add_skill(AgentSkill(
            name="data_processing",
            skill_type=SkillType.TECHNICAL,
            description="Process and transform data",
            parameters={
                "data_formats": ["json", "csv", "xml", "text"],
                "processing_types": ["filtering", "transformation", "aggregation"],
                "validation": True
            }
        ))
        
        # Communication Skill
        self.add_skill(AgentSkill(
            name="communication",
            skill_type=SkillType.COMMUNICATION,
            description="Communicate results and status",
            parameters={
                "communication_types": ["reports", "notifications", "summaries"],
                "formats": ["text", "structured", "visual"],
                "audiences": ["technical", "business", "general"]
            }
        ))
        
        # Problem Solving Skill
        self.add_skill(AgentSkill(
            name="problem_solving",
            skill_type=SkillType.COGNITIVE,
            description="Analyze and solve problems",
            parameters={
                "problem_types": ["analytical", "creative", "technical"],
                "solution_approaches": ["systematic", "iterative", "collaborative"],
                "complexity_levels": ["simple", "moderate", "complex"]
            }
        ))
    
    def _add_custom_skills(self):
        """Add skills from custom configuration"""
        
        for skill_config in self.custom_skills_config:
            try:
                # Parse skill type
                skill_type_str = skill_config.get("skill_type", "cognitive")
                skill_type = SkillType(skill_type_str.lower())
                
                # Create and add skill
                skill = AgentSkill(
                    name=skill_config.get("name", "unnamed_skill"),
                    skill_type=skill_type,
                    description=skill_config.get("description", "Custom skill"),
                    parameters=skill_config.get("parameters", {}),
                    performance_metrics=skill_config.get("performance_metrics", {})
                )
                
                self.add_skill(skill)
                
            except Exception as e:
                # Log error but continue with other skills
                print(f"Error adding custom skill {skill_config.get('name', 'unknown')}: {e}")
    
    def _initialize_capabilities(self):
        """Initialize custom capabilities based on configuration"""
        
        # If no custom capabilities provided, create default ones
        if not self.custom_capabilities_config:
            self._add_default_capabilities()
        else:
            self._add_custom_capabilities()
    
    def _add_default_capabilities(self):
        """Add default general-purpose capabilities"""
        
        # General Task Execution
        self.capabilities["general_task_execution"] = AgentCapability(
            name="general_task_execution",
            description="Execute general tasks and requests",
            required_skills=["general_task_processing"],
            optional_skills=["communication"],
            complexity_level=3
        )
        
        # Data Analysis and Processing
        self.capabilities["data_analysis_processing"] = AgentCapability(
            name="data_analysis_processing",
            description="Analyze and process data",
            required_skills=["data_processing"],
            optional_skills=["problem_solving", "communication"],
            complexity_level=4
        )
        
        # Problem Resolution
        self.capabilities["problem_resolution"] = AgentCapability(
            name="problem_resolution",
            description="Analyze and resolve problems",
            required_skills=["problem_solving"],
            optional_skills=["data_processing", "communication"],
            complexity_level=4
        )
    
    def _add_custom_capabilities(self):
        """Add capabilities from custom configuration"""
        
        for cap_config in self.custom_capabilities_config:
            try:
                capability = AgentCapability(
                    name=cap_config.get("name", "unnamed_capability"),
                    description=cap_config.get("description", "Custom capability"),
                    required_skills=cap_config.get("required_skills", []),
                    optional_skills=cap_config.get("optional_skills", []),
                    complexity_level=cap_config.get("complexity_level", 3)
                )
                
                self.capabilities[capability.name] = capability
                
            except Exception as e:
                # Log error but continue with other capabilities
                print(f"Error adding custom capability {cap_config.get('name', 'unknown')}: {e}")
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom agent tasks"""
        
        self.set_status(AgentStatus.BUSY)
        start_time = asyncio.get_event_loop().time()
        
        try:
            task_type = task.get("type", "general")
            
            # Route to appropriate handler based on task type
            if task_type == "general" or task_type == "custom":
                result = await self._execute_general_task(task)
            elif task_type == "data_processing":
                result = await self._execute_data_processing_task(task)
            elif task_type == "problem_solving":
                result = await self._execute_problem_solving_task(task)
            elif task_type == "analysis":
                result = await self._execute_analysis_task(task)
            else:
                # Try to handle unknown task types generically
                result = await self._execute_generic_task(task)
            
            execution_time = asyncio.get_event_loop().time() - start_time
            self.update_performance_metrics(execution_time, result.get("success", False))
            self.set_status(AgentStatus.ACTIVE)
            
            return result
            
        except Exception as e:
            execution_time = asyncio.get_event_loop().time() - start_time
            self.update_performance_metrics(execution_time, False)
            self.set_status(AgentStatus.ERROR)
            
            return {
                "success": False,
                "error": str(e),
                "task_type": task_type
            }
    
    async def _execute_general_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute general/custom tasks"""
        
        task_description = task.get("description", "")
        task_parameters = task.get("parameters", {})
        expected_output = task.get("expected_output", "structured_response")
        
        # Simulate general task processing
        await asyncio.sleep(2)
        
        # Analyze task requirements
        complexity = self._analyze_task_complexity(task_description)
        required_skills = self._identify_required_skills(task_description)
        
        result = {
            "task_analysis": {
                "description": task_description,
                "complexity": complexity,
                "required_skills": required_skills,
                "estimated_effort": self._estimate_effort(complexity)
            },
            "execution_plan": {
                "approach": "adaptive_processing",
                "steps": [
                    "Analyze task requirements",
                    "Identify relevant skills and capabilities",
                    "Execute task using available skills",
                    "Validate and format results",
                    "Provide recommendations if applicable"
                ],
                "risk_assessment": "low" if complexity <= 3 else "medium"
            },
            "results": {
                "status": "completed",
                "output_type": expected_output,
                "key_findings": [
                    "Task successfully analyzed and processed",
                    f"Applied {len(required_skills)} relevant skills",
                    "Results formatted according to requirements"
                ],
                "recommendations": [
                    "Consider breaking down complex tasks into smaller components",
                    "Provide more specific requirements for better results",
                    "Regular feedback helps improve task execution"
                ]
            },
            "performance_metrics": {
                "processing_time": "2.0 seconds",
                "accuracy_estimate": 0.85,
                "confidence_level": 0.78,
                "resource_utilization": "moderate"
            }
        }
        
        return {
            "success": True,
            "task_type": "general",
            "results": result,
            "agent_id": self.agent_id,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    async def _execute_data_processing_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data processing tasks"""
        
        data_input = task.get("data", {})
        processing_type = task.get("processing_type", "analysis")
        output_format = task.get("output_format", "json")
        
        # Simulate data processing
        await asyncio.sleep(1.5)
        
        processing_results = {
            "data_summary": {
                "input_size": len(str(data_input)) if data_input else 0,
                "processing_type": processing_type,
                "output_format": output_format,
                "processing_time": "1.5 seconds"
            },
            "processing_steps": [
                "Data validation and cleaning",
                "Structure analysis and parsing",
                "Processing according to specified type",
                "Result formatting and validation"
            ],
            "results": {
                "processed_records": 100,  # Mock value
                "data_quality_score": 0.92,
                "processing_success_rate": 0.98,
                "output_size": "estimated_based_on_input"
            },
            "insights": [
                "Data processing completed successfully",
                "High data quality score indicates clean input",
                "Processing efficiency within expected parameters"
            ]
        }
        
        return {
            "success": True,
            "task_type": "data_processing",
            "results": processing_results,
            "agent_id": self.agent_id,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    async def _execute_problem_solving_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute problem-solving tasks"""
        
        problem_description = task.get("problem", "")
        context = task.get("context", {})
        solution_approach = task.get("approach", "systematic")
        
        # Simulate problem solving
        await asyncio.sleep(2.5)
        
        problem_solving_results = {
            "problem_analysis": {
                "problem_type": self._classify_problem_type(problem_description),
                "complexity_level": self._analyze_problem_complexity(problem_description),
                "solution_approach": solution_approach,
                "context_factors": len(context) if context else 0
            },
            "solution_process": [
                "Problem definition and scope clarification",
                "Root cause analysis",
                "Solution alternatives generation",
                "Solution evaluation and selection",
                "Implementation planning"
            ],
            "proposed_solutions": [
                {
                    "solution_id": 1,
                    "description": "Primary solution based on systematic analysis",
                    "feasibility": "high",
                    "estimated_effort": "medium",
                    "risk_level": "low",
                    "expected_outcome": "Addresses core problem effectively"
                },
                {
                    "solution_id": 2,
                    "description": "Alternative approach with different trade-offs",
                    "feasibility": "medium",
                    "estimated_effort": "low",
                    "risk_level": "medium",
                    "expected_outcome": "Quick fix with potential limitations"
                }
            ],
            "recommendations": [
                "Implement primary solution for long-term resolution",
                "Consider alternative solution for immediate relief",
                "Monitor implementation progress and adjust as needed",
                "Document lessons learned for future reference"
            ],
            "success_metrics": [
                "Problem resolution effectiveness",
                "Implementation time and cost",
                "Stakeholder satisfaction",
                "Prevention of problem recurrence"
            ]
        }
        
        return {
            "success": True,
            "task_type": "problem_solving",
            "results": problem_solving_results,
            "agent_id": self.agent_id,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    async def _execute_analysis_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis tasks"""
        
        analysis_subject = task.get("subject", "")
        analysis_type = task.get("analysis_type", "general")
        depth_level = task.get("depth", "standard")
        
        # Simulate analysis
        await asyncio.sleep(2)
        
        analysis_results = {
            "analysis_overview": {
                "subject": analysis_subject,
                "analysis_type": analysis_type,
                "depth_level": depth_level,
                "analysis_duration": "2.0 seconds"
            },
            "methodology": {
                "approach": "multi_dimensional_analysis",
                "techniques_used": ["descriptive", "comparative", "trend_analysis"],
                "data_sources": ["provided_input", "contextual_knowledge"],
                "validation_methods": ["consistency_check", "logical_validation"]
            },
            "key_findings": [
                "Analysis completed within specified parameters",
                "Multiple perspectives considered in evaluation",
                "Results validated for consistency and accuracy"
            ],
            "detailed_analysis": {
                "strengths_identified": [
                    "Clear analysis objectives",
                    "Appropriate methodology selection",
                    "Comprehensive evaluation approach"
                ],
                "areas_for_improvement": [
                    "Could benefit from additional data sources",
                    "Deeper analysis possible with more time",
                    "Stakeholder input would enhance insights"
                ],
                "risk_factors": [
                    "Limited data availability",
                    "Time constraints on analysis depth",
                    "Assumptions made due to incomplete information"
                ]
            },
            "conclusions": [
                "Analysis objectives successfully met",
                "Results provide actionable insights",
                "Recommendations based on evidence and best practices"
            ],
            "next_steps": [
                "Review analysis results with stakeholders",
                "Implement recommended actions",
                "Monitor outcomes and adjust approach as needed",
                "Schedule follow-up analysis if required"
            ]
        }
        
        return {
            "success": True,
            "task_type": "analysis",
            "results": analysis_results,
            "agent_id": self.agent_id,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    async def _execute_generic_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute unknown/generic task types"""
        
        task_type = task.get("type", "unknown")
        
        # Simulate generic task processing
        await asyncio.sleep(1)
        
        generic_results = {
            "task_info": {
                "original_type": task_type,
                "processing_approach": "generic_handler",
                "adaptation_strategy": "best_effort_processing"
            },
            "processing_notes": [
                f"Task type '{task_type}' not specifically recognized",
                "Applied generic processing capabilities",
                "Results may be limited compared to specialized handlers"
            ],
            "results": {
                "status": "processed",
                "approach": "adaptive",
                "confidence": "moderate",
                "recommendations": [
                    "Consider using a more specific task type for better results",
                    "Provide additional context or parameters",
                    "Use specialized agents for domain-specific tasks"
                ]
            }
        }
        
        return {
            "success": True,
            "task_type": task_type,
            "results": generic_results,
            "agent_id": self.agent_id,
            "timestamp": asyncio.get_event_loop().time()
        }
    
    def _analyze_task_complexity(self, description: str) -> int:
        """Analyze task complexity (1-5 scale)"""
        if not description:
            return 1
        
        # Simple heuristic based on description length and keywords
        complexity_indicators = ["complex", "comprehensive", "detailed", "multi", "advanced"]
        complexity_score = 1
        
        if len(description) > 100:
            complexity_score += 1
        if len(description) > 200:
            complexity_score += 1
        
        for indicator in complexity_indicators:
            if indicator in description.lower():
                complexity_score += 1
                break
        
        return min(complexity_score, 5)
    
    def _identify_required_skills(self, description: str) -> List[str]:
        """Identify skills likely needed for the task"""
        available_skills = list(self.skills.keys())
        
        if not description:
            return available_skills[:2]  # Return first 2 skills as default
        
        # Simple keyword matching to identify relevant skills
        relevant_skills = []
        description_lower = description.lower()
        
        skill_keywords = {
            "data_processing": ["data", "process", "transform", "analyze"],
            "problem_solving": ["problem", "solve", "issue", "challenge"],
            "communication": ["report", "communicate", "present", "explain"],
            "general_task_processing": ["task", "execute", "perform", "handle"]
        }
        
        for skill_name, keywords in skill_keywords.items():
            if skill_name in available_skills:
                if any(keyword in description_lower for keyword in keywords):
                    relevant_skills.append(skill_name)
        
        # If no specific skills identified, return general ones
        if not relevant_skills:
            relevant_skills = [skill for skill in available_skills if "general" in skill][:2]
        
        return relevant_skills
    
    def _estimate_effort(self, complexity: int) -> str:
        """Estimate effort based on complexity"""
        effort_map = {
            1: "minimal",
            2: "low",
            3: "moderate",
            4: "high",
            5: "extensive"
        }
        return effort_map.get(complexity, "moderate")
    
    def _classify_problem_type(self, problem: str) -> str:
        """Classify the type of problem"""
        if not problem:
            return "unspecified"
        
        problem_lower = problem.lower()
        
        if any(word in problem_lower for word in ["technical", "system", "code", "bug"]):
            return "technical"
        elif any(word in problem_lower for word in ["business", "process", "workflow"]):
            return "business"
        elif any(word in problem_lower for word in ["data", "analysis", "information"]):
            return "analytical"
        else:
            return "general"
    
    def _analyze_problem_complexity(self, problem: str) -> str:
        """Analyze problem complexity"""
        if not problem:
            return "low"
        
        complexity_score = 0
        problem_lower = problem.lower()
        
        # Increase complexity based on indicators
        if len(problem) > 100:
            complexity_score += 1
        if any(word in problem_lower for word in ["multiple", "complex", "interconnected"]):
            complexity_score += 1
        if any(word in problem_lower for word in ["system", "integration", "architecture"]):
            complexity_score += 1
        
        if complexity_score >= 2:
            return "high"
        elif complexity_score == 1:
            return "medium"
        else:
            return "low"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomAgent':
        """Create CustomAgent from dictionary representation"""
        
        custom_skills = data.get('custom_skills', [])
        custom_capabilities = data.get('custom_capabilities', [])
        
        agent = cls(
            agent_id=data['agent_id'],
            name=data['name'],
            description=data['description'],
            configuration=data.get('configuration', {}),
            custom_skills=custom_skills,
            custom_capabilities=custom_capabilities
        )
        
        # Restore performance metrics if available
        if 'performance_metrics' in data:
            agent.performance_metrics = data['performance_metrics']
        
        return agent
