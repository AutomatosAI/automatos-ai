
"""
Enhanced Orchestrator with Detailed Logging and Agent Communication
==================================================================

This is the enhanced version of the orchestrator with comprehensive logging,
agent communication tracking, and performance monitoring capabilities.
"""

import asyncio
import json
import logging
import os
import subprocess
import time
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, TypedDict, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path

from crewai import Agent, Task, Crew
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks import get_openai_callback
from langchain.evaluation import load_evaluator
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from context_manager import EnhancedContextManager
from mcp_bridge import EnhancedMCPBridge
from ssh_manager import EnhancedSSHManager, SSHConnection, SecurityLevel
from ai_module_parser import AIModuleParser, AIModuleConfig, ModuleType
from security import SecurityAuditLogger, create_security_event, EventType, get_audit_logger
from llm_provider import LLMManager, LLMResponse
from logging_utils import EnhancedWorkflowLogger, PerformanceTimer, timed_operation, TokenTracker
from agent_comm import AgentCommunicationHub, AgentCommunicationClient, MessageType, MessagePriority
from dotenv import load_dotenv
from context_integration import get_context_manager, initialize_context_engineering

load_dotenv()

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('orchestrator.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class WorkflowType(Enum):
    AI_MODULE = "ai_module"
    TASK_PROMPT = "task_prompt"

class DeploymentStatus(Enum):
    PENDING = "pending"
    BUILDING = "building"
    TESTING = "testing"
    DEPLOYING = "deploying"
    RUNNING = "running"
    FAILED = "failed"
    STOPPED = "stopped"

@dataclass
class WorkflowState:
    workflow_id: str
    workflow_type: WorkflowType
    status: DeploymentStatus
    repository_url: str
    target_host: str
    project_path: str
    config: Optional[AIModuleConfig] = None
    task_prompt: Optional[str] = None
    environment_variables: Dict[str, str] = None
    deployment_logs: List[str] = None
    created_at: datetime = None
    updated_at: datetime = None

class EnhancedTwoTierOrchestrator:
    """Enhanced Two-Tiered Multi-Agent Orchestration System with Detailed Logging"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
        # Generate unique workflow ID
        self.workflow_id = f"workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(config)) % 10000:04d}"
        
        # Initialize enhanced logging
        self.workflow_logger = EnhancedWorkflowLogger(self.workflow_id)
        self.workflow_logger.log_workflow_step("init", "Initialize Orchestrator", "system", "in_progress")
        
        # Initialize agent communication hub
        self.comm_hub = AgentCommunicationHub()
        self.agent_client = AgentCommunicationClient(
            "orchestrator", 
            "main", 
            ["coordination", "planning", "deployment"], 
            self.comm_hub
        )
        
        # Initialize components
        self.context_manager = EnhancedContextManager()
        self.mcp_bridge = EnhancedMCPBridge()
        self.ssh_manager = EnhancedSSHManager()
        self.ai_module_parser = AIModuleParser()
        self.audit_logger = get_audit_logger()
        
        # Initialize AI components
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.1,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize flexible LLM manager
        self.llm_manager = LLMManager()
        
        # Initialize Context Engineering Manager
        self.context_engineering = None
        
        # Workflow tracking
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.rollback_stack: List[Dict[str, Any]] = []
        
        # SSH configuration
        self.default_ssh_config = SSHConnection(
            host=os.getenv("DEPLOY_HOST", "mcp.xplaincrypto.ai"),
            port=int(os.getenv("DEPLOY_PORT", "22")),
            username=os.getenv("DEPLOY_USER", "root"),
            key_path=os.getenv("DEPLOY_KEY_PATH", "/root/.ssh/id_rsa"),
            security_level=SecurityLevel.HIGH
        )
        
        # Register specialized agents
        self._register_specialized_agents()
        
        self.workflow_logger.log_workflow_step("init", "Initialize Orchestrator", "system", "completed")
        self.workflow_logger.log_agent_status("orchestrator", "main", "System initialized", "idle", 100.0)
        
        logger.info(f"Enhanced Two-Tier Orchestrator initialized - Workflow ID: {self.workflow_id}")
        
        # Initialize context engineering asynchronously
        asyncio.create_task(self._initialize_context_engineering())
    
    def _register_specialized_agents(self):
        """Register specialized agents with the communication hub"""
        agents = [
            ("task_analyzer", "specialist", ["task_analysis", "planning", "breakdown"]),
            ("code_generator", "specialist", ["code_generation", "python", "javascript", "typescript"]),
            ("git_manager", "specialist", ["git_operations", "version_control", "commits"]),
            ("deployment_manager", "specialist", ["deployment", "docker", "ssh", "monitoring"]),
            ("quality_assurance", "specialist", ["testing", "validation", "quality_control"])
        ]
        
        for agent_id, agent_type, capabilities in agents:
            AgentCommunicationClient(agent_id, agent_type, capabilities, self.comm_hub)
            self.workflow_logger.log_agent_status(agent_id, agent_type, "Registered and ready", "idle", 0.0)
    
    # ============================================================================
    # ENHANCED COGNITIVE FUNCTIONS WITH DETAILED LOGGING
    # ============================================================================
    
    @timed_operation("cognitive_task_breakdown")
    async def cognitive_task_breakdown(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """AI-powered task analysis and decomposition with detailed logging"""
        
        step_id = "task_breakdown"
        self.workflow_logger.log_workflow_step(step_id, "Cognitive Task Breakdown", "task_analyzer", "in_progress")
        self.workflow_logger.log_agent_status("task_analyzer", "specialist", "Analyzing task complexity", "working", 10.0)
        
        # Agent communication
        self.agent_client.send_message(
            "task_analyzer",
            MessageType.TASK_HANDOFF,
            {
                "task_description": task_description,
                "context": context,
                "action": "analyze_and_breakdown"
            },
            priority=MessagePriority.HIGH
        )
        
        try:
            self.workflow_logger.log_agent_status("task_analyzer", "specialist", "Preparing context analysis", "working", 25.0)
            
            # Prepare context information
            context_info = ""
            if context:
                if context.get("repository_structure"):
                    context_info += f"Repository structure: {context['repository_structure']}\n"
                if context.get("tech_stack"):
                    context_info += f"Technology stack: {', '.join(context['tech_stack'])}\n"
                if context.get("existing_files"):
                    context_info += f"Existing files: {', '.join(context['existing_files'][:10])}\n"
            
            self.workflow_logger.log_agent_status("task_analyzer", "specialist", "Generating breakdown prompt", "working", 50.0)
            
            # Create system prompt for task breakdown
            system_prompt = """You are an expert software architect and project manager. Your task is to analyze a development request and break it down into specific, actionable subtasks.

For each task breakdown, provide:
1. A list of specific subtasks with clear descriptions
2. Complexity score (1-10) for each subtask
3. Estimated duration in minutes for each subtask
4. Dependencies between subtasks
5. Priority level (high, medium, low)
6. Required skills/technologies
7. Overall project complexity assessment

Return your response as a structured JSON object."""
            
            user_prompt = f"""Please break down this development task:

TASK: {task_description}

CONTEXT:
{context_info}

Analyze this task and provide a detailed breakdown with specific subtasks, complexity analysis, and execution recommendations."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            self.workflow_logger.log_agent_status("task_analyzer", "specialist", "Calling LLM for analysis", "working", 75.0)
            
            # Generate response using LLM manager
            response = await self.llm_manager.generate_response(messages)
            
            # Log token usage
            if response.usage:
                cost = TokenTracker.calculate_cost(
                    response.model or "gpt-4",
                    response.usage.get("prompt_tokens", 0),
                    response.usage.get("completion_tokens", 0)
                )
                self.workflow_logger.log_performance_metric(
                    "task_breakdown_llm_call",
                    1.0,  # Duration will be calculated by timer
                    tokens_used=response.usage.get("total_tokens", 0),
                    estimated_cost=cost
                )
            
            self.workflow_logger.log_agent_status("task_analyzer", "specialist", "Processing LLM response", "working", 90.0)
            
            # Try to parse JSON response
            try:
                breakdown_data = json.loads(response.content)
                self.workflow_logger.log_agent_status("task_analyzer", "specialist", "JSON parsing successful", "working", 95.0)
            except json.JSONDecodeError:
                self.workflow_logger.log_error_with_recovery(
                    Exception("JSON parsing failed"),
                    "Creating structured fallback response",
                    {"raw_response": response.content[:500]}
                )
                # If JSON parsing fails, create structured response from text
                breakdown_data = {
                    "subtasks": [
                        {
                            "id": 1,
                            "description": task_description,
                            "complexity": 5,
                            "estimated_duration": 60,
                            "priority": "high",
                            "dependencies": [],
                            "skills_required": ["general_development"]
                        }
                    ],
                    "overall_complexity": 5,
                    "total_estimated_duration": 60,
                    "analysis": response.content
                }
            
            # Store in database for future reference
            breakdown_result = {
                "task_description": task_description,
                "breakdown_data": breakdown_data,
                "context": context,
                "llm_response": response.content,
                "provider_info": self.llm_manager.get_provider_info(),
                "created_at": datetime.now().isoformat()
            }
            
            # Log detailed breakdown results
            subtasks_count = len(breakdown_data.get('subtasks', []))
            total_complexity = breakdown_data.get('overall_complexity', 0)
            total_duration = breakdown_data.get('total_estimated_duration', 0)
            
            self.workflow_logger.log_agent_status("task_analyzer", "specialist", "Task breakdown completed", "completed", 100.0, {
                "subtasks_identified": subtasks_count,
                "overall_complexity": total_complexity,
                "estimated_duration_minutes": total_duration
            })
            
            # Agent communication - broadcast results
            self.agent_client.broadcast(
                MessageType.STATUS_UPDATE,
                {
                    "message": "Task breakdown completed",
                    "subtasks_count": subtasks_count,
                    "complexity_score": total_complexity,
                    "estimated_duration": total_duration
                }
            )
            
            self.workflow_logger.log_workflow_step(step_id, "Cognitive Task Breakdown", "task_analyzer", "completed", {
                "subtasks_identified": subtasks_count,
                "complexity_assessment": total_complexity
            })
            
            logger.info(f"Task breakdown completed: {subtasks_count} subtasks identified, complexity: {total_complexity}/10")
            return breakdown_result
            
        except Exception as e:
            self.workflow_logger.log_workflow_step(step_id, "Cognitive Task Breakdown", "task_analyzer", "failed")
            self.workflow_logger.log_agent_status("task_analyzer", "specialist", "Task breakdown failed", "error", 0.0)
            self.workflow_logger.log_error_with_recovery(e, "Returning fallback breakdown")
            
            # Report error through agent communication
            self.agent_client.report_error({
                "operation": "cognitive_task_breakdown",
                "error": str(e),
                "task_description": task_description[:200]
            })
            
            logger.error(f"Cognitive task breakdown failed: {e}")
            return {
                "error": str(e),
                "task_description": task_description,
                "fallback_subtasks": [
                    {
                        "id": 1,
                        "description": task_description,
                        "complexity": 5,
                        "estimated_duration": 60,
                        "priority": "high"
                    }
                ]
            }
    
    @timed_operation("cognitive_content_generation")
    async def cognitive_content_generation(self, 
                                         content_type: str, 
                                         specifications: Dict[str, Any],
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Production-ready code and content generation with detailed logging"""
        
        step_id = f"content_gen_{content_type}"
        self.workflow_logger.log_workflow_step(step_id, f"Generate {content_type}", "code_generator", "in_progress")
        self.workflow_logger.log_agent_status("code_generator", "specialist", f"Generating {content_type}", "working", 5.0)
        
        # Agent communication
        self.agent_client.send_message(
            "code_generator",
            MessageType.TASK_HANDOFF,
            {
                "content_type": content_type,
                "specifications": specifications,
                "context": context,
                "action": "generate_content"
            },
            priority=MessagePriority.NORMAL
        )
        
        try:
            self.workflow_logger.log_agent_status("code_generator", "specialist", "Analyzing specifications", "working", 15.0)
            
            # Prepare context information
            context_info = ""
            if context:
                if context.get("project_structure"):
                    context_info += f"Project structure: {context['project_structure']}\n"
                if context.get("coding_standards"):
                    context_info += f"Coding standards: {context['coding_standards']}\n"
                if context.get("dependencies"):
                    context_info += f"Dependencies: {', '.join(context['dependencies'])}\n"
                if context.get("target_environment"):
                    context_info += f"Target environment: {context['target_environment']}\n"
            
            self.workflow_logger.log_agent_status("code_generator", "specialist", "Preparing generation prompts", "working", 30.0)
            
            # Create specialized prompts based on content type
            if content_type.lower() in ["code", "python", "javascript", "typescript", "java", "go"]:
                system_prompt = """You are an expert software developer with deep knowledge of best practices, design patterns, and production-ready code standards.

Generate high-quality, production-ready code that follows:
1. Clean code principles
2. Proper error handling
3. Comprehensive documentation
4. Security best practices
5. Performance optimization
6. Testability and maintainability

Always include:
- Clear comments and docstrings
- Input validation
- Error handling
- Type hints (where applicable)
- Security considerations
- Performance considerations"""
                
            elif content_type.lower() in ["documentation", "readme", "api_docs"]:
                system_prompt = """You are a technical documentation expert who creates clear, comprehensive, and user-friendly documentation.

Generate documentation that includes:
1. Clear overview and purpose
2. Installation/setup instructions
3. Usage examples
4. API reference (if applicable)
5. Configuration options
6. Troubleshooting guide
7. Contributing guidelines (if applicable)

Use proper markdown formatting and structure."""
                
            elif content_type.lower() in ["config", "configuration", "yaml", "json", "dockerfile"]:
                system_prompt = """You are a DevOps and configuration expert who creates robust, secure, and maintainable configuration files.

Generate configuration that follows:
1. Security best practices
2. Environment-specific settings
3. Proper validation and defaults
4. Clear documentation and comments
5. Scalability considerations
6. Monitoring and logging setup"""
                
            else:
                system_prompt = f"""You are an expert in creating {content_type} content. Generate high-quality, professional content that meets industry standards and best practices."""
            
            # Prepare specifications string
            specs_text = ""
            for key, value in specifications.items():
                specs_text += f"{key}: {value}\n"
            
            user_prompt = f"""Generate {content_type} content based on these specifications:

SPECIFICATIONS:
{specs_text}

CONTEXT:
{context_info}

Please generate complete, production-ready content that meets all specifications and follows best practices."""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            self.workflow_logger.log_agent_status("code_generator", "specialist", "Calling LLM for generation", "working", 60.0)
            
            # Generate content using LLM manager
            response = await self.llm_manager.generate_response(messages)
            
            # Log token usage
            if response.usage:
                cost = TokenTracker.calculate_cost(
                    response.model or "gpt-4",
                    response.usage.get("prompt_tokens", 0),
                    response.usage.get("completion_tokens", 0)
                )
                self.workflow_logger.log_performance_metric(
                    f"{content_type}_generation_llm_call",
                    1.0,
                    tokens_used=response.usage.get("total_tokens", 0),
                    estimated_cost=cost
                )
            
            self.workflow_logger.log_agent_status("code_generator", "specialist", "Processing generated content", "working", 80.0)
            
            # Extract code blocks if present
            generated_content = response.content
            code_blocks = []
            
            # Simple code block extraction
            import re
            code_pattern = r'```(\w+)?\n(.*?)\n```'
            matches = re.findall(code_pattern, generated_content, re.DOTALL)
            
            for language, code in matches:
                code_blocks.append({
                    "language": language or "text",
                    "content": code.strip()
                })
            
            self.workflow_logger.log_agent_status("code_generator", "specialist", "Assessing content quality", "working", 90.0)
            
            # Quality assessment (basic heuristics)
            quality_score = self._assess_content_quality(generated_content, content_type)
            
            generation_result = {
                "content_type": content_type,
                "generated_content": generated_content,
                "code_blocks": code_blocks,
                "specifications": specifications,
                "context": context,
                "quality_score": quality_score,
                "provider_info": self.llm_manager.get_provider_info(),
                "usage_stats": response.usage,
                "created_at": datetime.now().isoformat()
            }
            
            # Log generation completion
            content_length = len(generated_content)
            blocks_count = len(code_blocks)
            
            self.workflow_logger.log_agent_status("code_generator", "specialist", "Content generation completed", "completed", 100.0, {
                "content_length": content_length,
                "code_blocks_extracted": blocks_count,
                "quality_score": quality_score
            })
            
            # Log code generation progress
            self.workflow_logger.log_code_generation_progress(
                f"{content_type}_content",
                "completed",
                lines_generated=content_length // 50,  # Rough estimate
                metadata={
                    "quality_score": quality_score,
                    "code_blocks": blocks_count
                }
            )
            
            # Agent communication
            self.agent_client.broadcast(
                MessageType.STATUS_UPDATE,
                {
                    "message": f"{content_type} generation completed",
                    "content_length": content_length,
                    "quality_score": quality_score,
                    "code_blocks": blocks_count
                }
            )
            
            self.workflow_logger.log_workflow_step(step_id, f"Generate {content_type}", "code_generator", "completed", {
                "content_length": content_length,
                "quality_score": quality_score
            })
            
            logger.info(f"Content generation completed: {content_type} - {content_length} characters, quality: {quality_score}/10")
            return generation_result
            
        except Exception as e:
            self.workflow_logger.log_workflow_step(step_id, f"Generate {content_type}", "code_generator", "failed")
            self.workflow_logger.log_agent_status("code_generator", "specialist", "Content generation failed", "error", 0.0)
            self.workflow_logger.log_error_with_recovery(e, "Returning fallback content")
            
            # Report error
            self.agent_client.report_error({
                "operation": "cognitive_content_generation",
                "content_type": content_type,
                "error": str(e)
            })
            
            logger.error(f"Cognitive content generation failed: {e}")
            return {
                "error": str(e),
                "content_type": content_type,
                "specifications": specifications,
                "fallback_content": f"# {content_type.title()} Content\n\n# TODO: Implement based on specifications\n# Error occurred during generation: {str(e)}"
            }
    
    def _assess_content_quality(self, content: str, content_type: str) -> float:
        """Assess the quality of generated content using heuristics"""
        score = 5.0  # Base score
        
        # Length check
        if len(content) > 100:
            score += 1.0
        if len(content) > 500:
            score += 1.0
            
        # Code-specific checks
        if content_type.lower() in ["code", "python", "javascript", "typescript"]:
            if "def " in content or "function " in content or "class " in content:
                score += 1.0
            if "try:" in content or "catch" in content:
                score += 0.5
            if "#" in content or "//" in content or "/*" in content:
                score += 0.5
                
        # Documentation checks
        if content_type.lower() in ["documentation", "readme"]:
            if "##" in content or "###" in content:
                score += 1.0
            if "```" in content:
                score += 0.5
                
        return min(score, 10.0)
    
    @timed_operation("cognitive_git_operations")
    async def cognitive_git_operations(self, 
                                     repository_path: str, 
                                     operation: str,
                                     files: List[str] = None,
                                     commit_message: str = None,
                                     branch: str = None) -> Dict[str, Any]:
        """Smart Git workflow with intelligent commits and detailed logging"""
        
        step_id = f"git_{operation}"
        self.workflow_logger.log_workflow_step(step_id, f"Git {operation}", "git_manager", "in_progress")
        self.workflow_logger.log_agent_status("git_manager", "specialist", f"Executing git {operation}", "working", 10.0)
        
        # Agent communication
        self.agent_client.send_message(
            "git_manager",
            MessageType.TASK_HANDOFF,
            {
                "repository_path": repository_path,
                "operation": operation,
                "files": files,
                "commit_message": commit_message,
                "branch": branch,
                "action": "execute_git_operation"
            },
            priority=MessagePriority.NORMAL
        )
        
        try:
            self.workflow_logger.log_agent_status("git_manager", "specialist", "Preparing git commands", "working", 25.0)
            
            deploy_mode = os.getenv("DEPLOY", "false").lower() == "true"
            
            # Helper function to execute git commands
            async def execute_git_command(command: str) -> Dict[str, Any]:
                full_command = f"cd {repository_path} && git {command}"
                
                if deploy_mode:
                    result = await self.ssh_manager.execute_command(
                        self.default_ssh_config,
                        full_command
                    )
                    return {
                        "success": result.success,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "command": full_command
                    }
                else:
                    try:
                        result = subprocess.run(
                            full_command,
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=60
                        )
                        return {
                            "success": result.returncode == 0,
                            "stdout": result.stdout,
                            "stderr": result.stderr,
                            "command": full_command
                        }
                    except subprocess.TimeoutExpired:
                        return {
                            "success": False,
                            "stdout": "",
                            "stderr": "Command timed out",
                            "command": full_command
                        }
            
            operation_results = []
            
            if operation == "status":
                self.workflow_logger.log_agent_status("git_manager", "specialist", "Getting repository status", "working", 50.0)
                status_result = await execute_git_command("status --porcelain")
                operation_results.append(status_result)
                
            elif operation == "add":
                if files:
                    for i, file in enumerate(files):
                        progress = 50.0 + (i / len(files)) * 30.0
                        self.workflow_logger.log_agent_status("git_manager", "specialist", f"Adding file: {file}", "working", progress)
                        add_result = await execute_git_command(f"add {file}")
                        operation_results.append(add_result)
                else:
                    self.workflow_logger.log_agent_status("git_manager", "specialist", "Adding all changes", "working", 60.0)
                    add_result = await execute_git_command("add .")
                    operation_results.append(add_result)
                    
            elif operation == "commit":
                self.workflow_logger.log_agent_status("git_manager", "specialist", "Preparing commit", "working", 40.0)
                
                # Generate intelligent commit message if not provided
                if not commit_message:
                    self.workflow_logger.log_agent_status("git_manager", "specialist", "Generating commit message", "working", 60.0)
                    commit_message = await self._generate_intelligent_commit_message(repository_path)
                
                self.workflow_logger.log_agent_status("git_manager", "specialist", "Creating commit", "working", 80.0)
                commit_result = await execute_git_command(f'commit -m "{commit_message}"')
                operation_results.append(commit_result)
                
            elif operation == "push":
                self.workflow_logger.log_agent_status("git_manager", "specialist", "Pushing to remote", "working", 70.0)
                push_result = await execute_git_command("push")
                operation_results.append(push_result)
                
            elif operation == "branch":
                if branch:
                    self.workflow_logger.log_agent_status("git_manager", "specialist", f"Creating branch: {branch}", "working", 60.0)
                    branch_result = await execute_git_command(f"checkout -b {branch}")
                    operation_results.append(branch_result)
                else:
                    self.workflow_logger.log_agent_status("git_manager", "specialist", "Listing branches", "working", 60.0)
                    branch_result = await execute_git_command("branch -a")
                    operation_results.append(branch_result)
                    
            elif operation == "full_workflow":
                # Complete workflow: add, commit, push
                self.workflow_logger.log_agent_status("git_manager", "specialist", "Adding all changes", "working", 30.0)
                add_result = await execute_git_command("add .")
                operation_results.append(add_result)
                
                if add_result["success"]:
                    if not commit_message:
                        self.workflow_logger.log_agent_status("git_manager", "specialist", "Generating commit message", "working", 50.0)
                        commit_message = await self._generate_intelligent_commit_message(repository_path)
                    
                    self.workflow_logger.log_agent_status("git_manager", "specialist", "Creating commit", "working", 70.0)
                    commit_result = await execute_git_command(f'commit -m "{commit_message}"')
                    operation_results.append(commit_result)
                    
                    if commit_result["success"]:
                        self.workflow_logger.log_agent_status("git_manager", "specialist", "Pushing changes", "working", 90.0)
                        push_result = await execute_git_command("push")
                        operation_results.append(push_result)
            
            # Get final repository status
            final_status = await execute_git_command("status --porcelain")
            
            git_operation_result = {
                "operation": operation,
                "repository_path": repository_path,
                "files": files,
                "commit_message": commit_message,
                "branch": branch,
                "operation_results": operation_results,
                "final_status": final_status,
                "success": all(result.get("success", False) for result in operation_results),
                "created_at": datetime.now().isoformat()
            }
            
            # Log completion
            success = git_operation_result['success']
            self.workflow_logger.log_agent_status("git_manager", "specialist", 
                                                f"Git {operation} {'completed' if success else 'failed'}", 
                                                "completed" if success else "error", 
                                                100.0 if success else 0.0)
            
            # Log git operation details
            self.workflow_logger.log_git_operation(
                operation=operation,
                files=files or [],
                commit_message=commit_message,
                success=success,
                details=f"Executed {len(operation_results)} git commands"
            )
            
            # Agent communication
            self.agent_client.broadcast(
                MessageType.STATUS_UPDATE,
                {
                    "message": f"Git {operation} {'completed' if success else 'failed'}",
                    "operation": operation,
                    "success": success,
                    "commit_message": commit_message
                }
            )
            
            self.workflow_logger.log_workflow_step(step_id, f"Git {operation}", "git_manager", 
                                                 "completed" if success else "failed", {
                                                     "commands_executed": len(operation_results),
                                                     "commit_message": commit_message
                                                 })
            
            logger.info(f"Git operation completed: {operation} - {'SUCCESS' if success else 'FAILED'}")
            return git_operation_result
            
        except Exception as e:
            self.workflow_logger.log_workflow_step(step_id, f"Git {operation}", "git_manager", "failed")
            self.workflow_logger.log_agent_status("git_manager", "specialist", f"Git {operation} failed", "error", 0.0)
            self.workflow_logger.log_error_with_recovery(e, "Git operation failed")
            
            # Report error
            self.agent_client.report_error({
                "operation": "cognitive_git_operations",
                "git_operation": operation,
                "error": str(e),
                "repository_path": repository_path
            })
            
            logger.error(f"Cognitive git operation failed: {e}")
            return {
                "operation": operation,
                "repository_path": repository_path,
                "error": str(e),
                "success": False,
                "created_at": datetime.now().isoformat()
            }
    
    async def _generate_intelligent_commit_message(self, repository_path: str) -> str:
        """Generate intelligent commit message based on git diff with logging"""
        try:
            self.workflow_logger.log_agent_status("git_manager", "specialist", "Analyzing git diff for commit message", "working", 65.0)
            
            deploy_mode = os.getenv("DEPLOY", "false").lower() == "true"
            
            # Get git diff
            if deploy_mode:
                diff_result = await self.ssh_manager.execute_command(
                    self.default_ssh_config,
                    f"cd {repository_path} && git diff --cached"
                )
                git_diff = diff_result.stdout if diff_result.success else ""
            else:
                result = subprocess.run(
                    f"cd {repository_path} && git diff --cached",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                git_diff = result.stdout if result.returncode == 0 else ""
            
            if not git_diff.strip():
                return "Update project files"
            
            # Use LLM to generate commit message
            system_prompt = """You are an expert at writing clear, concise git commit messages. 

Generate a commit message that:
1. Follows conventional commit format when appropriate
2. Is concise but descriptive
3. Focuses on what changed and why
4. Uses imperative mood (e.g., "Add feature" not "Added feature")
5. Is under 72 characters for the first line

Return only the commit message, nothing else."""
            
            user_prompt = f"""Based on this git diff, generate an appropriate commit message:

{git_diff[:2000]}  # Limit diff size for token efficiency
"""
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = await self.llm_manager.generate_response(messages)
            commit_message = response.content.strip().strip('"').strip("'")
            
            # Fallback to simple message if generation fails
            if not commit_message or len(commit_message) > 100:
                commit_message = "Update project files"
            
            self.workflow_logger.log_agent_status("git_manager", "specialist", f"Generated commit message: {commit_message[:50]}...", "working", 75.0)
            return commit_message
            
        except Exception as e:
            self.workflow_logger.log_error_with_recovery(e, "Using fallback commit message")
            logger.warning(f"Failed to generate intelligent commit message: {e}")
            return "Update project files"
    
    # ============================================================================
    # WORKFLOW EXECUTION WITH ENHANCED LOGGING
    # ============================================================================
    
    async def execute_complex_development_task(self, task_description: str, project_path: str = None) -> Dict[str, Any]:
        """Execute a complex development task with full workflow visibility"""
        
        self.workflow_logger.update_phase("Task Execution Started", 0.0)
        
        try:
            # Phase 1: Task Analysis and Breakdown
            self.workflow_logger.update_phase("Analyzing Task", 10.0)
            breakdown_result = await self.cognitive_task_breakdown(task_description)
            
            if "error" in breakdown_result:
                raise Exception(f"Task breakdown failed: {breakdown_result['error']}")
            
            subtasks = breakdown_result.get("breakdown_data", {}).get("subtasks", [])
            
            # Phase 2: Project Setup
            self.workflow_logger.update_phase("Setting Up Project", 20.0)
            if not project_path:
                project_path = f"/tmp/automotas_project_{self.workflow_id}"
                os.makedirs(project_path, exist_ok=True)
            
            # Phase 3: Content Generation
            self.workflow_logger.update_phase("Generating Content", 30.0)
            generated_files = []
            
            for i, subtask in enumerate(subtasks):
                progress = 30.0 + (i / len(subtasks)) * 40.0
                self.workflow_logger.update_phase(f"Processing Subtask: {subtask.get('description', 'Unknown')[:50]}...", progress)
                
                # Determine content type based on subtask
                content_type = self._determine_content_type(subtask)
                
                # Generate content for subtask
                generation_result = await self.cognitive_content_generation(
                    content_type=content_type,
                    specifications=subtask,
                    context={"project_path": project_path, "task_description": task_description}
                )
                
                if "error" not in generation_result:
                    # Write generated content to files
                    file_path = self._determine_file_path(subtask, content_type)
                    write_result = await self._write_file_to_project(
                        file_path=file_path,
                        content=generation_result["generated_content"],
                        project_path=project_path
                    )
                    
                    if write_result["success"]:
                        generated_files.append({
                            "file_path": file_path,
                            "content_type": content_type,
                            "subtask_id": subtask.get("id"),
                            "size_bytes": write_result["file_size"]
                        })
            
            # Phase 4: Git Operations
            self.workflow_logger.update_phase("Version Control Operations", 80.0)
            
            # Initialize git repository if needed
            git_init_result = await self.cognitive_git_operations(
                repository_path=project_path,
                operation="status"
            )
            
            # Add and commit all generated files
            commit_message = f"Implement: {task_description[:50]}..."
            git_workflow_result = await self.cognitive_git_operations(
                repository_path=project_path,
                operation="full_workflow",
                files=[f["file_path"] for f in generated_files],
                commit_message=commit_message
            )
            
            # Phase 5: Final Validation
            self.workflow_logger.update_phase("Final Validation", 95.0)
            
            # Validate generated files
            validation_results = []
            for file_info in generated_files:
                full_path = os.path.join(project_path, file_info["file_path"])
                if os.path.exists(full_path):
                    validation_results.append({
                        "file": file_info["file_path"],
                        "exists": True,
                        "size": os.path.getsize(full_path)
                    })
            
            # Complete workflow
            self.workflow_logger.update_phase("Completed", 100.0)
            
            execution_result = {
                "success": True,
                "workflow_id": self.workflow_id,
                "task_description": task_description,
                "project_path": project_path,
                "subtasks_processed": len(subtasks),
                "files_generated": len(generated_files),
                "generated_files": generated_files,
                "git_operations": git_workflow_result,
                "validation_results": validation_results,
                "breakdown_data": breakdown_result,
                "execution_summary": self.workflow_logger.get_workflow_summary()
            }
            
            # Log successful completion
            self.workflow_logger.log_workflow_completion(
                success=True,
                final_message=f"Successfully completed complex development task: {len(generated_files)} files generated"
            )
            
            return execution_result
            
        except Exception as e:
            self.workflow_logger.log_workflow_completion(
                success=False,
                final_message=f"Task execution failed: {str(e)}"
            )
            
            logger.error(f"Complex development task execution failed: {e}")
            return {
                "success": False,
                "workflow_id": self.workflow_id,
                "error": str(e),
                "task_description": task_description,
                "execution_summary": self.workflow_logger.get_workflow_summary()
            }
    
    def _determine_content_type(self, subtask: Dict[str, Any]) -> str:
        """Determine content type based on subtask description"""
        description = subtask.get("description", "").lower()
        
        if any(keyword in description for keyword in ["api", "endpoint", "route", "server"]):
            return "python"
        elif any(keyword in description for keyword in ["frontend", "ui", "component", "react"]):
            return "javascript"
        elif any(keyword in description for keyword in ["test", "testing", "unit test"]):
            return "python"
        elif any(keyword in description for keyword in ["config", "configuration", "docker"]):
            return "config"
        elif any(keyword in description for keyword in ["readme", "documentation", "docs"]):
            return "documentation"
        else:
            return "code"
    
    def _determine_file_path(self, subtask: Dict[str, Any], content_type: str) -> str:
        """Determine appropriate file path based on subtask and content type"""
        description = subtask.get("description", "").lower()
        subtask_id = subtask.get("id", 1)
        
        if content_type == "python":
            if "main" in description or "app" in description:
                return "main.py"
            elif "test" in description:
                return f"test_{subtask_id}.py"
            else:
                return f"module_{subtask_id}.py"
        elif content_type == "javascript":
            if "main" in description or "app" in description:
                return "app.js"
            else:
                return f"component_{subtask_id}.js"
        elif content_type == "config":
            if "docker" in description:
                return "Dockerfile"
            else:
                return "config.yaml"
        elif content_type == "documentation":
            return "README.md"
        else:
            return f"file_{subtask_id}.txt"
    
    async def _write_file_to_project(self, 
                                   file_path: str, 
                                   content: str, 
                                   project_path: str,
                                   backup: bool = True) -> Dict[str, Any]:
        """Write generated content to project files with backup and validation"""
        try:
            self.workflow_logger.log_code_generation_progress(file_path, "started")
            
            # Construct full path
            full_path = os.path.join(project_path, file_path)
            directory = os.path.dirname(full_path)
            
            # Create directory if it doesn't exist
            if directory:
                os.makedirs(directory, exist_ok=True)
            
            # Backup existing file if requested
            backup_path = None
            if backup and os.path.exists(full_path):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{full_path}.backup_{timestamp}"
                
                deploy_mode = os.getenv("DEPLOY", "false").lower() == "true"
                
                if deploy_mode:
                    # Use SSH for deployment
                    backup_result = await self.ssh_manager.execute_command(
                        self.default_ssh_config,
                        f"cp {full_path} {backup_path}"
                    )
                    if not backup_result.success:
                        logger.warning(f"Failed to create backup: {backup_result.stderr}")
                else:
                    # Use local file operations
                    import shutil
                    shutil.copy2(full_path, backup_path)
                    logger.info(f"Created backup: {backup_path}")
            
            # Write content to file
            deploy_mode = os.getenv("DEPLOY", "false").lower() == "true"
            
            if deploy_mode:
                # Use SSH for deployment
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.tmp') as tmp_file:
                    tmp_file.write(content)
                    tmp_file_path = tmp_file.name
                
                # Upload file via SSH
                upload_result = await self.ssh_manager.upload_file(
                    self.default_ssh_config,
                    tmp_file_path,
                    full_path
                )
                
                # Clean up temporary file
                os.unlink(tmp_file_path)
                
                if not upload_result.success:
                    raise Exception(f"Failed to upload file via SSH: {upload_result.stderr}")
                    
            else:
                # Use local file operations
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Validate file was written correctly
            file_size = len(content.encode('utf-8'))
            lines_count = content.count('\n') + 1
            
            self.workflow_logger.log_code_generation_progress(
                file_path, 
                "completed", 
                lines_generated=lines_count,
                metadata={"file_size_bytes": file_size}
            )
            
            write_result = {
                "file_path": file_path,
                "full_path": full_path,
                "backup_path": backup_path,
                "file_size": file_size,
                "content_length": len(content),
                "lines_count": lines_count,
                "success": True,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"File write completed: {file_path} ({file_size} bytes, {lines_count} lines)")
            return write_result
            
        except Exception as e:
            self.workflow_logger.log_code_generation_progress(file_path, "failed")
            self.workflow_logger.log_error_with_recovery(e, f"Failed to write file: {file_path}")
            
            logger.error(f"File write failed for {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "success": False,
                "created_at": datetime.now().isoformat()
            }
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get comprehensive workflow status for UI display"""
        return {
            "workflow_id": self.workflow_id,
            "workflow_summary": self.workflow_logger.get_workflow_summary(),
            "agent_communications": self.comm_hub.get_recent_messages(20),
            "communication_stats": self.comm_hub.get_communication_stats(),
            "active_agents": self.comm_hub.get_all_agents()
        }

# Example usage and testing
if __name__ == "__main__":
    async def test_enhanced_orchestrator():
        """Test the enhanced orchestrator with a complex development task"""
        
        orchestrator = EnhancedTwoTierOrchestrator()
        
        # Test with a complex development task
        complex_task = """
        Build a REST API with JWT authentication, user registration/login, protected routes, and comprehensive tests.
        
        Requirements:
        - FastAPI framework
        - JWT token authentication
        - User registration and login endpoints
        - Protected routes that require authentication
        - SQLite database for user storage
        - Password hashing with bcrypt
        - Comprehensive unit tests
        - API documentation
        - Docker configuration
        - Requirements.txt file
        """
        
        print(f"🚀 Starting complex development task...")
        print(f"Task: {complex_task[:100]}...")
        
        result = await orchestrator.execute_complex_development_task(complex_task)
        
        if result["success"]:
            print(f"✅ Task completed successfully!")
            print(f"📁 Project path: {result['project_path']}")
            print(f"📄 Files generated: {result['files_generated']}")
            print(f"🔧 Subtasks processed: {result['subtasks_processed']}")
            
            # Display generated files
            print("\n📋 Generated Files:")
            for file_info in result["generated_files"]:
                print(f"  - {file_info['file_path']} ({file_info['content_type']}, {file_info['size_bytes']} bytes)")
            
        else:
            print(f"❌ Task failed: {result['error']}")
        
        # Display workflow summary
        summary = result["execution_summary"]
        print(f"\n📊 Workflow Summary:")
        print(f"  - Duration: {summary['total_duration_seconds']:.2f} seconds")
        print(f"  - Steps completed: {summary['completed_steps']}/{summary['total_steps']}")
        print(f"  - Agents used: {summary['total_agents']}")
        print(f"  - Total tokens: {summary['total_tokens_used']}")
        print(f"  - Estimated cost: ${summary['total_estimated_cost']:.4f}")
        
        return result
    
    # Run the test
    import asyncio
    asyncio.run(test_enhanced_orchestrator())
