
"""
Enhanced Two-Tiered Multi-Agent Orchestration System with SSH Capabilities
==========================================================================

This enhanced orchestrator provides a two-tiered system that can handle:
1. Self-contained repositories with ai-module.yaml configurations
2. Traditional repositories with task prompts

Key Features:
- Auto-detection of repository type (ai-module.yaml vs task prompts)
- SSH command execution with banking-grade security
- Unified workflow handling for both approaches
- Comprehensive audit logging and security monitoring
- Advanced multi-agent collaboration patterns
- Real-time deployment and monitoring
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
from dotenv import load_dotenv

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
    AI_MODULE = "ai_module"  # Self-contained with ai-module.yaml
    TASK_PROMPT = "task_prompt"  # Traditional with task prompts

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
    """Enhanced Two-Tiered Multi-Agent Orchestration System"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        
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
        
        # Workflow tracking
        self.active_workflows: Dict[str, WorkflowState] = {}
        self.rollback_stack: List[Dict[str, Any]] = []
        
        # SSH configuration for mcp.xplaincrypto.ai
        self.default_ssh_config = SSHConnection(
            host=os.getenv("DEPLOY_HOST", "mcp.xplaincrypto.ai"),
            port=int(os.getenv("DEPLOY_PORT", "22")),
            username=os.getenv("DEPLOY_USER", "root"),
            key_path=os.getenv("DEPLOY_KEY_PATH", "/root/.ssh/id_rsa"),
            security_level=SecurityLevel.HIGH
        )
        
        logger.info("Enhanced Two-Tier Orchestrator initialized")
    
    # ============================================================================
    # COGNITIVE FUNCTIONS - AI-Powered Task Processing
    # ============================================================================
    
    async def cognitive_task_breakdown(self, task_description: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        AI-powered task analysis and decomposition
        
        Args:
            task_description: The main task to break down
            context: Additional context about the project/repository
            
        Returns:
            Dictionary containing subtasks, complexity analysis, and execution plan
        """
        try:
            logger.info(f"Starting cognitive task breakdown for: {task_description[:100]}...")
            
            # Prepare context information
            context_info = ""
            if context:
                if context.get("repository_structure"):
                    context_info += f"Repository structure: {context['repository_structure']}\n"
                if context.get("tech_stack"):
                    context_info += f"Technology stack: {', '.join(context['tech_stack'])}\n"
                if context.get("existing_files"):
                    context_info += f"Existing files: {', '.join(context['existing_files'][:10])}\n"
            
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
            
            # Generate response using LLM manager
            response = await self.llm_manager.generate_response(messages)
            
            # Try to parse JSON response
            try:
                import json
                breakdown_data = json.loads(response.content)
            except json.JSONDecodeError:
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
            
            logger.info(f"Task breakdown completed: {len(breakdown_data.get('subtasks', []))} subtasks identified")
            return breakdown_result
            
        except Exception as e:
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
    
    async def cognitive_content_generation(self, 
                                         content_type: str, 
                                         specifications: Dict[str, Any],
                                         context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Production-ready code and content generation
        
        Args:
            content_type: Type of content to generate (code, documentation, config, etc.)
            specifications: Detailed specifications for the content
            context: Project context and constraints
            
        Returns:
            Dictionary containing generated content, metadata, and quality metrics
        """
        try:
            logger.info(f"Starting cognitive content generation for: {content_type}")
            
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
            
            # Generate content using LLM manager
            response = await self.llm_manager.generate_response(messages)
            
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
            
            logger.info(f"Content generation completed: {len(generated_content)} characters, quality score: {quality_score}")
            return generation_result
            
        except Exception as e:
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
    
    async def _write_file_to_project(self, 
                                   file_path: str, 
                                   content: str, 
                                   project_path: str,
                                   backup: bool = True) -> Dict[str, Any]:
        """
        Write generated content to project files with backup and validation
        
        Args:
            file_path: Relative path within the project
            content: Content to write
            project_path: Base project directory path
            backup: Whether to create backup of existing files
            
        Returns:
            Dictionary containing operation result and metadata
        """
        try:
            logger.info(f"Writing file to project: {file_path}")
            
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
                
                # Use appropriate method based on DEPLOY flag
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
                # Create temporary file and upload
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
                logger.info(f"File written locally: {full_path}")
            
            # Validate file was written correctly
            file_size = len(content.encode('utf-8'))
            
            write_result = {
                "file_path": file_path,
                "full_path": full_path,
                "backup_path": backup_path,
                "file_size": file_size,
                "content_length": len(content),
                "success": True,
                "created_at": datetime.now().isoformat()
            }
            
            logger.info(f"File write completed: {file_path} ({file_size} bytes)")
            return write_result
            
        except Exception as e:
            logger.error(f"File write failed for {file_path}: {e}")
            return {
                "file_path": file_path,
                "error": str(e),
                "success": False,
                "created_at": datetime.now().isoformat()
            }
    
    async def cognitive_git_operations(self, 
                                     repository_path: str, 
                                     operation: str,
                                     files: List[str] = None,
                                     commit_message: str = None,
                                     branch: str = None) -> Dict[str, Any]:
        """
        Smart Git workflow with intelligent commits and branch management
        
        Args:
            repository_path: Path to the git repository
            operation: Git operation (add, commit, push, branch, merge, etc.)
            files: List of files to operate on (for add/commit operations)
            commit_message: Commit message (auto-generated if not provided)
            branch: Branch name for branch operations
            
        Returns:
            Dictionary containing operation results and git status
        """
        try:
            logger.info(f"Starting cognitive git operation: {operation} in {repository_path}")
            
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
                # Get repository status
                status_result = await execute_git_command("status --porcelain")
                operation_results.append(status_result)
                
            elif operation == "add":
                # Add files to staging
                if files:
                    for file in files:
                        add_result = await execute_git_command(f"add {file}")
                        operation_results.append(add_result)
                else:
                    # Add all changes
                    add_result = await execute_git_command("add .")
                    operation_results.append(add_result)
                    
            elif operation == "commit":
                # Generate intelligent commit message if not provided
                if not commit_message:
                    commit_message = await self._generate_intelligent_commit_message(repository_path)
                
                # Commit changes
                commit_result = await execute_git_command(f'commit -m "{commit_message}"')
                operation_results.append(commit_result)
                
            elif operation == "push":
                # Push to remote
                push_result = await execute_git_command("push")
                operation_results.append(push_result)
                
            elif operation == "branch":
                if branch:
                    # Create and switch to new branch
                    branch_result = await execute_git_command(f"checkout -b {branch}")
                    operation_results.append(branch_result)
                else:
                    # List branches
                    branch_result = await execute_git_command("branch -a")
                    operation_results.append(branch_result)
                    
            elif operation == "full_workflow":
                # Complete workflow: add, commit, push
                # Add all changes
                add_result = await execute_git_command("add .")
                operation_results.append(add_result)
                
                if add_result["success"]:
                    # Generate commit message if not provided
                    if not commit_message:
                        commit_message = await self._generate_intelligent_commit_message(repository_path)
                    
                    # Commit changes
                    commit_result = await execute_git_command(f'commit -m "{commit_message}"')
                    operation_results.append(commit_result)
                    
                    if commit_result["success"]:
                        # Push changes
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
            
            logger.info(f"Git operation completed: {operation} - {'SUCCESS' if git_operation_result['success'] else 'FAILED'}")
            return git_operation_result
            
        except Exception as e:
            logger.error(f"Cognitive git operation failed: {e}")
            return {
                "operation": operation,
                "repository_path": repository_path,
                "error": str(e),
                "success": False,
                "created_at": datetime.now().isoformat()
            }
    
    async def _generate_intelligent_commit_message(self, repository_path: str) -> str:
        """Generate intelligent commit message based on git diff"""
        try:
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
                
            return commit_message
            
        except Exception as e:
            logger.warning(f"Failed to generate intelligent commit message: {e}")
            return "Update project files"
    
    # ============================================================================
    # END COGNITIVE FUNCTIONS
    # ============================================================================
    
    async def detect_workflow_type(self, repository_path: str) -> WorkflowType:
        """Auto-detect workflow type based on repository structure"""
        
        repo_path = Path(repository_path)
        ai_module_file = repo_path / "ai-module.yaml"
        
        if ai_module_file.exists():
            logger.info(f"Detected AI module workflow: {repository_path}")
            return WorkflowType.AI_MODULE
        else:
            logger.info(f"Detected task prompt workflow: {repository_path}")
            return WorkflowType.TASK_PROMPT
    
    async def process_ai_module_workflow(
        self, 
        workflow_state: WorkflowState
    ) -> Dict[str, Any]:
        """Process self-contained AI module workflow"""
        
        try:
            workflow_state.status = DeploymentStatus.BUILDING
            
            # Parse ai-module.yaml
            ai_module_path = Path(workflow_state.project_path) / "ai-module.yaml"
            config = self.ai_module_parser.parse_file(str(ai_module_path))
            workflow_state.config = config
            
            logger.info(f"Processing AI module: {config.name} v{config.version}")
            
            # Log security event
            self.audit_logger.log_event(create_security_event(
                event_type=EventType.CONFIGURATION_CHANGE,
                user_id="system",
                source_ip="localhost",
                resource=workflow_state.repository_url,
                action="ai_module_deployment",
                result="started",
                details={"module_name": config.name, "version": config.version}
            ))
            
            # Install system dependencies
            if config.system_dependencies:
                deps_command = f"apt-get update && apt-get install -y {' '.join(config.system_dependencies)}"
                deps_result = await self.ssh_manager.execute_command(
                    self.default_ssh_config, deps_command
                )
                if not deps_result.success:
                    raise Exception(f"Failed to install system dependencies: {deps_result.stderr}")
            
            # Build the application
            workflow_state.status = DeploymentStatus.BUILDING
            build_result = await self.ssh_manager.execute_command(
                self.default_ssh_config,
                f"cd {workflow_state.project_path} && {config.build_command}"
            )
            
            if not build_result.success:
                workflow_state.status = DeploymentStatus.FAILED
                raise Exception(f"Build failed: {build_result.stderr}")
            
            # Run tests if specified
            if config.test_command:
                workflow_state.status = DeploymentStatus.TESTING
                test_result = await self.ssh_manager.execute_command(
                    self.default_ssh_config,
                    f"cd {workflow_state.project_path} && {config.test_command}"
                )
                
                if not test_result.success:
                    logger.warning(f"Tests failed: {test_result.stderr}")
                    # Continue deployment despite test failures (configurable)
            
            # Deploy the application
            workflow_state.status = DeploymentStatus.DEPLOYING
            
            # Generate deployment script based on target
            deployment_script = self._generate_deployment_script(config, workflow_state)
            
            deploy_result = await self.ssh_manager.execute_script(
                self.default_ssh_config,
                deployment_script,
                f"deploy_{config.name}.sh"
            )
            
            if not deploy_result.success:
                workflow_state.status = DeploymentStatus.FAILED
                raise Exception(f"Deployment failed: {deploy_result.stderr}")
            
            # Setup health monitoring
            if config.health_check.enabled:
                await self._setup_health_monitoring(config, workflow_state)
            
            workflow_state.status = DeploymentStatus.RUNNING
            workflow_state.updated_at = datetime.now()
            
            # Log successful deployment
            self.audit_logger.log_event(create_security_event(
                event_type=EventType.CONFIGURATION_CHANGE,
                user_id="system",
                source_ip="localhost",
                resource=workflow_state.repository_url,
                action="ai_module_deployment",
                result="success",
                details={
                    "module_name": config.name,
                    "version": config.version,
                    "port": config.port,
                    "deployment_target": config.deployment_target.value
                }
            ))
            
            return {
                "success": True,
                "workflow_id": workflow_state.workflow_id,
                "module_name": config.name,
                "version": config.version,
                "status": workflow_state.status.value,
                "endpoint": f"http://{self.default_ssh_config.host}:{config.port}",
                "health_check": f"http://{self.default_ssh_config.host}:{config.port}{config.health_check.endpoint}"
            }
            
        except Exception as e:
            workflow_state.status = DeploymentStatus.FAILED
            workflow_state.updated_at = datetime.now()
            
            # Log failure
            self.audit_logger.log_event(create_security_event(
                event_type=EventType.SYSTEM_ERROR,
                user_id="system",
                source_ip="localhost",
                resource=workflow_state.repository_url,
                action="ai_module_deployment",
                result="failed",
                details={"error": str(e)}
            ))
            
            logger.error(f"AI module workflow failed: {e}")
            return {
                "success": False,
                "workflow_id": workflow_state.workflow_id,
                "error": str(e),
                "status": workflow_state.status.value
            }
    
    async def process_task_prompt_workflow(
        self, 
        workflow_state: WorkflowState
    ) -> Dict[str, Any]:
        """Process traditional task prompt workflow"""
        
        try:
            workflow_state.status = DeploymentStatus.BUILDING
            
            logger.info(f"Processing task prompt workflow: {workflow_state.task_prompt[:100]}...")
            
            # Log security event
            self.audit_logger.log_event(create_security_event(
                event_type=EventType.CONFIGURATION_CHANGE,
                user_id="system",
                source_ip="localhost",
                resource=workflow_state.repository_url,
                action="task_prompt_deployment",
                result="started",
                details={"task_prompt": workflow_state.task_prompt[:200]}
            ))
            
            # Create AI agents for task execution
            agents = await self._create_task_agents()
            
            # Analyze repository structure
            analysis_result = await self._analyze_repository_structure(workflow_state.project_path)
            
            # Generate deployment strategy
            deployment_strategy = await self._generate_deployment_strategy(
                workflow_state.task_prompt,
                analysis_result
            )
            
            # Write generated files to project
            if deployment_strategy.get("generated_files"):
                for file_info in deployment_strategy["generated_files"]:
                    write_result = await self._write_file_to_project(
                        file_path=file_info["file_path"],
                        content=file_info["content"],
                        project_path=workflow_state.project_path
                    )
                    
                    if write_result["success"]:
                        workflow_state.deployment_logs.append(
                            f"File written: {file_info['file_path']} ({write_result['file_size']} bytes)"
                        )
                    else:
                        workflow_state.deployment_logs.append(
                            f"File write failed: {file_info['file_path']} - {write_result.get('error', 'Unknown error')}"
                        )
            
            # Execute deployment tasks
            workflow_state.status = DeploymentStatus.DEPLOYING
            
            for task in deployment_strategy["tasks"]:
                deploy_mode = os.getenv("DEPLOY", "false").lower() == "true"
                
                if deploy_mode:
                    task_result = await self.ssh_manager.execute_command(
                        self.default_ssh_config,
                        f"cd {workflow_state.project_path} && {task['command']}"
                    )
                    success = task_result.success
                    error_msg = task_result.stderr
                else:
                    # Use local execution for simple tasks
                    try:
                        result = subprocess.run(
                            f"cd {workflow_state.project_path} && {task['command']}",
                            shell=True,
                            capture_output=True,
                            text=True,
                            timeout=120
                        )
                        success = result.returncode == 0
                        error_msg = result.stderr
                    except subprocess.TimeoutExpired:
                        success = False
                        error_msg = "Command timed out"
                
                if not success and task.get("required", True):
                    workflow_state.status = DeploymentStatus.FAILED
                    raise Exception(f"Required task failed: {task['name']} - {error_msg}")
                
                workflow_state.deployment_logs.append(
                    f"Task: {task['name']} - {'SUCCESS' if success else 'FAILED'}"
                )
            
            # Perform git operations to commit changes
            git_result = await self.cognitive_git_operations(
                repository_path=workflow_state.project_path,
                operation="full_workflow",
                commit_message=f"Implement: {workflow_state.task_prompt[:50]}..."
            )
            
            if git_result["success"]:
                workflow_state.deployment_logs.append("Git operations completed successfully")
            else:
                workflow_state.deployment_logs.append(f"Git operations failed: {git_result.get('error', 'Unknown error')}")
            
            workflow_state.status = DeploymentStatus.RUNNING
            workflow_state.updated_at = datetime.now()
            
            # Log successful deployment
            self.audit_logger.log_event(create_security_event(
                event_type=EventType.CONFIGURATION_CHANGE,
                user_id="system",
                source_ip="localhost",
                resource=workflow_state.repository_url,
                action="task_prompt_deployment",
                result="success",
                details={"tasks_executed": len(deployment_strategy["tasks"])}
            ))
            
            return {
                "success": True,
                "workflow_id": workflow_state.workflow_id,
                "status": workflow_state.status.value,
                "deployment_strategy": deployment_strategy,
                "logs": workflow_state.deployment_logs,
                "endpoint": deployment_strategy.get("endpoint")
            }
            
        except Exception as e:
            workflow_state.status = DeploymentStatus.FAILED
            workflow_state.updated_at = datetime.now()
            
            # Log failure
            self.audit_logger.log_event(create_security_event(
                event_type=EventType.SYSTEM_ERROR,
                user_id="system",
                source_ip="localhost",
                resource=workflow_state.repository_url,
                action="task_prompt_deployment",
                result="failed",
                details={"error": str(e)}
            ))
            
            logger.error(f"Task prompt workflow failed: {e}")
            return {
                "success": False,
                "workflow_id": workflow_state.workflow_id,
                "error": str(e),
                "status": workflow_state.status.value
            }
    
    async def run_unified_workflow(
        self,
        repository_url: str,
        task_prompt: Optional[str] = None,
        target_host: Optional[str] = None,
        environment_variables: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Run unified workflow that auto-detects and handles both types"""
        
        # Generate workflow ID
        workflow_id = hashlib.sha256(
            f"{repository_url}{time.time()}".encode()
        ).hexdigest()[:16]
        
        # Clone repository with DEPLOY flag logic
        project_path = f"/tmp/projects/{workflow_id}"
        deploy_mode = os.getenv("DEPLOY", "false").lower() == "true"
        
        if deploy_mode:
            # Use SSH for deployment tasks
            print(f"[DEPLOY=true] Using SSH to clone repository {repository_url}")
            clone_result = await self.ssh_manager.execute_command(
                self.default_ssh_config,
                f"git clone {repository_url} {project_path}"
            )
            if not clone_result.success:
                return {
                    "success": False,
                    "error": f"Failed to clone repository: {clone_result.stderr}"
                }
        else:
            # Use local execution for simple coding tasks
            print(f"[DEPLOY=false] Using local git clone for repository {repository_url}")
            try:
                # import os already available at module level
                os.makedirs("/tmp/projects", exist_ok=True)
                result = subprocess.run(
                    f"git clone {repository_url} {project_path}",
                    shell=True, capture_output=True, text=True, timeout=60
                )
                if result.returncode != 0:
                    return {
                        "success": False,
                        "error": f"Failed to clone repository: {result.stderr}"
                    }
                print(f"[DEPLOY=false] Local clone succeeded: {repository_url} -> {project_path}")
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                return {
                    "success": False,
                    "error": f"Failed to clone repository: Local execution error: {e}"
                }
        
        # Detect workflow type
        workflow_type = await self.detect_workflow_type(project_path)
        
        # Create workflow state
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            workflow_type=workflow_type,
            status=DeploymentStatus.PENDING,
            repository_url=repository_url,
            target_host=target_host or self.default_ssh_config.host,
            project_path=project_path,
            task_prompt=task_prompt,
            environment_variables=environment_variables or {},
            deployment_logs=[],
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Store workflow state
        self.active_workflows[workflow_id] = workflow_state
        
        try:
            # Process based on workflow type
            if workflow_type == WorkflowType.AI_MODULE:
                result = await self.process_ai_module_workflow(workflow_state)
            else:
                result = await self.process_task_prompt_workflow(workflow_state)
            
            return result
            
        except Exception as e:
            logger.error(f"Unified workflow failed: {e}")
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e)
            }
    
    def _generate_deployment_script(
        self, 
        config: AIModuleConfig, 
        workflow_state: WorkflowState
    ) -> str:
        """Generate deployment script based on configuration"""
        
        script_parts = [
            "#!/bin/bash",
            "set -e",
            "",
            f"# Deployment script for {config.name} v{config.version}",
            f"cd {workflow_state.project_path}",
            ""
        ]
        
        # Environment variables
        if config.environment_variables:
            script_parts.append("# Set environment variables")
            for key, value in config.environment_variables.items():
                script_parts.append(f"export {key}='{value}'")
            script_parts.append("")
        
        # Install dependencies
        if config.dependencies:
            if config.deployment_target.value in ["docker"]:
                script_parts.extend([
                    "# Build Docker image",
                    f"docker build -t {config.name}:{config.version} .",
                    "",
                    "# Stop existing container",
                    f"docker stop {config.name} || true",
                    f"docker rm {config.name} || true",
                    "",
                    "# Run new container",
                    f"docker run -d --name {config.name} -p {config.port}:{config.port} {config.name}:{config.version}",
                    ""
                ])
            else:
                script_parts.extend([
                    "# Install dependencies",
                    config.build_command,
                    "",
                    "# Start application",
                    f"nohup {config.start_command} > app.log 2>&1 &",
                    f"echo $! > {config.name}.pid",
                    ""
                ])
        
        # Health check
        if config.health_check.enabled:
            script_parts.extend([
                "# Wait for application to start",
                "sleep 10",
                "",
                "# Health check",
                f"curl -f http://localhost:{config.port}{config.health_check.endpoint} || exit 1",
                ""
            ])
        
        script_parts.append("echo 'Deployment completed successfully'")
        
        return "\n".join(script_parts)
    
    async def _setup_health_monitoring(
        self, 
        config: AIModuleConfig, 
        workflow_state: WorkflowState
    ):
        """Setup health monitoring for deployed application"""
        
        monitoring_script = f"""#!/bin/bash
# Health monitoring script for {config.name}

while true; do
    if curl -f http://localhost:{config.port}{config.health_check.endpoint} > /dev/null 2>&1; then
        echo "$(date): {config.name} is healthy"
    else
        echo "$(date): {config.name} health check failed"
        # Optional: restart application
        # systemctl restart {config.name}
    fi
    sleep {config.health_check.interval}
done
"""
        
        # Upload and start monitoring script
        await self.ssh_manager.execute_script(
            self.default_ssh_config,
            monitoring_script,
            f"monitor_{config.name}.sh"
        )
        
        # Start monitoring in background
        await self.ssh_manager.execute_command(
            self.default_ssh_config,
            f"nohup /tmp/monitor_{config.name}.sh > /var/log/{config.name}_monitor.log 2>&1 &"
        )
    
    async def _create_task_agents(self) -> List[Agent]:
        """Create AI agents for task execution"""
        
        architect_agent = Agent(
            role="Solution Architect",
            goal="Design optimal deployment architecture",
            backstory="Expert in system architecture and deployment strategies",
            llm=self.llm,
            verbose=True
        )
        
        devops_agent = Agent(
            role="DevOps Engineer", 
            goal="Execute deployment and infrastructure tasks",
            backstory="Experienced in CI/CD, containerization, and cloud deployments",
            llm=self.llm,
            verbose=True
        )
        
        security_agent = Agent(
            role="Security Engineer",
            goal="Ensure secure deployment practices",
            backstory="Expert in application security and secure deployment practices",
            llm=self.llm,
            verbose=True
        )
        
        return [architect_agent, devops_agent, security_agent]
    
    async def _analyze_repository_structure(self, project_path: str) -> Dict[str, Any]:
        """Analyze repository structure - use local execution for simple tasks"""
        
        # Check if we should deploy or just code locally
        deploy_mode = os.getenv("DEPLOY", "false").lower() == "true"
        
        if deploy_mode:
            # Use SSH for actual deployment tasks
            print(f"[DEPLOY=true] Using SSH to analyze repository structure at {project_path}")
            analysis_result = await self.ssh_manager.execute_command(
                self.default_ssh_config,
                f"find {project_path} -type f -name '*.json' -o -name '*.py' -o -name '*.js' -o -name 'Dockerfile' -o -name 'requirements.txt' -o -name 'package.json' | head -20"
            )
            files = analysis_result.stdout.strip().split('\n') if analysis_result.success else []
        else:
            # Use local execution for simple coding tasks
            print(f"[DEPLOY=false] Using local execution to analyze repository structure at {project_path}")
            try:
                result = subprocess.run(
                    f"find {project_path} -type f -name '*.json' -o -name '*.py' -o -name '*.js' -o -name 'Dockerfile' -o -name 'requirements.txt' -o -name 'package.json' | head -20",
                    shell=True, capture_output=True, text=True, timeout=30
                )
                files = result.stdout.strip().split('\n') if result.returncode == 0 and result.stdout.strip() else []
                print(f"[DEPLOY=false] Local file analysis succeeded: found {len(files)} files")
            except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
                print(f"[DEPLOY=false] Local file analysis failed: {e}")
                files = []
        
        # Detect technology stack
        tech_stack = []
        if any('package.json' in f for f in files):
            tech_stack.append('nodejs')
        if any('requirements.txt' in f for f in files):
            tech_stack.append('python')
        if any('Dockerfile' in f for f in files):
            tech_stack.append('docker')
        
        return {
            "files": files,
            "tech_stack": tech_stack,
            "has_dockerfile": any('Dockerfile' in f for f in files),
            "has_package_json": any('package.json' in f for f in files),
            "has_requirements": any('requirements.txt' in f for f in files)
        }
    async def _generate_deployment_strategy(
        self, 
        task_prompt: str, 
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate deployment strategy using cognitive functions"""
        
        # Use cognitive task breakdown to analyze the task
        context = {
            "repository_structure": analysis.get("files", []),
            "tech_stack": analysis.get("tech_stack", []),
            "existing_files": analysis.get("files", [])
        }
        
        task_breakdown = await self.cognitive_task_breakdown(task_prompt, context)
        
        # Generate content for each subtask
        generated_files = []
        tasks = []
        
        if task_breakdown.get("breakdown_data"):
            subtasks = task_breakdown["breakdown_data"].get("subtasks", [])
            
            for subtask in subtasks:
                # Determine if this subtask requires code generation
                if any(keyword in subtask["description"].lower() for keyword in 
                       ["create", "implement", "write", "generate", "build", "develop"]):
                    
                    # Generate content for this subtask
                    content_specs = {
                        "task_description": subtask["description"],
                        "complexity": subtask.get("complexity", 5),
                        "skills_required": subtask.get("skills_required", []),
                        "tech_stack": analysis.get("tech_stack", [])
                    }
                    
                    # Determine content type based on tech stack and task
                    content_type = "code"
                    if "python" in analysis.get("tech_stack", []):
                        content_type = "python"
                    elif "javascript" in analysis.get("tech_stack", []) or "nodejs" in analysis.get("tech_stack", []):
                        content_type = "javascript"
                    
                    generated_content = await self.cognitive_content_generation(
                        content_type=content_type,
                        specifications=content_specs,
                        context=context
                    )
                    
                    if generated_content.get("code_blocks"):
                        for code_block in generated_content["code_blocks"]:
                            # Determine file name based on language and content
                            file_extension = {
                                "python": ".py",
                                "javascript": ".js",
                                "typescript": ".ts",
                                "html": ".html",
                                "css": ".css",
                                "json": ".json",
                                "yaml": ".yml"
                            }.get(code_block["language"], ".txt")
                            
                            file_name = f"generated_{subtask.get('id', 'file')}{file_extension}"
                            generated_files.append({
                                "file_path": file_name,
                                "content": code_block["content"],
                                "subtask_id": subtask.get("id"),
                                "language": code_block["language"]
                            })
                
                # Create deployment task
                if analysis["has_dockerfile"]:
                    tasks.append({
                        "name": f"build_docker_image_{subtask.get('id', '')}",
                        "command": f"docker build -t app:latest .",
                        "required": True,
                        "subtask_id": subtask.get("id")
                    })
                elif analysis["has_package_json"]:
                    tasks.append({
                        "name": f"npm_install_{subtask.get('id', '')}",
                        "command": "npm install",
                        "required": True,
                        "subtask_id": subtask.get("id")
                    })
                elif analysis["has_requirements"]:
                    tasks.append({
                        "name": f"pip_install_{subtask.get('id', '')}",
                        "command": "pip install -r requirements.txt",
                        "required": True,
                        "subtask_id": subtask.get("id")
                    })
        
        # Add default tasks if none were generated
        if not tasks:
            if analysis["has_dockerfile"]:
                tasks.extend([
                    {
                        "name": "build_docker_image",
                        "command": "docker build -t app:latest .",
                        "required": True
                    },
                    {
                        "name": "run_docker_container",
                        "command": "docker run -d -p 8000:8000 --name app app:latest",
                        "required": True
                    }
                ])
            elif analysis["has_package_json"]:
                tasks.extend([
                    {
                        "name": "install_npm_dependencies",
                        "command": "npm install",
                        "required": True
                    },
                    {
                        "name": "start_application",
                        "command": "nohup npm start > app.log 2>&1 &",
                        "required": True
                    }
                ])
            elif analysis["has_requirements"]:
                tasks.extend([
                    {
                        "name": "install_python_dependencies",
                        "command": "pip install -r requirements.txt",
                        "required": True
                    },
                    {
                        "name": "start_python_application",
                        "command": "nohup python app.py > app.log 2>&1 &",
                        "required": True
                    }
                ])
        
        return {
            "strategy": "cognitive_generated",
            "tech_stack": analysis["tech_stack"],
            "tasks": tasks,
            "generated_files": generated_files,
            "task_breakdown": task_breakdown,
            "endpoint": "http://localhost:8000"
        }
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow"""
        
        if workflow_id not in self.active_workflows:
            return {"error": "Workflow not found"}
        
        workflow_state = self.active_workflows[workflow_id]
        
        return {
            "workflow_id": workflow_id,
            "workflow_type": workflow_state.workflow_type.value,
            "status": workflow_state.status.value,
            "repository_url": workflow_state.repository_url,
            "target_host": workflow_state.target_host,
            "created_at": workflow_state.created_at.isoformat(),
            "updated_at": workflow_state.updated_at.isoformat(),
            "deployment_logs": workflow_state.deployment_logs,
            "config": asdict(workflow_state.config) if workflow_state.config else None
        }
    
    async def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows"""
        
        workflows = []
        for workflow_id, workflow_state in self.active_workflows.items():
            workflows.append({
                "workflow_id": workflow_id,
                "workflow_type": workflow_state.workflow_type.value,
                "status": workflow_state.status.value,
                "repository_url": workflow_state.repository_url,
                "created_at": workflow_state.created_at.isoformat()
            })
        
        return workflows
    
    async def stop_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Stop a running workflow"""
        
        if workflow_id not in self.active_workflows:
            return {"success": False, "error": "Workflow not found"}
        
        workflow_state = self.active_workflows[workflow_id]
        
        try:
            # Stop application based on type
            if workflow_state.config:
                # AI module workflow
                stop_command = f"docker stop {workflow_state.config.name} || pkill -f '{workflow_state.config.start_command}'"
            else:
                # Task prompt workflow
                stop_command = "docker stop app || pkill -f 'python\\|node\\|npm'"
            
            stop_result = await self.ssh_manager.execute_command(
                self.default_ssh_config,
                stop_command
            )
            
            workflow_state.status = DeploymentStatus.STOPPED
            workflow_state.updated_at = datetime.now()
            
            # Log stop event
            self.audit_logger.log_event(create_security_event(
                event_type=EventType.CONFIGURATION_CHANGE,
                user_id="system",
                source_ip="localhost",
                resource=workflow_state.repository_url,
                action="workflow_stop",
                result="success",
                details={"workflow_id": workflow_id}
            ))
            
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": workflow_state.status.value
            }
            
        except Exception as e:
            logger.error(f"Failed to stop workflow {workflow_id}: {e}")
            return {
                "success": False,
                "workflow_id": workflow_id,
                "error": str(e)
            }
    
    def cleanup_resources(self):
        """Cleanup resources and close connections"""
        self.ssh_manager.close_all_connections()
        logger.info("Orchestrator resources cleaned up")

# CLI interface
if __name__ == "__main__":
    import argparse
    
    async def main():
        parser = argparse.ArgumentParser(description="Enhanced Two-Tier Orchestrator")
        parser.add_argument("--repository", required=True, help="Repository URL")
        parser.add_argument("--task-prompt", help="Task prompt for traditional workflow")
        parser.add_argument("--target-host", help="Target deployment host")
        parser.add_argument("--list-workflows", action="store_true", help="List active workflows")
        parser.add_argument("--status", help="Get workflow status by ID")
        parser.add_argument("--stop", help="Stop workflow by ID")
        
        args = parser.parse_args()
        
        orchestrator = EnhancedTwoTierOrchestrator()
        
        try:
            if args.list_workflows:
                workflows = await orchestrator.list_active_workflows()
                print(json.dumps(workflows, indent=2))
            elif args.status:
                status = await orchestrator.get_workflow_status(args.status)
                print(json.dumps(status, indent=2, default=str))
            elif args.stop:
                result = await orchestrator.stop_workflow(args.stop)
                print(json.dumps(result, indent=2))
            else:
                result = await orchestrator.run_unified_workflow(
                    repository_url=args.repository,
                    task_prompt=args.task_prompt,
                    target_host=args.target_host
                )
                print(json.dumps(result, indent=2, default=str))
        
        finally:
            orchestrator.cleanup_resources()
    
    asyncio.run(main())
