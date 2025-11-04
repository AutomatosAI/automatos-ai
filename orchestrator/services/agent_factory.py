"""
Agent Factory - User-Defined Agents with Flexible Metadata
===========================================================

Pure execution layer for agents. Users define their own agent types.
The orchestrator handles all prompt engineering using Context Engineering.
Multiple agents of different types can run simultaneously.
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import json

from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from services.llm_provider import (
    LLMManager, LLMConfig, LLMProvider, LLMResponse,
    create_llm_manager
)
from database.models import (
    Agent, Skill, PriorityLevel, Base,
    AgentToolAssignment, MCPTool  # Phase 3: MCP Tools
)
from services.skill_loader import get_skill_loader

# Import new services (lazy import to avoid circular deps)
def get_action_executor():
    from services.agent_action_executor import get_action_executor as _get_executor
    return _get_executor()

def get_monitoring_service():
    from services.monitoring_service import get_monitoring_service as _get_monitor
    return _get_monitor()

def get_rag_service():
    from services.rag_service import get_rag_service as _get_rag
    return _get_rag()

def get_unified_tool_executor(db_session: Session):
    """PRD-17 Phase 3: Get UnifiedToolExecutor instance"""
    from services.unified_tool_executor import UnifiedToolExecutor
    return UnifiedToolExecutor(db_session)

def _build_tool_schemas(required_tools: List[str]) -> List[Dict]:
    """
    PRD-17: Convert tool categories to OpenAI function calling schema.
    
    Args:
        required_tools: List of tool categories (e.g., ['research', 'file_ops'])
        
    Returns:
        List of OpenAI function schemas
    """
    tools = []
    
    # Research tools
    if "research" in required_tools:
        tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "search_knowledge",
                    "description": "Search the knowledge base (RAG) for information about a topic. Returns relevant documentation chunks.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant information"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "semantic_search",
                    "description": "Find semantically similar content in the document database using vector similarity.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The concept or text to find similar content for"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (default: 3)",
                                "default": 3
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_codebase",
                    "description": "Search the codebase using CodeGraph for classes, functions, or code patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Class name, function name, or code pattern to search for"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ])
    
    # File operation tools
    if "file_ops" in required_tools:
        tools.extend([
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file from the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to read"
                            }
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "description": "Write content to a file in the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path to the file to write"
                            },
                            "content": {
                                "type": "string",
                                "description": "Content to write to the file"
                            }
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List files and directories in a path.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Directory path to list (default: workspace root)",
                                "default": "."
                            }
                        },
                        "required": []
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_directory",
                    "description": "Create a new directory in the workspace.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {
                                "type": "string",
                                "description": "Path of the directory to create"
                            }
                        },
                        "required": ["path"]
                    }
                }
            }
        ])
    
    # Shell command tools
    if "shell" in required_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": "execute_command",
                "description": "Execute a shell command in the workspace. Use with caution.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute"
                        }
                    },
                    "required": ["command"]
                }
            }
        })
    
    # PRD-17 Phase 3: MCP tools (GitHub integration)
    if "mcp" in required_tools:
        tools.append({
            "type": "function",
            "function": {
                "name": "GitHub MCP",
                "description": "Access GitHub API for repository operations, PRs, issues, and file management.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "description": "GitHub API method (repos.get, repos.getContent, pulls.create, issues.create, etc.)",
                            "enum": ["repos.get", "repos.getContent", "repos.listForAuthenticatedUser", "pulls.create", "pulls.list", "issues.create", "issues.list"]
                        },
                        "owner": {
                            "type": "string",
                            "description": "Repository owner (username or organization)",
                            "default": "AutomatosAI"
                        },
                        "repo": {
                            "type": "string",
                            "description": "Repository name"
                        },
                        "path": {
                            "type": "string",
                            "description": "File path in repository (for getContent)"
                        },
                        "title": {
                            "type": "string",
                            "description": "Title for PR or issue"
                        },
                        "body": {
                            "type": "string",
                            "description": "Description/body for PR or issue"
                        },
                        "head": {
                            "type": "string",
                            "description": "Head branch for PR"
                        },
                        "base": {
                            "type": "string",
                            "description": "Base branch for PR (default: main)"
                        }
                    },
                    "required": ["method"]
                }
            }
        })
    
    return tools

logger = logging.getLogger(__name__)

# Agent lifecycle states
class AgentLifecycle(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    LEARNING = "learning"
    HIBERNATING = "hibernating"
    RETIRED = "retired"

# Default LLM configuration for agents
DEFAULT_LLM_CONFIG = {
    "provider": "openai",  # Use environment default
    "model": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "context_window": 8192
}

# PRD-15: Multi-Model Configuration
@dataclass
class ModelConfiguration:
    """
    Complete model configuration for an agent (PRD-15).
    
    This dataclass encapsulates all model-specific settings including
    provider, model ID, and generation parameters.
    """
    provider: str  # 'openai', 'anthropic', 'huggingface'
    model_id: str  # 'gpt-4', 'claude-3-sonnet-20240229', etc.
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    fallback_model_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "fallback_model_id": self.fallback_model_id
        }
    
    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'ModelConfiguration':
        """Create from dictionary"""
        return ModelConfiguration(
            provider=data.get("provider", "openai"),
            model_id=data.get("model_id", "gpt-4"),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 2000),
            top_p=data.get("top_p", 1.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            fallback_model_id=data.get("fallback_model_id")
        )
    
    @staticmethod
    def get_default() -> 'ModelConfiguration':
        """Get default configuration"""
        return ModelConfiguration(
            provider="openai",
            model_id="gpt-4",
            temperature=0.7,
            max_tokens=2000
        )

@dataclass
class AgentMetadata:
    """
    User-defined agent metadata - completely flexible.
    
    Enhanced in PRD-15 to support full model configuration.
    Maintains backward compatibility with deprecated fields.
    """
    name: str
    agent_type: str  # User-defined type (e.g., "financial_analyst", "code_reviewer")
    description: Optional[str] = None
    skills: List[str] = field(default_factory=list)  # Semantic tags for matching
    
    # PRD-15: New model configuration
    model_config: Optional[ModelConfiguration] = None
    
    # Deprecated: Keep for backward compatibility
    preferred_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    
    custom_metadata: Dict[str, Any] = field(default_factory=dict)  # Any user data
    
    def get_model_config(self) -> ModelConfiguration:
        """
        Get model configuration with fallbacks.
        
        Priority:
        1. model_config (new)
        2. deprecated fields (backward compatibility)
        3. default configuration
        
        Returns:
            ModelConfiguration object
        """
        # Use new model_config if available
        if self.model_config:
            return self.model_config
        
        # Fall back to deprecated fields for backward compatibility
        if self.preferred_model:
            provider = "openai"
            if "claude" in self.preferred_model.lower():
                provider = "anthropic"
            elif "llama" in self.preferred_model.lower() or "mistral" in self.preferred_model.lower():
                provider = "huggingface"
            
            return ModelConfiguration(
                provider=provider,
                model_id=self.preferred_model,
                temperature=self.temperature or 0.7,
                max_tokens=self.max_tokens or 2000
            )
        
        # Use default
        return ModelConfiguration.get_default()
    
    def get_llm_config(self) -> Dict[str, Any]:
        """
        Get LLM configuration dict (backward compatible).
        
        Deprecated: Use get_model_config() instead.
        Maintained for backward compatibility.
        """
        model_config = self.get_model_config()
        return {
            "provider": model_config.provider,
            "model": model_config.model_id,
            "temperature": model_config.temperature,
            "max_tokens": model_config.max_tokens,
            "context_window": self.context_window or 8192
        }

@dataclass
class AgentRuntime:
    """Runtime representation of an agent"""
    agent_id: int
    metadata: AgentMetadata
    llm_manager: LLMManager
    lifecycle_state: AgentLifecycle
    created_at: datetime
    execution_count: int = 0
    total_tokens_used: int = 0
    last_execution: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)  # Short-term memory
    tools: List[Dict[str, Any]] = field(default_factory=list)  # Phase 3: MCP Tools assigned to agent
    tool_executor: Any = None  # PRD-17: Shared UnifiedToolExecutor (initialized once, reused)
    
    def update_metrics(self, execution_time: float, tokens_used: int, success: bool):
        """Update agent performance metrics"""
        self.execution_count += 1
        self.total_tokens_used += tokens_used
        self.last_execution = datetime.now()
        
        # Update rolling metrics
        if "avg_execution_time" not in self.performance_metrics:
            self.performance_metrics["avg_execution_time"] = execution_time
        else:
            # Rolling average
            avg = self.performance_metrics["avg_execution_time"]
            self.performance_metrics["avg_execution_time"] = (
                (avg * (self.execution_count - 1) + execution_time) / self.execution_count
            )
        
        # Success rate
        if "success_count" not in self.performance_metrics:
            self.performance_metrics["success_count"] = 0
        if success:
            self.performance_metrics["success_count"] += 1
        
        self.performance_metrics["success_rate"] = (
            self.performance_metrics["success_count"] / self.execution_count
        )

class AgentFactory:
    """
    Creates and manages user-defined agents.
    Pure execution layer - the orchestrator handles all prompt engineering.
    Can manage multiple agents of different types simultaneously.
    """
    
    def __init__(self, db_session: Session = None):
        # Use centralized database session
        if db_session:
            self.db_session = db_session
        else:
            from database.database import SessionLocal
            self.db_session = SessionLocal()
        
        self.active_agents: Dict[int, AgentRuntime] = {}
        self.logger = logging.getLogger(__name__)
    
    def _build_tools_prompt(self, required_tools: List[str]) -> str:
        """
        PRD-17: Build dynamic tools prompt based on required tool categories.
        
        Args:
            required_tools: List of tool categories (e.g., ['research', 'file_ops', 'shell'])
            
        Returns:
            Formatted prompt string with available tools
        """
        tools_sections = []
        
        # Research tools (search_knowledge, semantic_search, search_codebase)
        if "research" in required_tools:
            tools_sections.append("""
## 🔍 RESEARCH TOOLS

Available when you need to find information:
1. **search_knowledge** - Search documentation and knowledge base
   {"action": "search_knowledge", "params": {"query": "your search query", "limit": 5}}

2. **semantic_search** - Find semantically similar content
   {"action": "semantic_search", "params": {"query": "concept to find", "limit": 5}}

3. **search_codebase** - Search code implementations  
   {"action": "search_codebase", "params": {"query": "function or class name"}}

Use these tools if you need to understand something before acting.""")
        
        # File operation tools (read_file, write_file, list_directory, create_directory)
        if "file_ops" in required_tools:
            tools_sections.append("""
## 📁 FILE OPERATION TOOLS

Available File Tools:
1. **read_file** - Read file contents
   {"action": "read_file", "params": {"path": "path/to/file.py"}}

2. **write_file** - Create or update files
   {"action": "write_file", "params": {"path": "path/to/file.py", "content": "file contents here"}}

3. **list_directory** - List directory contents
   {"action": "list_directory", "params": {"path": "path/to/directory"}}

4. **create_directory** - Create a new directory
   {"action": "create_directory", "params": {"path": "path/to/new_dir"}}""")
        
        # Shell command tools (execute_command)
        if "shell" in required_tools:
            tools_sections.append("""
## 💻 SHELL COMMAND TOOLS

Available Shell Tools:
1. **execute_command** - Run shell commands (use with caution!)
   {"action": "execute_command", "params": {"command": "ls -la", "timeout": 30}}

⚠️  Use shell tools carefully - always validate commands before execution""")
        
        # Build final prompt
        if not tools_sections:
            return ""  # No tools needed
        
        final_prompt = "\n".join(tools_sections)
        final_prompt += """

**EXECUTION RULES**:
- You have tools available - use them when needed
- For simple operations (create file, run command), just execute directly
- For complex tasks requiring understanding, search first then act
- Be efficient - don't over-research simple tasks
"""
        return final_prompt
    
    async def create_agent(
        self,
        metadata: Union[AgentMetadata, Dict[str, Any]],
        auto_verify: bool = True
    ) -> AgentRuntime:
        """
        Create an agent from user-defined metadata.
        
        Args:
            metadata: AgentMetadata or dict with agent configuration
            auto_verify: Verify LLM connection immediately
            
        Returns:
            AgentRuntime with active LLM connection
        """
        start_time = time.time()
        
        # Convert dict to AgentMetadata if needed
        if isinstance(metadata, dict):
            # PRD-15: Handle model_config from dict
            model_config = None
            if "model_config" in metadata:
                model_config = ModelConfiguration.from_dict(metadata["model_config"])
            
            metadata = AgentMetadata(
                name=metadata.get("name", "Unnamed Agent"),
                agent_type=metadata.get("type", "generic"),
                description=metadata.get("description"),
                skills=metadata.get("skills", []),
                model_config=model_config,
                # Deprecated fields for backward compatibility
                preferred_model=metadata.get("preferred_model"),
                temperature=metadata.get("temperature"),
                max_tokens=metadata.get("max_tokens"),
                context_window=metadata.get("context_window"),
                custom_metadata=metadata.get("metadata", {})
            )
        
        # PRD-15: Get model configuration
        model_config = metadata.get_model_config()
        
        # Create database record
        db_agent = Agent(
            name=metadata.name,
            description=metadata.description or f"User-defined {metadata.agent_type} agent",
            agent_type=metadata.agent_type,  # User-defined type
            status=AgentLifecycle.INITIALIZING.value,
            configuration={
                "skills": metadata.skills,
                "llm_config": metadata.get_llm_config(),  # Backward compatibility
                "custom_metadata": metadata.custom_metadata
            },
            model_config=model_config.to_dict(),  # PRD-15: Store model config
            priority_level=PriorityLevel.MEDIUM.value,
            max_concurrent_tasks=5,
            auto_start=False,
            created_by="agent_factory"
        )
        
        self.db_session.add(db_agent)
        self.db_session.commit()
        
        # PRD-15: Initialize LLM connection with model configuration
        try:
            llm_manager = await self._create_llm_manager(model_config, db_agent.name)
            
            # Verify connection if requested
            if auto_verify:
                verification_result = await self._verify_llm_connection(llm_manager)
                if not verification_result["success"]:
                    # PRD-15: Try fallback model if configured
                    if model_config.fallback_model_id:
                        self.logger.warning(
                            f"Primary model '{model_config.model_id}' failed, "
                            f"trying fallback '{model_config.fallback_model_id}'"
                        )
                        fallback_config = ModelConfiguration(
                            provider=model_config.provider,
                            model_id=model_config.fallback_model_id,
                            temperature=model_config.temperature,
                            max_tokens=model_config.max_tokens
                        )
                        llm_manager = await self._create_llm_manager(fallback_config, db_agent.name)
                        verification_result = await self._verify_llm_connection(llm_manager)
                        
                        if verification_result["success"]:
                            # Update db_agent with fallback model
                            db_agent.model_config = fallback_config.to_dict()
                            self.db_session.commit()
                            self.logger.info(f"Fallback model '{model_config.fallback_model_id}' succeeded")
                    
                    if not verification_result["success"]:
                        self.db_session.delete(db_agent)
                        self.db_session.commit()
                        raise Exception(f"LLM verification failed: {verification_result['error']}")
                
                self.logger.info(
                    f"✓ Agent '{metadata.name}' LLM verified: "
                    f"model={model_config.model_id}, provider={model_config.provider}, "
                    f"response_time={verification_result['response_time']:.2f}s"
                )
            
            # Phase 3: Load agent's tools from database
            agent_tools = await self._load_agent_tools(db_agent.id)
            
            # Create runtime agent
            agent_runtime = AgentRuntime(
                agent_id=db_agent.id,
                metadata=metadata,
                llm_manager=llm_manager,
                lifecycle_state=AgentLifecycle.ACTIVE,
                created_at=datetime.now(),
                tools=agent_tools  # Phase 3: MCP Tools
            )
            
            # Update database status
            db_agent.status = AgentLifecycle.ACTIVE.value
            self.db_session.commit()
            
            # Store in active agents
            self.active_agents[db_agent.id] = agent_runtime
            
            self.logger.info(
                f"Agent '{metadata.name}' (type: {metadata.agent_type}) created in "
                f"{time.time() - start_time:.2f}s"
            )
            
            return agent_runtime
            
        except Exception as e:
            self.logger.error(f"Failed to create agent: {str(e)}")
            if db_agent.id:
                self.db_session.delete(db_agent)
                self.db_session.commit()
            raise
    
    async def _create_llm_manager(self, model_config: ModelConfiguration, agent_name: str = "") -> LLMManager:
        """
        Create LLM manager from model configuration (PRD-15).
        
        Args:
            model_config: ModelConfiguration with provider and model settings
            agent_name: Agent name for logging
            
        Returns:
            Initialized LLMManager
            
        Raises:
            ValueError: If provider is unsupported
        """
        from services.llm_provider import LLMConfig, LLMProvider as LLMProviderEnum
        
        # Map provider string to enum
        provider_map = {
            "openai": LLMProviderEnum.OPENAI,
            "anthropic": LLMProviderEnum.ANTHROPIC
        }
        
        if model_config.provider not in provider_map:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        
        provider = provider_map[model_config.provider]
        
        # Get API key from credential resolver
        from services.credential_resolver import get_credential_resolver
        resolver = get_credential_resolver()
        
        api_key = None
        if provider == LLMProviderEnum.OPENAI:
            api_key = resolver.get_credential_field("development_openai", "api_key")
        elif provider == LLMProviderEnum.ANTHROPIC:
            api_key = resolver.get_credential_field("development_anthropic", "api_key")
        
        if not api_key:
            raise ValueError(f"API key not found for provider: {model_config.provider}")
        
        # Create LLM config
        llm_config = LLMConfig(
            provider=provider,
            model=model_config.model_id,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            api_key=api_key
        )
        
        self.logger.info(
            f"Creating LLM manager for {agent_name or 'agent'}: "
            f"provider={model_config.provider}, model={model_config.model_id}"
        )
        
        return LLMManager(config=llm_config)
    
    async def _verify_llm_connection(self, llm_manager: LLMManager) -> Dict[str, Any]:
        """Verify LLM connection with minimal test"""
        try:
            start_time = time.time()
            
            # Minimal verification - no embedded prompts
            messages = [
                {"role": "user", "content": "Respond with 'OK' to confirm connection."}
            ]
            
            response = await llm_manager.generate_response(messages)
            response_time = time.time() - start_time
            
            if response and response.content:
                return {
                    "success": True,
                    "response_time": response_time,
                    "tokens_used": response.usage.get("total_tokens", 0) if response.usage else 0
                }
            else:
                return {"success": False, "error": "No response from LLM"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def activate_agent(self, agent_id: int) -> Optional[AgentRuntime]:
        """
        Load an agent from database and activate it in runtime.
        
        Args:
            agent_id: ID of agent to activate
            
        Returns:
            AgentRuntime if successful, None if agent not found or activation failed
        """
        try:
            # Check if already active
            if agent_id in self.active_agents:
                self.logger.info(f"Agent {agent_id} already active in runtime")
                return self.active_agents[agent_id]
            
            # Load from database
            db_agent = self.db_session.query(Agent).filter(Agent.id == agent_id).first()
            if not db_agent:
                self.logger.error(f"Agent {agent_id} not found in database")
                return None
            
            # Get LLM config from agent configuration
            config = db_agent.configuration or {}
            llm_config_dict = config.get("llm_config")
            
            if not llm_config_dict:
                # Use default if not configured
                self.logger.warning(f"Agent {agent_id} has no llm_config, using DEFAULT_LLM_CONFIG")
                llm_config_dict = DEFAULT_LLM_CONFIG.copy()
            
            # Create LLM manager
            provider = LLMProvider(llm_config_dict.get("provider", "openai"))
            llm_config = LLMConfig(
                provider=provider,
                model=llm_config_dict.get("model", "gpt-4"),
                temperature=llm_config_dict.get("temperature", 0.7),
                max_tokens=llm_config_dict.get("max_tokens", 2000),
            )
            llm_manager = LLMManager(llm_config)
            
            # Create metadata from database agent
            metadata = AgentMetadata(
                name=db_agent.name,
                agent_type=db_agent.agent_type,
                description=db_agent.description,
                skills=config.get("skills", []),
                custom_metadata=config.get("custom_metadata", {})
            )
            
            # Load agent's tools
            agent_tools = await self._load_agent_tools(agent_id)
            
            # Create runtime
            agent_runtime = AgentRuntime(
                agent_id=agent_id,
                metadata=metadata,
                llm_manager=llm_manager,
                lifecycle_state=AgentLifecycle.ACTIVE,
                created_at=datetime.now(),
                tools=agent_tools,
                tool_executor=get_unified_tool_executor(self.db_session)  # PRD-17: Initialize once, reuse
            )
            
            # Add to active agents
            self.active_agents[agent_id] = agent_runtime
            
            # Update database status
            db_agent.status = AgentLifecycle.ACTIVE.value
            self.db_session.commit()
            
            self.logger.info(f"✅ Activated agent {agent_id} ({db_agent.name}) with {llm_config_dict.get('model')}")
            
            return agent_runtime
            
        except Exception as e:
            self.logger.error(f"Failed to activate agent {agent_id}: {str(e)}")
            return None
    
    async def execute_with_prompt(
        self,
        agent: Union[int, AgentRuntime],
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        use_memory: bool = True,
        max_retries: int = 2,
        enable_actions: bool = True,
        action_executor: Optional[Any] = None,
        required_tools: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Execute a task with orchestrator-provided prompt.
        
        The orchestrator provides the fully engineered prompt using Context Engineering.
        
        Args:
            agent: Agent ID or runtime
            prompt: User prompt from orchestrator (may be atomic, molecular, or cellular)
            system_prompt: System prompt from orchestrator (with context, examples, etc.)
            context: Additional structured context
            use_memory: Include agent's short-term memory
            max_retries: Number of retries on failure
            
        Returns:
            Execution result with LLM response
        """
        start_time = time.time()
        
        # Get agent runtime - auto-activate if needed
        if isinstance(agent, int):
            agent_runtime = self.active_agents.get(agent)
            if not agent_runtime:
                # Agent not in runtime - try to activate it
                self.logger.info(f"Agent {agent} not in runtime, attempting to activate...")
                agent_runtime = await self.activate_agent(agent)
                if not agent_runtime:
                    return {
                        "status": "error",
                        "error": f"Agent {agent} could not be activated"
                    }
        else:
            agent_runtime = agent
        
        # Update state
        agent_runtime.lifecycle_state = AgentLifecycle.BUSY
        
        try:
            # Build messages - orchestrator provides the engineered prompts
            messages = []
            
            # System prompt from orchestrator (contains context engineering)
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Add short-term memory if enabled
            if use_memory and agent_runtime.memory:
                # Only include recent relevant memory (cells concept)
                recent_memory = agent_runtime.memory[-3:]  # Last 3 interactions
                for mem in recent_memory:
                    if "user_prompt" in mem:
                        messages.append({"role": "user", "content": mem["user_prompt"]})
                    if "assistant_response" in mem:
                        messages.append({"role": "assistant", "content": mem["assistant_response"]})
            
            # Check if actions are needed and add capabilities
            action_executor = None
            if enable_actions and self._requires_actions(prompt):
                action_executor = get_action_executor()
                action_prompt = """\n\nYou have access to perform real actions:
- read_file(path) - Read file contents
- write_file(path, content) - Create/update files
- execute_command(cmd) - Run shell commands
- list_directory(path) - List directory contents

To use actions, respond with JSON blocks like:
{"action": "write_file", "params": {"path": "test.py", "content": "print('hello')"}}
\nYou can include multiple action blocks in your response."""
                prompt = prompt + action_prompt
            
            # PRD-17: Dynamic tool injection based on required_tools
            # Default to research tools for backwards compatibility
            if required_tools is None:
                required_tools = ["research"]
            
            # Build OpenAI function calling schemas
            tool_schemas = _build_tool_schemas(required_tools)
            self.logger.info(f"📦 PRD-17: Providing {len(tool_schemas)} tools to agent: {[t['function']['name'] for t in tool_schemas]}")
            
            action_executor = action_executor or get_action_executor()  # Ensure executor exists
            
            # Add the main prompt from orchestrator
            messages.append({"role": "user", "content": prompt})
            
            # Execute with retries (at least 1 attempt)
            last_error = None
            for attempt in range(max(1, max_retries)):
                try:
                    # REAL LLM API CALL WITH FUNCTION CALLING
                    response = await agent_runtime.llm_manager.generate_response(messages, tools=tool_schemas)
                    execution_time = time.time() - start_time
                    
                    # PRD-17: Handle function calling responses
                    if response and response.tool_calls:
                        self.logger.info(f"🔧 PRD-17: Agent called {len(response.tool_calls)} tool(s) via function calling")
                        tool_results = []
                        tool_executor = agent_runtime.tool_executor  # PRD-17: Reuse executor (no re-init!)
                        
                        for tool_call in response.tool_calls:
                            func_name = tool_call['function']['name']
                            func_args = json.loads(tool_call['function']['arguments'])
                            self.logger.info(f"  🛠️  Calling {func_name}({func_args})")
                            
                            try:
                                result = await tool_executor.execute_tool(
                                    tool_name=func_name,
                                    parameters=func_args,
                                    agent_id=agent_runtime.agent_id
                                )
                                tool_results.append({
                                    "tool_call_id": tool_call['id'],
                                    "role": "tool",
                                    "name": func_name,
                                    "content": json.dumps(result)
                                })
                                self.logger.info(f"    ✅ {func_name} completed successfully")
                            except Exception as e:
                                self.logger.error(f"    ❌ {func_name} failed: {e}")
                                tool_results.append({
                                    "tool_call_id": tool_call['id'],
                                    "role": "tool",
                                    "name": func_name,
                                    "content": json.dumps({"error": str(e)})
                                })
                        
                        # Add assistant's tool call message and tool results to conversation
                        messages.append({
                            "role": "assistant",
                            "content": response.content or "",
                            "tool_calls": response.tool_calls
                        })
                        messages.extend(tool_results)
                        
                        # Call LLM again to process tool results
                        self.logger.info("  🔄 Calling LLM again with tool results...")
                        response = await agent_runtime.llm_manager.generate_response(messages, tools=tool_schemas)
                        execution_time = time.time() - start_time
                        
                        # PRD-17: If LLM returns empty content, use tool results as the response
                        if not response.content or response.content.strip() == "":
                            self.logger.warning("  ⚠️  LLM returned empty content after tool use, using tool results")
                            # Format tool results into readable response
                            tool_summary = "\n\n".join([
                                f"**{tr['name']}**: {tr['content'][:500]}..." 
                                for tr in tool_results
                            ])
                            response.content = f"Based on the tool results:\n\n{tool_summary}"
                        
                        self.logger.info("  ✅ LLM provided final answer after processing tool results")
                    
                    # Process any action requests in the response and iterate if needed (fallback for old JSON format)
                    action_results = []
                    if action_executor and response and response.content and '{"action"' in response.content:
                        self.logger.info(f"🔧 Agent requested tool calls, executing...")
                        action_results = await self._process_agent_actions(response.content, action_executor, agent_runtime)
                        
                        # If tools were executed, feed results back to agent for final answer
                        if action_results:
                            self.logger.info(f"  ✅ {len(action_results)} tool(s) executed, feeding results back to agent")
                            
                            # Add agent's tool request to messages
                            messages.append({"role": "assistant", "content": response.content})
                            
                            # Create tool results message - agent-friendly format
                            tool_results_text = "Research Results:\n\n"
                            for idx, result in enumerate(action_results, 1):
                                action_name = result.get('action', result.get('tool', 'unknown'))
                                tool_results_text += f"=== Tool {idx}: {action_name} ===\n"
                                
                                if result.get('success', result.get('status') == 'success'):
                                    # Success case - show actual content
                                    result_data = result.get('result', result.get('data', []))
                                    count = result.get('count', 0)
                                    
                                    if isinstance(result_data, list) and len(result_data) > 0:
                                        tool_results_text += f"Found {count} results:\n\n"
                                        # Show top 3 results with full content
                                        for i, item in enumerate(result_data[:3], 1):
                                            if isinstance(item, dict):
                                                content = item.get('content', str(item))
                                                relevance = item.get('relevance', item.get('similarity', 0))
                                                source = item.get('source', 'Unknown')
                                                tool_results_text += f"{i}. {content}\n"
                                                tool_results_text += f"   [Relevance: {relevance:.2f}, Source: {source}]\n\n"
                                            else:
                                                tool_results_text += f"{i}. {str(item)[:500]}\n\n"
                                        
                                        if count > 3:
                                            tool_results_text += f"(+ {count - 3} more results available)\n\n"
                                    else:
                                        tool_results_text += f"Result: {str(result_data)[:800]}\n\n"
                                else:
                                    # Error case
                                    error_msg = result.get('error', result.get('result', 'Unknown error'))
                                    tool_results_text += f"ERROR: {error_msg}\n"
                                    tool_results_text += f"Action: Continue with available knowledge.\n\n"
                            
                            tool_results_text += "\n📝 INSTRUCTIONS: Use the research results above to provide a detailed, accurate answer to the original task. Cite sources where appropriate."
                            
                            messages.append({"role": "user", "content": tool_results_text})
                            
                            # Call LLM again with tool results
                            self.logger.info("  🔄 Calling agent again with tool results for final answer...")
                            response = await agent_runtime.llm_manager.generate_response(messages)
                            execution_time = time.time() - start_time
                            self.logger.info("  ✅ Agent provided final answer after research")
                    
                    if response and response.content:
                        # Success! Update metrics
                        tokens_used = response.usage.get("total_tokens", 0) if response.usage else 0
                        agent_runtime.update_metrics(execution_time, tokens_used, True)
                        
                        # Store in memory
                        memory_entry = {
                            "task": prompt[:200],  # Truncate for storage
                            "response": response.content[:500],  # Truncate for storage
                            "summary": f"Executed: {prompt[:100]}",
                            "timestamp": datetime.now().isoformat(),
                            "tokens": tokens_used,
                            "execution_time": execution_time
                        }
                        agent_runtime.memory.append(memory_entry)
                        
                        # Keep memory size manageable
                        if len(agent_runtime.memory) > 20:
                            agent_runtime.memory = agent_runtime.memory[-20:]
                        
                        # Update lifecycle state
                        agent_runtime.lifecycle_state = AgentLifecycle.ACTIVE
                        
                        # Record in monitoring service
                        monitoring = get_monitoring_service()
                        monitoring.record_agent_execution(
                            agent_id=agent_runtime.agent_id,
                            agent_name=agent_runtime.metadata.name,
                            task=prompt[:100],
                            execution_time_ms=execution_time * 1000,
                            tokens_used=tokens_used,
                            success=True
                        )
                        
                        # Return successful result
                        return {
                            "status": "success",
                            "result": response.content,
                            "agent": {
                                "id": agent_runtime.agent_id,
                                "name": agent_runtime.metadata.name,
                                "type": agent_runtime.metadata.agent_type
                            },
                            "execution": {
                                "time": execution_time,
                                "tokens_used": tokens_used,
                                "model": response.model,
                                "provider": response.provider,
                                "attempt": attempt + 1,
                                "actions_enabled": enable_actions,
                                "actions_executed": len(action_results)
                            },
                            "action_results": action_results,
                            "metrics": {
                                "total_executions": agent_runtime.execution_count,
                                "success_rate": agent_runtime.performance_metrics.get("success_rate", 1.0),
                                "avg_execution_time": agent_runtime.performance_metrics.get("avg_execution_time", execution_time)
                            }
                        }
                    else:
                        last_error = "Empty response from LLM"
                        
                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                    
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff
            
            # All retries failed
            agent_runtime.update_metrics(time.time() - start_time, 0, False)
            agent_runtime.lifecycle_state = AgentLifecycle.ACTIVE
            
            return {
                "status": "error",
                "error": f"Task execution failed after {max_retries} attempts: {last_error}",
                "agent": {
                    "id": agent_runtime.agent_id,
                    "name": agent_runtime.metadata.name,
                    "type": agent_runtime.metadata.agent_type
                }
            }
            
        except Exception as e:
            agent_runtime.lifecycle_state = AgentLifecycle.ACTIVE
            self.logger.error(f"Task execution error: {str(e)}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def apply_skills(
        self,
        agent: AgentRuntime,
        new_skills: List[str]
    ) -> AgentRuntime:
        """
        Apply new skills to an agent by enhancing its system prompt.
        
        Args:
            agent: Agent runtime to enhance
            new_skills: List of skill names to add
            
        Returns:
            Updated agent runtime
        """
        skill_enhancements = []
        
        for skill in new_skills:
            if skill in SKILL_PROMPTS and skill not in agent.metadata.skills:
                skill_enhancements.append(SKILL_PROMPTS[skill])
                agent.metadata.skills.append(skill)
                self.logger.info(f"Applied skill '{skill}' to agent '{agent.metadata.name}'")
        
        if skill_enhancements:
            # Append to existing system prompt
            if "\n\n## Specialized Skills:" not in agent.system_prompt:
                agent.system_prompt += "\n\n## Specialized Skills:"
            agent.system_prompt += "".join(skill_enhancements)
            
            self.logger.info(
                f"Enhanced agent '{agent.metadata.name}' with {len(skill_enhancements)} new skills"
            )
        
        return agent

    # ======================================================================
    # PRD-22: Build agent system prompt with progressive skill injection
    # ======================================================================
    def _build_agent_system_prompt(
        self,
        agent: Agent,
        task_context: Optional[str] = None,
        db: Optional[Session] = None,
        required_tools: Optional[List[str]] = None
    ) -> str:
        """
        Build the agent system prompt, injecting PRD-22 skill content (Level 2).

        - Loads core content for each assigned skill via SkillLoader (progressive disclosure)
        - Falls back to skill.description for legacy/seed skills without filesystem content
        - Keeps tool schemas separate; optionally appends a tools section if provided
        """
        sections: List[str] = []

        # Identity
        sections.append(f"# Agent: {agent.name}")
        if agent.description:
            sections.append(agent.description)

        # Task context (optional)
        if task_context:
            sections.append("\n## Task Context\n" + str(task_context))

        # Skills (progressive disclosure)
        if getattr(agent, 'skills', None):
            sections.append("\n## Your Specialized Skills\n")
            loader = get_skill_loader(db) if db is not None else None

            for skill in agent.skills:
                sections.append(f"### {skill.name}")

                core_content = None
                if loader is not None:
                    try:
                        # Level 2: core content from SKILL.md body or prompt_template
                        core_content = loader.load_skill_core(skill.name, db=db)
                    except Exception:
                        core_content = None

                if core_content and isinstance(core_content, str) and core_content.strip():
                    sections.append(core_content)
                else:
                    # Fallback for legacy/seed skills
                    fallback = skill.prompt_template or skill.description or ""
                    if fallback:
                        sections.append(str(fallback))

        # Optional tools section (kept minimal; main tool wiring remains elsewhere)
        if required_tools:
            try:
                tool_schemas = _build_tool_schemas(required_tools)
                sections.append("\n## Available Tools\n")
                sections.append(json.dumps(tool_schemas))
            except Exception:
                pass

        return "\n\n".join([s for s in sections if s is not None])
    
    async def get_agent_status(self, agent_id: int) -> Dict[str, Any]:
        """
        Get detailed status of an agent.
        
        Returns real-time agent information.
        """
        agent_runtime = self.active_agents.get(agent_id)
        
        if not agent_runtime:
            # Try database
            db_agent = self.db_session.query(Agent).filter_by(id=agent_id).first()
            if db_agent:
                return {
                    "status": "inactive",
                    "agent": {
                        "id": db_agent.id,
                        "name": db_agent.metadata.name,
                        "type": db_agent.agent_type,
                        "database_status": db_agent.status,
                        "created_at": db_agent.created_at.isoformat() if db_agent.created_at else None
                    },
                    "runtime": None,
                    "message": "Agent exists in database but not in runtime. Use activate_agent() to initialize."
                }
            else:
                return {
                    "status": "not_found",
                    "error": f"Agent {agent_id} does not exist"
                }
        
        # Get provider info
        provider_info = agent_runtime.llm_manager.get_provider_info()
        
        return {
            "status": "active",
            "agent": {
                "id": agent_runtime.agent_id,
                "name": agent_runtime.metadata.name,
                "type": agent_runtime.metadata.agent_type,
                "lifecycle_state": agent_runtime.lifecycle_state.value,
                "skills": agent_runtime.metadata.skills
            },
            "runtime": {
                "created_at": agent_runtime.created_at.isoformat(),
                "last_execution": agent_runtime.last_execution.isoformat() if agent_runtime.last_execution else None,
                "execution_count": agent_runtime.execution_count,
                "total_tokens_used": agent_runtime.total_tokens_used,
                "memory_size": len(agent_runtime.memory)
            },
            "llm": provider_info,
            "metrics": agent_runtime.performance_metrics
        }
    
    async def test_agent_capabilities(self, agent: AgentRuntime) -> Dict[str, Any]:
        """
        Run comprehensive tests on agent capabilities.
        
        Returns detailed test results.
        """
        test_results = {
            "agent_id": agent.agent_id,
            "agent_name": agent.metadata.name,
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
        # Test 1: Basic response
        test1 = await self.execute_with_prompt(
            agent,
            "What are your primary capabilities?",
            use_memory=False
        )
        test_results["tests"].append({
            "name": "basic_response",
            "success": test1["status"] == "success",
            "execution_time": test1.get("execution", {}).get("time"),
            "tokens": test1.get("execution", {}).get("tokens_used")
        })
        
        # Test 2: Skill-specific task
        if agent.metadata.skills:
            skill_task = f"Demonstrate your {agent.metadata.skills[0]} capability with a brief example."
            test2 = await self.execute_with_prompt(agent, skill_task)
            test_results["tests"].append({
                "name": f"skill_test_{agent.metadata.skills[0]}",
                "success": test2["status"] == "success",
                "execution_time": test2.get("execution", {}).get("time"),
                "tokens": test2.get("execution", {}).get("tokens_used")
            })
        
        # Test 3: Context handling
        test3 = await self.execute_with_prompt(
            agent,
            "Analyze this context and provide insights",
            context={"data": "test", "value": 42, "items": ["a", "b", "c"]}
        )
        test_results["tests"].append({
            "name": "context_handling",
            "success": test3["status"] == "success",
            "execution_time": test3.get("execution", {}).get("time"),
            "tokens": test3.get("execution", {}).get("tokens_used")
        })
        
        # Summary
        successful_tests = sum(1 for t in test_results["tests"] if t["success"])
        test_results["summary"] = {
            "total_tests": len(test_results["tests"]),
            "successful": successful_tests,
            "success_rate": successful_tests / len(test_results["tests"]) if test_results["tests"] else 0,
            "total_time": sum(t.get("execution_time", 0) for t in test_results["tests"] if t.get("execution_time")),
            "total_tokens": sum(t.get("tokens", 0) for t in test_results["tests"] if t.get("tokens"))
        }
        
        return test_results
    
    # ======================================================================
    # ACTION EXECUTOR HELPER METHODS
    # ======================================================================
    
    def _requires_actions(self, prompt: str) -> bool:
        """Check if the prompt requires action capabilities"""
        action_keywords = [
            'write', 'create', 'file', 'save', 'execute', 'run',
            'command', 'shell', 'list', 'read', 'directory', 'folder',
            'delete', 'remove', 'mkdir', 'code', 'script', 'program'
        ]
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in action_keywords)
    
    async def _process_agent_actions(self, response: str, action_executor, agent_runtime=None) -> List[Dict[str, Any]]:
        """Process action requests from agent response"""
        import re
        results = []
        
        # Find all JSON action blocks in the response - match complete JSON objects
        action_pattern = r'\{"action":\s*"[^"]+",\s*"params":\s*\{[^}]*\}\}'
        matches = re.finditer(action_pattern, response)
        
        for match in matches:
            try:
                action_json = match.group(0)
                action_data = json.loads(action_json)
                
                action_type = action_data.get('action')
                params = action_data.get('params', {})
                
                # PRD-17 Phase 3: Reuse agent's tool_executor (initialized once)
                tool_executor = agent_runtime.tool_executor if agent_runtime else get_unified_tool_executor(self.db_session)
                result = await tool_executor.execute_tool(
                    tool_name=action_type,
                    parameters=params,
                    agent_id=0
                )
                
                # Enhanced logging for research tools
                if action_type == 'search_knowledge':
                    self.logger.info(f"  🔍 Executing search_knowledge with params: {params}")
                    self.logger.info(f"  📊 Result: success={result.get('success')}, count={result.get('count', 0)}")
                    if result.get('results'):
                        self.logger.info(f"  📄 First result preview: {str(result.get('results', [{}])[0])[:200]}")
                    self.logger.info(f"  ✅ Knowledge search: {result.get('count', 0)} results found")
                elif action_type in ['semantic_search', 'search_codebase']:
                    self.logger.info(f"  ✅ {action_type}: {result.get('count', 0)} results found")
                
                results.append(result)
                    
            except json.JSONDecodeError as e:
                # Try to fix common JSON issues
                self.logger.warning(f"JSON parse error at column {e.colno}: {e.msg}")
                self.logger.warning(f"  Full problematic JSON: {match.group(0)}")
                try:
                    fixed_json = match.group(0)
                    fixed_json = fixed_json.replace("'", '"')  # Single to double quotes
                    fixed_json = re.sub(r',\s*}', '}', fixed_json)  # Trailing commas in objects
                    fixed_json = re.sub(r',\s*\]', ']', fixed_json)  # Trailing commas in arrays
                    action_data = json.loads(fixed_json)
                    self.logger.info(f"  ✅ JSON fixed and parsed successfully")
                    # Note: Would need to reprocess the fixed action_data here
                except Exception as fix_error:
                    self.logger.error(f"Failed to fix JSON: {fix_error}")
            except Exception as e:
                self.logger.error(f"Failed to process action: {e}")
                results.append({
                    'action': 'unknown',
                    'params': {},
                    'success': False,
                    'result': str(e)
                })
        
        return results
    
    # ======================================================================
    # PHASE 3: MCP TOOLS INTEGRATION METHODS
    # ======================================================================
    
    async def _load_agent_tools(self, agent_id: int) -> List[Dict[str, Any]]:
        """
        Load MCP tools assigned to an agent from the database.
        
        Phase 3: Tools Integration
        Returns tool metadata for agent's assigned tools (only enabled ones).
        """
        try:
            # Query agent_tool_assignments with eagerly loaded tool data
            from sqlalchemy.orm import joinedload
            
            assignments = (
                self.db_session.query(AgentToolAssignment)
                .options(joinedload(AgentToolAssignment.tool))
                .filter(
                    AgentToolAssignment.agent_id == agent_id,
                    AgentToolAssignment.enabled == True
                )
                .all()
            )
            
            tools = []
            for assignment in assignments:
                if assignment.tool:  # Tool exists
                    tools.append({
                        "tool_id": assignment.tool.id,
                        "name": assignment.tool.name,
                        "description": assignment.tool.description,
                        "provider": assignment.tool.provider,
                        "category": assignment.tool.category,
                        "icon": assignment.tool.icon,
                        "mcp_server_url": assignment.tool.mcp_server_url,
                        "capabilities": assignment.tool.capabilities or {},
                        "permissions": assignment.permissions or {},
                        "configuration": assignment.configuration or {},
                        "assigned_at": assignment.assigned_at.isoformat() if assignment.assigned_at else None
                    })
            
            if tools:
                self.logger.info(f"✅ Loaded {len(tools)} tools for agent {agent_id}")
            
            return tools
            
        except Exception as e:
            self.logger.warning(f"Failed to load tools for agent {agent_id}: {e}")
            return []
    
    def get_agent_tool_capability(self, agent_runtime: AgentRuntime, capability: str) -> bool:
        """
        Check if an agent has a specific tool capability.
        
        Phase 3: Used by IntelligentAgentSelector for tool-based matching.
        """
        for tool in agent_runtime.tools:
            tool_capabilities = tool.get("capabilities", {})
            if isinstance(tool_capabilities, dict):
                methods = tool_capabilities.get("methods", [])
                if capability in methods:
                    return True
        
        return False
    
    def get_agent_tools_summary(self, agent_runtime: AgentRuntime) -> Dict[str, Any]:
        """Get summary of agent's tools for display/logging"""
        return {
            "total_tools": len(agent_runtime.tools),
            "tools": [
                {
                    "name": tool.get("name"),
                    "category": tool.get("category"),
                    "provider": tool.get("provider")
                }
                for tool in agent_runtime.tools
            ],
            "categories": list(set(tool.get("category") for tool in agent_runtime.tools if tool.get("category")))
        }
    
    def cleanup(self):
        """Clean up resources"""
        if self.db_session:
            self.db_session.close()


# Convenience function for quick agent creation
async def create_specialized_agent(
    name: str,
    agent_type: str,
    skills: List[str] = None,
    auto_test: bool = True
) -> Dict[str, Any]:
    """
    Quick function to create and test a specialized agent.
    
    Returns agent info and test results.
    """
    factory = AgentFactory()
    
    try:
        # Create agent
        agent = await factory.create_agent(
            name=name,
            agent_type=agent_type,
            skills=skills,
            auto_verify=True
        )
        
        result = {
            "agent": {
                "id": agent.agent_id,
                "name": agent.metadata.name,
                "type": agent.agent_type.value,
                "skills": agent.metadata.skills,
                "status": "created"
            }
        }
        
        # Run tests if requested
        if auto_test:
            test_results = await factory.test_agent_capabilities(agent)
            result["tests"] = test_results
        
        return result
    
    finally:
        factory.cleanup()


# Example usage and testing
if __name__ == "__main__":
    async def test_agent_factory():
        """Test the agent factory with REAL LLM connections"""
        
        print("=" * 60)
        print("AGENT FACTORY TEST - REAL LLM CONNECTIONS")
        print("=" * 60)
        
        factory = AgentFactory()
        
        try:
            # Test 1: Create a Code Architect agent
            print("\n1. Creating Code Architect agent...")
            architect = await factory.create_agent(
                metadata=AgentMetadata(
                    name="CodeMaster",
                    agent_type="code_architect",
                    description="Expert in code architecture and design",
                    skills=["code_analysis", "api_design", "system_design"]
                ),
                auto_verify=True
            )
            print(f"✓ Created: {architect.metadata.name} (ID: {architect.agent_id})")
            
            # Test 2: Execute a real task
            print("\n2. Executing code review task...")
            code_task = """
            Review this Python function and suggest improvements:
            
            def calc(x, y, op):
                if op == '+':
                    return x + y
                elif op == '-':
                    return x - y
                elif op == '*':
                    return x * y
                elif op == '/':
                    return x / y
            """
            
            result = await factory.execute_with_prompt(architect, code_task)
            
            if result["status"] == "success":
                print(f"✓ Task executed successfully")
                print(f"  Execution time: {result['execution']['time']:.2f}s")
                print(f"  Tokens used: {result['execution']['tokens_used']}")
                print(f"  Model: {result['execution']['model']}")
                print(f"  Provider: {result['execution']['provider']}")
                print("\n  Response preview:")
                print("  " + result["result"][:200] + "...")
            else:
                print(f"✗ Task failed: {result['error']}")
            
            # Test 3: Create a Security Expert
            print("\n3. Creating Security Expert agent...")
            security_expert = await factory.create_agent(
                metadata=AgentMetadata(
                    name="SecGuardian",
                    agent_type="security_expert",
                    description="Expert in security auditing and testing",
                    skills=["security_audit", "penetration_testing"]
                ),
                auto_verify=True
            )
            print(f"✓ Created: {security_expert.metadata.name} (ID: {security_expert.agent_id})")
            
            # Test 4: Security analysis task
            print("\n4. Executing security analysis...")
            security_task = "What are the top 3 security vulnerabilities in web applications?"
            
            sec_result = await factory.execute_with_prompt(security_expert, security_task)
            
            if sec_result["status"] == "success":
                print(f"✓ Security analysis completed")
                print(f"  Tokens used: {sec_result['execution']['tokens_used']}")
            
            # Test 5: Get agent status
            print("\n5. Checking agent status...")
            status = await factory.get_agent_status(architect.agent_id)
            print(f"✓ Agent status retrieved")
            print(f"  Lifecycle: {status['agent']['lifecycle_state']}")
            print(f"  Executions: {status['runtime']['execution_count']}")
            print(f"  Total tokens: {status['runtime']['total_tokens_used']}")
            
            # Test 6: Run capability tests
            print("\n6. Running capability tests...")
            test_results = await factory.test_agent_capabilities(architect)
            print(f"✓ Tests completed")
            print(f"  Success rate: {test_results['summary']['success_rate']*100:.0f}%")
            print(f"  Total time: {test_results['summary']['total_time']:.2f}s")
            print(f"  Total tokens: {test_results['summary']['total_tokens']}")
            
            print("\n" + "=" * 60)
            print("ALL TESTS COMPLETED SUCCESSFULLY")
            print("Agents are connected to REAL LLM services")
            print("=" * 60)
            
        except Exception as e:
            print(f"\n✗ Test failed: {str(e)}")
            import traceback
            traceback.print_exc()
        
        finally:
            factory.cleanup()
    
    # Run the test
    asyncio.run(test_agent_factory())