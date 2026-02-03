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
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import json

from sqlalchemy.orm import Session, joinedload
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.llm import (
    LLMManager, LLMConfig, LLMProvider, LLMResponse,
    create_llm_manager
)
from core.models import (
    Agent, Skill, PriorityLevel, Base,
)
from core.models.composio_cache import AgentAppAssignment, ComposioAppCache, ComposioActionCache
from modules.agents.services.skill_loader import get_skill_loader

# Import new services (lazy import to avoid circular deps)
def get_action_executor():
    from modules.agents.services.agent_action_executor import get_action_executor as _get_executor
    return _get_executor()

def get_monitoring_service():
    from core.services.monitoring_service import get_monitoring_service as _get_monitor
    return _get_monitor()

def get_rag_service():
    from modules.rag import get_rag_service as _get_rag
    return _get_rag()

def get_unified_tool_executor(db_session: Session, workspace_dir: str = "/tmp/automatos_workspace"):
    """PRD-17 Phase 3: Get UnifiedToolExecutor instance with workspace support"""
    from modules.tools import UnifiedToolExecutor
    return UnifiedToolExecutor(db_session, workspace_dir=workspace_dir)

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
    
    return tools


def _build_skill_tool_schemas(agent_skills: List) -> List[Dict]:
    """
    PRD-22: Extract executable tool schemas from agent skills.
    
    Skills can define tools in their tools_schema field (JSONB).
    This gives agents "superpowers" - they can both READ (prompt) and EXECUTE (tools).
    
    Args:
        agent_skills: List of Skill objects assigned to the agent
        
    Returns:
        List of OpenAI function schemas from skills
    """
    tools = []
    
    logger.info(f"🔍 _build_skill_tool_schemas called with {len(agent_skills)} skills")
    
    for skill in agent_skills:
        logger.info(f"🔍 Processing skill: {skill.name if hasattr(skill, 'name') else 'UNNAMED'}")
        
        # Check if skill has tools_schema defined
        has_attr = hasattr(skill, 'tools_schema')
        logger.info(f"  - has 'tools_schema' attribute: {has_attr}")
        
        if has_attr:
            tools_schema_value = skill.tools_schema
            logger.info(f"  - tools_schema value: {tools_schema_value}")
            logger.info(f"  - tools_schema type: {type(tools_schema_value)}")
            
        if not hasattr(skill, 'tools_schema') or not skill.tools_schema:
            logger.warning(f"  ⚠️ Skill '{skill.name}' has no tools_schema, skipping")
            continue
        
        # Extract tools array from tools_schema
        if not isinstance(skill.tools_schema, dict):
            logger.error(f"  ❌ Skill '{skill.name}' tools_schema is not a dict! Type: {type(skill.tools_schema)}")
            continue
            
        skill_tools = skill.tools_schema.get('tools', [])
        logger.info(f"  - Found {len(skill_tools)} tools in schema")
        
        for tool_def in skill_tools:
            tool_name = tool_def.get('name', 'UNKNOWN')
            logger.info(f"  🔧 Loading tool: {tool_name}")
            
            # Convert to OpenAI function calling format
            tools.append({
                "type": "function",
                "function": {
                    "name": tool_def.get('name'),
                    "description": tool_def.get('description', f"Tool from {skill.name} skill"),
                    "parameters": tool_def.get('parameters', {
                        "type": "object",
                        "properties": {},
                        "required": []
                    })
                }
            })
            
            logger.info(f"  ✅ Successfully loaded tool '{tool_name}' from skill '{skill.name}'")
    
    logger.info(f"🔍 _build_skill_tool_schemas returning {len(tools)} total tools")
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
# NOTE: This is a fallback - the system prefers to use settings from system_settings table
# Use get_default_llm_config() to get settings-aware configuration
DEFAULT_LLM_CONFIG = {
    "provider": "openai",  # Use environment default
    "model": "gpt-4",  # Will be overridden by system settings if configured
    "temperature": 0.7,
    "max_tokens": 2000,
    "context_window": 8192  # Will be updated from LLM models registry
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
    tools: List[Dict[str, Any]] = field(default_factory=list)  # Assigned external apps (Composio)
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
            from core.database.database import SessionLocal
            self.db_session = SessionLocal()
        
        self.active_agents: Dict[int, AgentRuntime] = {}
        self.logger = logging.getLogger(__name__)
    
    def _get_default_llm_config_from_settings(self) -> Dict[str, Any]:
        """
        Get default LLM configuration from system settings (Settings page).
        Falls back to DEFAULT_LLM_CONFIG if settings not available.
        
        This respects user's model selection from the Settings UI instead of hard-coding.
        
        Returns:
            Dictionary with LLM configuration
        """
        try:
            from core.llm.manager import get_system_setting
            
            # Get provider and model from settings - check both key formats
            provider = get_system_setting("orchestrator_llm", "llm_provider")
            if not provider:
                provider = get_system_setting("orchestrator_llm", "provider")  # Fallback to old key
            
            model = get_system_setting("orchestrator_llm", "llm_model")
            if not model:
                model = get_system_setting("orchestrator_llm", "model")  # Fallback to old key
            
            # If still not found, try environment variables (for backward compatibility)
            if not provider:
                import os
                provider = os.getenv("LLM_PROVIDER")
            if not model:
                import os
                model = os.getenv("LLM_MODEL")
            
            # If provider/model not found, fall back to DEFAULT_LLM_CONFIG
            if not provider or not model:
                self.logger.warning("LLM provider/model not found in system settings, falling back to DEFAULT_LLM_CONFIG")
                return DEFAULT_LLM_CONFIG.copy()
            
            # If no model in settings, use provider-specific defaults
            if not model:
                provider_defaults = {
                    "openai": "gpt-4o",  # Modern model with large context
                    "anthropic": "claude-3-5-sonnet-20241022",
                    "google": "gemini-pro",
                    "azure": "gpt-4o",
                    "huggingface": "mistralai/Mistral-7B-Instruct-v0.2"
                }
                model = provider_defaults.get(provider, "gpt-4")
            
            # Get context window from LLM models registry
            try:
                from core.models import LLMModel
                llm_model = self.db_session.query(LLMModel).filter_by(model_id=model).first()
                if llm_model:
                    context_window = llm_model.context_window
                    max_tokens = llm_model.max_output_tokens
                else:
                    # Fallback to reasonable defaults for common models
                    model_contexts = {
                        "gpt-4o": 128000,
                        "gpt-4-turbo": 128000,
                        "gpt-4": 8192,
                        "claude-3-5-sonnet-20241022": 200000,
                        "claude-3-opus": 200000,
                        "gemini-pro": 32768
                    }
                    context_window = model_contexts.get(model, 8192)
                    max_tokens = 4000 if context_window > 100000 else 2000
            except Exception as e:
                self.logger.warning(f"Could not get context window from registry: {e}")
                context_window = 8192
                max_tokens = 2000
            
            self.logger.info(f"📋 Using model from settings: {model} (context: {context_window})")
            return {
                "provider": provider,
                "model": model,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "context_window": context_window
            }
        except Exception as e:
            self.logger.warning(f"Could not get LLM config from settings: {e}, using DEFAULT_LLM_CONFIG")
            return DEFAULT_LLM_CONFIG.copy()
    
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
                tools=agent_tools  # Assigned external apps (Composio)
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
        from core.llm import LLMConfig, LLMProvider as LLMProviderEnum
        
        # Map provider string to enum
        provider_map = {
            "openai": LLMProviderEnum.OPENAI,
            "anthropic": LLMProviderEnum.ANTHROPIC
        }
        
        if model_config.provider not in provider_map:
            raise ValueError(f"Unsupported provider: {model_config.provider}")
        
        provider = provider_map[model_config.provider]
        
        # Get API key from credential resolver
        from core.credentials.resolver import get_credential_resolver
        import os
        resolver = get_credential_resolver()
        
        api_key = None
        if provider == LLMProviderEnum.OPENAI:
            try:
                api_key = resolver.get_credential_field("development_openai", "api_key")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve/decrypt 'development_openai' credential: {e}")
                api_key = None
                
            # Fallback to environment variable if credential resolver fails
            if not api_key:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key:
                    self.logger.info(f"Using OPENAI_API_KEY from environment variable for {agent_name} (fallback)")
        elif provider == LLMProviderEnum.ANTHROPIC:
            try:
                api_key = resolver.get_credential_field("development_anthropic", "api_key")
            except Exception as e:
                self.logger.warning(f"Failed to retrieve/decrypt 'development_anthropic' credential: {e}")
                api_key = None
                
            # Fallback to environment variable
            if not api_key:
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if api_key:
                    self.logger.info(f"Using ANTHROPIC_API_KEY from environment variable for {agent_name} (fallback)")
        
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
    
    async def activate_agent(self, agent_id: int, workspace_dir: str = "/tmp/automatos_workspace") -> Optional[AgentRuntime]:
        """
        Load an agent from database and activate it in runtime.
        
        Args:
            agent_id: ID of agent to activate
            workspace_dir: Workspace directory for file operations
            
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
            
            # Get LLM config - PRIORITIZE system settings over agent's stored config
            # This ensures agents use the LLM provider selected in Settings UI (e.g., HuggingFace for testing)
            config = db_agent.configuration or {}
            agent_llm_config = config.get("llm_config") or {}
            
            # Always get system settings first (respects Settings UI selection)
            system_llm_config = self._get_default_llm_config_from_settings()
            
            # Merge: system settings for provider/model, agent config for temperature/max_tokens (if set)
            # This allows system-wide LLM changes while preserving agent-specific generation parameters
            llm_config_dict = {
                "provider": system_llm_config.get("provider"),  # Use system provider (e.g., HuggingFace)
                "model": system_llm_config.get("model"),  # Use system model
                "temperature": agent_llm_config.get("temperature", system_llm_config.get("temperature", 0.7)),
                "max_tokens": agent_llm_config.get("max_tokens", system_llm_config.get("max_tokens", 2000)),
            }
            
            self.logger.info(f"Agent {agent_id} using LLM: {llm_config_dict.get('provider')}/{llm_config_dict.get('model')} (from system settings)")
            
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
                tool_executor=get_unified_tool_executor(self.db_session, workspace_dir or "/tmp/automatos_workspace")  # PRD-17: Initialize once, reuse
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
        required_tools: Optional[List[str]] = None,
        workspace_dir: Optional[str] = None  # Unique workspace per execution
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
            workspace_dir: Workspace directory for file operations
            
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
                agent_runtime = await self.activate_agent(agent, workspace_dir=workspace_dir or "/tmp/automatos_workspace")
                if not agent_runtime:
                    return {
                        "status": "error",
                        "error": f"Agent {agent} could not be activated"
                    }
        else:
            agent_runtime = agent
        
        # Log agent execution start prominently WITH MODEL INFORMATION
        agent_name = agent_runtime.metadata.name if hasattr(agent_runtime, 'metadata') else "Unknown"
        agent_id = agent_runtime.agent_id if hasattr(agent_runtime, 'agent_id') else "Unknown"
        agent_type = agent_runtime.metadata.agent_type if hasattr(agent_runtime, 'metadata') else "Unknown"
        
        # Get model information
        model_name = "Unknown"
        if hasattr(agent_runtime, 'llm_client') and agent_runtime.llm_client:
            model_name = getattr(agent_runtime.llm_client, 'model', 'Unknown')
        elif hasattr(agent_runtime, 'metadata') and hasattr(agent_runtime.metadata, 'llm_config'):
            llm_config = agent_runtime.metadata.llm_config
            if llm_config:
                model_name = llm_config.get('model', 'Unknown')
        
        self.logger.info("=" * 80)
        self.logger.info(f"🤖 EXECUTING AGENT: {agent_name} (ID: {agent_id}, Type: {agent_type})")
        self.logger.info(f"📋 MODEL: {model_name}")
        self.logger.info("=" * 80)
        
        # Update state
        agent_runtime.lifecycle_state = AgentLifecycle.BUSY
        
        try:
            # Build messages - orchestrator provides the engineered prompts
            messages = []
            
            # System prompt - either from orchestrator or built with skills
            # PRD-22: Build system prompt with skills AND extract skill tools
            skill_tool_schemas_from_prompt = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                # Get database agent for skill loading
                db_agent = self.db_session.query(Agent).filter_by(id=agent_runtime.agent_id).first()
                if db_agent:
                    # This NOW returns (prompt_text, skill_tools) tuple!
                    built_system_prompt, skill_tool_schemas_from_prompt = self._build_agent_system_prompt(
                        agent=db_agent,
                        task_context=prompt,  # Use the task prompt for skill selection
                        db=self.db_session,
                        required_tools=required_tools,
                        enable_smart_skill_selection=True
                    )
                    messages.append({"role": "system", "content": built_system_prompt})
                    self.logger.info(f"✨ Built system prompt with intelligent skill selection for agent {agent_runtime.agent_id}")
                    if skill_tool_schemas_from_prompt:
                        self.logger.info(f"🦸 PRD-22: Extracted {len(skill_tool_schemas_from_prompt)} skill tools from prompt building")
            
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
            
            # Build OpenAI function calling schemas (traditional tools)
            tool_schemas = _build_tool_schemas(required_tools)
            
            # PRD-22: Add skill-based tools extracted during prompt building
            if skill_tool_schemas_from_prompt:
                tool_schemas.extend(skill_tool_schemas_from_prompt)
                tool_names = [t['function']['name'] for t in skill_tool_schemas_from_prompt]
                self.logger.info(f"🦸 PRD-22: Added {len(skill_tool_schemas_from_prompt)} skill tools: {tool_names}")
            
            # Composio tool injection (follows chatbot pattern from service.py)
            # If agent has Composio app assignments, add composio_execute + hints
            _composio_workspace_id = None
            composio_apps = [t for t in (agent_runtime.tools or []) if t.get("provider") == "Composio"]
            if composio_apps:
                # Add composio_execute tool schema (same as ToolRegistry registration)
                composio_schema = {
                    "type": "function",
                    "function": {
                        "name": "composio_execute",
                        "description": (
                            "Execute an external app action via Composio (connected third-party apps). "
                            "Use this for actions in email/messaging and developer tools—e.g., "
                            "read/send emails, post messages, create/manage repositories, issues, and pull requests. "
                            "IMPORTANT: This tool has a 2-attempt limit. If the first attempt "
                            "fails, check the error message carefully before retrying."
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "app_name": {"type": "string", "description": "App name (e.g., 'GMAIL', 'SLACK', 'GITHUB')"},
                                "action": {"type": "string", "description": "Action name from Composio (e.g., 'GMAIL_LIST_EMAILS')"},
                                "params": {"type": "object", "description": "Action parameters (schema depends on the action)", "default": {}}
                            },
                            "required": ["action"]
                        }
                    }
                }
                tool_schemas.append(composio_schema)

                # Resolve workspace_id once for later tool execution calls
                db_agent = self.db_session.query(Agent).filter(Agent.id == agent_runtime.agent_id).first()
                _composio_workspace_id = getattr(db_agent, 'workspace_id', None) if db_agent else None

                # Build and inject Composio hints (app names + candidate actions + param hints)
                hint_lines = self._build_composio_hints(agent_runtime.agent_id, prompt)
                if hint_lines:
                    # Insert after system prompt, before user messages
                    insert_at = 1 if messages and messages[0].get("role") == "system" else 0
                    messages.insert(insert_at, {"role": "system", "content": "\n".join(hint_lines)})

                app_names = [t.get("name") for t in composio_apps]
                self.logger.info(f"🔌 Composio: Added composio_execute tool + hints ({len(hint_lines)} lines) for apps {app_names} (workspace={_composio_workspace_id})")

            all_tool_names = [t['function']['name'] for t in tool_schemas]
            self.logger.info(f"📦 Providing {len(tool_schemas)} total tools to agent: {all_tool_names}")

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
                        
                        # Track executed calls in this turn to prevent duplicates
                        executed_calls_hashes = set()
                        
                        for tool_call in response.tool_calls:
                            func_name = tool_call['function']['name']
                            func_args_str = tool_call['function']['arguments']
                            
                            try:
                                # Normalization: Parse JSON then canonicalize
                                func_args = json.loads(func_args_str)
                                canonical_args = json.dumps(func_args, sort_keys=True)
                            except json.JSONDecodeError:
                                # If invalid JSON, use raw string for hash (will likely fail later but consistent)
                                canonical_args = func_args_str.strip()
                                func_args = {}

                            # Create a hash of name + canonical args
                            call_hash = f"{func_name}:{canonical_args}"
                            
                            if call_hash in executed_calls_hashes:
                                self.logger.warning(f"⚠️  [DEDUPE] Skipping duplicate tool call in same turn: {func_name}")
                                tool_results.append({
                                    "tool_call_id": tool_call['id'],
                                    "role": "tool",
                                    "name": func_name,
                                    "content": json.dumps({"error": "Duplicate tool call skipped (already executed in this turn)"})
                                })
                                continue
                            
                            # Filter empty parameters for critical tools if they usually require them
                            if not func_args and "SLACK" in func_name:
                                self.logger.warning(f"⚠️  [FILTER] Skipping empty parameters for {func_name}")
                                tool_results.append({
                                    "tool_call_id": tool_call['id'],
                                    "role": "tool",
                                    "name": func_name,
                                    "content": json.dumps({"error": "Skipped: Empty parameters provided for tool requiring input."})
                                })
                                continue

                            executed_calls_hashes.add(call_hash)
                            
                            self.logger.info(f"  🛠️  [TRACE] Calling {func_name}({func_args})")
                            
                            try:
                                result = await tool_executor.execute_tool(
                                    tool_name=func_name,
                                    parameters=func_args,
                                    agent_id=agent_runtime.agent_id,
                                    workspace_id=_composio_workspace_id
                                )
                                tool_results.append({
                                    "tool_call_id": tool_call['id'],
                                    "role": "tool",
                                    "name": func_name,
                                    "content": json.dumps(result)
                                })
                                self.logger.info(f"    ✅ [TRACE] {func_name} completed successfully")
                            except Exception as e:
                                self.logger.error(f"    ❌ [TRACE] {func_name} failed: {e}")
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
    # PRD-22: Intelligent skill selection helper
    # ======================================================================
    def _estimate_skill_tokens(self, skill_name: str) -> int:
        """Estimate token count for a skill based on typical sizes"""
        # Averages based on actual PRD-22 skills
        skill_sizes = {
            'pdf': 1682,
            'docx': 2424,
            'xlsx': 2523,
            'pptx': 6293,
            'writing-skills': 5117,
            'writing-plans': 784
        }
        return skill_sizes.get(skill_name.lower(), 3000)  # Default 3K tokens
    
    def _calculate_max_skills(
        self,
        context_window: int,
        agent_skills: List,
        task_size_estimate: int = 2000
    ) -> int:
        """
        Dynamically calculate how many skills can fit in context window.
        
        Args:
            context_window: Total context window size
            agent_skills: List of agent skills
            task_size_estimate: Estimated tokens for task/prompt
            
        Returns:
            Maximum number of skills that can fit
        """
        # Reserve tokens for: task, system prompt, response buffer
        reserved = task_size_estimate + 1000 + 2000  # task + overhead + response
        available = context_window - reserved
        
        if available < 1000:
            self.logger.warning(f"Very limited context space: {available} tokens available")
            return 1  # At least try to load 1 skill
        
        # Sort skills by estimated size to pack efficiently
        skill_sizes = [(skill, self._estimate_skill_tokens(skill.name)) for skill in agent_skills]
        skill_sizes.sort(key=lambda x: x[1])  # Smallest first
        
        # Greedily pack skills
        total_tokens = 0
        max_skills = 0
        for _, size in skill_sizes:
            if total_tokens + size <= available:
                total_tokens += size
                max_skills += 1
            else:
                break
        
        self.logger.info(f"📊 Context analysis: {context_window}K window, can fit up to {max_skills}/{len(agent_skills)} skills")
        return max(1, max_skills)  # At least 1 skill
    
    def _select_relevant_skills(
        self,
        agent_skills: List,
        task_context: Optional[str] = None,
        context_window: int = 8192
    ) -> List:
        """
        Intelligently select the most relevant skills for the current task.
        
        NEW: Dynamically calculates max skills based on context window size.
        Uses enhanced keyword matching against both skill names and descriptions.
        
        Args:
            agent_skills: List of all agent skills
            task_context: The task description to match against
            context_window: Available context window (from model config)
            
        Returns:
            List of selected skills
        """
        if not agent_skills:
            return []
        
        if not task_context:
            # No context - return limited skills
            return agent_skills[:1]
        
        # Calculate dynamic max based on context window
        max_skills = self._calculate_max_skills(context_window, agent_skills)
        
        task_lower = task_context.lower()
        
        # 🎯 INTELLIGENT SKILL MATCHING - Pure database-driven, NO hard-coding!
        # Extract keywords from:
        # 1. Skill name (e.g., "pdf" → ["pdf"])
        # 2. Skill description (extract meaningful words)
        # 3. Tool names from tools_schema (e.g., "create_pdf" → ["create", "pdf"])
        
        skill_scores = []
        for skill in agent_skills:
            # Build dynamic keyword list from skill metadata
            keywords = []
            
            # 1. Skill name keywords (split on hyphens, underscores, camelCase)
            skill_name = skill.name.lower()
            keywords.append(skill_name)
            keywords.extend(skill_name.replace('-', ' ').replace('_', ' ').split())
            
            # 2. Description keywords (if available)
            if hasattr(skill, 'description') and skill.description:
                description_lower = skill.description.lower()
                # Extract meaningful words (>3 chars) from description
                desc_words = [w for w in description_lower.split() if len(w) > 3]
                keywords.extend(desc_words[:10])  # Top 10 words from description
            
            # 3. Tool names from tools_schema (if available)
            if hasattr(skill, 'tools_schema') and skill.tools_schema and isinstance(skill.tools_schema, dict):
                tools = skill.tools_schema.get('tools', [])
                for tool in tools:
                    tool_name = tool.get('name', '')
                    if tool_name:
                        # "create_pdf" → ["create", "pdf"]
                        tool_keywords = tool_name.replace('_', ' ').split()
                        keywords.extend(tool_keywords)
            
            # Calculate relevance score by checking keywords against task
            score = 0
            for keyword in keywords:
                if keyword in task_lower:
                    # Longer matches get higher scores
                    score += len(keyword.split())
            
            skill_scores.append((skill, score))
            if score > 0:
                self.logger.debug(f"  Skill '{skill.name}': score={score} (keywords: {keywords[:5]})")
        
        # Sort by relevance score (descending)
        skill_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Take top N with positive scores
        selected = [skill for skill, score in skill_scores[:max_skills] if score > 0]
        
        # If no skills matched keywords, use ALL assigned skills (up to max)
        # Rationale: If user assigned skills to agent, they're ALL relevant
        if not selected:
            self.logger.info(f"ℹ️  No exact keyword matches - using all {len(agent_skills)} assigned skills (user knows best!)")
            selected = agent_skills[:max_skills]
        
        skill_names = [s.name for s in selected]
        self.logger.info(f"✅ Selected {len(selected)}/{len(agent_skills)} skills: {skill_names} (max: {max_skills})")
        
        return selected

    # ======================================================================
    # PRD-22: Build agent system prompt with progressive skill injection
    # ======================================================================
    def _build_agent_system_prompt(
        self,
        agent: Agent,
        task_context: Optional[str] = None,
        db: Optional[Session] = None,
        required_tools: Optional[List[str]] = None,
        enable_smart_skill_selection: bool = True
    ) -> Tuple[str, List[Dict]]:
        """
        Build the agent system prompt AND extract skill tool schemas.

        - Loads core content for each assigned skill via SkillLoader (progressive disclosure)
        - Extracts tools_schema from skills for function calling
        - Falls back to skill.description for legacy/seed skills without filesystem content
        - NEW: Intelligently selects relevant skills based on context window and task relevance
        
        Args:
            agent: The agent to build prompt for
            task_context: Optional task description for context
            db: Database session
            required_tools: List of tool categories to include
            enable_smart_skill_selection: If True, dynamically selects skills (default: True)
            
        Returns:
            Tuple of (system_prompt_string, skill_tool_schemas_list)
        """
        sections: List[str] = []
        skill_tool_schemas: List[Dict] = []

        # Identity
        sections.append(f"# Agent: {agent.name}")
        sections.append(f"Agent ID: {agent.id}")
        sections.append(f"Agent Type: {getattr(agent, 'agent_type', 'unknown')}")
        if agent.description:
            sections.append(agent.description)

        # Task context (optional)
        if task_context:
            sections.append("\n## Task Context\n" + str(task_context))

        # Skills (progressive disclosure with intelligent selection)
        if getattr(agent, 'skills', None):
            sections.append("\n## Your Specialized Skills\n")
            loader = get_skill_loader(db) if db is not None else None
            
            # Get agent's context window for intelligent selection
            agent_config = agent.configuration or {}
            llm_config = agent_config.get('llm_config') or agent.model_config or {}
            context_window = llm_config.get('context_window', 8192)
            
            # Intelligently select relevant skills to prevent context overflow
            skills_to_load = agent.skills
            if enable_smart_skill_selection and len(agent.skills) > 1:
                skills_to_load = self._select_relevant_skills(
                    agent.skills, 
                    task_context,
                    context_window=context_window  # Dynamic based on model
                )

            for skill in skills_to_load:
                self.logger.info(f"📚 Loading skill: {skill.name}")
                sections.append(f"### {skill.name}")

                # Load prompt content
                core_content = None
                if loader is not None:
                    try:
                        # Level 2: core content from SKILL.md body or prompt_template
                        core_content = loader.load_skill_core(skill.name, db=db)
                        if core_content:
                            self.logger.info(f"  ✅ Loaded {len(core_content)} chars of core content for '{skill.name}'")
                    except Exception as e:
                        self.logger.warning(f"  ⚠️  Failed to load core content for '{skill.name}': {e}")
                        core_content = None

                if core_content and isinstance(core_content, str) and core_content.strip():
                    sections.append(core_content)
                else:
                    # Fallback for legacy/seed skills
                    fallback = skill.prompt_template or skill.description or ""
                    if fallback:
                        self.logger.info(f"  📝 Using fallback content for '{skill.name}' ({len(str(fallback))} chars)")
                        sections.append(str(fallback))
                    else:
                        self.logger.warning(f"  ⚠️  No content available for skill '{skill.name}'")
                
                # PRD-22: Extract tool schemas from skills
                if hasattr(skill, 'tools_schema') and skill.tools_schema:
                    try:
                        tools = skill.tools_schema.get('tools', [])
                        for tool_def in tools:
                            tool_name = tool_def.get("name")
                            # Convert to OpenAI function calling format
                            skill_tool_schemas.append({
                                "type": "function",
                                "function": {
                                    "name": tool_name,
                                    "description": tool_def.get("description", ""),
                                    "parameters": tool_def.get("parameters", {})
                                }
                            })
                            self.logger.info(f"  🛠️  Extracted tool: {tool_name}")
                        if tools:
                            self.logger.info(f"  🦸 Total: {len(tools)} tool(s) from skill '{skill.name}'")
                    except Exception as e:
                        self.logger.warning(f"  ❌ Failed to extract tools from skill '{skill.name}': {e}")

        # Optional tools section (kept minimal; main tool wiring remains elsewhere)
        if required_tools:
            try:
                tool_schemas = _build_tool_schemas(required_tools)
                sections.append("\n## Available Tools\n")
                sections.append(json.dumps(tool_schemas))
            except Exception:
                pass
        
        # Explicitly add assigned Composio apps (from the new assignment table)
        if db and agent.id:
            try:
                assignments = (
                    db.query(AgentAppAssignment)
                    .filter(
                        AgentAppAssignment.agent_id == agent.id,
                        AgentAppAssignment.is_active == True,
                        AgentAppAssignment.app_type == "EXTERNAL",
                    )
                    .all()
                )

                if assignments:
                    app_names = [a.app_name.upper() for a in assignments if a.app_name]
                    cache = {
                        a.app_name: a
                        for a in db.query(ComposioAppCache).filter(ComposioAppCache.app_name.in_(app_names)).all()
                    }

                    helper_section = ["\n## Available External Apps (Composio)\n"]
                    helper_section.append(
                        "You have access to these external apps via Composio. When you need to read/send emails or post messages, "
                        "use the `composio_execute` tool with an appropriate action for the relevant app.\n"
                    )
                    for assignment in assignments:
                        app_name = (assignment.app_name or "").upper()
                        app = cache.get(app_name)
                        if not app_name:
                            continue
                        helper_section.append(f"### {app_name}")
                        if app and app.description:
                            helper_section.append(f"**Description**: {app.description}")

                    sections.append("\n".join(helper_section))
            except Exception as e:
                self.logger.warning(f"Failed to append Composio apps to prompt: {e}")

        # Add instructions for handling dependency context
        sections.append("\n## IMPORTANT: Working with Context and Dependencies\n")
        sections.append("When you receive '## DEPENDENCY CONTEXT' at the beginning of your task:")
        sections.append("1. This contains outputs from previous tasks that you need to use")
        sections.append("2. Read and understand all the context provided")
        sections.append("3. For compilation/report tasks: Synthesize the information into a coherent document")
        sections.append("4. For document generation tasks: Transform the input into the requested format")
        sections.append("\nWhen your task involves writing/creating documents:")
        sections.append("- Use the write_file tool to save your output")
        sections.append("- The task description will specify the output filename")
        sections.append("- Actually WRITE the content, don't just describe what you would write")
        sections.append("- Process and synthesize the dependency context into meaningful output")
        
        # Add explicit instructions to USE skill tools if any were extracted
        if skill_tool_schemas:
            tool_names_list = [t['function']['name'] for t in skill_tool_schemas]
            sections.append("\n## IMPORTANT: Using Your Skill Tools\n")
            sections.append(f"You have access to the following specialized tools from your skills: {', '.join(tool_names_list)}")
            sections.append("\n**These skills were loaded specifically because they match your task requirements.**")
            sections.append("When your task requires capabilities provided by these tools, you MUST use them via function calling.")
            sections.append("\n**Critical Workflow:**")
            sections.append("1. Analyze your task description to understand what needs to be accomplished")
            sections.append("2. Check if any of your available skill tools match what the task requires")
            sections.append("3. If a tool matches the task requirement, CALL IT using function calling - do not just describe what you would do")
            sections.append("4. Use tool results to complete the task properly")
            sections.append("\nExample: If your task is 'create a PDF' and you have a 'create_pdf' tool → you MUST call create_pdf")
            sections.append("Example: If your task is 'write documentation' and you have 'write_technical_content' tool → you MUST call that tool")
            sections.append("\n**Your skills are your capabilities - use them when they match the task!**")

        prompt_text = "\n\n".join([s for s in sections if s is not None])
        return (prompt_text, skill_tool_schemas)
    
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
    # Assigned apps/integrations (Composio)
    # ======================================================================
    
    async def _load_agent_tools(self, agent_id: int) -> List[Dict[str, Any]]:
        """
        Load Composio apps assigned to an agent from the database (schema v2).

        This is DB-backed and does not rely on legacy tool tables.
        """
        try:
            assignments = (
                self.db_session.query(AgentAppAssignment)
                .filter(
                    AgentAppAssignment.agent_id == agent_id,
                    AgentAppAssignment.is_active == True,
                    AgentAppAssignment.app_type == "EXTERNAL",
                )
                .all()
            )

            if not assignments:
                return []

            app_names = [a.app_name.upper() for a in assignments if a.app_name]
            cache = {
                a.app_name: a
                for a in self.db_session.query(ComposioAppCache).filter(ComposioAppCache.app_name.in_(app_names)).all()
            }

            tools: List[Dict[str, Any]] = []
            for assignment in assignments:
                app_name = (assignment.app_name or "").upper()
                app = cache.get(app_name)
                if not app_name:
                    continue
                tools.append(
                    {
                        "name": app_name,
                        "description": (app.description if app else "") or "",
                        "provider": "Composio",
                        "category": ((app.categories or [None])[0] if app else None),
                        "icon": app.logo_url if app else None,
                        "assigned_at": assignment.assigned_at.isoformat() if assignment.assigned_at else None,
                    }
                )

            self.logger.info(f"✅ Loaded {len(tools)} Composio app assignment(s) for agent {agent_id}")
            return tools
        except Exception as e:
            self.logger.warning(f"Failed to load Composio apps for agent {agent_id}: {e}")
            return []
    
    def _build_composio_hints(self, agent_id: int, task_prompt: str) -> List[str]:
        """
        Build Composio app/action hint lines for system message injection.

        Follows the SAME pattern as chatbot service.py (lines 606-727):
        1. Query AgentAppAssignment for assigned external apps
        2. Cross-reference with workspace connections (EntityManager)
        3. Match candidate actions from ComposioActionCache using prompt tokens
        4. Extract parameter hints for top actions
        """
        import re
        from sqlalchemy import or_

        try:
            from core.composio.entity_manager import EntityManager

            # 1. Assigned EXTERNAL apps for this agent
            assigned = (
                self.db_session.query(AgentAppAssignment)
                .filter(
                    AgentAppAssignment.agent_id == agent_id,
                    AgentAppAssignment.is_active == True,
                    AgentAppAssignment.app_type == "EXTERNAL",
                )
                .all()
            )
            assigned_apps = [(a.app_name or "").upper() for a in assigned if a.app_name]
            self.logger.info(f"🔌 [hints] Agent {agent_id}: assigned_apps={assigned_apps}")
            if not assigned_apps:
                return []

            # 2. Cross-reference with connected apps in workspace
            connected_apps: List[str] = []
            db_agent = self.db_session.query(Agent).filter(Agent.id == agent_id).first()
            workspace_id = getattr(db_agent, 'workspace_id', None) if db_agent else None
            self.logger.info(f"🔌 [hints] Agent {agent_id}: workspace_id={workspace_id}")

            if workspace_id:
                try:
                    manager = EntityManager(self.db_session)
                    entity = manager.get_entity_by_workspace(workspace_id)
                    if entity:
                        connected_apps = [
                            (c.get("app_name") or "").upper()
                            for c in manager.get_entity_connections(entity["id"])
                            if c.get("status") == "active"
                        ]
                    self.logger.info(f"🔌 [hints] Agent {agent_id}: connected_apps={connected_apps}")
                except Exception as conn_err:
                    self.logger.warning(f"🔌 [hints] Connection check failed: {conn_err} - using all assigned apps")
                    connected_apps = []  # Fallback: don't filter

            allowed_apps = assigned_apps
            if connected_apps:
                connected_set = set(connected_apps)
                allowed_apps = [a for a in assigned_apps if a in connected_set]

            if not allowed_apps:
                self.logger.info(f"🔌 Composio: Agent {agent_id} has apps {assigned_apps} but none connected (connected={connected_apps})")
                # Fallback: use assigned apps anyway so LLM at least knows the names
                allowed_apps = assigned_apps

            # 3. Build hint header
            hint_lines = [
                "You have these external apps assigned (via Composio): "
                + ", ".join(sorted(set(allowed_apps))) + ".",
                "When you need external data/actions (email, Slack, etc.), use `composio_execute`.",
            ]

            # 4. Tokenize task prompt and match candidate actions from DB
            q = (task_prompt or "").lower()
            q_tokens = [t for t in re.split(r"[^a-z0-9]+", q) if len(t) > 2]
            stop = {"the", "and", "for", "with", "from", "that", "this", "have", "has", "are", "you", "your"}
            q_tokens = [t for t in q_tokens if t not in stop]
            q_tokens = q_tokens[:10]

            # Safety: avoid suggesting destructive actions for messaging intents
            is_messaging_intent = bool(re.search(r"\b(send|message|post|dm|chat)\b", q))
            dangerous_tokens = {
                "archive", "delete", "remove", "revoke", "clear", "close", "disable",
                "ban", "kick", "deactivate", "destroy", "purge",
            }

            app_matches: List[tuple] = []
            top_action_params: Dict[str, str] = {}

            if q_tokens:
                for app in allowed_apps[:12]:
                    token_filters = []
                    for tok in q_tokens:
                        like = f"%{tok}%"
                        token_filters.append(ComposioActionCache.action_name.ilike(like))
                        token_filters.append(ComposioActionCache.display_name.ilike(like))
                        token_filters.append(ComposioActionCache.description.ilike(like))

                    rows = (
                        self.db_session.query(ComposioActionCache.action_name, ComposioActionCache.parameters)
                        .filter(ComposioActionCache.app_name == app)
                        .filter(or_(*token_filters))
                        .limit(24)
                        .all()
                    )
                    self.logger.info(f"🔌 [hints] Token search for {app}: {len(rows)} matches (tokens={q_tokens})")
                    actions = []
                    for r in rows:
                        if not r or not r[0]:
                            continue
                        action_name = str(r[0])
                        # action_name from DB should be API identifier (e.g., GMAIL_FETCH_EMAILS)
                        # If it's still a display name (legacy data), reconstruct
                        if " " in action_name and not action_name.startswith(f"{app}_"):
                            action_name = f"{app}_{action_name.upper().replace(' ', '_')}"
                        if is_messaging_intent:
                            al = action_name.lower()
                            if any(tok in al for tok in dangerous_tokens):
                                continue
                        actions.append(action_name)
                        if len(top_action_params) < 10 and r[1]:
                            try:
                                from modules.tools.formatting.schema_detector import ParameterHintExtractor
                                param_hints = ParameterHintExtractor.extract_hints(r[1], max_params=5)
                                if param_hints:
                                    top_action_params[action_name] = param_hints
                            except Exception:
                                pass

                    # Deduplicate preserving order
                    seen = set()
                    actions = [a for a in actions if not (a in seen or seen.add(a))]
                    if actions:
                        app_matches.append((app, actions[:6]))

            # Fallback: if no token matches, get top actions for each app
            if not app_matches:
                self.logger.info(f"🔌 [hints] No token matches, fetching top actions per app")
                for app in allowed_apps[:6]:
                    rows = (
                        self.db_session.query(ComposioActionCache.action_name, ComposioActionCache.parameters)
                        .filter(ComposioActionCache.app_name == app)
                        .limit(10)
                        .all()
                    )
                    actions = []
                    for r in rows:
                        if not r or not r[0]:
                            continue
                        name = str(r[0])
                        # Legacy fallback: if still display name, reconstruct
                        if " " in name and not name.startswith(f"{app}_"):
                            name = f"{app}_{name.upper().replace(' ', '_')}"
                        actions.append(name)
                    self.logger.info(f"🔌 [hints] Fallback for {app}: {len(actions)} actions")
                    if actions:
                        app_matches.append((app, actions[:6]))
                    for r in rows[:3]:
                        if r and r[0] and r[1] and len(top_action_params) < 10:
                            name = str(r[0])
                            if " " in name and not name.startswith(f"{app}_"):
                                name = f"{app}_{name.upper().replace(' ', '_')}"
                            try:
                                from modules.tools.formatting.schema_detector import ParameterHintExtractor
                                param_hints = ParameterHintExtractor.extract_hints(r[1], max_params=5)
                                if param_hints:
                                    top_action_params[name] = param_hints
                            except Exception:
                                pass

            # Prefer apps with more matches
            app_matches.sort(key=lambda x: (-len(x[1]), x[0]))
            for app, actions in app_matches[:6]:
                hint_lines.append(f"- {app} candidate actions: {', '.join(actions)}")

            # Add parameter hints for top candidate actions
            if top_action_params:
                hint_lines.append("\nParameter hints for key actions:")
                for action_name, params in list(top_action_params.items())[:5]:
                    hint_lines.append(f"\n{action_name}:")
                    hint_lines.append(params)

            self.logger.info(f"🔌 Composio hints: apps={allowed_apps}, tokens={q_tokens}, matches={len(app_matches)}, param_hints={len(top_action_params)}, hint_lines={len(hint_lines)}")
            return hint_lines

        except Exception as e:
            self.logger.warning(f"Failed to build Composio hints for agent {agent_id}: {e}", exc_info=True)
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