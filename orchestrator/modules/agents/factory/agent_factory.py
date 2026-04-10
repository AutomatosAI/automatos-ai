"""
Agent Factory — Clean Rewrite
==============================

Pure execution layer for agents. Users define their own agent types.
The orchestrator handles all prompt engineering using Context Engineering.
Multiple agents of different types can run simultaneously.

Tool Source: ONE path — get_tools_for_agent() from tool_router.py.
No hardcoded tool schemas, no legacy JSON action format, no mid-execution discovery.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy.orm import Session

from config import config
from core.llm import LLMConfig, LLMManager, LLMProvider, LLMResponse, create_llm_manager
from core.models import Agent, Base, PriorityLevel, Skill
from core.models.composio_cache import AgentAppAssignment, ComposioAppCache


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy imports (avoid circular deps)
# ---------------------------------------------------------------------------

def get_monitoring_service():
    from core.services.monitoring_service import get_monitoring_service as _get_monitor
    return _get_monitor()


def get_unified_tool_executor(db_session: Session, workspace_dir: str = "/tmp/automatos_workspace"):
    from modules.tools import UnifiedToolExecutor
    return UnifiedToolExecutor(db_session, workspace_dir=workspace_dir)


# ---------------------------------------------------------------------------
# Enums & Dataclasses (KEPT — battle-tested)
# ---------------------------------------------------------------------------

class AgentLifecycle(Enum):
    INITIALIZING = "initializing"
    ACTIVE = "active"
    BUSY = "busy"
    LEARNING = "learning"
    HIBERNATING = "hibernating"
    RETIRED = "retired"


@dataclass
class ModelConfiguration:
    """Complete model configuration for an agent (PRD-15)."""
    provider: str
    model_id: str
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    fallback_model_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model_id": self.model_id,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty,
            "fallback_model_id": self.fallback_model_id,
        }

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ModelConfiguration":
        return ModelConfiguration(
            provider=data.get("provider", "openai"),
            model_id=data.get("model_id", config.LLM_MODEL),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", 2000),
            top_p=data.get("top_p", 1.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            fallback_model_id=data.get("fallback_model_id"),
        )

    @staticmethod
    def get_default() -> "ModelConfiguration":
        return ModelConfiguration(provider=config.LLM_PROVIDER, model_id=config.LLM_MODEL)


@dataclass
class AgentMetadata:
    """User-defined agent metadata — completely flexible."""
    name: str
    agent_type: str
    description: Optional[str] = None
    skills: List[str] = field(default_factory=list)
    model_config: Optional[ModelConfiguration] = None
    # Deprecated — keep for backward compat
    preferred_model: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    context_window: Optional[int] = None
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

    def get_model_config(self) -> ModelConfiguration:
        if self.model_config:
            return self.model_config
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
                max_tokens=self.max_tokens or 2000,
            )
        return ModelConfiguration.get_default()

    def get_llm_config(self) -> Dict[str, Any]:
        mc = self.get_model_config()
        return {
            "provider": mc.provider,
            "model": mc.model_id,
            "temperature": mc.temperature,
            "max_tokens": mc.max_tokens,
            "context_window": self.context_window or 8192,
        }


@dataclass
class ResolvedKey:
    """Result of API key resolution with source metadata."""
    api_key: str
    source: str  # "byok", "platform", "env"
    is_byok: bool
    provider: str = ""


@dataclass
class AgentRuntime:
    """Runtime representation of an agent."""
    agent_id: int
    metadata: AgentMetadata
    llm_manager: LLMManager
    lifecycle_state: AgentLifecycle
    created_at: datetime
    execution_count: int = 0
    total_tokens_used: int = 0
    last_execution: Optional[datetime] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    memory: List[Dict[str, Any]] = field(default_factory=list)
    tools: List[Dict[str, Any]] = field(default_factory=list)  # Composio app assignments
    tool_executor: Any = None
    is_byok: bool = False
    resolved_provider: str = ""
    workspace_id: Optional[Any] = None

    def update_metrics(self, execution_time: float, tokens_used: int, success: bool):
        self.execution_count += 1
        self.total_tokens_used += tokens_used
        self.last_execution = datetime.now()
        if "avg_execution_time" not in self.performance_metrics:
            self.performance_metrics["avg_execution_time"] = execution_time
        else:
            avg = self.performance_metrics["avg_execution_time"]
            self.performance_metrics["avg_execution_time"] = (
                (avg * (self.execution_count - 1) + execution_time) / self.execution_count
            )
        if "success_count" not in self.performance_metrics:
            self.performance_metrics["success_count"] = 0
        if success:
            self.performance_metrics["success_count"] += 1
        self.performance_metrics["success_rate"] = (
            self.performance_metrics["success_count"] / self.execution_count
        )


# ---------------------------------------------------------------------------
# AgentFactory
# ---------------------------------------------------------------------------

class AgentFactory:
    """
    Creates and manages user-defined agents.
    Pure execution layer — the orchestrator handles all prompt engineering.
    """

    def __init__(self, db_session: Session = None):
        if db_session:
            self.db_session = db_session
        else:
            from core.database.database import SessionLocal
            self.db_session = SessionLocal()
        self.active_agents: Dict[int, AgentRuntime] = {}
        self.logger = logging.getLogger(__name__)

    # ==================================================================
    # LLM Config Resolution
    # ==================================================================

    def _get_default_llm_config_from_settings(self) -> Dict[str, Any]:
        """Get default LLM config from system settings, falling back to config.py."""
        try:
            from core.llm.manager import get_system_setting

            provider = get_system_setting("orchestrator_llm", "llm_provider")
            if not provider:
                provider = get_system_setting("orchestrator_llm", "provider")
            model = get_system_setting("orchestrator_llm", "llm_model")
            if not model:
                model = get_system_setting("orchestrator_llm", "model")
            if not provider:
                provider = config.LLM_PROVIDER
            if not model:
                model = config.LLM_MODEL
            if not provider or not model:
                self.logger.warning("LLM provider/model not in system settings, using config defaults")
                return {
                    "provider": config.LLM_PROVIDER,
                    "model": config.LLM_MODEL,
                    "temperature": 0.7,
                    "max_tokens": 2000,
                    "context_window": 8192,
                }

            # Lookup context window from LLM models registry
            context_window = 8192
            max_tokens = 2000
            try:
                from core.models import LLMModel
                llm_model = self.db_session.query(LLMModel).filter_by(model_id=model).first()
                if llm_model:
                    context_window = llm_model.context_window
                    max_tokens = llm_model.max_output_tokens
            except Exception as e:
                self.logger.warning(f"Could not get context window from registry: {e}")

            self.logger.info(f"Using model from settings: {model} (context: {context_window})")
            return {
                "provider": provider,
                "model": model,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "context_window": context_window,
            }
        except Exception as e:
            self.logger.warning(f"Could not get LLM config from settings: {e}, using config defaults")
            return {
                "provider": config.LLM_PROVIDER,
                "model": config.LLM_MODEL,
                "temperature": 0.7,
                "max_tokens": 2000,
                "context_window": 8192,
            }

    def _resolve_provider_for_model(self, provider_str: str, model_id: str) -> str:
        """Auto-detect and correct provider-model mismatches."""
        DIRECT_PROVIDERS = {
            "openai", "anthropic", "google", "grok", "azure", "azure_openai",
            "aws_bedrock", "bedrock", "huggingface", "openrouter",
        }
        model_lower = model_id.lower()

        # Unknown provider + slash format = OpenRouter marketplace model
        if provider_str not in DIRECT_PROVIDERS and "/" in model_id:
            self.logger.info(f"Provider '{provider_str}' not recognized, routing '{model_id}' through OpenRouter")
            return "openrouter"

        # Slash-format model IDs are OpenRouter marketplace models
        if "/" in model_id and provider_str != "openrouter":
            prefix = model_id.split("/")[0].lower()
            if prefix == provider_str.lower() or prefix not in DIRECT_PROVIDERS:
                self.logger.info(
                    f"Slash-format model '{model_id}' with provider='{provider_str}' "
                    f"detected as OpenRouter marketplace model. Routing through OpenRouter."
                )
                return "openrouter"

        # Fix provider-model mismatches
        inferred = None
        if model_lower.startswith("gemini"):
            inferred = "google"
        elif model_lower.startswith("claude"):
            inferred = "anthropic"
        elif model_lower.startswith(("gpt-", "o1", "o3", "o4")):
            inferred = "openai"
        elif model_lower.startswith("grok"):
            inferred = "grok"

        if inferred and inferred != provider_str and provider_str in DIRECT_PROVIDERS:
            self.logger.warning(
                f"Provider-model mismatch: provider='{provider_str}' but model='{model_id}' "
                f"suggests '{inferred}'. Auto-correcting."
            )
            return inferred

        return provider_str

    async def _resolve_api_key(self, provider_name: str, agent_name: str = "", workspace_id=None) -> Optional[ResolvedKey]:
        """
        Resolve API key: BYOK → credential store → env vars.
        """
        from core.credentials.resolver import get_credential_resolver

        resolver = get_credential_resolver()

        # 1. Check BYOK
        if workspace_id:
            try:
                from core.models.workspaces import Workspace
                from core.models.core import UserApiKey
                from core.credentials.encryption import get_encryption_service

                workspace = self.db_session.query(Workspace).get(workspace_id)
                byok_overrides = (workspace.settings or {}).get("byok_overrides", {}) if workspace else {}

                if byok_overrides.get(provider_name, False):
                    byok_key = (
                        self.db_session.query(UserApiKey)
                        .filter(
                            UserApiKey.workspace_id == workspace_id,
                            UserApiKey.provider == provider_name,
                            UserApiKey.is_active == True,
                        )
                        .order_by(UserApiKey.last_used_at.desc().nullslast())
                        .first()
                    )
                    if byok_key:
                        encryption = get_encryption_service()
                        decrypted = encryption.decrypt(byok_key.encrypted_key)
                        self.logger.info(f"Resolved BYOK API key for '{provider_name}' workspace={workspace_id}")
                        return ResolvedKey(api_key=decrypted, source="byok", is_byok=True, provider=provider_name)
                    else:
                        self.logger.info(f"BYOK enabled but no active key for '{provider_name}', falling through")
            except Exception as e:
                self.logger.error(f"BYOK key lookup failed for {provider_name}: {e}")

        # 2. Credential store (platform keys)
        cred_names = [
            f"development_{provider_name}_api",
            f"development_{provider_name}",
            f"{provider_name}_api",
            provider_name,
        ]
        for cred_name in cred_names:
            try:
                key = resolver.get_credential_field(cred_name, "api_key")
                if not key:
                    key = resolver.get_credential_field(cred_name, "api_token")
                if key:
                    self.logger.info(f"Resolved platform API key from credential '{cred_name}' for {agent_name}")
                    return ResolvedKey(api_key=key, source="platform", is_byok=False, provider=provider_name)
            except Exception:
                continue

        # 3. Config env vars
        from config import config as _cfg
        config_map = {
            "openai": _cfg.OPENAI_API_KEY,
            "anthropic": _cfg.ANTHROPIC_API_KEY,
            "google": _cfg.GOOGLE_API_KEY,
            "openrouter": _cfg.OPENROUTER_API_KEY,
            "grok": _cfg.XAI_API_KEY,
            "azure": _cfg.AZURE_OPENAI_API_KEY,
            "azure_openai": _cfg.AZURE_OPENAI_API_KEY,
            "aws_bedrock": _cfg.AWS_ACCESS_KEY_ID,
            "bedrock": _cfg.AWS_ACCESS_KEY_ID,
        }
        key = config_map.get(provider_name)
        if key:
            self.logger.info(f"Using config API key for {provider_name} for {agent_name}")
            return ResolvedKey(api_key=key, source="env", is_byok=False, provider=provider_name)

        return None

    async def _create_llm_manager(self, model_config: ModelConfiguration, agent_name: str = "", workspace_id=None) -> Tuple[LLMManager, ResolvedKey]:
        """Create LLM manager with API key resolution (PRD-15, PRD-54)."""
        from core.llm import LLMConfig, LLMProvider as LLMProviderEnum

        provider_map = {
            "openai": LLMProviderEnum.OPENAI,
            "anthropic": LLMProviderEnum.ANTHROPIC,
            "google": LLMProviderEnum.GOOGLE,
            "openrouter": LLMProviderEnum.OPENROUTER,
            "grok": LLMProviderEnum.GROK,
            "huggingface": LLMProviderEnum.HUGGINGFACE,
            "azure": LLMProviderEnum.AZURE,
            "azure_openai": LLMProviderEnum.AZURE,
            "aws_bedrock": LLMProviderEnum.AWS_BEDROCK,
            "bedrock": LLMProviderEnum.AWS_BEDROCK,
        }

        effective_provider = self._resolve_provider_for_model(model_config.provider, model_config.model_id)
        if effective_provider not in provider_map:
            raise ValueError(f"Unsupported provider: {effective_provider}")

        provider = provider_map[effective_provider]
        resolved = await self._resolve_api_key(effective_provider, agent_name, workspace_id=workspace_id)
        if not resolved:
            raise ValueError(f"API key not found for provider: {effective_provider}")

        llm_config = LLMConfig(
            provider=provider,
            model=model_config.model_id,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            api_key=resolved.api_key,
        )

        # Bedrock uses IAM auth
        if effective_provider in ("aws_bedrock", "bedrock"):
            llm_config.secret_key = config.AWS_SECRET_ACCESS_KEY
            llm_config.base_url = config.AWS_REGION or "us-east-1"

        self.logger.info(
            f"Creating LLM manager for {agent_name or 'agent'}: "
            f"provider={effective_provider}, model={model_config.model_id}, source={resolved.source}"
        )

        manager = LLMManager(
            config=llm_config,
            workspace_id=workspace_id,
            agent_id=None,
            is_byok=resolved.is_byok,
        )
        return manager, resolved

    async def _verify_llm_connection(self, llm_manager: LLMManager) -> Dict[str, Any]:
        """Verify LLM connection with minimal test."""
        try:
            start_time = time.time()
            messages = [{"role": "user", "content": "Respond with 'OK' to confirm connection."}]
            response = await llm_manager.generate_response(messages)
            response_time = time.time() - start_time
            if response and response.content:
                return {
                    "success": True,
                    "response_time": response_time,
                    "tokens_used": response.usage.get("total_tokens", 0) if response.usage else 0,
                }
            return {"success": False, "error": "No response from LLM"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    # ==================================================================
    # Agent Lifecycle
    # ==================================================================

    async def create_agent(
        self,
        metadata: Union[AgentMetadata, Dict[str, Any]],
        auto_verify: bool = True,
    ) -> AgentRuntime:
        """Create an agent from user-defined metadata."""
        start_time = time.time()

        if isinstance(metadata, dict):
            model_config = None
            if "model_config" in metadata:
                model_config = ModelConfiguration.from_dict(metadata["model_config"])
            metadata = AgentMetadata(
                name=metadata.get("name", "Unnamed Agent"),
                agent_type=metadata.get("type", "generic"),
                description=metadata.get("description"),
                skills=metadata.get("skills", []),
                model_config=model_config,
                preferred_model=metadata.get("preferred_model"),
                temperature=metadata.get("temperature"),
                max_tokens=metadata.get("max_tokens"),
                context_window=metadata.get("context_window"),
                custom_metadata=metadata.get("metadata", {}),
            )

        model_config = metadata.get_model_config()

        db_agent = Agent(
            name=metadata.name,
            description=metadata.description or f"User-defined {metadata.agent_type} agent",
            agent_type=metadata.agent_type,
            status=AgentLifecycle.INITIALIZING.value,
            configuration={
                "skills": metadata.skills,
                "llm_config": metadata.get_llm_config(),
                "custom_metadata": metadata.custom_metadata,
            },
            model_config=model_config.to_dict(),
            priority_level=PriorityLevel.MEDIUM.value,
            max_concurrent_tasks=5,
            auto_start=False,
            created_by="agent_factory",
        )

        self.db_session.add(db_agent)
        self.db_session.commit()

        try:
            llm_manager, resolved = await self._create_llm_manager(model_config, db_agent.name, workspace_id=db_agent.workspace_id)

            if auto_verify:
                verification_result = await self._verify_llm_connection(llm_manager)
                if not verification_result["success"] and model_config.fallback_model_id:
                    self.logger.warning(
                        f"Primary model '{model_config.model_id}' failed, trying fallback '{model_config.fallback_model_id}'"
                    )
                    fallback_config = ModelConfiguration(
                        provider=model_config.provider,
                        model_id=model_config.fallback_model_id,
                        temperature=model_config.temperature,
                        max_tokens=model_config.max_tokens,
                    )
                    llm_manager, resolved = await self._create_llm_manager(fallback_config, db_agent.name, workspace_id=db_agent.workspace_id)
                    verification_result = await self._verify_llm_connection(llm_manager)
                    if verification_result["success"]:
                        db_agent.model_config = fallback_config.to_dict()
                        self.db_session.commit()

                if not verification_result["success"]:
                    self.db_session.delete(db_agent)
                    self.db_session.commit()
                    raise Exception(f"LLM verification failed: {verification_result['error']}")

            agent_tools = await self._load_agent_tools(db_agent.id)

            agent_runtime = AgentRuntime(
                agent_id=db_agent.id,
                metadata=metadata,
                llm_manager=llm_manager,
                lifecycle_state=AgentLifecycle.ACTIVE,
                created_at=datetime.now(),
                tools=agent_tools,
                is_byok=resolved.is_byok,
                resolved_provider=resolved.provider,
                workspace_id=db_agent.workspace_id,
            )

            db_agent.status = AgentLifecycle.ACTIVE.value
            self.db_session.commit()
            self.active_agents[db_agent.id] = agent_runtime

            self.logger.info(
                f"Agent '{metadata.name}' (type: {metadata.agent_type}) created in {time.time() - start_time:.2f}s"
            )
            return agent_runtime

        except Exception as e:
            self.logger.error(f"Failed to create agent: {e}")
            if db_agent.id:
                self.db_session.delete(db_agent)
                self.db_session.commit()
            raise

    async def activate_agent(
        self,
        agent_id: int,
        workspace_dir: str = "/tmp/automatos_workspace",
        use_system_llm: bool = False,
    ) -> Optional[AgentRuntime]:
        """Load an agent from database and activate it in runtime."""
        try:
            if agent_id in self.active_agents:
                self.logger.info(f"Agent {agent_id} already active in runtime")
                return self.active_agents[agent_id]

            from modules.agents.queries import get_agent_with_context

            db_agent = get_agent_with_context(self.db_session, agent_id)
            if not db_agent:
                self.logger.error(f"Agent {agent_id} not found in database")
                return None

            # Resolve LLM config: agent's own model_config → system settings
            agent_model_config = db_agent.model_config or {}
            agent_config = db_agent.configuration or {}
            agent_llm_config = agent_config.get("llm_config") or {}

            agent_has_model = agent_model_config.get("model_id") and agent_model_config.get("provider")

            if agent_has_model and not use_system_llm:
                llm_config_dict = {
                    "provider": agent_model_config["provider"],
                    "model": agent_model_config["model_id"],
                    "temperature": agent_model_config.get("temperature", 0.7),
                    "max_tokens": agent_model_config.get("max_tokens", 2000),
                }
                self.logger.info(f"Agent {agent_id} using LLM: {llm_config_dict['provider']}/{llm_config_dict['model']} (agent model_config)")
            else:
                reason = "use_system_llm=True" if use_system_llm else "no agent model_config"
                system_llm_config = self._get_default_llm_config_from_settings()
                llm_config_dict = {
                    "provider": system_llm_config.get("provider"),
                    "model": system_llm_config.get("model"),
                    "temperature": agent_llm_config.get("temperature", system_llm_config.get("temperature", 0.7)),
                    "max_tokens": agent_llm_config.get("max_tokens", system_llm_config.get("max_tokens", 2000)),
                }
                self.logger.info(f"Agent {agent_id} using LLM: {llm_config_dict.get('provider')}/{llm_config_dict.get('model')} ({reason})")

            provider_str = llm_config_dict.get("provider", "openai")
            model_id_str = llm_config_dict.get("model", config.LLM_MODEL)
            provider_str = self._resolve_provider_for_model(provider_str, model_id_str)

            provider = LLMProvider(provider_str)
            resolved = await self._resolve_api_key(provider_str, db_agent.name, workspace_id=db_agent.workspace_id)

            # Fallback: if direct provider has no credential, try OpenRouter
            if (not resolved or not resolved.api_key) and provider_str != "openrouter":
                self.logger.warning(
                    f"No credential for provider '{provider_str}' — falling back to OpenRouter for '{model_id_str}'"
                )
                provider_str = "openrouter"
                if "/" not in model_id_str:
                    original_provider = llm_config_dict.get("provider", "").lower()
                    model_id_str = f"{original_provider}/{model_id_str}"
                provider = LLMProvider(provider_str)
                resolved = await self._resolve_api_key(provider_str, db_agent.name, workspace_id=db_agent.workspace_id)

            llm_config = LLMConfig(
                provider=provider,
                model=model_id_str,
                temperature=llm_config_dict.get("temperature", 0.7),
                max_tokens=llm_config_dict.get("max_tokens", 2000),
                api_key=resolved.api_key if resolved else None,
            )
            llm_manager = LLMManager(
                config=llm_config,
                workspace_id=db_agent.workspace_id,
                agent_id=agent_id,
                is_byok=resolved.is_byok if resolved else False,
            )

            metadata = AgentMetadata(
                name=db_agent.name,
                agent_type=db_agent.agent_type,
                description=db_agent.description,
                skills=agent_config.get("skills", []),
                custom_metadata=agent_config.get("custom_metadata", {}),
            )

            agent_tools = await self._load_agent_tools(agent_id)

            agent_runtime = AgentRuntime(
                agent_id=agent_id,
                metadata=metadata,
                llm_manager=llm_manager,
                lifecycle_state=AgentLifecycle.ACTIVE,
                created_at=datetime.now(),
                tools=agent_tools,
                tool_executor=get_unified_tool_executor(self.db_session, workspace_dir or "/tmp/automatos_workspace"),
                is_byok=resolved.is_byok if resolved else False,
                resolved_provider=resolved.provider if resolved else provider_str,
                workspace_id=db_agent.workspace_id,
            )

            self.active_agents[agent_id] = agent_runtime
            db_agent.status = AgentLifecycle.ACTIVE.value
            self.db_session.commit()

            self.logger.info(f"Activated agent {agent_id} ({db_agent.name}) with {llm_config_dict.get('model')}")
            return agent_runtime

        except Exception as e:
            self.logger.error(f"Failed to activate agent {agent_id}: {e}")
            return None

    # ==================================================================
    # execute_with_prompt — CLEAN REWRITE
    # ==================================================================
    # ONE tool source: get_tools_for_agent() from tool_router.py
    # Tool loop with max_tool_iterations (default 10)
    # No hardcoded schemas, no legacy JSON actions, no mid-execution discovery

    async def execute_with_prompt(
        self,
        agent: Union[int, AgentRuntime],
        prompt: str,
        system_prompt: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        use_memory: bool = True,
        max_retries: int = 2,
        max_tool_iterations: int = 10,
        composio_action_names: Optional[set] = None,
        context_mode: Optional[str] = None,  # ContextMode enum value — overrides default TASK_EXECUTION
        attachment_ids: Optional[List[str]] = None,  # PRD-127: ephemeral attachments
        # Legacy params — accepted but ignored (callers may still pass them)
        enable_actions: bool = True,
        action_executor: Optional[Any] = None,
        required_tools: Optional[List[str]] = None,
        workspace_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a task with orchestrator-provided prompt.

        Tools are determined by agent assignments via get_tools_for_agent() —
        the single source of truth. No required_tools parameter needed.
        """
        start_time = time.time()

        # --- Resolve agent runtime ---
        if isinstance(agent, int):
            agent_runtime = self.active_agents.get(agent)
            if not agent_runtime:
                self.logger.info(f"Agent {agent} not in runtime, activating...")
                agent_runtime = await self.activate_agent(agent, workspace_dir=workspace_dir or "/tmp/automatos_workspace")
                if not agent_runtime:
                    return {"status": "error", "error": f"Agent {agent} could not be activated"}
        else:
            agent_runtime = agent

        agent_name = agent_runtime.metadata.name
        agent_id = agent_runtime.agent_id

        self.logger.info("=" * 80)
        self.logger.info(f"EXECUTING AGENT: {agent_name} (ID: {agent_id}, Type: {agent_runtime.metadata.agent_type})")
        self.logger.info("=" * 80)

        agent_runtime.lifecycle_state = AgentLifecycle.BUSY

        try:
            # --- Build messages ---
            messages = []

            # System prompt: explicit OR ContextService
            context_result = None  # Populated when ContextService builds the prompt

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                # --- ContextService path (the single prompt builder) ---
                from modules.agents.queries import get_agent_with_context
                from modules.context import ContextService, ContextMode

                mode = ContextMode(context_mode) if context_mode else ContextMode.TASK_EXECUTION
                db_agent = get_agent_with_context(self.db_session, agent_runtime.agent_id)
                if db_agent:
                    context_result = await ContextService(self.db_session).build_context(
                        mode=mode,
                        agent=db_agent,
                        workspace_id=agent_runtime.workspace_id,
                        task_description=prompt,
                        attachment_ids=attachment_ids,  # PRD-127
                    )
                    messages.append({"role": "system", "content": context_result.system_prompt})
                else:
                    # Last resort — should not happen
                    messages.append({"role": "system", "content": f"You are agent {agent_runtime.agent_id}."})

            # Short-term memory
            if use_memory and agent_runtime.memory:
                for mem in agent_runtime.memory[-3:]:
                    if "user_prompt" in mem:
                        messages.append({"role": "user", "content": mem["user_prompt"]})
                    if "assistant_response" in mem:
                        messages.append({"role": "assistant", "content": mem["assistant_response"]})

            # Recipe step context injection
            _has_step_outputs = context and context.get("step_outputs")
            _has_step_results = context and context.get("step_results")
            if _has_step_outputs or _has_step_results:
                recipe_step = context.get("step", "?")
                total_steps = context.get("total_steps", "?")
                if _has_step_outputs:
                    completed_count = len(context["step_outputs"])
                else:
                    completed_count = len([sr for sr in context["step_results"] if sr.get("status") == "completed"])

                if completed_count > 0:
                    recipe_ctx = (
                        f"You are executing step {recipe_step} of {total_steps} in a recipe.\n"
                        "Previous steps have already completed. Their outputs are provided "
                        "in a separate system context message. When the user's task mentions "
                        "'results', 'output', 'data', or 'findings', it refers to that "
                        "previous step content. Use it directly — do not invent or fabricate data."
                    )
                    messages.append({"role": "system", "content": recipe_ctx})

            # Preserve original prompt for Composio hint generation
            original_user_prompt = prompt

            # --- Build tools ---
            if context_result is not None:
                # ContextService already loaded tools via ToolsSection
                tool_schemas = list(context_result.tools)
            else:
                # Explicit system_prompt path: load tools directly
                from modules.tools.tool_router import get_tools_for_agent

                tool_schemas = get_tools_for_agent(
                    agent_id=agent_runtime.agent_id,
                    db_session=self.db_session,
                    workspace_id=agent_runtime.workspace_id,
                )

            # Composio hint injection (enriches composio_execute with action enum + hints)
            workspace_id = agent_runtime.workspace_id
            composio_apps = [t for t in (agent_runtime.tools or []) if t.get("provider") == "Composio"]
            if composio_apps:
                if composio_action_names:
                    # Recipe path: pre-resolved action names
                    self._inject_composio_recipe_hints(
                        tool_schemas, messages, composio_apps, composio_action_names,
                    )
                else:
                    # Default path: hint service
                    self._inject_composio_hints(
                        tool_schemas, messages, agent_runtime, original_user_prompt, workspace_id,
                    )

            all_tool_names = [t["function"]["name"] for t in tool_schemas]
            self.logger.info(f"Providing {len(tool_schemas)} tools to agent: {all_tool_names}")

            # Add user prompt
            messages.append({"role": "user", "content": prompt})

            # --- Execute with retries ---
            last_error = None
            messages_snapshot = list(messages)  # snapshot before retry loop
            for attempt in range(max(1, max_retries)):
                try:
                    messages = list(messages_snapshot)  # reset each attempt
                    response = await agent_runtime.llm_manager.generate_response(messages, tools=tool_schemas)
                    execution_time = time.time() - start_time

                    # --- Tool loop: iterate until no more tool calls or max iterations ---
                    tool_iteration = 0
                    tool_results = []
                    while response and response.tool_calls and tool_iteration < max_tool_iterations:
                        tool_iteration += 1
                        self.logger.info(
                            f"Tool iteration {tool_iteration}/{max_tool_iterations}: "
                            f"{len(response.tool_calls)} tool call(s)"
                        )

                        tool_results = await self._execute_tool_calls(
                            response.tool_calls,
                            agent_runtime,
                            workspace_id,
                        )

                        # Append assistant message + tool results to conversation
                        messages.append({
                            "role": "assistant",
                            "content": response.content or "",
                            "tool_calls": response.tool_calls,
                        })
                        messages.extend(tool_results)

                        # Call LLM again with tool results
                        response = await agent_runtime.llm_manager.generate_response(messages, tools=tool_schemas)
                        execution_time = time.time() - start_time

                    if tool_iteration >= max_tool_iterations:
                        self.logger.warning(f"Hit max tool iterations ({max_tool_iterations}) for agent {agent_id}")

                    # Synthesize empty response from tool results
                    if response and (not response.content or not response.content.strip()):
                        if tool_results:
                            tool_summary = "\n\n".join([
                                f"**{tr['name']}**: {tr['content'][:500]}"
                                for tr in tool_results
                                if tr.get("role") == "tool"
                            ])
                            response.content = f"Based on the tool results:\n\n{tool_summary}"

                    # Handle truncation: continue generating if output was cut off
                    max_continuations = 2
                    continuation = 0
                    while (
                        response
                        and getattr(response, "finish_reason", None) == "length"
                        and response.content
                        and continuation < max_continuations
                    ):
                        continuation += 1
                        completion_tokens = (response.usage or {}).get("completion_tokens", 0)
                        self.logger.info(
                            "Output truncated for agent %s (continuation %d/%d, %d tokens so far). "
                            "Requesting continuation...",
                            agent_id, continuation, max_continuations, completion_tokens,
                        )
                        # Append partial output and ask to continue
                        messages.append({"role": "assistant", "content": response.content})
                        messages.append({
                            "role": "user",
                            "content": "Your response was truncated. Continue exactly where you left off — do not repeat any content.",
                        })
                        continuation_response = await agent_runtime.llm_manager.generate_response(messages, tools=tool_schemas)
                        if continuation_response and continuation_response.content:
                            response.content += continuation_response.content
                            if continuation_response.usage:
                                prev_usage = response.usage or {}
                                response.usage = {
                                    "total_tokens": prev_usage.get("total_tokens", 0) + continuation_response.usage.get("total_tokens", 0),
                                    "completion_tokens": prev_usage.get("completion_tokens", 0) + continuation_response.usage.get("completion_tokens", 0),
                                    "prompt_tokens": continuation_response.usage.get("prompt_tokens", 0),
                                }
                            response.finish_reason = getattr(continuation_response, "finish_reason", None)
                            execution_time = time.time() - start_time
                        else:
                            break

                    if continuation > 0:
                        self.logger.info(
                            "Completed %d continuation(s) for agent %s. Total output: %d chars",
                            continuation, agent_id, len(response.content) if response and response.content else 0,
                        )

                    if response and response.content:
                        tokens_used = response.usage.get("total_tokens", 0) if response.usage else 0
                        agent_runtime.update_metrics(execution_time, tokens_used, True)

                        # Store in short-term memory
                        agent_runtime.memory.append({
                            "task": prompt[:200],
                            "response": response.content[:500],
                            "summary": f"Executed: {prompt[:100]}",
                            "timestamp": datetime.now().isoformat(),
                            "tokens": tokens_used,
                            "execution_time": execution_time,
                        })
                        if len(agent_runtime.memory) > 20:
                            agent_runtime.memory = agent_runtime.memory[-20:]

                        agent_runtime.lifecycle_state = AgentLifecycle.ACTIVE

                        # Record monitoring
                        monitoring = get_monitoring_service()
                        monitoring.record_agent_execution(
                            agent_id=agent_runtime.agent_id,
                            agent_name=agent_runtime.metadata.name,
                            task=prompt[:100],
                            execution_time_ms=execution_time * 1000,
                            tokens_used=tokens_used,
                            success=True,
                        )

                        return {
                            "status": "success",
                            "result": response.content,
                            "agent": {
                                "id": agent_runtime.agent_id,
                                "name": agent_runtime.metadata.name,
                                "type": agent_runtime.metadata.agent_type,
                            },
                            "execution": {
                                "time": execution_time,
                                "tokens_used": tokens_used,
                                "model": response.model,
                                "provider": response.provider,
                                "attempt": attempt + 1,
                                "tool_iterations": tool_iteration,
                            },
                            "metrics": {
                                "total_executions": agent_runtime.execution_count,
                                "success_rate": agent_runtime.performance_metrics.get("success_rate", 1.0),
                                "avg_execution_time": agent_runtime.performance_metrics.get("avg_execution_time", execution_time),
                            },
                        }
                    else:
                        last_error = "Empty response from LLM"

                except Exception as e:
                    last_error = str(e)
                    self.logger.warning(f"Attempt {attempt + 1} failed: {last_error}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)

            # All retries failed
            agent_runtime.update_metrics(time.time() - start_time, 0, False)
            agent_runtime.lifecycle_state = AgentLifecycle.ACTIVE
            return {
                "status": "error",
                "error": f"Task execution failed after {max_retries} attempts: {last_error}",
                "agent": {
                    "id": agent_runtime.agent_id,
                    "name": agent_runtime.metadata.name,
                    "type": agent_runtime.metadata.agent_type,
                },
            }

        except Exception as e:
            agent_runtime.lifecycle_state = AgentLifecycle.ACTIVE
            self.logger.error(f"Task execution error: {e}")
            return {"status": "error", "error": str(e)}

    # ==================================================================
    # Tool Execution (extracted from execute_with_prompt)
    # ==================================================================

    async def _execute_tool_calls(
        self,
        tool_calls: List[Dict],
        agent_runtime: AgentRuntime,
        workspace_id: Optional[Any],
    ) -> List[Dict]:
        """Execute tool calls from LLM response, deduplicating within a turn."""
        tool_results = []
        executed_hashes: set = set()
        tool_executor = agent_runtime.tool_executor

        for tool_call in tool_calls:
            func_name = tool_call["function"]["name"]
            func_args_str = tool_call["function"]["arguments"]

            try:
                func_args = json.loads(func_args_str)
                canonical_args = json.dumps(func_args, sort_keys=True)
            except json.JSONDecodeError:
                canonical_args = func_args_str.strip()
                func_args = {}

            call_hash = f"{func_name}:{canonical_args}"

            if call_hash in executed_hashes:
                self.logger.warning(f"[DEDUPE] Skipping duplicate tool call: {func_name}")
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps({"error": "Duplicate tool call skipped"}),
                })
                continue

            # Filter empty params for critical tools
            if not func_args and "SLACK" in func_name:
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps({"error": "Empty parameters for tool requiring input"}),
                })
                continue

            executed_hashes.add(call_hash)
            self.logger.info(f"  [TRACE] Calling {func_name}({func_args})")

            try:
                result = await tool_executor.execute_tool(
                    tool_name=func_name,
                    parameters=func_args,
                    agent_id=agent_runtime.agent_id,
                    workspace_id=workspace_id,
                )
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps(result),
                })
                self.logger.info(f"    [TRACE] {func_name} completed")
            except Exception as e:
                self.logger.error(f"    [TRACE] {func_name} failed: {e}")
                tool_results.append({
                    "tool_call_id": tool_call["id"],
                    "role": "tool",
                    "name": func_name,
                    "content": json.dumps({"error": str(e)}),
                })

        return tool_results

    # ==================================================================
    # Composio Hint Injection
    # ------------------------------------------------------------------
    # These methods are COMPLEMENTARY to ComposioSection (modules/context/
    # sections/composio.py), NOT redundant. ComposioSection renders static
    # app-level descriptions in the system prompt. These methods do dynamic
    # per-request work: _inject_composio_hints() uses ComposioHintService
    # to semantically match user prompts to relevant actions AND constrains
    # the composio_execute tool schema enum. _inject_composio_recipe_hints()
    # does the same for recipe paths with pre-resolved action names.
    # Future: absorb hint logic into a ComposioHintSection or expand
    # ComposioSection with action-level detail. See PRD-81 Task 5.7.
    # ==================================================================

    def _inject_composio_recipe_hints(
        self,
        tool_schemas: List[Dict],
        messages: List[Dict],
        composio_apps: List[Dict],
        composio_action_names: set,
    ):
        """Inject pre-resolved Composio actions (recipe path)."""
        sorted_actions = sorted(composio_action_names)

        # Find composio_execute schema and constrain its action enum
        for schema in tool_schemas:
            if schema.get("function", {}).get("name") == "composio_execute":
                schema["function"]["parameters"]["properties"]["action"] = {
                    "type": "string",
                    "description": "Action name — must be one of these actions.",
                    "enum": sorted_actions,
                }
                break

        app_names = [t.get("name") for t in composio_apps]
        hint_lines = [
            f"You have Composio apps connected: {', '.join(app_names)}.",
            f"Available actions (use exactly these names): {', '.join(sorted_actions)}.",
            "Call composio_execute with the action name and required params.",
        ]
        insert_at = 1 if messages and messages[0].get("role") == "system" else 0
        messages.insert(insert_at, {"role": "system", "content": "\n".join(hint_lines)})

        self.logger.info(
            f"Composio (semantic): constrained to {len(sorted_actions)} actions: {sorted_actions}"
        )

    def _inject_composio_hints(
        self,
        tool_schemas: List[Dict],
        messages: List[Dict],
        agent_runtime: AgentRuntime,
        original_user_prompt: str,
        workspace_id: Optional[Any],
    ):
        """Inject Composio per-action tools (primary) or hint fallback.

        Primary: ComposioToolService returns per-action OpenAI function schemas
        (e.g. TAVILY_SEARCH with typed params). Strips composio_execute and
        replaces with per-action tools.

        Fallback: ComposioHintService injects action names as system prompt
        hints and constrains composio_execute's action enum.
        """
        try:
            from modules.tools.services.composio_tool_service import ComposioToolService

            composio_svc = ComposioToolService(self.db_session)
            composio_result = composio_svc.get_tools_for_step(
                agent_id=agent_runtime.agent_id,
                workspace_id=workspace_id,
                task_prompt=original_user_prompt,
            )

            if composio_result and composio_result.tools:
                # Strip composio_execute, add per-action schemas
                tool_schemas[:] = [
                    t for t in tool_schemas
                    if t.get("function", {}).get("name") != "composio_execute"
                ] + composio_result.tools

                # Add scope message so the LLM knows which apps are available
                from api.recipe_executor import _composio_scope_message
                insert_at = 1 if messages and messages[0].get("role") == "system" else 0
                messages.insert(insert_at, {
                    "role": "system",
                    "content": _composio_scope_message(composio_result.app_names),
                })

                self.logger.info(
                    f"Composio (per-action): strategy={composio_result.strategy} "
                    f"actions={len(composio_result.action_set)} "
                    f"search_ms={composio_result.search_ms}"
                )
                return

        except Exception as e:
            self.logger.warning(f"ComposioToolService failed, falling back to hints: {e}")

        # Fallback: hint-based injection
        try:
            from modules.tools.services.composio_hint_service import ComposioHintService

            hint_service = ComposioHintService(self.db_session)
            hint_result = hint_service.build_hints(
                agent_id=agent_runtime.agent_id,
                prompt=original_user_prompt,
                workspace_id=workspace_id,
            )

            if hint_result.hint_lines:
                insert_at = 1 if messages and messages[0].get("role") == "system" else 0
                messages.insert(insert_at, {"role": "system", "content": "\n".join(hint_result.hint_lines)})

            if hint_result.matched_actions:
                for schema in tool_schemas:
                    if schema.get("function", {}).get("name") == "composio_execute":
                        schema["function"]["parameters"]["properties"]["action"] = {
                            "type": "string",
                            "description": "Action name — must be one of the candidate actions listed above.",
                            "enum": hint_result.matched_actions,
                        }
                        break

            self.logger.info(
                f"Composio (hints fallback): strategy={hint_result.strategy_used}, "
                f"constrained_actions={len(hint_result.matched_actions)}, "
                f"apps={hint_result.allowed_apps}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to inject Composio hints: {e}")

    # ==================================================================
    # Composio App Loading
    # ==================================================================

    async def _load_agent_tools(self, agent_id: int) -> List[Dict[str, Any]]:
        """Load Composio apps assigned to an agent from the database."""
        try:
            assignments = (
                self.db_session.query(AgentAppAssignment)
                .filter(
                    AgentAppAssignment.agent_id == agent_id,
                    AgentAppAssignment.is_active.is_(True),
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
                tools.append({
                    "name": app_name,
                    "description": (app.description if app else "") or "",
                    "provider": "Composio",
                    "category": ((app.categories or [None])[0] if app else None),
                    "icon": app.logo_url if app else None,
                    "assigned_at": assignment.assigned_at.isoformat() if assignment.assigned_at else None,
                })

            self.logger.info(f"Loaded {len(tools)} Composio app(s) for agent {agent_id}")
            return tools
        except Exception as e:
            self.logger.warning(f"Failed to load Composio apps for agent {agent_id}: {e}")
            return []

    # ==================================================================
    # Status & Utility
    # ==================================================================

    async def get_agent_status(self, agent_id: int) -> Dict[str, Any]:
        """Get detailed status of an agent."""
        agent_runtime = self.active_agents.get(agent_id)

        if not agent_runtime:
            db_agent = self.db_session.query(Agent).filter_by(id=agent_id).first()
            if db_agent:
                return {
                    "status": "inactive",
                    "agent": {
                        "id": db_agent.id,
                        "name": db_agent.name,
                        "type": db_agent.agent_type,
                        "database_status": db_agent.status,
                    },
                    "runtime": None,
                    "message": "Agent exists in database but not in runtime.",
                }
            return {"status": "not_found", "error": f"Agent {agent_id} does not exist"}

        provider_info = agent_runtime.llm_manager.get_provider_info()
        return {
            "status": "active",
            "agent": {
                "id": agent_runtime.agent_id,
                "name": agent_runtime.metadata.name,
                "type": agent_runtime.metadata.agent_type,
                "lifecycle_state": agent_runtime.lifecycle_state.value,
                "skills": agent_runtime.metadata.skills,
            },
            "runtime": {
                "created_at": agent_runtime.created_at.isoformat(),
                "last_execution": agent_runtime.last_execution.isoformat() if agent_runtime.last_execution else None,
                "execution_count": agent_runtime.execution_count,
                "total_tokens_used": agent_runtime.total_tokens_used,
                "memory_size": len(agent_runtime.memory),
            },
            "llm": provider_info,
            "metrics": agent_runtime.performance_metrics,
        }

    def get_agent_tool_capability(self, agent_runtime: AgentRuntime, capability: str) -> bool:
        """Check if an agent has a specific tool capability."""
        for tool in agent_runtime.tools:
            tool_capabilities = tool.get("capabilities", {})
            if isinstance(tool_capabilities, dict):
                if capability in tool_capabilities.get("methods", []):
                    return True
        return False

    def get_agent_tools_summary(self, agent_runtime: AgentRuntime) -> Dict[str, Any]:
        """Get summary of agent's tools for display/logging."""
        return {
            "total_tools": len(agent_runtime.tools),
            "tools": [
                {"name": t.get("name"), "category": t.get("category"), "provider": t.get("provider")}
                for t in agent_runtime.tools
            ],
            "categories": list(set(t.get("category") for t in agent_runtime.tools if t.get("category"))),
        }

    def cleanup(self):
        """Clean up resources."""
        if self.db_session:
            self.db_session.close()
