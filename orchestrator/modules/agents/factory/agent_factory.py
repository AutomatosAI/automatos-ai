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
from core.models.composio_cache import AgentAppAssignment, ComposioActionCache, ComposioAppCache
from modules.agents.services.skill_loader import get_skill_loader


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
    system_prompt: Optional[str] = None
    skill_tool_schemas: List[Dict[str, Any]] = field(default_factory=list)

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

            db_agent = self.db_session.query(Agent).filter(Agent.id == agent_id).first()
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

            # Build system prompt at activation time (single injection point)
            system_prompt, skill_tool_schemas = self._build_agent_system_prompt(
                agent=db_agent,
                db=self.db_session,
            )

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
                system_prompt=system_prompt,
                skill_tool_schemas=skill_tool_schemas,
            )

            self.active_agents[agent_id] = agent_runtime
            db_agent.status = AgentLifecycle.ACTIVE.value
            self.db_session.commit()

            self.logger.info(f"Activated agent {agent_id} ({db_agent.name}) with {llm_config_dict.get('model')}")
            return agent_runtime

        except Exception as e:
            self.logger.error(f"Failed to activate agent {agent_id}: {e}")
            return None

    def refresh_agent_prompt(self, agent_id: int) -> bool:
        """Rebuild the system prompt for an already-active agent."""
        runtime = self.active_agents.get(agent_id)
        if not runtime:
            return False

        db_agent = self.db_session.query(Agent).filter(Agent.id == agent_id).first()
        if not db_agent:
            return False

        system_prompt, skill_tool_schemas = self._build_agent_system_prompt(
            agent=db_agent,
            db=self.db_session,
        )
        runtime.system_prompt = system_prompt
        runtime.skill_tool_schemas = skill_tool_schemas
        self.logger.info(f"Refreshed system prompt for agent {agent_id}")
        return True

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

            # System prompt: explicit > cached > ContextService
            skill_tool_schemas_from_prompt = []
            context_result = None  # Populated when ContextService builds the prompt

            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            elif getattr(agent_runtime, "system_prompt", None):
                messages.append({"role": "system", "content": agent_runtime.system_prompt})
                skill_tool_schemas_from_prompt = getattr(agent_runtime, "skill_tool_schemas", [])
            else:
                # --- ContextService path (replaces _build_agent_system_prompt) ---
                from modules.context import ContextService, ContextMode

                db_agent = self.db_session.query(Agent).filter_by(id=agent_runtime.agent_id).first()
                if db_agent:
                    context_result = await ContextService(self.db_session).build_context(
                        mode=ContextMode.TASK_EXECUTION,
                        agent=db_agent,
                        workspace_id=agent_runtime.workspace_id,
                        task_description=prompt,
                    )
                    messages.append({"role": "system", "content": context_result.system_prompt})

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
                # Explicit/cached system_prompt path: load tools directly
                from modules.tools.tool_router import get_tools_for_agent

                tool_schemas = get_tools_for_agent(
                    agent_id=agent_runtime.agent_id,
                    db_session=self.db_session,
                    workspace_id=agent_runtime.workspace_id,
                )

                # Add skill-based tools from prompt building
                if skill_tool_schemas_from_prompt:
                    tool_schemas.extend(skill_tool_schemas_from_prompt)
                    tool_names = [t["function"]["name"] for t in skill_tool_schemas_from_prompt]
                    self.logger.info(f"Added {len(skill_tool_schemas_from_prompt)} skill tools: {tool_names}")

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
        """Inject Composio hints via hint service (default/chatbot path)."""
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

            composio_apps = [t for t in (agent_runtime.tools or []) if t.get("provider") == "Composio"]
            app_names = [t.get("name") for t in composio_apps]
            self.logger.info(
                f"Composio: hints={len(hint_result.hint_lines)} lines, "
                f"strategy={hint_result.strategy_used}, "
                f"constrained_actions={len(hint_result.matched_actions)}, "
                f"apps={app_names}"
            )
        except Exception as e:
            self.logger.warning(f"Failed to inject Composio hints: {e}")

    # ==================================================================
    # System Prompt Builder
    # ==================================================================

    def _build_agent_system_prompt(
        self,
        agent: Agent,
        task_context: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> Tuple[str, List[Dict]]:
        """
        Build the agent system prompt AND extract skill tool schemas.

        Single injection point — loads ALL assigned skills AND plugin content.
        """
        sections: List[str] = []
        skill_tool_schemas: List[Dict] = []

        # Identity
        sections.append(f"# Agent: {agent.name}")
        sections.append(f"Agent ID: {agent.id}")
        sections.append(f"Agent Type: {getattr(agent, 'agent_type', 'unknown')}")
        if agent.description:
            sections.append(agent.description)

        # Persona
        try:
            if getattr(agent, "use_custom_persona", False) and agent.custom_persona_prompt:
                sections.append(f"\n## Persona & Communication Style\n{agent.custom_persona_prompt}")
            elif getattr(agent, "persona_id", None) and getattr(agent, "persona", None):
                persona_prompt = agent.persona.system_prompt or ""
                if persona_prompt:
                    sections.append(f"\n## Persona & Communication Style\n{persona_prompt}")
        except Exception as e:
            self.logger.warning(f"Failed to load persona for agent {agent.id}: {e}")

        # Task context
        if task_context:
            sections.append("\n## Task Context\n" + str(task_context))

        # Skills
        if getattr(agent, "skills", None):
            active_skills = [s for s in agent.skills if s.is_active]
            if active_skills:
                sections.append("\n## Your Specialized Skills\n")
                loader = get_skill_loader(db) if db is not None else None
                loaded_skill_ids: set = set()

                for skill in active_skills:
                    if skill.id in loaded_skill_ids:
                        continue
                    loaded_skill_ids.add(skill.id)
                    self.logger.info(f"Loading skill: {skill.name}")
                    sections.append(f"### {skill.name}")

                    # Load prompt content
                    core_content = None
                    if loader is not None:
                        try:
                            core_content = loader.load_skill_core(skill.name, db=db)
                        except Exception as e:
                            self.logger.warning(f"Failed to load core content for '{skill.name}': {e}")

                    if core_content and isinstance(core_content, str) and core_content.strip():
                        sections.append(core_content)
                    else:
                        fallback = skill.prompt_template or skill.description or ""
                        if fallback:
                            sections.append(str(fallback))

                    # Extract tool schemas from skills
                    if hasattr(skill, "tools_schema") and skill.tools_schema and isinstance(skill.tools_schema, dict):
                        try:
                            for tool_def in skill.tools_schema.get("tools", []):
                                tool_name = tool_def.get("name")
                                skill_tool_schemas.append({
                                    "type": "function",
                                    "function": {
                                        "name": tool_name,
                                        "description": tool_def.get("description", ""),
                                        "parameters": tool_def.get("parameters", {}),
                                    },
                                })
                                self.logger.info(f"  Extracted tool: {tool_name}")
                        except Exception as e:
                            self.logger.warning(f"Failed to extract tools from skill '{skill.name}': {e}")

                self.logger.info(f"Loaded {len(active_skills)} skill(s) for agent {agent.id}")

        # Plugin content (non-materialized only)
        try:
            from core.services.plugin_context_service import PluginContextService

            plugin_svc = PluginContextService(db) if db else None
            if plugin_svc:
                plugin_rows = plugin_svc.get_assigned_plugins(agent.id)
                if plugin_rows:
                    non_materialized = []
                    for row in plugin_rows:
                        _aap, plugin = row if isinstance(row, tuple) else (row, getattr(row, "plugin", row))
                        materialized_ids = getattr(plugin, "materialized_skill_ids", None) or []
                        if not materialized_ids:
                            non_materialized.append(row)

                    if non_materialized:
                        tier1 = plugin_svc.build_tier1_summary(non_materialized)
                        if tier1:
                            sections.append(tier1)
                        tier2 = plugin_svc.build_tier2_content_sync(non_materialized, task_context=task_context)
                        if tier2:
                            sections.append(tier2)
        except Exception as e:
            self.logger.warning(f"Failed to load plugins for agent {agent.id}: {e}")

        # Composio apps section in prompt
        if db and agent.id:
            try:
                assignments = (
                    db.query(AgentAppAssignment)
                    .filter(
                        AgentAppAssignment.agent_id == agent.id,
                        AgentAppAssignment.is_active.is_(True),
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
                        "You have access to these external apps via Composio. "
                        "Use the `composio_execute` tool with an appropriate action.\n"
                    )
                    for assignment in assignments:
                        app_name = (assignment.app_name or "").upper()
                        app = cache.get(app_name)
                        if app_name:
                            helper_section.append(f"### {app_name}")
                            if app and app.description:
                                helper_section.append(f"**Description**: {app.description}")
                    sections.append("\n".join(helper_section))
            except Exception as e:
                self.logger.warning(f"Failed to append Composio apps to prompt: {e}")

        # Dependency context instructions
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

        # Skill tool usage instructions
        if skill_tool_schemas:
            tool_names_list = [t["function"]["name"] for t in skill_tool_schemas]
            sections.append("\n## IMPORTANT: Using Your Skill Tools\n")
            sections.append(f"You have access to: {', '.join(tool_names_list)}")
            sections.append("When your task requires capabilities provided by these tools, you MUST use them via function calling.")
            sections.append("Analyze your task, check if any tools match, and CALL them — do not just describe what you would do.")

        # Response formatting
        sections.append("\n## Response Formatting Rules\n")
        sections.append("When you receive API/tool results:")
        sections.append("- Synthesize data into clear, human-friendly prose — do NOT dump raw JSON")
        sections.append("- NEVER use code blocks or inline code backticks")
        sections.append("- Use bullet points or short paragraphs for a non-technical reader")

        prompt_text = "\n\n".join([s for s in sections if s is not None])
        return (prompt_text, skill_tool_schemas)

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
