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
from core.llm.defaults import DEFAULT_MAX_OUTPUT_TOKENS
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
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
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
        from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
        return ModelConfiguration(
            provider=data.get("provider") or DEFAULT_LLM_PROVIDER,
            model_id=data.get("model_id", DEFAULT_LLM_MODEL),
            temperature=data.get("temperature", 0.7),
            max_tokens=data.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
            top_p=data.get("top_p", 1.0),
            frequency_penalty=data.get("frequency_penalty", 0.0),
            presence_penalty=data.get("presence_penalty", 0.0),
            fallback_model_id=data.get("fallback_model_id"),
        )

    @staticmethod
    def get_default() -> "ModelConfiguration":
        from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
        return ModelConfiguration(provider=DEFAULT_LLM_PROVIDER, model_id=DEFAULT_LLM_MODEL)


@dataclass
class AgentMetadata:
    """User-defined agent metadata — completely flexible."""
    name: str
    agent_type: str
    description: Optional[str] = None
    persona: Optional[str] = None
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
            from core.llm.defaults import DEFAULT_LLM_PROVIDER
            provider = DEFAULT_LLM_PROVIDER
            if "claude" in self.preferred_model.lower():
                provider = "anthropic"
            elif "llama" in self.preferred_model.lower() or "mistral" in self.preferred_model.lower():
                provider = "huggingface"
            return ModelConfiguration(
                provider=provider,
                model_id=self.preferred_model,
                temperature=self.temperature or 0.7,
                max_tokens=self.max_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
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
        """Get default LLM config — used when agent has no model_config.

        The single source of truth is Settings > Orchestrator (Auto agent row).
        This fallback is only hit for non-Auto agents without their own config.
        We use safe defaults (OpenRouter/Gemini Flash) instead of reading from
        system_settings which may have stale cached values.
        """
        try:
            from core.llm.manager import get_system_setting

            provider = get_system_setting("orchestrator_llm", "provider")
            if not provider:
                provider = get_system_setting("orchestrator_llm", "llm_provider")
            model = get_system_setting("orchestrator_llm", "model")
            if not model:
                model = get_system_setting("orchestrator_llm", "llm_model")
            from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
            if not provider:
                provider = DEFAULT_LLM_PROVIDER
            if not model:
                model = DEFAULT_LLM_MODEL
            if not provider or not model:
                self.logger.warning("LLM provider/model not in system settings, using defaults")
                return {
                    "provider": DEFAULT_LLM_PROVIDER,
                    "model": DEFAULT_LLM_MODEL,
                    "temperature": 0.7,
                    "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                    "context_window": 8192,
                }

            # Lookup context window from LLM models registry
            context_window = 8192
            max_tokens = DEFAULT_MAX_OUTPUT_TOKENS
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
            from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
            self.logger.warning(f"Could not get LLM config from settings: {e}, using defaults")
            return {
                "provider": DEFAULT_LLM_PROVIDER,
                "model": DEFAULT_LLM_MODEL,
                "temperature": 0.7,
                "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                "context_window": 8192,
            }

    def _get_system_llm_config_from_settings(self) -> Dict[str, Any]:
        """Get System LLM config — the cheap/fast tier for Light power mode."""
        try:
            from core.llm.manager import get_system_setting
            from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL

            provider = get_system_setting("system_llm", "provider") or get_system_setting("system_llm", "llm_provider")
            model = get_system_setting("system_llm", "model") or get_system_setting("system_llm", "llm_model")

            if not provider:
                provider = DEFAULT_LLM_PROVIDER
            if not model:
                model = DEFAULT_LLM_MODEL

            context_window = 8192
            max_tokens = DEFAULT_MAX_OUTPUT_TOKENS
            try:
                from core.models import LLMModel
                llm_model = self.db_session.query(LLMModel).filter_by(model_id=model).first()
                if llm_model:
                    context_window = llm_model.context_window
                    max_tokens = llm_model.max_output_tokens
            except Exception as e:
                self.logger.warning(f"Could not get context window for system_llm: {e}")

            self.logger.info(f"System LLM from settings: {model} (context: {context_window})")
            return {
                "provider": provider,
                "model": model,
                "temperature": 0.7,
                "max_tokens": max_tokens,
                "context_window": context_window,
            }
        except Exception as e:
            from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
            self.logger.warning(f"Could not get system_llm config: {e}, using defaults")
            return {
                "provider": DEFAULT_LLM_PROVIDER,
                "model": DEFAULT_LLM_MODEL,
                "temperature": 0.7,
                "max_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
                "context_window": 8192,
            }

    def _model_max_output_tokens(self, model_id: Optional[str]) -> int:
        """The selected model's own output ceiling from the LLM registry, or the
        canonical default if the model isn't found. Lets an agent that hasn't set
        an explicit Max Output Tokens default to what its model actually supports
        — never a hardcoded literal."""
        if model_id and self.db_session is not None:
            try:
                from core.models import LLMModel
                m = self.db_session.query(LLMModel).filter_by(model_id=model_id).first()
                if m and m.max_output_tokens:
                    return m.max_output_tokens
            except Exception as e:
                self.logger.warning(f"max_output_tokens lookup failed for {model_id}: {e}")
        return DEFAULT_MAX_OUTPUT_TOKENS

    def _resolve_provider_for_model(self, provider_str: str, model_id: str) -> tuple[str, str]:
        """Auto-detect and correct provider-model mismatches.

        Returns (provider, model_id) — both may be rewritten.
        """
        DIRECT_PROVIDERS = {
            "openai", "anthropic", "google", "grok", "azure", "azure_openai",
            "aws_bedrock", "bedrock", "huggingface", "openrouter",
        }
        # Known OpenRouter vendor prefixes for bare model-id recovery
        VENDOR_PREFIX_RULES = (
            ("llama", "meta-llama"),
            ("qwen", "qwen"),
            ("deepseek", "deepseek-ai"),
            ("mistral", "mistralai"),
            ("mixtral", "mistralai"),
            ("gemma", "google"),
            ("phi", "microsoft"),
            ("yi-", "01-ai"),
        )
        model_lower = model_id.lower()

        # Unknown/deprecated provider (e.g. legacy 'aiml') → route through OpenRouter
        if provider_str not in DIRECT_PROVIDERS:
            # Already slash-format: pass straight to OpenRouter
            if "/" in model_id:
                self.logger.info(
                    f"Provider '{provider_str}' not recognized, routing '{model_id}' through OpenRouter"
                )
                return "openrouter", model_id
            # Bare id → infer vendor prefix so OpenRouter can resolve it
            for needle, vendor in VENDOR_PREFIX_RULES:
                if model_lower.startswith(needle):
                    rewritten = f"{vendor}/{model_id}"
                    self.logger.warning(
                        f"Provider '{provider_str}' unknown and model '{model_id}' has no slash; "
                        f"rewriting to '{rewritten}' and routing through OpenRouter"
                    )
                    return "openrouter", rewritten
            # Last resort: assume OpenRouter can handle it
            self.logger.warning(
                f"Provider '{provider_str}' unknown; routing bare model '{model_id}' through OpenRouter"
            )
            return "openrouter", model_id

        # Slash-format model IDs are OpenRouter marketplace models
        if "/" in model_id and provider_str != "openrouter":
            prefix = model_id.split("/")[0].lower()
            if prefix == provider_str.lower() or prefix not in DIRECT_PROVIDERS:
                self.logger.info(
                    f"Slash-format model '{model_id}' with provider='{provider_str}' "
                    f"detected as OpenRouter marketplace model. Routing through OpenRouter."
                )
                return "openrouter", model_id

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
            return inferred, model_id

        return provider_str, model_id

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

        # 1.5 Operator workspace key (PLATFORM_KEY_WORKSPACE_ID) — the pilot
        # "platform key" lane, read live from user_api_keys instead of a
        # duplicated credential-store copy that can drift (2026-07-30).
        from core.llm.workspace_keys import get_platform_workspace_key
        ws_key = get_platform_workspace_key(provider_name)
        if ws_key:
            self.logger.info(
                f"Resolved platform key from operator workspace store for '{provider_name}' ({agent_name})"
            )
            return ResolvedKey(api_key=ws_key, source="platform_workspace", is_byok=False, provider=provider_name)

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

        effective_provider, effective_model_id = self._resolve_provider_for_model(
            model_config.provider, model_config.model_id
        )
        if effective_provider not in provider_map:
            raise ValueError(f"Unsupported provider: {effective_provider}")

        provider = provider_map[effective_provider]
        resolved = await self._resolve_api_key(effective_provider, agent_name, workspace_id=workspace_id)
        if not resolved:
            raise ValueError(
                f"No API key available for {effective_provider}. "
                f"Add one in Settings → API Keys, or set platform fallback "
                f"({effective_provider.upper()}_API_KEY env var on the API service)."
            )

        llm_config = LLMConfig(
            provider=provider,
            model=effective_model_id,
            temperature=model_config.temperature,
            max_tokens=model_config.max_tokens,
            api_key=resolved.api_key,
            top_p=getattr(model_config, 'top_p', None),
            frequency_penalty=getattr(model_config, 'frequency_penalty', None),
            presence_penalty=getattr(model_config, 'presence_penalty', None),
            stop=getattr(model_config, 'stop', None),
            timeout=getattr(model_config, 'timeout', None),
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
        use_orchestrator_llm: bool = False,
        force_llm_tier: Optional[str] = None,
    ) -> Optional[AgentRuntime]:
        """Load an agent from database and activate it in runtime.

        PRD-137 Fix #2: parameter renamed from ``use_system_llm`` to
        ``use_orchestrator_llm`` to match what the code actually does.

        ``force_llm_tier``: when set to ``"orchestrator_llm"`` or
        ``"system_llm"``, overrides the agent's own model with the
        corresponding tier from system_settings. Used by mission
        power modes (Light → system_llm, Max → orchestrator_llm).
        """
        try:
            if agent_id in self.active_agents:
                self.logger.info(f"Agent {agent_id} already active in runtime")
                return self.active_agents[agent_id]

            from modules.agents.queries import get_agent_with_context

            db_agent = get_agent_with_context(self.db_session, agent_id)
            if not db_agent:
                self.logger.error(f"Agent {agent_id} not found in database")
                return None

            # Resolve LLM config: agent's own model_config → orchestrator-tier defaults
            agent_model_config = db_agent.model_config or {}
            agent_config = db_agent.configuration or {}
            agent_llm_config = agent_config.get("llm_config") or {}

            agent_has_model = agent_model_config.get("model_id") and agent_model_config.get("provider")

            if force_llm_tier == "system_llm":
                tier_config = self._get_system_llm_config_from_settings()
                llm_config_dict = {
                    "provider": tier_config.get("provider"),
                    "model": tier_config.get("model"),
                    "temperature": agent_llm_config.get("temperature", tier_config.get("temperature", 0.7)),
                    "max_tokens": tier_config.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
                }
                self.logger.info(f"Agent {agent_id} using LLM: {llm_config_dict.get('provider')}/{llm_config_dict.get('model')} (force_llm_tier=system_llm)")
            elif force_llm_tier == "orchestrator_llm" or use_orchestrator_llm:
                reason = f"force_llm_tier={force_llm_tier}" if force_llm_tier else "use_orchestrator_llm=True"
                orchestrator_llm_config = self._get_default_llm_config_from_settings()
                llm_config_dict = {
                    "provider": orchestrator_llm_config.get("provider"),
                    "model": orchestrator_llm_config.get("model"),
                    "temperature": agent_llm_config.get("temperature", orchestrator_llm_config.get("temperature", 0.7)),
                    "max_tokens": agent_llm_config.get("max_tokens", orchestrator_llm_config.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
                }
                self.logger.info(f"Agent {agent_id} using LLM: {llm_config_dict.get('provider')}/{llm_config_dict.get('model')} ({reason})")
            elif agent_has_model:
                llm_config_dict = {
                    "provider": agent_model_config["provider"],
                    "model": agent_model_config["model_id"],
                    "temperature": agent_model_config.get("temperature", 0.7),
                    "max_tokens": agent_model_config.get("max_tokens") or self._model_max_output_tokens(agent_model_config.get("model_id")),
                }
                self.logger.info(f"Agent {agent_id} using LLM: {llm_config_dict['provider']}/{llm_config_dict['model']} (agent model_config)")
            else:
                orchestrator_llm_config = self._get_default_llm_config_from_settings()
                llm_config_dict = {
                    "provider": orchestrator_llm_config.get("provider"),
                    "model": orchestrator_llm_config.get("model"),
                    "temperature": agent_llm_config.get("temperature", orchestrator_llm_config.get("temperature", 0.7)),
                    "max_tokens": agent_llm_config.get("max_tokens", orchestrator_llm_config.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS)),
                }
                self.logger.info(f"Agent {agent_id} using LLM: {llm_config_dict.get('provider')}/{llm_config_dict.get('model')} (no agent model_config)")

            # PRD-223 W0: the orchestrator seat is policy-gated. This is the
            # last-mile check — every writer path (settings route, per-agent
            # route, chat tool, seeds) converges here, so a quarantined model
            # cannot reach Auto's chair no matter how it was written. On a
            # block, degrade to a trusted brain (orchestrator default, then
            # platform default) — never a dead chat.
            is_orchestrator_seat = (
                use_orchestrator_llm
                or force_llm_tier == "orchestrator_llm"
                or (
                    getattr(db_agent, "name", "") == "Auto"
                    and bool(getattr(db_agent, "is_system_agent", False))
                )
            )
            if is_orchestrator_seat:
                from core.llm.model_policy import check_orchestrator_model

                allowed, reason = check_orchestrator_model(llm_config_dict.get("model"))
                if not allowed:
                    blocked_model = llm_config_dict.get("model")
                    fallback = self._get_default_llm_config_from_settings()
                    fb_ok, _ = check_orchestrator_model(fallback.get("model"))
                    if not fb_ok:
                        from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
                        fallback = {"provider": DEFAULT_LLM_PROVIDER, "model": DEFAULT_LLM_MODEL}
                    llm_config_dict["provider"] = fallback.get("provider")
                    llm_config_dict["model"] = fallback.get("model")
                    self.logger.critical(
                        f"[model-policy] Agent {agent_id} orchestrator-seat model "
                        f"'{blocked_model}' BLOCKED ({reason}) — substituting "
                        f"'{llm_config_dict['provider']}/{llm_config_dict['model']}' (PRD-223)"
                    )

            from core.llm.defaults import DEFAULT_LLM_PROVIDER, DEFAULT_LLM_MODEL
            provider_str = llm_config_dict.get("provider") or DEFAULT_LLM_PROVIDER
            model_id_str = llm_config_dict.get("model", DEFAULT_LLM_MODEL)
            provider_str, model_id_str = self._resolve_provider_for_model(provider_str, model_id_str)

            try:
                provider = LLMProvider(provider_str)
            except ValueError:
                # Safety net: any provider that survived the resolver without being
                # normalised (e.g. stale enum value) is forced through OpenRouter.
                self.logger.warning(
                    f"Provider '{provider_str}' is not a valid LLMProvider enum; forcing OpenRouter"
                )
                provider_str = "openrouter"
                if "/" not in model_id_str:
                    original = (llm_config_dict.get("provider") or "").lower()
                    if original and original != provider_str:
                        model_id_str = f"{original}/{model_id_str}"
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
                max_tokens=llm_config_dict.get("max_tokens", DEFAULT_MAX_OUTPUT_TOKENS),
                api_key=resolved.api_key if resolved else None,
                top_p=llm_config_dict.get("top_p"),
                frequency_penalty=llm_config_dict.get("frequency_penalty"),
                presence_penalty=llm_config_dict.get("presence_penalty"),
                stop=llm_config_dict.get("stop"),
                timeout=llm_config_dict.get("timeout"),
            )
            llm_manager = LLMManager(
                config=llm_config,
                workspace_id=db_agent.workspace_id,
                agent_id=agent_id,
                is_byok=resolved.is_byok if resolved else False,
            )

            persona_text = ""
            if getattr(db_agent, "use_custom_persona", False) and getattr(db_agent, "custom_persona_prompt", None):
                persona_text = db_agent.custom_persona_prompt
            elif getattr(db_agent, "persona", None) and getattr(db_agent.persona, "system_prompt", None):
                persona_text = db_agent.persona.system_prompt

            metadata = AgentMetadata(
                name=db_agent.name,
                agent_type=db_agent.agent_type,
                description=db_agent.description,
                persona=persona_text or None,
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
                    # Note: attachment_ids intentionally NOT passed here. AgentFactory
                    # builds its own messages list and only consumes context_result's
                    # system_prompt/tools, so any attachment parts ContextService injects
                    # into context_result.messages would be discarded. Resolution happens
                    # below after the user prompt is appended — single chokepoint for
                    # both system_prompt-bypass and ContextService paths.
                    context_result = await ContextService(self.db_session).build_context(
                        mode=mode,
                        agent=db_agent,
                        workspace_id=agent_runtime.workspace_id,
                        task_description=prompt,
                        # Narrow the dispatcher enum to task-relevant actions —
                        # without a query this lane shipped all 137 every run.
                        query=prompt,
                    )
                    # PRD-201 S4: carry the assembler's cache-stable prefix on the
                    # system message so the Anthropic client can place its
                    # cache_control breakpoint there. Non-Anthropic providers
                    # ignore the extra key.
                    messages.append({
                        "role": "system",
                        "content": context_result.system_prompt,
                        "cache_prefix": context_result.cacheable_prefix,
                    })
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

            # Preserve original prompt for Composio hint generation
            original_user_prompt = prompt

            # --- Build tools ---
            if context_result is not None:
                # ContextService already loaded tools via ToolsSection
                tool_schemas = list(context_result.tools)
            else:
                # Explicit system_prompt path: load tools directly
                from modules.tools.tool_router import get_tools_for_agent_async

                # PRD-138 US-009: pass the user prompt so the dispatcher
                # enum narrows in lockstep with the prompt summary. Falls
                # back to the full enum if SEMANTIC_TOOL_ROUTING is off,
                # the prompt is empty, or ranking fails. Awaited on this
                # loop — never bridged through a helper thread.
                tool_schemas = await get_tools_for_agent_async(
                    agent_id=agent_runtime.agent_id,
                    db_session=self.db_session,
                    workspace_id=agent_runtime.workspace_id,
                    query=prompt,
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

            # PRD-127: Single chokepoint for attachment resolution in non-chat LLM calls.
            # Works for both branches above (system_prompt bypass + ContextService path)
            # because it runs on the final messages list. Any entry point that forwards
            # attachment_ids through to execute_with_prompt gets multimodal support for
            # free — no per-handler wiring, no duplicated vision checks.
            if attachment_ids:
                try:
                    from uuid import UUID as _UUID
                    from modules.attachments.resolver import (
                        AttachmentResolver,
                        VisionNotSupportedError,
                        inject_parts_into_last_user_message,
                    )
                    _llm_cfg = getattr(agent_runtime.llm_manager, "config", None)
                    _model_id = getattr(_llm_cfg, "model", None) if _llm_cfg else None
                    _resolver = AttachmentResolver(db_session=self.db_session)
                    _parts, _att_failures = await _resolver.resolve(
                        attachment_ids=[_UUID(a) for a in attachment_ids],
                        workspace_id=_UUID(str(agent_runtime.workspace_id)),
                        model_id=_model_id or "",
                    )
                    if _parts:
                        messages = inject_parts_into_last_user_message(messages, _parts)
                        self.logger.info(
                            f"[PRD-127] AgentFactory resolved {len(_parts)} attachment parts "
                            f"from {len(attachment_ids)} ids for agent {agent_id}"
                        )
                    if _att_failures:
                        # PRD-223 S0.3: marker part already rides in _parts.
                        self.logger.warning(
                            f"[PRD-223] AgentFactory: {len(_att_failures)} attachment(s) "
                            f"unavailable for agent {agent_id}"
                        )
                except VisionNotSupportedError as _vne:
                    self.logger.warning(
                        f"[PRD-127] AgentFactory vision not supported for agent {agent_id}: {_vne}"
                    )
                except Exception as _att_err:
                    self.logger.error(
                        f"[PRD-127] AgentFactory attachment resolution failed: {_att_err}",
                        exc_info=True,
                    )

            # --- Execute with retries ---
            last_error = None
            messages_snapshot = list(messages)  # snapshot before retry loop

            # PRD-201 S5: generalise ContextGuard beyond chat — missions and
            # heartbeats now get the same model-aware compaction the chat path
            # already had, run once before the tool loop. Guarded: a compaction
            # fault must never fail the run.
            try:
                from core.context_guard import ContextGuard as _ContextGuard
                _guard_model = getattr(
                    getattr(agent_runtime.llm_manager, "config", None), "model", None
                ) or ""
                _compacted, _was_compacted, _guarded_tools = await _ContextGuard().check_and_compact(
                    messages=messages_snapshot,
                    model_name=_guard_model,
                    llm_manager=agent_runtime.llm_manager,
                    workspace_id=str(agent_runtime.workspace_id),
                    agent_id=agent_runtime.agent_id,
                    db_session=self.db_session,
                    tools=tool_schemas,
                )
                if _was_compacted:
                    messages_snapshot = _compacted
                    self.logger.info(
                        "[PRD-201 S5] ContextGuard compacted headless context for agent %s",
                        agent_id,
                    )
                if _guarded_tools is None and tool_schemas:
                    tool_schemas = []
                    self.logger.warning(
                        "[PRD-201 S5] ContextGuard dropped tools (over budget) for agent %s",
                        agent_id,
                    )
            except Exception as _cg_err:
                self.logger.debug("[PRD-201 S5] ContextGuard skipped: %s", _cg_err)

            from core.llm.request_scope import headless_run

            for attempt in range(max(1, max_retries)):
                try:
                    messages = list(messages_snapshot)  # reset each attempt
                    # PRD-201 S5: mark the Anthropic call as a headless run so the
                    # client seam emits context-editing + the memory tool.
                    with headless_run():
                        response = await agent_runtime.llm_manager.generate_response(messages, tools=tool_schemas)
                    execution_time = time.time() - start_time

                    # --- Converged tool loop (PRD-142 W3-S4 / G6): same executor as chat ---
                    from modules.tools.execution.tool_loop import ToolLoopExecutor

                    async def _agent_llm_cb(msgs, tls):
                        # PRD-201 S5: keep the headless scope across the tool loop's
                        # re-invocations so every iteration emits context-editing.
                        with headless_run():
                            return await agent_runtime.llm_manager.generate_response(msgs, tools=tls)

                    # PRD-178 S1 (F020): thread the calling task's field context
                    # so PlatformActionExecutor binds field tools to THIS run's
                    # field_id — never a `.first()` guess over concurrent missions.
                    _field_caller_context = (
                        {"field_context": context}
                        if context and context.get("field_id") else None
                    )
                    # PRD-193 S1/S4 (P2-12): a confirmation ask fired on a
                    # board-task run must link its grant back to the task
                    # (details.board_task_id), so the grant API's existing
                    # board re-queue resumes the run into the now-active
                    # grant. api/board_tasks passes source/task_id in context.
                    if (
                        context
                        and context.get("source") == "board_task"
                        and context.get("task_id") is not None
                    ):
                        _field_caller_context = {
                            **(_field_caller_context or {}),
                            "board_task_id": context.get("task_id"),
                        }

                    async def _agent_tool_cb(name, args, call_id, ws_id):
                        # PRD-201 S5: the Anthropic memory tool is client-executed —
                        # run it against the durable store with the /memories
                        # traversal guard, never through the platform tool registry.
                        if name == "memory":
                            from modules.memory.memory_tool import (
                                DurableMemoryStoreBackend,
                                MemoryToolBackend,
                            )
                            _mem_result = await MemoryToolBackend(
                                DurableMemoryStoreBackend(), workspace_id=ws_id
                            ).handle(args or {})
                            return {
                                "success": True,
                                "llm_context": _mem_result,
                                "raw_result": _mem_result,
                            }
                        # SLACK empty-params guard.
                        if not args and "SLACK" in name:
                            return {
                                "success": False,
                                "llm_context": json.dumps(
                                    {"error": "Empty parameters for tool requiring input"}
                                ),
                            }
                        try:
                            # PRD-192 S3: turn-level budget estimate on the
                            # agent lane (same shared estimator as chat) so
                            # the policy gate prices this call.
                            from core.context_guard import estimate_turn_budget
                            _turn_budget = estimate_turn_budget(
                                agent_runtime.llm_manager, messages
                            )
                            _cb_caller_context = (
                                {**(_field_caller_context or {}), **_turn_budget}
                                or None
                            )
                            raw = await agent_runtime.tool_executor.execute_tool(
                                tool_name=name,
                                parameters=args,
                                agent_id=agent_runtime.agent_id,
                                workspace_id=ws_id,
                                caller_context=_cb_caller_context,
                            )
                            return {
                                "success": True,
                                "llm_context": json.dumps(raw),
                                "raw_result": raw,
                            }
                        except Exception as tool_err:
                            self.logger.error(f"    [TRACE] {name} failed: {tool_err}")
                            return {
                                "success": False,
                                "llm_context": json.dumps({"error": str(tool_err)}),
                            }

                    loop_executor = ToolLoopExecutor(
                        llm_callback=_agent_llm_cb,
                        tool_callback=_agent_tool_cb,
                        max_iterations=max_tool_iterations,
                        content_truncate_tokens=0,  # agent path historically did not truncate
                    )
                    loop_result = await loop_executor.run(
                        initial_response=response,
                        messages=messages,
                        tools=tool_schemas,
                        workspace_id=workspace_id,
                    )
                    response = loop_result.response
                    tool_iteration = loop_result.iterations
                    execution_time = time.time() - start_time

                    # Recover the last round's tool messages for empty-response synthesis.
                    tool_results: List[Dict[str, Any]] = []
                    for msg in reversed(messages):
                        if msg.get("role") == "tool":
                            tool_results.append(msg)
                        elif tool_results:
                            break
                    tool_results.reverse()

                    if loop_result.max_iterations_reached:
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
