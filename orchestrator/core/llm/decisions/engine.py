"""PRD-248 — the decision engine: dials, backend choice, one ``decide()``.

Callers (AutoBrain today, the tool rerank next) never see a provider. They ask
``get_decision_engine().decide(state=..., questions=...)`` and get a
``DecisionResult`` or ``None``; nothing here raises into a chat turn.

The operating dials live in ``system_settings`` category ``decision_engine``
so a PoC flips from Settings → System without a restart; they are read with
defaults (a missing row is "off", never an error) and cached for a short TTL
so a turn costs one settings read, not five.

Shadow rows go to a JSON-lines file (``DECISION_SHADOW_LOG_PATH``) — durable
where container stdout is not, and what ``scripts.eval.decision_shadow.score``
reads.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from config import config

from .llm_adapter import ADAPTER_MIN_TIMEOUT_S, PROVIDER_LLM, LLMDecisionAdapter
from .questions import DecisionResult, Question
from .typesafe_client import (
    DEFAULT_MODELS,
    PROVIDER_OPENROUTER,
    PROVIDER_TYPESAFE,
    TypeSafeDecisionClient,
)

logger = logging.getLogger(__name__)

SETTINGS_CATEGORY = "decision_engine"

MODE_OFF = "off"
MODE_SHADOW = "shadow"
MODE_LIVE = "live"
MODES = (MODE_OFF, MODE_SHADOW, MODE_LIVE)

PROVIDERS = (PROVIDER_OPENROUTER, PROVIDER_TYPESAFE, PROVIDER_LLM)

DIALS_TTL_SECONDS = 30.0
TIMEOUT_MIN_S = 0.2
TIMEOUT_MAX_S = 30.0


@dataclass(frozen=True)
class Dials:
    """The ``decision_engine`` settings, defaulted. ``model`` empty means the
    route's pinned default."""

    provider: str = PROVIDER_OPENROUTER
    model: str = ""
    timeout_seconds: float = 2.5
    min_confidence: float = 0.7
    classifier_mode: str = MODE_OFF
    tool_rerank_mode: str = MODE_OFF

    @property
    def any_on(self) -> bool:
        return self.classifier_mode != MODE_OFF or self.tool_rerank_mode != MODE_OFF


DEFAULT_DIALS = Dials()

SettingsReader = Callable[[str, str, Optional[str]], Optional[str]]


def _default_reader(category: str, key: str, default: Optional[str]) -> Optional[str]:
    from core.llm.manager import get_system_setting

    return get_system_setting(category, key, default)


def _as_mode(raw: str, default: str) -> str:
    value = (raw or "").strip().lower()
    return value if value in MODES else default


def _as_float(raw: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(raw)))
    except (TypeError, ValueError):
        return default


class DecisionEngine:
    def __init__(
        self,
        *,
        settings_reader: Optional[SettingsReader] = None,
        backend_factory: Optional[Callable[[Dials], Any]] = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._read = settings_reader or _default_reader
        self._backend_factory = backend_factory
        self._clock = clock
        self._dials: Optional[Dials] = None
        self._dials_at = 0.0
        self._backend: Any = None
        self._backend_dials: Optional[Dials] = None
        self._lock = threading.Lock()

    # -- dials --------------------------------------------------------------

    def dials(self, force: bool = False) -> Dials:
        now = self._clock()
        with self._lock:
            fresh = self._dials is not None and (now - self._dials_at) < DIALS_TTL_SECONDS
            if fresh and not force:
                return self._dials  # type: ignore[return-value]
        loaded = self._load_dials()
        with self._lock:
            self._dials = loaded
            self._dials_at = now
        return loaded

    def reload(self) -> Dials:
        return self.dials(force=True)

    def _setting(self, key: str, default: str) -> str:
        try:
            value = self._read(SETTINGS_CATEGORY, key, None)
        except Exception:  # noqa: BLE001 — a settings read must never break a turn
            logger.debug("[decision] settings read failed for %s", key, exc_info=True)
            value = None
        if value is None or str(value).strip() == "":
            return default
        return str(value).strip()

    def _load_dials(self) -> Dials:
        d = DEFAULT_DIALS
        provider = self._setting("provider", d.provider).lower()
        if provider not in PROVIDERS:
            logger.warning("[decision] unknown provider %r — using %s", provider, d.provider)
            provider = d.provider
        return Dials(
            provider=provider,
            model=self._setting("model", d.model),
            timeout_seconds=_as_float(
                self._setting("timeout_seconds", str(d.timeout_seconds)),
                d.timeout_seconds, TIMEOUT_MIN_S, TIMEOUT_MAX_S,
            ),
            min_confidence=_as_float(
                self._setting("min_confidence", str(d.min_confidence)), d.min_confidence, 0.0, 1.0
            ),
            classifier_mode=_as_mode(self._setting("classifier_mode", d.classifier_mode), d.classifier_mode),
            tool_rerank_mode=_as_mode(self._setting("tool_rerank_mode", d.tool_rerank_mode), d.tool_rerank_mode),
        )

    # -- backend ------------------------------------------------------------

    def backend(self, dials: Optional[Dials] = None) -> Any:
        """The backend for the current dials, or None when the route has no key.
        Rebuilt only when the routing dials change (the HTTP client keeps its
        connection pool across turns)."""
        d = dials or self.dials()
        routing = replace(d, classifier_mode=MODE_OFF, tool_rerank_mode=MODE_OFF, min_confidence=0.0)
        with self._lock:
            if self._backend is not None and self._backend_dials == routing:
                return self._backend
        backend = self._build_backend(d)
        with self._lock:
            self._backend = backend
            self._backend_dials = routing
        return backend

    def _build_backend(self, d: Dials) -> Any:
        if self._backend_factory is not None:
            return self._backend_factory(d)
        if d.provider == PROVIDER_LLM:
            return LLMDecisionAdapter(timeout_s=max(d.timeout_seconds, ADAPTER_MIN_TIMEOUT_S))
        client = TypeSafeDecisionClient(
            provider=d.provider,
            model=d.model or DEFAULT_MODELS[d.provider],
            timeout_s=d.timeout_seconds,
        )
        if not client.is_available():
            logger.info("[decision] route %s has no API key — engine idle", d.provider)
            return None
        return client

    # -- the call -----------------------------------------------------------

    async def decide(
        self,
        *,
        state: Any,
        questions: Mapping[str, Question],
        workspace_id: Any,
        agent_id: Optional[int] = None,
        purpose: str = "decision",
    ) -> Optional[DecisionResult]:
        """A result, or None. Bounded by the backend's timeout plus a grace
        second; never raises."""
        d = self.dials()
        backend = self.backend(d)
        if backend is None:
            return None
        try:
            return await asyncio.wait_for(
                backend.decide(
                    state=state,
                    questions=questions,
                    workspace_id=workspace_id,
                    agent_id=agent_id,
                    purpose=purpose,
                ),
                timeout=float(getattr(backend, "timeout_s", d.timeout_seconds)) + 1.0,
            )
        except asyncio.TimeoutError:
            logger.warning("[decision] %s exceeded its budget on %s", d.provider, purpose)
            return None
        except Exception as exc:  # noqa: BLE001 — never into a turn
            logger.warning("[decision] %s raised on %s: %s", d.provider, purpose, exc, exc_info=True)
            return None

    # -- shadow log ---------------------------------------------------------

    def record_shadow(self, row: Mapping[str, Any]) -> None:
        """Append one JSON line; a write failure is logged at debug and dropped."""
        try:
            path = Path(config.DECISION_SHADOW_LOG_PATH)
            path.parent.mkdir(parents=True, exist_ok=True)
            line = json.dumps({"ts": round(time.time(), 3), **dict(row)}, ensure_ascii=False, default=str)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception:  # noqa: BLE001
            logger.debug("[decision] shadow row not written", exc_info=True)


_engine_lock = threading.Lock()
_engine: Optional[DecisionEngine] = None


def get_decision_engine() -> DecisionEngine:
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is None:
            _engine = DecisionEngine()
    return _engine
