"""PRD-248 — the System One HTTP backend (TypeSafe Jev), direct or via OpenRouter.

One request shape for both routes: ``{"model", "state", "questions"}`` POSTed
with a bearer key, ``answers`` keyed by question id in the reply. Direct goes to
TypeSafe's ``/v1/systemone`` with a TypeSafe key; the OpenRouter route goes to
OpenRouter's decisions endpoint with the platform's existing OpenRouter key and
bills on that account.

Posture, the same as every other optional outbound seam here: one attempt, a
hard timeout, never raises — a failed decision returns ``None`` and the caller
carries on as if the engine were switched off. Every call books one
``llm_usage`` row (lane ``decision``) priced at the configured per-token rate,
output tokens free.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Dict, Mapping, Optional

import httpx

from config import config

from .questions import DecisionResult, Question, parse_answers, to_wire

logger = logging.getLogger(__name__)

PROVIDER_TYPESAFE = "typesafe"
PROVIDER_OPENROUTER = "openrouter"

# The model id each route understands. Pinned versions, not ``latest``: the
# dials and thresholds tuned against one version must not drift under a silent
# upgrade (TypeSafe's own advice).
DEFAULT_MODELS: Dict[str, str] = {
    PROVIDER_TYPESAFE: "jev-1.13.0",
    PROVIDER_OPENROUTER: "typesafe/jev-1.13",
}

STATUS_TIMEOUT = "timeout"


def resolve_api_key(provider: str) -> Optional[str]:
    """The bearer key for a route, from where the platform already keeps keys.

    OpenRouter: the operator workspace key store, then the credential store,
    then the env (the same order the embedding client uses). TypeSafe: the
    workspace key store under provider ``typesafe`` (if someone adds one), then
    ``TYPESAFE_API_KEY``. Never raises."""
    try:
        from core.llm.workspace_keys import get_platform_workspace_key

        key = get_platform_workspace_key(provider)
        if key:
            return key
    except Exception:  # noqa: BLE001 — a lookup failure is a miss, not an outage
        logger.debug("[decision] workspace key lookup failed for %s", provider, exc_info=True)

    if provider == PROVIDER_OPENROUTER:
        try:
            from core.llm.embedding_manager import get_credential_field

            key = get_credential_field(PROVIDER_OPENROUTER)
            if key:
                return key
        except Exception:  # noqa: BLE001
            logger.debug("[decision] credential store lookup failed", exc_info=True)
        return config.OPENROUTER_API_KEY or None

    if provider == PROVIDER_TYPESAFE:
        return config.TYPESAFE_API_KEY or None
    return None


def endpoint_for(provider: str) -> str:
    if provider == PROVIDER_OPENROUTER:
        return config.OPENROUTER_DECISIONS_URL
    return config.TYPESAFE_API_URL


class TypeSafeDecisionClient:
    """One route (``typesafe`` or ``openrouter``), one model, one timeout."""

    def __init__(
        self,
        provider: str,
        model: Optional[str] = None,
        timeout_s: float = 2.5,
        client_factory: Optional[Callable[[], httpx.AsyncClient]] = None,
    ) -> None:
        if provider not in DEFAULT_MODELS:
            raise ValueError(f"unknown decision route {provider!r}")
        self.provider = provider
        self.model = model or DEFAULT_MODELS[provider]
        self.timeout_s = float(timeout_s)
        self._client_factory = client_factory
        self._client: Optional[httpx.AsyncClient] = None
        self._client_loop: Optional[asyncio.AbstractEventLoop] = None
        self._api_key: Optional[str] = None
        self._key_resolved = False

    # -- availability -------------------------------------------------------

    def api_key(self) -> Optional[str]:
        if not self._key_resolved:
            self._api_key = resolve_api_key(self.provider)
            self._key_resolved = True
        return self._api_key

    def is_available(self) -> bool:
        return bool(self.api_key())

    # -- transport ----------------------------------------------------------

    def _get_client(self) -> httpx.AsyncClient:
        # Keep the connection pool per running loop (the rerank manager's
        # lesson): a warm connection saves the TLS handshake — measured at
        # ~360 ms to TypeSafe from Europe — on every decision after the first.
        loop = asyncio.get_running_loop()
        if self._client is None or self._client.is_closed or self._client_loop is not loop:
            if self._client_factory is not None:
                self._client = self._client_factory()
            else:
                self._client = httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_s))
            self._client_loop = loop
        return self._client

    def _headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self.api_key()}",
            "Content-Type": "application/json",
        }
        if self.provider == PROVIDER_OPENROUTER:
            headers["X-Title"] = "Automatos"
        return headers

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
        """One attempt, bounded by ``timeout_s``; ``None`` on any failure."""
        if not self.is_available():
            logger.info("[decision] no %s key — %s skipped", self.provider, purpose)
            return None

        body = {"model": self.model, "state": state, "questions": to_wire(questions)}
        started = time.monotonic()
        try:
            client = self._get_client()
            response = await client.post(
                endpoint_for(self.provider),
                headers=self._headers(),
                json=body,
                timeout=httpx.Timeout(self.timeout_s),
            )
            latency_ms = int((time.monotonic() - started) * 1000)
            if response.status_code >= 400:
                detail = response.text[:200]
                logger.warning(
                    "[decision] %s HTTP %s on %s: %s",
                    self.provider, response.status_code, purpose, detail,
                )
                self._record_usage(None, latency_ms, workspace_id, agent_id, error=f"HTTP {response.status_code}")
                return None
            data = response.json()
        except httpx.TimeoutException:
            latency_ms = int((time.monotonic() - started) * 1000)
            logger.warning("[decision] %s timed out after %sms on %s", self.provider, latency_ms, purpose)
            self._record_usage(None, latency_ms, workspace_id, agent_id, error=STATUS_TIMEOUT)
            return None
        except Exception as exc:  # noqa: BLE001 — the seam must never break a turn
            latency_ms = int((time.monotonic() - started) * 1000)
            logger.warning("[decision] %s failed on %s: %s", self.provider, purpose, exc)
            self._record_usage(None, latency_ms, workspace_id, agent_id, error=str(exc)[:200])
            return None

        answers = parse_answers(data.get("answers") if isinstance(data, dict) else None)
        usage = (data.get("usage") if isinstance(data, dict) else None) or {}
        result = DecisionResult(
            answers=answers,
            provider=self.provider,
            model=str(data.get("model") or self.model) if isinstance(data, dict) else self.model,
            latency_ms=latency_ms,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
        )
        self._record_usage(result, latency_ms, workspace_id, agent_id)
        if not answers:
            logger.warning("[decision] %s returned no parseable answers on %s", self.provider, purpose)
            return None
        return result

    # -- receipt ------------------------------------------------------------

    def _record_usage(
        self,
        result: Optional[DecisionResult],
        latency_ms: int,
        workspace_id: Any,
        agent_id: Optional[int],
        error: Optional[str] = None,
    ) -> None:
        """One ``llm_usage`` row per call. Input tokens at the configured rate,
        output free — the seam's whole economic point, booked honestly."""
        try:
            from core.llm.usage_context import LANE_DECISION, current_usage_scope
            from core.llm.usage_tracker import STATUS_ERROR, STATUS_SUCCESS, UsageTracker

            # The attribution in force for this task: a simulation campaign
            # (PRD-247 encodes ``sim:<campaign>:<scenario>:<run>`` in
            # execution_id), a watch, a board task — so a decision's cost sits
            # on the same row family as the turn it served.
            scope = current_usage_scope()
            input_tokens = result.input_tokens if result else 0
            output_tokens = result.output_tokens if result else 0
            usd_in = input_tokens / 1_000_000.0 * float(config.DECISION_ENGINE_USD_PER_MTOK_IN)
            UsageTracker.track(
                workspace_id=workspace_id if workspace_id is not None else scope.get("workspace_id"),
                model_id=result.model if result else self.model,
                provider=self.provider,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                agent_id=agent_id if agent_id is not None else scope.get("agent_id"),
                execution_id=scope.get("execution_id"),
                request_type=LANE_DECISION,
                latency_ms=latency_ms,
                status=STATUS_ERROR if error else STATUS_SUCCESS,
                error_message=error,
                tier="direct",
                cost_override=(usd_in, 0.0),
            )
        except Exception as exc:  # noqa: BLE001 — never fail a decision over its receipt
            logger.debug("[decision] usage not recorded: %s", exc)
