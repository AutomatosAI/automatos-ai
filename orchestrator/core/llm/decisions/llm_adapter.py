"""PRD-248 — the same typed questions answered by the platform's own system LLM.

This is the baseline and the no-cloud-key path: the identical ``decide()``
contract served by whatever model the ``system_llm`` tier is configured with,
through a JSON prompt. Its probabilities are the model's stated numbers (no
logprobs), so treat its confidence as a proxy — the point of running it is to
have a like-for-like comparison in the same shadow log, and to let the local
edition exercise the seam without a TypeSafe or OpenRouter key.

Usage is booked by the LLM manager itself under request type
``decision_adapter`` (a ``system_llm`` service).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any, Callable, Dict, List, Mapping, Optional

from .questions import (
    TYPE_CHOICE,
    TYPE_NOUL,
    TYPE_SCORE,
    Choice,
    DecisionAnswer,
    DecisionResult,
    Noul,
    Question,
    Score,
    _clamp,
    to_wire,
)

logger = logging.getLogger(__name__)

PROVIDER_LLM = "llm"
SERVICE_NAME = "decision_adapter"
MAX_STATE_CHARS = 12_000
# The adapter is the slow baseline, never the fast path: it gets more rope
# than a Jev call so the comparison is between answers, not timeouts.
ADAPTER_MIN_TIMEOUT_S = 20.0

_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def build_adapter_prompt(state: Any, wire_questions: Mapping[str, Any]) -> str:
    state_text = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, default=str)
    return (
        "You answer typed questions about a state. Do not explain, do not add keys.\n\n"
        f"STATE:\n{state_text[:MAX_STATE_CHARS]}\n\n"
        f"QUESTIONS (JSON, keyed by id):\n{json.dumps(wire_questions, ensure_ascii=False)}\n\n"
        "Return ONLY a JSON object keyed by question id. Per question type:\n"
        '- choice: {"choice": "<one option key exactly as given>", '
        '"probabilities": {"<option>": p, ...}} with probabilities summing to 1\n'
        '- score: {"score": <0-based level index, may be fractional>, '
        '"probabilities": [p per level, low to high]} summing to 1\n'
        '- noul: {"noul": <probability that the answer is yes, 0..1>}\n'
        "Be calibrated: spread probability when you are unsure."
    )


def _nonneg(value: Any) -> float:
    """A stated weight: negatives and junk count as zero, anything above one
    is kept — the model may answer in ratios, and normalisation sorts it out."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if v != v:  # NaN
        return 0.0
    return max(0.0, v)


def _normalise(values: List[float]) -> List[float]:
    total = sum(v for v in values if v > 0)
    if total <= 0:
        return values
    return [max(0.0, v) / total for v in values]


def _answer_choice(q: Choice, raw: Mapping[str, Any]) -> Optional[DecisionAnswer]:
    options = q.options
    probs_raw = raw.get("probabilities")
    probs = [_nonneg((probs_raw or {}).get(o, 0.0)) if isinstance(probs_raw, Mapping) else 0.0 for o in options]
    choice = raw.get("choice")
    choice = str(choice) if choice is not None else None
    if choice not in options:
        choice = None
    if sum(probs) <= 0:
        if choice is None:
            return None
        probs = [1.0 if o == choice else 0.0 for o in options]
    probs = _normalise(probs)
    if choice is None:
        choice = options[max(range(len(options)), key=lambda i: probs[i])]
    return DecisionAnswer(
        type=TYPE_CHOICE,
        choice=choice,
        probabilities=dict(zip(options, probs)),
        confidence=max(probs),
    )


def _answer_score(q: Score, raw: Mapping[str, Any]) -> Optional[DecisionAnswer]:
    n = len(q.criteria)
    probs_raw = raw.get("probabilities")
    probs = [_nonneg(v) for v in probs_raw] if isinstance(probs_raw, (list, tuple)) else []
    if len(probs) != n or sum(probs) <= 0:
        score = raw.get("score")
        if score is None:
            return None
        try:
            idx = int(round(_clamp(score, 0.0, float(n - 1))))
        except (TypeError, ValueError):
            return None
        probs = [1.0 if i == idx else 0.0 for i in range(n)]
    probs = _normalise(probs)
    score_f = float(sum(i * p for i, p in enumerate(probs)))
    return DecisionAnswer(type=TYPE_SCORE, score=score_f, probabilities=probs, confidence=max(probs))


def _answer_noul(raw: Mapping[str, Any]) -> Optional[DecisionAnswer]:
    if raw.get("noul") is None:
        return None
    return DecisionAnswer(type=TYPE_NOUL, noul=_clamp(raw.get("noul")))


def answers_from_adapter(
    questions: Mapping[str, Question], data: Mapping[str, Any]
) -> Dict[str, DecisionAnswer]:
    """Coerce the model's JSON into the same answer shapes the API returns."""
    out: Dict[str, DecisionAnswer] = {}
    for qid, q in questions.items():
        raw = data.get(qid)
        if not isinstance(raw, Mapping):
            continue
        parsed: Optional[DecisionAnswer]
        if isinstance(q, Choice):
            parsed = _answer_choice(q, raw)
        elif isinstance(q, Score):
            parsed = _answer_score(q, raw)
        elif isinstance(q, Noul):
            parsed = _answer_noul(raw)
        else:
            parsed = None
        if parsed is not None:
            out[qid] = parsed
    return out


class LLMDecisionAdapter:
    """``decide()`` over the system LLM. ``manager_factory`` defaults to
    ``create_llm_manager`` and is injectable for tests."""

    provider = PROVIDER_LLM

    def __init__(
        self,
        timeout_s: float = ADAPTER_MIN_TIMEOUT_S,
        manager_factory: Optional[Callable[..., Any]] = None,
    ) -> None:
        self.timeout_s = max(float(timeout_s), ADAPTER_MIN_TIMEOUT_S)
        self.model = "system_llm"
        self._manager_factory = manager_factory

    def is_available(self) -> bool:
        return True

    async def decide(
        self,
        *,
        state: Any,
        questions: Mapping[str, Question],
        workspace_id: Any,
        agent_id: Optional[int] = None,
        purpose: str = "decision",
    ) -> Optional[DecisionResult]:
        prompt = build_adapter_prompt(state, to_wire(questions))
        started = time.monotonic()
        try:
            factory = self._manager_factory
            if factory is None:
                from core.llm.manager import create_llm_manager

                factory = create_llm_manager
            llm = factory(
                service_name=SERVICE_NAME,
                workspace_id=workspace_id,
                agent_id=agent_id,
                request_type=SERVICE_NAME,
            )
            response = await asyncio.wait_for(
                llm.generate_response(messages=[{"role": "user", "content": prompt}]),
                timeout=self.timeout_s,
            )
        except asyncio.TimeoutError:
            logger.warning("[decision] llm adapter timed out on %s", purpose)
            return None
        except Exception as exc:  # noqa: BLE001 — the seam must never break a turn
            logger.warning("[decision] llm adapter failed on %s: %s", purpose, exc)
            return None

        latency_ms = int((time.monotonic() - started) * 1000)
        content = response.content if hasattr(response, "content") else str(response)
        match = _JSON_BLOCK.search(content or "")
        if not match:
            logger.warning("[decision] llm adapter returned no JSON on %s", purpose)
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("[decision] llm adapter returned invalid JSON on %s", purpose)
            return None
        answers = answers_from_adapter(questions, data if isinstance(data, Mapping) else {})
        if not answers:
            return None
        usage = getattr(response, "usage", None) or {}
        return DecisionResult(
            answers=answers,
            provider=self.provider,
            model=str(getattr(response, "model", None) or self.model),
            latency_ms=latency_ms,
            input_tokens=int(usage.get("input_tokens") or usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or usage.get("completion_tokens") or 0),
        )
