"""PRD-248 S4 — rerank a candidate list with typed decisions (pure).

The retriever proposes; the decision engine judges. Each candidate gets one
yes/no question — would this action help with the request? — answered in a
single call with a probability per candidate. The cut keeps the candidates
above a probability floor, ordered by probability, tops up to a minimum so an
unsure turn never strips the surface, and caps at the caller's top-K. When no
candidate clears the floor the cut carries a "nothing fits" signal.

Shared by the production tool router (``modules.tools.discovery.decision_rerank``)
and the offline eval harness (``scripts.eval.tool_routing``). No I/O here.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from .questions import DecisionResult, Noul, Question

PURPOSE = "tool_rerank"
QUERY_MAX_CHARS = 2000
DESCRIPTION_MAX_CHARS = 160
DEFAULT_CANDIDATES = 30
DEFAULT_MIN_PROBABILITY = 0.5
DEFAULT_MIN_KEEP = 5
# Fewer answered candidates than this share of the asked ones = a miss.
MIN_ANSWERED_SHARE = 0.5

Candidate = Tuple[str, str]  # (name, description)
Decide = Callable[..., Awaitable[Optional[DecisionResult]]]


def build_state(query: str) -> Dict[str, Any]:
    return {"request": (query or "")[:QUERY_MAX_CHARS]}


def build_questions(candidates: Sequence[Candidate]) -> Dict[str, Question]:
    """One Noul per candidate, keyed by name, the description folded into the
    proposition. Terse on purpose: wordier instructions measured worse
    calibration in the early-access tests."""
    questions: Dict[str, Question] = {}
    for name, description in candidates:
        if not name:
            continue
        desc = (description or "").strip()[:DESCRIPTION_MAX_CHARS]
        text = f"Calling the action `{name}`"
        if desc:
            text += f" ({desc})"
        questions[name] = Noul(text + " would help with the request.")
    if not questions:
        raise ValueError("a rerank needs at least one candidate")
    return questions


@dataclass
class RerankCut:
    kept: List[str]
    probabilities: Dict[str, float]
    nothing_fits: bool
    answered: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kept": list(self.kept),
            "probabilities": {k: round(v, 4) for k, v in self.probabilities.items()},
            "nothing_fits": self.nothing_fits,
            "answered": self.answered,
        }


def apply_rerank(
    result: DecisionResult,
    candidates: Sequence[str],
    *,
    top_k: int,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    min_keep: int = DEFAULT_MIN_KEEP,
) -> Optional[RerankCut]:
    """The cut. None when the engine answered fewer than half the candidates —
    a partial answer is a miss, never a narrower surface."""
    names = [n for n in candidates if n]
    if not names:
        return None
    probabilities: Dict[str, float] = {}
    for name in names:
        answer = result.get(name)
        if answer is not None and answer.noul is not None:
            probabilities[name] = float(answer.noul)
    if len(probabilities) < max(1, int(len(names) * MIN_ANSWERED_SHARE)):
        return None

    order = {n: i for i, n in enumerate(names)}
    ordered = sorted(probabilities, key=lambda n: (-probabilities[n], order[n]))
    kept = [n for n in ordered if probabilities[n] >= float(min_probability)]
    nothing_fits = not kept
    if len(kept) < int(min_keep):
        for name in ordered:
            if len(kept) >= int(min_keep):
                break
            if name not in kept:
                kept.append(name)
    if top_k and int(top_k) > 0:
        kept = kept[: int(top_k)]
    return RerankCut(
        kept=kept, probabilities=probabilities, nothing_fits=nothing_fits, answered=len(probabilities)
    )


def compare_surfaces(embedding_cut: Sequence[str], reranked: Sequence[str]) -> Dict[str, Any]:
    e = [n for n in (embedding_cut or []) if n]
    r = [n for n in (reranked or []) if n]
    es, rs = set(e), set(r)
    return {
        "embedding_size": len(e),
        "rerank_size": len(r),
        "overlap": len(es & rs),
        "dropped": [n for n in e if n not in rs],
        "added": [n for n in r if n not in es],
        "same_top": bool(e and r and e[0] == r[0]),
    }


def query_digest(query: str) -> str:
    return hashlib.sha256((query or "").lower().strip().encode()).hexdigest()[:16]


async def rerank_candidates(
    *,
    query: str,
    candidates: Sequence[Candidate],
    decide: Decide,
    top_k: int,
    min_probability: float = DEFAULT_MIN_PROBABILITY,
    min_keep: int = DEFAULT_MIN_KEEP,
    workspace_id: Any = None,
) -> Tuple[Optional[RerankCut], Optional[DecisionResult]]:
    """One engine call over the candidates, then the cut. ``(None, None)`` on
    a miss; ``(None, result)`` when the engine answered too little."""
    if not candidates:
        return None, None
    result = await decide(
        state=build_state(query),
        questions=build_questions(candidates),
        workspace_id=workspace_id,
        purpose=PURPOSE,
    )
    if result is None:
        return None, None
    cut = apply_rerank(
        result,
        [name for name, _ in candidates],
        top_k=top_k,
        min_probability=min_probability,
        min_keep=min_keep,
    )
    return cut, result
