"""PRD-248 — typed questions and answers for the decision seam (pure).

Three primitives, the exact shapes TypeSafe's System One API speaks, so a
question built here is sent verbatim to Jev and the LLM-backed adapter answers
the identical contract:

* ``Choice`` — pick one option from a described set (2..255 options); answered
  with the pick, a probability per option and a confidence.
* ``Score``  — a position on 2..10 ordered, described levels; answered with a
  probability-weighted score, a probability per level and a confidence.
* ``Noul``   — a yes/no proposition; answered with P(yes).

No I/O and no config in this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

CHOICE_MIN_OPTIONS = 2
CHOICE_MAX_OPTIONS = 255
SCORE_MIN_LEVELS = 2
SCORE_MAX_LEVELS = 10

TYPE_CHOICE = "choice"
TYPE_SCORE = "score"
TYPE_NOUL = "noul"


def _clamp(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return lo
    if v != v:  # NaN
        return lo
    return max(lo, min(hi, v))


@dataclass(frozen=True)
class Choice:
    """One option out of a described set. ``criteria`` maps option -> description
    (``None`` sends an option with no description)."""

    instructions: str
    criteria: Mapping[str, Optional[str]]

    def __post_init__(self) -> None:
        n = len(self.criteria)
        if not CHOICE_MIN_OPTIONS <= n <= CHOICE_MAX_OPTIONS:
            raise ValueError(
                f"Choice needs {CHOICE_MIN_OPTIONS}..{CHOICE_MAX_OPTIONS} options, got {n}"
            )

    @property
    def options(self) -> List[str]:
        return list(self.criteria.keys())

    def to_wire(self) -> Dict[str, Any]:
        return {
            "type": TYPE_CHOICE,
            "instructions": self.instructions,
            "criteria": {k: (v if v else None) for k, v in self.criteria.items()},
        }


@dataclass(frozen=True)
class Score:
    """A position on ordered levels, described low to high."""

    instructions: str
    criteria: Sequence[str]

    def __post_init__(self) -> None:
        n = len(self.criteria)
        if not SCORE_MIN_LEVELS <= n <= SCORE_MAX_LEVELS:
            raise ValueError(
                f"Score needs {SCORE_MIN_LEVELS}..{SCORE_MAX_LEVELS} levels, got {n}"
            )

    def to_wire(self) -> Dict[str, Any]:
        return {
            "type": TYPE_SCORE,
            "instructions": self.instructions,
            "criteria": list(self.criteria),
        }


@dataclass(frozen=True)
class Noul:
    """A yes/no proposition. Optional ``criteria`` describe what true and false
    look like (both keys or neither)."""

    instructions: str
    criteria: Optional[Mapping[str, str]] = None

    def to_wire(self) -> Dict[str, Any]:
        wire: Dict[str, Any] = {"type": TYPE_NOUL, "instructions": self.instructions}
        if self.criteria:
            wire["criteria"] = {
                "true": str(self.criteria.get("true", "")),
                "false": str(self.criteria.get("false", "")),
            }
        return wire


Question = Union[Choice, Score, Noul]


def to_wire(questions: Mapping[str, Question]) -> Dict[str, Dict[str, Any]]:
    """The ``questions`` object of a System One request, keyed by question id."""
    if not questions:
        raise ValueError("a decision needs at least one question")
    return {qid: q.to_wire() for qid, q in questions.items()}


# ---------------------------------------------------------------------------
# Answers
# ---------------------------------------------------------------------------


@dataclass
class DecisionAnswer:
    """One answer. ``probabilities`` is a dict for a Choice (option -> p) and a
    list for a Score (one p per level); a Noul carries only ``noul`` = P(yes)."""

    type: str
    choice: Optional[str] = None
    score: Optional[float] = None
    noul: Optional[float] = None
    probabilities: Optional[Any] = None
    confidence: Optional[float] = None

    @property
    def certainty(self) -> float:
        """One 0..1 number to threshold on whatever the type: the reported
        confidence for Choice/Score (falling back to the winning probability),
        and for a Noul how far from even it sits — 0.95 yes and 0.05 no are
        equally certain."""
        if self.type == TYPE_NOUL:
            return abs(_clamp(self.noul) - 0.5) * 2.0
        if self.confidence is not None:
            return _clamp(self.confidence)
        if isinstance(self.probabilities, dict) and self.probabilities:
            return _clamp(max(_clamp(v) for v in self.probabilities.values()))
        if isinstance(self.probabilities, (list, tuple)) and self.probabilities:
            return _clamp(max(_clamp(v) for v in self.probabilities))
        return 0.0

    @property
    def yes(self) -> Optional[bool]:
        """A Noul's verdict at the even split; None for other types."""
        if self.type != TYPE_NOUL or self.noul is None:
            return None
        return _clamp(self.noul) >= 0.5

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"type": self.type}
        if self.choice is not None:
            out["choice"] = self.choice
        if self.score is not None:
            out["score"] = round(float(self.score), 4)
        if self.noul is not None:
            out["noul"] = round(_clamp(self.noul), 4)
        if isinstance(self.probabilities, dict):
            out["probabilities"] = {k: round(_clamp(v), 4) for k, v in self.probabilities.items()}
        elif isinstance(self.probabilities, (list, tuple)):
            out["probabilities"] = [round(_clamp(v), 4) for v in self.probabilities]
        if self.confidence is not None:
            out["confidence"] = round(_clamp(self.confidence), 4)
        return out


def parse_answer(raw: Any) -> Optional[DecisionAnswer]:
    """One answer object as the API returns it. Tolerant: the ``type`` may be
    absent (inferred from the value key) and numbers may arrive as strings.
    Returns None when nothing usable is in it."""
    if not isinstance(raw, Mapping):
        return None
    kind = str(raw.get("type") or "").lower()
    if not kind:
        if "choice" in raw:
            kind = TYPE_CHOICE
        elif "score" in raw:
            kind = TYPE_SCORE
        elif "noul" in raw:
            kind = TYPE_NOUL
        else:
            return None

    if kind == TYPE_NOUL:
        if raw.get("noul") is None:
            return None
        return DecisionAnswer(type=TYPE_NOUL, noul=_clamp(raw.get("noul")))

    probs = raw.get("probabilities")
    conf = raw.get("confidence")
    confidence = _clamp(conf) if conf is not None else None

    if kind == TYPE_CHOICE:
        choice = raw.get("choice")
        probabilities = (
            {str(k): _clamp(v) for k, v in probs.items()} if isinstance(probs, Mapping) else None
        )
        if choice is None and probabilities:
            choice = max(probabilities.items(), key=lambda kv: kv[1])[0]
        if choice is None:
            return None
        return DecisionAnswer(
            type=TYPE_CHOICE, choice=str(choice), probabilities=probabilities, confidence=confidence
        )

    if kind == TYPE_SCORE:
        score = raw.get("score")
        probabilities = (
            [_clamp(v) for v in probs] if isinstance(probs, (list, tuple)) else None
        )
        if score is None and probabilities:
            score = float(sum(i * p for i, p in enumerate(probabilities)))
        if score is None:
            return None
        try:
            score_f = float(score)
        except (TypeError, ValueError):
            return None
        return DecisionAnswer(
            type=TYPE_SCORE, score=score_f, probabilities=probabilities, confidence=confidence
        )
    return None


def parse_answers(raw: Any) -> Dict[str, DecisionAnswer]:
    """The ``answers`` map of a response; unparseable entries are dropped."""
    out: Dict[str, DecisionAnswer] = {}
    if not isinstance(raw, Mapping):
        return out
    for qid, value in raw.items():
        parsed = parse_answer(value)
        if parsed is not None:
            out[str(qid)] = parsed
    return out


@dataclass
class DecisionResult:
    """What a backend returns for one request: the answers plus the receipt."""

    answers: Dict[str, DecisionAnswer]
    provider: str
    model: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0

    def get(self, qid: str) -> Optional[DecisionAnswer]:
        return self.answers.get(qid)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "answers": {k: v.to_dict() for k, v in self.answers.items()},
        }
