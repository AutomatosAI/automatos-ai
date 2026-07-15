"""PRD-201 S2 — one tokenizer + token-aware truncation.

Pure. The live build path now counts and truncates through
``core.context_guard`` (tiktoken) — the char/4 ``TokenEstimator`` is deleted.
Includes the source-grep guard that no char/4 estimator caller survives on the
assembly path (repointed here in the same PR the symbol moved).
"""

import sys
from pathlib import Path

import pytest

_ORCH = Path(__file__).resolve().parent.parent.parent
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.context_guard import count_tokens, truncate_to_token_budget

# The live-build files S2 unifies. The grep guard is scoped to exactly these —
# the assembly path — not the whole package (field_scoring's PRD-166 length
# proxy and the F079 rival budgeters are out of S2's stated scope, §8-Q7).
_ASSEMBLY_FILES = [
    _ORCH / "modules" / "context" / "service.py",
    _ORCH / "modules" / "context" / "budget.py",
    _ORCH / "modules" / "context" / "sections" / "base.py",
    _ORCH / "modules" / "context" / "sections" / "conversation.py",
]


# --- delegation ---


def test_count_tokens_is_the_one_counter():
    assert count_tokens("") == 0
    assert count_tokens("hello world this is a token test") > 0
    # Same content → same number every call (one definition of size).
    text = "The quick brown fox jumps over the lazy dog. " * 5
    assert count_tokens(text) == count_tokens(text)


def test_base_section_estimate_delegates_to_count_tokens():
    from modules.context.sections.base import BaseSection

    class _Probe(BaseSection):
        name = "probe"
        priority = 5

        async def render(self, ctx):  # pragma: no cover - not called
            return ""

    text = '{"key": "value", "nested": {"a": 1, "b": [2, 3]}}' * 4
    assert _Probe().estimate_tokens(text) == count_tokens(text)


# --- token-aware truncation ---


def test_truncation_lands_on_token_boundary():
    # A JSON-ish payload longer than the budget truncates to a valid token
    # prefix — decodes cleanly (no mid-token cut) and is under the budget.
    payload = ('{"records": [' + ",".join('{"id": %d, "v": "x"}' % i for i in range(400)) + "]}")
    budget = 50
    out = truncate_to_token_budget(payload, budget, suffix="")
    assert out != payload
    assert len(out) < len(payload)
    # The retained prefix is at most the budget in tokens (boundary, not chars).
    assert count_tokens(out) <= budget + 1


def test_truncation_noop_when_under_budget():
    assert truncate_to_token_budget("short", 1000) == "short"
    assert truncate_to_token_budget("", 10) == ""
    assert truncate_to_token_budget("abc", 0) == "abc"


# --- source-grep guard: no char/4 estimator callers remain on the build path ---


def test_char4_estimator_deleted():
    assert not (_ORCH / "modules" / "context" / "estimator.py").exists()


@pytest.mark.parametrize("path", _ASSEMBLY_FILES, ids=lambda p: p.name)
def test_no_char4_estimator_callers(path):
    src = path.read_text(encoding="utf-8")
    assert "TokenEstimator" not in src, f"{path.name} still references TokenEstimator"
    assert "modules.context.estimator" not in src, f"{path.name} still imports the deleted estimator"
    assert "_estimator.estimate" not in src, f"{path.name} still calls the char/4 estimator"
    # No char-slice truncation ("[: ... * 4]" / "max_tokens * 4") on the build path.
    assert "* 4" not in src, f"{path.name} still uses a char/4 (* 4) slice"
