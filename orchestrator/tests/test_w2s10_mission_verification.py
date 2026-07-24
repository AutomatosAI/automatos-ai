"""PRD-142 Wave 2 · WS-I / W2-S10 — mission-verification tests.

The verification stack decides ``pass | fail | partial`` for *every* task a
mission produces. It was wholly untested. The two layers below the LLM judge
are fully deterministic, so their verdicts can be pinned exactly with no spend
and no network:

* ``DeterministicChecker`` — eight pure check handlers (length, regex, JSON
  schema, keywords, sections, URLs, word-count) plus the ``must_pass``
  short-circuit. A ``must_pass`` failure aborts before the LLM judge is ever
  reached; ``required_sections`` is special-cased to *never* short-circuit.
* ``VerificationService`` — the empty-output guard, the deterministic
  short-circuit, and the per-run output-hash cache (PRD-82B US-007) that skips
  a second LLM call when a retry produces byte-identical output.
* ``_select_verifier_model`` — the cross-model guarantee: the verifier must be
  a *different* model family than the executor, so a model never grades itself.

The LLM judge itself (Stage 2) is stubbed everywhere here — these tests assert
the deterministic gating *around* it, and prove the judge is not called when
the deterministic layer already has a verdict (the cost guard).
"""

from __future__ import annotations

import os
import sys
import types
from uuid import uuid4

# config import pulls Postgres env; seed harmless defaults. Nothing here
# touches a real DB, vector store, or network.
for _k in ("POSTGRES_USER", "POSTGRES_PASSWORD", "POSTGRES_DB"):
    os.environ.setdefault(_k, "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ.setdefault("POSTGRES_PORT", "5432")

# Defensive: some coordination imports can reach the RAG package, which pulls
# camelot (an optional PDF dep absent from the test env). Stub it.
sys.modules.setdefault("camelot", types.ModuleType("camelot"))

from unittest.mock import AsyncMock  # noqa: E402

import pytest  # noqa: E402

from modules.coordination import verification as verif  # noqa: E402
from modules.coordination.deterministic_checks import (  # noqa: E402
    DeterministicChecker,
)
from modules.coordination.verification import (  # noqa: E402
    VERDICT_FAIL,
    VERDICT_PASS,
    VerificationResult,
    VerificationService,
    _select_verifier_model,
)


# ================================================ DeterministicChecker
# Eight pure handlers — no LLM, no DB. Pin pass + fail for each.


def _check(output, criteria):
    return DeterministicChecker().check(output, criteria)


def test_no_criteria_passes_trivially():
    out = _check("anything at all", None)
    assert out.passed is True
    assert out.failures == []
    assert out.short_circuited is False


def test_min_and_max_length():
    assert _check("short", [{"type": "min_length", "value": 100}]).passed is False
    assert _check("x" * 200, [{"type": "min_length", "value": 100}]).passed is True
    assert _check("x" * 200, [{"type": "max_length", "value": 100}]).passed is False
    assert _check("ok", [{"type": "max_length", "value": 100}]).passed is True


def test_format_regex():
    crit = [{"type": "format_regex", "value": r"^\d{4}-\d{2}-\d{2}$"}]
    assert _check("2026-06-05", crit).passed is True
    assert _check("not a date", crit).passed is False


def test_contains_keywords_is_case_insensitive():
    crit = [{"type": "contains_keywords", "value": ["Revenue", "GROWTH"]}]
    assert _check("Q4 revenue and growth were strong", crit).passed is True
    out = _check("only revenue here", crit)
    assert out.passed is False
    assert "growth" in out.failures[0].description.lower()


def test_word_count_range():
    crit = [{"type": "word_count_range", "value": [3, 5]}]
    assert _check("one two three four", crit).passed is True
    assert _check("one two", crit).passed is False
    assert _check("one two three four five six", crit).passed is False


def test_json_schema_validates_structure_and_required_props():
    crit = [
        {
            "type": "json_schema",
            "value": {
                "type": "object",
                "required": ["name", "count"],
                "properties": {"name": {"type": "string"}, "count": {"type": "integer"}},
            },
        }
    ]
    assert _check('{"name": "x", "count": 3}', crit).passed is True
    assert _check("not json at all", crit).passed is False
    assert _check('{"name": "x"}', crit).passed is False  # missing 'count'
    assert _check('{"name": "x", "count": "3"}', crit).passed is False  # wrong type


def test_url_valid_passes_when_no_urls_present():
    # No URLs in the text → nothing to validate → pass.
    assert _check("plain prose, no links", [{"type": "url_valid", "value": None}]).passed is True
    assert _check(
        "see https://example.com/path", [{"type": "url_valid", "value": None}]
    ).passed is True


def test_unknown_check_type_is_skipped_not_failed():
    # An unknown handler is logged and skipped, not counted as a failure.
    out = _check("whatever", [{"type": "no_such_check", "value": 1}])
    assert out.passed is True
    assert out.failures == []


# ---- short-circuit semantics (the cost-relevant invariants) ----


def test_must_pass_failure_short_circuits():
    out = _check(
        "tiny",
        [
            {"type": "min_length", "value": 100, "must_pass": True},
            {"type": "contains_keywords", "value": ["unreached"]},
        ],
    )
    assert out.passed is False
    assert out.short_circuited is True
    # Stops at the first must_pass failure — the second check never runs.
    assert len(out.failures) == 1
    assert out.failures[0].check_type == "min_length"


def test_non_must_pass_failures_accumulate_without_short_circuit():
    out = _check(
        "tiny",
        [
            {"type": "min_length", "value": 100},
            {"type": "contains_keywords", "value": ["missing"]},
        ],
    )
    assert out.passed is False
    assert out.short_circuited is False
    assert len(out.failures) == 2


def test_required_sections_never_short_circuits_even_if_must_pass():
    """An LLM planner can't predict the exact headings an agent will emit, so
    ``required_sections`` is force-demoted out of must_pass. A miss is recorded
    as a soft failure, never a short-circuit."""
    out = _check(
        "# Intro\n\nbody only, no summary",
        [{"type": "required_sections", "value": ["Summary"], "must_pass": True}],
    )
    assert out.short_circuited is False
    assert out.passed is False
    assert out.failures[0].must_pass is False


# ================================================== VerificationService
# Deterministic gating around the (stubbed) LLM judge.


@pytest.mark.asyncio
async def test_empty_output_fails_without_touching_the_judge():
    svc = VerificationService()
    svc._run_llm_judge = AsyncMock(side_effect=AssertionError("judge must not run"))
    out = await svc.verify_task("t", "desc", "   ", None)
    assert out.verdict == VERDICT_FAIL
    assert out.deterministic_passed is False
    svc._run_llm_judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_must_pass_failure_fails_closed_without_the_judge():
    """Deterministic short-circuit → FAIL with no LLM spend (cost guard)."""
    svc = VerificationService()
    svc._run_llm_judge = AsyncMock(side_effect=AssertionError("judge must not run"))
    out = await svc.verify_task(
        "Report",
        "Write a long report",
        "too short",
        [{"type": "min_length", "value": 5000, "must_pass": True}],
    )
    assert out.verdict == VERDICT_FAIL
    assert out.deterministic_passed is False
    assert out.deterministic_failures  # populated
    svc._run_llm_judge.assert_not_awaited()


@pytest.mark.asyncio
async def test_output_hash_cache_skips_second_judge_call():
    """PRD-82B US-007: identical output on retry returns the cached verdict
    without a second LLM call. Cache is class-level and keyed by
    (run_id, task_id, sha256(output))."""
    run_id, task_id = uuid4(), uuid4()
    VerificationService.clear_cache(run_id)  # isolate from any prior run
    svc = VerificationService()
    judged = VerificationResult(verdict=VERDICT_PASS, reasoning="judged once")
    svc._run_llm_judge = AsyncMock(return_value=judged)

    try:
        first = await svc.verify_task(
            "t", "d", "identical output", None, run_id=run_id, task_id=task_id
        )
        second = await svc.verify_task(
            "t", "d", "identical output", None, run_id=run_id, task_id=task_id
        )

        assert first is judged
        assert second is judged
        svc._run_llm_judge.assert_awaited_once()  # the retry hit the cache
    finally:
        # Class-level cache: always purge, even if an assertion above fails, so
        # this run's entries can never bleed into another test.
        VerificationService.clear_cache(run_id)


def test_output_hash_is_deterministic_and_input_sensitive():
    h = VerificationService._output_hash
    assert h("same") == h("same")
    assert h("a") != h("b")
    assert len(h("x")) == 64  # sha256 hexdigest


def test_clear_cache_is_scoped_to_one_run():
    run_a, run_b = uuid4(), uuid4()
    task = uuid4()
    res = VerificationResult(verdict=VERDICT_PASS)
    VerificationService._cache[(run_a, task, "h1")] = res
    VerificationService._cache[(run_a, task, "h2")] = res
    VerificationService._cache[(run_b, task, "h3")] = res

    try:
        removed = VerificationService.clear_cache(run_a)

        assert removed == 2
        assert not any(k[0] == run_a for k in VerificationService._cache)
        assert (run_b, task, "h3") in VerificationService._cache
    finally:
        # Leave the class-level cache exactly as we found it, pass or fail.
        VerificationService.clear_cache(run_a)
        VerificationService.clear_cache(run_b)


# ============================================== cross-model verifier guarantee


def test_verifier_is_always_a_different_family_than_executor(monkeypatch):
    """A model must never grade its own family. Given a mapping, an Anthropic
    executor is verified by a non-Anthropic model and vice-versa."""
    monkeypatch.setattr(
        verif,
        "_parse_verifier_model_mapping",
        lambda: {"anthropic": "gpt-4o-mini", "openai": "claude-3-5-sonnet"},
    )

    for_claude = _select_verifier_model("claude-opus-4-1")
    for_gpt = _select_verifier_model("gpt-4o")

    assert "claude" not in for_claude.lower()  # Anthropic executor → non-Anthropic verifier
    assert "gpt" not in for_gpt.lower()        # OpenAI executor → non-OpenAI verifier


def test_verifier_falls_back_when_executor_family_unknown(monkeypatch):
    monkeypatch.setattr(verif, "_parse_verifier_model_mapping", lambda: {})
    monkeypatch.setattr(
        verif.Config, "COORDINATOR_VERIFIER_FALLBACK_MODEL", "fallback-model", raising=False
    )
    # No executor and no mapping → the configured fallback.
    assert _select_verifier_model(None) == "fallback-model"
