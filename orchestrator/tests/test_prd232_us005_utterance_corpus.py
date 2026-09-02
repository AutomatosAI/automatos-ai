"""
PRD-232 US-005 — the synthetic utterance corpus (coverage linter).
==================================================================

The corpus (``orchestrator/core/seeds/utterances/<category>.yaml``) is
hand-authored seed data (decision §6.3): 15-25 diverse utterances per registered
non-super_admin_only action, folding in every ``ActionDefinition.examples``
string and every ``_PLATFORM_KEYWORDS`` phrase (provenance-tagged). These tests
ARE the linter — they call ``scripts.generate_utterance_corpus.validate`` (which
loads the live registry lightweight — no transformers/torch — and AST-parses the
phrase map) and assert the coverage contract of US-005.

The generator authors nothing and calls no LLM/network; it is the format
contract + coverage checker. See its module docstring.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

_ORCH_ROOT = Path(__file__).resolve().parent.parent
if str(_ORCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_ORCH_ROOT))

import scripts.generate_utterance_corpus as gen  # noqa: E402


@pytest.fixture(scope="module")
def report():
    """One validation pass, shared across the assertions (registry + phrase map
    are loaded once)."""
    return gen.validate()


# ── AC4 / overall: the corpus validates clean ────────────────────────────────
def test_corpus_validates_clean(report):
    assert report.ok, (
        "utterance corpus failed validation — first errors:\n  "
        + "\n  ".join(report.errors[:20])
    )


# ── AC1: >= 90% of non-su actions carry >= 15 utterances ─────────────────────
def test_coverage_at_least_90_percent(report):
    assert report.n_nonsu_actions >= 150, (
        f"expected the full non-su surface (~183), got {report.n_nonsu_actions}"
    )
    assert report.coverage_pct >= gen.COVERAGE_FLOOR_PCT, (
        f"only {report.coverage_pct:.1f}% of {report.n_nonsu_actions} non-su actions "
        f"have >= {gen.MIN_UTTERANCES} utterances (floor {gen.COVERAGE_FLOOR_PCT:.0f}%)"
    )


# ── AC1: the VECTOR vocabulary is present on the board-write action ───────────
def test_update_task_status_has_close_ticket_blocked():
    corpus = {name: utts for cf in gen.load_corpus() for name, utts in cf.actions.items()}
    assert "platform_update_task_status" in corpus, "board-write action missing from corpus"
    blob = " ".join(u.text.lower() for u in corpus["platform_update_task_status"])
    for token in ("close", "ticket", "blocked"):
        assert token in blob, (
            f"platform_update_task_status utterances must include {token!r} "
            f"(the 2026-08-28 VECTOR failure vocabulary)"
        )
    # the canonical VECTOR sentence itself is seeded
    assert "close all the blocked tickets from vector" in blob


# ── AC2: every ActionDefinition.examples string appears (provenance-tagged) ───
def test_every_registry_example_present(report):
    missing = {
        name: r.missing_examples
        for name, r in report.per_action.items()
        if r.missing_examples
    }
    assert not missing, f"registry examples absent from corpus: {missing}"


# ── AC2: every mappable _PLATFORM_KEYWORDS phrase appears (provenance-tagged) ─
def test_every_phrase_map_phrase_present(report):
    missing = {
        name: r.missing_phrases
        for name, r in report.per_action.items()
        if r.missing_phrases
    }
    assert not missing, f"phrase-map phrases absent from corpus: {missing}"


def test_phrase_map_skips_are_only_su_or_unregistered(report):
    """Every phrase-map key we did NOT require is skipped for a legitimate
    reason — su-only (excluded) or an unregistered legacy name with no corpus
    home. No key is silently dropped."""
    assert report.skipped_phrase_keys, "expected some su/unregistered phrase keys"
    for key, reason in report.skipped_phrase_keys.items():
        assert ("su-only" in reason) or ("unregistered" in reason), (
            f"phrase-map key {key!r} skipped for an unexpected reason: {reason}"
        )


# ── AC3: su-only actions are excluded (fail-closed, from the live flag) ───────
def test_super_admin_only_actions_excluded(report):
    assert report.su_present == [], (
        f"super_admin_only actions leaked into the corpus: {report.su_present}"
    )
    # spot-check known su-only actions are genuinely absent from the corpus files
    corpus_names = {name for cf in gen.load_corpus() for name in cf.actions}
    su_names = {a.name for a in gen.load_registry_actions() if a.super_admin_only}
    assert su_names, "registry reports no super_admin_only actions — flag read broken?"
    assert su_names.isdisjoint(corpus_names), (
        f"su actions present: {su_names & corpus_names}"
    )


def test_no_unknown_or_stale_action_names(report):
    assert report.unknown_actions == [], (
        f"corpus references non-registered actions: {report.unknown_actions}"
    )


# ── AC3: each file documents its schema + provenance at the top ──────────────
def test_each_file_documents_schema():
    files = sorted(gen.CORPUS_DIR.glob("*.yaml"))
    assert files, f"no corpus YAML files under {gen.CORPUS_DIR}"
    for path in files:
        text = path.read_text()
        head = text[:1600].lower()
        assert "schema:" in head, f"{path.name}: schema block not documented at top"
        assert "source" in head and "provenance" in head, (
            f"{path.name}: provenance of source tags not documented"
        )
        assert f"category: {path.stem}" in text, f"{path.name}: category != filename stem"
        assert "version: 1" in text, f"{path.name}: missing schema version"


# ── provenance integrity: example/phrase_map tags are truthful ───────────────
def test_no_mistagged_provenance(report):
    mistagged = {
        name: r.mistagged for name, r in report.per_action.items() if r.mistagged
    }
    assert not mistagged, f"utterances carry untruthful source tags: {mistagged}"


# ── design guard: the linter loads the registry WITHOUT the torch chain ──────
def test_registry_load_is_lightweight():
    """``load_registry_actions`` leaf-loads the discovery package so
    ``modules/tools/__init__`` (execution -> agents -> llm ->
    sentence_transformers -> torch) is never imported. Proven in a clean
    subprocess so a sibling test that already imported torch can't mask it."""
    code = (
        "import sys;"
        "import scripts.generate_utterance_corpus as g;"
        "acts=g.load_registry_actions();"
        "nonsu=[a for a in acts if not a.super_admin_only];"
        "assert len(nonsu)>=150, len(nonsu);"
        "assert 'torch' not in sys.modules, 'torch leaked';"
        "assert 'sentence_transformers' not in sys.modules, 'sentence_transformers leaked';"
        "print('OK', len(nonsu))"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_ORCH_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, (
        f"lightweight-load guard failed:\nSTDOUT: {proc.stdout}\nSTDERR: {proc.stderr[-2000:]}"
    )
    assert "OK" in proc.stdout
