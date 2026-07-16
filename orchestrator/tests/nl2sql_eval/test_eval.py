"""PRD-160 S5 — NL2SQL eval entry point.

  pytest tests/nl2sql_eval -q

Always-on (no LLM, no external DB):
  * every golden SQL executes on the seeded DB (catches a broken golden);
  * every golden SQL passes the S2 AST validator against the seeded schema.

LLM-gated (set RUN_NL2SQL_EVAL=1 with a configured embedding/LLM provider):
  * run the real generator over the seeded questions, compute execution
    accuracy, print it, and assert no regression vs baseline.json.
"""
from __future__ import annotations

import json
import os

import pytest

from tests.nl2sql_eval import harness


def test_goldens_execute_on_seed():
    """Identity eval: each golden runs cleanly on the seed → accuracy 1.0."""
    report = harness.check_goldens()
    broken = [r.id for r in report.results if not r.correct]
    assert report.accuracy == 1.0, f"golden SQL failed to execute on seed: {broken}"


def _load_validator():
    """Load validator.py standalone (it imports only sqlglot + stdlib) so the
    eval stays self-contained — no heavy modules.nl2sql package init, no DB."""
    import importlib.util
    import pathlib

    vpath = pathlib.Path(__file__).resolve().parents[2] / "modules" / "nl2sql" / "query" / "validator.py"
    spec = importlib.util.spec_from_file_location("nl2sql_validator_standalone", vpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.SQLValidator


def test_goldens_pass_validator():
    """Every golden is SELECT-only and references only seeded tables."""
    SQLValidator = _load_validator()

    conn = harness.build_seeded_db()
    schema = harness.introspect_schema(conn)
    conn.close()
    validator = SQLValidator()
    failures = []
    for q in harness.load_questions():
        try:
            validator.validate_and_rewrite(q["golden_sql"], schema_metadata=schema)
        except Exception as e:  # noqa: BLE001
            failures.append((q["id"], str(e)))
    assert not failures, f"goldens rejected by validator: {failures}"


def test_eval_set_is_substantial():
    """Guard against silently shrinking the eval set (PRD-199 S3 grew it to
    32 across joins/dates/negations/ambiguity; PRD-160's own bar wants 30+)."""
    assert len(harness.load_questions()) >= 30


@pytest.mark.skipif(
    os.getenv("RUN_NL2SQL_EVAL") != "1",
    reason="generation accuracy eval is opt-in (set RUN_NL2SQL_EVAL=1 with an LLM)",
)
def test_generation_accuracy_no_regression(capsys):
    from modules.nl2sql.query.nl2sql_service import NaturalLanguageToSQLService
    from modules.nl2sql.query.validator import SQLValidator
    from core.llm import create_llm_manager

    generator = NaturalLanguageToSQLService(
        llm_provider=create_llm_manager(service_name="orchestrator")
    )

    def generate_fn(question, schema):
        sql, _expl, meta = generator.generate_sql(
            question=question, schema_metadata=schema, dialect="sqlite",
        )
        if not sql or meta.get("success") is False:
            raise RuntimeError(meta.get("error") or "generation failed")
        validated, _ = SQLValidator().validate_and_rewrite(sql, schema_metadata=schema)
        return validated

    report = harness.evaluate(generate_fn)
    baseline = harness.load_baseline().get("accuracy", 0.0)

    # surfaced in the PR (CI captures stdout for non-required jobs)
    with capsys.disabled():
        print(f"\n[nl2sql-eval] accuracy={report.accuracy:.3f} "
              f"({report.correct}/{report.total})  baseline={baseline:.3f}")
        for r in report.results:
            if not r.correct:
                print(f"  MISS {r.id}: {r.error or 'result mismatch'}")

    # write the run for the PR artifact / baseline bump
    out = {"accuracy": round(report.accuracy, 4), "correct": report.correct,
           "total": report.total}
    (harness.HERE / "last_run.json").write_text(json.dumps(out, indent=2))

    tolerance = 0.02
    assert report.accuracy >= baseline - tolerance, (
        f"accuracy {report.accuracy:.3f} regressed vs baseline {baseline:.3f}"
    )


@pytest.mark.skipif(
    os.getenv("RUN_NL2SQL_EVAL") != "1",
    reason="semantic A/B eval is opt-in (set RUN_NL2SQL_EVAL=1 with an LLM)",
)
def test_semantic_ab_reports_delta(capsys):
    """PRD-199 S4: semantic-on vs semantic-off on the same LLM and set.
    The delta is REPORTED (printed + ab_run.json), never asserted against a
    target — the target is set after the first honest measurement (§8-Q1)."""
    from modules.nl2sql.query.nl2sql_service import NaturalLanguageToSQLService
    from modules.nl2sql.query.validator import SQLValidator
    from core.llm import create_llm_manager

    generator = NaturalLanguageToSQLService(
        llm_provider=create_llm_manager(service_name="orchestrator")
    )

    def factory(semantic_doc):
        def generate_fn(question, schema):
            sql, _expl, meta = generator.generate_sql(
                question=question, schema_metadata=schema, dialect="sqlite",
                semantic_layer=semantic_doc,
            )
            if not sql or meta.get("success") is False:
                raise RuntimeError(meta.get("error") or "generation failed")
            validated, _ = SQLValidator().validate_and_rewrite(sql, schema_metadata=schema)
            return validated
        return generate_fn

    ab = harness.evaluate_ab(factory)

    with capsys.disabled():
        print(f"\n[nl2sql-eval] semantic A/B: with={ab['acc_with_semantic']:.3f} "
              f"without={ab['acc_without_semantic']:.3f} delta_s={ab['delta_s']:+.3f}")

    (harness.HERE / "ab_run.json").write_text(json.dumps(ab, indent=2))
    assert ab["total"] > 0  # shape only — the number is the deliverable
