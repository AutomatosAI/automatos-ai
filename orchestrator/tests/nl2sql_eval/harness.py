"""PRD-160 S5 — NL2SQL regression eval harness.

sql-eval-style *execution accuracy*: for each seeded question we run a candidate
SQL and the golden SQL against the same seeded database and compare their result
sets (order-insensitive, float-rounded). The harness is self-contained — it
seeds an in-memory SQLite DB from ``seed_schema.sql`` — so it runs anywhere with
no external service, which is what lets it be a non-required CI job.

The harness is generation-agnostic: ``evaluate(generate_fn)`` takes a callable
``(question, schema_metadata) -> sql``. The pytest entry wires in the real
NaturalLanguageToSQLService when an LLM is configured, and otherwise only checks
that the golden SQL itself is valid + executable (always-on integrity).
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
SEED_SQL = HERE / "seed_schema.sql"
QUESTIONS = HERE / "questions.json"
BASELINE = HERE / "baseline.json"


def build_seeded_db() -> sqlite3.Connection:
    """Return an in-memory SQLite connection seeded from seed_schema.sql."""
    conn = sqlite3.connect(":memory:")
    conn.executescript(SEED_SQL.read_text())
    return conn


def introspect_schema(conn: sqlite3.Connection) -> Dict[str, Any]:
    """Build the schema_metadata dict the generator/validator expect."""
    tables = []
    names = [
        r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
    ]
    for tname in names:
        cols = []
        for row in conn.execute(f'PRAGMA table_info("{tname}")').fetchall():
            # row = (cid, name, type, notnull, dflt, pk)
            cols.append({"name": row[1], "type": (row[2] or "").lower(),
                         "primary_key": bool(row[5])})
        tables.append({"name": tname, "columns": cols})
    return {"tables": tables}


def load_questions() -> List[Dict[str, Any]]:
    return json.loads(QUESTIONS.read_text())["questions"]


def load_baseline() -> Dict[str, Any]:
    if BASELINE.exists():
        return json.loads(BASELINE.read_text())
    return {"accuracy": 0.0}


def run_sql(conn: sqlite3.Connection, sql: str) -> List[Tuple[Any, ...]]:
    return conn.execute(sql).fetchall()


def _norm_value(v: Any) -> Any:
    if isinstance(v, (float, Decimal)):
        return round(float(v), 4)
    return v


def normalize_rows(rows: List[Tuple[Any, ...]]) -> List[Tuple[Any, ...]]:
    """Order-insensitive, float-rounded representation for comparison."""
    return sorted(tuple(_norm_value(v) for v in row) for row in rows)


def result_sets_match(a: List[Tuple], b: List[Tuple]) -> bool:
    try:
        return normalize_rows(a) == normalize_rows(b)
    except TypeError:
        # unsortable mixed types — fall back to multiset of stringified rows
        sa = sorted(str(tuple(_norm_value(v) for v in r)) for r in a)
        sb = sorted(str(tuple(_norm_value(v) for v in r)) for r in b)
        return sa == sb


@dataclass
class QuestionResult:
    id: str
    question: str
    golden_sql: str
    candidate_sql: Optional[str] = None
    correct: bool = False
    error: Optional[str] = None


@dataclass
class EvalReport:
    total: int = 0
    correct: int = 0
    results: List[QuestionResult] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return (self.correct / self.total) if self.total else 0.0


def evaluate(generate_fn: Callable[[str, Dict[str, Any]], str]) -> EvalReport:
    """Run every seeded question through ``generate_fn`` and score by execution
    accuracy against the golden SQL."""
    conn = build_seeded_db()
    schema = introspect_schema(conn)
    report = EvalReport()

    for q in load_questions():
        report.total += 1
        qr = QuestionResult(id=q["id"], question=q["question"], golden_sql=q["golden_sql"])
        try:
            golden_rows = run_sql(conn, q["golden_sql"])
        except Exception as e:  # a broken golden is a harness bug, surface it
            qr.error = f"golden failed: {e}"
            report.results.append(qr)
            continue
        try:
            candidate_sql = generate_fn(q["question"], schema)
            qr.candidate_sql = candidate_sql
            cand_rows = run_sql(conn, candidate_sql)
            qr.correct = result_sets_match(cand_rows, golden_rows)
        except Exception as e:
            qr.error = str(e)
        if qr.correct:
            report.correct += 1
        report.results.append(qr)

    conn.close()
    return report


def check_goldens() -> EvalReport:
    """Always-on integrity: every golden SQL must execute on the seed (an
    identity eval where the candidate IS the golden — accuracy must be 1.0)."""
    return evaluate(lambda question, schema, _q=load_questions(): next(
        x["golden_sql"] for x in _q
        if x["question"] == question
    ))


# ---------------------------------------------------------------------------
# PRD-199 S4 — the semantic lever, measured (semantic-on vs semantic-off)
# ---------------------------------------------------------------------------

SEMANTIC_FIXTURE = HERE / "semantic_fixture.json"


def load_semantic_fixture() -> Dict[str, Any]:
    """The seeded canonical semantic doc for the A/B eval. Its definitions
    deliberately AGREE with the goldens' interpretation — the A/B measures
    whether *stating* the vocabulary helps the LLM match it."""
    return json.loads(SEMANTIC_FIXTURE.read_text())


def evaluate_ab(
    generate_fn_factory: Callable[
        [Optional[Dict[str, Any]]], Callable[[str, Dict[str, Any]], str]
    ],
) -> Dict[str, Any]:
    """Run the full set twice — with the seeded semantic doc and without —
    and report the measured uplift ΔS = acc_with − acc_without.

    ΔS is REPORTED, never asserted against a target: the ~20%-class uplift
    is Snowflake's number on Snowflake's datasets; the target here is set
    only after the first honest measurement (§8-Q1, ⏸ PENDING REAL EVAL).
    ΔS is also what sizes how much semantic-editor UI is worth building.
    """
    semantic_doc = load_semantic_fixture()
    with_report = evaluate(generate_fn_factory(semantic_doc))
    without_report = evaluate(generate_fn_factory(None))
    return {
        "acc_with_semantic": round(with_report.accuracy, 4),
        "acc_without_semantic": round(without_report.accuracy, 4),
        "delta_s": round(with_report.accuracy - without_report.accuracy, 4),
        "total": with_report.total,
        "with_correct": with_report.correct,
        "without_correct": without_report.correct,
    }
