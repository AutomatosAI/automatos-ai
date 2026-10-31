"""PRD-268 §6.7 (US-C13, K14) — the proactive evals' command line.

    cd orchestrator
    python -m evals.proactive <triage|silence|reply_chips|spoken|precision|one_brief> [--json]
    python -m evals.proactive precision --workspace <workspace id>

Runs one eval and prints its bar, its numbers and PASS or MISS (``--json``: the score as JSON), then
its audit lines: every spoken answer for the hand audit, each workspace's precision. Precision reads
each workspace's suggestions, or one workspace's with ``--workspace``; no other eval takes it.
It exits 0 on a pass and 1 on a miss, and 2 when the eval cannot run to a result: no gold set
can be read (``gold.GoldSetUnavailable``), no workspace is set, or a model call failed
(``runtime.EvalCannotRun`` and any other failure, logged). The model is the platform's
``orchestrator_llm`` (Haiku 5.5 by default) on the workspace ``PROACTIVE_EVAL_WORKSPACE_ID``
names; the gold sets are read from ``PROACTIVE_EVAL_GOLD_DIR``, outside the repository.
Precision and one brief call no model and read no gold set: precision reads the configured
database in a READ ONLY transaction, and one brief writes a scratch workspace there and removes it.

Each eval's runner is its own module here (``evals.proactive.<eval>``), imported when it runs,
with ``async def evaluate() -> Score``. The owner runs them by hand, never CI; a run under a bar
is a miss, reported as it is. Printing happens here only (ruff exempts ``__main__.py`` from T20).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import sys
from typing import Any, Dict, List, Optional, Union

from evals.proactive.bars import BAR_TEXT
from evals.proactive.gold import GoldSetUnavailable
from evals.proactive.runtime import EvalCannotRun
from evals.proactive.scoring import Score

logger = logging.getLogger(__name__)

EVALS = ("triage", "silence", "reply_chips", "spoken", "precision", "one_brief")
# The eval that takes --workspace, as its evaluate()'s workspace_id.
BY_WORKSPACE = "precision"
WORKSPACE_ONLY_FOR = "--workspace is for the precision eval only"
RUNNER_PACKAGE = "evals.proactive"
EXIT_PASS, EXIT_MISS, EXIT_CANNOT_RUN = 0, 1, 2
CANNOT_RUN = "CANNOT RUN"
FAILED = "the run failed ({error}); the log has the detail"
DIGITS = 3


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m evals.proactive",
                                     description="PRD-268 §6.7: run one proactive eval against its bar.")
    parser.add_argument("eval", choices=EVALS, help="the eval to run")
    parser.add_argument("--json", action="store_true", help="print the score as JSON")
    parser.add_argument("--workspace", metavar="ID", help="precision only: one workspace's suggestions")
    return parser


async def _evaluate(name: str, options: Dict[str, Any]) -> Score:
    runner = importlib.import_module(f"{RUNNER_PACKAGE}.{name}")
    return await runner.evaluate(**options)


def _shown(value: Any) -> str:
    return f"{value:.{DIGITS}f}" if isinstance(value, float) else str(value)


def _text(score: Score) -> str:
    lines = [f"{score.name}: {score.verdict}", f"bar: {score.bar}"]
    lines += [f"  {key}: {_shown(value)}" for key, value in score.numbers.items()]
    return "\n".join([*lines, *score.audit])


def _refusal(name: str, reason: str, as_json: bool) -> str:
    if as_json:
        return json.dumps({"eval": name, "bar": BAR_TEXT[name], "verdict": CANNOT_RUN, "reason": reason}, indent=2)
    return f"{name}: {CANNOT_RUN}\nbar: {BAR_TEXT[name]}\n  {reason}"


def _run(name: str, options: Dict[str, Any]) -> Union[Score, str]:
    """The eval's score, or why it could not run."""
    try:
        return asyncio.run(_evaluate(name, options))
    except (GoldSetUnavailable, EvalCannotRun) as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001 — logged, and reported as a run that could not finish (exit 2)
        logger.exception("[evals.proactive] %s failed", name)
        return FAILED.format(error=type(exc).__name__)


def main(argv: Optional[List[str]] = None) -> int:
    """Run the eval ``argv`` names, print its result, and return the exit code."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.workspace is not None and args.eval != BY_WORKSPACE:
        parser.error(WORKSPACE_ONLY_FOR)
    outcome = _run(args.eval, {"workspace_id": args.workspace} if args.workspace is not None else {})
    if not isinstance(outcome, Score):
        print(_refusal(args.eval, outcome, args.json))
        return EXIT_CANNOT_RUN
    print(json.dumps(outcome.as_dict(), indent=2, default=str) if args.json else _text(outcome))
    return EXIT_PASS if outcome.passed else EXIT_MISS


if __name__ == "__main__":
    sys.exit(main())
