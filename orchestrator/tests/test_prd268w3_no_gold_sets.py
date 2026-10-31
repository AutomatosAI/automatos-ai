"""PRD-268 US-C13a (§6.7, K14): no gold set is ever committed under orchestrator/evals/proactive.

The repository is public; the proactive evals read their gold sets from PROACTIVE_EVAL_GOLD_DIR,
outside it. This guard fails on any *.jsonl tracked under the evals folder (the git index when git
is there, else a walk of the folder, as in an image without .git), and checks that .gitignore keeps
one copied in there out of the index.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import List

ORCHESTRATOR = Path(__file__).resolve().parents[1]
EVALS = ORCHESTRATOR / "evals" / "proactive"
REPO = ORCHESTRATOR.parent
RULE = "orchestrator/evals/proactive/**/*.jsonl"
GOLD_SUFFIX = ".jsonl"


def _tracked_by_git(folder: Path) -> List[str] | None:
    """The files git tracks under ``folder``, or None when git or a checkout is not there."""
    if shutil.which("git") is None:
        return None
    proc = subprocess.run(["git", "ls-files", "-z", "--", "."], cwd=folder, capture_output=True, timeout=30)
    if proc.returncode != 0:
        return None
    return [path for path in proc.stdout.decode("utf-8").split("\0") if path]


def _walked(folder: Path) -> List[str]:
    """Every file under ``folder``, relative to it: what an image without .git ships."""
    return sorted(str(path.relative_to(folder)) for path in folder.rglob("*") if path.is_file())


def _gold_sets(folder: Path) -> List[str]:
    files = _tracked_by_git(folder)
    listed = _walked(folder) if files is None else files
    return sorted(path for path in listed if path.lower().endswith(GOLD_SUFFIX))


def test_the_evals_folder_is_there():
    assert (EVALS / "gold.py").is_file(), "the guard must look at the folder the evals live in"


def test_no_gold_set_is_tracked_under_the_proactive_evals():
    found = _gold_sets(EVALS)
    assert not found, (
        f"gold data is tracked under {EVALS}: {found}. Gold sets stay local (public repo): keep them in "
        "the folder PROACTIVE_EVAL_GOLD_DIR names, outside the repository, and git rm --cached them.")


def test_the_walk_finds_a_gold_set_where_there_is_no_git(tmp_path):
    (tmp_path / "live").mkdir()
    (tmp_path / "live" / "triage.jsonl").write_text('{"expected": "act_today"}\n', encoding="utf-8")
    (tmp_path / "notes.md").write_text("not gold\n", encoding="utf-8")
    walked = [path for path in _walked(tmp_path) if path.endswith(GOLD_SUFFIX)]
    assert walked == [str(Path("live") / "triage.jsonl")]


def test_gitignore_keeps_a_gold_set_copied_into_the_folder_out():
    """Checked where a checkout is there; an image or a mounted orchestrator folder ships no .gitignore."""
    gitignore = REPO / ".gitignore"
    if gitignore.is_file():
        assert RULE in gitignore.read_text(encoding="utf-8").splitlines(), f".gitignore lacks {RULE}"
    if _tracked_by_git(EVALS) is None:
        return
    for name in ("orchestrator/evals/proactive/triage.jsonl", "orchestrator/evals/proactive/live/silence.jsonl"):
        proc = subprocess.run(["git", "check-ignore", "-q", "--no-index", name], cwd=REPO, timeout=30)
        assert proc.returncode == 0, f"git would track {name}"
    proc = subprocess.run(["git", "check-ignore", "-q", "--no-index", "orchestrator/evals/proactive/scoring.py"],
                          cwd=REPO, timeout=30)
    assert proc.returncode == 1, "the rule must ignore gold sets only, never the eval code"
