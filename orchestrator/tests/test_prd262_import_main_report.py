"""PRD-262 US-032 (FR-6): what `import main` loads, and how long it takes.

test.yml's backend job runs scripts/ci/import_main_report.py once per run. It imports
`main` in a fresh interpreter under `python -X importtime`, fails when a module FR-6
names is loaded (naming the import chain that loaded it), writes the 25 slowest
imports and the wall time to the step summary, and judges the wall time against a
budget once one is set. This suite checks:

- the FR-6 list the script holds is the PRD's;
- importtime's rows parse, with each module's depth, and the chain that loaded a
  module is read back from the tree;
- a loaded FR-6 module, or a submodule of one, is found; the top rows are the slowest;
- the end-to-end run on a throwaway `main.py`: a clean one passes, one that imports a
  stand-in `cv2` fails with its chain, and a budget it misses fails while an empty one
  only reports;
- the backend job runs the script on the first shard, after the tests and their
  coverage, with an empty budget until the first week's numbers set it.
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ci" / "import_main_report.py"


def _load():
    spec = importlib.util.spec_from_file_location("import_main_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["import_main_report"] = module  # before exec: its NamedTuples resolve their module
    spec.loader.exec_module(module)
    return module


report = _load()

# What `python -X importtime -c 'import main'` prints: a module after its imports.
IMPORTTIME = """\
import time: self [us] | cumulative | imported package
import time:       120 |        120 | _json
import time:       300 |        420 | json
import time:       900 |        900 |         cv2.data
import time:     50000 |      50900 |       cv2
import time:      2000 |      52900 |     camelot
import time:       400 |      53300 |   modules.rag
import time:       700 |        700 |   api.chat
import time:      1500 |      55500 | main
"""


def test_the_forbidden_list_is_fr6():
    assert report.FR6_MODULES == (
        "camelot", "cv2", "matplotlib", "pandas", "nltk", "google.generativeai", "googleapiclient")
    assert report.TOP_IMPORTS == 25


def test_importtime_rows_parse_with_their_depth():
    rows = report.parse_importtime(IMPORTTIME + "INFO some log line\n")
    assert [row.name for row in rows] == ["_json", "json", "cv2.data", "cv2", "camelot", "modules.rag",
                                          "api.chat", "main"]
    assert rows[2] == report.Row(900, 900, 4, "cv2.data")
    assert rows[-1] == report.Row(1500, 55500, 0, "main")


def test_the_chain_that_loaded_a_module_is_read_back_from_the_tree():
    rows = report.parse_importtime(IMPORTTIME)
    assert report.import_chain(rows, "cv2") == ["main", "modules.rag", "camelot", "cv2"]
    assert report.import_chain(rows, "cv2.data") == ["main", "modules.rag", "camelot", "cv2", "cv2.data"]
    assert report.import_chain(rows, "api.chat") == ["main", "api.chat"]
    assert report.import_chain(rows, "json") == ["json"]
    assert report.import_chain(rows, "pandas") == []


def test_a_loaded_fr6_module_or_a_submodule_of_one_is_found():
    modules = ["json", "cv2", "cv2.data", "google", "google.generativeai.client", "googleapis_common"]
    found = report.loaded(modules, report.FR6_MODULES)
    assert found == {"cv2": "cv2", "google.generativeai": "google.generativeai.client"}
    assert report.loaded(["google", "googleapis_common", "pandasai"], report.FR6_MODULES) == {}


def test_the_top_rows_are_the_slowest_cumulative():
    rows = report.parse_importtime(IMPORTTIME)
    assert [row.name for row in report.top(rows, 3)] == ["main", "modules.rag", "camelot"]


def _app(tmp_path: Path, main_source: str) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    (app / "main.py").write_text(main_source, encoding="utf-8")
    (app / "cv2.py").write_text("VERSION = 'stand-in'\n", encoding="utf-8")
    return app


def _run(tmp_path: Path, main_source: str, *extra: str) -> tuple[int, str]:
    summary = tmp_path / "summary.md"
    code = report.main(["--cwd", str(_app(tmp_path, main_source)), "--summary", str(summary), *extra])
    return code, summary.read_text(encoding="utf-8")


def test_a_main_that_loads_no_fr6_module_passes_and_reports_its_time(tmp_path, capsys):
    # The sleep makes main the slowest import, whatever the interpreter loads at start-up.
    code, summary = _run(tmp_path, "import json\nimport time\ntime.sleep(0.3)\n", "--budget-seconds", "")
    assert code == 0, capsys.readouterr().out
    assert "FR-6 modules loaded by `import main`: none." in summary
    assert "no budget yet: it is set from the first week's numbers" in summary
    assert re.search(r"^\| 1 \| [\d,.]+ ms \| [\d,.]+ ms \| `main` \|$", summary, re.M), summary
    assert len(re.findall(r"^\| \d+ \|", summary, re.M)) <= report.TOP_IMPORTS


def test_a_main_that_loads_one_fails_and_names_the_chain(tmp_path, capsys):
    code, summary = _run(tmp_path, "import cv2\n")
    out = capsys.readouterr().out
    assert code == 1
    assert "::error title=import main (PRD-262 US-032)::`import main` loads cv2 (main → cv2)" in out
    assert "- `cv2`: main → cv2" in summary


def test_a_budget_is_judged_once_set(tmp_path, capsys):
    code, summary = _run(tmp_path, "import time\ntime.sleep(0.3)\n", "--budget-seconds", "0.1")
    assert code == 1
    assert "over its 0.1 s budget" in capsys.readouterr().out
    assert "budget 0.1 s, **" in summary


def test_a_main_that_fails_to_import_fails_the_step(tmp_path, capsys):
    code, summary = _run(tmp_path, "raise SystemExit('config refused')\n")
    assert code == 1
    assert "`import main` failed with exit 1" in capsys.readouterr().out
    assert "**Failed:**" in summary


def test_the_backend_job_runs_it_once_after_the_tests():
    job = yaml.safe_load((ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8"))["jobs"][
        "orchestrator-tests-shard"]
    assert job["defaults"]["run"]["working-directory"] == "orchestrator"
    steps = job["steps"]
    (at,) = [i for i, step in enumerate(steps) if "import_main_report.py" in step.get("run", "")]
    names = [step.get("name") for step in steps]
    assert at > names.index("Run test net (with coverage)") and at > names.index("Keep this shard's coverage data")
    step = steps[at]
    assert step["if"] == "${{ !cancelled() && matrix.shard == 0 && steps.deps.outcome == 'success' }}"
    assert steps[names.index("Install Python dependencies")]["id"] == "deps"
    assert step["env"] == {"IMPORT_MAIN_BUDGET_SECONDS": ""}, "set from the first week's numbers (PRD-262 US-032)"
    assert step["run"] == ('python ../scripts/ci/import_main_report.py --budget-seconds '
                           '"$IMPORT_MAIN_BUDGET_SECONDS" --summary "$GITHUB_STEP_SUMMARY"')
