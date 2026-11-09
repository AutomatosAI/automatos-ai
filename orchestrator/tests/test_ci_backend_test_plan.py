"""8 Oct ("it's getting so so slow to pass a PR"): orchestrator-tests runs as shards.

``scripts/ci/backend_test_plan.py`` gives every test file to exactly one shard, so
the shards together run every test the single job ran: none is skipped, none runs
twice. A PR that only touches the frontend, the docs or a top-level markdown file
runs only the test files that name the folder it touched, and the ones that read the
whole tracked tree; anything else runs them all.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / "ci" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


plan = _load("backend_test_plan")
TEST_FILES = plan.read_test_files()
SECONDS = plan.json.loads(plan.DURATIONS.read_text(encoding="utf-8"))


@pytest.mark.parametrize("count", [1, 2, 3, 4, 6])
def test_every_test_file_lands_in_exactly_one_shard(count):
    shards = plan.balance(plan.select(TEST_FILES, []), SECONDS, count)

    taken = [name for shard in shards for name in shard]
    assert sorted(taken) == sorted(TEST_FILES)
    assert len(taken) == len(set(taken))


def test_this_file_is_one_of_the_planned_test_files():
    assert "tests/test_ci_backend_test_plan.py" in TEST_FILES


def test_shards_carry_near_equal_recorded_seconds():
    shards = plan.balance(sorted(SECONDS), SECONDS, 4)

    loads = [sum(SECONDS[name] for name in shard) for shard in shards]
    assert max(loads) - min(loads) <= max(SECONDS.values())


def test_a_file_with_no_record_weighs_the_median():
    shards = plan.balance(["tests/new.py", "tests/a.py", "tests/b.py"], {"tests/a.py": 10.0, "tests/b.py": 2.0}, 2)

    assert shards == [["tests/a.py"], ["tests/b.py", "tests/new.py"]]


@pytest.mark.parametrize(
    "paths",
    [
        ["orchestrator/main.py"],
        # A lock change runs every test, test_prd262_api_locks.py among them (PRD-262 US-022).
        ["frontend/app/page.tsx", "orchestrator/requirements.lock"],
        [".github/workflows/test.yml"],
        ["docker-compose.yml"],
        ["services/workspace-worker/main.py"],
        ["scripts/ci/backend_test_plan.py"],
        ["LICENSE"],
    ],
)
def test_a_change_outside_the_selectable_paths_runs_every_test(paths):
    assert plan.scope(paths, TEST_FILES) == {"mode": "full", "keywords": []}


def test_a_frontend_and_readme_change_runs_the_tests_that_name_them():
    scope = plan.scope(["frontend/app/page.tsx", "README.md", ""], TEST_FILES)

    assert scope == {"mode": "select", "keywords": ["README", "frontend/app/"]}
    selected = plan.select(TEST_FILES, scope["keywords"])
    assert selected and len(selected) < len(TEST_FILES)
    assert all("frontend" in TEST_FILES[name] or "README" in TEST_FILES[name] or plan.WHOLE_TREE.search(TEST_FILES[name])
               for name in selected)


def test_another_workflow_runs_the_tests_that_name_its_folder():
    assert plan.scope([".github/workflows/codeql.yml"], TEST_FILES)["keywords"] == [".github/workflows/"]


def test_a_change_no_test_names_runs_none():
    files = {"tests/test_a.py": "frontend", "tests/test_b.py": "docs"}

    assert plan.scope(["graphify-out/GRAPH_REPORT.md"], files) == {"mode": "none", "keywords": ["graphify-out/"]}
    assert plan.scope([], files)["mode"] == "none"


def test_select_takes_only_the_files_naming_a_keyword():
    files = {"tests/test_a.py": 'ROOT / "frontend" / "x.ts"', "tests/test_b.py": "nothing", "tests/test_c.py": "README.md"}

    assert plan.select(files, ["frontend/", "README"]) == ["tests/test_a.py", "tests/test_c.py"]
    assert plan.select(files, []) == sorted(files)


# #1455: a PRD-only change ran the 102 test files that say "docs" (/docs routes, docstrings).
def test_a_folder_is_named_as_a_path_to_it_or_a_parent_not_as_a_word():
    files = {
        "tests/test_joined.py": '_ROOT / "docs" / "PRDS" / "PRD-143-OBS-TIER-MANIFEST.md"',
        "tests/test_string.py": 'MANIFEST = "docs/PRDS/PRD-152.md"',
        "tests/test_parent.py": 'DOCS = REPO / "docs"\nfor path in DOCS.rglob("*.md"): ...',
        "tests/test_relative.py": 'Path("../docs/PRDS")',
        "tests/test_sibling.py": 'GUIDE = REPO / "docs" / "getting-started" / "self-hosting.md"',
        "tests/test_route.py": '"""The API docs stay up."""\nclient.get("/docs")',
        "tests/test_longer.py": '"docs/PRDS-archive/x.md"',
    }

    assert plan.scope(["docs/PRDS/PRD-274-AUTOMATOS-FOR-PRODUCTS.md"], files)["keywords"] == ["docs/PRDS/"]
    assert plan.select(files, ["docs/PRDS/"]) == [
        "tests/test_joined.py", "tests/test_parent.py", "tests/test_relative.py", "tests/test_string.py"]


def test_a_test_reading_the_whole_tracked_tree_runs_for_any_change():
    files = {
        "tests/test_listed.py": '["git", "ls-files"]',
        "tests/test_grepped.py": '["git", "grep", "-l", name, "--", "."]',
        "tests/test_helper.py": '_git_grep("-n", "-F", *patterns)',
        "tests/test_other.py": "nothing",
    }

    assert plan.select(files, ["e2e/browser/"]) == ["tests/test_grepped.py", "tests/test_helper.py", "tests/test_listed.py"]


def test_a_prd_change_runs_a_fraction_of_the_tests_that_say_docs():
    selected = plan.select(TEST_FILES, ["docs/PRDS/"])

    assert "tests/test_prd143_boundary_sweep.py" in selected  # _REPO_ROOT / "docs" / "PRDS" / ...
    assert "tests/test_prd262_dead_weight_retired.py" in selected  # git grep over every tracked file
    assert len(selected) * 3 < len([name for name, source in TEST_FILES.items() if "docs" in source])


def test_durations_come_from_the_gaps_between_result_lines():
    log = [
        "﻿2026-10-07T21:37:00.0000000Z tests/test_a.py::test_one PASSED [  0%]",
        "2026-10-07T21:37:01.5000000Z tests/test_a.py::test_two PASSED [  0%]",
        "2026-10-07T21:37:02.0000000Z some other line",
        "2026-10-07T21:37:04.0000000Z tests/test_b.py::test_x[1] FAILED [  1%]",
    ]

    assert plan.durations_from_log(log) == {"tests/test_a.py": 1.5, "tests/test_b.py": 2.5}


def test_plan_prints_one_shard_of_the_selected_files(capsys):
    assert plan.main(["plan", "--shard", "1/3"]) == 0

    printed = capsys.readouterr().out.split()
    expected = plan.balance(plan.select(TEST_FILES, []), SECONDS, 3)[1]
    assert printed == expected
