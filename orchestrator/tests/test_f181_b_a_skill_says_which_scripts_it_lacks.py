"""F181 (night 6) — a loaded skill says which of the scripts it names are not installed.

spreadsheet-qa came from OneWave-AI/claude-skills without its scripts: skill
164 has no skill_files rows. Agent 325 loaded it at 03:25:37 and 03:29:17 and
ran `python3 scripts/profile.py …` at 03:30:07, and the workspace had no such
file. load_skill now ends the skill's instructions with a note naming each
script they name that did not come with the skill, and telling the agent to do
those steps with its own code.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace as NS

import pytest

from modules.tools.discovery import handlers_skill_runtime as runtime

WS = "dacae30f-7840-40c1-8d03-25c3910affd0"
BODY = ("# Spreadsheet QA\n\n```bash\npython3 scripts/profile.py path/to/*.csv path/to/book.xlsx --out sqa_out\n"
        "```\nWrite a question spec and run it:\n```bash\npython3 scripts/ask.py spec.json --out sqa_out/q1\n```\n"
        "`python3 tests/run_tests.py` rebuilds the fixtures (`tests/questions/*.json`).")


class _Db:
    def __init__(self, shipped):
        self.shipped = shipped

    def query(self, *_a):
        return self

    def filter(self, *_a):
        return self

    def all(self):
        return [(path,) for path in self.shipped]


@pytest.fixture
def load(monkeypatch):
    skill = NS(id=164, name="spreadsheet-qa", prompt_template=BODY)
    monkeypatch.setattr(runtime, "_resolve_visible_skill", lambda db, ws, name="", skill_id=None: skill)
    return lambda shipped: asyncio.run(runtime.load_skill(_Db(shipped), WS, {"name": "spreadsheet-qa"}))


def test_a_skill_without_its_scripts_says_not_to_run_them(load):
    out = load([])

    assert out["success"] is True and out["content"].startswith(BODY)
    assert out["content"].endswith(
        "Not installed here: scripts/ask.py, scripts/profile.py, tests/run_tests.py are named above but did not "
        "come with this skill, so do not run them. Do those steps with your own code (python3 and its csv "
        "module), and say in your answer that the skill's scripts were missing.")


def test_a_skill_that_brought_its_scripts_is_as_written(load):
    assert load(["scripts/profile.py", "./scripts/ask.py", "tests/run_tests.py"])["content"] == BODY


def test_one_missing_script_is_named_alone(load):
    content = load(["scripts/ask.py", "tests/run_tests.py"])["content"]
    assert content.endswith("Not installed here: scripts/profile.py is named above but did not come with this "
                            "skill, so do not run it. Do those steps with your own code (python3 and its csv "
                            "module), and say in your answer that the skill's scripts were missing.")
