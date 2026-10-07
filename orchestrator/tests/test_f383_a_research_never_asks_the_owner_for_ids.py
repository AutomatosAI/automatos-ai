"""F383 (night 11, 7 Oct): the content-bank research never asks the owner for document ids.

Task 2153 (Content bank research, started by saving a plan) blocked twice asking the shop
owner for document IDs ("the 'ref' parameter in the 'facts' object"), and none of its six
platform_add_social_topics calls added a topic, each answering success. Now:

* search_knowledge's text gives each source's document id, the ref a fact cites;
* a fact of kind note needs no ref (its label defaults to "Note");
* the research prompt says where each ref comes from and never to ask the owner for one,
  and every installed copy of the Playbook its owner never edited takes that prompt at
  boot; an owner-edited copy keeps its own;
* a call that adds nothing fails, the refusals' reasons first.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.seeds import seed_socials_package as seed  # noqa: E402
from core.seeds.socials_playbook_copies import refresh_installed_copies  # noqa: E402
from modules.socials import topics  # noqa: E402
from modules.tools.discovery import handlers_socials  # noqa: E402
from modules.tools.formatting.llm_summary_parts import source_line  # noqa: E402
from modules.tools.formatting.result_formatter import ToolResultFormatter  # noqa: E402
import tests.test_prd251bw2_topics as topic_tests  # noqa: E402
from tests.test_prd251bw2_topics import _create_plan, _tool, _topic  # noqa: E402

api = topic_tests.api
bank = topic_tests.bank

(RESEARCH,) = [spec for spec in seed.SOCIALS_PLAYBOOKS if spec["template_id"] == seed.RESEARCH_PLAYBOOK_TEMPLATE_ID]
NOTE_FACT = {"text": "The Harvest Club box goes on sale Monday 12 October.", "source": {"kind": "note"}}


# ── the ref a fact cites is in the search's text ────────────────────────────

def test_search_knowledge_gives_each_source_its_document_id():
    raw = {"success": True, "results": [
        {"content": "Wholesale terms: 30 days.", "source": "wholesale-terms.md", "similarity": 0.61, "document_id": 717},
        {"content": "Q2 revenue up 12%.", "source": "q2-2026-numbers.csv", "similarity": 0.44, "document_id": 724},
    ]}

    text = ToolResultFormatter.format_for_llm(raw, "search_knowledge")

    assert "[Source 1: wholesale-terms.md] (document 717; 61.0%)" in text      # night 11: no id anywhere
    assert "[Source 2: q2-2026-numbers.csv] (document 724; 44.0%)" in text
    assert "never ask the owner for one" in text
    assert "only a file named here" in text                                     # F088's rule stays


def test_a_source_with_no_document_id_reads_as_before():
    assert source_line(1, {"filename": "notes.md", "similarity": 0.5}) == "\n[Source 1: notes.md] (50.0%)"


# ── a note is its own source ────────────────────────────────────────────────

def test_a_note_needs_no_ref_and_its_label_defaults():
    fact = topics.validate_fact(NOTE_FACT, "facts[0]")

    assert fact["source"] == {"kind": "note", "ref": None, "label": topics.NOTE_LABEL}
    labelled = {**NOTE_FACT, "source": {"kind": "note", "label": "The plan's notes"}}
    assert topics.validate_fact(labelled, "facts[0]")["source"]["label"] == "The plan's notes"


@pytest.mark.parametrize("kind", ["knowledge", "deliverable", "web", "github"])
def test_any_other_source_still_needs_its_ref_and_label(kind):
    with pytest.raises(topics.InvalidTopic, match="source.ref is required"):
        topics.validate_fact({"text": "A figure.", "source": {"kind": kind, "label": "x"}}, "facts[0]")
    with pytest.raises(topics.InvalidTopic, match="source.label is required"):
        topics.validate_fact({"text": "A figure.", "source": {"kind": kind, "ref": "717"}}, "facts[0]")


# ── the tool's answer ───────────────────────────────────────────────────────

def test_research_adds_a_topic_resting_on_a_note(bank):
    plan = _create_plan(bank)
    answer = _tool(handlers_socials.add_social_topics, bank, plan_id=plan["id"],
                   topics=[_topic(title="Harvest Club in October", facts=[NOTE_FACT])])

    assert answer["success"] is True and [t["title"] for t in answer["added"]] == ["Harvest Club in October"]


def test_a_call_that_adds_nothing_fails_with_the_reasons_first(bank):
    plan = _create_plan(bank)
    unsourced = {"text": "1,240 bags roasted.", "source": {"kind": "knowledge", "label": "September report"}}
    answer = _tool(handlers_socials.add_social_topics, bank, plan_id=plan["id"],
                   topics=[_topic(title="September in numbers", facts=[unsourced])])

    assert answer["success"] is False and answer["added"] == []                # night 11: success, nothing added
    assert answer["error"].startswith('Nothing was added to the bank. "September in numbers": ')
    assert "source.ref is required" in answer["error"]
    assert "Never ask the owner for an id." in answer["error"]
    assert [r["index"] for r in answer["refused"]] == ["0"]


def test_the_topics_tool_says_where_a_ref_comes_from():
    from modules.tools.discovery.action_registry import get_action_registry

    action = get_action_registry().get("platform_add_social_topics")
    source = action.parameters["properties"]["topics"]["items"]["properties"]["facts"]["items"]["properties"]["source"]

    assert source["required"] == ["kind"]
    assert "search_knowledge" in source["properties"]["ref"]["description"]
    assert "Never ask the owner" in source["properties"]["ref"]["description"]


# ── the research prompt, and the copies installed with an older one ─────────

def test_the_research_prompt_says_where_each_ref_comes_from():
    (step,) = RESEARCH["steps"]
    prompt = step["prompt_template"]

    assert "search_knowledge source line" in prompt and "kind note, with no ref" in prompt
    assert "Never ask the owner for an id, a ref or a source" in prompt
    assert seed._RESEARCH_PROMPT_251C in seed.SEEDED_BEFORE[seed.RESEARCH_PLAYBOOK_TEMPLATE_ID]["research"]
    assert seed._RESEARCH_PROMPT_251C != prompt


def _copy(prompt, copy_id):
    step = {"step_id": "research", "order": 1, "agent_id": 412, "prompt_template": prompt}
    return NS(id=copy_id, workspace_id="febae41b", owner_type="workspace", steps=[step])


class _Rows:
    """db.query(...).filter(...): the marketplace row (first) and its workspace copies (all)."""

    def __init__(self, row, copies):
        self.row, self.copies = row, copies

    def query(self, *entities):
        return self

    def filter(self, *clauses):
        return self

    def first(self):
        return self.row

    def all(self):
        return self.copies


def test_an_unedited_copy_takes_the_new_prompt_and_an_edited_one_keeps_its_own():
    (step,) = RESEARCH["steps"]
    unedited, older, edited = (_copy(seed._RESEARCH_PROMPT_251C, 1), _copy(seed._RESEARCH_PROMPT_251B, 2),
                               _copy("Our own research: the roadmap first.", 3))
    row = NS(id=99, owner_type="marketplace", steps=[dict(unedited.steps[0], agent_id=11)])

    assert seed._ensure_playbook(_Rows(row, [unedited, older, edited]), RESEARCH, {}) == seed.UPDATED

    assert unedited.steps[0]["prompt_template"] == older.steps[0]["prompt_template"] == step["prompt_template"]
    assert unedited.steps[0]["agent_id"] == 412                                 # the copy's own agent stays
    assert edited.steps == [{"step_id": "research", "order": 1, "agent_id": 412,
                             "prompt_template": "Our own research: the roadmap first."}]
    assert refresh_installed_copies(_Rows(row, [unedited, edited]), row,
                                    lambda steps: seed.refreshed_steps(RESEARCH, steps)) == 0  # up to date now
