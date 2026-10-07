"""F378 (night 11, 7 Oct): sources come from the brief, hold the figure, and are never invented.

B1: the Infographic compose bound the owner's café numbers to three unrelated reports (the
render then refused: "row 1 shows 'Lantern Kitchen (Bristol) 96 kg'; the report has 'task id
2,150'") and printed the invented source label "Internal Report, 2026-10-06". Pinned:

* candidate sources are searched with the brief's words, never with an empty query;
* a claim bound to a source that does not hold its figure is unbound, with a warning; one
  that does stays bound; an address the brief quotes is the owner's citation;
* the source line names the bound source (its title and day), or words the brief gives;
  otherwise it is left for the owner: a question, never asked of the model again.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

import api.socials_compose as compose_api  # noqa: E402
from core.chart_binding import chip_text  # noqa: E402
from modules.socials import compose, compose_checks, compose_sources  # noqa: E402

BRIEF = "Infographic: our wholesale in October. Lantern Kitchen 96 kg, Quay Street Coffee 80 kg. Source: our own till."
SCHEMA = {
    "headline": {"type": "text", "label": "Headline", "max_chars": 60},
    "row_1_value": {"type": "text", "label": "Row 1: the figure", "claim": True, "default": ""},
    "source_label": {"type": "text", "label": "Source", "max_chars": 90},
}
CHART = {"id": "chart", "name": "Infographic", "format": "social_image", "sizes": ["1080x1350"], "variables_schema": SCHEMA}
UNRELATED = {"kind": "report", "ref": "r-task", "title": "Task: Tide Café wholesale proposal", "detail": "task id 2,150",
             "as_of": "2026-10-06T10:00:00+00:00"}
RELATED = {"kind": "report", "ref": "r-week", "title": "Wholesale October", "detail": "Lantern Kitchen 96 kg",
           "as_of": "2026-10-05T10:00:00+00:00"}


def _ctx(brief=BRIEF, candidates=(UNRELATED, RELATED)):
    return compose.ComposeContext(brief=brief, format="infographic", channels=[{"toolkit": "instagram", "label": "Instagram"}],
                                  templates=[CHART], candidates=list(candidates))


def _raw(source=None, label=None):
    variables = {"headline": "Our wholesale", "row_1_value": "96 kg"}
    if label is not None:
        variables["source_label"] = label
    return {"title": "Wholesale", "format": "infographic", "template_id": "chart", "copy": {"base": "Our wholesale.", "channels": {}},
            "variables": variables, "sources": {"row_1_value": source} if source else {}}


def test_candidates_are_searched_with_the_briefs_words(monkeypatch):
    asked = []

    def search(db, workspace_id, *, kind=None, q=None, limit=10):
        asked.append((kind, q, limit))
        return [{"kind": "report", "ref": "r-week", "title": "Wholesale October"}]

    monkeypatch.setattr(compose_api.post_sources, "search", search)
    found = compose_api.candidate_sources(None, uuid.uuid4(), BRIEF + " https://tide.cafe/numbers")
    terms = compose_sources.brief_terms(BRIEF)
    assert terms and all(q for _kind, q, _limit in asked)
    assert [q for kind, q, _limit in asked if kind is None] == terms
    assert ("url", "https://tide.cafe/numbers", 1) in asked
    assert found == [{"kind": "report", "ref": "r-week", "title": "Wholesale October"}]  # each once


def test_the_briefs_terms_are_its_distinctive_words():
    terms = compose_sources.brief_terms("Please post about our Harvest Club subscribers this week https://a.b/c")
    assert terms == ["subscribers", "harvest", "club"]


def test_a_claim_bound_to_a_source_without_its_figure_is_unbound():
    proposal = compose_checks.checked_proposal(_raw({"kind": "report", "ref": "r-task"}), _ctx())
    assert proposal["sources"] == {}
    assert "The figure of row_1_value (96 kg) is not in its source 'Task: Tide Café wholesale proposal'; it was unbound" in proposal["warnings"]


def test_a_claim_bound_to_a_source_holding_its_figure_stays_and_names_the_source_line():
    proposal = compose_checks.checked_proposal(_raw({"kind": "report", "ref": "r-week"}, label="Internal Report, 2026-10-06"), _ctx())
    assert proposal["sources"]["row_1_value"]["ref"] == "r-week"
    assert proposal["variables"]["source_label"]["value"] == chip_text("Wholesale October", RELATED["as_of"], 90)


def test_an_address_the_brief_quotes_is_the_owners_citation():
    url = {"kind": "url", "ref": "https://tide.cafe/numbers", "title": "tide.cafe"}
    proposal = compose_checks.checked_proposal(_raw({"kind": "url", "ref": url["ref"]}), _ctx(candidates=[url]))
    assert proposal["sources"]["row_1_value"]["kind"] == "url"


def test_a_source_line_nobody_gave_is_left_for_the_owner():
    proposal = compose_checks.checked_proposal(_raw(label="Internal Report, 2026-10-06"), _ctx())
    assert "source_label" not in proposal["variables"]
    assert any(w.startswith("The source line 'Internal Report, 2026-10-06' names no source") for w in proposal["warnings"])
    kept = compose_checks.checked_proposal(_raw(label="our own till"), _ctx())
    assert kept["variables"]["source_label"]["value"] == "our own till"


class _Model:
    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    async def generate_response(self, messages, tools=None):
        self.asked.append(messages)
        answer = self.answers.pop(0)
        return SimpleNamespace(content=answer if isinstance(answer, str) else json.dumps(answer))


def test_the_model_is_never_asked_for_a_source_line_it_cannot_know():
    model = _Model([_raw(label="Internal Report, 2026-10-06")])
    proposal = asyncio.run(compose.propose(_ctx(brief=BRIEF.replace(" Source: our own till.", "")), lambda: model, 5.0))
    assert len(model.asked) == 1  # no follow-up for the source line
    assert "Source" in proposal["questions"]
