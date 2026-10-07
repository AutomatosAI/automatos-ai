"""F378 (night 11, 7 Oct): no template placeholder and no bare "true" reaches a post.

B15: compose wrote "We sold {retail_bags_sold} retail bags…" and the save kept the braces.
B-i3-5: Auto's quote card printed "true" as its eyebrow. Pinned:

* the template contract refuses a text value that holds a placeholder (``{name}``,
  ``{{ name }}``, ``[a_name]``, ``[Your name]``) or is a bare literal word, so a post's
  render, the composer and an agent's ``generate_document`` all refuse it; a value too
  long says how long it was;
* a post's save refuses copy or a variable holding a placeholder, and a variable that is
  a bare literal, naming the field; plain words pass;
* the composer asks once for copy that holds a placeholder, with the reason; what is still
  there is a warning and an owner's question in plain words;
* the skills' braces are explained to the model: a post never carries them.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_HOST", "127.0.0.1")
os.environ.setdefault("POSTGRES_PORT", "59432")
os.environ.setdefault("POSTGRES_DB", "test")

_ORCH = Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from core.social_templates import InvalidVariableValues, resolve_variables  # noqa: E402
from core.social_text_values import placeholder_in, placeholder_label  # noqa: E402
from modules.documents.generation_service import DocumentGenerationService  # noqa: E402
from modules.documents.social_starters import social_starters  # noqa: E402
from modules.socials import compose, service  # noqa: E402

QUOTE = next(starter for starter in social_starters() if starter["name"] == "Quote card")
SCHEMA = {"headline": {"type": "text", "label": "Headline", "max_chars": 24}, "note": {"type": "text", "default": ""}}
CARD = {"id": "card", "name": "Fact card", "format": "social_image", "sizes": ["1080x1350"], "variables_schema": SCHEMA}


@pytest.mark.parametrize("text, found", [
    ("We sold {retail_bags_sold} retail bags", "{retail_bags_sold}"),
    ("Hello {{ name }}", "{{ name }}"),
    ("Up [growth_percent] on last year", "[growth_percent]"),
    ("Love, [Your Name]", "[Your Name]"),
    ("Harbour Blend, £32 a bag", None),
    ("Tide {Café} opens", None),  # a word in braces is no field name
])
def test_a_placeholder_is_found_and_plain_words_are_not(text, found):
    assert placeholder_in(text) == found


def test_a_placeholder_reads_as_plain_words_in_a_question():
    assert placeholder_label("{retail_bags_sold}") == "Retail bags sold"
    assert placeholder_label("[Your name]") == "Name"


@pytest.mark.parametrize("value", ["true", "False", " null ", "None", "undefined", "{harvest_club_subscribers}"])
def test_the_contract_refuses_a_bare_literal_or_a_placeholder_as_text(value):
    schema = QUOTE["blocks"]["variables_schema"]
    resolved = resolve_variables(schema, {**QUOTE["sample_data"], "eyebrow": value})
    assert "eyebrow" not in resolved.values
    assert any(problem.startswith("eyebrow ") for problem in resolved.invalid)


def test_generate_document_refuses_the_quote_cards_true_eyebrow():
    template = SimpleNamespace(format="social_image", blocks=QUOTE["blocks"], name="Quote card")
    with pytest.raises(InvalidVariableValues) as refused:
        DocumentGenerationService._social_values(template, {**QUOTE["sample_data"], "eyebrow": "true"}, "social_image")
    assert "eyebrow is the word 'true'" in str(refused.value)


def test_a_value_too_long_says_how_long_it_was():
    (problem,) = resolve_variables(SCHEMA, {"headline": "x" * 31}).invalid
    assert problem == "headline is longer than 24 characters (31 given)"


def test_a_save_refuses_copy_or_a_variable_holding_a_placeholder_and_names_the_field():
    with pytest.raises(service.InvalidPost, match=r"copy\.channels\.twitter holds the placeholder \{retail_bags_sold\}"):
        service._validate_copy({"base": "Fine.", "channels": {"twitter": "We sold {retail_bags_sold} bags."}})
    with pytest.raises(service.InvalidPost, match=r"copy\.base holds the placeholder \{\{ name \}\}"):
        service._validate_copy({"base": "Hi {{ name }}"})
    with pytest.raises(service.InvalidPost, match=r"variables\.eyebrow is the word 'true'"):
        service._validate_variables({"eyebrow": {"value": "true", "claim": False}})
    with pytest.raises(service.InvalidPost, match=r"variables\.headline holds the placeholder \[growth_percent\]"):
        service._validate_variables({"headline": {"value": "Up [growth_percent]", "claim": False}})


def test_a_save_keeps_plain_words_and_real_switches():
    copy = {"base": "We sold 412 bags.", "channels": {"twitter": "412 bags. Thank you."}}
    assert service._validate_copy(copy) == copy
    variables = {"headline": {"value": "412 bags", "claim": True}, "show": {"value": True, "claim": False}}
    assert service._validate_variables(variables) == variables


class _Model:
    def __init__(self, answers):
        self.answers, self.asked = list(answers), []

    async def generate_response(self, messages, tools=None):
        self.asked.append(messages)
        answer = self.answers.pop(0)
        return SimpleNamespace(content=answer if isinstance(answer, str) else json.dumps(answer))


def _ctx(**overrides):
    base = dict(brief="Three numbers: 412 retail bags, 96 Harvest Club members.", format="image",
                channels=[{"toolkit": "twitter", "label": "X"}], templates=[CARD], candidates=[])
    return compose.ComposeContext(**{**base, **overrides})


def _answer(base):
    return {"title": "Our numbers", "format": "image", "template_id": "card", "variables": {"headline": "Our numbers"},
            "copy": {"base": base, "channels": {"twitter": base}}}


def _propose(model, ctx=None):
    return asyncio.run(compose.propose(ctx or _ctx(), lambda: model, 5.0))


def test_the_composer_asks_once_for_copy_holding_a_placeholder_with_the_reason():
    fixed = {"copy": {"base": "We sold 412 retail bags.", "channels": {"twitter": "412 retail bags sold."}}}
    model = _Model([_answer("We sold {retail_bags_sold} retail bags."), fixed])
    proposal = _propose(model)
    assert len(model.asked) == 2
    note, listed = model.asked[1][-1]["content"].split("\n", 1)
    assert note == compose.COPY_FIX_NOTE and "{retail_bags_sold}" in listed
    assert proposal["copy"] == fixed["copy"]
    assert proposal["questions"] == [] and not any("placeholder" in w for w in proposal["warnings"])


def test_a_placeholder_still_there_is_a_warning_and_an_owners_question():
    model = _Model([_answer("We sold {retail_bags_sold} retail bags."), "no idea"])
    proposal = _propose(model)
    assert len(model.asked) == 2  # asked once, never again
    assert "The copy holds the placeholder {retail_bags_sold}: write the real words before saving" in proposal["warnings"]
    assert proposal["questions"] == ["Retail bags sold"]


def test_the_model_is_told_the_skills_braces_never_reach_a_post():
    system = compose.build_messages(_ctx(skills={"social-template-payloads": "{ \"post_id\": \"{id}\" }"}))[0]["content"]
    assert compose.SKILLS_NOTE in system and system.index(compose.SKILLS_NOTE) < system.index("## Skill:")
    assert compose.SKILLS_NOTE not in compose.build_messages(_ctx())[0]["content"]
