"""F055 — a playbook step never hands an agent an unfilled placeholder.

Night 2, ticket #387: step 1 of a playbook reached its agent reading "List every
number, date, price … and tasting note in {input}". The engine knew only
``{input.<field>}``; the bare ``{input}`` people naturally write passed through
verbatim. The agent refused to invent a document and asked — the right call,
but it should never have been given a template variable to puzzle over.
"""
from __future__ import annotations

from api.recipe_executor import (
    fill_step_placeholders,
    substitute_playbook_input,
    unresolved_input_placeholders,
)

TICKET_387 = ("List every number, date, price, address, name, founding year, origin, "
              "altitude, varietal and tasting note in {input}. One line each.")


def test_bare_input_is_the_whole_input_when_there_is_one_field():
    out = substitute_playbook_input(TICKET_387, {"document": "Harbourline price list …"})
    assert "{input}" not in out and "Harbourline price list" in out


def test_bare_input_prefers_the_content_of_a_trigger_or_upload():
    out = substitute_playbook_input("Summarise {input}", {"content": "the body", "source": "gmail"})
    assert out == "Summarise the body"


def test_bare_input_falls_back_to_every_field_as_lines():
    out = substitute_playbook_input("Use {input}", {"a": 1, "b": 2})
    assert out == "Use a: 1\nb: 2"


def test_named_fields_still_work_and_win_over_the_bare_form():
    out = substitute_playbook_input("{input.city} then {input}", {"city": "Porto"})
    assert out == "Porto then Porto"


def test_a_run_with_no_input_leaves_the_placeholder_to_be_caught_not_invented():
    out = substitute_playbook_input(TICKET_387, None)
    assert unresolved_input_placeholders(out) == ["{input}"]
    assert unresolved_input_placeholders(substitute_playbook_input("{input.city}", {"x": 1})) == ["{input.city}"]


def test_braces_that_are_not_input_variables_are_the_authors_text():
    text = 'Return JSON like {"name": "x"} and keep {previous_output} for later'
    assert substitute_playbook_input(text, {"a": 1}) == text
    assert unresolved_input_placeholders(text) == []


def test_the_step_resolver_understands_bare_input_too():
    assert "{input}" not in fill_step_placeholders(TICKET_387, {"doc": "the doc"})
