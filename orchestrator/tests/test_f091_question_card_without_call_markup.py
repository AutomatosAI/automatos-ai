"""F091 (night 3) — a question card shows the question, not the tool call's markup.

A question card ended in raw "</question><options>…</invoke>": the asking model
wrote its tool call's own markup into the question text. Every ask goes through
``stage_question``, which now cuts the text at the first call-markup tag and
recovers the options that markup carried when none were passed separately.
"""
from __future__ import annotations

import inspect

from modules.tools.discovery import handlers_asks
from modules.tools.discovery.handlers_asks import clean_question


def test_the_markup_is_cut_and_its_options_recovered():
    assert clean_question(
        'Which wholesale terms should I use?</question><options>["30 days", "14 days"]</options></invoke>', None,
    ) == ("Which wholesale terms should I use?", ["30 days", "14 days"])
    assert clean_question("Allow `rm -rf build`?</question>\n<options>\n- Allow\n- Deny\n</options>\n</invoke>",
                          None) == ("Allow `rm -rf build`?", ["Allow", "Deny"])


def test_options_passed_separately_win_and_a_plain_question_is_untouched():
    assert clean_question('Which one?</question><options>["a"]</options></invoke>', ["x", "y"]) == (
        "Which one?", ["x", "y"])
    assert clean_question("Is **Tuesday** the delivery day?", None) == ("Is **Tuesday** the delivery day?", None)
    assert clean_question("", None) == ("", None)


def test_every_ask_is_cleaned_before_it_is_staged():
    source = inspect.getsource(handlers_asks.stage_question)
    assert source.index("clean_question(question, options)") < source.index("create_grant(")
