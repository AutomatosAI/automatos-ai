"""F182 (CodeQL py/polynomial-redos) — an owner's answer line is read in linear time.

inputs_from_answer matched each "name: value" line with a lazy value before a
trailing \\s*$, which backtracked quadratically on a long run of spaces inside a
value: an answer of a few tens of kilobytes held the event loop for seconds. The
value is now the rest of the line, stripped in code.
"""
from __future__ import annotations

import time

from core.services.playbook_inputs import inputs_from_answer

CONTRACT = {"cafe_name": {"required": True}, "hours": {"required": True}}
LONG_RUN = " " * 30_000


def test_a_long_run_of_spaces_in_a_value_is_read_at_once():
    started = time.perf_counter()
    values = inputs_from_answer(f"cafe_name: Harbour{LONG_RUN}Line", ["cafe_name"], CONTRACT)
    assert time.perf_counter() - started < 0.5
    assert values == {"cafe_name": f"Harbour{LONG_RUN}Line"}


def test_each_line_still_gives_its_value_stripped():
    answer = "- cafe_name:  Harbourline Café  \nhours = 8am to 4pm"
    assert inputs_from_answer(answer, ["cafe_name", "hours"], CONTRACT) == {
        "cafe_name": "Harbourline Café", "hours": "8am to 4pm"}
