"""F460 (night 18): "Two things to remember, please: from November we post on Tuesdays. And Quay Coffee
House moves to 30-day terms from November." stored "From November we post on Tuesdays." as a memory,
not a rule: the lead-in counted as part of the sentence (6 of 11 words, under the 0.6 a rule keeps),
and the Thursday rule it replaced stayed, since only a standing-rule write retires one.

A lead-in that names several things heads a list, so the clause after its ':' is weighed on its own;
a lead-in with a condition, a negation or a scope never is (N17-4's lead-ins are unchanged).
"""
from __future__ import annotations

import pytest

from modules.tools.discovery.memory_write import STANDING_RULE_COVERAGE
from modules.tools.discovery.rule_wording import in_their_words, their_sentence
from tests import test_prd256_fix_one_number_one_rule as one_rule

TWO_THINGS = ("Two things to remember, please: from November we post on Tuesdays. And Quay Coffee House moves "
              "to 30-day terms from November.")
TUESDAYS = "From November we post on Tuesdays."
THURSDAYS = "Change of plan - from November we post on Thursdays, not Wednesdays. Replace that rule."

qdrant = one_rule.qdrant
owner_says = one_rule.owner_says


def test_the_clause_after_a_lists_lead_in_is_the_owners_sentence():
    assert in_their_words(TUESDAYS, [TWO_THINGS], STANDING_RULE_COVERAGE) is True
    assert their_sentence(TUESDAYS, [TWO_THINGS], STANDING_RULE_COVERAGE) == TUESDAYS          # verbatim, as stored


@pytest.mark.parametrize("said", [
    "Only if Nadia agrees: post on Tuesdays",                                   # a condition
    "Two things, only if Nadia agrees: post on Tuesdays.",
    "Two things we never do: post on Tuesdays.",                                # a negation
    "Two rules for the Falmouth shop: post on Tuesdays. Close at five.",        # a scope
    "For the Falmouth shop: post on Tuesdays.",                                 # N17-4: no list, kept whole
])
def test_a_lead_in_that_changes_what_the_rule_says_stays_in_it(said):
    assert in_their_words("Post on Tuesdays.", [said], STANDING_RULE_COVERAGE) is False


def test_storing_it_through_the_memory_write_makes_it_a_rule_and_retires_thursdays(qdrant, owner_says):
    owner_says.append(THURSDAYS)
    assert one_rule._store(THURSDAYS)["standing_rule"] is True
    owner_says.insert(0, TWO_THINGS)                                            # newest first

    answer = one_rule._store(TUESDAYS, replaces=THURSDAYS)

    assert answer["success"] is True and answer["standing_rule"] is True and answer["retired"] == [THURSDAYS]
    assert one_rule._rules() == [TUESDAYS]
