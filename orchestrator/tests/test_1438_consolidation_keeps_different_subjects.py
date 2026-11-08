"""#1438: consolidation never deletes a true fact about someone else phrased the same way.

``is_contradiction`` called two facts a contradiction when they shared most words and
each had a word the other lacked. Facts written to one template ("The owner of Bloom is
Mia Chen" / "The owner of Initech is Aiko Hooli") pass both, so the hourly consolidation
superseded and deleted the older: in #1437's eval, 332 of 414 distinct facts.

A contradiction is "a different value for a shared slot" (the module's own words): word
by word, the two facts differ in exactly one place where both sides say something
different. A word added on one side only ("…now") is elaboration, not a second slot.
Different subjects differ in several places and are both kept.
"""
from __future__ import annotations

import pytest

from modules.memory.operations.contradiction import is_contradiction, plan_consolidation


def _m(id_, text, created):
    return {"id": id_, "memory": text, "importance": 0.5, "created_at": created, "metadata": {"tier": "l3"}}


DIFFERENT_SUBJECTS = [
    ("The owner of Bloom is Mia Chen. Mia handles all wedding enquiries personally and wants them flagged immediately.",
     "The owner of Initech is Aiko Hooli. Aiko handles all wedding enquiries personally and wants them flagged immediately."),
    ("Acme Analytics is billed annually on the enterprise plan; the subscription renews every year on March 1st.",
     "Initech Aiko is billed annually on the enterprise plan; the subscription renews every year on March 3rd."),
    ("Northstar has 9 consultants: 3 partners, 2 senior consultants, and 4 analysts.",
     "Initech has 19 consultants: 7 partners, 5 senior consultants, and 9 analysts."),
    ("The account manager for the Leeds office is Jory Hobb, who joined from the Bristol team last spring.",
     "The account manager for the Glasgow office is Priya Nair, who joined from the Bristol team last spring."),
    # one slot, but the subject's: it opens the sentence, or owns what follows
    ("Acme does not want any AI-written content published externally without a human review.",
     "Initech does not want any AI-written content published externally without a human review."),
    ("The Midnight Garden arrangement is only available from October to December.",
     "The Harvest Moon arrangement is only available from October to December."),
    ("Bloom's brand voice avoids the word cheap; use affordable or great value instead.",
     "Initech's brand voice avoids the word cheap; use affordable or great value instead."),
    ("When preparing Northstar's client decks, redact any client name other than the recipient.",
     "When preparing Initech's client decks, redact any client name other than the recipient."),
]

ONE_SLOT_CONTRADICTIONS = [
    ("The deploy target is the staging cluster.", "The deploy target is the production cluster now."),
    ("Primary on-call is Alice for releases.", "Primary on-call is Bob for releases now."),
    ("The primary contact at Acme is Dana Ruiz.", "The primary contact at Acme is Lena Fox."),
    ("Acme's subscription renews every year on March 1st.", "Acme's subscription renews every year on June 1st."),
    ("Decided to keep invoicing in Xero rather than migrate.", "Decided to keep invoicing in QuickBooks rather than migrate."),
    ("The courier collects weekday parcels at 4pm from the warehouse.",
     "The courier collects weekday parcels at 6pm from the warehouse."),
]

# Where the rule cannot tell a value from a subject, it keeps both: a stale fact stays rather
# than a true one being deleted. Pinned so a change here is a decision, not an accident.
KEPT_WHEN_UNSURE = [
    ("Alice is the primary on-call for releases.", "Bob is the primary on-call for releases."),       # opens with the value
    ("Acme renews annually on March 1st.", "Acme renews monthly on the 1st."),                        # two slots changed
]


@pytest.mark.parametrize("older,newer", DIFFERENT_SUBJECTS)
def test_same_shaped_facts_about_different_subjects_are_not_a_contradiction(older, newer):
    assert not is_contradiction(_m("a", older, "2026-11-01"), _m("b", newer, "2026-11-02"))


@pytest.mark.parametrize("older,newer", DIFFERENT_SUBJECTS)
def test_consolidation_keeps_both(older, newer):
    plan = plan_consolidation([_m("a", older, "2026-11-01"), _m("b", newer, "2026-11-02")])
    assert plan.supersessions == [] and plan.merges == []


@pytest.mark.parametrize("older,newer", KEPT_WHEN_UNSURE)
def test_when_the_rule_cannot_tell_it_keeps_both(older, newer):
    assert plan_consolidation([_m("a", older, "2026-11-01"), _m("b", newer, "2026-11-02")]).supersessions == []


@pytest.mark.parametrize("older,newer", ONE_SLOT_CONTRADICTIONS)
def test_a_new_value_for_one_slot_still_supersedes_the_old(older, newer):
    plan = plan_consolidation([_m("a", older, "2026-11-01"), _m("b", newer, "2026-11-02")])
    assert [(s.winner["id"], s.loser["id"]) for s in plan.supersessions] == [("b", "a")]


def test_a_mixed_workspace_loses_only_the_superseded_value():
    mems = [
        _m("1", "The owner of Bloom is Mia Chen.", "2026-11-01"),
        _m("2", "The owner of Initech is Aiko Hooli.", "2026-11-02"),
        _m("3", "The deploy target is the staging cluster.", "2026-11-01"),
        _m("4", "The deploy target is the production cluster now.", "2026-11-03"),
    ]
    plan = plan_consolidation(mems)
    assert [s.loser["id"] for s in plan.supersessions] == ["3"]
