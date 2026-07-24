"""PRD-187 S4 — L2→L3 promotion can finally fire (memory J4 / C.3-C.4, P2-06).

The old gate — ``importance > 0.7 AND access_count > 3`` — was a bootstrap
deadlock (promotion needs access → access needs recall → recall couldn't
match) over a population averaging 0.40–0.60 importance: ZERO promotions in
the table's lifetime was the forced output. These tests pin the new policy:

1. A high-importance distilled fact promotes WITHOUT any access-count
   requirement; a high-signal type (``user_fact``) promotes from the lower bar.
2. Dropping the access gate does NOT open the floodgate: noise types
   (``playbook_summary`` / ``heartbeat_log``) never promote, at any
   importance, and ordinary low-importance rows still don't clear the bar.
3. Field→durable promotion is a DIFFERENT policy and keeps its access gate
   (there, access_count is genuine usage).

Pure — the eligibility predicate is fed plain values; no DB.
"""
from __future__ import annotations

import os
import pathlib
import sys

import pytest

for _k, _v in {
    "POSTGRES_USER": "test", "POSTGRES_PASSWORD": "test",
    "POSTGRES_HOST": "localhost", "POSTGRES_PORT": "5432", "POSTGRES_DB": "test",
}.items():
    os.environ.setdefault(_k, _v)

_ORCH = pathlib.Path(__file__).resolve().parents[1]
if str(_ORCH) not in sys.path:
    sys.path.insert(0, str(_ORCH))

from modules.memory.promotion_policy import (  # noqa: E402
    high_signal_types,
    promotion_eligible,
)

_POLICY = dict(
    min_importance=0.7,
    high_signal_min_importance=0.5,
    high_signal=frozenset({"user_fact", "preference", "procedure"}),
)


# ---------------------------------------------------------------------------
# 1. Promotion fires on importance — no access-count gate
# ---------------------------------------------------------------------------

def test_promotion_fires_on_importance():
    # The exact row shape that could NEVER promote before: importance clears
    # the bar, access_count is irrelevant (there is no access argument at all).
    assert promotion_eligible("user_fact", 0.75, **_POLICY) is True
    assert promotion_eligible("exchange", 0.75, **_POLICY) is True


def test_high_signal_types_promote_from_lower_bar():
    # 0.55 < the 0.7 general bar, but user_fact/preference/procedure are what
    # durable memory exists for.
    assert promotion_eligible("user_fact", 0.55, **_POLICY) is True
    assert promotion_eligible("preference", 0.5, **_POLICY) is True
    assert promotion_eligible("procedure", 0.6, **_POLICY) is True
    # the same importance on an ordinary type does NOT clear the general bar
    assert promotion_eligible("exchange", 0.55, **_POLICY) is False


# ---------------------------------------------------------------------------
# 2. No floodgate: noise never promotes; low importance still gated
# ---------------------------------------------------------------------------

def test_promotion_still_excludes_noise():
    # Even at absurd importance, operational chatter never becomes durable —
    # had the old gate ever fired, failure spam would have been copied
    # verbatim into L3 (memory dossier §C.4).
    assert promotion_eligible("playbook_summary", 0.99, **_POLICY) is False
    assert promotion_eligible("heartbeat_log", 0.99, **_POLICY) is False
    assert promotion_eligible("recipe_summary", 0.99, **_POLICY) is False


def test_low_importance_still_not_promoted():
    assert promotion_eligible("exchange", 0.4, **_POLICY) is False
    assert promotion_eligible("user_fact", 0.3, **_POLICY) is False
    assert promotion_eligible(None, 0.9, **_POLICY) is True  # untyped, high importance
    assert promotion_eligible("exchange", None, **_POLICY) is False


def test_high_signal_types_config_parse():
    assert high_signal_types("user_fact, preference,procedure") == frozenset(
        {"user_fact", "preference", "procedure"}
    )
    assert high_signal_types("") == frozenset()


# ---------------------------------------------------------------------------
# 3. Field→durable promotion keeps its access gate
# ---------------------------------------------------------------------------

def test_field_promotion_keeps_access_gate():
    # The field job's gate is genuine usage of a live recall path — S4 changes
    # the L2→L3 policy ONLY. Assert the field job still reads its access-count
    # knob and the knob still exists in config.
    src = (_ORCH / "jobs" / "promote_field_memory.py").read_text()
    assert "FIELD_PROMOTION_MIN_ACCESS_COUNT" in src

    import re
    cfg_src = (_ORCH / "config.py").read_text()
    assert re.search(r"FIELD_PROMOTION_MIN_ACCESS_COUNT", cfg_src)
    # and the retired L2 gate is really gone from config
    assert "MEMORY_PROMOTION_MIN_ACCESS_COUNT" not in cfg_src
