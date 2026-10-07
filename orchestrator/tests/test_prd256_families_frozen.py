"""PRD-256 US-002 (D10) — the claim families are frozen.

What was not done is said from the turn's receipts now (``consumers/chatbot/receipts.py``); the
regex families only drive the in-loop nudge until Wave 2 US-012 deletes them, once the receipts
rule is proven. Until then they may not grow: no new family, no new pattern. This pins today's
counts (7 Oct 2026). A change that needs one more pattern is a change to the receipts rule.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

EXECUTION = Path(__file__).resolve().parents[1] / "modules" / "tools" / "execution"
# The families' entries (``_Family(``, and document_claims' ``family(`` helper) and their patterns.
COUNTS = {
    "action_claims.py": {"_Family(": 23, "family(": 0, "re.compile(": 41},
    "document_claims.py": {"_Family(": 1, "family(": 13, "re.compile(": 16},
    "shop_and_team_claims.py": {"_Family(": 0, "family(": 0, "re.compile(": 14},
    "social_post_claims.py": {"_Family(": 6, "family(": 0, "re.compile(": 6},
}
_COUNTED = {"_Family(": re.compile(r"_Family\("), "family(": re.compile(r"(?<![\w])family\("),
            "re.compile(": re.compile(r"re\.compile\(")}


@pytest.mark.parametrize("module", sorted(COUNTS))
def test_prd256_families_frozen(module):
    text = (EXECUTION / module).read_text(encoding="utf-8")
    counted = {what: len(pattern.findall(text)) for what, pattern in _COUNTED.items()}
    assert counted == COUNTS[module], f"{module} grew: the families are frozen (PRD-256 D10)"


def test_no_new_claim_family_module():
    assert sorted(path.name for path in EXECUTION.glob("*_claims.py")) == sorted(COUNTS)
