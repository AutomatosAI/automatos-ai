"""F365 (a) (night 10c) — a proposal its agent calls a probe never reaches the owner.

On #2141 ("more space between sections") the Brand Designer filed question #1788 with
the whole reason "VALIDATION PROBE ONLY" and ``spacing_unit_pt`` 4 → 6, a change its
own notes did not propose (they said 4 → 8): it was checking whether the kit took the
value. The owner got it as the real proposal ("I don't know what I'm approving").

``platform_propose_brand_kit`` has no test mode. The kit's checks already run before
any card is filed (a proposal the kit refuses asks nothing), so a probe is never
needed, and a card whose reason says it is not real is refused: nothing is drawn,
written or asked. The tool's description says so, in both its forms.
"""
from __future__ import annotations

import pytest

from modules.tools.discovery import brand_proposal_card as card
from services.session_work_tools import PROPOSE_BRAND_KIT_SPEC
from tests import test_prd255w2_designer_card as designer

filed = designer.filed      # the board render, the worker and the card filing, faked
_Db, _Worker, _propose, _session = designer._Db, designer._Worker, designer._propose, designer._session
SPACING_PROBE = {"spacing_unit_pt": 6}


@pytest.mark.parametrize("why", [
    "VALIDATION PROBE ONLY",
    "Dry run: checking the field is accepted.",
    "Just testing whether the kit takes 6pt.",
    "Test card, please ignore this.",
    "Don’t approve — not a real proposal.",
])
def test_a_card_whose_reason_says_it_is_not_real_is_never_filed(filed, why):
    db = _Db()
    answer = _propose(_session(brand_kit=SPACING_PROBE, why=why), db)

    assert answer["success"] is False and "there is no test or probe mode" in answer["error"]
    assert "Nothing was asked" in answer["error"]
    assert filed["session"] == [] and filed["board_card"] == [] and filed["board"] == []
    assert _Worker.written == {} and db.workspace.settings == designer.STORED


def test_1788s_reason_is_named_back_to_the_agent(filed):
    answer = _propose(_session(brand_kit=SPACING_PROBE, why="VALIDATION PROBE ONLY"))
    assert "'VALIDATION PROBE'" in answer["error"]


@pytest.mark.parametrize("why", [
    "More room between sections: every gap doubles, so the letter runs to two pages.",
    "The accent passes the contrast test on white and on sand.",
    "Replaces the placeholder sign-off with your own name.",
    "",
])
def test_a_real_reason_still_files_the_card(filed, why):
    answer = _propose(_session(brand_kit=SPACING_PROBE, why=why))

    assert answer["success"] is True and answer["ask_id"] == 901
    assert len(filed["session"]) == 1 and "`spacing_unit_pt`" in filed["session"][0]["question"]


def test_the_tool_says_it_has_no_test_mode():
    from modules.tools.discovery.action_registry import ActionRegistry
    from modules.tools.discovery.actions_brand_proposals import register_brand_proposal_actions

    registry = ActionRegistry()
    register_brand_proposal_actions(registry)
    action = registry._actions["platform_propose_brand_kit"]     # no lazy load of every other action
    assert "there is no test mode" in action.description
    assert "there is no test mode" in PROPOSE_BRAND_KIT_SPEC["description"]


def test_the_check_is_on_the_reason_alone():
    assert card.not_a_real_proposal("VALIDATION PROBE ONLY") == "VALIDATION PROBE"
    assert card.not_a_real_proposal(None) is None
    assert card.not_a_real_proposal("Warmer: the logo's clay, as the accent for highlights only.") is None
