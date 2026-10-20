"""P256-FIX-N17-5 (F441, nights 16–17): a write that went through is never denied by the not-done line.

Night 17 idx 217: "Done. `harbourline-ruling-2026-10-12.md` now says … Just to be clear: I haven't replaced
anything" over a platform_upload_document that was done; night 16 idx 352 the same over two done
platform_store_memory calls ("I've replaced the Wednesday rule"). The update family (claims_backed.FAMILIES)
was backed only by update_/configure_/set_/edit_/rename_/patch_/revert_: not by an upload (a same-name upload
replaces the document, F087) nor by a memory stored with ``replaces``.

Now the update family takes upload_, and "replaced" is its own family, backed by a memory or any of those writes.
"changed" stays the update family's alone: a stored memory backs no "I've changed Scout's model." (whether it
should, night 16 idx 111, is Gerard's call), and A697's "reverted" over a memory still draws the line.
"""
from __future__ import annotations

from consumers.chatbot.claim_removed import without_unbacked_claims
from consumers.chatbot.claims_backed import FAMILIES
from tests.helpers_receipts_rule import call, tracker_of

UPLOADED = call("platform_upload_document", {"filename": "harbourline-ruling-2026-10-12.md"},
                {"success": True, "document_id": 81, "filename": "harbourline-ruling-2026-10-12.md"})
RULE_STORED = call("platform_store_memory", {"content": "From November we post on Wednesdays.", "standing_rule": True})
PLAYBOOK_REFUSED = call("platform_create_playbook", {"name": "Heartbeat"}, {"success": False, "error": "steps must be a list"})


def _removed(answer, *calls):
    """The reply as the owner gets it in Auto's voice; None when nothing is taken out."""
    removed = without_unbacked_claims(answer, tracker_of(calls).outcomes, True)
    return None if removed is None else removed.text


def test_the_idx_217_reply_over_a_read_and_a_done_upload_is_left_alone():
    said = ("Done. `harbourline-ruling-2026-10-12.md` now says the Wednesday van ruling stands. I've replaced the "
            "old copy with it.")
    assert _removed(said, "platform_list_documents", UPLOADED) is None


def test_an_upload_backs_updated():
    assert _removed("I've updated the rulings file.", UPLOADED) is None


def test_two_stored_rules_back_replaced():
    assert _removed("I've replaced the Wednesday rule.", RULE_STORED, RULE_STORED) is None


def test_a_stored_memory_still_backs_no_change():
    assert _removed("I've changed Scout's model.", RULE_STORED) == (
        "Just to be clear: I haven't changed anything in this reply. Ask me again if you want it done.")


def test_a697_reverted_over_a_memory_still_draws_the_line():
    said = "I've saved your preference. I've reverted the heartbeat to every 30 minutes."
    assert _removed(said, PLAYBOOK_REFUSED, RULE_STORED) == (
        "I've saved your preference. Just to be clear: I haven't reverted anything in this reply; I tried to create "
        "the playbook \"Heartbeat\" and it didn't go through. Ask me again if you want it done.")


def test_replaced_is_its_own_family_and_the_update_family_takes_an_upload():
    update = next(backs for verbs, backs in FAMILIES if "updated" in verbs)
    replaced = next((verbs, backs) for verbs, backs in FAMILIES if "replaced" in verbs)
    upload = {"action": "platform_upload_document", "kind": "write", "status": "done"}
    memory = {"action": "platform_store_memory", "kind": "write", "status": "done"}
    assert replaced[0] == frozenset({"replaced"})
    assert update(upload) and not update(memory)
    assert replaced[1](upload) and replaced[1](memory)
