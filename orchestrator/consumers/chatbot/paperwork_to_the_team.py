"""F337(c) (night 10): customer paperwork asked for in plain English goes to the team.

Night 10, 37 documents each way: the ones the owner's agents made on tickets graded 4 or
better 57% of the time, the ones Auto wrote itself in chat 8%. "Auto, get the team to do it"
already routed each in 5-7 seconds, and the same agent made the same quality whether the
card came from the owner or from Auto. Night 10b: given a named template and its fields
(F351-c), Auto made 6 of 7 right first time.

So the owner's call (5 Oct): a template named, Auto fills it now (``named_template_note``);
a letter, invoice, quote, flyer or the like asked for with no template named goes to the
agent whose job it is. The turn gets a note saying so: recommend the agent, file the card
with every fact the owner gave and start it, say who has it, and offer to make it on one of
their templates instead. Asking Auto to do it "yourself" keeps it with Auto. Only the
owner's latest message is read, and only to tell a request for paperwork from a question
about it ("did Rosa pay the invoice?" asks nothing to be made).
"""
from __future__ import annotations

import re
from typing import Optional

# What paperwork for a customer is called, and the asking that makes one.
_PAPER = (r"(?P<kind>letters?|invoices?|quotes?|quotations?|estimates?|proposals?|agreements?|contracts?|"
          r"flyers?|leaflets?|posters?|brochures?|price ?lists?|welcome sheets?|receipts?|newsletters?|"
          r"one-pagers?|menus?)")
_MAKE = (r"\b(?:make|write|draft|create|prepare|produce|generate|design|put together|draw up|knock up|mock up|"
         r"do (?:me |us )?(?:a|an|the))\b")
_ASKS_FOR = re.compile(_MAKE + r"[^.!?\n]{0,80}?\b" + _PAPER + r"\b", re.I)
# The owner keeps it with Auto: "can you do it yourself", "don't bother the team".
_AUTO_ITSELF = re.compile(r"\byourself\b|\b(?:don['’]?t|do not|no need to)\s+(?:bother|involve|use|ask)\s+the\s+team\b"
                          r"|\bwithout the team\b", re.I)

VOWELS = "aeiou"
TEAM_NOTE = (
    "The owner asked for {article} {kind} for their customers and named none of their templates. Customer paperwork "
    "goes to the team (the owner's choice): don't write it yourself in this reply. Call platform_recommend_agent "
    "with the request (platform_list_agents is the plain roster), then file it for the agent whose job it is: "
    "platform_create_task with assigned_agent_name and a self-contained description, written as the dispatch "
    "contract below, that carries every name, address, figure, date and term exactly as the owner gave them and "
    "invents none. Start it (platform_update_task_status to 'in_progress'). Tell the owner in one line who has it "
    "and its card number, and offer to make it yourself now on one of their templates if they'd rather."
)


def asks_for_paperwork(text: object) -> Optional[str]:
    """The kind of customer paperwork ``text`` asks to be made ("price list"), or None: a
    question about one, a message that keeps it with Auto, or nothing of the kind."""
    said = str(text or "")
    if _AUTO_ITSELF.search(said):
        return None
    found = _ASKS_FOR.search(said)
    return " ".join(found.group("kind").lower().split()) if found else None


def team_note(text: object) -> Optional[str]:
    """The hand-to-the-team note for the owner's latest message, or None when it asks for no
    paperwork. The ASSIGN lane's dispatch contract and board rule go with it."""
    kind = asks_for_paperwork(text)
    if kind is None:
        return None
    from consumers.chatbot.auto import BOARD_MOVES_THE_CARD
    from modules.coordination.dispatch_contract import DISPATCH_CONTRACT_FRAGMENT

    article = "an" if kind[:1] in VOWELS else "a"
    return f"{TEAM_NOTE.format(article=article, kind=kind)}\n\n{DISPATCH_CONTRACT_FRAGMENT}\n{BOARD_MOVES_THE_CARD}"


__all__ = ["TEAM_NOTE", "asks_for_paperwork", "team_note"]
