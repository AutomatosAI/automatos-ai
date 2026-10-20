"""F460 (night 18): Auto spoke of the owner in the third person, to the owner.

"Ask Gerard whether…", "He hasn't confirmed", "needs Gerard's decision", all said to Gerard. The
prompt's only line about who Auto talks to was "You're talking to Gerard", while the documents and
the cards name him. Both lanes' identity clause
(``personality.get_base_system_prompt`` and ``service.atom_identity_clause``) now say that the person
those name is the one Auto is talking to, by name only: "the owner" is someone else in a team
member's chat. With no name known, neither prompt changes.
"""
from __future__ import annotations

# By name only: "the owner" is the person talking only when they own the workspace, and a team member's
# chat is not the owner's (FIXER review).
OWNER_IS_YOU = ("When a document, card or tool note names {name}, that is the person you are talking to: say "
                "'you' and 'your', never '{name}' or a third-person pronoun, except inside a draft addressed to "
                "someone else.")


def owner_is_you(name: str) -> str:
    """The clause for a known ``name``: the person the documents name is the one Auto talks to."""
    return OWNER_IS_YOU.format(name=name)


__all__ = ["OWNER_IS_YOU", "owner_is_you"]
