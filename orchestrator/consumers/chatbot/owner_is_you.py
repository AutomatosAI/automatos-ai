"""F460 (night 18): Auto spoke of the owner in the third person, to the owner.

"Ask Gerard whether…", "He hasn't confirmed", "needs Gerard's decision", all said to Gerard. The
prompt's only line about who Auto talks to was "You're talking to Gerard", while the documents, the
cards and the tools' notes name him or call him "the owner". Both lanes' identity clause
(``personality.get_base_system_prompt`` and ``service.atom_identity_clause``) now say that the person
those name is the one Auto is talking to. With no name known, neither prompt changes.
"""
from __future__ import annotations

OWNER_IS_YOU = ("When a document, card or tool note names {name} or 'the owner', that is the person you are "
                "talking to: say 'you' and 'your', never '{name}', 'he' or 'the owner', except inside a draft "
                "addressed to someone else.")


def owner_is_you(name: str) -> str:
    """The clause for a known ``name``: the person the documents name is the one Auto talks to."""
    return OWNER_IS_YOU.format(name=name)


__all__ = ["OWNER_IS_YOU", "owner_is_you"]
