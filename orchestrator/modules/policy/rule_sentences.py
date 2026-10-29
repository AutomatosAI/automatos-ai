"""PRD-268 §6.11.4, K16 — the rule that asked, said in words.

Before this table, the only names a rule had were machine text: the posture router's reason
(``"posture={p} routes {risk} to ask"``, ``modules/policy/gate.py``) and a board approval's raw
policy key. Each rule an approval can come from now has one plain sentence, in English and
Portuguese (Brazil):

* ``confirm.<risk class>``: the confirmation gate's card (``modules/tools/discovery/
  confirmation_gate.py``), by ``classify_action``'s risk class;
* ``posture.<posture>.<risk class>``: every posture × risk-class pair whose routing table sends it
  to ask (``PolicyDocument.route_for``);
* ``approval_policy.<policy>``: the Mission and board approval policy (``core/services/
  approval_policy.py``) when it holds the work: always_ask, auto_below_budget over the ceiling,
  full_auto without the autonomy gate.

``rule_sentence`` never returns machine text: an id with no sentence reads ``FALLBACK``'s, and a
language with no table reads English. Clients draw the sentence; they never compose it.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping, Tuple

from core.services.approval_policy import ALWAYS_ASK, AUTO_BELOW_BUDGET, FULL_AUTO
from modules.policy.policy_document import (
    BALANCED,
    PERMISSIVE,
    RISK_DESTRUCTIVE,
    RISK_EXTERNAL,
    RISK_INTERNAL_WRITE,
    RISK_PUBLISH,
    RISK_READ,
    STRICT,
)
from modules.proactive.locale import DEFAULT_LOCALE, ENGLISH, PORTUGUESE_BRAZIL

__all__ = [
    "FALLBACK", "RULE_IDS", "STRINGS", "approval_policy_rule", "confirm_rule", "known_rule", "posture_rule",
    "rule_sentence",
]

CONFIRM = "confirm"
POSTURE = "posture"
APPROVAL_POLICY = "approval_policy"
# The sentence an id without one reads: still words, never the id.
FALLBACK = "fallback"
_SEP = "."


def confirm_rule(risk_class: str) -> str:
    """The confirmation gate's rule id for a call of ``risk_class``."""
    return _SEP.join((CONFIRM, str(risk_class)))


def posture_rule(posture: str, risk_class: str) -> str:
    """The posture router's rule id for ``risk_class`` under ``posture``."""
    return _SEP.join((POSTURE, str(posture), str(risk_class)))


def approval_policy_rule(policy: str) -> str:
    """The approval policy's rule id for work it holds under ``policy``."""
    return _SEP.join((APPROVAL_POLICY, str(policy)))


# Sentences shared by the confirmation gate's rules and the postures that ask for the same class.
_EN_SENDS = "anything that sends to other people or acts in another app"
_EN_DELETES = "anything that deletes or can't be undone"
_EN_PUBLISHES = "anything that publishes"
_PT_SENDS = "qualquer coisa que envie para outras pessoas ou aja em outro app"
_PT_DELETES = "qualquer coisa que apague ou não possa ser desfeita"
_PT_PUBLISHES = "qualquer coisa que publique"

_EN = {
    confirm_rule(RISK_READ): "Ask before an action marked for confirmation, even one that only reads",
    confirm_rule(RISK_INTERNAL_WRITE): "Ask before a change in this workspace that is marked for confirmation",
    confirm_rule(RISK_EXTERNAL): f"Ask before {_EN_SENDS}",
    confirm_rule(RISK_DESTRUCTIVE): f"Ask before {_EN_DELETES}",
    confirm_rule(RISK_PUBLISH): f"Ask before {_EN_PUBLISHES}",
    posture_rule(BALANCED, RISK_PUBLISH): f"Balanced policy: ask before {_EN_PUBLISHES}",
    posture_rule(BALANCED, RISK_EXTERNAL): f"Balanced policy: ask before {_EN_SENDS}",
    posture_rule(BALANCED, RISK_DESTRUCTIVE): f"Balanced policy: ask before {_EN_DELETES}",
    posture_rule(STRICT, RISK_INTERNAL_WRITE): "Strict policy: ask before any change in this workspace",
    posture_rule(STRICT, RISK_PUBLISH): f"Strict policy: ask before {_EN_PUBLISHES}",
    posture_rule(STRICT, RISK_EXTERNAL): f"Strict policy: ask before {_EN_SENDS}",
    posture_rule(STRICT, RISK_DESTRUCTIVE): f"Strict policy: ask before {_EN_DELETES}",
    posture_rule(PERMISSIVE, RISK_DESTRUCTIVE): f"Permissive policy: ask only before {_EN_DELETES}",
    approval_policy_rule(ALWAYS_ASK): "Ask before any Mission or task runs",
    approval_policy_rule(AUTO_BELOW_BUDGET): "Ask before work that costs more than your approval ceiling",
    approval_policy_rule(FULL_AUTO): "Ask before work runs while full autonomy is off",
    FALLBACK: "Ask before this runs",
}
_PT_BR = {
    confirm_rule(RISK_READ): "Perguntar antes de uma ação marcada para confirmação, mesmo uma que só lê",
    confirm_rule(RISK_INTERNAL_WRITE): "Perguntar antes de uma mudança neste workspace marcada para confirmação",
    confirm_rule(RISK_EXTERNAL): f"Perguntar antes de {_PT_SENDS}",
    confirm_rule(RISK_DESTRUCTIVE): f"Perguntar antes de {_PT_DELETES}",
    confirm_rule(RISK_PUBLISH): f"Perguntar antes de {_PT_PUBLISHES}",
    posture_rule(BALANCED, RISK_PUBLISH): f"Política equilibrada: perguntar antes de {_PT_PUBLISHES}",
    posture_rule(BALANCED, RISK_EXTERNAL): f"Política equilibrada: perguntar antes de {_PT_SENDS}",
    posture_rule(BALANCED, RISK_DESTRUCTIVE): f"Política equilibrada: perguntar antes de {_PT_DELETES}",
    posture_rule(STRICT, RISK_INTERNAL_WRITE): "Política rígida: perguntar antes de qualquer mudança neste workspace",
    posture_rule(STRICT, RISK_PUBLISH): f"Política rígida: perguntar antes de {_PT_PUBLISHES}",
    posture_rule(STRICT, RISK_EXTERNAL): f"Política rígida: perguntar antes de {_PT_SENDS}",
    posture_rule(STRICT, RISK_DESTRUCTIVE): f"Política rígida: perguntar antes de {_PT_DELETES}",
    posture_rule(PERMISSIVE, RISK_DESTRUCTIVE): f"Política permissiva: perguntar só antes de {_PT_DELETES}",
    approval_policy_rule(ALWAYS_ASK): "Perguntar antes de qualquer Missão ou tarefa rodar",
    approval_policy_rule(AUTO_BELOW_BUDGET): "Perguntar antes de um trabalho que custe mais que seu teto de aprovação",
    approval_policy_rule(FULL_AUTO): "Perguntar antes de um trabalho rodar enquanto a autonomia total está desligada",
    FALLBACK: "Perguntar antes de isto rodar",
}

STRINGS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    ENGLISH: MappingProxyType(_EN),
    PORTUGUESE_BRAZIL: MappingProxyType(_PT_BR),
})
RULE_IDS: Tuple[str, ...] = tuple(key for key in _EN if key != FALLBACK)


def known_rule(rule_id: str) -> str:
    """``rule_id`` when the table holds it, else ``FALLBACK``: a client never sees an id it has no sentence for."""
    return rule_id if rule_id in STRINGS[DEFAULT_LOCALE] else FALLBACK


def rule_sentence(rule_id: str, language: str = DEFAULT_LOCALE) -> str:
    """The plain sentence for ``rule_id`` in ``language``: English when the language has no
    table, ``FALLBACK``'s sentence when the id has none. Never the id itself."""
    table = STRINGS.get(language, STRINGS[DEFAULT_LOCALE])
    return table.get(rule_id) or table[FALLBACK]
