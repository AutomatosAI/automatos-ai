"""PRD-268 §6.11.1, K13 — what a row's next step says, from the strings table.

Server copy, never the model's: each needs-you kind and each reason a ticket is stuck
(the ``STUCK_*`` constants of ``services.needs_you``) has one sentence, in English and
Portuguese (Brazil), chosen by the request's ``Accept-Language``
(``modules.proactive.locale.request_locale``). Clients draw the sentence; they never
compose it. Every sentence is at most ``NEXT_STEP_CHARS`` long, so it fits on the line under
a row's status.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from modules.proactive.locale import DEFAULT_LOCALE
from services.needs_you import (
    STUCK_MISSION_ENDED,
    STUCK_NO_AGENT,
    STUCK_NO_HOST,
    STUCK_NOT_PICKED_UP,
    STUCK_OUT_OF_CREDIT,
    STUCK_OVER_BUDGET,
    STUCK_STEP_FAILED,
)

__all__ = ["APPROVAL_PLAN", "FAILED_MISSION", "NEXT_STEP_CHARS", "STRINGS", "STUCK_REASONS", "say"]

# The longest next step a row carries (§6.11.1).
NEXT_STEP_CHARS = 90
# Every reason a stuck ticket gives (services/needs_you.py), each with its own sentence.
STUCK_REASONS = (STUCK_NO_HOST, STUCK_NO_AGENT, STUCK_NOT_PICKED_UP, STUCK_MISSION_ENDED,
                 STUCK_STEP_FAILED, STUCK_OVER_BUDGET, STUCK_OUT_OF_CREDIT)
# A Mission waiting for its plan's approval, and a Mission's own card that failed: their
# kind's sentence would tell the person to allow an action or rerun a ticket.
APPROVAL_PLAN = "approval_plan"
FAILED_MISSION = "failed_mission"

STRINGS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "en": MappingProxyType({
        "review": "Look over the work: approve it, or send it back with what to change.",
        "question": "Answer the question so the work can go on.",
        "approval": "Allow the action it asks for, or refuse.",
        APPROVAL_PLAN: "Approve the Mission's plan, or reject it.",
        "stuck": "Nothing moves this until you do: open it to see why.",
        "failed": "It couldn't finish: try again, or close it if it's no longer needed.",
        FAILED_MISSION: "The Mission couldn't finish: replan it to try again, or let it go.",
        STUCK_NO_HOST: "No computer that runs this agent's CLI is online: start one, or pick another agent.",
        STUCK_NO_AGENT: "Assign an agent so the work can start.",
        STUCK_NOT_PICKED_UP: "The board never runs a Playbook's card: run the Playbook again, or close it.",
        STUCK_MISSION_ENDED: "Its Mission ended without it: open the Mission to resume it, or let it go.",
        STUCK_STEP_FAILED: "It failed its Mission's check: open the Mission to retry it, or let it go.",
        STUCK_OVER_BUDGET: "The Mission paused at its budget: raise the budget, then resume it.",
        STUCK_OUT_OF_CREDIT: "The Mission paused when the AI credit ran out: top up, then resume it.",
    }),
    "pt-BR": MappingProxyType({
        "review": "Revise o trabalho: aprove, ou devolva dizendo o que mudar.",
        "question": "Responda à pergunta para o trabalho continuar.",
        "approval": "Autorize a ação pedida, ou recuse.",
        APPROVAL_PLAN: "Aprove o plano da Missão, ou rejeite.",
        "stuck": "Nada anda até você agir: abra para ver o motivo.",
        "failed": "Não deu para terminar: tente de novo, ou feche se não precisar mais.",
        FAILED_MISSION: "A Missão não terminou: replaneje para tentar de novo, ou deixe para lá.",
        STUCK_NO_HOST: "Nenhum computador que roda a CLI deste agente está online: ligue um, ou troque o agente.",
        STUCK_NO_AGENT: "Atribua um agente para o trabalho começar.",
        STUCK_NOT_PICKED_UP: "O quadro nunca roda o cartão de um Playbook: rode o Playbook de novo, ou feche.",
        STUCK_MISSION_ENDED: "A Missão terminou sem ele: abra a Missão para retomar, ou deixe para lá.",
        STUCK_STEP_FAILED: "Falhou na verificação da Missão: abra a Missão para tentar de novo, ou deixe para lá.",
        STUCK_OVER_BUDGET: "A Missão pausou no orçamento: aumente o orçamento e retome.",
        STUCK_OUT_OF_CREDIT: "A Missão pausou quando o crédito de IA acabou: recarregue e retome.",
    }),
})


def say(locale: str, key: str) -> str:
    """The sentence ``key`` in ``locale``, English when the locale has no table."""
    return STRINGS.get(locale, STRINGS[DEFAULT_LOCALE])[key]
