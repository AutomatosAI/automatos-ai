"""PRD-268 §6.11.3, K13 — what readiness says, from the strings table.

Server copy, never the model's: each item's label, each finding's detail, each fix's label
and the routes' refusals, in English and Portuguese (Brazil), chosen by the request's
``Accept-Language`` (``modules.proactive.locale.request_locale``). Clients draw these; they
never compose them. No sentence names a host or a latency: the local edition adds those
(``services.readiness.service``), SaaS never does.
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

from modules.proactive.locale import DEFAULT_LOCALE
from services.readiness import items as i

__all__ = ["CHECKED_RECENTLY", "NO_WORKSPACE", "STRINGS", "say"]

# The routes' refusals.
CHECKED_RECENTLY = "checked_recently"
NO_WORKSPACE = "no_workspace"

STRINGS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "en": MappingProxyType({
        i.CORE: "Automatos server",
        i.MODEL_ROUTE: "Auto's model",
        i.WEB_SEARCH: "Web search",
        i.PUSH: "Phone notifications",
        i.CORE_READY: "The server and its database answer.",
        i.CORE_DOWN: "The database didn't answer.",
        i.MODEL_READY: "Auto's model is chosen and has a key to run with.",
        i.MODEL_UNSET: "No model is chosen for Auto.",
        i.MODEL_NO_KEY: "Auto's model has no key to run with.",
        i.SEARCH_OPENROUTER: "Searches go through OpenRouter.",
        i.SEARCH_COMPOSIO: "Searches go through Composio Search.",
        i.SEARCH_SEARXNG: "Searches go through this install's SearXNG.",
        i.SEARCH_NONE: "No search engine is set up.",
        i.SEARCH_OFF: "Web search is switched off on this server.",
        i.PUSH_READY: "Notifications can reach your phone.",
        i.PUSH_UNANSWERED: "Nobody has said yet whether this install may send notifications through Expo.",
        i.PUSH_DECLINED: "This install was told not to send notifications through Expo.",
        i.NOT_ANSWERED: "Didn't answer in time.",
        i.UNREADABLE: "Couldn't be read. The server logged why.",
        i.CHOOSE_MODEL: "Choose Auto's model",
        i.ADD_KEY: "Add a key",
        i.ANSWER_PUSH: "Answer in Phones & devices",
        CHECKED_RECENTLY: "Checked a moment ago. Try again in a minute.",
        NO_WORKSPACE: "There's no workspace to check.",
    }),
    "pt-BR": MappingProxyType({
        i.CORE: "Servidor do Automatos",
        i.MODEL_ROUTE: "Modelo do Auto",
        i.WEB_SEARCH: "Busca na web",
        i.PUSH: "Notificações no celular",
        i.CORE_READY: "O servidor e o banco de dados respondem.",
        i.CORE_DOWN: "O banco de dados não respondeu.",
        i.MODEL_READY: "O modelo do Auto está escolhido e tem uma chave para funcionar.",
        i.MODEL_UNSET: "Nenhum modelo foi escolhido para o Auto.",
        i.MODEL_NO_KEY: "O modelo do Auto não tem uma chave para funcionar.",
        i.SEARCH_OPENROUTER: "As buscas passam pelo OpenRouter.",
        i.SEARCH_COMPOSIO: "As buscas passam pelo Composio Search.",
        i.SEARCH_SEARXNG: "As buscas passam pelo SearXNG desta instalação.",
        i.SEARCH_NONE: "Nenhum mecanismo de busca está configurado.",
        i.SEARCH_OFF: "A busca na web está desligada neste servidor.",
        i.PUSH_READY: "As notificações podem chegar ao seu celular.",
        i.PUSH_UNANSWERED: "Ninguém disse ainda se esta instalação pode enviar notificações pelo Expo.",
        i.PUSH_DECLINED: "Esta instalação foi instruída a não enviar notificações pelo Expo.",
        i.NOT_ANSWERED: "Não respondeu a tempo.",
        i.UNREADABLE: "Não foi possível ler. O servidor registrou o motivo.",
        i.CHOOSE_MODEL: "Escolher o modelo do Auto",
        i.ADD_KEY: "Adicionar uma chave",
        i.ANSWER_PUSH: "Responder à pergunta sobre o Expo",
        CHECKED_RECENTLY: "Verificado há pouco. Tente de novo em um minuto.",
        NO_WORKSPACE: "Não há um espaço de trabalho para verificar.",
    }),
})


def say(locale: str, key: str) -> str:
    """The sentence ``key`` in ``locale``, English when the locale has no table."""
    return STRINGS.get(locale, STRINGS[DEFAULT_LOCALE])[key]
