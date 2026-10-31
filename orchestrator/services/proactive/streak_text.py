"""PRD-268 §6.2.2, K13 — what a failing streak's card says, from the strings table.

The card is the platform's, never the model's: its title and detail are built from
structured fields only (the agent's name, the error class, the count, when the streak
began and the intervals). Server copy comes from ``STRINGS`` in English and Portuguese
(Brazil); a card is written in the background, so in English (K13).

* The title says who fails and why: "VECTOR's check-ins fail: the AI provider refuses its
  key (HTTP 401)".
* The detail says since when and how often, that no ticket was filed, from the third
  failure the slower interval ("It now checks in every 2 hours instead of every hour"), and
  what Start does.
* When a check-in works again, the detail says so ("VECTOR's check-ins work again").
"""
from __future__ import annotations

from datetime import datetime, timezone
from types import MappingProxyType
from typing import Mapping, Optional

from core.models.proactive import DETAIL_LENGTH, ERROR_CLASS_LENGTH, TITLE_LENGTH
from modules.proactive.streaks import DOUBLING_FROM
from modules.tools.discovery.suggestion_fields import plain_text
from services.proactive.failures import OUT_OF_CREDIT, TIMEOUT, http_status
from services.proactive.loop import COST_CEILING

__all__ = [
    "BACKGROUND_LOCALE",
    "STRINGS",
    "duration",
    "streak_detail",
    "streak_title",
    "works_again_detail",
]

BACKGROUND_LOCALE = "en"
# The agent's name, as far as a title has room for it.
NAME_CHARS = 40
# How a streak's start is written: the same in every locale, and in UTC.
SINCE_FORMAT = "%Y-%m-%d %H:%M"
MINUTES_PER_HOUR = 60
# The provider refused the key itself: unauthorised or forbidden.
KEY_STATUSES = frozenset({"401", "403"})

STRINGS: Mapping[str, Mapping[str, str]] = MappingProxyType({
    "en": MappingProxyType({
        "title": "{name}'s check-ins fail: {reason}",
        "title_bare": "{name}'s check-ins fail ({error})",
        "reason_key": "the AI provider refuses its key (HTTP {status})",
        "reason_status": "the AI provider refuses the request (HTTP {status})",
        "reason_credit": "the AI credit ran out",
        "reason_timeout": "they time out",
        "reason_cost": "they pass their cost ceiling",
        "failing_one": "Its check-in at {since} UTC failed ({error}). No ticket was filed.",
        "failing_many": "Since {since} UTC, its last {count} check-ins failed ({error}). No ticket was filed for them.",
        "slower": "It now checks in every {interval} instead of every {base}, until a check-in works.",
        "start": "Start gives Auto a ticket to find out why and fix it.",
        "works_again_one": "{name}'s check-ins work again, after one failure ({error}).",
        "works_again_many": "{name}'s check-ins work again, after {count} failures in a row ({error}).",
        "hour": "hour",
        "hours": "{n} hours",
        "minutes": "{n} minutes",
    }),
    "pt-BR": MappingProxyType({
        "title": "Os check-ins de {name} falham: {reason}",
        "title_bare": "Os check-ins de {name} falham ({error})",
        "reason_key": "o provedor de IA recusa a chave (HTTP {status})",
        "reason_status": "o provedor de IA recusa o pedido (HTTP {status})",
        "reason_credit": "o crédito de IA acabou",
        "reason_timeout": "o tempo se esgota",
        "reason_cost": "passam do teto de custo",
        "failing_one": "O check-in das {since} UTC falhou ({error}). Nenhum ticket foi criado.",
        "failing_many": ("Desde {since} UTC, os últimos {count} check-ins falharam ({error}). "
                         "Nenhum ticket foi criado para eles."),
        "slower": "Agora o check-in é feito a cada {interval}, em vez de a cada {base}, até um check-in funcionar.",
        "start": "Iniciar dá ao Auto um ticket para descobrir o motivo e corrigir.",
        "works_again_one": "Os check-ins de {name} funcionam de novo, depois de uma falha ({error}).",
        "works_again_many": "Os check-ins de {name} funcionam de novo, depois de {count} falhas seguidas ({error}).",
        "hour": "hora",
        "hours": "{n} horas",
        "minutes": "{n} minutos",
    }),
})


def _table(locale: str) -> Mapping[str, str]:
    return STRINGS.get(locale, STRINGS[BACKGROUND_LOCALE])


def _name(name: str) -> str:
    """The agent's name as plain text on one line (no control or format characters), as
    far as a title has room for it."""
    return plain_text(str(name or ""))[:NAME_CHARS]


def _error(error_class: str) -> str:
    """The error class as plain text, as long as its column holds."""
    return plain_text(str(error_class or ""))[:ERROR_CLASS_LENGTH]


def _reason(error_class: str, table: Mapping[str, str]) -> Optional[str]:
    status = http_status(error_class)
    if status is not None:
        return table["reason_key" if status in KEY_STATUSES else "reason_status"].format(status=status)
    named = {OUT_OF_CREDIT: "reason_credit", TIMEOUT: "reason_timeout", COST_CEILING: "reason_cost"}
    key = named.get(error_class)
    return table[key] if key else None


def duration(minutes: int, locale: str = BACKGROUND_LOCALE) -> str:
    """An interval in words: "hour", "2 hours", "90 minutes" (after "every")."""
    table = _table(locale)
    if minutes % MINUTES_PER_HOUR:
        return table["minutes"].format(n=minutes)
    hours = minutes // MINUTES_PER_HOUR
    return table["hour"] if hours == 1 else table["hours"].format(n=hours)


def streak_title(name: str, error_class: str, locale: str = BACKGROUND_LOCALE) -> str:
    """The card's title: who fails and why."""
    table = _table(locale)
    reason = _reason(error_class, table)
    if reason is None:
        return table["title_bare"].format(name=_name(name), error=_error(error_class))[:TITLE_LENGTH]
    return table["title"].format(name=_name(name), reason=reason)[:TITLE_LENGTH]


def _since(moment: datetime) -> str:
    """``moment`` in UTC; a time without a zone is taken as UTC, as the columns store it."""
    aware = moment if moment.tzinfo is not None else moment.replace(tzinfo=timezone.utc)
    return aware.astimezone(timezone.utc).strftime(SINCE_FORMAT)


def streak_detail(error_class: str, count: int, since: datetime, base_minutes: int, interval_minutes: int,
                  locale: str = BACKGROUND_LOCALE) -> str:
    """The card's detail: since when and how often it failed, the slower interval from the
    third failure, and what Start does."""
    table = _table(locale)
    key = "failing_one" if count == 1 else "failing_many"
    lines = [table[key].format(since=_since(since), count=count, error=_error(error_class))]
    if count >= DOUBLING_FROM and interval_minutes > base_minutes:
        lines.append(table["slower"].format(interval=duration(interval_minutes, locale),
                                            base=duration(base_minutes, locale)))
    lines.append(table["start"])
    return " ".join(lines)[:DETAIL_LENGTH]


def works_again_detail(name: str, error_class: str, count: int, locale: str = BACKGROUND_LOCALE) -> str:
    """The detail of a streak's card once a check-in works again."""
    table = _table(locale)
    key = "works_again_one" if count == 1 else "works_again_many"
    return table[key].format(name=_name(name), count=count, error=_error(error_class))[:DETAIL_LENGTH]
