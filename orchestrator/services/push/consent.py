"""PRD-266 §4.4, D21 (US-010c, K13) — the operator's answer to "Send notifications through Expo's push service?".

A push to Automatos Go leaves the install for Expo's push service (``services/push/expo.py``).
On SaaS that is the platform's own service, so it is always allowed. A self-hosted (local)
install sends nothing until its operator agrees, once, and may change their mind later.

The answer is ONE install-level setting in the DB settings table, not an environment
setting, so nothing here is in ``config.py`` (K13): category ``notifications``,

* ``push_via_expo`` — ``'true'`` or ``'false'``;
* ``push_via_expo_answered_at`` — when it was last answered (ISO-8601, UTC).

``expo_push_allowed`` reads through ``core.llm.manager.get_system_setting`` (its own short
session, fail-soft): a read that fails is "not agreed", so a broken settings table never sends
anything off the install. ``consent_state`` reads strictly (``read_system_setting``) and raises
when it cannot read, so the routes never show "could not read" as "never answered" (Studio
would ask again). The writer follows ``modules/voice/live_settings.set_voice_setting``: it
updates the row, creating it when it is missing, flushes, and leaves the commit to its caller
(the ``PUT /api/v1/mobile/push-consent`` route in ``api/mobile_push.py``). ``system_settings``
has no unique (category, key), so it updates every matching row: a duplicate pair can never
disagree once an answer is written.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from sqlalchemy.orm import Session

from config import config
from core.llm.manager import get_system_setting, read_system_setting
from core.models.system_settings import SettingCategory, SystemSetting

CONSENT_CATEGORY = SettingCategory.NOTIFICATIONS.value
KEY_PUSH_VIA_EXPO = "push_via_expo"
KEY_ANSWERED_AT = "push_via_expo_answered_at"
AGREED = "true"
DECLINED = "false"
BOOLEAN_TYPE = "boolean"
STRING_TYPE = "string"
CREATED_BY = "prd266"
SETTING_DESCRIPTIONS = {
    KEY_PUSH_VIA_EXPO: "The operator's answer to 'Send notifications through Expo's push service?' "
                       "(PRD-266 D21). Nothing is pushed from this install until it is true.",
    KEY_ANSWERED_AT: "When the operator last answered push_via_expo (ISO-8601, UTC).",
}


@dataclass(frozen=True)
class ConsentState:
    """The operator's answer as the routes show it."""

    agreed: bool
    answered: bool
    answered_at: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        """``{agreed, answered, answered_at}``, the routes' body."""
        return {"agreed": self.agreed, "answered": self.answered, "answered_at": self.answered_at}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _is_agreed(value: Optional[str]) -> bool:
    """Only the word ``'true'`` agrees; anything else, or nothing, does not."""
    return str(value or "").strip().lower() == AGREED


def expo_push_allowed() -> bool:
    """May this install send notifications through Expo's push service?

    Always on SaaS. On a local install only once the operator agreed (``push_via_expo`` is
    ``'true'``); not answered, declined, or a failed read is False.
    """
    if not config.IS_LOCAL_EDITION:
        return True
    return _is_agreed(get_system_setting(CONSENT_CATEGORY, KEY_PUSH_VIA_EXPO, DECLINED))


def consent_state() -> ConsentState:
    """The stored answer: agreed, whether it was ever answered, and when (or None).

    Raises (``read_system_setting``'s error) when the settings cannot be read.
    """
    agreed = _is_agreed(read_system_setting(CONSENT_CATEGORY, KEY_PUSH_VIA_EXPO))
    answered_at = (read_system_setting(CONSENT_CATEGORY, KEY_ANSWERED_AT) or "").strip() or None
    return ConsentState(agreed=agreed, answered=answered_at is not None, answered_at=answered_at)


def _write_setting(db: Session, key: str, value: str, value_type: str) -> None:
    """Update every ``notifications.<key>`` row, or add one when this install has none yet."""
    updated = (
        db.query(SystemSetting)
        .filter(SystemSetting.category == CONSENT_CATEGORY, SystemSetting.key == key)
        .update({SystemSetting.value: value}, synchronize_session="fetch")
    )
    if updated:
        return
    db.add(SystemSetting(
        category=CONSENT_CATEGORY,
        key=key,
        value=value,
        value_type=value_type,
        description=SETTING_DESCRIPTIONS[key],
        is_sensitive=False,
        created_by=CREATED_BY,
    ))


def set_push_consent(db: Session, agreed: bool) -> ConsentState:
    """Record the operator's answer and when it was given; flushes, never commits.

    Returns the state as written, so the caller can answer with it once it commits.
    """
    answered_at = _now().isoformat()
    _write_setting(db, KEY_PUSH_VIA_EXPO, AGREED if agreed else DECLINED, BOOLEAN_TYPE)
    _write_setting(db, KEY_ANSWERED_AT, answered_at, STRING_TYPE)
    db.flush()
    return ConsentState(agreed=agreed, answered=True, answered_at=answered_at)


__all__ = [
    "CONSENT_CATEGORY",
    "KEY_ANSWERED_AT",
    "KEY_PUSH_VIA_EXPO",
    "ConsentState",
    "consent_state",
    "expo_push_allowed",
    "set_push_consent",
]
