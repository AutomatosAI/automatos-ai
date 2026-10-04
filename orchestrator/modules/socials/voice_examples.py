"""PRD-251C (C8, US-C406): Auto learns the owner's voice from their edits.

* **What Auto drafted.** When an agent writes a post's copy (the plan maker, or an agent's
  draft or edit, US-116), the history entry it logs keeps the base copy it wrote
  (:func:`draft_entry`).
* **Kept on approval.** When a person approves the post (``needs_approval`` → approved or
  scheduled) and the copy they approve differs from Auto's last draft, the pair is kept as a
  voice example (``social_voice_examples``); the workspace keeps its newest
  ``SOCIALS_VOICE_EXAMPLES``. Copy that did not change teaches nothing and is not kept.
* **Used.** The composer gets the newest examples with the brand kit's voice
  (``api/socials_compose.compose_context``).
* **Controlled.** Brand kit, Voice lists them, and an owner or admin removes any; a removed
  one is never used again.

Keeping an example never fails the approval: it runs after the approval's commit, and a
failure is logged.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional
from uuid import UUID

from config import config
from core.models.socials import SocialVoiceExample

logger = logging.getLogger(__name__)

AUTO_COPY_KEY = "copy"  # the history entry's key for the base copy an agent wrote
NEEDS_APPROVAL = "needs_approval"
APPROVED_STATUSES = ("approved", "scheduled")
EXAMPLE_MAX_CHARS = 4000


def _base(copy: Any) -> str:
    return str((copy or {}).get("base") or "").strip() if isinstance(copy, Mapping) else ""


def draft_entry(copy: Any) -> Dict[str, str]:
    """The history entry's extra for an agent's write of the copy: its base text, when it has one."""
    base = _base(copy)
    return {AUTO_COPY_KEY: base[:EXAMPLE_MAX_CHARS]} if base else {}


def auto_draft(post: Any) -> Optional[str]:
    """The base copy an agent last wrote for the post, from its history; ``None`` when none did."""
    entries = [entry for entry in (getattr(post, "review_log", None) or []) if isinstance(entry, Mapping)]
    found = next((entry for entry in reversed(entries) if entry.get("agent") and entry.get(AUTO_COPY_KEY)), None)
    return str(found[AUTO_COPY_KEY]) if found else None


def _same(first: str, second: str) -> bool:
    return " ".join(first.split()) == " ".join(second.split())


def _trim(db: Any, workspace_id: UUID) -> None:
    """Keep the workspace's newest ``SOCIALS_VOICE_EXAMPLES``."""
    keep = max(1, int(config.SOCIALS_VOICE_EXAMPLES))
    old = (
        db.query(SocialVoiceExample.id)
        .filter(SocialVoiceExample.workspace_id == workspace_id)
        .order_by(SocialVoiceExample.created_at.desc(), SocialVoiceExample.id)
        .offset(keep)
        .all()
    )
    if old:
        db.query(SocialVoiceExample).filter(SocialVoiceExample.id.in_([row.id for row in old])).delete(synchronize_session=False)


def keep(db: Any, post: Any, now: Optional[datetime] = None) -> Optional[SocialVoiceExample]:
    """The post's voice example, kept and committed when its approved copy differs from Auto's draft."""
    draft, approved = auto_draft(post), _base(getattr(post, "copy", None))
    if not draft or not approved or _same(draft, approved):
        return None
    # created_at from the clock, not now(): examples one transaction keeps keep their order (F209).
    example = SocialVoiceExample(workspace_id=post.workspace_id, post_id=post.id, draft=draft, approved=approved[:EXAMPLE_MAX_CHARS],
                                 created_at=now or datetime.now(timezone.utc))
    db.add(example)
    db.flush()
    _trim(db, post.workspace_id)
    db.commit()
    return example


def keep_if_approved(db: Any, before: str, post: Any) -> None:
    """After a commit that approved ``post`` (from ``needs_approval``): its example kept. Never raises."""
    if before != NEEDS_APPROVAL or getattr(post, "status", None) not in APPROVED_STATUSES:
        return
    try:
        keep(db, post)
    except Exception:  # noqa: BLE001 — the approval stands; the example is a courtesy, logged
        logger.exception("[Socials] the voice example of post %s was not kept", getattr(post, "id", None))
        db.rollback()


def examples(db: Any, workspace_id: UUID, limit: Optional[int] = None) -> List[SocialVoiceExample]:
    """The workspace's examples, newest first."""
    count = max(1, int(limit or config.SOCIALS_VOICE_EXAMPLES))
    return (
        db.query(SocialVoiceExample)
        .filter(SocialVoiceExample.workspace_id == workspace_id)
        .order_by(SocialVoiceExample.created_at.desc(), SocialVoiceExample.id)
        .limit(count)
        .all()
    )


def remove(db: Any, workspace_id: UUID, example_id: UUID) -> bool:
    """Remove one of the workspace's examples; ``False`` when it has no such one."""
    found = db.query(SocialVoiceExample).filter(SocialVoiceExample.workspace_id == workspace_id, SocialVoiceExample.id == example_id).first()
    if found is None:
        return False
    db.delete(found)
    db.commit()
    return True


def for_composer(db: Any, workspace_id: UUID) -> List[Dict[str, str]]:
    """The newest examples as the composer reads them: what Auto wrote, and what was approved."""
    return [{"auto_wrote": row.draft, "approved": row.approved} for row in examples(db, workspace_id)]
