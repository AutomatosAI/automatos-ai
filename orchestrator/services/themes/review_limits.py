"""PRD-261 C-02a — what theme reviews may spend.

The review stage (``review_stage.py``) runs inline at a Marketplace submission and at a private
upload, and its LLM check can make dozens of paid calls. Four limits cap that spend:

* **Stages at once** (``THEME_REVIEW_MAX_CONCURRENT``, default 2): a process-wide count taken
  with a try (never a wait) before the route holds any database connection for the stage; when
  it is full the route answers 503 with ``Retry-After``. Each uvicorn worker has its own count.
* **Submissions per person** (``THEME_SUBMISSIONS_PER_DAY``, default 10) and **private uploads
  per workspace** (``THEME_UPLOADS_PER_DAY``, default 20) in a rolling 24 hours, counted from
  ``theme_packages`` rows and checked before the stage: past it, 429 with a plain sentence.
* **LLM tokens per day** (``THEME_REVIEW_DAILY_TOKENS``, default 2,000,000) for every theme
  review together, in a rolling 24 hours. What was spent is read from the rows' own scan reports
  (``checks.llm.tokensUsed``), so no new table: ``llm_usage`` is a per-workspace billing ledger
  (cost columns, no index on its type or time) and the review's tokens are the platform's, not
  the workspace's. A stage reserves its estimate in this process before its calls
  (``reserve_tokens``), so two stages at once never both spend the last of the budget here; one
  that doesn't fit doesn't call the LLM ("LLM stage did not run: daily budget reached", so
  ``review_required``, never safe).

Not counted: a stage whose row is never saved (a lost race, a store failure; the claim and the
read-only checks before the stage keep that rare), and a call still in flight at the stage's
deadline (at most ``llm_stage.CONCURRENCY`` per stage, each ending at its own timeout). An
unusable reply's tokens are counted.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterator, Mapping, Optional

from sqlalchemy.orm import Session

from config import config
from core.models.themes import ThemePackage

logger = logging.getLogger(__name__)

DEFAULT_MAX_CONCURRENT = 2
DEFAULT_SUBMISSIONS_PER_DAY = 10
DEFAULT_UPLOADS_PER_DAY = 20
DEFAULT_DAILY_TOKENS = 2_000_000
WINDOW = timedelta(hours=24)
BUSY_RETRY_SECONDS = 30
OWNER_MARKETPLACE, OWNER_WORKSPACE = "marketplace", "workspace"

REVIEWS_BUSY = "Theme reviews are busy right now. Try again in a minute."
SUBMISSIONS_QUOTA = (
    "You've submitted {limit} themes in the last 24 hours, the most allowed. Try again tomorrow."
)
UPLOADS_QUOTA = (
    "This workspace has uploaded {limit} themes in the last 24 hours, the most allowed. Try again tomorrow."
)


def positive_setting(name: str, default: int) -> int:
    """A positive whole-number setting from config, else the default (a typo never lifts a cap)."""
    try:
        value = int(getattr(config, name, default) or default)
    except (TypeError, ValueError):
        logger.warning("[Themes] %s is not a whole number: using %d", name, default)
        return default
    return value if value > 0 else default


class _Gauge:
    """A number shared by this process's threads, changed under a lock."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value = 0

    def try_add(self, amount: int, ceiling: int) -> bool:
        """Add ``amount`` when the total stays at or under ``ceiling``; False (and no change) otherwise."""
        with self._lock:
            if self._value + amount > ceiling:
                return False
            self._value += amount
            return True

    def take(self, amount: int) -> None:
        with self._lock:
            self._value = max(self._value - amount, 0)

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


_STAGES = _Gauge()
_RESERVED_TOKENS = _Gauge()


@contextmanager
def review_slot() -> Iterator[bool]:
    """Yields True when this request may run a review stage now, False when the process is full."""
    held = _STAGES.try_add(1, positive_setting("THEME_REVIEW_MAX_CONCURRENT", DEFAULT_MAX_CONCURRENT))
    try:
        yield held
    finally:
        if held:
            _STAGES.take(1)


def busy_headers() -> Dict[str, str]:
    """The 503's ``Retry-After``."""
    return {"Retry-After": str(BUSY_RETRY_SECONDS)}


def stages_running() -> int:
    """Review stages running in this process now."""
    return _STAGES.value


def _since() -> datetime:
    return datetime.now(timezone.utc) - WINDOW


def _recent(db: Session, owner_type: str):
    return db.query(ThemePackage).filter(ThemePackage.owner_type == owner_type, ThemePackage.created_at >= _since())


def submission_quota_refusal(db: Session, user_pk: Optional[int]) -> Optional[str]:
    """Why this person may not submit again today, or None. The local operator with no user row counts as one person."""
    limit = positive_setting("THEME_SUBMISSIONS_PER_DAY", DEFAULT_SUBMISSIONS_PER_DAY)
    column = ThemePackage.submitted_by
    mine = column.is_(None) if user_pk is None else column == user_pk
    if _recent(db, OWNER_MARKETPLACE).filter(mine).count() < limit:
        return None
    logger.info("[Themes] Submission refused: user %s reached %d submissions in 24 hours", user_pk, limit)
    return SUBMISSIONS_QUOTA.format(limit=limit)


def upload_quota_refusal(db: Session, workspace_id: Any) -> Optional[str]:
    """Why this workspace may not upload another private theme today, or None."""
    limit = positive_setting("THEME_UPLOADS_PER_DAY", DEFAULT_UPLOADS_PER_DAY)
    if _recent(db, OWNER_WORKSPACE).filter(ThemePackage.workspace_id == workspace_id).count() < limit:
        return None
    logger.info("[Themes] Upload refused: workspace %s reached %d uploads in 24 hours", workspace_id, limit)
    return UPLOADS_QUOTA.format(limit=limit)


def _llm_tokens(report: Any) -> int:
    """``checks.llm.tokensUsed`` of one scan report; 0 when it has none."""
    checks = report.get("checks") if isinstance(report, Mapping) else None
    llm = checks.get("llm") if isinstance(checks, Mapping) else None
    used = llm.get("tokensUsed") if isinstance(llm, Mapping) else 0
    return used if isinstance(used, int) and not isinstance(used, bool) and used > 0 else 0


def tokens_used_today(db: Session) -> int:
    """LLM tokens every theme review stored in the last 24 hours, private and Marketplace alike."""
    rows = db.query(ThemePackage.scan_report).filter(ThemePackage.created_at >= _since()).all()
    return sum(_llm_tokens(row[0]) for row in rows)


def tokens_left_today(db: Session) -> int:
    """``THEME_REVIEW_DAILY_TOKENS`` less what the last 24 hours' reviews used (never below 0)."""
    budget = positive_setting("THEME_REVIEW_DAILY_TOKENS", DEFAULT_DAILY_TOKENS)
    return max(budget - tokens_used_today(db), 0)


@contextmanager
def reserve_tokens(estimate: int, tokens_left: Optional[int]) -> Iterator[bool]:
    """Yields True with ``estimate`` held for the block when it fits in ``tokens_left`` less what
    other stages in this process hold; False (nothing held) when it doesn't. ``None`` is no cap."""
    if tokens_left is None:
        yield True
        return
    held = _RESERVED_TOKENS.try_add(estimate, tokens_left)
    try:
        yield held
    finally:
        if held:
            _RESERVED_TOKENS.take(estimate)


def tokens_reserved() -> int:
    """Tokens the stages running in this process hold now."""
    return _RESERVED_TOKENS.value
