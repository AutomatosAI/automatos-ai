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
  the workspace's.
* **A hold per request** (``token_hold``): the route opens it before it reads what is left today
  and keeps it until the request ends, after the row is committed. Before each LLM call the
  stage holds that call's upper bound (``TokenHold.try_hold``); when the reply comes back the
  bound becomes what the reply says it cost (``TokenHold.settle``). A call that doesn't fit what
  is left, less what every other request in this process holds or spent, is not made ("daily
  budget reached", so ``review_required``, never safe). A hold released after this request
  opened its own still counts against it (its row may have been committed after this request
  read the database), so one worker never spends the same tokens twice.

Not counted: a stage whose row is never saved (a lost race, a store failure) once its request
has ended; and every uvicorn worker keeps its own holds, so workers that start stages at the same
moment can each spend what is left once. An unusable reply's tokens are counted, and a call still
in flight at the stage's deadline counts at its upper bound.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

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


class TokenHold:
    """One request's LLM tokens in this process: the bounds of its calls in flight and what its
    answered calls cost, against ``budget`` (what was left today when it opened; None is no cap)."""

    def __init__(self, seq: int) -> None:
        self.seq = seq
        self.budget: Optional[int] = None
        self.in_flight = 0
        self.spent = 0

    @property
    def total(self) -> int:
        """What this request holds: its calls in flight at their bound, and its answered calls."""
        return self.in_flight + self.spent

    def try_hold(self, bound: int) -> bool:
        """Hold ``bound`` for one call when it fits; False (nothing held) when it doesn't."""
        return _LEDGER.try_hold(self, bound)

    def settle(self, bound: int, cost: int) -> None:
        """A call held at ``bound`` came back having cost ``cost``."""
        _LEDGER.settle(self, bound, cost)


class _Ledger:
    """Every open hold in this process, and the holds released while another was open."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._seq = 0
        self._open: Dict[int, TokenHold] = {}
        self._released: List[Tuple[int, int]] = []  # (seq at release, tokens it held)

    def open(self) -> TokenHold:
        with self._lock:
            self._seq += 1
            hold = TokenHold(self._seq)
            self._open[hold.seq] = hold
            return hold

    def _others(self, hold: TokenHold) -> int:
        """What other requests hold now, plus what those released since ``hold`` opened held."""
        live = sum(other.total for seq, other in self._open.items() if seq != hold.seq)
        return live + sum(tokens for seq, tokens in self._released if seq > hold.seq)

    def try_hold(self, hold: TokenHold, bound: int) -> bool:
        with self._lock:
            if hold.budget is not None and hold.total + self._others(hold) + bound > hold.budget:
                return False
            hold.in_flight += bound
            return True

    def settle(self, hold: TokenHold, bound: int, cost: int) -> None:
        with self._lock:
            hold.in_flight = max(hold.in_flight - bound, 0)
            hold.spent += max(cost, 0)

    def release(self, hold: TokenHold) -> None:
        with self._lock:
            self._open.pop(hold.seq, None)
            self._seq += 1
            if hold.total:
                self._released.append((self._seq, hold.total))
            oldest = min(self._open, default=self._seq)
            self._released = [(seq, tokens) for seq, tokens in self._released if seq > oldest]

    def held(self) -> int:
        with self._lock:
            return sum(hold.total for hold in self._open.values())


_LEDGER = _Ledger()


@contextmanager
def token_hold(tokens_left: Callable[[], Optional[int]]) -> Iterator[TokenHold]:
    """This request's hold: opened first, then ``tokens_left()`` is read as its budget (``None``
    is no cap), released when the block ends. Keep it open until the stage's row is committed."""
    hold = _LEDGER.open()
    try:
        hold.budget = tokens_left()
        yield hold
    finally:
        _LEDGER.release(hold)


def tokens_reserved() -> int:
    """Tokens the requests open in this process hold now (in flight and spent)."""
    return _LEDGER.held()
