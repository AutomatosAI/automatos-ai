"""The moderation post: a new content report's summary to the Automatos inbox (PRD-266 US-016b).

On SaaS, each NEW report (never a duplicate, never one that failed) is posted, after
the response, to ``MODERATION_WEBHOOK_URL`` through the existing generic webhook
driver (``channels/drivers/webhook.py``: httpx, a 15 s timeout, the body
``{text, message, ai_generated}``). A local install posts nothing, even when the
setting is set: its reports stay with its operator (D15).

The summary carries the report id, the workspace id, the target type and id, the
reason, an excerpt of the reported content and the reporter's note, each cut to
``EXCERPT_MAX_CHARS``. It never carries the reporter's name, email or id, or the
workspace's name: ``FiledReport`` holds none of them.

The URL is a credential: it is never logged, and neither is the driver's error
text (an httpx exception can quote the URL). A failed post is logged with the
report id, the HTTP status when there is one and whether it is retryable.
"""

from __future__ import annotations

import functools
import logging
import re
from typing import Optional
from urllib.parse import urlsplit

from channels.drivers import get_driver
from config import config
from services.content_reports import EXCERPT_MAX_CHARS
from services.content_reports.service import FiledReport
from services.content_reports.targets import clip_excerpt

logger = logging.getLogger(__name__)

# The existing generic outbound driver; no client of our own.
WEBHOOK_PLATFORM = "webhook"
REQUIRED_URL_SCHEME = "https"

NO_NOTE = "(none)"

# Slack's three control characters (mentions, links), sent as their entities so a
# note or an excerpt cannot ping a channel or carry a disguised link.
_CONTROL_ENTITIES: dict[str, str] = {"&": "&amp;", "<": "&lt;", ">": "&gt;"}

# The driver's own failure text for an HTTP error; only the status is read from it.
_DRIVER_HTTP_ERROR = re.compile(r"webhook returned HTTP (\d{3})")


def _plain(text: Optional[str]) -> str:
    """``text`` on one line with Slack's control characters escaped, at most ``EXCERPT_MAX_CHARS`` long.

    The cut counts the posted characters (an entity counts its full length) and
    never splits an entity.
    """
    units: list[str] = []
    length = 0
    for char in clip_excerpt(text or ""):
        unit = _CONTROL_ENTITIES.get(char, char)
        if length + len(unit) > EXCERPT_MAX_CHARS:
            break
        units.append(unit)
        length += len(unit)
    return "".join(units)


def moderation_summary(report: FiledReport) -> str:
    """The text posted to the moderation inbox for ``report``: no reporter, no workspace name.

    The excerpt and the note are each at most ``EXCERPT_MAX_CHARS`` (200) characters as
    posted, on one line, so neither can add a line of its own.
    """
    note = _plain(report.note) or NO_NOTE
    return "\n".join((
        f"Automatos content report {report.report_id}",
        f"Workspace: {report.workspace_id}",
        f"Target: {report.target_type} {report.target_id}",
        f"Reason: {report.reason}",
        f"Excerpt: {_plain(report.excerpt)}",
        f"Note: {note}",
    ))


def _is_https_with_host(value: str) -> bool:
    """Whether ``value`` is an https URL that names a host."""
    try:
        parts = urlsplit(value)
    except ValueError:
        return False
    return parts.scheme == REQUIRED_URL_SCHEME and bool(parts.hostname)


@functools.lru_cache(maxsize=1)
def _warn_unusable_setting() -> None:
    """Say once per process that the setting is unusable; never its value."""
    logger.warning(
        "[content_reports] MODERATION_WEBHOOK_URL is set but is not an https URL with a host: "
        "content reports are not posted to the moderation inbox"
    )


def moderation_url() -> Optional[str]:
    """The inbox URL when this install posts reports (SaaS and an https URL), else ``None``."""
    if not config.IS_SAAS_EDITION:
        return None
    value = (config.MODERATION_WEBHOOK_URL or "").strip()
    if not value:
        return None
    if not _is_https_with_host(value):
        _warn_unusable_setting()
        return None
    return value


def _http_status(error: Optional[str]) -> Optional[str]:
    """The HTTP status in the driver's failure text, or ``None`` (a network error has none)."""
    match = _DRIVER_HTTP_ERROR.fullmatch(error or "")
    return match.group(1) if match else None


async def post_to_moderation(summary: str, *, report_id: int, workspace_id: str) -> bool:
    """Post ``summary`` to the moderation inbox; ``True`` when the inbox took it.

    ``False``, with no network call, unless this install posts reports
    (``moderation_url``). A failure is logged at ERROR without the URL and
    without the driver's error text, and is never raised: the report stands.
    """
    url = moderation_url()
    if url is None:
        return False
    try:
        result = await get_driver(WEBHOOK_PLATFORM)().send(
            workspace_id=workspace_id, config={"webhook_url": url}, target=None, text=summary,
        )
    except Exception as exc:  # noqa: BLE001 — its text can quote the URL: logged by class only, returns False
        logger.error(
            "[content_reports] report %s: the moderation post raised %s (not retried)",
            report_id, type(exc).__name__,
        )
        return False
    if result.ok:
        return True
    logger.error(
        "[content_reports] report %s: the moderation post failed (HTTP status %s, retryable %s)",
        report_id, _http_status(result.error) or "none", result.retryable,
    )
    return False


__all__ = [
    "WEBHOOK_PLATFORM",
    "moderation_summary",
    "moderation_url",
    "post_to_moderation",
]
