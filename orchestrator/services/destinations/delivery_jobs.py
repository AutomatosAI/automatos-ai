"""A shopper's callback, delivered as durable jobs (PRD-276 §6.5, Wave 1b).

One ``widget_destination.deliver`` job per destination, queued as the request is accepted,
retried at ``DELIVERY_BACKOFF`` up to ``DELIVERY_ATTEMPTS``, and never lost to a restart.

**The contact details.** The shopper's name and phone are the one thing the job can't re-read:
Automatos keeps only the phone's hash (``api/widgets/callback.py``). So the job carries them
encrypted, and the queue removes them from the row the moment the job ends, done or dead (the
kind's ``scrub``). Plaintext never reaches the database, and the ciphertext lives only for the
retry window. A dead delivery stays listed, but a retry can't send it again: its details are
gone, and its event log says so.

**At least once.** Every send happens under the job's fence (§6.8). Telegram, Slack and WhatsApp
messages carry no delivery id, so a worker that dies between a send and its record can repeat a
message. A webhook gets ``X-Automatos-Delivery`` (the job's id, the same on every attempt) and can
drop one it has seen.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from typing import Any, Optional
from uuid import UUID

from core.jobs.registry import job_handler
from services.destinations.base import CallbackPayload, DispatchResult
from services.destinations.dispatcher import destination_type, dispatch_via_channel, record_attempt

logger = logging.getLogger(__name__)

DELIVER_KIND = "widget_destination.deliver"
CONTACT_KEY = "contact"
DELIVERY_BACKOFF = (5, 30, 120, 600)  # seconds after attempts 1-4: long enough to ride out an outage
DELIVERY_ATTEMPTS = 5
SEND_TIMEOUT_SECONDS = 15  # the slowest channel driver's HTTP timeout (Slack, WhatsApp, webhook)
SEND_MARGIN_SECONDS = 5
CONTACT_GONE = "The shopper's contact details were erased when this delivery ended, so it can't be sent again."


class NotYetDelivered(RuntimeError):
    """A retryable failure: the job goes back in the queue at its next delay."""


def enqueue_callback_dispatch(*, site: Any, session_id: str, request_id: str, payload: CallbackPayload,
                              destinations: list[dict]) -> int:
    """Queues one delivery job per destination, the contact details sealed, on a session of its own
    (the endpoint's own writes are already committed); how many went in.

    Never raises, as the in-process dispatch it replaces never did: the request is already logged,
    so failing it would only have the shopper's retry answered as a duplicate with nothing queued.
    A failure is logged and recorded on the widget error tile, and answers 0."""
    from core.database.database import SessionLocal
    from core.utils.exception_telemetry import record_error

    db = SessionLocal()
    try:
        return queue_deliveries(db, site=site, session_id=session_id, request_id=request_id, payload=payload,
                                destinations=destinations)
    except Exception as exc:  # noqa: BLE001 — logged and recorded; the endpoint's answer stands
        logger.exception("[widget] could not queue callback %s for site %s", request_id, site.id)
        record_error(subsystem="widget", operation="queue_callback", error=exc, workspace_id=site.workspace_id,
                     extra={"site_id": str(site.id), "request_id": request_id})
        return 0
    finally:
        db.close()


def queue_deliveries(db: Any, *, site: Any, session_id: str, request_id: str, payload: CallbackPayload,
                     destinations: list[dict]) -> int:
    """``enqueue_callback_dispatch`` on the caller's session; how many jobs went in (one per
    destination and request, never twice). Commits."""
    from core.credentials.encryption import get_encryption_service
    from core.jobs import enqueue

    sealed = get_encryption_service().encrypt(json.dumps(asdict(payload)))
    queued = 0
    for n, destination in enumerate(destinations):
        job = {"site_id": str(site.id), "session_id": session_id, "request_id": request_id,
               "destination": destination, CONTACT_KEY: sealed}
        queued += enqueue(db, DELIVER_KIND, job, workspace_id=site.workspace_id, key=f"{request_id}:{n}") is not None
    db.commit()
    return queued


def _contact(payload: dict) -> Optional[CallbackPayload]:
    sealed = payload.get(CONTACT_KEY)
    if not sealed:
        return None
    from core.credentials.encryption import get_encryption_service

    return CallbackPayload(**json.loads(get_encryption_service().decrypt(sealed)))


def _final(result: DispatchResult, job: Any) -> bool:
    return result.success or not result.retryable or job.attempts >= job.max_attempts


@job_handler(DELIVER_KIND, backoff=DELIVERY_BACKOFF, max_attempts=DELIVERY_ATTEMPTS, scrub=(CONTACT_KEY,))
async def deliver_widget_callback(job: Any) -> Optional[bool]:
    """One attempt to one destination, logged; True when delivered. A retryable failure raises,
    and the worker re-queues the job. Nothing is sent once the claim is lost or the workspace closed."""
    from core.database.database import SessionLocal

    contact = _contact(job.payload)
    db = SessionLocal()
    try:
        if contact is None:
            result = DispatchResult(success=False, destination_type=destination_type(job.payload["destination"]),
                                    latency_ms=0, error=CONTACT_GONE, retryable=False)
        else:
            result = await _send_under_fence(db, job, contact)
        if result is None:
            return None
        final = _final(result, job)
        await record_attempt(db, site_id=UUID(job.payload["site_id"]), workspace_id=job.workspace_id,
                             session_id=job.payload["session_id"], request_id=job.payload["request_id"],
                             destination=job.payload["destination"], result=result, attempt=job.attempts, final=final)
    finally:
        db.close()
    if not final:
        raise NotYetDelivered(result.error or "callback not yet delivered")
    return result.success


async def _send_under_fence(db: Any, job: Any, contact: CallbackPayload) -> Optional[DispatchResult]:
    """§6.8: the fence (the job row locked while the claim holds, the workspace live, the lease
    stretched past the send), the send, and on a final answer the job done, in one transaction."""
    from core.jobs.queue import complete, fence

    if not await asyncio.to_thread(fence, db, job, SEND_TIMEOUT_SECONDS + SEND_MARGIN_SECONDS):
        await asyncio.to_thread(db.rollback)
        logger.info("[widget] delivery job %s lost its claim or its workspace closed; nothing sent", job.id)
        return None
    result = await dispatch_via_channel(destination=job.payload["destination"], payload=contact, db=db,
                                        workspace_id=job.workspace_id, delivery_id=str(job.id))
    if _final(result, job):
        await asyncio.to_thread(complete, db, job, commit=False)
    await asyncio.to_thread(db.commit)
    return result
