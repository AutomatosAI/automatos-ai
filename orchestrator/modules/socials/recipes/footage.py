"""PRD-251 S1.8 (D12, D13): footage and stills from the workspace's Composio generation toolkit.

A post asks its template's slots to be filled (``social_posts.footage``:
``{slot: {"prompt"}}``). Its render generates them first, before the voice and
before media-render (``modules/socials/render.py``):

1. **The plan** (:func:`plan_for`, as the render starts). A slot the template
   does not have, or marks ``"generate": false`` (the app's own screen
   recording: never generated UI), is not generated. A slot already made for its
   prompt is reused. Every other slot is routed to the first toolkit, in
   ``SOCIALS_FOOTAGE_TOOLKITS`` order, whose recipe makes its kind (footage or a
   still) with every action it calls on offer in the media capability registry
   (``footage_toolkits.py``). With no such toolkit the slot plays the template's
   own motion graphics, and the render's report says why.
2. **The money** (D13), inside one footage window per workspace across every
   worker process (a Postgres advisory lock), so two renders never spend the
   same headroom: every shot is priced (fal.ai by its estimate action, the
   others at ``SOCIALS_FOOTAGE_CEILING_*_USD``) and the total must fit the
   post's cap and the workspace's monthly media cap
   (``modules/socials/media_caps.py``). Over a cap nothing is submitted, and the
   render fails saying why.
3. **Submit, then poll**, never one long call: every shot is submitted, then
   polled every ``SOCIALS_FOOTAGE_POLL_SECONDS``, all of them within
   ``SOCIALS_FOOTAGE_MAX_WAIT_SECONDS`` of the window opening (one deadline for
   the render's footage, however many toolkits make it), through
   ``ComposioToolExecutor``, which checks the Wave 0 deny list again.
4. **Copied the moment it completes** (provider links expire): the file is
   fetched from a public address only, pinned and size-capped (``files.fetch``),
   checked to be the slot's kind by its first bytes, stored at
   ``social-media/{workspace}/{post}/footage-<slot>-<sha8>.<ext>`` (D9) and
   registered as a Deliverable. Only then is the slot marked done on the post
   (``service.record_footage``), where the next render reuses it.
5. **Booked** on the media lane against the post, before the window closes so
   the next cap check sees it: each shot fal.ai completed at its estimate; a
   credit-billed toolkit's balance difference (Kie.ai, Higgsfield), read before
   the first submit and after the last job ends inside the toolkit's credit
   window, at ``SOCIALS_*_USD_PER_CREDIT``.

Generated clips fill template slots only (the hook and the b-roll): product
screens stay HTML, and every word on screen is template text.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import tempfile
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union
from uuid import UUID

from config import config
from core.composio.tool_executor import ComposioToolExecutor
from core.llm.usage_context import LANE_MEDIA, usage_scope
from core.llm.usage_tracker import UsageTracker
from core.models.socials import SocialPost
from core.models.workspaces import Workspace
from core.social_templates import IMAGE_SLOT, VIDEO_SLOT, slot_generatable
from modules.socials import service
from modules.socials.capabilities import MediaCapabilities
from modules.socials.media_caps import MediaCapExceeded, check_caps, media_spend, post_execution_id
from modules.socials.media_store import MediaNameError, MediaStore, media_key, media_route, valid_file_name
from modules.socials.recipes.files import FileOutputError, ReturnedFile, fetch
from modules.socials.recipes.footage_toolkits import (
    CREDITS,
    FAILED,
    KIND_CAPABILITY,
    KIND_WORDS,
    RECIPES,
    RUNNING,
    FootageError,
    FootageRecipe,
    FootageRefused,
    FootageToolError,
    JobState,
    Route,
    Shot,
    aspect_ratio,
)
from modules.socials.recipes.toolkit import (
    ToolkitSchemaChanged,
    balance_of,
    call,
    credit_window,
    error_of,
    output_of,
    params_for,
    window,
)

logger = logging.getLogger(__name__)

# The footage window's advisory lock namespace ('socf'): one window per workspace.
FOOTAGE_LOCK_NAMESPACE = 0x736F6366
# A job whose status cannot be read this many times running is given up on.
MAX_POLL_FAILURES = 5
FOOTAGE_FILE_PREFIX = "footage-"
DIGEST_CHARS = 8
DELIVERABLE_SOURCE_TYPE = "social_post"
AVAILABLE, CONNECT, UNAVAILABLE = "available", "connect", "unavailable"
# Generated media by its first bytes → (kind, extension, content type). An
# ftyp box whose brand is an audio one is not footage.
AUDIO_BRANDS = (b"M4A ", b"M4B ", b"M4P ")


# ── the plan ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Kept:
    """A file a render already generated for the slot's prompt, in our storage."""

    slot: str
    path: str
    name: str


@dataclass(frozen=True)
class FootagePlan:
    """What a render does about the footage its post asks for, resolved as it starts."""

    shots: Tuple[Tuple[Shot, Route], ...] = ()
    kept: Tuple[Kept, ...] = ()
    # slot → why the template's own motion graphics play there instead.
    fallback: Mapping[str, str] = field(default_factory=dict)

    @property
    def shown(self) -> Tuple[str, ...]:
        """The slots this render shows: those it generates and those it reuses."""
        return tuple(shot.slot for shot, _ in self.shots) + tuple(kept.slot for kept in self.kept)

    def report(self) -> Dict[str, Any]:
        """The plan, for the render's report in the review log."""
        out: Dict[str, Any] = {}
        if self.shots:
            out["generated"] = {shot.slot: route.recipe.toolkit for shot, route in self.shots}
        if self.kept:
            out["reused"] = [kept.slot for kept in self.kept]
        if self.fallback:
            out["motion_graphics"] = dict(self.fallback)
        return out


def preferred_toolkits() -> Tuple[str, ...]:
    """The generation toolkits a render tries, in order (``SOCIALS_FOOTAGE_TOOLKITS``)."""
    names = (name.strip().lower() for name in (config.SOCIALS_FOOTAGE_TOOLKITS or "").split(","))
    return tuple(dict.fromkeys(name for name in names if name in RECIPES))


def route_for(kind: str, caps: MediaCapabilities) -> Union[Route, str]:
    """The first toolkit route that makes ``kind`` here; else why none does."""
    if caps.problem:
        return caps.problem
    reasons = []
    for toolkit in preferred_toolkits():
        if toolkit not in caps.connected:
            continue
        route, why = RECIPES[toolkit].route(kind, caps)
        if route is not None:
            return route
        reasons.append(why)
    if reasons:
        return "; ".join(reasons)
    connectable = [RECIPES[t].label for t in preferred_toolkits() if t in caps.connectable(KIND_CAPABILITY[kind])]
    hint = f": connect {' or '.join(connectable)} in Composio" if connectable else ""
    return f"no generation toolkit that makes {KIND_WORDS[kind]} is connected{hint}"


def plan_for(footage: Any, slots: Any, caps: MediaCapabilities, *, width: int, height: int) -> Optional[FootagePlan]:
    """The plan for the post's ``footage`` over its template's ``slots``; ``None`` when it asks for none."""
    asked = footage if isinstance(footage, Mapping) else {}
    if not asked:
        return None
    specs = slots if isinstance(slots, Mapping) else {}
    ratio = aspect_ratio(width, height)
    shots: List[Tuple[Shot, Route]] = []
    kept: List[Kept] = []
    fallback: Dict[str, str] = {}
    routes: Dict[str, Union[Route, str]] = {}
    for slot, record in asked.items():
        spec = specs.get(slot)
        if not isinstance(spec, Mapping):
            fallback[slot] = "the template has no such slot"
            continue
        label = str(spec.get("label") or slot)
        if not slot_generatable(spec):
            fallback[slot] = f"{label} takes the workspace's own file, never generated footage"
            continue
        if not isinstance(record, Mapping) or not isinstance(record.get("prompt"), str):
            fallback[slot] = "it asks for no prompt"
            continue
        if record.get("status") == service.FOOTAGE_DONE and valid_file_name(record.get("name")):
            kept.append(Kept(slot=slot, path=spec["path"], name=record["name"]))
            continue
        kind = spec["kind"]
        if kind not in routes:
            routes[kind] = route_for(kind, caps)
        route = routes[kind]
        if isinstance(route, str):
            fallback[slot] = route
            continue
        shots.append((Shot(slot=slot, kind=kind, path=spec["path"], label=label, prompt=record["prompt"], aspect_ratio=ratio), route))
    return FootagePlan(shots=tuple(shots), kept=tuple(kept), fallback=dict(fallback))


def footage_sources(caps: MediaCapabilities) -> Dict[str, Any]:
    """What a post's slots can be filled with here (D15, D16): per kind, the toolkit
    a render would use; per generation toolkit, available, to connect, or why not."""
    kinds = {}
    for kind in (VIDEO_SLOT, IMAGE_SLOT):
        route = route_for(kind, caps)
        kinds[kind] = (
            {"available": True, "toolkit": route.recipe.toolkit, "label": route.recipe.label, "model": route.model}
            if not isinstance(route, str)
            else {"available": False, "reason": route}
        )
    toolkits = []
    for toolkit in preferred_toolkits():
        recipe = RECIPES[toolkit]
        entry: Dict[str, Any] = {"toolkit": toolkit, "label": recipe.label}
        if toolkit not in caps.connected:
            offers = any(toolkit in caps.connectable(KIND_CAPABILITY[kind]) for kind in (VIDEO_SLOT, IMAGE_SLOT))
            if offers and not caps.problem:
                toolkits.append({**entry, "status": CONNECT})
            continue
        made = {kind: recipe.route(kind, caps) for kind in (VIDEO_SLOT, IMAGE_SLOT)}
        makes = [kind for kind, (route, _) in made.items() if route is not None]
        if makes:
            toolkits.append({**entry, "status": AVAILABLE, "makes": makes})
        else:
            toolkits.append({**entry, "status": UNAVAILABLE, "reason": made[VIDEO_SLOT][1]})
    return {"kinds": kinds, "toolkits": toolkits, "problem": caps.problem}


# ── the files ───────────────────────────────────────────────────────────────
def media_type(data: bytes) -> Optional[Tuple[str, str, str]]:
    """(kind, extension, content type) of generated media by its first bytes; ``None`` when neither."""
    head = data[:16]
    if head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand in AUDIO_BRANDS:
            return None
        return (VIDEO_SLOT, "mov", "video/quicktime") if brand.startswith(b"qt") else (VIDEO_SLOT, "mp4", "video/mp4")
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return VIDEO_SLOT, "webm", "video/webm"
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return IMAGE_SLOT, "png", "image/png"
    if head[:3] == b"\xff\xd8\xff":
        return IMAGE_SLOT, "jpg", "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return IMAGE_SLOT, "webp", "image/webp"
    return None


def footage_file_name(slot: str, digest: str, extension: str) -> str:
    """``footage-<slot>-<sha8>.<ext>``: a new file never replaces one an earlier Deliverable names."""
    return f"{FOOTAGE_FILE_PREFIX}{slot.lower()}-{digest[:DIGEST_CHARS]}.{extension}"


@dataclass(frozen=True)
class Made:
    """A shot generated, copied into our storage, registered and recorded on the post."""

    slot: str
    path: str
    key: str
    name: str
    toolkit: str
    model: str
    bytes: int
    sha256: str
    estimate_usd: float


def _store(store: MediaStore, key: str, data: bytes, content_type: str) -> None:
    with tempfile.TemporaryDirectory(prefix="socials-footage-") as scratch:
        path = Path(scratch) / "shot"
        path.write_bytes(data)
        store.put_file(key, path, content_type)


def _register(session_factory: Callable[[], Any], *, workspace_id: UUID, post_id: UUID, title: str, shot: Shot,
              route: Route, key: str, name: str, size: int, digest: str, estimate_usd: float) -> str:
    from services.deliverable_service import DeliverableService, _infer_artifact_type

    db = session_factory()
    try:
        result = DeliverableService(db, workspace_id).register(
            file_path=key,
            title=f"{title}: {shot.label} (generated)",
            source_type=DELIVERABLE_SOURCE_TYPE,
            source_id=str(post_id),
            artifact_type=_infer_artifact_type(name),
            storage_type="s3",
            file_type=Path(name).suffix.lstrip("."),
            file_size_bytes=size,
            preview_url=media_route(post_id, name),
            preview_type="file",
            extra={
                "social_post_id": str(post_id),
                "sha256": digest,
                "footage": {
                    "slot": shot.slot, "toolkit": route.recipe.toolkit, "model": route.model,
                    "prompt": shot.prompt, "estimate_usd": estimate_usd,
                },
            },
        )
    finally:
        db.close()
    if not result.get("success") or not result.get("deliverable_id"):
        raise FootageError(f"the {shot.label.lower()} could not be saved as a Deliverable: {result.get('error')}")
    return str(result["deliverable_id"])


def _locked_post(db: Any, workspace_id: UUID, post_id: UUID) -> Optional[SocialPost]:
    """The post, its row locked until the transaction ends (footage is written whole)."""
    return (
        db.query(SocialPost)
        .filter(SocialPost.workspace_id == workspace_id, SocialPost.id == post_id)
        .with_for_update()
        .first()
    )


def _record(session_factory: Callable[[], Any], workspace_id: UUID, post_id: UUID, slot: str, record: Mapping[str, Any]) -> bool:
    """Mark the slot done on the post with its file; ``False`` when the post no longer asks for it."""
    db = session_factory()
    try:
        post = _locked_post(db, workspace_id, post_id)
        if post is None or not service.record_footage(post, slot, record):
            db.rollback()
            return False
        db.commit()
        return True
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _record_costs(session_factory: Callable[[], Any], workspace_id: UUID, post_id: UUID, costs: Mapping[str, float]) -> None:
    """A credit-billed toolkit's cost per slot, once its balance difference is known."""
    db = session_factory()
    try:
        post = _locked_post(db, workspace_id, post_id)
        footage = dict(post.footage) if post is not None and isinstance(post.footage, dict) else {}
        changed = False
        for slot, usd in costs.items():
            record = footage.get(slot)
            if isinstance(record, dict) and record.get("status") == service.FOOTAGE_DONE:
                footage[slot] = {**record, "cost_usd": usd}
                changed = True
        if changed:
            post.footage = footage
            db.commit()
        else:
            db.rollback()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def _keep(store: MediaStore, session_factory: Callable[[], Any], shot: Shot, route: Route, estimate_usd: float,
                returned: ReturnedFile, *, workspace_id: UUID, post_id: UUID, title: str) -> Made:
    """Fetch the shot now (its link expires), check it, store it, register it, then mark it done."""
    label = route.recipe.label
    try:
        data = returned.data or await fetch(
            returned.url,
            max_bytes=config.SOCIALS_FOOTAGE_MAX_BYTES,
            timeout_seconds=config.SOCIALS_MEDIA_FETCH_TIMEOUT_SECONDS,
        )
    except FileOutputError as exc:
        raise FootageError(f"{label}'s file could not be fetched: {exc}") from exc
    if len(data) > config.SOCIALS_FOOTAGE_MAX_BYTES:
        raise FootageError(f"{label}'s file is larger than the {config.SOCIALS_FOOTAGE_MAX_BYTES}-byte limit")
    found = media_type(data)
    if found is None or found[0] != shot.kind:
        raise FootageError(f"{label} returned something that is not {KIND_WORDS[shot.kind]}")
    _, extension, content_type = found
    digest = hashlib.sha256(data).hexdigest()
    name = footage_file_name(shot.slot, digest, extension)
    try:
        key = media_key(workspace_id, post_id, name)
    except MediaNameError as exc:
        raise FootageError(f"the {shot.label.lower()} cannot be stored: {exc}") from exc
    try:
        await asyncio.to_thread(_store, store, key, data, content_type)
    except Exception as exc:  # noqa: BLE001 — any storage error fails the render, loudly
        logger.exception("[SocialsFootage] storing %s failed", key)
        raise FootageError(f"{label}'s file could not be stored") from exc
    deliverable_id = await asyncio.to_thread(
        _register, session_factory, workspace_id=workspace_id, post_id=post_id, title=title, shot=shot,
        route=route, key=key, name=name, size=len(data), digest=digest, estimate_usd=estimate_usd,
    )
    record = {
        "prompt": shot.prompt,
        "toolkit": route.recipe.toolkit,
        "model": route.model,
        "deliverable_id": deliverable_id,
        "name": name,
        "sha256": digest,
        "bytes": len(data),
        "content_type": content_type,
        "estimate_usd": estimate_usd,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if not route.recipe.credit_billed:
        record["cost_usd"] = estimate_usd
    if not await asyncio.to_thread(_record, session_factory, workspace_id, post_id, shot.slot, record):
        logger.warning("[SocialsFootage] post %s no longer asks for %s as generated; it is kept as a Deliverable", post_id, shot.slot)
    return Made(
        slot=shot.slot, path=shot.path, key=key, name=name, toolkit=route.recipe.toolkit, model=route.model,
        bytes=len(data), sha256=digest, estimate_usd=estimate_usd,
    )


# ── calling the toolkit ─────────────────────────────────────────────────────
def _release(executor: Any) -> None:
    """End the executor session's transaction: a job is polled for minutes, and
    no read may stay open (idle in transaction) between two calls."""
    db = getattr(executor, "db", None)
    if db is not None:
        db.rollback()


def _asker(executor: Any, workspace_id: UUID, route: Route) -> Callable[..., Any]:
    """``ask(role, wanted, required)``: the route's action for ``role`` on the workspace's connection."""

    async def ask(role: str, wanted: Mapping[str, Any], required: Sequence[str] = ()) -> Any:
        action = route.actions[role]
        try:
            params = params_for(action, wanted, required)
        except ToolkitSchemaChanged as exc:
            raise FootageToolError(str(exc)) from None
        try:
            result = await call(executor, workspace_id, action, params)
        finally:
            await asyncio.to_thread(_release, executor)
        if not result.get("success"):
            raise FootageToolError(f"{route.recipe.label}'s {action.slug} failed: {error_of(result)}")
        return output_of(result)

    return ask


async def _read_balance(executor: Any, workspace_id: UUID, route: Route) -> Optional[Decimal]:
    try:
        credits = await _asker(executor, workspace_id, route)(CREDITS, {})
    except FootageToolError as exc:
        logger.warning("[SocialsFootage] %s's balance could not be read: %s", route.recipe.label, exc)
        return None
    except Exception:  # noqa: BLE001 — a balance that cannot be read is said so by the caller
        logger.exception("[SocialsFootage] %s's balance could not be read", route.recipe.label)
        return None
    return balance_of(credits, route.recipe.balance_keys)


def _what(shots: Sequence[Shot]) -> str:
    if len(shots) == 1:
        return f"The {shots[0].label.lower()}"
    return f"{len(shots)} generated shots ({', '.join(shot.label.lower() for shot in shots)})"


async def _price(executor: Any, workspace_id: UUID, shots: Sequence[Tuple[Shot, Route]]) -> List[Tuple[Shot, Route, float]]:
    """Each shot's price (D13): a pricing toolkit's estimate, asked once per model; else its ceiling."""
    prices: Dict[str, float] = {}
    by_model: Dict[Tuple[str, str], List[Tuple[Shot, Route]]] = {}
    for shot, route in shots:
        if route.recipe.prices:
            by_model.setdefault((route.recipe.toolkit, route.model), []).append((shot, route))
        else:
            prices[shot.slot] = route.recipe.ceiling(shot.kind)
    for (_, model), items in by_model.items():
        route = items[0][1]
        try:
            total = await route.recipe.estimate(_asker(executor, workspace_id, route), model, len(items))
        except FootageToolError as exc:
            raise FootageRefused(f"{route.recipe.label} could not price the shots ({exc}): nothing was submitted.") from exc
        for shot, _ in items:
            prices[shot.slot] = total / len(items)
    return [(shot, route, prices[shot.slot]) for shot, route in shots]


def _check_caps(session_factory: Callable[[], Any], workspace_id: UUID, post_id: UUID, price_usd: float, what: str) -> None:
    db = session_factory()
    try:
        workspace = db.get(Workspace, workspace_id)
        if workspace is None:
            raise FootageRefused("the workspace is gone: nothing was submitted.")
        check_caps(media_spend(db, workspace, post_id), price_usd, what)
    except MediaCapExceeded as exc:
        raise FootageRefused(str(exc)) from exc
    finally:
        db.close()


# ── polling ─────────────────────────────────────────────────────────────────
@dataclass
class _Job:
    shot: Shot
    route: Route
    price: float
    handle: Any
    state: JobState = field(default_factory=lambda: JobState(RUNNING))
    poll_failures: int = 0
    # Given up on because its status could not be read: the provider may still finish it.
    unreadable: bool = False


async def _poll(executor: Any, workspace_id: UUID, job: _Job) -> None:
    try:
        job.state = await job.route.recipe.check(_asker(executor, workspace_id, job.route), job.shot, job.route, job.handle)
        job.poll_failures = 0
    except FootageToolError as exc:
        # The provider keeps the job: a status that cannot be read is asked again.
        job.poll_failures += 1
        logger.warning("[SocialsFootage] polling the %s failed (%d): %s", job.shot.slot, job.poll_failures, exc)
        if job.poll_failures >= MAX_POLL_FAILURES:
            job.state = JobState(FAILED, reason=f"its status could not be read: {exc}")
            job.unreadable = True


async def _wait(executor: Any, workspace_id: UUID, jobs: Sequence[_Job], deadline: float) -> None:
    """Poll every job until each ends or the footage's deadline passes."""
    while True:
        for job in jobs:
            if job.state.status == RUNNING:
                await _poll(executor, workspace_id, job)
        if all(job.state.status != RUNNING for job in jobs) or time.monotonic() >= deadline:
            return
        await asyncio.sleep(config.SOCIALS_FOOTAGE_POLL_SECONDS)


# ── booking (D13) ───────────────────────────────────────────────────────────
def _book_rows(workspace_id: UUID, post_id: UUID, rows: Sequence[Mapping[str, Any]]) -> None:
    """Book on the media lane against the post. Called off the event loop, the
    booking is written before this returns: the next cap check sees it."""
    with usage_scope(request_type=LANE_MEDIA, execution_id=post_execution_id(post_id), workspace_id=workspace_id):
        for row in rows:
            UsageTracker.track_media(**row)


def _units(shot: Shot) -> int:
    """What a shot is counted in: its seconds of footage, or one still."""
    return int(config.SOCIALS_FOOTAGE_CLIP_SECONDS) if shot.kind == VIDEO_SLOT else 1


def _priced_rows(jobs: Sequence[_Job], latency_ms: int) -> List[Dict[str, Any]]:
    """A pricing toolkit bills what it completes: each shot at its estimate, unless
    the toolkit said the job failed. A job still running when the wait ended, or
    whose status could not be read, is booked too: the toolkit may finish it."""
    return [
        {"provider": job.route.recipe.toolkit, "model_id": job.route.model, "units": _units(job.shot),
         "usd": job.price, "latency_ms": latency_ms}
        for job in jobs
        if job.state.status != FAILED or job.unreadable
    ]


def _credit_rows(recipe: FootageRecipe, jobs: Sequence[_Job], before: Optional[Decimal], after: Optional[Decimal],
                 latency_ms: int) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """A credit-billed toolkit's balance difference, one row, and each shot's share
    of it (by its price). A difference that cannot be read books the submitted
    shots at their ceilings, and says so."""
    if not jobs:
        return [], {}
    budget = sum(job.price for job in jobs)
    problem = None
    if before is None or after is None:
        usd, used = budget, 0.0
        problem = f"{recipe.label}'s balance could not be read after the jobs: booked at the ceiling"
        logger.error("[SocialsFootage] %s", problem)
    else:
        used = float(max(before - after, Decimal(0)))
        usd = used * recipe.usd_per_credit()
    models = sorted({job.route.model for job in jobs})
    row = {"provider": recipe.toolkit, "model_id": "+".join(models), "units": math.ceil(used), "usd": usd,
           "latency_ms": latency_ms, "error_message": problem}
    shares = {job.shot.slot: usd * (job.price / budget if budget else 1 / len(jobs)) for job in jobs}
    return [row], shares


# ── generating ──────────────────────────────────────────────────────────────
async def _run_toolkit(executor: Any, store: MediaStore, session_factory: Callable[[], Any],
                       group: Sequence[Tuple[Shot, Route, float]], *, workspace_id: UUID, post_id: UUID,
                       title: str, deadline: float) -> Tuple[Dict[str, Made], Dict[str, str]]:
    """One toolkit's shots: submit them all, poll them together, keep each that
    completes, and book what the toolkit spent, whatever happened."""
    recipe = group[0][1].recipe
    made: Dict[str, Made] = {}
    failed: Dict[str, str] = {}
    jobs: List[_Job] = []
    window_for_credits = credit_window(session_factory, workspace_id, recipe.toolkit) if recipe.credit_billed else nullcontext()
    async with window_for_credits:
        started = time.monotonic()
        before = await _read_balance(executor, workspace_id, group[0][1]) if recipe.credit_billed else None
        if recipe.credit_billed and before is None:
            why = (
                f"{recipe.label}'s balance could not be read, so what it would spend could not be booked: "
                "nothing was submitted to it"
            )
            return made, {shot.slot: why for shot, _, _ in group}
        shares: Dict[str, float] = {}
        try:
            for shot, route, price in group:
                try:
                    handle = await recipe.submit(_asker(executor, workspace_id, route), shot, route)
                except FootageToolError as exc:
                    failed[shot.slot] = str(exc)
                    continue
                jobs.append(_Job(shot=shot, route=route, price=price, handle=handle))
            await _wait(executor, workspace_id, jobs, deadline)
            for job in jobs:
                if job.state.status == RUNNING:
                    minutes = config.SOCIALS_FOOTAGE_MAX_WAIT_SECONDS // 60
                    failed[job.shot.slot] = f"{recipe.label} did not finish it within {minutes} minutes"
                elif job.state.status == FAILED:
                    failed[job.shot.slot] = job.state.reason or f"{recipe.label} could not make it"
                else:
                    try:
                        made[job.shot.slot] = await _keep(
                            store, session_factory, job.shot, job.route, job.price, job.state.file,
                            workspace_id=workspace_id, post_id=post_id, title=title,
                        )
                    except FootageError as exc:
                        failed[job.shot.slot] = str(exc)
        finally:
            latency_ms = int((time.monotonic() - started) * 1000)
            if recipe.credit_billed:
                after = await _read_balance(executor, workspace_id, group[0][1])
                rows, shares = _credit_rows(recipe, jobs, before, after, latency_ms)
            else:
                rows = _priced_rows(jobs, latency_ms)
            if rows:
                await asyncio.to_thread(_book_rows, workspace_id, post_id, rows)
    kept_shares = {slot: usd for slot, usd in shares.items() if slot in made}
    if kept_shares:
        await asyncio.to_thread(_record_costs, session_factory, workspace_id, post_id, kept_shares)
    return made, failed


async def generate(plan: FootagePlan, *, workspace_id: UUID, post_id: UUID, title: str,
                   session_factory: Callable[[], Any], store: MediaStore) -> Dict[str, Made]:
    """Generate the plan's shots (D12, D13): priced and capped first, then
    submitted, polled, copied into our storage, registered and marked done, and
    booked. :class:`FootageRefused` when nothing was submitted, and why;
    :class:`FootageError` naming each shot that failed (what was made is kept
    on the post, and the next render reuses it)."""
    if not plan.shots:
        return {}
    made: Dict[str, Made] = {}
    failed: Dict[str, str] = {}
    db = session_factory()
    try:
        executor = ComposioToolExecutor(db)
        async with window(session_factory, FOOTAGE_LOCK_NAMESPACE, str(workspace_id)):
            deadline = time.monotonic() + config.SOCIALS_FOOTAGE_MAX_WAIT_SECONDS
            priced = await _price(executor, workspace_id, plan.shots)
            total = sum(price for _, _, price in priced)
            shots = [shot for shot, _, _ in priced]
            await asyncio.to_thread(_check_caps, session_factory, workspace_id, post_id, total, _what(shots))
            groups: Dict[str, List[Tuple[Shot, Route, float]]] = {}
            for item in priced:
                groups.setdefault(item[1].recipe.toolkit, []).append(item)
            for group in groups.values():
                done, broken = await _run_toolkit(
                    executor, store, session_factory, group, workspace_id=workspace_id, post_id=post_id,
                    title=title, deadline=deadline,
                )
                made.update(done)
                failed.update(broken)
    finally:
        db.close()
    if failed:
        labels = {shot.slot: shot.label for shot, _ in plan.shots}
        raise FootageError("; ".join(f"{labels.get(slot, slot)}: {why}" for slot, why in failed.items()))
    return made
