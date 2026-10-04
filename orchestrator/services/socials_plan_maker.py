"""PRD-251B Wave 2 (B7; US-B205): a plan's posts are made on their day.

The scheduler leader runs :func:`run_tick` every ``SOCIALS_PLAN_TICK_SECONDS``
(:func:`register`, called by ``services/schedule_reconcile.start_schedule_reconcile``).
For each active plan of a workspace with Socials on, each due slot
(``plans.due_slots``: its make time has come, its time has not, no post holds its key)
is made, at most ``make.max_per_day`` a local day and ``SOCIALS_PLAN_MAX_SLOTS_PER_TICK``
a tick:

1. A channel it names must be connected and post its format, and the render it needs
   must fit the workspace's render minutes. Otherwise the slot is skipped and the
   workspace told: never half-made.
2. The next unused topic that suits its format, one pinned to its day first. An empty
   bank skips the slot and tells the workspace, once a day.
3. The claim: a draft with the slot's key (unique per plan), its planned time and the
   plan as its campaign, the topic marked used, committed. A tick that loses the race
   for the slot moves on.
4. The composer writes it, the topic's facts as its claims' sources; the draft takes
   the proposal and the row's channels, then renders (a template) or goes for approval
   (text). Either way it ends in ``needs_approval`` and approvers are told.
5. Once per plan and day: "Today's posts are ready".

PRD-251C (C1, US-C203): a weekly or monthly plan makes its whole batch once its moment
comes (``modules/socials/batches.py``), each post carrying the batch's key. A slot it cannot
make is skipped and recorded on the plan, so the batch can end; once every slot of a batch
is made or skipped, "Your week is ready" (or month) goes out once, linked to the Queue,
instead of the daily notice.

A slot's work runs in a worker thread on its own session; the composer and the render
start run on the event loop from there, as the routes do (F105).

The plan's visual mix (US-B305, ``modules/socials/plan_visuals.py``): each slot draws its
visual from the mix. AI images or AI footage ask the template's slots for it, each with the
composer's prompt (the render makes them through the workspace's default toolkit, capped and
booked); the library makes a fitting image or video Deliverable the post's media, as the
editor's Library does, and the post goes for approval with no render.
"""
from __future__ import annotations

import functools
import logging
from dataclasses import replace
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple
from uuid import UUID

import anyio
import sqlalchemy as sa
from sqlalchemy.exc import IntegrityError

from config import config
from core import media_render_quota as render_quota
from core.models.core import DocumentTemplate
from core.models.socials import SocialCampaign, SocialPost, SocialTopic
from core.models.workspaces import Workspace
from core.social_cuts import cut_to_length
from modules.socials import batches, compose, plan_notify, plan_store, plan_visuals, plans, render, service, topics, upload_crops
from modules.socials.capabilities import social_channels
from modules.socials.settings import socials_off_reason

logger = logging.getLogger(__name__)

PLAN_TICK_JOB_ID = "socials_plan_tick"
PLAN_AGENT = "the plan"
MADE, TAKEN, GONE, NO_TOPIC, SKIPPED, FAILED = "made", "taken", "gone", "no_topic", "skipped", "failed"
READY_NOTE = "Made from the plan's content bank: {topic}."
# PRD-251C (C1): why a batch slot whose time came before the tick made it is recorded as skipped.
PASSED_UNMADE = "its time came before it could be made"
# The post kinds a post format publishes as, most fitting first (the composer's order).
FORMAT_KINDS: Mapping[str, Tuple[str, ...]] = {
    "video": ("video", "reel", "short"),
    "image": ("image",),
    "carousel": ("carousel", "image"),
    "fact_card": ("image",),
    "infographic": ("image",),
    "text": ("text",),
}
# A fact's source kind (B8) → the post source kind a claim binds to (D7). A note backs no claim.
FACT_SOURCE_KINDS = {"knowledge": "document", "deliverable": "deliverable", "web": "url", "github": "url"}
STILL_FORMATS = ("image", "carousel", "fact_card", "infographic")
# The library's newest Deliverables of the kind a post needs, which a library visual picks from.
LIBRARY_CANDIDATES = 50
_LIBRARY = sa.text(
    """
    SELECT id, title, summary FROM deliverables
     WHERE workspace_id = :workspace_id AND deleted_at IS NULL AND artifact_type = :kind
     ORDER BY created_at DESC LIMIT :limit
    """
)


def _session() -> Any:
    from core.database.database import SessionLocal

    return SessionLocal()


def _posts_api() -> Any:
    from api import socials

    return socials


def _local_today(plan: SocialCampaign, now: datetime) -> date:
    return now.astimezone(plans.zone_of(plan)).date()


def _tell_once(db: Any, plan: SocialCampaign, event: tuple, title: str, now: datetime) -> None:
    """A per-day plan notice, once (the plan remembers the day); committed first."""
    if plan_notify.once_today(plan, event[0], _local_today(plan, now)):
        db.commit()
        plan_notify.notify_plan(plan.workspace_id, plan.id, event, title)


# ── which slots ────────────────────────────────────────────────────────────


def _within_daily_cap(plan: SocialCampaign, slots: List[plans.Slot], taken: set) -> List[plans.Slot]:
    cap = plans.make_settings(plan)["max_per_day"]
    if not cap:
        return slots
    counts: Dict[date, int] = {}
    for key in taken:
        parsed = plans.parse_slot_key(key)
        if parsed is not None:
            counts[parsed[1]] = counts.get(parsed[1], 0) + 1
    kept = []
    for slot in slots:
        if counts.get(slot.local_date, 0) < cap:
            counts[slot.local_date] = counts.get(slot.local_date, 0) + 1
            kept.append(slot)
    return kept


def _plan_due(db: Any, plan: SocialCampaign, now: datetime) -> List[plans.Slot]:
    workspace = db.get(Workspace, plan.workspace_id)
    if workspace is None or socials_off_reason(workspace) is not None:
        return []
    taken = plan_store.taken_keys(db, plan)
    found = batches.due_slots(plan, now, taken) if batches.is_batched(plan) else plans.due_slots(plan, now, taken)
    return _within_daily_cap(plan, found, taken)


def collect_due(now: datetime) -> List[Tuple[UUID, str]]:
    """The (plan, slot key) pairs to make now, earliest first, at most a tick's worth."""
    db = _session()
    try:
        due: List[Tuple[datetime, UUID, str]] = []
        for plan in plan_store.active_plans(db):
            due += [(slot.at, plan.id, slot.key) for slot in _plan_due(db, plan, now)]
        return [(plan_id, key) for _at, plan_id, key in sorted(due)][: int(config.SOCIALS_PLAN_MAX_SLOTS_PER_TICK)]
    finally:
        db.close()


# ── one slot ───────────────────────────────────────────────────────────────


def _kind_for(channel: Any, post_format: str, row_kind: Optional[str] = None) -> Optional[str]:
    """The kind a channel posts the slot as: a story row's story (PRD-251C), else the
    format's most fitting kind the channel offers now."""
    available = [kind.kind for kind in channel.post_kinds if kind.available]
    wanted = (row_kind,) if row_kind else FORMAT_KINDS.get(post_format, ())
    return next((kind for kind in wanted if kind in available), None)


def slot_targets(db: Any, workspace_id: UUID, slot: plans.Slot) -> List[Dict[str, Any]]:
    """The slot's channels that are connected and post its format (or its row's kind), each with its kind."""
    channels = {channel.toolkit: channel for channel in social_channels(db, workspace_id)}
    found = [(toolkit, _kind_for(channels[toolkit], slot.format, slot.kind)) for toolkit in slot.channels if toolkit in channels]
    return [{"toolkit": toolkit, "post_kind": kind, "options": {}} for toolkit, kind in found if kind]


def render_seconds(db: Any, workspace_id: UUID, slot: plans.Slot) -> int:
    """The render minutes the slot's post needs, in seconds: a video's length; stills and text none."""
    if slot.format != "video":
        return 0
    template = db.get(DocumentTemplate, UUID(slot.template_id)) if slot.template_id else None
    if template is None or template.workspace_id != workspace_id or not isinstance(template.blocks, dict):
        return int(slot.length_seconds or config.SOCIALS_RENDER_MAX_DURATION_SECONDS)
    blocks = cut_to_length(template.blocks, slot.length_seconds) if slot.length_seconds else template.blocks
    return render_quota.declared_seconds(blocks, template.format)


def fits_quota(db: Any, workspace: Workspace, seconds: int) -> bool:
    if seconds <= 0:
        return True
    quota = render_quota.render_quota(db, workspace)
    if quota.quota_minutes is None:
        return True
    reserved = render_quota.render_seconds_reserved(db, workspace.id)
    return quota.used_seconds + reserved + seconds <= quota.quota_minutes * 60


def _refusal(db: Any, workspace: Workspace, slot: plans.Slot) -> Optional[str]:
    if not slot_targets(db, workspace.id, slot):
        return f"no connected channel posts a {slot.kind or slot.format} for {', '.join(slot.channels)}"
    if not fits_quota(db, workspace, render_seconds(db, workspace.id, slot)):
        return "this month's render minutes are used up"
    return None


def topic_brief(topic: SocialTopic) -> str:
    lines = [topic.title]
    if topic.angle:
        lines.append(topic.angle)
    lines += [f"- {fact['text']} ({fact['source']['label']})" for fact in topic.facts or [] if isinstance(fact, dict)]
    return "\n".join(lines)


def fact_candidates(topic: SocialTopic, now: datetime) -> List[Dict[str, Any]]:
    """The topic's facts as sources a claim may bind to (a note backs none)."""
    found = []
    for fact in topic.facts or []:
        source = fact.get("source") or {}
        kind = FACT_SOURCE_KINDS.get(source.get("kind"))
        if kind:
            found.append({"kind": kind, "ref": source.get("ref"), "title": source.get("label"), "detail": fact.get("text"), "as_of": now.isoformat()})
    return found


def claim(db: Any, plan: SocialCampaign, slot: plans.Slot, topic: SocialTopic, now: datetime) -> Optional[SocialPost]:
    """The slot's draft, committed with the topic marked used; ``None`` when another tick holds the slot."""
    post = service.create_draft(
        db, workspace_id=plan.workspace_id, created_by=plan.created_by, title=topic.title, brief=topic_brief(topic),
        format=slot.format, template_id=slot.template_id, length_seconds=slot.length_seconds, agent=PLAN_AGENT,
    )
    post.campaign_id, post.slot_key = plan.id, slot.key
    post.batch_key = batches.batch_key(plan, slot.local_date)  # PRD-251C: None for a daily plan
    service.set_planned_for(post, slot.at, plan.timezone)
    topics.mark_used(topic, post, now)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        return None
    return post


def _propose(db: Any, plan: SocialCampaign, slot: plans.Slot, topic: SocialTopic, now: datetime,
             visual_slots: List[Dict[str, str]]) -> Dict[str, Any]:
    from api import socials_compose as compose_api

    request = compose_api.ComposeRequest(
        brief=topic_brief(topic)[: compose_api.BRIEF_MAX_CHARS], channels=list(slot.channels), format=slot.format,
        template_id=UUID(slot.template_id) if slot.template_id else None, length_seconds=slot.length_seconds,
    )
    context = compose_api.compose_context(db, plan.workspace_id, request)
    context = replace(context, candidates=[*fact_candidates(topic, now), *context.candidates], visual_slots=tuple(visual_slots))
    timeout = float(config.SOCIALS_COMPOSE_TIMEOUT_SECONDS)
    return anyio.from_thread.run(compose.propose, context, compose.llm_factory(plan.workspace_id), timeout)


def _changes(proposal: Mapping[str, Any], slot: plans.Slot) -> Dict[str, Any]:
    copy = proposal.get("copy") or {}
    changes = {
        "title": proposal.get("title") or None,
        "copy": {"base": copy.get("base") or "", "channels": dict(copy.get("per_channel") or {})},
        "variables": dict(proposal.get("variables") or {}),
        "sources": dict(proposal.get("sources") or {}),
    }
    if slot.template_id is None and proposal.get("template_id"):
        changes["template_id"] = proposal["template_id"]
    return {key: value for key, value in changes.items() if value is not None}


def template_blocks(db: Any, workspace_id: UUID, template_id: Any) -> Optional[Mapping[str, Any]]:
    """The composition of the workspace's template ``template_id``; ``None`` for anything else."""
    template = db.get(DocumentTemplate, UUID(str(template_id))) if template_id else None
    return render.composition_of(template) if template is not None and template.workspace_id == workspace_id else None


def library_media(db: Any, workspace_id: UUID, post_format: str, topic: SocialTopic) -> Optional[Dict[str, Any]]:
    """The edit that makes a library Deliverable fitting the topic the post's media, as the
    editor's Library does; ``None`` when nothing in the library fits."""
    kind = "video" if post_format == "video" else "image"
    rows = db.execute(_LIBRARY, {"workspace_id": workspace_id, "kind": kind, "limit": LIBRARY_CANDIDATES}).mappings().all()
    found = plan_visuals.best_fit(rows, topic.title, topic.angle)
    if found is None:
        return None
    return {"media": {"original": [str(found["id"])]}, "template_id": None, "length_seconds": None, "format": kind}


def _visual_changes(db: Any, plan: SocialCampaign, slot: plans.Slot, topic: SocialTopic, proposal: Mapping[str, Any],
                    visual: str) -> Dict[str, Any]:
    """The edit the slot's visual adds to the composer's: AI-made slots, or a library file."""
    if visual == plan_visuals.TEMPLATES:
        return {}
    if visual == plan_visuals.LIBRARY:
        return library_media(db, plan.workspace_id, slot.format, topic) or {}
    asked = plan_visuals.ai_slots(template_blocks(db, plan.workspace_id, proposal.get("template_id")), visual)
    if not asked:
        return {}
    fallback = plan_visuals.topic_prompt(topic.title, topic.angle)
    return {"footage": plan_visuals.footage_asks(asked, proposal.get("visual_prompts") or {}, fallback, slot.visual_toolkit)}


def write(db: Any, workspace: Workspace, plan: SocialCampaign, slot: plans.Slot, topic: SocialTopic, post: SocialPost, now: datetime) -> None:
    """The claimed draft written by the composer with the slot's visual, given its channels,
    then rendered or submitted."""
    from api import socials_targets

    posts_api, actor = _posts_api(), plan.created_by
    visual = slot.visual_source or plan_visuals.visual_for(plans.make_settings(plan)["visual_mix"], slot.key)  # PRD-251C US-C302
    known = plan_visuals.ai_slots(template_blocks(db, plan.workspace_id, slot.template_id), visual) if visual in plan_visuals.SLOT_KIND else []
    proposal = _propose(db, plan, slot, topic, now, known)
    changes = {**_changes(proposal, slot), **_visual_changes(db, plan, slot, topic, proposal, visual)}
    anyio.from_thread.run(functools.partial(posts_api.edit_post, db, post, actor, changes, agent=PLAN_AGENT))
    socials_targets.set_post_targets(db, post, actor, slot_targets(db, plan.workspace_id, slot), agent=PLAN_AGENT)
    if post.template_id is not None or upload_crops.own_still(post) is not None:  # PRD-251C US-C303: a still is cropped
        anyio.from_thread.run(posts_api.render_post, db, workspace, post, actor)
    else:
        posts_api.submit_post(db, post, actor, note=READY_NOTE.format(topic=topic.title))


def _skip(db: Any, plan: SocialCampaign, slot: plans.Slot, refusal: str, now: datetime) -> None:
    """A slot the plan cannot make: told once a day. A batch also records it, so it can end."""
    key = batches.batch_key(plan, slot.local_date)
    if key is not None:
        plan.make = batches.with_skip(plan, key, slot.key, refusal, _local_today(plan, now))
        db.commit()
    _tell_once(db, plan, plan_notify.SLOT_SKIPPED, f"{plan.name}: {refusal}", now)


def _make(db: Any, plan_id: UUID, key: str, now: datetime) -> str:
    plan = db.get(SocialCampaign, plan_id)
    slot = plans.slot_for_key(plan, key) if plan is not None and plan.status == plans.ACTIVE else None
    if slot is None or slot.at <= now:
        return GONE
    workspace = db.get(Workspace, plan.workspace_id)
    refusal = _refusal(db, workspace, slot)
    if refusal is not None:
        _skip(db, plan, slot, refusal, now)
        return SKIPPED
    topic = topics.next_topic(db, plan, slot.format, slot.local_date)
    if topic is None:
        _tell_once(db, plan, plan_notify.BANK_EMPTY, f"{plan.name}: research again or add topics", now)
        return NO_TOPIC
    post = claim(db, plan, slot, topic, now)
    if post is None:
        return TAKEN
    try:
        write(db, workspace, plan, slot, topic, post, now)
    except Exception:  # noqa: BLE001 — the claimed draft stays for a person; logged and told
        logger.exception("[Socials] plan %s: the post for slot %s could not be written", plan_id, key)
        db.rollback()
        plan_notify.notify_plan(plan.workspace_id, plan.id, plan_notify.SLOT_SKIPPED, f"{plan.name}: the draft for {topic.title} waits for you")
        return FAILED
    return MADE


def make_slot(plan_id: UUID, key: str, now: datetime) -> str:
    """Make one slot's post (the module docstring), on its own session. Never raises."""
    db = _session()
    try:
        return _make(db, plan_id, key, now)
    except Exception:  # noqa: BLE001 — one slot never stops the tick; logged
        logger.exception("[Socials] plan %s: slot %s failed", plan_id, key)
        db.rollback()
        return FAILED
    finally:
        db.close()


def tell_ready(made: Mapping[UUID, int], now: datetime) -> None:
    """"Today's posts are ready", once per day for each daily plan (a batch has its own notice)."""
    db = _session()
    try:
        for plan_id, count in made.items():
            plan = db.get(SocialCampaign, plan_id)
            if plan is not None and not batches.is_batched(plan):
                _tell_once(db, plan, plan_notify.READY, f"{plan.name}: {count} waiting for approval", now)
    finally:
        db.close()


def _batch_post_count(db: Any, plan: SocialCampaign, key: str) -> int:
    return db.query(SocialPost.id).filter(SocialPost.campaign_id == plan.id, SocialPost.batch_key == key).count()


def _ready_title(plan: SocialCampaign, window: batches.Window, count: int, passed: int) -> str:
    """"Countdown: 6 posts for the week of 19 Oct; 1 passed before it could be made"."""
    title = f"{plan.name}: {count} post{'' if count == 1 else 's'} for {batches.label(plan, window)}"
    if not passed:
        return title
    return f"{title}; {passed} passed before {'it' if passed == 1 else 'they'} could be made"


def _announce_if_complete(db: Any, plan: SocialCampaign, window: batches.Window, now: datetime) -> None:
    """"Your week is ready", once, when every slot of the batch is made or skipped. A slot whose
    time came before the tick made it is recorded as skipped then, and the notice names it."""
    taken = plan_store.taken_keys(db, plan)
    if batches.announced(plan, window.key) or batches.pending(plan, window.key, now, taken):
        return
    passed = [slot.key for slot in batches.passed_unmade(plan, window.key, now, taken)]
    count = _batch_post_count(db, plan, window.key)
    if count == 0 and not passed:
        return
    today = _local_today(plan, now)
    if passed:
        plan.make = batches.with_skips(plan, window.key, passed, PASSED_UNMADE, today)
    plan.make = batches.with_record(plan, window.key, {batches.ANNOUNCED: now.isoformat()}, today)
    db.commit()
    event = plan_notify.MONTH_READY if batches.rhythm_of(plan) == plans.MONTHLY else plan_notify.WEEK_READY
    plan_notify.notify_review(plan.workspace_id, plan.id, event, _ready_title(plan, window, count, len(passed)))


def announce_batches(touched: Mapping[UUID, Iterable[str]], now: datetime) -> None:
    """Each batch the tick made or skipped slots of, told once it is complete (C1)."""
    db = _session()
    try:
        for plan_id, keys in touched.items():
            plan = db.get(SocialCampaign, plan_id)
            if plan is None or not batches.is_batched(plan):
                continue
            days = {parsed[1] for parsed in (plans.parse_slot_key(key) for key in keys) if parsed}
            windows = {window.key: window for window in (batches.window_of(plan, day) for day in days) if window}
            for window in windows.values():
                _announce_if_complete(db, plan, window, now)
    except Exception:  # noqa: BLE001 — a notice never fails the tick; the next tick tries again
        logger.exception("[Socials] the batch notices could not be sent")
        db.rollback()
    finally:
        db.close()


async def run_tick(now: Optional[datetime] = None) -> Dict[str, int]:
    """One pass over every active plan's due slots, the batches', the evening reminders and the
    weekly notes (PRD-251C), then due research (US-B204)."""
    from services import socials_plan_reminders, socials_plan_research, socials_weekly_notes

    now = now or datetime.now(timezone.utc)
    outcomes: Dict[str, int] = {}
    made: Dict[UUID, int] = {}
    touched: Dict[UUID, Set[str]] = {}
    try:
        due = await anyio.to_thread.run_sync(collect_due, now)
    except Exception:  # noqa: BLE001 — the next tick tries again
        logger.exception("[Socials] the plan tick could not read its due slots")
        return outcomes
    for plan_id, key in due:
        outcome = await anyio.to_thread.run_sync(make_slot, plan_id, key, now)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == MADE:
            made[plan_id] = made.get(plan_id, 0) + 1
        if outcome in (MADE, SKIPPED):
            touched[plan_id] = {*touched.get(plan_id, set()), key}
    if made:
        await anyio.to_thread.run_sync(tell_ready, made, now)
    if touched:
        await anyio.to_thread.run_sync(announce_batches, touched, now)
    outcomes["reminded"] = await anyio.to_thread.run_sync(socials_plan_reminders.remind_due, _session, now)
    outcomes["weekly_notes"] = await anyio.to_thread.run_sync(socials_weekly_notes.send_due, _session, now)
    outcomes["research"] = await anyio.to_thread.run_sync(socials_plan_research.launch_due, now)
    return outcomes


def register(scheduler: Any) -> bool:
    """The tick on the leader's scheduler; ``False`` when this worker hosts none."""
    if scheduler is None or not getattr(scheduler, "running", False):
        return False
    from apscheduler.triggers.interval import IntervalTrigger

    scheduler.add_job(
        run_tick, IntervalTrigger(seconds=int(config.SOCIALS_PLAN_TICK_SECONDS)), id=PLAN_TICK_JOB_ID,
        replace_existing=True, max_instances=1, coalesce=True,
    )
    logger.info("[Socials] the plan tick runs every %ds", int(config.SOCIALS_PLAN_TICK_SECONDS))
    return True
