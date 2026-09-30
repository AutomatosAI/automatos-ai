"""PRD-251 D8, D14 (US-301): one target's steps, run through Composio.

This is where a post leaves Automatos, and the only code that passes the post gate's
way through (``PLATFORM_PUBLISHER``, ``core/composio/post_gate.py``). Every call goes
through ``ComposioToolExecutor.execute_with_uploads`` on the workspace's own
connection: the Wave 0 deny list first (a denied step fails its target, or is skipped
when optional), the post gate, then exactly the step's ``files`` as the call's own
upload spec, never the executor's global ``UPLOAD_ACTIONS``.

* **Files and links.** A media param in the step's ``files`` is staged from our
  storage as a local file (:class:`Stager`) and uploaded by the executor; one in
  ``urls`` takes a presigned inline link a platform can fetch (D9,
  ``media_urls.public_media_url``). With no public storage an optional step is
  skipped and the receipt says so; a required one fails its target saying so.
* **Status steps** are called every ``SOCIALS_PUBLISH_POLL_SECONDS`` until their
  ``until`` says done or failed, or ``SOCIALS_PUBLISH_MAX_WAIT_SECONDS`` pass.
* **Failures** (D8) carry the platform's own message and whether they are transient:
  a timeout, a 5xx, a 429 or a connection error. A 4xx, a refusal by the deny list
  or the post gate, or a platform's refusal is not.
* **Resuming.** ``outputs`` keeps what each step returned, so a retried attempt
  starts at the step that failed: a file uploaded once is not uploaded again.

The channel differences are the adapter DATA: nothing here names a channel.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Awaitable, Callable, Dict, List, Mapping, MutableMapping, Optional, Tuple
from urllib.parse import quote

from config import config
from core.composio.deny_list import ERROR_TYPE_DENIED
from core.composio.post_gate import ERROR_TYPE_POST_GATE, PLATFORM_PUBLISHER
from modules.socials import media_urls
from modules.socials.capabilities import PUBLISH, ChannelStep
from modules.socials.media_urls import MediaFile, MediaHostingError
from modules.socials.publish_sources import MediaParam, SourceMissing, TargetContext, resolve_params
from modules.socials.recipes.toolkit import PLATFORM_AGENT_ID, output_of
from modules.socials.step_results import DONE, ID, PERMALINK, poll_state, returned

logger = logging.getLogger(__name__)

ERROR_MAX_CHARS = 1000
FAILED_STATE = "failed"
# Transient (D8): worth another attempt. Anything else is the platform's answer: a
# message that carries a 4xx status (429 aside) never is, whatever else it says.
_TRANSIENT = re.compile(
    r"time[d ]?\s?out|timeout|temporar|connection (?:error|reset|refused|aborted)|connecterror|"
    r"remote ?protocol|\b5\d\d\b(?!\s*(?:char|byte|word|item|second|ms\b|px|kb|mb))|\b429\b|"
    r"rate.?limit|too many requests|service unavailable|bad gateway|gateway time",
    re.IGNORECASE,
)
_CLIENT_ERROR = re.compile(r"\b4(?!29)\d\d\b(?!\s*(?:char|byte|word|item|second|ms\b|px|kb|mb))")
_NEVER_TRANSIENT = frozenset({ERROR_TYPE_DENIED, ERROR_TYPE_POST_GATE})


class StepFailure(Exception):
    """A step did not do its work: the platform's message, and whether to try again."""

    def __init__(self, message: str, *, transient: bool) -> None:
        self.message = message[:ERROR_MAX_CHARS]
        self.transient = transient
        super().__init__(self.message)


class Stager:
    """Stages a post's stored file as a local file the executor uploads."""

    def __init__(self, workdir: Path, client: Any = None) -> None:
        self.workdir = workdir
        self._client = client
        self._staged: Dict[str, Path] = {}

    def _download(self, media: MediaFile) -> Path:
        from core.storage import get_s3_client

        if media.key is None:
            raise media_urls.MediaUnavailable(media.error or media_urls.NOT_IN_STORAGE)
        folder = self.workdir / media.deliverable_id
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / PurePosixPath(media.key).name
        (self._client or get_s3_client()).download_file(config.S3_DOCUMENTS_BUCKET, media.key, str(path))
        return path

    async def stage(self, media: MediaFile) -> Path:
        if media.deliverable_id not in self._staged:
            self._staged[media.deliverable_id] = await asyncio.to_thread(self._download, media)
        return self._staged[media.deliverable_id]


@dataclass
class Runtime:
    """What a publish runs with: tests hand in a fake executor, sleep and clock."""

    executor: Any
    stager: Stager
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep
    clock: Callable[[], float] = time.monotonic
    public_link: Callable[[Any, MediaFile], str] = media_urls.public_media_url


@dataclass
class TargetProgress:
    """What a target's steps have done so far, kept across its attempts."""

    outputs: MutableMapping[str, Mapping[str, Any]] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)


# ---- one call ---------------------------------------------------------------


def _envelope_error(result: Mapping[str, Any]) -> Optional[str]:
    """Composio's own ``{"successful": false, "error"}`` inside a successful call."""
    data = result.get("data")
    if isinstance(data, Mapping) and data.get("successful") is False:
        return str(data.get("error") or "the platform refused the call")
    return None


def _failure(step: ChannelStep, result: Mapping[str, Any]) -> StepFailure:
    message = str(result.get("error") or _envelope_error(result) or "no reason given").strip()
    transient = (
        result.get("error_type") not in _NEVER_TRANSIENT
        and not _CLIENT_ERROR.search(message)
        and bool(_TRANSIENT.search(message))
    )
    return StepFailure(f"{step.action}: {message}", transient=transient)


async def _call(step: ChannelStep, params: Dict[str, Any], ctx: TargetContext, rt: Runtime) -> Any:
    result = await rt.executor.execute_with_uploads(
        step.action,
        params,
        agent_id=PLATFORM_AGENT_ID,
        workspace_id=ctx.workspace_id,
        app_name=ctx.toolkit.upper(),
        upload_params=step.files,
        way_through=PLATFORM_PUBLISHER,
    )
    if not result.get("success") or _envelope_error(result):
        raise _failure(step, result)
    return output_of(result)


async def _poll(step: ChannelStep, params: Dict[str, Any], ctx: TargetContext, rt: Runtime) -> Any:
    budget = config.SOCIALS_PUBLISH_MAX_WAIT_SECONDS
    deadline = rt.clock() + budget
    while True:
        output = None
        try:
            output = await _call(step, params, ctx, rt)
        except StepFailure as failure:
            if not failure.transient:
                raise
        if output is not None:
            state, why = poll_state(output, step.until)
            if state == DONE:
                return output
            if state == FAILED_STATE:
                raise StepFailure(f"{step.action} reported a failure: {why}", transient=False)
        if rt.clock() >= deadline:
            raise StepFailure(f"{step.action} did not finish within {budget} seconds", transient=False)
        await rt.sleep(config.SOCIALS_PUBLISH_POLL_SECONDS)


# ---- files and links ----------------------------------------------------------


async def _materialize(name: str, value: MediaParam, step: ChannelStep, ctx: TargetContext, rt: Runtime) -> Any:
    if name in step.files:
        items = [await rt.stager.stage(media) for media in value.files]
    else:
        post = _PostRef(ctx.post_id, ctx.workspace_id)
        items = [await asyncio.to_thread(rt.public_link, post, media) for media in value.files]
    return items if value.many else items[0]


@dataclass(frozen=True)
class _PostRef:
    """What ``media_urls.public_media_url`` reads of a post."""

    id: Any
    workspace_id: Any


async def _params(step: ChannelStep, ctx: TargetContext, rt: Runtime, progress: TargetProgress) -> Dict[str, Any]:
    try:
        params = resolve_params(step, ctx, progress.outputs)
        for name, value in list(params.items()):
            if isinstance(value, MediaParam):
                params[name] = await _materialize(name, value, step, ctx, rt)
    except (SourceMissing, MediaHostingError) as exc:
        raise StepFailure(f"{step.action}: {exc}", transient=False) from exc
    return params


async def _run_step(step: ChannelStep, ctx: TargetContext, rt: Runtime, progress: TargetProgress) -> None:
    params = await _params(step, ctx, rt, progress)
    output = await (_poll if step.until else _call)(step, params, ctx, rt)
    progress.outputs[step.id] = returned(output, step.returns)


# ---- a target ---------------------------------------------------------------


def receipt(steps: Tuple[ChannelStep, ...], outputs: Mapping[str, Mapping[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    """The target's remote id (its last publish step's ``id``) and its permalink (the
    last one a step returned, else the publish step's link template)."""
    published = [step for step in steps if step.step_class == PUBLISH and step.id in outputs]
    last = published[-1] if published else None
    remote = outputs[last.id].get(ID) if last else None
    link = next((outputs[s.id][PERMALINK] for s in reversed(steps) if PERMALINK in outputs.get(s.id, {})), None)
    if link is None and remote is not None and last is not None and last.permalink:
        link = last.permalink.replace("{id}", quote(str(remote), safe=":"))
    return (str(remote) if remote is not None else None), (str(link) if link is not None else None)


async def run_steps(
    steps: Tuple[ChannelStep, ...], ctx: TargetContext, rt: Runtime, progress: TargetProgress,
) -> Tuple[Optional[str], Optional[str]]:
    """Run the steps not done yet, in order: the receipt, or :class:`StepFailure`.
    An optional step that cannot run is skipped, and ``progress.notes`` says why."""
    for step in steps:
        if step.id in progress.outputs:
            continue
        try:
            await _run_step(step, ctx, rt, progress)
        except StepFailure as failure:
            if not step.optional:
                raise
            logger.info("[Socials] optional step %s of post %s skipped: %s", step.action, ctx.post_id, failure.message)
            progress.notes.append(f"Skipped: {failure.message}")
            progress.outputs[step.id] = {}
    return receipt(steps, progress.outputs)
