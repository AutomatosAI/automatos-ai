"""PRD-251 D12: one small recipe per Composio generation toolkit (fal.ai, Kie.ai, Higgsfield MCP).

Each recipe maps "a 5 s 9:16 shot from this prompt", or a still, onto its
toolkit's own actions, always as submit then poll, never one long call:

* **fal.ai** (``fal_ai``): ``FAL_AI_ESTIMATE_PRICING`` prices a call first; then
  ``FAL_AI_SUBMIT_ASYNC_JOB`` → ``FAL_AI_QUEUE_GET_STATUS`` →
  ``FAL_AI_GET_QUEUE_REQUEST_RESULT``. fal bills dollars: a completed shot is
  booked at its estimate.
* **Kie.ai** (``kieai``): ``KIEAI_GENERATE_VEO_VIDEO`` → ``KIEAI_GET_VEO_VIDEO_DETAILS``,
  and ``KIEAI_GENERATE_FLUX_KONTEXT_IMAGE`` → ``KIEAI_GET_FLUX_KONTEXT_IMAGE_DETAILS``
  for a still. It bills credit (``KIEAI_GET_ACCOUNT_CREDITS``).
* **Higgsfield MCP** (``higgsfield_mcp``, the customer's Higgsfield account):
  ``HIGGSFIELD_MCP_GENERATE_VIDEO`` (or ``_GENERATE_IMAGE``), which takes ONE
  ``params`` JSON string, → ``HIGGSFIELD_MCP_JOBS_WAIT``. It bills credit
  (``HIGGSFIELD_MCP_BALANCE``).

A recipe names the slugs it calls, and the media capability registry decides
whether the workspace may call each (connected, allowlisted for that
capability, in the cached schemas, not denied). Parameter names are as
docs.composio.dev lists the actions (checked 2026-09-26); a parameter the
cached schema does not list is left out. Higgsfield's job and result fields are
not documented there, so its answers are read by the keys Higgsfield's platform
API uses, defensively; the owner's small-spend test confirms them (PRD-251
"Verify at build"). The default model per toolkit is config (open question 7).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional, Tuple

from config import config
from core.social_templates import IMAGE_SLOT, VIDEO_SLOT
from modules.socials.capabilities import (
    BALANCE,
    ESTIMATE,
    GENERATE_IMAGE,
    GENERATE_VIDEO,
    STATUS,
    MediaCapabilities,
    OfferedAction,
)
from modules.socials.recipes.files import ReturnedFile, returned_file
from modules.socials.recipes.toolkit import MAX_DEPTH, children, decimal_of

# What a slot's kind asks of a toolkit, and how the messages name it.
KIND_CAPABILITY = {VIDEO_SLOT: GENERATE_VIDEO, IMAGE_SLOT: GENERATE_IMAGE}
KIND_WORDS = {VIDEO_SLOT: "footage", IMAGE_SLOT: "a still"}

# The roles a recipe's actions play.
SUBMIT, POLL, RESULT, PRICE, CREDITS = "submit", "poll", "result", "price", "credits"

# A job's state as a poll reads it.
RUNNING, DONE, FAILED = "running", "done", "failed"

# The aspect ratios generation models take; a composition's size maps to the nearest.
COMMON_RATIOS = ("9:16", "16:9", "1:1", "4:5", "3:4", "4:3", "2:3", "3:2", "21:9")

# Where an answer carries the generated file, per kind, in the order trusted.
LINK_KEYS = {
    VIDEO_SLOT: ("video", "videos", "resultUrls", "result_urls", "video_url", "videoUrl", "outputs", "output", "results", "url"),
    IMAGE_SLOT: ("images", "image", "resultImageUrl", "result_image_url", "image_url", "imageUrl", "resultUrls",
                 "outputs", "output", "results", "url"),
}
URL_FIELDS = ("url", "video_url", "image_url", "file_url", "download_url", "uri")


class FootageError(Exception):
    """Footage could not be generated or kept; the render fails saying why."""


class FootageRefused(FootageError):
    """Nothing was submitted: a cap, a price or a balance said no."""


class FootageToolError(FootageError):
    """The toolkit answered a call with an error, or with nothing a recipe can use."""


@dataclass(frozen=True)
class Shot:
    """One slot to generate: what the post asks, and the shape the template shows it in."""

    slot: str
    kind: str  # video | image (the slot's kind)
    path: str  # the template's slot path, e.g. assets/slots/hook.mp4
    label: str
    prompt: str  # as the post asks it
    aspect_ratio: str


@dataclass(frozen=True)
class Route:
    """How one toolkit makes one kind: its recipe, its model and the offered actions it calls."""

    recipe: "FootageRecipe"
    kind: str
    model: str
    actions: Mapping[str, OfferedAction]  # role → the offered action


@dataclass(frozen=True)
class JobState:
    status: str  # RUNNING | DONE | FAILED
    file: Optional[ReturnedFile] = None
    reason: Optional[str] = None


# ``ask(role, params, required)``: the route's action for ``role``, called on the
# workspace's connection; its output, or FootageToolError.
Ask = Callable[..., Awaitable[Any]]


# ── reading an answer ───────────────────────────────────────────────────────
def find(answer: Any, keys: Tuple[str, ...], want: Callable[[Any], bool] = lambda value: value not in (None, "", [], {})) -> Any:
    """The first value under any of ``keys`` that ``want`` accepts, shallowest first."""
    frontier = [answer]
    for _ in range(MAX_DEPTH):
        objects = [item for item in frontier if isinstance(item, Mapping)]
        for key in keys:
            for obj in objects:
                if key in obj and want(obj[key]):
                    return obj[key]
        frontier = [value for item in frontier for value in children(item)]
        if not frontier:
            break
    return None


def _is_text(value: Any) -> bool:
    return (isinstance(value, str) and bool(value.strip())) or (isinstance(value, int) and not isinstance(value, bool))


def find_text(answer: Any, keys: Tuple[str, ...]) -> Optional[str]:
    value = find(answer, keys, _is_text)
    return str(value).strip() if value is not None else None


def _http_link(value: Any) -> Optional[str]:
    text = value.strip() if isinstance(value, str) else ""
    if text.lower().startswith(("https://", "http://")) and not any(c.isspace() for c in text):
        return text
    return None


def link_in(value: Any, depth: int = 0) -> Optional[str]:
    """A file link in ``value``: the text itself, an object's url, or a list's first."""
    if depth > MAX_DEPTH:
        return None
    if isinstance(value, str):
        return _http_link(value)
    if isinstance(value, Mapping):
        for field_name in URL_FIELDS:
            link = link_in(value.get(field_name), depth + 1)
            if link:
                return link
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            link = link_in(item, depth + 1)
            if link:
                return link
    return None


def media_file(answer: Any, kind: str) -> Optional[ReturnedFile]:
    """The generated file in a finished job's answer: a link under the kind's keys,
    else a Composio file output, a plain link or the bytes inline."""
    for key in LINK_KEYS[kind]:
        value = find(answer, (key,), lambda candidate: link_in(candidate) is not None)
        if value is not None:
            return ReturnedFile(url=link_in(value))
    return returned_file(answer)


def aspect_ratio(width: int, height: int) -> str:
    """The common aspect ratio nearest ``width`` × ``height`` (1080×1920 → 9:16)."""
    wanted = width / height if height else 1.0

    def distance(ratio: str) -> float:
        across, down = (int(part) for part in ratio.split(":"))
        return abs(across / down - wanted)

    return min(COMMON_RATIOS, key=distance)


def prompt_for(prompt: str) -> str:
    """The prompt as submitted: it ends with the guard words (no readable text, no
    logos), since every word on screen is template text (D12)."""
    guard = config.SOCIALS_FOOTAGE_PROMPT_GUARD
    text = prompt.strip()
    if not guard or guard.lower() in text.lower():
        return text
    return f"{text.rstrip('.').rstrip()}, {guard}"


# ── the recipes ─────────────────────────────────────────────────────────────
class FootageRecipe:
    """How one generation toolkit makes footage and stills (D12)."""

    toolkit = ""
    label = ""
    # fal prices a call before any spend; the others are budgeted at the ceilings.
    prices = False
    # A credit-billed toolkit books its balance difference (D13): the keys its
    # balance action carries the credit under, in the order trusted.
    credit_billed = False
    balance_keys: Tuple[str, ...] = ()

    def roles(self, kind: str) -> Dict[str, Tuple[str, str]]:
        """role → (action slug, the capability the allowlist must give it)."""
        raise NotImplementedError

    def model(self, kind: str) -> str:
        raise NotImplementedError

    def usd_per_credit(self) -> float:
        return 0.0

    def ceiling(self, kind: str) -> float:
        """The budget for a shot this toolkit prices nothing for (D13)."""
        return float(config.SOCIALS_FOOTAGE_CEILING_VIDEO_USD if kind == VIDEO_SLOT else config.SOCIALS_FOOTAGE_CEILING_IMAGE_USD)

    def route(self, kind: str, caps: MediaCapabilities) -> Tuple[Optional[Route], Optional[str]]:
        """This toolkit's route for ``kind`` when the registry offers every action it calls; else why not."""
        actions: Dict[str, OfferedAction] = {}
        for role, (slug, capability) in self.roles(kind).items():
            action = caps.action(self.toolkit, slug)
            if action is None or capability not in action.capabilities:
                return None, (
                    f"{self.label} does not offer {slug} here (not on the Socials media allowlist as {capability}, "
                    "not in its synced actions, or blocked)"
                )
            actions[role] = action
        model = self.model(kind)
        if not model:
            return None, f"{self.label} has no {kind} model set"
        return Route(recipe=self, kind=kind, model=model, actions=actions), None

    async def estimate(self, ask: Ask, model: str, count: int) -> float:
        raise NotImplementedError

    async def submit(self, ask: Ask, shot: Shot, route: Route) -> Any:
        """Submit the shot; the job to poll."""
        raise NotImplementedError

    async def check(self, ask: Ask, shot: Shot, route: Route, job: Any) -> JobState:
        raise NotImplementedError


# fal's queue API (docs.composio.dev fal_ai, fal.ai platform API, checked 2026-09-26).
FAL_SUBMIT = "FAL_AI_SUBMIT_ASYNC_JOB"
FAL_STATUS = "FAL_AI_QUEUE_GET_STATUS"
FAL_RESULT = "FAL_AI_GET_QUEUE_REQUEST_RESULT"
FAL_ESTIMATE = "FAL_AI_ESTIMATE_PRICING"
FAL_RUNNING = frozenset({"IN_QUEUE", "IN_PROGRESS"})
FAL_COMPLETED = "COMPLETED"
# One call's price as fal has billed it: whatever unit the model is priced in
# (per second, per video, per image), one call is one shot.
FAL_ESTIMATE_TYPE = "historical_api_price"
FAL_CURRENCY = "USD"


class FalRecipe(FootageRecipe):
    toolkit = "fal_ai"
    label = "fal.ai"
    prices = True

    def roles(self, kind: str) -> Dict[str, Tuple[str, str]]:
        return {
            SUBMIT: (FAL_SUBMIT, KIND_CAPABILITY[kind]),
            POLL: (FAL_STATUS, STATUS),
            RESULT: (FAL_RESULT, STATUS),
            PRICE: (FAL_ESTIMATE, ESTIMATE),
        }

    def model(self, kind: str) -> str:
        return config.SOCIALS_FOOTAGE_FAL_VIDEO_MODEL if kind == VIDEO_SLOT else config.SOCIALS_FOOTAGE_FAL_IMAGE_MODEL

    async def estimate(self, ask: Ask, model: str, count: int) -> float:
        answer = await ask(
            PRICE,
            {"estimate_type": FAL_ESTIMATE_TYPE, "endpoints": {model: {"call_quantity": count}}},
            ("estimate_type", "endpoints"),
        )
        total = decimal_of(find(answer, ("total_cost", "totalCost"), lambda value: decimal_of(value) is not None))
        currency = find_text(answer, ("currency",)) or ""
        # A price the answer does not say is in dollars is no price: the caps are in dollars.
        if total is None or total < 0 or currency.upper() != FAL_CURRENCY:
            raise FootageRefused(
                f"fal.ai did not price {model} in dollars, so its spend could not be checked against the caps: "
                "nothing was submitted."
            )
        return float(total)

    async def submit(self, ask: Ask, shot: Shot, route: Route) -> Any:
        request: Dict[str, Any] = {"prompt": prompt_for(shot.prompt), "aspect_ratio": shot.aspect_ratio}
        if shot.kind == VIDEO_SLOT:
            request["duration"] = str(config.SOCIALS_FOOTAGE_CLIP_SECONDS)
        answer = await ask(SUBMIT, {"model_id": route.model, "input": request}, ("model_id", "input"))
        request_id = find_text(answer, ("request_id", "requestId"))
        if not request_id:
            raise FootageToolError(f"fal.ai took the {shot.label.lower()} but gave no request id")
        return request_id

    async def check(self, ask: Ask, shot: Shot, route: Route, job: Any) -> JobState:
        ids = {"model_id": route.model, "request_id": job}
        status = (find_text(await ask(POLL, ids, ("model_id", "request_id")), ("status",)) or "").upper()
        if not status or status in FAL_RUNNING:
            return JobState(RUNNING)
        if status != FAL_COMPLETED:
            return JobState(FAILED, reason=f"fal.ai's job ended {status}")
        found = media_file(await ask(RESULT, ids, ("model_id", "request_id")), shot.kind)
        if found is None:
            return JobState(FAILED, reason=f"fal.ai finished with no {KIND_WORDS[shot.kind]} in its answer")
        return JobState(DONE, file=found)


# Kie.ai's task APIs (docs.composio.dev kieai, checked 2026-09-26): the Veo
# details take ``taskId``, the Flux Kontext details ``task_id``. successFlag:
# 0 generating, 1 done, 2 and 3 failed.
KIE_VEO = "KIEAI_GENERATE_VEO_VIDEO"
KIE_VEO_DETAILS = "KIEAI_GET_VEO_VIDEO_DETAILS"
KIE_FLUX = "KIEAI_GENERATE_FLUX_KONTEXT_IMAGE"
KIE_FLUX_DETAILS = "KIEAI_GET_FLUX_KONTEXT_IMAGE_DETAILS"
KIE_CREDITS = "KIEAI_GET_ACCOUNT_CREDITS"
KIE_GENERATING, KIE_DONE = 0, 1
KIE_IMAGE_FORMAT = "png"


class KieRecipe(FootageRecipe):
    toolkit = "kieai"
    label = "Kie.ai"
    credit_billed = True
    # Kie answers {"code", "msg", "data": <credits>}.
    balance_keys = ("credits", "data", "balance")

    def roles(self, kind: str) -> Dict[str, Tuple[str, str]]:
        submit, poll = (KIE_VEO, KIE_VEO_DETAILS) if kind == VIDEO_SLOT else (KIE_FLUX, KIE_FLUX_DETAILS)
        return {SUBMIT: (submit, KIND_CAPABILITY[kind]), POLL: (poll, STATUS), CREDITS: (KIE_CREDITS, BALANCE)}

    def model(self, kind: str) -> str:
        return config.SOCIALS_FOOTAGE_KIEAI_VIDEO_MODEL if kind == VIDEO_SLOT else config.SOCIALS_FOOTAGE_KIEAI_IMAGE_MODEL

    def usd_per_credit(self) -> float:
        return float(config.SOCIALS_KIEAI_USD_PER_CREDIT)

    async def submit(self, ask: Ask, shot: Shot, route: Route) -> Any:
        wanted: Dict[str, Any] = {"prompt": prompt_for(shot.prompt), "model": route.model, "aspect_ratio": shot.aspect_ratio}
        if shot.kind == IMAGE_SLOT:
            wanted["output_format"] = KIE_IMAGE_FORMAT
        answer = await ask(SUBMIT, wanted, ("prompt",))
        task = find_text(answer, ("taskId", "task_id"))
        if not task:
            why = find_text(answer, ("msg", "message", "error")) or "no task id in its answer"
            raise FootageToolError(f"Kie.ai did not start the {shot.label.lower()}: {why}")
        return task

    async def check(self, ask: Ask, shot: Shot, route: Route, job: Any) -> JobState:
        param = "taskId" if shot.kind == VIDEO_SLOT else "task_id"
        answer = await ask(POLL, {param: job}, (param,))
        flag = find(answer, ("successFlag",), lambda value: isinstance(value, int) and not isinstance(value, bool))
        if flag is None or flag == KIE_GENERATING:
            return JobState(RUNNING)
        if flag != KIE_DONE:
            why = find_text(answer, ("errorMessage", "msg")) or f"its task ended with flag {flag}"
            return JobState(FAILED, reason=f"Kie.ai could not make it: {why}")
        found = media_file(answer, shot.kind)
        if found is None:
            return JobState(FAILED, reason=f"Kie.ai finished with no {KIND_WORDS[shot.kind]} in its answer")
        return JobState(DONE, file=found)


# Higgsfield MCP (docs.composio.dev higgsfield_mcp, checked 2026-09-26): generate
# takes one ``params`` JSON string; JOBS_WAIT long-polls at most 15 s.
HF_VIDEO = "HIGGSFIELD_MCP_GENERATE_VIDEO"
HF_IMAGE = "HIGGSFIELD_MCP_GENERATE_IMAGE"
HF_WAIT = "HIGGSFIELD_MCP_JOBS_WAIT"
HF_BALANCE = "HIGGSFIELD_MCP_BALANCE"
HF_LONG_POLL_SECONDS = 15
HF_JOB_ID_KEYS = ("job_id", "jobId", "id")
HF_DONE = frozenset({"completed", "succeeded", "success", "done", "finished"})
HF_FAILED = frozenset({"failed", "error", "nsfw", "canceled", "cancelled", "rejected", "expired"})


@dataclass(frozen=True)
class HiggsfieldJob:
    id: str
    entry: Mapping[str, Any]  # the job as generate returned it (JOBS_WAIT may want it whole)


def _jobs_param(action: OfferedAction, job: HiggsfieldJob) -> list:
    """``jobs`` for JOBS_WAIT: the job objects when its cached schema takes objects, else their ids."""
    properties = action.parameters.get("properties") if isinstance(action.parameters, Mapping) else None
    jobs = properties.get("jobs") if isinstance(properties, Mapping) else None
    items = jobs.get("items") if isinstance(jobs, Mapping) else None
    if isinstance(items, Mapping) and items.get("type") == "object" and job.entry:
        return [dict(job.entry)]
    return [job.id]


def _job_entry(answer: Any, job: HiggsfieldJob) -> Any:
    """This job's entry in a JOBS_WAIT answer (by its id), else the answer itself."""
    listed = find(answer, ("jobs",), lambda value: isinstance(value, list))
    for entry in listed or ():
        if isinstance(entry, Mapping) and find_text(entry, HF_JOB_ID_KEYS) in (None, job.id):
            return entry
    return answer


class HiggsfieldRecipe(FootageRecipe):
    toolkit = "higgsfield_mcp"
    label = "Higgsfield"
    credit_billed = True
    balance_keys = ("credits",)

    def roles(self, kind: str) -> Dict[str, Tuple[str, str]]:
        submit = HF_VIDEO if kind == VIDEO_SLOT else HF_IMAGE
        return {SUBMIT: (submit, KIND_CAPABILITY[kind]), POLL: (HF_WAIT, STATUS), CREDITS: (HF_BALANCE, BALANCE)}

    def model(self, kind: str) -> str:
        if kind == VIDEO_SLOT:
            return config.SOCIALS_FOOTAGE_HIGGSFIELD_VIDEO_MODEL
        return config.SOCIALS_FOOTAGE_HIGGSFIELD_IMAGE_MODEL

    def usd_per_credit(self) -> float:
        return float(config.SOCIALS_HIGGSFIELD_USD_PER_CREDIT)

    async def submit(self, ask: Ask, shot: Shot, route: Route) -> Any:
        spec: Dict[str, Any] = {"model": route.model, "prompt": prompt_for(shot.prompt), "aspect_ratio": shot.aspect_ratio}
        if shot.kind == VIDEO_SLOT:
            spec["duration"] = config.SOCIALS_FOOTAGE_CLIP_SECONDS
        answer = await ask(SUBMIT, {"params": json.dumps(spec)}, ("params",))
        listed = find(answer, ("results", "jobs"), lambda value: isinstance(value, list) and bool(value))
        entry = next((item for item in listed or () if isinstance(item, Mapping)), None)
        ident = find_text(entry, HF_JOB_ID_KEYS) if entry else None
        if not ident and listed and isinstance(listed[0], str) and listed[0].strip():
            ident = listed[0].strip()
        ident = ident or find_text(answer, ("job_id", "jobId"))
        if not ident:
            why = find_text(answer, ("error", "notice")) or "no job in its answer"
            raise FootageToolError(f"Higgsfield did not start the {shot.label.lower()}: {why}")
        return HiggsfieldJob(id=ident, entry=entry or {})

    async def check(self, ask: Ask, shot: Shot, route: Route, job: Any) -> JobState:
        params = {"jobs": _jobs_param(route.actions[POLL], job), "timeout_seconds": HF_LONG_POLL_SECONDS}
        entry = _job_entry(await ask(POLL, params, ("jobs",)), job)
        status = (find_text(entry, ("status", "state")) or "").lower()
        if status in HF_FAILED:
            why = find_text(entry, ("error", "reason", "message")) or status
            return JobState(FAILED, reason=f"Higgsfield could not make it: {why}")
        if status not in HF_DONE:
            return JobState(RUNNING)
        found = media_file(entry, shot.kind)
        if found is None:
            return JobState(FAILED, reason=f"Higgsfield finished with no {KIND_WORDS[shot.kind]} in its answer")
        return JobState(DONE, file=found)


RECIPES: Mapping[str, FootageRecipe] = {
    recipe.toolkit: recipe for recipe in (FalRecipe(), KieRecipe(), HiggsfieldRecipe())
}
