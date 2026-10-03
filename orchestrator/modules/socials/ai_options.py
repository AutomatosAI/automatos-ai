"""PRD-251B Wave 3 (B10, D12, D13; US-B305): AI-made visuals in the editor's Look.

**AI image**: OPTIONS stills for one of the template's image slots, made through the
workspace's AI-images toolkit (its default first, US-B304) with the brand kit's style
profile after the prompt (US-B303). Each is copied into our storage and kept as a
Deliverable, as a render's footage is; the post's ``footage[slot]`` then lists them
(``options``). The person picks one: it becomes the slot's file exactly as if a render had
made it (``status: done``), so the next render shows it and makes nothing more. Every word
on screen stays template text (D12).

**AI footage** for a video's hook and b-roll is the footage the post asks for: its prompts
on ``post.footage``, made by the next render through the footage default.

Making options is spend: each still is priced by its toolkit, checked against the post's
and the month's caps, and booked, through ``recipes/footage.generate`` (D13). It runs in the
background; the slot says ``making``, then the options, or why it failed.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple
from uuid import UUID

from core.social_templates import IMAGE_SLOT
from modules.socials import service
from modules.socials.recipes import footage as footage_recipes
from modules.socials.recipes.footage_toolkits import Route, Shot, aspect_ratio

logger = logging.getLogger(__name__)

OPTIONS = 4
OPTIONS_KEY = "options"
STATE_KEY = "options_state"
ERROR_KEY = "options_error"
MAKING, READY, FAILED = "making", "ready", "failed"
OPTIONS_FAILED = "The options could not be made. Try again."
CONTENT_TYPES = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp"}


class OptionsRefused(service.InvalidPost):
    """The slot cannot have AI options now (422), and why."""


def assert_editable(post: Any) -> None:
    """:class:`service.IllegalTransition` (409) for a post that can no longer change."""
    if post.status not in service.EDITABLE_STATUSES:
        raise service.IllegalTransition(post.status, service.ACTION_EDIT)


def option_slot(slot: str, n: int) -> str:
    return f"{slot}_option_{n}"


def image_slot(blocks: Mapping[str, Any], slot: str) -> Mapping[str, Any]:
    """The template's image slot a toolkit may fill, or OptionsRefused."""
    spec = (blocks.get("slots") or {}).get(slot)
    if not isinstance(spec, Mapping) or spec.get("kind") != IMAGE_SLOT:
        raise OptionsRefused(f"{slot} is not one of this template's image slots")
    if spec.get("generate") is False:
        raise OptionsRefused(f"{spec.get('label') or slot} takes the workspace's own file, never a generated one")
    return spec


def plan_options(spec: Mapping[str, Any], slot: str, prompt: str, route: Route, *, width: int, height: int, style: str,
                 references: Tuple[str, ...] = ()) -> footage_recipes.FootagePlan:
    """OPTIONS shots of the one slot, each under its own name so each is kept as a file; the
    brand kit's style follows the prompt, and its liked references go when the toolkit takes one."""
    label = str(spec.get("label") or slot)
    ratio = aspect_ratio(width, height)
    shots = tuple(
        (Shot(slot=option_slot(slot, n), kind=IMAGE_SLOT, path=str(spec["path"]), label=f"{label} option {n}", prompt=prompt,
              aspect_ratio=ratio, style=style, references=references, record=False), route)
        for n in range(1, OPTIONS + 1)
    )
    return footage_recipes.FootagePlan(shots=shots)


def option_records(made: Mapping[str, Any], prompt: str) -> List[Dict[str, Any]]:
    """The made stills as the slot lists them: each a record a pick turns into the slot's file."""
    records = []
    for item in made.values():
        extension = item.name.rsplit(".", 1)[-1]
        records.append({
            "prompt": prompt, "toolkit": item.toolkit, "model": item.model, "deliverable_id": item.deliverable_id,
            "name": item.name, "sha256": item.sha256,
            "bytes": item.bytes, "content_type": CONTENT_TYPES.get(extension, "image/png"),
            "estimate_usd": item.estimate_usd, "generated_at": datetime.now(timezone.utc).isoformat(),
        })
    return sorted(records, key=lambda record: record["name"])


def ask(post: Any, slot: str, prompt: str) -> None:
    """The slot asks for options of ``prompt``: they are being made (the caller commits)."""
    footage = dict(post.footage) if isinstance(post.footage, dict) else {}
    footage[slot] = {"prompt": prompt, OPTIONS_KEY: [], STATE_KEY: MAKING}
    post.footage = footage


def settle(post: Any, slot: str, prompt: str, options: List[Dict[str, Any]], error: Optional[str]) -> bool:
    """The options made for ``prompt``, and why any were not, while the slot still asks for them:
    ready with every option made (some may have failed: the error says which), else failed."""
    footage = dict(post.footage) if isinstance(post.footage, dict) else {}
    asked = footage.get(slot)
    if not isinstance(asked, dict) or asked.get("prompt") != prompt or asked.get(STATE_KEY) != MAKING:
        return False
    state = READY if options else FAILED
    footage[slot] = {"prompt": prompt, OPTIONS_KEY: options, STATE_KEY: state, **({ERROR_KEY: error} if error else {})}
    post.footage = footage
    return True


def pick(post: Any, slot: str, name: str) -> bool:
    """The option named ``name`` becomes the slot's file, as a render's would; ``False`` when there is no such option."""
    footage = dict(post.footage) if isinstance(post.footage, dict) else {}
    asked = footage.get(slot)
    options = asked.get(OPTIONS_KEY) if isinstance(asked, dict) else None
    chosen = next((o for o in options or [] if isinstance(o, dict) and o.get("name") == name), None)
    if chosen is None:
        return False
    footage[slot] = {**chosen, "prompt": asked["prompt"], "status": service.FOOTAGE_DONE}
    post.footage = footage
    return True


async def make(plan: footage_recipes.FootagePlan, *, workspace_id: UUID, post_id: UUID, title: str, slot: str, prompt: str,
               session_factory: Callable[[], Any], store: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Make the options (priced, capped, booked); the records of those made, and why any were not.
    Never raises: whatever happens, the slot is settled and leaves "making"."""
    try:
        made = await footage_recipes.generate(plan, workspace_id=workspace_id, post_id=post_id, title=title,
                                              session_factory=session_factory, store=store)
    except footage_recipes.FootageError as exc:
        logger.warning("[SocialsAIOptions] post %s slot %s: %s", post_id, slot, exc)
        return option_records(exc.made, prompt), str(exc)  # the ones made are kept and booked: offer them
    except Exception:  # any other failure still settles the slot, saying so
        logger.exception("[SocialsAIOptions] post %s slot %s: the options could not be made", post_id, slot)
        return [], OPTIONS_FAILED
    return option_records(made, prompt), None
