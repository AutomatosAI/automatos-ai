"""
Local-edition first-run seed — the two-minute demo (PRD-233 S3)
================================================================

Runs inside the existing entrypoint seed step (``core.database.load_seed_data``)
and only in the local edition: ``config.AUTH_EDITION == "local"`` with a
``config.DEFAULT_WORKSPACE_ID``. SaaS never enters this module's write path.

What it seeds, all scoped to the DEFAULT_WORKSPACE_ID workspace:

* the workspace row itself and the single operator ``users`` row — the same
  idempotent shape as ``docker-entrypoint.sh`` / ``scripts/init_test_db.py`` —
  so the content below is attributable on the very first boot regardless of
  which lifecycle step runs first;
* Auto, through the existing per-workspace seeder (``seed_auto_agent`` —
  reused, never duplicated);
* a three-agent starter roster (Researcher / Writer / Analyst) that uses the
  platform's native tools only — no Composio app assignments;
* one demo Playbook (``workflow_recipes``) the roster can run with nothing
  but an LLM key: no integrations, no worker;
* a welcome Deliverable — a ``blog_posts`` row, the one member of the
  ``v_workspace_outputs`` union whose content lives in the database, so the
  Deliverables tab opens it with no worker / object-storage round-trip.

Idempotent-REFRESH — the ``seed_auto_agent`` persona-hash pattern, generalised:

* identity per row is stable: agent ``slug`` / Playbook ``template_id`` /
  post ``slug``, each within the workspace;
* a row whose content fingerprint equals the CURRENT seed is left alone;
* a row whose fingerprint matches a PRIOR shipped version
  (``PRIOR_SEED_FINGERPRINTS``) is refreshed to the current content;
* any other fingerprint is a user edit — never overwritten (``customized``);
* a seeded row the user deleted is not resurrected: the workspace's
  ``settings["local_first_run"]`` ledger records what was seeded once.

Running the seed twice yields identical state — nothing is assigned unless it
differs, so no UPDATE is emitted. Seed content lives here; the database is the
runtime source (CLAUDE.md §4 — no file hacks).
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Optional
from uuid import UUID

from sqlalchemy import inspect as sa_inspect, text
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from config import config
from core.models.core import Agent, BlogPost, User, WorkflowTemplate
from core.models.workspaces import Workspace
from core.seeds.seed_auto_agent import seed_auto_agent

# NOTE: nothing from core.services is imported here on purpose. The seed
# loader (core/database/load_seed_data.py) puts /app/core at sys.path[0], so
# under it `core/services/__init__` → analytics_engine → core.redis resolves
# `import redis` to the app's own core/redis package and dies with
# "No module named 'redis.asyncio'". Keep this module's import chain to
# models + the Auto seeder.

logger = logging.getLogger(__name__)

# Same estimate BlogService uses (200 wpm, minimum 1) — kept local, see above.
_READING_WORDS_PER_MINUTE = 200

LEDGER_KEY = "local_first_run"  # workspaces.settings[LEDGER_KEY]
LEDGER_VERSION = 1

# Same shape as docker-entrypoint.sh ensure_local_workspace() and
# scripts/init_test_db.py: one workspace, one operator; users.id 1 is
# api/chat.py's own fallback (SELECT id FROM users WHERE id = 1).
LOCAL_WORKSPACE_NAME = "Local Workspace"
LOCAL_WORKSPACE_SLUG = "local"
LOCAL_OPERATOR_USER_ID = 1
LOCAL_OPERATOR_USERNAME = "local"
LOCAL_OPERATOR_PLACEHOLDER_NAME = "Local Operator"  # PRD-233 S6 makes it editable

# Fingerprints of PRIOR shipped versions of the content below. When you change
# any seed text, add the fingerprint the OLD content produced (capture it with
# ``current_fingerprints()`` before editing) so untouched rows keep refreshing.
# Rows hashing to none of current ∪ prior are user edits and are never touched.
PRIOR_SEED_FINGERPRINTS: frozenset[str] = frozenset()

# ── Roster ───────────────────────────────────────────────────────────────────

ROSTER_TEAM = "Starter team"
ROSTER_AGENT_TYPE = "specialized"

ROSTER: tuple[dict[str, Any], ...] = (
    {
        "slug": "local-researcher",
        "name": "Researcher",
        "job_title": "Research Analyst",
        "marketplace_category": "Research",
        "description": (
            "Gathers the facts on a topic — from the workspace's documents and "
            "knowledge and from what the model already knows — and hands back "
            "research notes that say plainly what is known and what is assumed."
        ),
        "persona": (
            "You are the Researcher on a small starter team.\n"
            "- Collect the facts that matter for the topic you are given; use the "
            "workspace's own documents and knowledge first when they exist.\n"
            "- Label every point as established, likely, or unverified — never "
            "present a guess as a fact.\n"
            "- Hand over compact research notes (bullets, with sources where you "
            "have them) that a writer can build on without re-reading everything."
        ),
        "responsibilities": [
            "Collect and organise the facts on a topic",
            "Separate what is known from what is assumed",
            "Hand over research notes the Writer can build on",
        ],
        "tags": ["research", "starter", "local-edition"],
    },
    {
        "slug": "local-writer",
        "name": "Writer",
        "job_title": "Content Writer",
        "marketplace_category": "Marketing",
        "description": (
            "Turns research notes into finished prose — briefs, summaries, "
            "posts, emails — in the length and tone asked for, without adding "
            "claims the notes do not support."
        ),
        "persona": (
            "You are the Writer on a small starter team.\n"
            "- Write from the material you are given; do not invent facts to "
            "fill gaps — flag them instead.\n"
            "- Plain language, short paragraphs, a clear structure, no hype.\n"
            "- Deliver the requested format (Markdown by default) and nothing "
            "else; the reader should be able to use it as-is."
        ),
        "responsibilities": [
            "Turn research notes into finished, structured prose",
            "Keep to the requested length, tone and format",
            "Flag gaps rather than filling them with guesses",
        ],
        "tags": ["writing", "starter", "local-edition"],
    },
    {
        "slug": "local-analyst",
        "name": "Analyst",
        "job_title": "Business Analyst",
        "marketplace_category": "Operations",
        "description": (
            "Reviews drafts, data and plans: finds the gaps, risks and "
            "unsupported claims, then returns a corrected version with a short "
            "list of what changed and what to do next."
        ),
        "persona": (
            "You are the Analyst on a small starter team.\n"
            "- Review what you are given critically: gaps, risks, unsupported "
            "claims, missing next steps.\n"
            "- Be specific and brief — one line per finding, most important first.\n"
            "- Always return the corrected work in full so it can be used "
            "immediately, and say when nothing needed changing."
        ),
        "responsibilities": [
            "Review drafts and plans for gaps, risks and unsupported claims",
            "Return corrected work ready to use",
            "Recommend concrete next steps",
        ],
        "tags": ["analysis", "review", "starter", "local-edition"],
    },
)

# ── Demo Playbook ────────────────────────────────────────────────────────────

PLAYBOOK_TEMPLATE_ID = "local-two-minute-brief"
PLAYBOOK_TOPIC_DEFAULT = "How a small team can put AI agents to work"

# Steps carry the roster slug; the stored rows carry the resolved ``agent_id``
# (what the executor and the create route's agent check read). ``agent`` is
# NOT a step key — GET /{id} reserves it for enrichment.
PLAYBOOK: dict[str, Any] = {
    "template_id": PLAYBOOK_TEMPLATE_ID,
    "name": "Two-minute brief",
    "description": (
        "Research, write and review a one-page brief on any topic. Three steps, "
        "three agents, nothing required beyond your LLM key — run it to see the "
        "starter roster work together, then change the topic or the prompts."
    ),
    "inputs": {
        "topic": {
            "type": "string",
            "required": True,
            "description": "What the brief should cover",
            "default": PLAYBOOK_TOPIC_DEFAULT,
        },
    },
    "outputs": {
        "final_brief": {
            "type": "string",
            "description": "The reviewed one-page brief (the Analyst's output)",
        },
    },
    "steps": (
        {
            "step_id": "research",
            "order": 1,
            "agent_slug": "local-researcher",
            "error_handling": "stop",
            "output_key": "research_notes",
            "prompt_template": (
                "Research this topic: {input.topic}\n\n"
                "Produce research notes for a one-page brief: 6-10 bullet points "
                "covering what it is, why it matters, the main options or "
                "approaches, and the pitfalls. Mark each point as established, "
                "likely, or unverified. Use the workspace's knowledge tools first "
                "if any are available; otherwise draw on what you know and say "
                "so. Finish with the three questions a decision-maker would "
                "still ask."
            ),
        },
        {
            "step_id": "write",
            "order": 2,
            "agent_slug": "local-writer",
            "error_handling": "stop",
            "output_key": "draft_brief",
            "prompt_template": (
                "Write a one-page brief on: {input.topic}\n\n"
                "Use the Researcher's notes from the previous step as your "
                "source material — do not add claims they do not support. "
                "Structure: a title, a two-sentence summary, three short "
                "sections with headings, and a closing 'Next steps' list of "
                "three actions. Plain language, no hype, under 450 words. "
                "Output the brief in Markdown."
            ),
        },
        {
            "step_id": "review",
            "order": 3,
            "agent_slug": "local-analyst",
            "error_handling": "stop",
            "output_key": "final_brief",
            "prompt_template": (
                "Review the brief from the previous step on: {input.topic}\n\n"
                "First list up to three gaps, risks or unsupported claims (one "
                "line each). Then output the corrected brief in full under the "
                "heading 'Final brief', with your fixes applied and nothing else "
                "changed. If nothing needs fixing, say so and repeat the brief "
                "unchanged."
            ),
        },
    ),
    # Timeouts in seconds (the executor normalises); same defaults the create
    # route applies when a caller omits execution_config.
    "execution_config": {
        "mode": "sequential",
        "max_retries": 1,
        "timeout_per_step": 300,
        "total_timeout": 900,
        "auto_learning": True,
    },
    "schedule_config": {"type": "manual"},
    "template_definition": {"steps": [], "agents": [], "config": {}, "variables": []},
    "tags": ["demo", "starter", "local-edition", "brief"],
    "version": "1.0",
}

# ── Welcome Deliverable ──────────────────────────────────────────────────────

WELCOME_SLUG = "welcome-to-automatos-local-edition"
WELCOME_TITLE = "Welcome to Automatos (local edition)"
WELCOME_AUTHOR = "Auto"
WELCOME_CATEGORY = "guide"
WELCOME_TAGS = ["welcome", "local-edition", "getting-started"]
WELCOME_EXCERPT = (
    "What is already in this workspace, the one thing to do first, and how to "
    "run the demo Playbook."
)
WELCOME_CONTENT = """# Welcome to Automatos (local edition)

This workspace runs entirely on your machine — the API, Postgres, Redis and object storage in Docker. There is no account and no login: the session you are using is the workspace operator.

## What is already here

- **Auto** — the workspace orchestrator. Chat with Auto to create agents, run Playbooks and manage the workspace.
- **A starter roster** — Researcher, Writer and Analyst, on the Agents page. They use the platform's native tools only.
- **A demo Playbook** — *Two-minute brief*, on the Playbooks page. Researcher → Writer → Analyst produce a reviewed one-page brief on a topic you choose.
- **This Deliverable** — the first entry in your Deliverables tab. Everything your agents produce lands there.

## The one thing to do first

Agents need a language model. Add an API key under **Settings → API Keys** (OpenRouter, OpenAI, Anthropic, Google and others are supported). Nothing answers until a key is present; everything listed above works with only that key.

## Try it

1. Open **Playbooks** and run *Two-minute brief* with a topic of your own.
2. Follow the execution log step by step; the final brief is the Analyst's output.
3. Ask Auto in chat: "What can I do here?"

## What is not here yet

- Third-party integrations (Slack, GitHub, Google Workspace and the rest) run through Composio. Without a `COMPOSIO_API_KEY` in your `.env` they are not offered.
- Agents acting on files on your own machine use the workspace worker service — the self-hosting guide covers how to enable it.

Everything seeded here is yours to edit or delete. Seeded items are refreshed by platform updates only while they are untouched; once you change one, it stays as you left it.
"""

# ── Fingerprints ─────────────────────────────────────────────────────────────

_AGENT_CONTENT_FIELDS: tuple[str, ...] = (
    "name", "description", "job_title", "team", "agent_type",
    "marketplace_category", "status", "use_custom_persona",
    "custom_persona_prompt", "tags", "responsibilities", "configuration",
    "model_config",
)
_PLAYBOOK_CONTENT_FIELDS: tuple[str, ...] = (
    "name", "description", "template_definition", "inputs", "outputs",
    "execution_config", "schedule_config", "tags", "recommended_agents",
    "required_tools", "is_public", "is_featured", "version",
)
_PLAYBOOK_REFRESH_FIELDS: tuple[str, ...] = _PLAYBOOK_CONTENT_FIELDS + ("steps",)
_POST_CONTENT_FIELDS: tuple[str, ...] = (
    "title", "excerpt", "content", "tags", "category", "status", "author_name",
)
_POST_REFRESH_FIELDS: tuple[str, ...] = _POST_CONTENT_FIELDS + ("reading_time_minutes",)


def _fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def agent_fingerprint(row: Any) -> str:
    """Content fingerprint of a roster row (ORM row or column namespace)."""
    return _fingerprint({f: getattr(row, f, None) for f in _AGENT_CONTENT_FIELDS})


def _normalized_steps(steps: list[dict[str, Any]], id_to_slug: dict[int, str]) -> list[dict[str, Any]]:
    """Steps with ``agent_id`` replaced by the roster slug — the stable identity —
    so a re-created roster agent (new id) still hashes as seed content."""
    normalized = []
    for step in steps:
        agent_id = step.get("agent_id")
        rest = {k: v for k, v in step.items() if k != "agent_id"}
        normalized.append({**rest, "agent_slug": id_to_slug.get(agent_id, agent_id)})
    return normalized


def playbook_fingerprint(row: Any, id_to_slug: dict[int, str]) -> str:
    """Content fingerprint of a Playbook row, agent ids normalised to slugs."""
    payload = {f: getattr(row, f, None) for f in _PLAYBOOK_CONTENT_FIELDS}
    payload["steps"] = _normalized_steps(list(getattr(row, "steps", None) or []), id_to_slug)
    return _fingerprint(payload)


def post_fingerprint(row: Any) -> str:
    """Content fingerprint of the welcome post."""
    return _fingerprint({f: getattr(row, f, None) for f in _POST_CONTENT_FIELDS})


# ── Column builders (the ONE definition of what a seeded row holds) ──────────

def _agent_columns(spec: dict[str, Any], workspace_id: UUID, operator: Optional[User]) -> dict[str, Any]:
    return {
        "name": spec["name"],
        "slug": spec["slug"],
        "description": spec["description"],
        "job_title": spec["job_title"],
        "team": ROSTER_TEAM,
        "agent_type": ROSTER_AGENT_TYPE,
        "marketplace_category": spec["marketplace_category"],
        "status": "active",
        "use_custom_persona": True,
        "custom_persona_prompt": spec["persona"],
        "tags": list(spec["tags"]),
        "responsibilities": list(spec["responsibilities"]),
        "configuration": {},
        # No model_config: like wizard-created agents, the roster resolves its
        # LLM at runtime from the workspace's configured provider/key.
        "model_config": None,
        "workspace_id": workspace_id,
        "owner_type": "workspace",
        "owner_id": str(workspace_id),
        "is_system_agent": False,
        "is_shared": True,
        "created_by": config.LOCAL_OPERATOR_EMAIL,
        "created_by_user_id": operator.id if operator else None,
    }


def _stored_steps(steps: tuple[dict[str, Any], ...], slug_to_id: dict[str, int]) -> list[dict[str, Any]]:
    stored = []
    for step in steps:
        rest = {k: deepcopy(v) for k, v in step.items() if k != "agent_slug"}
        stored.append({**rest, "agent_id": slug_to_id[step["agent_slug"]]})
    return stored


def _playbook_columns(
    spec: dict[str, Any], workspace_id: UUID, operator: Optional[User], slug_to_id: dict[str, int]
) -> dict[str, Any]:
    return {
        "template_id": spec["template_id"],
        "name": spec["name"],
        "description": spec["description"],
        "workspace_id": workspace_id,
        "owner_type": "workspace",
        "owner_id": str(workspace_id),
        "template_definition": deepcopy(spec["template_definition"]),
        "steps": _stored_steps(spec["steps"], slug_to_id),
        "inputs": deepcopy(spec["inputs"]),
        "outputs": deepcopy(spec["outputs"]),
        "execution_config": deepcopy(spec["execution_config"]),
        "schedule_config": deepcopy(spec["schedule_config"]),
        "tags": list(spec["tags"]),
        "recommended_agents": [],
        "required_tools": [],  # native tools only — nothing to install
        "is_public": True,
        "is_featured": True,
        "is_system": False,  # the operator may delete it; the ledger keeps it deleted
        "is_approved": False,
        "version": spec["version"],
        "created_by": config.LOCAL_OPERATOR_EMAIL,
        "created_by_user_id": operator.id if operator else None,
    }


def _reading_time_minutes(content: str) -> int:
    return max(1, math.ceil(len(content.split()) / _READING_WORDS_PER_MINUTE))


def _post_columns(workspace_id: UUID, auto_agent_id: Optional[int]) -> dict[str, Any]:
    return {
        "workspace_id": workspace_id,
        "author_agent_id": auto_agent_id,
        "author_name": WELCOME_AUTHOR,
        "title": WELCOME_TITLE,
        "slug": WELCOME_SLUG,
        "excerpt": WELCOME_EXCERPT,
        "content": WELCOME_CONTENT,
        "file_path": None,  # content stays in the DB — no workspace file to fetch
        "tags": list(WELCOME_TAGS),
        "category": WELCOME_CATEGORY,
        "status": "published",
        "reading_time_minutes": _reading_time_minutes(WELCOME_CONTENT),
    }


def validate_playbook(recipe: WorkflowTemplate) -> None:
    """The create route's validation path (api/workflow_recipes.create), raised
    instead of HTTP-mapped. Seed time and tests share it."""
    for check in (recipe.validate_steps, recipe.validate_execution_config, recipe.validate_schedule_config):
        ok, error = check()
        if not ok:
            raise ValueError(f"demo Playbook is invalid: {error}")


def current_fingerprints() -> dict[str, str]:
    """{identity: fingerprint} of the content as currently shipped. Capture
    BEFORE editing seed text and add the old values to PRIOR_SEED_FINGERPRINTS."""
    ws = UUID(int=0)
    slug_to_id = {spec["slug"]: index + 1 for index, spec in enumerate(ROSTER)}
    id_to_slug = {v: k for k, v in slug_to_id.items()}
    out = {
        spec["slug"]: agent_fingerprint(SimpleNamespace(**_agent_columns(spec, ws, None)))
        for spec in ROSTER
    }
    out[PLAYBOOK_TEMPLATE_ID] = playbook_fingerprint(
        SimpleNamespace(**_playbook_columns(PLAYBOOK, ws, None, slug_to_id)), id_to_slug
    )
    out[WELCOME_SLUG] = post_fingerprint(SimpleNamespace(**_post_columns(ws, None)))
    return out


# ── Refresh mechanics ────────────────────────────────────────────────────────

def _assign(row: Any, field: str, value: Any) -> bool:
    """Assign only when different (no UPDATE for identical content). New
    objects for JSON values; flagged so a reassigned JSON(B) column persists."""
    if getattr(row, field, None) == value:
        return False
    setattr(row, field, deepcopy(value))
    if isinstance(value, (dict, list)) and sa_inspect(row, raiseerr=False) is not None:
        flag_modified(row, field)
    return True


def _backfill_attribution(row: Any, columns: dict[str, Any]) -> None:
    """A seed-owned row created before the operator existed gets its
    created_by_user_id once the user row is there (never on customized rows)."""
    if not hasattr(row, "created_by_user_id") or columns.get("created_by_user_id") is None:
        return
    if getattr(row, "created_by_user_id", None) is None:
        row.created_by_user_id = columns["created_by_user_id"]


def _refresh(
    row: Any,
    columns: dict[str, Any],
    fields: tuple[str, ...],
    fingerprint: Callable[[Any], str],
    current: str,
) -> str:
    """current ⇒ untouched; a PRIOR shipped version ⇒ refreshed to the current
    content; anything else is the user's edit ⇒ customized, never overwritten."""
    found = fingerprint(row)
    if found == current:
        _backfill_attribution(row, columns)
        return "current"
    if found not in PRIOR_SEED_FINGERPRINTS:
        return "customized"
    for field in fields:
        _assign(row, field, columns[field])
    _backfill_attribution(row, columns)
    return "refreshed"


# ── Workspace, operator, ledger ──────────────────────────────────────────────

def _ensure_workspace(db: Session, workspace_id: UUID) -> str:
    result = db.execute(
        text(
            "INSERT INTO workspaces (id, name, slug, is_personal, is_active) "
            "VALUES (CAST(:id AS uuid), :name, :slug, TRUE, TRUE) "
            "ON CONFLICT (id) DO NOTHING"
        ),
        {"id": str(workspace_id), "name": LOCAL_WORKSPACE_NAME, "slug": LOCAL_WORKSPACE_SLUG},
    )
    return "created" if result.rowcount == 1 else "present"


def _ensure_operator_user(db: Session) -> tuple[Optional[User], str]:
    """The single operator, resolved by config.LOCAL_OPERATOR_EMAIL (how
    api/chat.py and hybrid.py find it). Inserted only when missing — a name the
    operator typed later (S6) is never touched."""
    email = config.LOCAL_OPERATOR_EMAIL
    user = db.query(User).filter(User.email == email).first()
    if user is not None:
        return user, "present"
    db.execute(
        text(
            "INSERT INTO users (id, username, email, name, is_active) "
            "VALUES (:id, :username, :email, :name, TRUE) ON CONFLICT DO NOTHING"
        ),
        {
            "id": LOCAL_OPERATOR_USER_ID,
            "username": LOCAL_OPERATOR_USERNAME,
            "email": email,
            "name": LOCAL_OPERATOR_PLACEHOLDER_NAME,
        },
    )
    db.execute(text(
        "SELECT setval(pg_get_serial_sequence('users','id'), GREATEST((SELECT max(id) FROM users), 1))"
    ))
    user = db.query(User).filter(User.email == email).first()
    if user is not None:
        return user, "created"
    logger.warning(
        "PRD-233 S3: could not seed the local operator (%s) — users.id %s or username %r "
        "is already taken by another row; seeded content will carry no user attribution",
        email, LOCAL_OPERATOR_USER_ID, LOCAL_OPERATOR_USERNAME,
    )
    return None, "unresolved"


@dataclass
class _SeedContext:
    db: Session
    workspace_id: UUID
    operator: Optional[User]
    ledger: dict[str, set[str]]


def _read_ledger(workspace: Workspace) -> dict[str, set[str]]:
    seeded = ((workspace.settings or {}).get(LEDGER_KEY) or {}).get("seeded") or {}
    return {kind: set(seeded.get(kind) or []) for kind in ("agents", "playbooks", "deliverables")}


def _present_identities(ctx: _SeedContext) -> dict[str, set[str]]:
    db, ws = ctx.db, ctx.workspace_id
    roster_slugs = [c for spec in ROSTER for c in _slug_candidates(spec["slug"], ws)]
    agents = {
        _base_slug(slug, ws) for (slug,) in db.query(Agent.slug)
        .filter(Agent.workspace_id == ws, Agent.slug.in_(roster_slugs)).all()
    }
    playbooks = {
        _base_slug(tid, ws) for (tid,) in db.query(WorkflowTemplate.template_id)
        .filter(WorkflowTemplate.workspace_id == ws, WorkflowTemplate.template_id.in_(_slug_candidates(PLAYBOOK_TEMPLATE_ID, ws))).all()
    }
    posts = {
        slug for (slug,) in db.query(BlogPost.slug)
        .filter(BlogPost.workspace_id == ws, BlogPost.slug == WELCOME_SLUG).all()
    }
    return {"agents": agents, "playbooks": playbooks, "deliverables": posts}


def _record_ledger(workspace: Workspace, ctx: _SeedContext) -> bool:
    """Remember every identity seeded at least once (rebuild, never mutate the
    stored dict). Returns True when the workspace row changed."""
    present = _present_identities(ctx)
    entry = {
        "version": LEDGER_VERSION,
        "seeded": {kind: sorted(ctx.ledger[kind] | present[kind]) for kind in ctx.ledger},
    }
    settings = dict(workspace.settings or {})
    if settings.get(LEDGER_KEY) == entry:
        return False
    workspace.settings = {**settings, LEDGER_KEY: entry}
    if sa_inspect(workspace, raiseerr=False) is not None:
        flag_modified(workspace, "settings")
    return True


# ── Upserts ──────────────────────────────────────────────────────────────────

# agents.slug is GLOBALLY unique (idx_agents_slug_unique), while the roster's
# identity is per workspace. One local install has one workspace, so the plain
# slug is the norm; when another workspace already owns it (the CI seed tests'
# throwaway workspaces, a second local workspace), THIS workspace stores a
# suffixed slug. Every lookup and fingerprint maps back to the base slug, so the
# seed's refresh contract is identical in both cases.
def _ws_suffix(workspace_id: UUID) -> str:
    return str(workspace_id).replace("-", "")[:8]


def _slug_candidates(base: str, workspace_id: UUID) -> tuple[str, str]:
    return (base, f"{base}-{_ws_suffix(workspace_id)}")


def _base_slug(stored: str, workspace_id: UUID) -> str:
    suffix = f"-{_ws_suffix(workspace_id)}"
    return stored[: -len(suffix)] if stored.endswith(suffix) else stored


def _free_slug(db, base: str, workspace_id: UUID) -> str:
    taken_elsewhere = (
        db.query(Agent.id).filter(Agent.slug == base, Agent.workspace_id != workspace_id).first() is not None
    )
    return _slug_candidates(base, workspace_id)[1] if taken_elsewhere else base


def _upsert_agent(ctx: _SeedContext, spec: dict[str, Any]) -> str:
    row = (
        ctx.db.query(Agent)
        .filter(
            Agent.workspace_id == ctx.workspace_id,
            Agent.slug.in_(_slug_candidates(spec["slug"], ctx.workspace_id)),
        )
        .first()
    )
    columns = _agent_columns(spec, ctx.workspace_id, ctx.operator)
    if row is None:
        if spec["slug"] in ctx.ledger["agents"]:
            return "deleted_by_user"
        columns["slug"] = _free_slug(ctx.db, spec["slug"], ctx.workspace_id)
        ctx.db.add(Agent(**columns))
        ctx.db.flush()
        return "created"
    columns["slug"] = row.slug  # identity is the base slug; the stored one stays
    current = agent_fingerprint(SimpleNamespace(**columns))
    return _refresh(row, columns, _AGENT_CONTENT_FIELDS, agent_fingerprint, current)


def _roster_ids(ctx: _SeedContext) -> dict[str, int]:
    candidates = [c for spec in ROSTER for c in _slug_candidates(spec["slug"], ctx.workspace_id)]
    rows = (
        ctx.db.query(Agent.slug, Agent.id)
        .filter(Agent.workspace_id == ctx.workspace_id, Agent.slug.in_(candidates))
        .all()
    )
    return {_base_slug(slug, ctx.workspace_id): agent_id for slug, agent_id in rows}


def _upsert_playbook(ctx: _SeedContext, spec: dict[str, Any], slug_to_id: dict[str, int]) -> str:
    needed = {step["agent_slug"] for step in spec["steps"]}
    if not needed <= set(slug_to_id):
        return "missing_agent"  # a roster agent it needs is gone — leave it alone
    # template_id is globally unique (ix_workflow_recipes_template_id) while the
    # Playbook's identity is per workspace — same rule as the roster slugs: the
    # plain id is the norm, a workspace suffix only when another workspace owns
    # the plain one. The ledger and the refresh contract key on the base id.
    row = (
        ctx.db.query(WorkflowTemplate)
        .filter(
            WorkflowTemplate.workspace_id == ctx.workspace_id,
            WorkflowTemplate.template_id.in_(_slug_candidates(spec["template_id"], ctx.workspace_id)),
        )
        .first()
    )
    columns = _playbook_columns(spec, ctx.workspace_id, ctx.operator, slug_to_id)
    fingerprint = partial(playbook_fingerprint, id_to_slug={v: k for k, v in slug_to_id.items()})
    if row is None:
        if spec["template_id"] in ctx.ledger["playbooks"]:
            return "deleted_by_user"
        taken_elsewhere = (
            ctx.db.query(WorkflowTemplate.id)
            .filter(WorkflowTemplate.template_id == spec["template_id"], WorkflowTemplate.workspace_id != ctx.workspace_id)
            .first()
            is not None
        )
        if taken_elsewhere:
            columns["template_id"] = _slug_candidates(spec["template_id"], ctx.workspace_id)[1]
        recipe = WorkflowTemplate(**columns)
        validate_playbook(recipe)
        ctx.db.add(recipe)
        ctx.db.flush()
        return "created"
    columns["template_id"] = row.template_id  # identity is the base id; the stored one stays
    return _refresh(row, columns, _PLAYBOOK_REFRESH_FIELDS, fingerprint, fingerprint(SimpleNamespace(**columns)))


def _upsert_welcome_post(ctx: _SeedContext, auto_agent_id: Optional[int]) -> str:
    row = (
        ctx.db.query(BlogPost)
        .filter(BlogPost.workspace_id == ctx.workspace_id, BlogPost.slug == WELCOME_SLUG)
        .first()
    )
    columns = _post_columns(ctx.workspace_id, auto_agent_id)
    if row is None:
        if WELCOME_SLUG in ctx.ledger["deliverables"]:
            return "deleted_by_user"
        ctx.db.add(BlogPost(**columns, published_at=datetime.now(timezone.utc)))
        ctx.db.flush()
        return "created"
    current = post_fingerprint(SimpleNamespace(**columns))
    return _refresh(row, columns, _POST_REFRESH_FIELDS, post_fingerprint, current)


# ── Entry point ──────────────────────────────────────────────────────────────

def seed_local_first_run(db: Session) -> dict[str, Any]:
    """Seed (or refresh) the local edition's first-run content. The caller owns
    the commit. Returns per-kind outcome counts; ``{"skipped": ...}`` outside
    the local edition — SaaS is never written to."""
    if config.AUTH_EDITION != "local":
        return {"skipped": "not-local"}
    raw_workspace_id = (config.DEFAULT_WORKSPACE_ID or "").strip()
    if not raw_workspace_id:
        return {"skipped": "no-default-workspace"}
    workspace_id = UUID(raw_workspace_id)

    result: dict[str, Any] = {"workspace_id": str(workspace_id)}
    result["workspace"] = _ensure_workspace(db, workspace_id)
    operator, result["operator_user"] = _ensure_operator_user(db)
    auto = seed_auto_agent(db, workspace_id)  # existing per-workspace seeder, idempotent
    result["auto_agent_id"] = auto.id

    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).one()
    ctx = _SeedContext(db=db, workspace_id=workspace_id, operator=operator, ledger=_read_ledger(workspace))

    result["agents"] = dict(Counter(_upsert_agent(ctx, spec) for spec in ROSTER))
    result["playbooks"] = dict(Counter([_upsert_playbook(ctx, PLAYBOOK, _roster_ids(ctx))]))
    result["deliverables"] = dict(Counter([_upsert_welcome_post(ctx, auto.id)]))
    result["ledger_updated"] = _record_ledger(workspace, ctx)
    db.flush()

    logger.info("PRD-233 S3 local first-run seed: %s", result)
    return result
