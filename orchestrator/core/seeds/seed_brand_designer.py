"""
The Brand designer, seeded per workspace (PRD-255 US-011)
=========================================================

Every workspace gets a designer on its team: the Socials package's Brand Designer
(``seed_socials_package``), one persona source. The workspace's row is a copy of the
marketplace row, made the way an install makes one (``cloned_from_id`` names the
marketplace row), so installing the Socials package later reuses it
(``workspace_clone_of``) instead of cloning a second designer.

Seeded wherever Auto is: ``seed_auto_agent`` calls :func:`seed_brand_designer`, so a
new hosted workspace (``core/auth/hybrid.py``), the lazy Auto seed
(``api/workspaces.py``) and the local first-run seed (every local boot) all get one.

Insert-if-absent: a workspace that already has its designer (seeded, or the Socials
package's clone) keeps it as it is. The workspace's settings remember the seed
(``LEDGER_KEY``), so a designer the owner removed is not brought back.

The runtime: the designer reads images (the logo, the pages it draws), so it runs as
a Claude Code session (``runtime: cli``, ``provider: claude``) wherever the instance
runs sessions (``config.CLI_RUNTIME_ENABLED``, the local edition). Elsewhere (hosted,
or local without session mode) it is an API agent on the workspace's own model, with
the same instructions. Either way it is created with no host: its tickets wait for
one as every session agent's do.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Mapping, Optional
from uuid import UUID, uuid4

from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from config import config
from core.cli_runtime import (
    CONFIG_MODEL_KEY,
    CONFIG_PROVIDER_KEY,
    CONFIG_RUNTIME_KEY,
    PROVIDER_CLAUDE,
    RUNTIME_API,
    RUNTIME_CLI,
    validate_runtime_configuration,
)
from core.models.core import Agent
from core.models.workspaces import Workspace
# The Socials seed's own helpers: the marketplace row is made and its skills linked
# exactly as that seed does it, never a second way.
from core.seeds.seed_socials_package import BRAND_DESIGNER, MARKETPLACE, SOCIALS_AGENTS, _ensure_agent, _link_skills

logger = logging.getLogger(__name__)

SEEDED_BY = "seed_brand_designer"
WORKSPACE_OWNER = "workspace"
# The Claude Code model a session designer runs on: judging a rendered page is the work.
CLI_MODEL = "opus"
# Workspace.settings: set once the workspace has had its designer.
LEDGER_KEY = "brand_designer_seeded"

BRAND_DESIGNER_SPEC: Mapping[str, Any] = next(spec for spec in SOCIALS_AGENTS if spec["slug"] == BRAND_DESIGNER)
# The name Auto files a brand ask to (PRD-255 US-014).
BRAND_DESIGNER_NAME: str = BRAND_DESIGNER_SPEC["name"]


def designer_slug(workspace_id: UUID) -> str:
    """The seeded row's slug: ``agents.slug`` is unique across workspaces, as Auto's is."""
    return f"{BRAND_DESIGNER}-{workspace_id}"


def designer_configuration(cli_enabled: bool) -> Dict[str, Any]:
    """The designer's runtime: a Claude Code session where sessions run, else the API."""
    if cli_enabled:
        return {CONFIG_RUNTIME_KEY: RUNTIME_CLI, CONFIG_PROVIDER_KEY: PROVIDER_CLAUDE, CONFIG_MODEL_KEY: CLI_MODEL}
    return {CONFIG_RUNTIME_KEY: RUNTIME_API}


def find_brand_designer(db: Session, workspace_id: UUID) -> Optional[Agent]:
    """The workspace's Brand designer: the seeded row, or the Socials package's clone."""
    seeded = (
        db.query(Agent)
        .filter(Agent.workspace_id == workspace_id, Agent.slug == designer_slug(workspace_id))
        .first()
    )
    if seeded is not None:
        return seeded
    marketplace = db.query(Agent).filter(Agent.slug == BRAND_DESIGNER, Agent.owner_type == MARKETPLACE).first()
    if marketplace is None:
        return None
    return (
        db.query(Agent)
        .filter(
            Agent.cloned_from_id == marketplace.id,
            Agent.workspace_id == workspace_id,
            Agent.owner_type == WORKSPACE_OWNER,
        )
        .first()
    )


def seed_brand_designer(db: Session, workspace_id: UUID) -> Optional[Agent]:
    """Create the workspace's Brand designer when it has none; return its designer.

    None when the owner removed the one seeded earlier. Idempotent; the caller commits.
    """
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    existing = find_brand_designer(db, workspace_id)
    if existing is not None:
        _remember_seeded(workspace)
        return existing
    if _was_seeded(workspace):
        logger.info("Brand designer: workspace %s removed its designer; not seeded again", workspace_id)
        return None
    agent = _create_designer(db, workspace_id)
    _remember_seeded(workspace)
    return agent


def _create_designer(db: Session, workspace_id: UUID) -> Agent:
    configuration = designer_configuration(bool(config.CLI_RUNTIME_ENABLED))
    errors = validate_runtime_configuration(configuration, cli_enabled=bool(config.CLI_RUNTIME_ENABLED))
    if errors:
        raise ValueError(f"The Brand designer's runtime is invalid: {'; '.join(errors)}")
    marketplace, _created = _ensure_agent(db, BRAND_DESIGNER_SPEC)
    agent = Agent(**_columns(workspace_id, marketplace.id, configuration))
    db.add(agent)
    db.flush()
    _link_skills(db, agent, BRAND_DESIGNER_SPEC["skills"])
    logger.info(
        "Brand designer: seeded for workspace %s (agent.id=%s, runtime=%s)",
        workspace_id, agent.id, configuration[CONFIG_RUNTIME_KEY],
    )
    return agent


def _columns(workspace_id: UUID, marketplace_id: Any, configuration: Dict[str, Any]) -> Dict[str, Any]:
    """The workspace row: the marketplace spec's, owned by the workspace, with no host."""
    spec = BRAND_DESIGNER_SPEC
    return {
        "public_id": uuid4(),
        "name": spec["name"],
        "slug": designer_slug(workspace_id),
        "description": spec["description"],
        "agent_type": spec["agent_type"],
        "team": spec["team"],
        "job_title": spec["job_title"],
        "tags": list(spec["tags"]),
        "custom_persona_prompt": spec["custom_persona_prompt"],
        "use_custom_persona": True,
        # No model_config: an API designer runs on the workspace's configured LLM.
        "model_config": None,
        "configuration": configuration,
        "status": "active",
        "is_system_agent": False,
        "owner_type": WORKSPACE_OWNER,
        "owner_id": str(workspace_id),
        "workspace_id": workspace_id,
        "cloned_from_id": marketplace_id,
        "is_approved": True,
        "is_featured": False,
        "created_by": SEEDED_BY,
    }


def _settings_of(workspace: Optional[Workspace]) -> Dict[str, Any]:
    settings = getattr(workspace, "settings", None)
    return dict(settings) if isinstance(settings, Mapping) else {}


def _was_seeded(workspace: Optional[Workspace]) -> bool:
    return _settings_of(workspace).get(LEDGER_KEY) is True


def _remember_seeded(workspace: Optional[Workspace]) -> None:
    """Record the seed on the workspace (a new dict, written only when it changes)."""
    if workspace is None or _was_seeded(workspace):
        return
    workspace.settings = {**_settings_of(workspace), LEDGER_KEY: True}
    if isinstance(workspace, Workspace):  # a mapped row: make sure the JSON column is written
        flag_modified(workspace, "settings")
