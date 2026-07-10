"""Policy plane — the Balanced workspace policy document (PRD-174 §5).

Encodes the act-vs-ask decision **once**, as the DB-configured workspace policy
the plane reads — not per-surface toggles (the three partial planes today
already contradict each other). Owner decision LOCKED 2026-07-02: **Balanced**.

Balanced defaults (all tunable per-workspace — config, not a deploy):

- **Auto (no ask):** reads; low-risk internal writes (draft docs, update own
  board-task status, internal memory writes, research/plan).
- **Ask (PRD-163 card / row):** destructive ops (deletes, board Run-Now);
  external side-effects (Composio sends/refunds/discounts, email/channel posts,
  Shopify writes); over-budget spend.
- **Templates / brand-kits:** Auto drafts; a human approves publish.
- **Board:** mutations stay chat/assignment-driven — no free create-task in v1.

Single canonical reader/writer for ``workspace.settings.policy_plane``, mirroring
the ``auto_autonomy`` / ``approval_policy`` services (settings live on the
Workspace JSON column; the caller owns the transaction). Fail-safe: an
unreadable / corrupt setting falls back to the Balanced defaults.

SQLAlchemy is imported lazily so this module loads stdlib-only for unit tests;
the *classifier* (:func:`classify_action`) is pure and needs no DB at all.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional
from uuid import UUID

logger = logging.getLogger(__name__)

# Posture names. Balanced is the locked default; the others exist so the same
# document can express "supervise everything" or "trust within the autonomy
# dial" without inventing new mechanisms.
BALANCED = "balanced"
STRICT = "strict"      # ask for every write + external + destructive
PERMISSIVE = "permissive"  # ask only for destructive + over-budget
VALID_POSTURES = frozenset({BALANCED, STRICT, PERMISSIVE})

# Risk classes an action can fall into (pure, tool-name + permission derived).
RISK_READ = "read"
RISK_INTERNAL_WRITE = "internal_write"
RISK_EXTERNAL = "external_side_effect"
RISK_DESTRUCTIVE = "destructive"
RISK_PUBLISH = "publish"  # templates / brand-kits publish step

# PRD-192 S1 (locked): the risk classes that FAIL CLOSED on a plane error under
# the enforce stages — and the exact set the `destructive` stage enforces.
# read / internal_write fail open (marked + counted); an unclassifiable call is
# treated destructive (closed). One frozenset so the stage boundary and the
# fail posture can never drift apart.
FAIL_CLOSED_RISK_CLASSES = frozenset({RISK_DESTRUCTIVE, RISK_EXTERNAL, RISK_PUBLISH})

# The Balanced routing table: risk class -> "auto" | "ask".
_BALANCED_ROUTING: Dict[str, str] = {
    RISK_READ: "auto",
    RISK_INTERNAL_WRITE: "auto",
    RISK_PUBLISH: "ask",
    RISK_EXTERNAL: "ask",
    RISK_DESTRUCTIVE: "ask",
}
_STRICT_ROUTING: Dict[str, str] = {
    RISK_READ: "auto",
    RISK_INTERNAL_WRITE: "ask",
    RISK_PUBLISH: "ask",
    RISK_EXTERNAL: "ask",
    RISK_DESTRUCTIVE: "ask",
}
_PERMISSIVE_ROUTING: Dict[str, str] = {
    RISK_READ: "auto",
    RISK_INTERNAL_WRITE: "auto",
    RISK_PUBLISH: "auto",
    RISK_EXTERNAL: "auto",
    RISK_DESTRUCTIVE: "ask",
}
_ROUTING_BY_POSTURE = {
    BALANCED: _BALANCED_ROUTING,
    STRICT: _STRICT_ROUTING,
    PERMISSIVE: _PERMISSIVE_ROUTING,
}

DEFAULTS: Dict[str, Any] = {
    "posture": BALANCED,
    # explicit, default-OFF: "agents inherit admin from the workspace owner".
    # F014 — with this off, admin_only actions require the *caller's* own admin
    # role, never a workspace-has-an-admin fallback.
    "agents_inherit_admin": False,
    # per-workspace overrides of the risk→route map, e.g.
    # {"external_side_effect": "auto"} to let a workspace auto-send.
    "route_overrides": {},
}


@dataclass(frozen=True)
class PolicyDocument:
    """The resolved, in-memory view of a workspace's policy posture."""

    posture: str
    agents_inherit_admin: bool
    route_overrides: Dict[str, str]

    def route_for(self, risk_class: str) -> str:
        """Return ``"auto"`` or ``"ask"`` for a risk class under this posture.

        A per-workspace ``route_overrides`` entry wins over the posture table.
        Unknown risk classes fail safe to ``"ask"`` (never silently auto).
        """
        override = self.route_overrides.get(risk_class)
        if override in ("auto", "ask"):
            return override
        table = _ROUTING_BY_POSTURE.get(self.posture, _BALANCED_ROUTING)
        return table.get(risk_class, "ask")

    def audit_snapshot(self) -> Dict[str, Any]:
        return {
            "posture": self.posture,
            "agents_inherit_admin": self.agents_inherit_admin,
            "route_overrides": dict(self.route_overrides),
        }


def _defaults_doc() -> PolicyDocument:
    return PolicyDocument(
        posture=DEFAULTS["posture"],
        agents_inherit_admin=DEFAULTS["agents_inherit_admin"],
        route_overrides={},
    )


def _enforcement_active() -> bool:
    """Guarded read of the enforce-stage flag — never raises (PRD-192 S1)."""
    try:
        from modules.policy.flag import enforcement_active

        return enforcement_active()
    except Exception:
        return False


def load_policy_document(db: Any, workspace_id: UUID | str) -> PolicyDocument:
    """Return the workspace's ``policy_plane`` settings as a :class:`PolicyDocument`.

    Fail-safe: missing workspace / corrupt setting ⇒ Balanced defaults.
    """
    if db is None or workspace_id is None:
        return _defaults_doc()
    try:
        from core.models.workspaces import Workspace

        ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
        if ws is None:
            return _defaults_doc()
        cfg = (ws.settings or {}).get("policy_plane") or {}
    except Exception:
        logger.warning(
            "[policy.document] read failed for workspace=%s — Balanced defaults",
            workspace_id, exc_info=True,
        )
        # PRD-192 S1: under an enforce stage a posture-read fault must not
        # silently pre-decide the routing — re-raise so the gate's single
        # except owns the fail posture (closed for high-risk classes). In
        # off/shadow the historical Balanced fallback stands.
        if _enforcement_active():
            raise
        return _defaults_doc()

    posture = cfg.get("posture")
    if not isinstance(posture, str) or posture not in VALID_POSTURES:
        posture = BALANCED

    inherit = cfg.get("agents_inherit_admin")
    inherit = bool(inherit) if isinstance(inherit, bool) else False

    overrides_raw = cfg.get("route_overrides")
    overrides: Dict[str, str] = {}
    if isinstance(overrides_raw, dict):
        for k, v in overrides_raw.items():
            if v in ("auto", "ask"):
                overrides[str(k)] = v

    return PolicyDocument(posture=posture, agents_inherit_admin=inherit, route_overrides=overrides)


def set_policy_document(
    db: Any,
    workspace_id: UUID | str,
    *,
    posture: Optional[str] = None,
    agents_inherit_admin: Optional[bool] = None,
    route_overrides: Optional[Dict[str, str]] = None,
) -> PolicyDocument:
    """Persist policy-plane fields to ``workspace.settings.policy_plane``.

    Only provided fields change. Caller owns the transaction (stages + flushes).
    """
    from core.models.workspaces import Workspace
    from sqlalchemy.orm.attributes import flag_modified

    if posture is not None and posture not in VALID_POSTURES:
        raise ValueError(
            f"invalid posture {posture!r}; expected one of {sorted(VALID_POSTURES)}"
        )

    ws = db.query(Workspace).filter(Workspace.id == workspace_id).first()
    if ws is None:
        raise ValueError(f"workspace {workspace_id} not found")

    settings = dict(ws.settings or {})
    current = dict(settings.get("policy_plane") or {})
    if posture is not None:
        current["posture"] = posture
    if agents_inherit_admin is not None:
        current["agents_inherit_admin"] = bool(agents_inherit_admin)
    if route_overrides is not None:
        current["route_overrides"] = {
            str(k): v for k, v in route_overrides.items() if v in ("auto", "ask")
        }
    settings["policy_plane"] = current
    ws.settings = settings
    flag_modified(ws, "settings")
    db.flush()
    return load_policy_document(db, workspace_id)


# ---------------------------------------------------------------------------
# Pure risk classifier — no DB. Maps a tool call to a risk class the routing
# table understands. Kept deliberately conservative: anything it can't
# confidently place as read/internal is treated as the higher-risk class.
# ---------------------------------------------------------------------------

# Composio actions carry an external side-effect by nature (they act on a
# third-party app). Sends/refunds/discounts/posts are the classic ask cases.
_EXTERNAL_TOOL_PREFIXES = ("composio_", "workspace_exec")


def classify_action(
    tool_name: str,
    *,
    permission_level: Optional[str] = None,
    is_composio: bool = False,
) -> str:
    """Classify a tool call into a risk class (pure).

    - Composio / external-exec tools ⇒ ``external_side_effect`` (they touch a
      third-party app), unless the registry marks them ``destructive``.
    - Platform/registry tools use ``permission_level``: ``destructive`` ⇒
      destructive; ``write`` ⇒ internal_write; ``read``/unknown ⇒ read.
    - The template/brand-kit *publish* step is flagged ``publish`` by name.
    """
    name = (tool_name or "").lower()

    if permission_level == "destructive":
        return RISK_DESTRUCTIVE

    if is_composio or name.startswith(_EXTERNAL_TOOL_PREFIXES):
        return RISK_EXTERNAL

    # Template / brand-kit publish is the human-approves-publish case (§5).
    if "publish" in name and ("template" in name or "brand" in name):
        return RISK_PUBLISH

    if permission_level == "write":
        return RISK_INTERNAL_WRITE
    if permission_level == "read":
        return RISK_READ

    # Unknown platform tool with no permission signal: treat a bare
    # ``platform_*`` read-shaped name as read, everything else as internal
    # write (fail toward asking, never toward silent external action).
    return RISK_READ if name.startswith("platform_") and _looks_readonly(name) else RISK_INTERNAL_WRITE


_READONLY_HINTS = ("list", "get", "search", "read", "grep", "summary", "stats", "query", "describe")


def _looks_readonly(name: str) -> bool:
    return any(h in name for h in _READONLY_HINTS)
