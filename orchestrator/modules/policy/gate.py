"""Policy plane — the one typed chokepoint (PRD-174 Step 2, F085/F060).

`PolicyGate.check()` is the single enforcement point every tool call on every
surface passes through when the plane is on. It folds together, for *all* tools
(not just `platform_*`), the guardrails that today are scattered across router
deps, action-registry flags and the mission approval engine — and that
Composio / workspace / registry tools bypass entirely (`unified_executor`
routes them around the gate stack).

What the gate evaluates, in order (deny > ask > allow — first blocking wins):

1. **Super-admin gate** (fail-closed) — `super_admin_only` actions: only a
   literal ``system_role == 'super_admin'`` caller passes. (F043 helper.)
2. **Admin gate** — `admin_only` actions require the *caller's own* admin role
   (F014); the "agents inherit admin from the workspace owner" fallback is an
   explicit, default-OFF workspace policy, not an implicit workspace-has-an-
   admin flip.
3. **Budget admission** (F086 + F059) — model-aware pre-call spend check against
   the workspace budget; over budget ⇒ ask/deny with errors-as-data.
4. **Act-vs-ask routing** (Balanced, §5) — destructive / external side-effect /
   publish ⇒ ask (PRD-163 card/row); read / low-risk-internal ⇒ auto. Skipped
   when the workspace is dialled to full autonomy (the auto dial) — that path
   still can't satisfy the super-admin gate or the destructive backstop, both
   of which live in `platform_executor` downstream.

The gate does NOT re-implement the PRD-140 hierarchy check or the per-(workspace,
agent) rate limiter — those stay in `platform_executor` for platform actions and
remain authoritative. The gate is the *universal* layer on top, so external
tools stop being ungoverned.

Returns a :class:`~modules.policy.types.Verdict`. Callers execute only on
allow; deny/ask stop the call and surface `verdict.error` (errors-as-data) as
tool content the model reads.

SQLAlchemy / registry imports are lazy so this loads stdlib-only for unit tests;
`check()` needs a live `db` only to price budgets and read the workspace policy.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from modules.policy import budget as _budget
from modules.policy import policy_document as _policy_doc
from modules.policy import pricing as _pricing
from modules.policy import roles as _roles
from modules.policy.types import Decision, PolicyError, Verdict

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolCall:
    """The subject of a policy decision — one tool call, tenant-scoped.

    Deliberately small and typed so the gate has no dependency on the caller's
    request shape.
    """

    tool_name: str
    parameters: Dict[str, Any]
    workspace_id: Any
    agent_id: Optional[int] = None
    caller_context: Optional[Dict[str, Any]] = None
    # Optional model-aware budget inputs for the pending call (F086/F059). When
    # a caller knows the model + token estimate it passes them; otherwise the
    # budget check uses spend-to-date only (still catches an already-over budget).
    model_id: Optional[str] = None
    est_input_tokens: int = 0
    est_output_tokens: int = 0
    # PRD-192 S1: the EXECUTOR's knowledge of whether this call is a Composio
    # action (it routes per-action SDK names like GMAIL_SEND_EMAIL via its
    # `composio_actions` dict / registry metadata — names the gate's prefix
    # check cannot recognise). ``None`` ⇒ the gate falls back to the prefix.
    is_composio: Optional[bool] = None


class PolicyGate:
    """Stateless evaluator. Construct per-request with the live `db` session."""

    def __init__(self, db: Any) -> None:
        self.db = db

    # -- public API --------------------------------------------------------

    def check(self, call: ToolCall) -> Verdict:
        """Evaluate one tool call. Returns the merged verdict (deny > ask > allow)."""
        doc = _policy_doc.load_policy_document(self.db, call.workspace_id)
        action_def = self._lookup_action(call.tool_name)
        # PRD-192 S1: the caller's hint wins (the executor knows its per-action
        # Composio routing); the prefix check is only the fallback — otherwise
        # SDK per-action sends classify internal_write and auto-run under
        # Balanced even with the plane ON.
        is_composio = (
            call.is_composio
            if call.is_composio is not None
            else self._is_composio(call.tool_name)
        )

        # 1) super-admin gate (fail-closed) — highest precedence.
        if action_def is not None and getattr(action_def, "super_admin_only", False):
            role = (call.caller_context or {}).get("system_role")
            if not _roles.is_super_admin(role):
                return Verdict.deny(
                    PolicyError(
                        code="super_admin_required",
                        message_for_model=(
                            f"Action '{call.tool_name}' is restricted to the platform "
                            "super admin (observability tier)."
                        ),
                        remediation="This action cannot be taken by an agent or workspace user.",
                        retryable=False,
                    )
                )

        # 2) admin gate — caller's own role (F014). The workspace-owner fallback
        #    is gated behind the explicit, default-OFF `agents_inherit_admin`.
        if action_def is not None and getattr(action_def, "admin_only", False):
            admin_verdict = self._admin_gate(call, doc)
            if admin_verdict.decision is Decision.DENY:
                return admin_verdict

        # 3) budget admission (model-aware). May deny (hard ceiling breach).
        budget_verdict = self._budget_gate(call)
        if budget_verdict.decision is not Decision.ALLOW:
            return budget_verdict

        # 4) act-vs-ask routing under the workspace posture (Balanced by default).
        route_verdict = self._route_gate(call, doc, action_def, is_composio)
        return route_verdict

    # -- gate stages -------------------------------------------------------

    def _admin_gate(self, call: ToolCall, doc: _policy_doc.PolicyDocument) -> Verdict:
        ctx = call.caller_context
        # Full autonomy: Auto runs as admin (the existing dial). Kept here so the
        # gate matches platform_executor's admin path when the flag is on.
        if self._full_autonomy(call.workspace_id):
            return Verdict.allow("full-autonomy: agent runs as admin")

        if ctx is not None:
            role = ctx.get("system_role")
            ws_role = ctx.get("workspace_role")
            if _roles.is_admin(role) or ws_role in ("owner", "admin"):
                return Verdict.allow("caller has admin/owner role")
            # Explicit non-admin caller — deny (F014: no workspace-has-an-admin flip).
            return Verdict.deny(
                PolicyError(
                    code="admin_required",
                    message_for_model=(
                        f"Action '{call.tool_name}' requires workspace admin or owner role."
                    ),
                    remediation="Ask a workspace admin to perform this, or escalate.",
                    retryable=False,
                )
            )

        # No caller identity (heartbeat / agent factory). Only the explicit,
        # default-OFF workspace policy lets an agent inherit admin here (F014).
        if doc.agents_inherit_admin and self._workspace_has_admin_owner(call.workspace_id):
            return Verdict.allow("agents_inherit_admin policy on + workspace has admin owner")
        return Verdict.deny(
            PolicyError(
                code="admin_required",
                message_for_model=(
                    f"Action '{call.tool_name}' requires admin. No admin caller identity "
                    "and this workspace does not grant agents inherited admin."
                ),
                remediation="Enable the 'agents inherit admin' workspace policy, or run as an admin.",
                retryable=False,
            )
        )

    def _budget_gate(self, call: ToolCall) -> Verdict:
        projected_cost = 0.0
        if call.model_id and (call.est_input_tokens or call.est_output_tokens):
            priced = _pricing.estimate_cost_usd(
                self.db, call.model_id, call.est_input_tokens, call.est_output_tokens
            )
            projected_cost = priced or 0.0
        decision = _budget.check_budget(
            self.db,
            call.workspace_id,
            projected_cost_usd=projected_cost,
            projected_tokens=call.est_input_tokens + call.est_output_tokens,
        )
        if decision.allowed:
            return Verdict.allow(decision.reason)
        return Verdict.deny(
            PolicyError(
                code="budget_exceeded",
                message_for_model=(
                    f"This call would exceed the workspace budget: {decision.reason}."
                ),
                remediation=(
                    "Raise the workspace budget ceiling, wait for the window to roll "
                    "over, or ask a human to approve the overage."
                ),
                retryable=False,
            ),
            reason=decision.reason,
        )

    def _route_gate(
        self,
        call: ToolCall,
        doc: _policy_doc.PolicyDocument,
        action_def: Any,
        is_composio: bool,
    ) -> Verdict:
        # Full autonomy short-circuits the ask routing (auto dial). The
        # destructive backstop + super-admin gate are unaffected (they ran
        # above / run downstream in platform_executor).
        if self._full_autonomy(call.workspace_id):
            return Verdict.allow("full-autonomy: act without asking")

        permission_level = getattr(action_def, "permission_level", None)
        risk = _policy_doc.classify_action(
            call.tool_name, permission_level=permission_level, is_composio=is_composio
        )
        route = doc.route_for(risk)
        if route == "auto":
            return Verdict.allow(f"posture={doc.posture} routes {risk} to auto")
        return Verdict.ask(
            PolicyError(
                code="approval_required",
                message_for_model=(
                    f"Action '{call.tool_name}' ({risk}) requires human approval under "
                    f"the workspace's {doc.posture} policy. It has NOT been executed."
                ),
                remediation=(
                    "A human must approve this in the approval card/queue before it runs."
                ),
                retryable=True,
            ),
            reason=f"posture={doc.posture} routes {risk} to ask",
        )

    # -- lazy lookups (DB / registry) --------------------------------------

    def _lookup_action(self, tool_name: str) -> Any:
        """Resolve the ActionDefinition for a platform action, or None.

        Handles both the ``platform_execute`` meta-tool (action nested in
        params) — not seen here, callers pass the resolved action name — and
        direct ``platform_*`` names. Non-platform tools return None.
        """
        try:
            from modules.tools.discovery import get_action_registry

            return get_action_registry().get(tool_name)
        except Exception:
            return None

    @staticmethod
    def _is_composio(tool_name: str) -> bool:
        """Prefix-based FALLBACK only (PRD-192 S1): callers that know better
        (the executor's `composio_actions` / registry metadata) pass the
        ``ToolCall.is_composio`` hint, which takes precedence in ``check()``."""
        name = (tool_name or "")
        return name.startswith("composio_") or name == "composio_execute"

    def _full_autonomy(self, workspace_id: Any) -> bool:
        try:
            from core.services.auto_autonomy import is_full_autonomy

            return bool(is_full_autonomy(self.db, workspace_id))
        except Exception:
            logger.warning(
                "[policy.gate] autonomy read failed for workspace=%s — supervised",
                workspace_id, exc_info=True,
            )
            return False

    def _workspace_has_admin_owner(self, workspace_id: Any) -> bool:
        try:
            from core.workspaces.models import WorkspaceMember

            member = (
                self.db.query(WorkspaceMember)
                .filter(
                    WorkspaceMember.workspace_id == workspace_id,
                    WorkspaceMember.role.in_(("owner", "admin")),
                    WorkspaceMember.is_active.is_(True),
                )
                .first()
            )
            return member is not None
        except Exception:
            logger.warning(
                "[policy.gate] workspace-admin-owner read failed for %s",
                workspace_id, exc_info=True,
            )
            return False
