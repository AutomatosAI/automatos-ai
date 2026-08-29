"""
Onboarding Section — PRD-222 W1S2 / US-009: the stage-aware conversational spine.
================================================================================

Auto runs the whole onboarding journey in chat, resumable at any point. This
section injects the guidance for *exactly one* stage — the workspace's current
``onboarding.stage`` — so Auto always knows where the user is and what happens
next.

Trigger (replaces the old ``agent_count == 0`` heuristic):
1. The workspace's ``onboarding.stage`` is NOT terminal (not ``completed`` /
   ``skipped``) — the server-side state machine (US-001) is the single source of
   truth, so the flow survives reloads and resumes from the recorded stage; OR
2. the user explicitly re-triggers with a phrase like "set up my workspace".

Otherwise the section renders ``""``.

The spine (each stage below maps to one guidance block):
    questions → teach → proposal → building → boom → powerup → (completed)

Hard rules baked into every rendered variant:
- Auto records each advance via the ``platform_update_onboarding`` tool (US-003)
  — the section NEVER advances state itself, and Auto must never assume a stage
  moved without that tool call succeeding.
- Auto adapts its register to ``segment.comfort``.
- Auto consults the capability report (US-007) and degrades honestly — e.g. it
  does not offer a site scan when Firecrawl is not configured.
- Onboarding must NEVER set ``skip_verification`` or ``auto_approve`` (the trust
  rule locked by US-010) — every build is verified and user-approved.
"""

import logging
from typing import Any, Optional

from modules.context.sections.base import BaseSection, SectionContext
from services import onboarding_state

logger = logging.getLogger(__name__)

# Trigger phrases that re-activate onboarding for an already-onboarded workspace.
_TRIGGER_PHRASES = frozenset({
    "set up my workspace",
    "help me get started",
    "mission zero",
    "reconfigure my workspace",
    "setup my workspace",
})

# Stages whose guidance is "ask the three questions". ``not_started`` also lands
# here (the user's first message) with a greeting prefix.
_QUESTION_STAGES = frozenset({onboarding_state.INITIAL_STAGE, "questions"})


# --------------------------------------------------------------------------- #
# Content — the COMMON rules (every active stage) + one per-stage block.
# The section renders COMMON + exactly ONE stage block, never all of them, so
# the largest rendered variant is COMMON + the biggest single block.
# --------------------------------------------------------------------------- #

_HEADER = "## Onboarding — you are guiding this workspace through setup ({stage}{comfort})\n"

_COMMON_RULES = """\
Guide the user one step at a time, conversationally — never a form, never a wall of text.

Rules at every stage:
- Record progress with the `platform_update_onboarding` tool: pass `advance_to` \
when the user finishes a step, and `segment` ({business, goal, comfort}) as you \
learn it. A stage only advances when that tool call succeeds — never assume it did.
- Match the user's AI comfort: plain-language and benefits-first for newcomers; \
precise and technical, with "or do it yourself" shortcuts, for technical users.
- NEVER create a mission or run a tool with `skip_verification` or `auto_approve` \
set — every build is verified and the user approves it before anything runs.
- Size the build: 3 or fewer agents AND 2 or fewer Playbooks → make the changes \
with direct tool calls right after the user's explicit yes; anything larger → \
create a normal mission (it defaults to awaiting_approval; the user approves it \
on the mission surface).
"""

# PRD-230 US-002 — the capability doctrine ("Auto knows its own shop"). These are
# REFLEXES for every active stage, kept tight: the heavy per-vertical depth is
# pulled on match via the package manifest (D8), never inlined here (Q7 budget).
_CAPABILITY_DOCTRINE = """\
Know your shop (reflexes):
- Connect apps via Composio through the chat **connect card** (Shopify first-class). \
No connect tool? Route to the card — never apologise or improvise.
- Shopify is two-step, told honestly: Composio connect = store data now; the Automatos \
Shopify app then adds a **Site** under Settings → Widget SDK → sync → Knowledge Graph + \
widgets. Tiered, never oversold.
- A URL → call `platform_scan_business_site` now (Firecrawl is prod; degrade honestly if not).
- Staff marketplace-first: search prebuilt agents, tools and Playbooks before building custom.
- Say it straight — "no CSVs — we sync directly, and our Shopify package includes \
widgets and agents." And early: "you're on Basic while we set up — we'll pick your \
plan together shortly."
- Stages are EXACTLY `not_started`, `questions`, `teach`, `proposal`, `building`, \
`boom`, `powerup`, `completed`, `skipped` — pass one to `platform_update_onboarding`; \
never invent one.
"""

_FIRST_MESSAGE_PREFIX = (
    "This is the user's first message. Greet them warmly as Auto in one line, "
    "then begin.\n\n"
)

_STAGE_QUESTIONS = """\
### Now: the three questions
Ask these one at a time, in your own words, waiting for each answer:
1. What's your business? (what you do, who you serve)
2. What's the first thing you'd want handled for you?
3. How comfortable are you with AI — brand new, or very technical?
Save each answer with `platform_update_onboarding` (segment.business / .goal / \
.comfort). When you have all three, advance_to `teach`.
"""

_STAGE_TEACH = """\
### Now: teach Auto their business
Offer three ways — their choice:
- Scan their website: call `platform_scan_business_site` with their domain.{firecrawl_note}
- Upload documents.
- Just tell you in chat.
When the reading finishes, play back what you learned in plain words and ask them \
to correct you — corrections matter. Then advance_to `proposal`.
"""

_NO_SCAN_NOTE = (
    " (Site scanning is NOT available in this deployment — do not offer the scan; "
    "steer to document upload or just talking it through.)"
)

_STAGE_PROPOSAL = """\
### Now: propose the setup — this is the approval gate
Start by matching a package: call `platform_search_packages` with their segment \
(business, goal, any store URL). If one matches, offer exactly ONE BY NAME with its \
contents — e.g. "Shopify Management: four agents (Operations, Support, Inventory, \
Business Analyst), a weekly-numbers report, and your store connect — want it?" If \
they defer the pick to you, choose sensibly (a store OWNER → Management, a builder \
→ Development). If NOTHING matches, don't force a package — custom-design their team, \
marketplace-first for each agent, tool and Playbook. Either way present ONE proposal \
sized to their business (a barber gets Auto + 1–2 helpers and ~2 Playbooks; a larger \
company more), what each piece does for THEM, the 1–2 apps to connect, and the cost \
("this build is covered by your trial credit"). Nothing is built before they say yes \
— let them edit conversationally. On an explicit yes, advance_to `building` and start.
Plan: {plan_recommendation}
"""

_STAGE_BUILDING = """\
### Now: build it — narrate every step
If they accepted a package, install it with `platform_install_package` (its slug) \
and narrate the manifest — the agents, skills, tools and Playbooks now registered \
to THEIR workspace, theirs to edit. Otherwise create the pieces directly. Then \
request the connections the setup needs through the chat connect card (never \
auto-connect); for Shopify, the two-step honestly — connect now for store data, \
then the Automatos app adds a Site under Settings → Widget SDK → sync. Narrate as \
you go ("Created your Marketing helper — it's on your Agents page"). When the build \
is complete and verified, advance_to `boom`.
"""

_STAGE_BOOM = """\
### Now: the payoff moment
Invite the user to ask you something about THEIR business, and answer it grounded \
in what you just learned — this is the value moment, still on their trial credit. \
Offer to put the team to work now — run their first Playbook or report; the setup \
checklist card carries the remaining steps. Once they've seen it, advance_to `powerup`.
"""

_STAGE_POWERUP = """\
### Now: keep Auto running — connect a key
Frame this as continuation, not a paywall: "Auto just read your business and built \
your team — keep him running." {trial_line}
Recommend ONE option first: **OpenRouter — one key, pay-as-you-go, access to 400+ \
models.** Offer the masked in-chat key entry. List other providers \
(OpenAI, Anthropic, …) collapsed beneath, for users who already have one.
A saved key is validated live and unlocks the full model catalogue. Declining is \
fine — the remaining trial credit keeps working.
Then present the run-and-learn checklist (connect a second app · invite a teammate \
· run your first mission · take the 10-minute course).
To finish, write the onboarding summary — what you built, why, and what happens \
next — with `platform_submit_report` (report_type `onboarding`, plus a title and \
content) so it lands as a Deliverable on their workspace; then advance_to `completed`.
"""

_RETRIGGER_NOTE = (
    "> The user asked to set up / reconfigure a workspace that has already "
    "onboarded. Offer to adjust the existing setup or start fresh — their call — "
    "then proceed from the three questions.\n"
)


def _trial_line(onboarding: dict[str, Any]) -> str:
    """A concrete trial-balance sentence for the power-up copy, when known."""
    trial = onboarding.get("trial") or {}
    granted = trial.get("granted_usd")
    if granted is None:
        return "Mention their remaining trial credit."
    spent = trial.get("spent_usd") or 0
    remaining = max(0.0, float(granted) - float(spent))
    return f"They have ${remaining:.2f} of ${float(granted):.2f} trial credit left."


class OnboardingSection(BaseSection):
    """PRD-222 US-009: stage-aware Mission Zero v2 prompt injection.

    Priority 2 (high — after identity, before skills/tools). Emits the guidance
    for the workspace's current onboarding stage, or ``""`` once the workspace
    has completed / skipped (unless the user manually re-triggers).
    """

    name: str = "onboarding"
    priority: int = 2
    # Budget: the section renders COMMON rules + one stage block. Largest measured
    # variant (powerup, with the dynamic trial line) is well under this cap — see
    # tests/test_prd222_onboarding_section.py::test_largest_variant_within_budget.
    max_tokens: Optional[int] = 800

    async def render(self, ctx: SectionContext) -> str:
        try:
            return await self._build(ctx)
        except Exception:
            logger.exception("OnboardingSection.render failed")
            return ""

    async def _build(self, ctx: SectionContext) -> str:
        workspace = self._load_workspace(ctx)
        stage = (
            onboarding_state.current_stage(workspace)
            if workspace is not None
            else None
        )
        is_active = stage is not None and stage not in onboarding_state.TERMINAL_STAGES
        is_manual = self._check_trigger_phrases(ctx)

        if not is_active and not is_manual:
            return ""

        onboarding = (
            onboarding_state.get_onboarding(workspace) if workspace is not None else {}
        )
        comfort = (onboarding.get("segment") or {}).get("comfort")

        # Re-trigger on a terminal (or unloadable) workspace restarts from the
        # questions; a re-trigger mid-flow just resumes the current stage.
        manual_note = ""
        render_stage = stage
        if is_manual and not is_active:
            render_stage = onboarding_state.INITIAL_STAGE
            manual_note = _RETRIGGER_NOTE

        return self._compose(ctx, render_stage, comfort, onboarding, manual_note)

    def _compose(
        self,
        ctx: SectionContext,
        stage: str,
        comfort: Optional[str],
        onboarding: dict[str, Any],
        manual_note: str,
    ) -> str:
        comfort_str = f" · comfort: {comfort}" if comfort else ""
        parts = [_HEADER.format(stage=stage, comfort=comfort_str)]
        if manual_note:
            parts.append(manual_note)
        parts.append(_COMMON_RULES)
        parts.append(_CAPABILITY_DOCTRINE)
        parts.append(self._stage_block(ctx, stage, onboarding))
        return "\n".join(p.strip() for p in parts if p and p.strip()) + "\n"

    def _stage_block(
        self, ctx: SectionContext, stage: str, onboarding: dict[str, Any]
    ) -> str:
        if stage in _QUESTION_STAGES:
            prefix = (
                _FIRST_MESSAGE_PREFIX
                if stage == onboarding_state.INITIAL_STAGE
                else ""
            )
            return prefix + _STAGE_QUESTIONS
        if stage == "teach":
            return _STAGE_TEACH.format(firecrawl_note=self._firecrawl_note(ctx))
        if stage == "proposal":
            return _STAGE_PROPOSAL.format(
                plan_recommendation=self._plan_recommendation(onboarding)
            )
        if stage == "building":
            return _STAGE_BUILDING
        if stage == "boom":
            return _STAGE_BOOM
        if stage == "powerup":
            return _STAGE_POWERUP.format(trial_line=_trial_line(onboarding))
        return ""  # defensive — terminal stages never reach here

    def _plan_recommendation(self, onboarding: dict[str, Any]) -> str:
        """The plan-recommendation line for the proposal stage (US-025).

        Derived from the stored segment via a pure helper; display prices come
        from PLAN_TIERS. Fail-safe: any error degrades to a plain instruction so
        a helper fault never blanks the proposal guidance.
        """
        try:
            from services.plan_tiers import plan_proposal_copy

            # Pass ONLY the segment — recommend_plan reads segment['team_size']
            # itself, so this display and the handler's plan_recommended funnel
            # stamp (which also passes only the segment) use identical inputs.
            segment = onboarding.get("segment") or {}
            return plan_proposal_copy(segment)
        except Exception:  # noqa: BLE001 — guidance must never blank on a helper error
            logger.warning("OnboardingSection._plan_recommendation failed", exc_info=True)
            return (
                "Recommend a plan (Basic $19 · Pro $49 · Business $99/mo, early access; "
                "Enterprise coming soon) and set the accepted tier via "
                "`platform_update_onboarding` (plan)."
            )

    def _firecrawl_note(self, ctx: SectionContext) -> str:
        """Honest-degrade: suppress the scan offer when Firecrawl is unconfigured.

        Reads the US-007 capability report. On any failure it defaults to
        offering the scan — ``platform_scan_business_site`` itself degrades
        honestly (US-008), so a report error never blocks the offer.
        """
        try:
            from services.capability_report import onboarding_capabilities

            caps = onboarding_capabilities(
                ctx.db_session, workspace_id=ctx.workspace_id
            )
            if not caps.get("firecrawl_configured", True):
                return _NO_SCAN_NOTE
        except Exception:
            logger.debug("OnboardingSection: capability check failed", exc_info=True)
        return ""

    def _load_workspace(self, ctx: SectionContext) -> Any:
        """Load the Workspace record (defensive; returns None on any miss)."""
        if not ctx.db_session or not ctx.workspace_id:
            return None
        try:
            from core.models.workspaces import Workspace

            return (
                ctx.db_session.query(Workspace)
                .filter(Workspace.id == ctx.workspace_id)
                .first()
            )
        except Exception as exc:
            logger.debug("OnboardingSection: workspace load failed: %s", exc)
            return None

    def _check_trigger_phrases(self, ctx: SectionContext) -> bool:
        """True if the last user message contains a manual re-trigger phrase."""
        messages = ctx.messages
        if not messages:
            return False
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = (msg.get("content") or "").lower().strip()
                return any(phrase in content for phrase in _TRIGGER_PHRASES)
        return False
