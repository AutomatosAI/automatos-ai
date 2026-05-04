"""
Prompt assembly for the tool-routing eval.

Three modes:

* `full`            — dump every ActionDefinition the registry knows about,
                      formatted to match production's `ActionRegistry.build_prompt_summary()`.
                      Tools list is also full (platform_execute.action.enum unset → free-form string).

* `filtered`        — prompt-only narrowing: delegate to the production
                      `ActionSemanticIndex` (PRD-138 US-003) to rank actions, then ask
                      the production `ActionRegistry.build_filtered_prompt_summary()` (US-002)
                      to render only those names. The dispatcher schema is UNCHANGED — its
                      action field stays free-form, so the LLM "could" pick anything; the
                      narrow prompt is steering, not enforcement. This is what shipped in
                      Phase 1 (US-001..US-005, MERGED).

* `filtered_schema` — prompt + schema narrowing: same prompt as `filtered`, AND the
                      `platform_execute` tool's `action.enum` is set to the same top-K
                      ranked names. This is the production behavior after Phase 1b
                      (US-008..US-010): both the prompt summary and the dispatcher
                      schema's enum narrow together.

Public surface:

    builder = PromptBuilder(actions=registry.get_all())
    prompt, surfaced = builder.build("list my agents", mode="full")
    prompt, surfaced = builder.build("list my agents", mode="filtered", top_k=15)
    prompt, surfaced = builder.build("list my agents", mode="filtered_schema", top_k=15)
    tools = builder.build_tools(TOP_LEVEL_TOOLS, "list my agents",
                                mode="filtered_schema", top_k=15)

Each `build()` call returns `(markdown, surfaced_action_names)` where
`surfaced_action_names` is the in-set candidate list used by `score.py`
to compute the in-set hit rate. `build_tools()` returns the tools list
ready to pass to OpenRouter — narrowed for `filtered_schema`, unchanged
for `full` and `filtered`.
"""

from __future__ import annotations

import asyncio
import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence, Tuple

logger = logging.getLogger(__name__)

# Mirror production's preamble/structure exactly so the only variable
# between modes is the action list itself.
_PREAMBLE = (
    "## Platform Actions\n\n"
    "You can execute these actions via `platform_execute`. Always specify the "
    "action name and include all required parameters. If an action fails, check "
    "the error and retry with corrected parameters — do not guess or fabricate "
    "results.\n\n"
)

_CATALOG_HEADER = (
    "\n## Available Platform Actions\n\n"
    "Use `platform_execute(action, params)` to call these. "
    "The `action` field must be the exact action name.\n\n"
)


# Modes that rank via ActionSemanticIndex (vs the full-dump baseline).
_RANKED_MODES = frozenset({"filtered", "filtered_schema"})


# ──────────────────────────────────────────────────────────────────
# Full-mode renderer (kept local so Appendix A's full-mode baseline
# stays bit-stable — the wave only rewires *filtered* mode).
# ──────────────────────────────────────────────────────────────────


def _format_action_line(action: Any) -> str:
    """Format a single action as one bullet, matching production exactly."""
    props = action.parameters.get("properties", {}) if action.parameters else {}
    required = action.parameters.get("required", []) if action.parameters else []
    param_hints: List[str] = []
    for pname in props.keys():
        marker = " (required)" if pname in required else ""
        param_hints.append(f"`{pname}`{marker}")
    param_str = f" — params: {', '.join(param_hints)}" if param_hints else ""
    return f"- `{action.name}`: {action.description}{param_str}"


def _render_catalog(actions: Iterable[Any]) -> str:
    """Render an action list as markdown, grouped by category, sorted within group."""
    by_category: Dict[str, List[Any]] = {}
    for a in actions:
        by_category.setdefault(a.category, []).append(a)

    lines = [_CATALOG_HEADER.rstrip("\n")]
    for category in sorted(by_category.keys()):
        lines.append("")
        lines.append(f"### {category.replace('_', ' ').title()}")
        for action in sorted(by_category[category], key=lambda a: a.name):
            lines.append(_format_action_line(action))
    lines.append("")
    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────────
# PromptBuilder
# ──────────────────────────────────────────────────────────────────


@dataclass
class PromptBuilder:
    actions: Sequence[Any]

    def build(self, query: str, mode: str, top_k: int = 15) -> Tuple[str, List[str]]:
        """
        Build the system-prompt action catalog for `query`.

        Returns (markdown, surfaced_action_names). The runner uses the second
        element to compute "in-set hit rate" — i.e. did the filter at least
        surface a correct action, even if the LLM picked the wrong one?
        """
        if mode == "full":
            return self._full(), [a.name for a in self.actions]
        if mode in _RANKED_MODES:
            # filtered and filtered_schema use the same prompt — they differ
            # only in whether the schema's action.enum is also narrowed.
            return self._filtered(query, top_k=top_k)
        raise ValueError(f"unknown mode: {mode!r}")

    def build_tools(
        self,
        top_level_tools: List[Dict[str, Any]],
        query: str,
        mode: str,
        top_k: int = 15,
        ranked_names: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """
        Return the OpenAI-compatible function-calling tools list, narrowed if needed.

        * `full`            — returns top_level_tools unchanged.
        * `filtered`        — returns top_level_tools unchanged (prompt narrows; schema does not).
        * `filtered_schema` — deep-copies top_level_tools and sets the
                              `platform_execute.parameters.properties.action.enum`
                              to the top-K ranked action names for `query`.

        If `ranked_names` is provided (e.g. surfaced list from a prior `build()`
        call) it's used directly — no re-ranking. Otherwise the ranker is invoked.

        Falls back to the unchanged tools list on any ranker error — same
        defensive pattern as production's `to_dispatcher_schema(allowed_names=[])`.
        """
        if mode != "filtered_schema":
            return top_level_tools

        if ranked_names is None:
            try:
                ranked_names = self._rank_action_names(query, top_k=top_k)
            except Exception:  # noqa: BLE001
                logger.warning(
                    "build_tools: ranker raised — falling back to full enum",
                    exc_info=True,
                )
                return top_level_tools

        if not ranked_names:
            logger.warning(
                "build_tools: ranker returned 0 names — falling back to full enum"
            )
            return top_level_tools

        narrowed = copy.deepcopy(top_level_tools)
        for tool in narrowed:
            fn = tool.get("function") or {}
            if fn.get("name") != "platform_execute":
                continue
            params = fn.get("parameters") or {}
            props = params.get("properties") or {}
            action_field = props.get("action")
            if not isinstance(action_field, dict):
                continue
            action_field["enum"] = list(ranked_names)
            break
        return narrowed

    # --------------------------------------------------------------

    def _full(self) -> str:
        return _PREAMBLE + _render_catalog(self.actions)

    def _filtered(self, query: str, top_k: int) -> Tuple[str, List[str]]:
        """
        Delegate ranking to the production ActionSemanticIndex (US-003) and
        rendering to ActionRegistry.build_filtered_prompt_summary (US-002).

        Calls the index with exclude_admin=False, exclude_promoted=False so
        the eval's tool surface stays identical to the 2026-05-03 baseline
        (prototype ranker over all 107 actions). The PRD's parity check is
        about ranker quality of the new index — not about reproducing the
        production caller's filter, which is a separate concern verified
        elsewhere (US-004 unit tests for PlatformActionsSection).
        """
        # Imported lazily so the bootstrap stubs in `_registry_bootstrap` are
        # already in place before we touch the production modules.
        from modules.tools.discovery.action_registry import get_action_registry

        top_names = self._rank_action_names(query, top_k=top_k)

        registry = get_action_registry()
        catalog = registry.build_filtered_prompt_summary(
            top_names,
            exclude_admin=False,
            exclude_promoted=False,
        )

        return _PREAMBLE + catalog, top_names

    def _rank_action_names(self, query: str, top_k: int) -> List[str]:
        """Run ActionSemanticIndex and return the ranked action names.

        Centralised here so both the prompt path (`_filtered`) and the schema
        path (`build_tools`) hit the same ranker with the same flags.
        """
        from modules.tools.discovery.action_semantic_index import (
            get_action_semantic_index,
        )

        index = get_action_semantic_index()
        ranked = asyncio.run(
            index.rank_actions(
                query=query,
                top_k=top_k,
                exclude_admin=False,
                exclude_promoted=False,
            )
        )
        return [name for name, _score in ranked]
