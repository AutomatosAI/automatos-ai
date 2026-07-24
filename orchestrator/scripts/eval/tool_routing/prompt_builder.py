"""
Prompt assembly for the tool-routing eval.

Four modes:

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

* `graph`           — graph-based ranking (PRD-139 US-007): delegate to the production
                      `GraphRouter` (US-004) which uses edges + affinities over the
                      embedding entry nodes. Prompt includes chain hints when chains have
                      depth > 1. Falls back to `filtered` when the graph is empty (no
                      edges), tagging the result as `graph (no-edges)`.

Public surface:

    builder = PromptBuilder(actions=registry.get_all())
    prompt, surfaced = builder.build("list my agents", mode="full")
    prompt, surfaced = builder.build("list my agents", mode="filtered", top_k=15)
    prompt, surfaced = builder.build("list my agents", mode="filtered_schema", top_k=15)
    prompt, surfaced, is_fallback = builder.build_graph("list my agents", top_k=15)
    tools = builder.build_tools(TOP_LEVEL_TOOLS, "list my agents",
                                mode="filtered_schema", top_k=15)

Each `build()` call returns `(markdown, surfaced_action_names)` where
`surfaced_action_names` is the in-set candidate list used by `score.py`
to compute the in-set hit rate. `build_graph()` adds a third element
`is_fallback` (True when graph had no edges and fell back to filtered).
`build_tools()` returns the tools list ready to pass to OpenRouter —
narrowed for `filtered_schema` and `graph`, unchanged for `full` and
`filtered`.
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

# Chain hint block injected above the filtered catalog when graph mode
# produces multi-action chains (depth > 1).
_CHAIN_HINT_HEADER = "\n## Likely Platform Action Chains\n\n"


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

        For graph mode, use `build_graph()` instead — it returns a third
        element indicating whether the graph fell back to filtered.
        """
        if mode == "full":
            return self._full(), [a.name for a in self.actions]
        if mode in _RANKED_MODES:
            # filtered and filtered_schema use the same prompt — they differ
            # only in whether the schema's action.enum is also narrowed.
            return self._filtered(query, top_k=top_k)
        if mode == "graph":
            prompt, surfaced, _is_fallback = self.build_graph(query, top_k=top_k)
            return prompt, surfaced
        raise ValueError(f"unknown mode: {mode!r}")

    def build_graph(
        self, query: str, top_k: int = 15
    ) -> Tuple[str, List[str], bool]:
        """Build prompt using GraphRouter for graph-mode evaluation.

        Returns (markdown, surfaced_action_names, is_fallback) where
        is_fallback is True when the graph had no edges and we fell back
        to the filtered (embedding-only) path.
        """
        return self._graph(query, top_k=top_k)

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
        if mode not in ("filtered_schema", "graph"):
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

    def _graph(self, query: str, top_k: int) -> Tuple[str, List[str], bool]:
        """Delegate ranking to the production GraphRouter (PRD-139 US-004).

        Algorithm:
        1. Call GraphRouter.rank_chains() to get scored chains.
        2. If chains are empty (no edges / cold start), fall back to the
           filtered (embedding-only) path and set is_fallback=True.
        3. Extract unique action names from chains (preserving rank order).
        4. Build the filtered prompt summary using those names.
        5. If any chains have depth > 1, prepend chain hint block.

        Returns (markdown, surfaced_action_names, is_fallback).
        """
        from modules.tools.discovery.graph_router import get_graph_router
        from modules.tools.discovery.action_registry import get_action_registry

        router = get_graph_router()
        try:
            chains = asyncio.run(
                router.rank_chains(
                    query,
                    # PRD-177 S5: offline eval harness reads the unscoped graph
                    # (no tenant context); production callers pass a real id.
                    workspace_id=None,
                    top_k=top_k,
                    exclude_admin=False,
                    exclude_promoted=False,
                )
            )
        except Exception:
            logger.warning(
                "_graph: GraphRouter.rank_chains() raised — falling back to filtered",
                exc_info=True,
            )
            prompt, surfaced = self._filtered(query, top_k=top_k)
            return prompt, surfaced, True

        if not chains:
            # Empty graph — fall back to filtered (embedding-only).
            prompt, surfaced = self._filtered(query, top_k=top_k)
            return prompt, surfaced, True

        # Check if all chains are single-action (no edges found in graph).
        has_multi_action = any(len(chain_actions) > 1 for _, _, chain_actions in chains)
        is_fallback = not has_multi_action

        # Extract unique action names from chains, preserving rank order.
        action_names: List[str] = list(dict.fromkeys(
            name
            for _, _, chain_actions in chains
            for name in chain_actions
        ))

        # Build filtered prompt summary using the graph-ranked names.
        registry = get_action_registry()
        catalog = registry.build_filtered_prompt_summary(
            action_names,
            exclude_admin=False,
            exclude_promoted=False,
        )

        # Build chain hints for multi-action chains.
        chain_hints = ""
        if has_multi_action:
            hint_lines = [_CHAIN_HINT_HEADER.rstrip("\n")]
            seen_pairs: set = set()
            for primary, _score, chain_actions in chains:
                if len(chain_actions) <= 1:
                    continue
                pair_key = tuple(chain_actions)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)
                hint_lines.append(
                    f"- {' -> '.join(f'`{a}`' for a in chain_actions)}"
                )
            hint_lines.append("")
            chain_hints = "\n".join(hint_lines) + "\n"

        return _PREAMBLE + chain_hints + catalog, action_names, is_fallback
