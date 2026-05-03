"""
Prompt assembly for the tool-routing eval.

Two modes:

* `full`     — dump every ActionDefinition the registry knows about,
               formatted to match production's `ActionRegistry.build_prompt_summary()`.
               This is what the LLM sees today (~107 actions, ~2K tokens).

* `filtered` — delegate to the production `ActionSemanticIndex` (PRD-138 US-003)
               to rank actions by cosine similarity for the query, then ask the
               production `ActionRegistry.build_filtered_prompt_summary()` (US-002)
               to render only those names. Eliminates the prototype duplicate-ranking
               implementation: the eval now measures the real index.

Public surface:

    builder = PromptBuilder(actions=registry.get_all())
    full_prompt     = builder.build("list my agents", mode="full")
    filtered_prompt = builder.build("list my agents", mode="filtered", top_k=15)

Each `build()` call returns `(markdown, surfaced_action_names)` where
`surfaced_action_names` is the in-set candidate list used by `score.py`
to compute the in-set hit rate.
"""

from __future__ import annotations

import asyncio
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
        if mode == "filtered":
            return self._filtered(query, top_k=top_k)
        raise ValueError(f"unknown mode: {mode!r}")

    # --------------------------------------------------------------

    def _full(self) -> str:
        return _PREAMBLE + _render_catalog(self.actions)

    def _filtered(self, query: str, top_k: int) -> Tuple[str, List[str]]:
        """
        Delegate ranking to the production ActionSemanticIndex (US-003) and
        rendering to ActionRegistry.build_filtered_prompt_summary (US-002).
        """
        # Imported lazily so the bootstrap stubs in `_registry_bootstrap` are
        # already in place before we touch the production modules.
        from modules.tools.discovery.action_registry import get_action_registry
        from modules.tools.discovery.action_semantic_index import (
            get_action_semantic_index,
        )

        index = get_action_semantic_index()
        ranked = asyncio.run(
            index.rank_actions(
                query=query,
                top_k=top_k,
                exclude_admin=True,
                exclude_promoted=True,
            )
        )
        top_names = [name for name, _score in ranked]

        registry = get_action_registry()
        catalog = registry.build_filtered_prompt_summary(
            top_names,
            exclude_admin=True,
            exclude_promoted=True,
        )

        return _PREAMBLE + catalog, top_names
