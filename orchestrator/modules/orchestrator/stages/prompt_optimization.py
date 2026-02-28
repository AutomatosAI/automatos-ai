"""
Prompt Optimization Stage (Stage 3b)
=====================================

PRD-59 / PRD-58 Integration: Before execution, check if PRD-58's
Prompt Registry has an optimized variant for the relevant system prompts.

If an A/B test is active, select the appropriate variant and inject it
into the agent's execution config. Track which variant was used for analytics.

Falls back gracefully if Prompt Registry is unavailable.
"""

import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PromptVariant:
    """A prompt variant from the Prompt Registry"""
    slug: str
    version: str
    content: str
    is_optimized: bool = False
    ab_group: Optional[str] = None  # "control" or "variant"


@dataclass
class PromptOptimizationResult:
    """Result of prompt optimization check"""
    subtask_variants: Dict[str, PromptVariant] = field(default_factory=dict)
    variants_applied: int = 0
    ab_tests_active: int = 0
    registry_available: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "variants_applied": self.variants_applied,
            "ab_tests_active": self.ab_tests_active,
            "registry_available": self.registry_available,
            "subtask_variants": {
                sid: {
                    "slug": v.slug,
                    "version": v.version,
                    "is_optimized": v.is_optimized,
                    "ab_group": v.ab_group,
                }
                for sid, v in self.subtask_variants.items()
            },
        }


class PromptOptimizationStage:
    """
    Stage 3b: Check Prompt Registry for optimized prompt variants.

    Integrates with PRD-58's PromptRegistry to:
    1. Look up system prompts by slug
    2. Check for active A/B tests
    3. Select the appropriate variant
    4. Inject into agent execution config
    """

    def __init__(self, db_session: Optional[Any] = None):
        self.db = db_session
        self._registry = None
        self._registry_checked = False

    def _get_registry(self):
        """Lazy-load Prompt Registry if available"""
        if self._registry_checked:
            return self._registry

        self._registry_checked = True
        try:
            from modules.prompts.registry import PromptRegistry
            self._registry = PromptRegistry(db_session=self.db)
            logger.info("✅ Prompt Registry available for Stage 3b")
        except ImportError:
            logger.info("ℹ️ Prompt Registry not available (PRD-58 not deployed)")
            self._registry = None
        except Exception as e:
            logger.warning(f"Failed to initialize Prompt Registry: {e}")
            self._registry = None

        return self._registry

    async def optimize_prompts(
        self,
        subtasks: List[Dict[str, Any]],
        agent_assignments: Dict[str, Any],
    ) -> PromptOptimizationResult:
        """
        Check for optimized prompt variants for each subtask's agent.

        Args:
            subtasks: List of subtask dicts
            agent_assignments: Map of subtask_id → agent info

        Returns:
            PromptOptimizationResult with any applied variants
        """
        registry = self._get_registry()
        if registry is None:
            return PromptOptimizationResult(registry_available=False)

        result = PromptOptimizationResult(registry_available=True)

        for i, subtask in enumerate(subtasks):
            subtask_id = subtask.get("subtask_id", f"subtask_{i}")
            agent_type = subtask.get("agent_type", "general")

            # Map agent type to prompt slug
            prompt_slug = self._get_prompt_slug(agent_type)
            if not prompt_slug:
                continue

            try:
                # Check for A/B variant
                variant = await self._get_variant(registry, prompt_slug)
                if variant:
                    result.subtask_variants[subtask_id] = variant
                    if variant.is_optimized:
                        result.variants_applied += 1
                    if variant.ab_group:
                        result.ab_tests_active += 1

                    # Inject into subtask config
                    subtask["optimized_system_prompt"] = variant.content
                    subtask["prompt_variant"] = {
                        "slug": variant.slug,
                        "version": variant.version,
                        "ab_group": variant.ab_group,
                    }

            except Exception as e:
                logger.debug(f"Prompt optimization failed for {subtask_id}: {e}")

        if result.variants_applied > 0:
            logger.info(
                f"✅ Stage 3b: Applied {result.variants_applied} optimized prompts, "
                f"{result.ab_tests_active} A/B tests active"
            )
        else:
            logger.info("ℹ️ Stage 3b: No optimized prompt variants available")

        return result

    async def _get_variant(
        self, registry: Any, prompt_slug: str
    ) -> Optional[PromptVariant]:
        """Get the best prompt variant from registry"""
        try:
            # Try A/B variant first
            if hasattr(registry, "get_ab_variant"):
                ab_result = await registry.get_ab_variant(prompt_slug)
                if ab_result:
                    return PromptVariant(
                        slug=prompt_slug,
                        version=ab_result.get("version", "unknown"),
                        content=ab_result.get("content", ""),
                        is_optimized=ab_result.get("is_optimized", False),
                        ab_group=ab_result.get("group", None),
                    )

            # Fall back to latest version
            if hasattr(registry, "get_prompt"):
                prompt = await registry.get_prompt(prompt_slug)
                if prompt:
                    return PromptVariant(
                        slug=prompt_slug,
                        version=prompt.get("version", "1.0"),
                        content=prompt.get("content", ""),
                        is_optimized=prompt.get("is_optimized", False),
                    )

        except Exception as e:
            logger.debug(f"Registry lookup failed for {prompt_slug}: {e}")

        return None

    @staticmethod
    def _get_prompt_slug(agent_type: str) -> Optional[str]:
        """Map agent type to prompt registry slug"""
        slug_map = {
            "orchestrator": "task-decomposer",
            "researcher": "research-agent",
            "developer": "code-agent",
            "writer": "writing-agent",
            "analyst": "analysis-agent",
            "data_analyst": "analysis-agent",
            "tester": "testing-agent",
            "designer": "design-agent",
            "general": None,  # No specific prompt for general agents
        }
        return slug_map.get(agent_type)
