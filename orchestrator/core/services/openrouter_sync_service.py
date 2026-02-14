"""
OpenRouterSyncService - Sync OpenRouter model catalog to local cache.

Fetches ALL models from https://openrouter.ai/api/v1/models and upserts
them into the openrouter_models_cache table. Pattern mirrors
MetadataSyncService (Composio) but is completely independent.

Usage:
    service = OpenRouterSyncService(db_session)
    result = service.run_full_sync()
"""

import logging
import httpx
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session
from sqlalchemy import func

from core.models.openrouter_cache import OpenRouterModelCache, OpenRouterSyncJob

logger = logging.getLogger(__name__)

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class OpenRouterSyncService:
    """Sync OpenRouter model catalog to local DB cache."""

    def __init__(self, db_session: Session):
        self.db = db_session
        self._sync_job: Optional[OpenRouterSyncJob] = None

    # =========================================================================
    # Main sync
    # =========================================================================

    def run_full_sync(self) -> Dict[str, Any]:
        """Fetch all models from OpenRouter and upsert into cache.

        Returns:
            Dict with sync statistics.
        """
        logger.info("Starting full OpenRouter model sync")
        start_time = datetime.utcnow()

        self._sync_job = OpenRouterSyncJob(
            job_type="full_sync",
            status="running",
            started_at=start_time,
        )
        self.db.add(self._sync_job)
        self.db.commit()

        errors: List[str] = []
        models_added = 0
        models_updated = 0

        try:
            # Fetch from OpenRouter (public endpoint, no auth required)
            raw_models = self._fetch_models()
            logger.info(f"Fetched {len(raw_models)} models from OpenRouter")

            for model_data in raw_models:
                try:
                    was_new = self._upsert_model(model_data)
                    if was_new:
                        models_added += 1
                    else:
                        models_updated += 1
                except Exception as e:
                    mid = model_data.get("id", "unknown")
                    error_msg = f"Failed to sync model {mid}: {str(e)}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            self.db.commit()

            self._sync_job.status = "completed"
            self._sync_job.models_synced = models_added + models_updated
            self._sync_job.models_added = models_added
            self._sync_job.models_updated = models_updated
            self._sync_job.errors_count = len(errors)
            self._sync_job.error_details = {"errors": errors} if errors else None

        except Exception as e:
            logger.error(f"OpenRouter sync failed: {e}")
            self._sync_job.status = "failed"
            self._sync_job.error_details = {"fatal_error": str(e)}
            errors.append(str(e))

        finally:
            end_time = datetime.utcnow()
            self._sync_job.completed_at = end_time
            self._sync_job.duration_ms = int((end_time - start_time).total_seconds() * 1000)
            self.db.commit()

        result = {
            "job_id": self._sync_job.id,
            "status": self._sync_job.status,
            "models_synced": models_added + models_updated,
            "models_added": models_added,
            "models_updated": models_updated,
            "errors_count": len(errors),
            "duration_ms": self._sync_job.duration_ms,
        }
        logger.info(f"OpenRouter sync completed: {result}")
        return result

    # =========================================================================
    # API fetching
    # =========================================================================

    def _fetch_models(self) -> List[Dict[str, Any]]:
        """Fetch all models from the OpenRouter public API."""
        with httpx.Client(timeout=60) as client:
            resp = client.get(OPENROUTER_MODELS_URL)
            resp.raise_for_status()
            data = resp.json()
        return data.get("data", [])

    # =========================================================================
    # Upsert logic
    # =========================================================================

    def _upsert_model(self, data: Dict[str, Any]) -> bool:
        """Upsert a single model into the cache.

        Returns True if model was newly created, False if updated.
        """
        model_id = data.get("id", "")
        if not model_id:
            return False

        # Derive provider from model_id (e.g. "anthropic/claude-3.5-sonnet" -> "anthropic")
        provider = model_id.split("/")[0] if "/" in model_id else "unknown"

        # Parse pricing
        pricing = data.get("pricing", {}) or {}
        prompt_cost = self._safe_float(pricing.get("prompt"))
        completion_cost = self._safe_float(pricing.get("completion"))
        image_cost = self._safe_float(pricing.get("image"))
        request_cost = self._safe_float(pricing.get("request"))
        cache_read_cost = self._safe_float(pricing.get("input_cache_read"))
        cache_write_cost = self._safe_float(pricing.get("input_cache_write"))

        # Parse architecture
        arch = data.get("architecture", {}) or {}
        modality = arch.get("modality", "")
        input_modalities = arch.get("input_modalities", []) or []
        output_modalities = arch.get("output_modalities", []) or []
        tokenizer = arch.get("tokenizer", "")

        # Parse top provider
        top_prov = data.get("top_provider", {}) or {}
        top_ctx = top_prov.get("context_length") or 0
        top_max = top_prov.get("max_completion_tokens") or 0
        is_mod = top_prov.get("is_moderated", False)

        # Capabilities from supported_parameters and modality
        supported_params = data.get("supported_parameters", []) or []
        default_params = data.get("default_parameters", {}) or {}
        supports_tools = "tools" in supported_params
        supports_vision = "image" in input_modalities
        supports_streaming = True  # OpenRouter supports streaming for all
        supports_json_mode = "response_format" in supported_params
        supports_reasoning = "reasoning" in supported_params or "reasoning" in str(data.get("name", "")).lower()

        # Categorize
        category = self._categorize_model(data, supports_vision, supports_reasoning, prompt_cost)
        tier = self._tier_model(prompt_cost)
        tags = self._generate_tags(data, supports_tools, supports_vision, supports_reasoning)

        # Context length
        context_length = data.get("context_length") or 0
        max_completion = top_max or (context_length // 4 if context_length else 0)

        existing = self.db.query(OpenRouterModelCache).filter_by(model_id=model_id).first()

        if existing:
            existing.slug = data.get("canonical_slug")
            existing.display_name = data.get("name", model_id)
            existing.description = data.get("description")
            existing.provider = provider
            existing.prompt_cost = prompt_cost
            existing.completion_cost = completion_cost
            existing.image_cost = image_cost
            existing.request_cost = request_cost
            existing.cache_read_cost = cache_read_cost
            existing.cache_write_cost = cache_write_cost
            existing.context_length = context_length
            existing.max_completion_tokens = max_completion
            existing.modality = modality
            existing.input_modalities = input_modalities
            existing.output_modalities = output_modalities
            existing.tokenizer = tokenizer
            existing.supports_tools = supports_tools
            existing.supports_vision = supports_vision
            existing.supports_streaming = supports_streaming
            existing.supports_json_mode = supports_json_mode
            existing.supports_reasoning = supports_reasoning
            existing.supported_parameters = supported_params
            existing.top_provider_context = top_ctx
            existing.top_provider_max_tokens = top_max
            existing.is_moderated = is_mod
            existing.category = category
            existing.tier = tier
            existing.tags = tags
            existing.created_timestamp = data.get("created")
            existing.raw_data = data
            existing.last_synced_at = datetime.utcnow()
            return False
        else:
            new_model = OpenRouterModelCache(
                model_id=model_id,
                slug=data.get("canonical_slug"),
                display_name=data.get("name", model_id),
                description=data.get("description"),
                provider=provider,
                prompt_cost=prompt_cost,
                completion_cost=completion_cost,
                image_cost=image_cost,
                request_cost=request_cost,
                cache_read_cost=cache_read_cost,
                cache_write_cost=cache_write_cost,
                context_length=context_length,
                max_completion_tokens=max_completion,
                modality=modality,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                tokenizer=tokenizer,
                supports_tools=supports_tools,
                supports_vision=supports_vision,
                supports_streaming=supports_streaming,
                supports_json_mode=supports_json_mode,
                supports_reasoning=supports_reasoning,
                supported_parameters=supported_params,
                top_provider_context=top_ctx,
                top_provider_max_tokens=top_max,
                is_moderated=is_mod,
                category=category,
                tier=tier,
                tags=tags,
                created_timestamp=data.get("created"),
                raw_data=data,
            )
            self.db.add(new_model)
            return True

    # =========================================================================
    # Helpers
    # =========================================================================

    @staticmethod
    def _safe_float(val) -> float:
        if val is None:
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    @staticmethod
    def _categorize_model(data: Dict, supports_vision: bool, supports_reasoning: bool, prompt_cost: float) -> str:
        """Categorize model by use-case."""
        name = (data.get("name") or "").lower()
        model_id = (data.get("id") or "").lower()

        if supports_reasoning or "reasoning" in name or "o1" in model_id or "o3" in model_id or "r1" in model_id:
            return "reasoning"
        if supports_vision and "vision" in name:
            return "vision"
        if any(kw in name for kw in ["code", "codex", "starcoder", "deepseek-coder", "codestral"]):
            return "coding"
        if prompt_cost == 0:
            return "free"
        if prompt_cost > 0.00001:  # >$10/1M tokens
            return "premium"
        if prompt_cost > 0.000001:  # >$1/1M tokens
            return "balanced"
        return "fast"

    @staticmethod
    def _tier_model(prompt_cost: float) -> str:
        """Assign a pricing tier."""
        if prompt_cost == 0:
            return "free"
        if prompt_cost <= 0.0000005:  # <= $0.50/1M tokens
            return "budget"
        if prompt_cost <= 0.000003:  # <= $3/1M tokens
            return "mid"
        return "premium"

    @staticmethod
    def _generate_tags(data: Dict, supports_tools: bool, supports_vision: bool, supports_reasoning: bool) -> List[str]:
        """Generate searchable tags."""
        tags = []
        name = (data.get("name") or "").lower()
        model_id = (data.get("id") or "").lower()

        if supports_tools:
            tags.append("function-calling")
        if supports_vision:
            tags.append("vision")
        if supports_reasoning:
            tags.append("reasoning")
        if "instruct" in name or "instruct" in model_id:
            tags.append("instruct")
        if "chat" in name:
            tags.append("chat")
        if "free" in name or (data.get("pricing", {}) or {}).get("prompt") == "0":
            tags.append("free")

        return tags

    # =========================================================================
    # Utility
    # =========================================================================

    def get_last_sync_time(self) -> Optional[datetime]:
        last_job = (
            self.db.query(OpenRouterSyncJob)
            .filter_by(status="completed")
            .order_by(OpenRouterSyncJob.completed_at.desc())
            .first()
        )
        return last_job.completed_at if last_job else None

    def get_sync_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        jobs = (
            self.db.query(OpenRouterSyncJob)
            .order_by(OpenRouterSyncJob.started_at.desc())
            .limit(limit)
            .all()
        )
        return [job.to_dict() for job in jobs]
