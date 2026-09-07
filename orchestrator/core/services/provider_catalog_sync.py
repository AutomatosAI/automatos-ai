"""
Provider catalogue sync (PRD-236 W1 S1.2)
=========================================

Fills ``llm_models`` with one row per ROUTE — ``(serving_provider, model_id)``.

- ``openrouter``: runs the existing cache sync (``OpenRouterSyncService``,
  OpenRouter → ``openrouter_models_cache``) and then PROJECTS the cache into
  ``llm_models`` rows served by OpenRouter, priced at OpenRouter's prices.
- ``nvidia``: reads build.nvidia.com's public model list
  (``GET {NVIDIA_BASE_URL}/models`` — ids only, no pricing or capability
  fields) and writes rows served by NVIDIA at price 0, borrowing context
  window, output cap, tool/vision flags, description and display name from
  the OpenRouter cache row of the same vendor id when one exists.
  Non-chat ids (embeddings, rerankers, reward models, OCR/parsers, safety
  classifiers) are skipped — this is the chat catalogue.

Direct providers (OpenAI, Anthropic, Google, DeepSeek…) keep their seeded
rows; their ``/models`` endpoints need a key and publish no prices.

Job history rides on ``openrouter_sync_jobs`` (``job_type`` distinguishes the
provider) — no new table (CLAUDE.md §4).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from core.llm import providers as registry
from core.models.core import LLMModel
from core.models.openrouter_cache import OpenRouterModelCache, OpenRouterSyncJob
from core.services.openrouter_sync_service import OpenRouterSyncService

logger = logging.getLogger(__name__)

SYNCABLE_PROVIDERS = ("openrouter", "nvidia")
JOB_TYPES = {"openrouter": "full_sync", "nvidia": "nvidia_sync"}

# NVIDIA lists embeddings, rerankers, reward models, OCR/parsers and safety
# classifiers next to chat models. None of them answer chat completions.
_NON_CHAT_ID = re.compile(
    r"(embed|reward|rerank|parse|ocr|deplot|kosmos|fuyu|safety|guard|diffusion|detector|nvclip|calibration)",
    re.I,
)
# NVIDIA vendor slug → OpenRouter vendor slug, for metadata borrowing.
_VENDOR_ALIASES = {"deepseek-ai": "deepseek", "meta": "meta-llama"}

NVIDIA_FETCH_TIMEOUT_S = 30


class ProviderCatalogSync:
    """One entry point per provider; every route lands in ``llm_models``."""

    def __init__(self, db: Session):
        self.db = db

    # ------------------------------------------------------------------ #
    # Dispatch
    # ------------------------------------------------------------------ #

    def sync(self, provider: str, api_key: Optional[str] = None) -> Dict[str, Any]:
        slug = registry.normalize_slug(provider)
        if slug == "openrouter":
            return self.sync_openrouter()
        if slug == "nvidia":
            return self.sync_nvidia(api_key=api_key)
        raise ValueError(f"Provider '{provider}' has no catalogue sync (syncable: {', '.join(SYNCABLE_PROVIDERS)})")

    def last_synced(self) -> Dict[str, Optional[str]]:
        """Latest completed sync per syncable provider, ISO timestamps."""
        out: Dict[str, Optional[str]] = {}
        for slug, job_type in JOB_TYPES.items():
            job = (
                self.db.query(OpenRouterSyncJob)
                .filter(OpenRouterSyncJob.job_type == job_type, OpenRouterSyncJob.status == "completed")
                .order_by(OpenRouterSyncJob.completed_at.desc().nullslast())
                .first()
            )
            # Naive UTC in the DB → explicit UTC on the wire, or the browser reads it as local time.
            out[slug] = (
                job.completed_at.replace(tzinfo=timezone.utc).isoformat() if job and job.completed_at else None
            )
        return out

    # ------------------------------------------------------------------ #
    # OpenRouter: cache sync + projection
    # ------------------------------------------------------------------ #

    def sync_openrouter(self) -> Dict[str, Any]:
        result = OpenRouterSyncService(self.db).run_full_sync()
        result["projection"] = self.project_openrouter_cache()
        return result

    def project_openrouter_cache(self) -> Dict[str, Any]:
        """Every active cache row becomes an OpenRouter-served catalogue row."""
        rows = (
            self.db.query(OpenRouterModelCache)
            .filter(OpenRouterModelCache.status == "active")
            .all()
        )
        for cached in rows:
            self._upsert_route("openrouter", cached.model_id, self._values_from_cache(cached))
        self.db.commit()
        logger.info("[CatalogSync] projected %d OpenRouter rows into llm_models", len(rows))
        return {"provider": "openrouter", "rows": len(rows)}

    @staticmethod
    def _values_from_cache(cached: OpenRouterModelCache) -> Dict[str, Any]:
        return dict(
            provider=cached.provider,
            display_name=cached.display_name,
            description=cached.description,
            model_family=cached.provider,
            context_window=int(cached.context_length or 0),
            max_output_tokens=int(cached.max_completion_tokens or 0),
            input_cost_per_1k_tokens=float(cached.prompt_cost or 0) * 1000,
            output_cost_per_1k_tokens=float(cached.completion_cost or 0) * 1000,
            supports_functions=bool(cached.supports_tools),
            supports_vision=bool(cached.supports_vision),
            supports_streaming=True if cached.supports_streaming is None else bool(cached.supports_streaming),
            status="active",
            sourcing="aggregator",
            category=cached.category,
            tags=list(cached.tags or []),
            capabilities={},
            recommended_for=[],
            external_id=cached.model_id,
            pricing_updated_at=datetime.utcnow(),
        )

    # ------------------------------------------------------------------ #
    # NVIDIA: public list + borrowed metadata
    # ------------------------------------------------------------------ #

    def sync_nvidia(self, api_key: Optional[str] = None) -> Dict[str, Any]:
        started = datetime.utcnow()
        job = OpenRouterSyncJob(job_type=JOB_TYPES["nvidia"], status="running", started_at=started)
        self.db.add(job)
        self.db.commit()

        try:
            ids = self._fetch_nvidia_ids(api_key)
            synced = borrowed = skipped = 0
            kept: List[str] = []
            for model_id in ids:
                if _NON_CHAT_ID.search(model_id):
                    skipped += 1
                    continue
                cached = self._borrow(model_id)
                if cached is not None:
                    borrowed += 1
                self._upsert_route("nvidia", model_id, self._values_for_nvidia(model_id, cached))
                kept.append(model_id)
                synced += 1
            # Routes NVIDIA no longer lists (or that the chat filter now excludes)
            # stop being offered; installs keep their row, marked deprecated.
            deprecated = (
                self.db.query(LLMModel)
                .filter(
                    LLMModel.serving_provider == "nvidia",
                    LLMModel.status == "active",
                    ~LLMModel.model_id.in_(kept),
                )
                .update({"status": "deprecated"}, synchronize_session=False)
            )

            finished = datetime.utcnow()
            job.status = "completed"
            job.models_synced = synced
            job.models_updated = synced
            job.completed_at = finished
            job.duration_ms = int((finished - started).total_seconds() * 1000)
            job.job_metadata = {
                "skipped_non_chat": skipped, "borrowed_metadata": borrowed,
                "listed": len(ids), "deprecated": int(deprecated or 0),
            }
            self.db.commit()
            logger.info(
                "[CatalogSync] NVIDIA: %d routes (%d with borrowed metadata, %d non-chat skipped, %d deprecated)",
                synced, borrowed, skipped, int(deprecated or 0),
            )
            return {
                "provider": "nvidia", "status": "completed", "models_synced": synced,
                "borrowed_metadata": borrowed, "skipped_non_chat": skipped, "listed": len(ids),
                "deprecated": int(deprecated or 0),
            }
        except Exception as exc:
            self.db.rollback()
            job.status = "failed"
            job.errors_count = 1
            job.error_details = {"error": str(exc)[:500]}
            job.completed_at = datetime.utcnow()
            self.db.add(job)
            self.db.commit()
            logger.error("[CatalogSync] NVIDIA sync failed: %s", exc)
            raise

    @staticmethod
    def _fetch_nvidia_ids(api_key: Optional[str]) -> List[str]:
        base = registry.base_url_for("nvidia")
        if not base:
            raise RuntimeError("NVIDIA base URL is not configured (NVIDIA_BASE_URL)")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        with httpx.Client(timeout=NVIDIA_FETCH_TIMEOUT_S) as client:
            resp = client.get(f"{base.rstrip('/')}/models", headers=headers)
            resp.raise_for_status()
            data = resp.json()
        return sorted({m["id"] for m in data.get("data", []) if isinstance(m, dict) and m.get("id")})

    def _borrow(self, model_id: str) -> Optional[OpenRouterModelCache]:
        """The OpenRouter cache row for the same vendor id, if any."""
        q = self.db.query(OpenRouterModelCache)
        exact = q.filter(OpenRouterModelCache.model_id == model_id).first()
        if exact is not None:
            return exact
        vendor, _, name = model_id.partition("/")
        alias = _VENDOR_ALIASES.get(vendor)
        if alias and name:
            aliased = q.filter(OpenRouterModelCache.model_id == f"{alias}/{name}").first()
            if aliased is not None:
                return aliased
        if name:
            return q.filter(OpenRouterModelCache.model_id.endswith(f"/{name}")).first()
        return None

    def _values_for_nvidia(self, model_id: str, cached: Optional[OpenRouterModelCache]) -> Dict[str, Any]:
        vendor, _, name = model_id.partition("/")
        if cached is not None:
            values = self._values_from_cache(cached)
        else:
            values = dict(
                display_name=self._display_name(name or model_id),
                description="Hosted by NVIDIA (build.nvidia.com). Context window not published by NVIDIA's catalogue.",
                model_family=vendor,
                context_window=0,
                max_output_tokens=0,
                supports_functions=False,
                supports_vision=False,
                supports_streaming=True,
                status="active",
                category="free",
                tags=[],
                capabilities={},
                recommended_for=[],
                pricing_updated_at=datetime.utcnow(),
            )
        values.update(
            provider=vendor or "nvidia",
            input_cost_per_1k_tokens=0.0,
            output_cost_per_1k_tokens=0.0,
            sourcing="hosted_open",
            category=values.get("category") or "free",
            tags=sorted(set(list(values.get("tags") or []) + ["free", "nvidia"])),
            external_id=model_id,
        )
        return values

    @staticmethod
    def _display_name(name: str) -> str:
        return " ".join(part[:1].upper() + part[1:] for part in name.replace("_", "-").split("-") if part)

    # ------------------------------------------------------------------ #
    # Upsert
    # ------------------------------------------------------------------ #

    def _upsert_route(self, serving_provider: str, model_id: str, values: Dict[str, Any]) -> None:
        """INSERT … ON CONFLICT (serving_provider, model_id) DO UPDATE.

        Workspace-facing state (install_count, featured/default flags,
        workspace_id, created_at) is never overwritten by a sync.
        """
        now = datetime.utcnow()
        update = dict(values, serving_provider=serving_provider, model_id=model_id, updated_at=now)
        insert = dict(
            update,
            created_at=now,
            install_count=0,
            popularity_score=0,
            is_featured=False,
            is_default=False,
            default_temperature=0.7,
            min_temperature=0.0,
            max_temperature=2.0,
        )
        stmt = (
            pg_insert(LLMModel.__table__)
            .values(**insert)
            .on_conflict_do_update(constraint="uq_llm_models_provider_model", set_=update)
        )
        self.db.execute(stmt)
