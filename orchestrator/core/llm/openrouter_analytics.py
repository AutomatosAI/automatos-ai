"""
OpenRouter Analytics Service
=============================

Fetches usage data from OpenRouter's management API endpoints:
- /api/v1/activity  — daily usage breakdown by model
- /api/v1/credits   — account credits balance
- /api/v1/key       — key limits and usage stats

Syncs activity data into the local llm_usage table for analytics queries.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

import httpx

from config import config

logger = logging.getLogger(__name__)

OPENROUTER_BASE = config.OPENROUTER_BASE_URL


class OpenRouterAnalyticsService:
    """Fetches and syncs OpenRouter analytics data."""

    def __init__(self):
        self._timeout = 30.0

    def _headers(self, api_key: str) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": config.OPENROUTER_SITE_URL,
            "X-Title": "Automatos AI",
        }

    # ------------------------------------------------------------------
    # Activity sync
    # ------------------------------------------------------------------

    async def sync_activity(self, api_key: str, workspace_id: UUID) -> Dict[str, Any]:
        """
        Fetch usage rows from OpenRouter /api/v1/activity and upsert
        into the local llm_usage table (deduped by workspace + model + date).

        Returns a summary dict with counts of synced/skipped rows.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{OPENROUTER_BASE}/activity",
                    headers=self._headers(api_key),
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "OpenRouter activity fetch failed: HTTP %s – %s",
                exc.response.status_code,
                exc.response.text[:200],
            )
            return {"synced": 0, "skipped": 0, "error": f"HTTP {exc.response.status_code}"}
        except Exception as exc:
            logger.warning("OpenRouter activity fetch error: %s", exc)
            return {"synced": 0, "skipped": 0, "error": str(exc)}

        # The response is {"data": [ {...}, ... ]}
        rows = data.get("data", [])
        if not rows:
            return {"synced": 0, "skipped": 0}

        return self._upsert_activity_rows(rows, workspace_id)

    def _upsert_activity_rows(
        self, rows: List[Dict[str, Any]], workspace_id: UUID
    ) -> Dict[str, Any]:
        """Upsert OpenRouter activity rows into llm_usage."""
        from core.database.database import SessionLocal
        from core.models.core import LLMUsage

        synced = 0
        skipped = 0
        db = SessionLocal()
        try:
            for row in rows:
                model_id = row.get("model", "unknown")
                # OpenRouter returns date as ISO string or YYYY-MM-DD
                date_str = row.get("date", "")
                try:
                    row_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    row_date = datetime.utcnow()

                # Check for existing row (dedup by workspace + model + date)
                existing = (
                    db.query(LLMUsage)
                    .filter(
                        LLMUsage.workspace_id == workspace_id,
                        LLMUsage.model_id == model_id,
                        LLMUsage.created_at == row_date,
                    )
                    .first()
                )
                if existing:
                    skipped += 1
                    continue

                cost = float(row.get("usage", 0) or 0)
                prompt_tokens = int(row.get("prompt_tokens", 0) or 0)
                completion_tokens = int(row.get("completion_tokens", 0) or 0)
                total_tokens = prompt_tokens + completion_tokens
                requests_count = int(row.get("requests", 0) or 0)
                is_byok = bool(row.get("byok_usage_inference", False))

                usage_row = LLMUsage(
                    workspace_id=workspace_id,
                    model_id=model_id,
                    provider="openrouter",
                    tier="aggregator",
                    request_type="activity_sync",
                    input_tokens=prompt_tokens,
                    output_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    input_cost=0.0,
                    output_cost=0.0,
                    total_cost=cost,
                    is_byok=is_byok,
                    latency_ms=None,
                    status="success",
                    error_message=None,
                    execution_id=f"openrouter_sync_{model_id}_{date_str}",
                    created_at=row_date,
                )
                db.add(usage_row)
                synced += 1

            db.commit()
        except Exception as exc:
            db.rollback()
            logger.warning("Failed to upsert OpenRouter activity: %s", exc)
            return {"synced": 0, "skipped": 0, "error": str(exc)}
        finally:
            db.close()

        return {"synced": synced, "skipped": skipped}

    # ------------------------------------------------------------------
    # Credits
    # ------------------------------------------------------------------

    async def get_credits(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Fetch account credits from OpenRouter /api/v1/credits.

        Returns dict with total_credits and total_usage, or None on error.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{OPENROUTER_BASE}/credits",
                    headers=self._headers(api_key),
                )
                resp.raise_for_status()
                data = resp.json()
            return {
                "total_credits": data.get("total_credits", 0),
                "total_usage": data.get("total_usage", 0),
            }
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "OpenRouter credits fetch failed: HTTP %s", exc.response.status_code
            )
            return None
        except Exception as exc:
            logger.warning("OpenRouter credits fetch error: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Key info
    # ------------------------------------------------------------------

    async def get_key_info(self, api_key: str) -> Optional[Dict[str, Any]]:
        """
        Fetch key limits/usage from OpenRouter /api/v1/key.

        Returns dict with limit, limit_remaining, usage stats, or None on error.
        """
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{OPENROUTER_BASE}/key",
                    headers=self._headers(api_key),
                )
                resp.raise_for_status()
                data = resp.json()

            key_data = data.get("data", data)
            return {
                "limit": key_data.get("limit"),
                "limit_remaining": key_data.get("limit_remaining"),
                "usage_daily": key_data.get("usage_daily", 0),
                "usage_weekly": key_data.get("usage_weekly", 0),
                "usage_monthly": key_data.get("usage_monthly", 0),
                "is_free_tier": key_data.get("is_free_tier", False),
                "rate_limit": key_data.get("rate_limit", {}),
            }
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "OpenRouter key info fetch failed: HTTP %s", exc.response.status_code
            )
            return None
        except Exception as exc:
            logger.warning("OpenRouter key info fetch error: %s", exc)
            return None
