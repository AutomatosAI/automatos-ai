"""
Automatos Alert Ingest — FastAPI Router

Receives AlertManager webhook payloads, deduplicates by fingerprint,
and stores in the infrastructure_alerts table.

Usage:
    from automatos_alerts import create_alerts_router
    app.include_router(create_alerts_router(get_db), prefix="/api")
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)

ALERT_INGEST_TOKEN = os.environ.get("ALERT_INGEST_TOKEN", "")


# ─────────────────────────────────────────────
# Pydantic Models
# ─────────────────────────────────────────────
class AlertLabel(BaseModel):
    alertname: str
    severity: Optional[str] = "unknown"
    service: Optional[str] = None
    instance: Optional[str] = None
    job: Optional[str] = None


class AlertAnnotation(BaseModel):
    summary: Optional[str] = None
    description: Optional[str] = None
    runbook_url: Optional[str] = None


class Alert(BaseModel):
    status: str  # "firing" or "resolved"
    labels: dict
    annotations: dict = {}
    startsAt: Optional[str] = None
    endsAt: Optional[str] = None
    generatorURL: Optional[str] = None
    fingerprint: Optional[str] = None


class AlertManagerPayload(BaseModel):
    version: Optional[str] = None
    groupKey: Optional[str] = None
    truncatedAlerts: Optional[int] = 0
    status: str  # "firing" or "resolved"
    receiver: Optional[str] = None
    groupLabels: dict = {}
    commonLabels: dict = {}
    commonAnnotations: dict = {}
    externalURL: Optional[str] = None
    alerts: list[Alert]


# ─────────────────────────────────────────────
# SQL for infrastructure_alerts table
# ─────────────────────────────────────────────
CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS infrastructure_alerts (
    id SERIAL PRIMARY KEY,
    fingerprint VARCHAR(64) NOT NULL,
    alertname VARCHAR(255) NOT NULL,
    severity VARCHAR(32) NOT NULL DEFAULT 'unknown',
    status VARCHAR(32) NOT NULL DEFAULT 'firing',
    service VARCHAR(255),
    instance VARCHAR(255),
    labels JSONB NOT NULL DEFAULT '{}',
    annotations JSONB NOT NULL DEFAULT '{}',
    starts_at TIMESTAMPTZ,
    ends_at TIMESTAMPTZ,
    generator_url TEXT,
    receiver VARCHAR(255),
    raw_payload JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    resolved_at TIMESTAMPTZ,
    UNIQUE(fingerprint, starts_at)
);

CREATE INDEX IF NOT EXISTS idx_infra_alerts_fingerprint ON infrastructure_alerts(fingerprint);
CREATE INDEX IF NOT EXISTS idx_infra_alerts_status ON infrastructure_alerts(status);
CREATE INDEX IF NOT EXISTS idx_infra_alerts_severity ON infrastructure_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_infra_alerts_alertname ON infrastructure_alerts(alertname);
CREATE INDEX IF NOT EXISTS idx_infra_alerts_created ON infrastructure_alerts(created_at DESC);
"""

# ─────────────────────────────────────────────
# Upsert query (dedup by fingerprint + starts_at)
# ─────────────────────────────────────────────
UPSERT_SQL = """
INSERT INTO infrastructure_alerts
    (fingerprint, alertname, severity, status, service, instance,
     labels, annotations, starts_at, ends_at, generator_url, receiver, raw_payload)
VALUES
    (:fingerprint, :alertname, :severity, :status, :service, :instance,
     :labels, :annotations, :starts_at, :ends_at, :generator_url, :receiver, :raw_payload)
ON CONFLICT (fingerprint, starts_at) DO UPDATE SET
    status = EXCLUDED.status,
    ends_at = EXCLUDED.ends_at,
    annotations = EXCLUDED.annotations,
    raw_payload = EXCLUDED.raw_payload,
    updated_at = NOW(),
    resolved_at = CASE
        WHEN EXCLUDED.status = 'resolved' THEN NOW()
        ELSE infrastructure_alerts.resolved_at
    END
RETURNING id;
"""


def _compute_fingerprint(labels: dict) -> str:
    """Stable fingerprint from sorted label pairs."""
    canonical = json.dumps(labels, sort_keys=True)
    return hashlib.sha256(canonical.encode()).hexdigest()[:16]


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts or ts == "0001-01-01T00:00:00Z":
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def create_alerts_router(get_db_dependency) -> APIRouter:
    """Create the alerts ingest router with the given DB dependency.

    Args:
        get_db_dependency: FastAPI dependency that yields a DB session
    """
    router = APIRouter(tags=["alerts"])

    @router.post("/alerts/ingest")
    async def ingest_alerts(
        payload: AlertManagerPayload,
        request: Request,
        authorization: Optional[str] = Header(None),
        x_alert_source: Optional[str] = Header(None),
        x_alert_priority: Optional[str] = Header(None),
        db=Depends(get_db_dependency),
    ):
        # Validate Bearer token
        if ALERT_INGEST_TOKEN:
            expected = f"Bearer {ALERT_INGEST_TOKEN}"
            if authorization != expected:
                raise HTTPException(status_code=401, detail="Invalid token")

        processed = 0
        errors = 0

        for alert in payload.alerts:
            try:
                fingerprint = alert.fingerprint or _compute_fingerprint(alert.labels)
                alertname = alert.labels.get("alertname", "unknown")
                severity = alert.labels.get("severity", "unknown")
                service = alert.labels.get("service", alert.labels.get("job", None))
                instance = alert.labels.get("instance", None)

                from sqlalchemy import text
                db.execute(
                    text(UPSERT_SQL),
                    {
                        "fingerprint": fingerprint,
                        "alertname": alertname,
                        "severity": severity,
                        "status": alert.status,
                        "service": service,
                        "instance": instance,
                        "labels": json.dumps(alert.labels),
                        "annotations": json.dumps(alert.annotations),
                        "starts_at": _parse_timestamp(alert.startsAt),
                        "ends_at": _parse_timestamp(alert.endsAt),
                        "generator_url": alert.generatorURL,
                        "receiver": payload.receiver,
                        "raw_payload": json.dumps(alert.dict()),
                    },
                )
                processed += 1
            except Exception as e:
                logger.error(f"Failed to process alert {alert.labels}: {e}")
                errors += 1

        db.commit()

        logger.info(
            f"Alert ingest: {processed} processed, {errors} errors, "
            f"source={x_alert_source}, priority={x_alert_priority}"
        )

        return {
            "status": "ok",
            "processed": processed,
            "errors": errors,
        }

    @router.get("/alerts")
    async def list_alerts(
        status: Optional[str] = None,
        severity: Optional[str] = None,
        limit: int = 50,
        db=Depends(get_db_dependency),
    ):
        from sqlalchemy import text
        conditions = []
        params = {"limit": limit}

        if status:
            conditions.append("status = :status")
            params["status"] = status
        if severity:
            conditions.append("severity = :severity")
            params["severity"] = severity

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = db.execute(
            text(f"""
                SELECT id, fingerprint, alertname, severity, status, service,
                       annotations, starts_at, resolved_at, created_at
                FROM infrastructure_alerts
                {where}
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            params,
        ).fetchall()

        return [dict(row._mapping) for row in rows]

    return router
