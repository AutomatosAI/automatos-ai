#!/usr/bin/env python3
"""
Setup Jira Trigger Subscription

Management script to register JIRA_NEW_ISSUE_TRIGGER for a Jira project
(default: PILOT) so new tickets fire webhooks to the orchestrator.

Usage:
    python scripts/setup_jira_trigger.py [--project PILOT] [--agent-id 1] [--workflow-id 2]

Environment:
    COMPOSIO_API_KEY or COMPOSIO_KEY — Composio API key (required)
    BACKEND_URL — Orchestrator backend URL (default: http://localhost:8000)
    DATABASE_URL or POSTGRES_* — Database connection (required)

Webhook URL Configuration:
    After running this script, ensure the webhook callback URL is registered
    in your Composio dashboard:
      1. Go to https://app.composio.dev → Project Settings → Webhooks
      2. Set the webhook URL to: {BACKEND_URL}/api/composio/webhook
      3. (Optional) Set a webhook secret and add it as COMPOSIO_WEBHOOK_SECRET in .env
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add orchestrator root to path so imports work
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Register JIRA_NEW_ISSUE_TRIGGER for a Jira project",
    )
    parser.add_argument(
        "--project",
        default="PILOT",
        help="Jira project key to monitor (default: PILOT)",
    )
    parser.add_argument(
        "--agent-id",
        type=int,
        default=None,
        help="Agent ID to route new issues to (optional)",
    )
    parser.add_argument(
        "--workflow-id",
        type=int,
        default=None,
        help="Workflow ID to route new issues to (optional)",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Read Composio API key from env / .env
    # ------------------------------------------------------------------
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path)

    composio_key = os.getenv("COMPOSIO_API_KEY") or os.getenv("COMPOSIO_KEY")
    if not composio_key:
        logger.error(
            "COMPOSIO_API_KEY or COMPOSIO_KEY not set. "
            "Set it in the environment or in orchestrator/.env"
        )
        sys.exit(1)

    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    callback_url = f"{backend_url}/api/composio/webhook"

    trigger_name = "JIRA_NEW_ISSUE_TRIGGER"
    project_key = args.project

    print(f"Setting up {trigger_name} for project: {project_key}")
    print(f"Callback URL: {callback_url}")

    # ------------------------------------------------------------------
    # 2. Call Composio SDK to create the trigger
    # ------------------------------------------------------------------
    from core.composio.client import ComposioClient

    client = ComposioClient(api_key=composio_key)

    # Get or create entity for the workspace
    from core.database.database import SessionLocal
    from core.composio.entity_manager import EntityManager
    from config import config, DEFAULT_TENANT_ID

    db = SessionLocal()
    try:
        entity_manager = EntityManager(db)
        entity = entity_manager.get_or_create_entity(DEFAULT_TENANT_ID)
        entity_id = entity["composio_entity_id"]
        entity_pk = entity["id"]

        print(f"Using entity: {entity_id} (DB id={entity_pk})")

        # Subscribe to trigger via Composio SDK
        result = client.subscribe_to_trigger(
            entity_id=entity_id,
            trigger_name=trigger_name,
            callback_url=callback_url,
        )

        composio_subscription_id = result.get("id", "")
        print(f"Composio subscription created: id={composio_subscription_id}")

        # ------------------------------------------------------------------
        # 3. Create TriggerSubscription record in database
        # ------------------------------------------------------------------
        from core.models.composio import TriggerSubscription

        subscription = TriggerSubscription(
            entity_id=entity_pk,
            trigger_name=trigger_name,
            callback_url=callback_url,
            composio_subscription_id=str(composio_subscription_id),
            agent_id=args.agent_id,
            workflow_id=args.workflow_id,
            is_active=True,
        )
        db.add(subscription)
        db.commit()
        db.refresh(subscription)

        # ------------------------------------------------------------------
        # 4. Print confirmation
        # ------------------------------------------------------------------
        print()
        print("=" * 60)
        print("Jira Trigger Subscription Created Successfully")
        print("=" * 60)
        print(f"  Trigger:          {trigger_name}")
        print(f"  Project:          {project_key}")
        print(f"  Subscription ID:  {subscription.id}")
        print(f"  Composio ID:      {composio_subscription_id}")
        print(f"  Callback URL:     {callback_url}")
        print(f"  Agent ID:         {args.agent_id or '(none — router decides)'}")
        print(f"  Workflow ID:      {args.workflow_id or '(none — router decides)'}")
        print()
        print("Next steps:")
        print("  1. Verify webhook URL in Composio dashboard:")
        print("     https://app.composio.dev → Project Settings → Webhooks")
        print(f"  2. Set webhook URL to: {callback_url}")
        print("  3. (Optional) Set COMPOSIO_WEBHOOK_SECRET in .env for signature verification")
        print("  4. Create a new issue in Jira project %s to test" % project_key)
        print("=" * 60)

    except Exception as e:
        logger.exception("Failed to set up Jira trigger")
        db.rollback()
        sys.exit(1)
    finally:
        db.close()


if __name__ == "__main__":
    main()
