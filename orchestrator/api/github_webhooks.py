"""
GitHub Webhook Integration
===========================

Handles GitHub webhooks for automated workflow triggers.
Specifically designed for PR review workflows.
"""

import hashlib
import hmac
import logging
from typing import Optional
from fastapi import APIRouter, Request, HTTPException, Header, Depends, BackgroundTasks
from sqlalchemy.orm import Session

from core.database.database import get_db
from api.workflows import execute_workflow

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/github", tags=["github"])


def verify_github_signature(payload: bytes, signature: str, secret: str) -> bool:
    """Verify GitHub webhook signature"""
    if not signature or not signature.startswith('sha256='):
        return False
    
    expected_signature = 'sha256=' + hmac.new(
        secret.encode('utf-8'),
        payload,
        hashlib.sha256
    ).hexdigest()
    
    return hmac.compare_digest(signature, expected_signature)


@router.post("/webhook")
async def github_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
    x_github_event: Optional[str] = Header(None),
    x_hub_signature_256: Optional[str] = Header(None),
    db: Session = Depends(get_db)
):
    """
    GitHub webhook endpoint for PR events
    
    Configure in GitHub:
    1. Go to repo Settings → Webhooks → Add webhook
    2. Payload URL: {API_URL}/api/github/webhook (replace {API_URL} with your API server URL)
    3. Content type: application/json
    4. Secret: (set GITHUB_WEBHOOK_SECRET env var)
    5. Events: Pull requests
    """
    
    # Get raw body for signature verification
    body = await request.body()
    payload = await request.json()
    
    # Verify signature (optional but recommended)
    import os
    webhook_secret = os.getenv("GITHUB_WEBHOOK_SECRET", "")
    if webhook_secret and x_hub_signature_256:
        if not verify_github_signature(body, x_hub_signature_256, webhook_secret):
            logger.warning("Invalid GitHub webhook signature")
            raise HTTPException(status_code=401, detail="Invalid signature")
    
    logger.info(f"📥 GitHub webhook: {x_github_event}")
    
    # Handle pull request events
    if x_github_event == "pull_request":
        action = payload.get("action")
        pr = payload.get("pull_request", {})
        repo = payload.get("repository", {})
        
        # Trigger on PR opened or synchronized (new commits)
        if action in ["opened", "synchronize", "reopened"]:
            pr_number = pr.get("number")
            pr_title = pr.get("title")
            pr_url = pr.get("html_url")
            pr_branch = pr.get("head", {}).get("ref")
            base_branch = pr.get("base", {}).get("ref")
            repo_name = repo.get("full_name")
            repo_url = repo.get("clone_url")
            
            logger.info(f"🔍 PR #{pr_number} {action}: {pr_title}")
            
            # Find the PR Review workflow by name
            from core.models import Workflow
            import os
            pr_workflow_name = os.getenv("GITHUB_PR_WORKFLOW_NAME", "PR Code Review")
            pr_workflow = db.query(Workflow).filter(
                Workflow.name == pr_workflow_name
            ).first()
            
            if not pr_workflow:
                logger.warning("PR Code Review workflow not found - creating from template")
                # Could auto-create here, but better to require manual setup
                return {
                    "status": "no_workflow",
                    "message": "PR Code Review workflow not found. Please create it first.",
                    "pr": pr_number
                }
            
            # Trigger workflow with PR context
            execution_data = {
                "workflow_id": pr_workflow.id,
                "options": {
                    "priority": "high"
                },
                "context": {
                    "pr_number": pr_number,
                    "pr_title": pr_title,
                    "pr_url": pr_url,
                    "pr_branch": pr_branch,
                    "base_branch": base_branch,
                    "repo_name": repo_name,
                    "repo_url": repo_url,
                    "action": action,
                    "triggered_by": "github_webhook"
                }
            }
            
            try:
                # Execute workflow asynchronously
                execution = await execute_workflow(
                    pr_workflow.id,
                    execution_data,
                    background_tasks,
                    db
                )
                
                logger.info(f"✅ Triggered PR review workflow for PR #{pr_number}")
                
                return {
                    "status": "success",
                    "message": f"PR review workflow triggered for PR #{pr_number}",
                    "execution_id": execution.get("id"),
                    "pr": pr_number,
                    "workflow": pr_workflow.name
                }
                
            except Exception as e:
                logger.error(f"Failed to trigger PR review: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    # Health check / ping event
    elif x_github_event == "ping":
        return {
            "status": "success",
            "message": "GitHub webhook configured successfully",
            "zen": payload.get("zen")
        }
    
    # Unsupported event
    return {
        "status": "ignored",
        "event": x_github_event,
        "message": "Event not configured for automation"
    }


@router.get("/health")
async def github_integration_health():
    """Health check for GitHub integration"""
    import os
    return {
        "status": "healthy",
        "webhook_secret_configured": bool(os.getenv("GITHUB_WEBHOOK_SECRET")),
        "endpoint": "/api/github/webhook"
    }

