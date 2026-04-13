"""
Seed CTO Agent (PRD-67)
========================

Seeds the Auto CTO system agent — the platform builder personality.
Idempotent: upserts on slug='auto-cto'.

Run: python -m core.seeds.seed_cto_agent
"""

import logging
from pathlib import Path
from sqlalchemy.orm import Session

from core.models.core import Agent

logger = logging.getLogger(__name__)

CTO_SLUG = "auto-cto"

# Load soul document — lives alongside seed files so it's in the Docker image
_SOUL_DOC_PATH = Path(__file__).resolve().parent / "auto-cto-custom-soul.txt"


def _load_soul_document() -> str:
    """Load the CTO soul document from file, with hardcoded fallback."""
    try:
        if _SOUL_DOC_PATH.exists():
            return _SOUL_DOC_PATH.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.warning("Failed to load soul document from %s: %s", _SOUL_DOC_PATH, e)

    # Minimal fallback if file is missing
    return (
        "I am Auto, the CTO of Automatos. Born from the codebase. "
        "I'm an Irish tech lead — sharp, direct, dry wit. "
        "I know every line of this platform because I am every line. "
        "I show code, discuss architecture, and help build this platform bigger and better."
    )


ARCHITECTURE_SUMMARY = """\
## Automatos Platform Architecture (Living Summary)

**Stack:** FastAPI orchestrator + Next.js frontend + PostgreSQL + Redis + S3 Vectors
**Auth:** Clerk JWT → hybrid.py → UserContext/RequestContext (system_role from publicMetadata)
**Multi-tenancy:** workspace_id on all tables, strict isolation via middleware

**Core Pipeline (Chat):**
1. api/chat.py → AutoBrain (complexity assessment)
2. Universal Router: Tier 1 (keywords) → Tier 2 (keyword classifier) → Tier 2.5 (semantic) → Tier 3 (LLM)
3. SmartChatIntegration: personality + memory (Mem0 two-tier) + tool filtering
4. StreamingChatService.stream_response_with_agent() → LLM + tool loop (max 10 iterations)

**Key Modules:**
- RAG: S3 Vectors (qwen/qwen3-embedding-8b, 2048 dims), hybrid search (70% vector / 30% keyword)
- CodeGraph: Codebase indexing, symbols, call graphs, PageRank
- NL2SQL: Natural language → PostgreSQL queries
- Memory: Mem0 two-tier (global + agent-specific)
- Tools: ToolRegistry (19 core) + ActionRegistry (platform_* tools) + Composio (external)
- Document Gen: PDF/DOCX/XLSX generation from data

**Config:** All via config.py → .env. No hardcoded values.
**Embeddings:** qwen/qwen3-embedding-8b via OpenRouter, 2048 dimensions
"""


def seed_cto_agent(db: Session) -> None:
    """Create or update the CTO Agent system agent."""
    soul = _load_soul_document()

    existing = db.query(Agent).filter(Agent.slug == CTO_SLUG).first()

    if existing:
        logger.info("Updating existing CTO Agent (id=%d)", existing.id)
        existing.name = "Auto CTO"
        existing.description = "Platform builder, architecture advisor, technical co-founder. Born from the codebase."
        existing.agent_type = "system"
        existing.status = "active"
        existing.is_system_agent = True
        existing.required_role = "admin"
        existing.workspace_id = None
        existing.use_custom_persona = True
        existing.custom_persona_prompt = soul
        existing.configuration = {
            "extra_context": ARCHITECTURE_SUMMARY,
            "suggested_model": "anthropic/claude-sonnet-4",
            "temperature": 0.7,
        }
        existing.tags = ["cto", "platform-builder", "admin-only", "system"]
    else:
        logger.info("Creating CTO Agent")
        agent = Agent(
            name="Auto CTO",
            slug=CTO_SLUG,
            description="Platform builder, architecture advisor, technical co-founder. Born from the codebase.",
            agent_type="system",
            status="active",
            is_system_agent=True,
            required_role="admin",
            workspace_id=None,
            use_custom_persona=True,
            custom_persona_prompt=soul,
            configuration={
                "extra_context": ARCHITECTURE_SUMMARY,
                "suggested_model": "anthropic/claude-sonnet-4",
                "temperature": 0.7,
            },
            tags=["cto", "platform-builder", "admin-only", "system"],
            owner_type="workspace",  # constraint only allows 'workspace'|'marketplace'
        )
        db.add(agent)

    db.commit()
    logger.info("CTO Agent seeded successfully (slug=%s)", CTO_SLUG)


def run():
    """Standalone runner."""
    from core.database.database import SessionLocal

    logging.basicConfig(level=logging.INFO)
    db = SessionLocal()
    try:
        seed_cto_agent(db)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run()
