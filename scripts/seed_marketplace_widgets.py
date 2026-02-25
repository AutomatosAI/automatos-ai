"""
Seed marketplace with internal Automatos AI widgets.

Usage:
    cd orchestrator && python -m scripts.seed_marketplace_widgets
    OR
    python scripts/seed_marketplace_widgets.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'orchestrator'))

from uuid import uuid4
from datetime import datetime, timezone
from random import randint, uniform
from sqlalchemy import text, create_engine
from core.database.database import get_database_url

# ── Widget definitions ──────────────────────────────────────────────────────

WIDGETS = [
    {
        "name": "chat-assistant",
        "display_name": "Chat Assistant",
        "description": (
            "AI-powered chat interface with streaming responses, tool execution, "
            "and conversation history."
        ),
        "categories": ["productivity", "ai"],
        "permissions": ["chat"],
        "icon_url": "/icons/widgets/chat.svg",
        "readme": (
            "# Chat Assistant\n\n"
            "A full-featured AI chat interface built into Automatos.\n\n"
            "## Features\n"
            "- Streaming token-by-token responses\n"
            "- Tool / function-call execution with inline results\n"
            "- Persistent conversation history per workspace\n"
            "- Markdown, code-block, and LaTeX rendering\n"
            "- File and image attachment support\n\n"
            "## Permissions\n"
            "Requires the **chat** permission to send and receive messages."
        ),
    },
    {
        "name": "data-explorer",
        "display_name": "Data Explorer",
        "description": (
            "Interactive data visualization with SQL queries, charts, and table views."
        ),
        "categories": ["analytics", "data"],
        "permissions": ["data:query"],
        "icon_url": "/icons/widgets/data-explorer.svg",
        "readme": (
            "# Data Explorer\n\n"
            "Run SQL queries against connected data sources and visualize results "
            "instantly.\n\n"
            "## Features\n"
            "- SQL editor with syntax highlighting and autocomplete\n"
            "- Bar, line, pie, and scatter chart types\n"
            "- Sortable, filterable data tables\n"
            "- Export to CSV and JSON\n\n"
            "## Permissions\n"
            "Requires the **data:query** permission to execute queries."
        ),
    },
    {
        "name": "document-viewer",
        "display_name": "Document Viewer",
        "description": (
            "Rich document viewer with search, annotations, and markdown rendering."
        ),
        "categories": ["productivity", "content"],
        "permissions": ["documents:read"],
        "icon_url": "/icons/widgets/document-viewer.svg",
        "readme": (
            "# Document Viewer\n\n"
            "View and annotate documents directly inside your workspace.\n\n"
            "## Features\n"
            "- PDF, Markdown, and plain-text rendering\n"
            "- Full-text search within documents\n"
            "- Highlight and annotation support\n"
            "- Side-by-side comparison mode\n\n"
            "## Permissions\n"
            "Requires the **documents:read** permission."
        ),
    },
    {
        "name": "code-inspector",
        "display_name": "Code Inspector",
        "description": (
            "Syntax-highlighted code viewer with line numbers, search, and diff support."
        ),
        "categories": ["development", "productivity"],
        "permissions": ["documents:read"],
        "icon_url": "/icons/widgets/code-inspector.svg",
        "readme": (
            "# Code Inspector\n\n"
            "Browse and review code with full syntax highlighting.\n\n"
            "## Features\n"
            "- 100+ language grammars via Tree-sitter\n"
            "- Line numbers and word-wrap toggle\n"
            "- Inline search and go-to-line\n"
            "- Unified and side-by-side diff views\n\n"
            "## Permissions\n"
            "Requires the **documents:read** permission."
        ),
    },
    {
        "name": "email-manager",
        "display_name": "Email Manager",
        "description": (
            "Read, compose, and reply to emails with Gmail and Outlook integration."
        ),
        "categories": ["communication", "productivity"],
        "permissions": ["chat"],
        "icon_url": "/icons/widgets/email-manager.svg",
        "readme": (
            "# Email Manager\n\n"
            "Manage your inbox without leaving Automatos.\n\n"
            "## Features\n"
            "- Gmail and Outlook OAuth integration\n"
            "- Compose, reply, and forward with rich text\n"
            "- Thread grouping and label/folder management\n"
            "- AI-suggested replies and summaries\n\n"
            "## Permissions\n"
            "Requires the **chat** permission for AI-assisted composition."
        ),
    },
    {
        "name": "terminal",
        "display_name": "Terminal",
        "description": (
            "Command execution terminal with ANSI color support and output history."
        ),
        "categories": ["development", "devops"],
        "permissions": ["agents:execute"],
        "icon_url": "/icons/widgets/terminal.svg",
        "readme": (
            "# Terminal\n\n"
            "Execute commands directly from your workspace.\n\n"
            "## Features\n"
            "- Full ANSI / xterm-256 color rendering\n"
            "- Scrollable output history\n"
            "- Copy-paste and keyboard shortcut support\n"
            "- Configurable shell (bash, zsh, fish)\n\n"
            "## Permissions\n"
            "Requires the **agents:execute** permission to run commands."
        ),
    },
    {
        "name": "workflow-monitor",
        "display_name": "Workflow Monitor",
        "description": (
            "Real-time workflow execution monitoring with step-by-step progress "
            "and controls."
        ),
        "categories": ["automation", "devops"],
        "permissions": ["workflows:read"],
        "icon_url": "/icons/widgets/workflow-monitor.svg",
        "readme": (
            "# Workflow Monitor\n\n"
            "Watch workflows execute in real time and intervene when needed.\n\n"
            "## Features\n"
            "- Live step-by-step progress visualization\n"
            "- Pause, resume, and cancel controls\n"
            "- Execution logs and error details\n"
            "- Historical run comparison\n\n"
            "## Permissions\n"
            "Requires the **workflows:read** permission."
        ),
    },
    {
        "name": "memory-inspector",
        "display_name": "Memory Inspector",
        "description": (
            "Browse, search, and manage AI agent memories with type filtering."
        ),
        "categories": ["ai", "productivity"],
        "permissions": ["chat"],
        "icon_url": "/icons/widgets/memory-inspector.svg",
        "readme": (
            "# Memory Inspector\n\n"
            "Explore and curate the memories your AI agents have stored.\n\n"
            "## Features\n"
            "- Full-text search across all memory entries\n"
            "- Filter by memory type (episodic, semantic, procedural)\n"
            "- Edit or delete individual memories\n"
            "- Bulk export and import\n\n"
            "## Permissions\n"
            "Requires the **chat** permission to access agent memory stores."
        ),
    },
]

# ── Seed logic ──────────────────────────────────────────────────────────────

INSERT_SQL = text("""
    INSERT INTO marketplace_widgets (
        id, name, display_name, description, developer_name,
        version, pricing_type, icon_url, readme,
        categories, permissions,
        install_count, rating_average, rating_count,
        status, published_at, created_at, updated_at
    ) VALUES (
        :id, :name, :display_name, :description, 'Automatos AI',
        '1.0.0', 'free', :icon_url, :readme,
        :categories, :permissions,
        :install_count, :rating_average, :rating_count,
        'published', :now, :now, :now
    )
    ON CONFLICT (name) DO NOTHING
""")


def seed_marketplace_widgets():
    """Insert internal widgets into marketplace_widgets (idempotent)."""
    print("Seeding marketplace_widgets with 8 internal widgets...")

    engine = create_engine(get_database_url())
    now = datetime.now(timezone.utc)

    with engine.connect() as conn:
        trans = conn.begin()
        try:
            inserted = 0
            skipped = 0

            for w in WIDGETS:
                result = conn.execute(INSERT_SQL, {
                    "id": str(uuid4()),
                    "name": w["name"],
                    "display_name": w["display_name"],
                    "description": w["description"],
                    "icon_url": w.get("icon_url", f"/icons/widgets/{w['name']}.svg"),
                    "readme": w.get("readme", ""),
                    "categories": w["categories"],
                    "permissions": w["permissions"],
                    "install_count": randint(50, 500),
                    "rating_average": round(uniform(4.0, 5.0), 2),
                    "rating_count": randint(5, 50),
                    "now": now,
                })

                if result.rowcount:
                    inserted += 1
                    print(f"  + {w['display_name']}")
                else:
                    skipped += 1
                    print(f"  ~ {w['display_name']} (already exists)")

            trans.commit()
            print(
                f"\nDone. Inserted {inserted}, skipped {skipped} "
                f"(total {inserted + skipped})."
            )

        except Exception as exc:
            trans.rollback()
            print(f"ERROR: {exc}")
            raise


if __name__ == "__main__":
    seed_marketplace_widgets()
