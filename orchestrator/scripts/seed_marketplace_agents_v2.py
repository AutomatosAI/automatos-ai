"""
Seed marketplace with 18 NEW professional agents (v2).

Expands the marketplace with real-world use cases across sales, finance,
recruiting, security, marketing, and operations. Tools map to real Composio
integrations. LLMs selected by cost-vs-capability fit.

Cost tiers used:
  - Budget:  llama-3.1-8b-instruct           (~$0.05/M tokens) — routine tasks
  - Mid:     llama-3.3-70b-instruct           (~$0.30/M tokens) — general reasoning
  - Mid+:    meta-llama/Meta-Llama-3.1-70B    (~$0.40/M tokens) — longer context
  - Premium: anthropic/claude-3.5-sonnet      (~$3.00/M tokens) — nuance/code/empathy
  - Reason:  deepseek-ai/DeepSeek-R1          (~$0.55/M tokens) — deep analysis
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from core.database.database import get_database_url

MARKETPLACE_AGENTS_V2 = [
    # ─── SALES & CRM ────────────────────────────────────────────
    {
        "name": "Sales Prospector",
        "agent_type": "custom",
        "description": "Finds, qualifies, and scores inbound and outbound leads automatically. Monitors HubSpot for new contacts, enriches profiles via LinkedIn data, and drafts personalized outreach emails in Gmail. Prioritizes leads by fit score and engagement signals so your sales team focuses on the hottest opportunities. Uses Llama 3.3 70B for smart lead scoring without burning through your LLM budget.",
        "category": "Sales",
        "tags": ["sales", "leads", "prospecting", "hubspot", "linkedin", "outreach", "crm"],
        "tools": ["HUBSPOT", "LINKEDIN", "GMAIL"],
        "model_id": "llama-3.3-70b-instruct",
        "skills": ["pattern_recognition", "Data Analysis"],
    },
    {
        "name": "Lead Nurture Agent",
        "agent_type": "custom",
        "description": "Keeps leads warm through automated, personalized follow-up sequences. Monitors HubSpot deal stages, sends timely emails via Gmail, and books meetings through Calendly when leads show buying intent. Adjusts messaging cadence based on engagement metrics. Perfect for sales teams that lose deals due to slow follow-up. Powered by Llama 3.1 70B for natural, non-robotic email copy.",
        "category": "Sales",
        "tags": ["sales", "nurture", "follow-up", "hubspot", "calendly", "email", "pipeline"],
        "tools": ["HUBSPOT", "GMAIL", "CALENDLY"],
        "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "skills": ["pattern_recognition", "task_decomposition"],
    },
    {
        "name": "CRM Pipeline Manager",
        "agent_type": "custom",
        "description": "Keeps your Salesforce pipeline clean and actionable. Identifies stale deals, flags missing fields, updates deal stages based on activity, and pushes weekly pipeline summaries to Slack and Google Sheets. Ensures your forecast is always accurate. Uses Llama 3.3 70B for reliable CRM data analysis at scale.",
        "category": "Sales",
        "tags": ["crm", "salesforce", "pipeline", "deals", "forecasting", "reporting"],
        "tools": ["SALESFORCE", "SLACK", "GOOGLESHEETS"],
        "model_id": "llama-3.3-70b-instruct",
        "skills": ["Data Analysis", "pattern_recognition"],
    },

    # ─── FINANCE & ACCOUNTING ────────────────────────────────────
    {
        "name": "Invoice & Billing Manager",
        "agent_type": "custom",
        "description": "Automates invoice creation, payment tracking, and overdue reminders. Syncs QuickBooks invoices with Stripe payment data, sends polite payment reminders via Gmail, and flags overdue accounts. Generates weekly AR aging reports. Ideal for small businesses without a dedicated accounts receivable team. Uses Llama 3.1 8B — invoicing is structured and repetitive, no need for expensive models.",
        "category": "Finance",
        "tags": ["invoicing", "billing", "quickbooks", "stripe", "payments", "accounts receivable"],
        "tools": ["QUICKBOOKS", "STRIPE", "GMAIL"],
        "model_id": "llama-3.1-8b-instruct",
        "skills": ["task_decomposition"],
    },
    {
        "name": "Bookkeeper",
        "agent_type": "custom",
        "description": "Categorizes transactions, reconciles accounts, and prepares financial summaries. Pulls data from QuickBooks and Stripe, cross-references with bank statements in Google Sheets, and flags anomalies via Slack. Generates monthly P&L snapshots and cash flow summaries. Essential for founders who want financial visibility without hiring a full-time bookkeeper. Powered by Llama 3.3 70B for accurate number crunching.",
        "category": "Finance",
        "tags": ["bookkeeping", "accounting", "quickbooks", "reconciliation", "finance", "cash flow"],
        "tools": ["QUICKBOOKS", "GOOGLESHEETS", "GMAIL", "SLACK"],
        "model_id": "llama-3.3-70b-instruct",
        "skills": ["Data Analysis", "pattern_recognition"],
    },
    {
        "name": "Expense Tracker",
        "agent_type": "custom",
        "description": "Scans Gmail for receipts and invoices, extracts amounts and categories, and logs them into Google Sheets automatically. Sends Slack alerts for unusual spending and generates weekly expense summaries. Perfect for freelancers, contractors, and small teams who hate manual expense entry. Uses Llama 3.1 8B — receipt parsing is structured extraction, keeping costs minimal.",
        "category": "Finance",
        "tags": ["expenses", "receipts", "tracking", "budget", "spend management"],
        "tools": ["GMAIL", "GOOGLESHEETS", "SLACK"],
        "model_id": "llama-3.1-8b-instruct",
        "skills": ["pattern_recognition"],
    },

    # ─── MARKETING ───────────────────────────────────────────────
    {
        "name": "Email Marketing Specialist",
        "agent_type": "custom",
        "description": "Manages email campaigns end-to-end. Creates audience segments in Mailchimp, drafts campaign copy, A/B tests subject lines, and tracks open/click rates in Google Sheets. Stores content drafts in Notion for team review. Identifies the best send times based on historical engagement. Uses Llama 3.1 70B for creative copy that converts while keeping costs reasonable.",
        "category": "Marketing",
        "tags": ["email marketing", "mailchimp", "campaigns", "newsletters", "automation"],
        "tools": ["MAILCHIMP", "GOOGLESHEETS", "NOTION"],
        "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "skills": ["Data Analysis", "pattern_recognition"],
    },
    {
        "name": "SEO Strategist",
        "agent_type": "custom",
        "description": "Drives organic traffic through data-driven SEO. Analyzes keyword opportunities, audits on-page SEO, tracks ranking changes, and generates content briefs optimized for search. Stores strategies in Notion and tracks metrics in Google Sheets. Produces weekly SEO performance reports. Uses Claude 3.5 Sonnet for the creative + analytical depth that SEO content planning demands.",
        "category": "Marketing",
        "tags": ["seo", "search", "keywords", "organic traffic", "content strategy", "rankings"],
        "tools": ["NOTION", "GOOGLESHEETS", "GOOGLEDOCS"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": ["Data Analysis", "pattern_recognition"],
    },
    {
        "name": "Competitive Intelligence Agent",
        "agent_type": "custom",
        "description": "Monitors competitors across social media, news, and product updates. Tracks Twitter and LinkedIn for competitor mentions, stores findings in Notion, and alerts the team via Slack on significant moves (funding, launches, pricing changes). Generates monthly competitive landscape reports. Uses Claude 3.5 Sonnet for nuanced analysis of competitive positioning.",
        "category": "Marketing",
        "tags": ["competitive analysis", "market intelligence", "monitoring", "research", "strategy"],
        "tools": ["TWITTER", "LINKEDIN", "NOTION", "SLACK"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": ["Data Analysis", "pattern_recognition"],
    },

    # ─── RECRUITING & HR ─────────────────────────────────────────
    {
        "name": "Recruitment Sourcer",
        "agent_type": "custom",
        "description": "Automates top-of-funnel recruiting. Searches LinkedIn for candidates matching job criteria, logs them in Ashby (ATS), drafts personalized outreach emails, and coordinates interview scheduling. Tracks pipeline metrics and flags bottlenecks via Slack. Saves recruiters 10+ hours per week on sourcing. Uses Llama 3.3 70B for personalized, non-templated outreach messages.",
        "category": "HR",
        "tags": ["recruiting", "sourcing", "hiring", "linkedin", "ats", "talent acquisition"],
        "tools": ["LINKEDIN", "GMAIL", "SLACK"],
        "model_id": "llama-3.3-70b-instruct",
        "skills": ["pattern_recognition", "task_decomposition"],
    },

    # ─── ENGINEERING & SECURITY ──────────────────────────────────
    {
        "name": "Bug Triage Agent",
        "agent_type": "custom",
        "description": "Automatically triages incoming bugs and issues. When a new Jira issue or Sentry alert arrives, analyzes severity, checks related GitHub commits for likely root causes, assigns to the right team member, and posts context-rich summaries in Slack. Reduces triage time from hours to seconds. Uses Claude 3.5 Sonnet because accurate code-level root cause analysis requires premium reasoning.",
        "category": "DevOps",
        "tags": ["bugs", "triage", "jira", "sentry", "incident management", "debugging"],
        "tools": ["JIRA", "GITHUB", "SLACK", "SENTRY"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": ["code_analysis", "pattern_recognition"],
    },
    {
        "name": "Security Monitor",
        "agent_type": "custom",
        "description": "Continuous security posture monitoring for your codebase and infrastructure. Scans GitHub repos for dependency vulnerabilities, monitors Sentry for suspicious error patterns, creates Jira tickets for security findings, and alerts the team via Slack. Generates weekly security posture reports. Uses Claude 3.5 Sonnet — security analysis demands the highest accuracy available.",
        "category": "DevOps",
        "tags": ["security", "vulnerabilities", "monitoring", "devsecops", "compliance", "alerts"],
        "tools": ["GITHUB", "SENTRY", "JIRA", "SLACK"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": ["code_analysis", "pattern_recognition"],
    },

    # ─── OPERATIONS & PRODUCTIVITY ───────────────────────────────
    {
        "name": "Meeting Coordinator",
        "agent_type": "custom",
        "description": "Eliminates meeting scheduling overhead. Manages Google Calendar availability, sends scheduling links via Gmail, prepares agendas in Google Docs from Slack discussion threads, and distributes meeting notes post-call. Detects scheduling conflicts and suggests optimal times. Uses Llama 3.1 8B — scheduling logic is straightforward, keeping per-meeting costs near zero.",
        "category": "Productivity",
        "tags": ["meetings", "scheduling", "calendar", "agendas", "notes", "coordination"],
        "tools": ["GOOGLECALENDAR", "SLACK", "GMAIL", "GOOGLEDOCS"],
        "model_id": "llama-3.1-8b-instruct",
        "skills": ["task_decomposition"],
    },
    {
        "name": "Document Workflow Manager",
        "agent_type": "custom",
        "description": "Routes, organizes, and manages documents across your organization. Monitors Google Drive for new uploads, categorizes files, moves them to correct folders, notifies relevant team members via Slack, and maintains a document index in Notion. Handles approval workflows and version tracking. Uses Llama 3.1 8B for cost-efficient document classification and routing.",
        "category": "Productivity",
        "tags": ["documents", "files", "google drive", "organization", "workflow", "approvals"],
        "tools": ["GOOGLEDRIVE", "GOOGLEDOCS", "SLACK", "NOTION"],
        "model_id": "llama-3.1-8b-instruct",
        "skills": ["task_decomposition", "pattern_recognition"],
    },
    {
        "name": "Startup Ops Assistant",
        "agent_type": "custom",
        "description": "The Swiss Army knife for early-stage startups. Manages Notion wikis, tracks OKRs in Google Sheets, monitors GitHub for engineering velocity, and keeps the team aligned via Slack standups. Generates weekly all-hands summaries and flags blockers before they become crises. Built for lean teams wearing multiple hats. Uses Llama 3.3 70B for versatile reasoning across diverse operational tasks.",
        "category": "Operations",
        "tags": ["startup", "operations", "okrs", "standup", "velocity", "all-hands"],
        "tools": ["SLACK", "NOTION", "GOOGLESHEETS", "GITHUB"],
        "model_id": "llama-3.3-70b-instruct",
        "skills": ["Data Analysis", "task_decomposition"],
    },

    # ─── RESEARCH & ANALYSIS ─────────────────────────────────────
    {
        "name": "Research Analyst",
        "agent_type": "custom",
        "description": "Deep-dives into topics with structured research methodology. Uses web search (Tavily) to gather sources, synthesizes findings in Google Docs, tracks research threads in Notion, and logs data in Google Sheets. Produces executive summaries with citations and confidence scores. Ideal for market research, due diligence, and strategic planning. Powered by DeepSeek R1 for chain-of-thought reasoning on complex research questions.",
        "category": "Research",
        "tags": ["research", "analysis", "market research", "due diligence", "reports", "insights"],
        "tools": ["TAVILY", "GOOGLEDOCS", "NOTION", "GOOGLESHEETS"],
        "model_id": "deepseek-ai/DeepSeek-R1",
        "skills": ["Data Analysis", "pattern_recognition"],
    },

    # ─── CLIENT SUCCESS ──────────────────────────────────────────
    {
        "name": "Client Success Manager",
        "agent_type": "custom",
        "description": "Proactively manages customer health and retention. Monitors HubSpot for usage drops and renewal dates, checks Intercom for support ticket trends, sends personalized check-in emails via Gmail, and escalates at-risk accounts via Slack. Generates customer health scorecards and QBR prep docs. Uses Claude 3.5 Sonnet because empathetic, context-aware customer communication requires premium language understanding.",
        "category": "Customer Support",
        "tags": ["customer success", "retention", "churn prevention", "health scores", "renewals"],
        "tools": ["HUBSPOT", "INTERCOM", "GMAIL", "SLACK"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": ["Data Analysis", "pattern_recognition"],
    },

    # ─── SCHEDULING ──────────────────────────────────────────────
    {
        "name": "Calendar Optimizer",
        "agent_type": "custom",
        "description": "Analyzes your calendar patterns and optimizes for deep work, meetings, and breaks. Syncs Google Calendar with Calendly availability, identifies scheduling conflicts, blocks focus time, and sends daily schedule summaries via Gmail. Suggests meeting consolidation to reduce context switching. Uses Llama 3.1 8B — calendar optimization is rule-based, keeping costs negligible.",
        "category": "Productivity",
        "tags": ["calendar", "scheduling", "time management", "focus time", "optimization"],
        "tools": ["GOOGLECALENDAR", "CALENDLY", "GMAIL", "SLACK"],
        "model_id": "llama-3.1-8b-instruct",
        "skills": ["Data Analysis", "task_decomposition"],
    },
]


def seed_marketplace_agents_v2():
    """Seed the marketplace with 18 new professional agents (v2 expansion)."""
    print("🌱 Seeding marketplace with v2 agents (18 new agent types)...")

    database_url = get_database_url()
    engine = create_engine(database_url)

    with engine.connect() as db:
        trans = db.begin()

        try:
            # Clear only v2 agents (identified by creator_name)
            db.execute(text(
                "DELETE FROM marketplace_items WHERE type = 'agent' "
                "AND creator_name = 'Automatos Team' "
                "AND name IN :names"
            ).bindparams(names=tuple(a["name"] for a in MARKETPLACE_AGENTS_V2)))
            print("✓ Cleared existing v2 agents (if any)\n")

            created = 0
            for agent_config in MARKETPLACE_AGENTS_V2:
                print(f"📦 Creating {agent_config['name']}...")

                # Look up tool IDs and icons from composio_apps_cache
                tool_names = agent_config['tools']
                tool_ids = []
                tool_icons = []

                for tool_name in tool_names:
                    tool_query = text("""
                        SELECT id, logo_url FROM composio_apps_cache
                        WHERE UPPER(app_name) = UPPER(:name)
                        LIMIT 1
                    """)
                    tool_result = db.execute(tool_query, {"name": tool_name}).first()
                    if tool_result:
                        tool_ids.append(tool_result[0])
                        tool_icons.append(tool_result[1] or "")
                    else:
                        print(f"   ⚠️  Tool '{tool_name}' not found in composio_apps_cache")

                metadata = {
                    "agent_type": agent_config['agent_type'],
                    "model_id": agent_config['model_id'],
                    "tools": tool_ids,
                    "tool_names": tool_names,
                    "tool_icons": tool_icons,
                    "skills": agent_config['skills'],
                    "priority_level": "medium",
                    "max_concurrent_tasks": 3,
                    "auto_start": False,
                    "tags": agent_config['tags'],
                    "configuration": {}
                }

                insert_query = text("""
                    INSERT INTO marketplace_items (
                        type, name, description, creator_name, category, tags,
                        install_count, is_featured, is_approved, version, metadata,
                        created_at, updated_at
                    ) VALUES (
                        'agent', :name, :description, 'Automatos Team', :category, :tags,
                        0, true, true, '1.0.0', :metadata,
                        NOW(), NOW()
                    )
                    RETURNING id
                """)

                result = db.execute(insert_query, {
                    "name": agent_config['name'],
                    "description": agent_config['description'],
                    "category": agent_config['category'],
                    "tags": json.dumps(agent_config['tags']),
                    "metadata": json.dumps(metadata)
                })

                new_id = result.scalar()
                created += 1
                print(f"   ✅ Created ID: {new_id}")
                print(f"      Tools: {', '.join(tool_names)} ({len(tool_ids)} mapped)")
                print(f"      LLM:   {agent_config['model_id']}")
                print(f"      Cat:   {agent_config['category']}\n")

            trans.commit()
            print(f"✨ Successfully seeded {created} new marketplace agents (v2)!")

        except Exception as e:
            trans.rollback()
            print(f"❌ Error seeding v2 agents: {e}")
            raise


if __name__ == "__main__":
    seed_marketplace_agents_v2()
