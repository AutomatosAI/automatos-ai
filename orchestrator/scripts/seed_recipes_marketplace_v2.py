"""
Seed marketplace with 15 NEW workflow recipes (v2).

Recipes combine multiple agents for real-world workflows.
Each recipe specifies:
  - Trigger type: 'manual' | 'cron' (scheduled) | 'trigger' (event-driven)
  - Which agents work together and in what order
  - Required tools across all participating agents
  - Estimated run time and cost tier

Trigger types map to real use cases:
  - cron/daily     → Morning briefings, overnight processing
  - cron/weekly    → Reports, reviews, reconciliation
  - cron/monthly   → Financial summaries, strategic reviews
  - trigger/jira   → New issue created
  - trigger/github → New PR, push to main, security alert
  - trigger/sentry → Error spike, new issue
  - trigger/email  → Specific email received
  - trigger/hubspot→ New lead, deal stage change
  - manual         → On-demand by user
"""

import sys
import os
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from core.database.database import get_database_url

MARKETPLACE_RECIPES_V2 = [
    # ─── TRIGGERED RECIPES ───────────────────────────────────────
    {
        "name": "Jira Issue Auto-Triage & Assign",
        "description": "Triggered when a new Jira issue is created. The Bug Triage Agent analyzes the issue title, description, and labels to determine severity and component. Cross-references recent GitHub commits and Sentry errors to identify likely root cause. Auto-assigns to the right engineer, sets priority, adds relevant labels, and posts a context-rich summary in Slack with linked commits and error traces. Reduces mean triage time from 45 minutes to under 60 seconds.",
        "category": "DevOps",
        "tags": ["jira", "triage", "automation", "bugs", "incident", "triggered"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["JIRA", "GITHUB", "SENTRY", "SLACK"],
            "required_skills": ["code_analysis", "pattern_recognition"],
            "suggested_agents": ["Bug Triage Agent"],
            "trigger": "jira_issue_created",
            "trigger_type": "trigger",
            "estimated_time": "30 seconds",
            "cost_tier": "premium",
            "steps": [
                "New Jira issue triggers the recipe",
                "Bug Triage Agent analyzes issue content and severity",
                "Cross-references GitHub commits and Sentry errors",
                "Auto-assigns engineer and sets priority",
                "Posts triage summary with context to Slack"
            ]
        }
    },
    {
        "name": "GitHub PR Security & Quality Gate",
        "description": "Triggered on every new pull request. The Security Monitor scans for dependency vulnerabilities, secret leaks, and OWASP patterns. The DevOps Engineer reviews code quality, test coverage, and architectural consistency. Results are posted as PR comments with pass/fail status. Blocks merge on critical findings and notifies the team in Slack. Your automated senior reviewer that never takes PTO.",
        "category": "DevOps",
        "tags": ["github", "security", "code review", "pr", "quality gate", "triggered"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["GITHUB", "SLACK", "SENTRY"],
            "required_skills": ["code_analysis", "pattern_recognition"],
            "suggested_agents": ["Security Monitor", "DevOps Engineer"],
            "trigger": "github_pr_opened",
            "trigger_type": "trigger",
            "estimated_time": "2-5 minutes per PR",
            "cost_tier": "premium",
            "steps": [
                "New PR triggers the recipe",
                "Security Monitor scans for vulnerabilities and secrets",
                "DevOps Engineer reviews code quality and patterns",
                "Results posted as PR review comments",
                "Critical findings block merge and alert Slack"
            ]
        }
    },
    {
        "name": "New Lead Qualification & Meeting Booking",
        "description": "Triggered when a new lead enters HubSpot (form fill, import, or API). The Sales Prospector enriches the lead with LinkedIn data and scores it against your ICP. If qualified, the Lead Nurture Agent drafts a personalized intro email and includes a Calendly link for booking. If unqualified, the lead is tagged and moved to a nurture sequence. Ensures no lead waits more than 5 minutes for first contact.",
        "category": "Sales",
        "tags": ["leads", "qualification", "sales", "hubspot", "calendly", "triggered"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["HUBSPOT", "LINKEDIN", "GMAIL", "CALENDLY"],
            "required_skills": ["pattern_recognition", "Data Analysis"],
            "suggested_agents": ["Sales Prospector", "Lead Nurture Agent"],
            "trigger": "hubspot_new_lead",
            "trigger_type": "trigger",
            "estimated_time": "2 minutes per lead",
            "cost_tier": "mid",
            "steps": [
                "New HubSpot lead triggers the recipe",
                "Sales Prospector enriches lead with LinkedIn data",
                "Lead scored against ICP criteria",
                "Qualified: personalized email + Calendly link sent",
                "Unqualified: tagged and added to nurture sequence"
            ]
        }
    },
    {
        "name": "Sentry Incident Response Pipeline",
        "description": "Triggered on a new Sentry error spike or critical issue. The Security Monitor assesses severity and blast radius. The Bug Triage Agent identifies the likely root cause from recent deployments. If critical, creates a Jira incident ticket, pages the on-call engineer via Slack, and starts an incident channel. Cuts incident response time from discovery to action by 80%.",
        "category": "DevOps",
        "tags": ["incident response", "sentry", "monitoring", "alerts", "on-call", "triggered"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["SENTRY", "JIRA", "GITHUB", "SLACK"],
            "required_skills": ["code_analysis", "pattern_recognition"],
            "suggested_agents": ["Security Monitor", "Bug Triage Agent"],
            "trigger": "sentry_issue_critical",
            "trigger_type": "trigger",
            "estimated_time": "1 minute",
            "cost_tier": "premium",
            "steps": [
                "Sentry critical alert triggers the recipe",
                "Security Monitor assesses severity and blast radius",
                "Bug Triage Agent identifies root cause from recent deploys",
                "Jira incident ticket created automatically",
                "Slack incident channel opened and on-call pinged"
            ]
        }
    },
    {
        "name": "New Hire Full Onboarding Workflow",
        "description": "Triggered when a new employee is added in BambooHR. The HR Assistant creates the onboarding checklist and tracks completion. The Meeting Coordinator schedules intro meetings with team leads and the buddy system. The Document Workflow Manager provisions access to Google Drive folders and shares the employee handbook via Notion. Welcome message posted in Slack. Turns a 3-day admin process into a 30-minute automated flow.",
        "category": "HR",
        "tags": ["onboarding", "hr", "new hire", "automation", "bamboohr", "triggered"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["BAMBOOHR", "SLACK", "GOOGLECALENDAR", "GOOGLEDRIVE", "NOTION", "GMAIL"],
            "required_skills": ["task_decomposition"],
            "suggested_agents": ["HR Assistant", "Meeting Coordinator", "Document Workflow Manager"],
            "trigger": "bamboohr_new_hire",
            "trigger_type": "trigger",
            "estimated_time": "15 minutes",
            "cost_tier": "budget",
            "steps": [
                "New hire record in BambooHR triggers the recipe",
                "HR Assistant creates onboarding checklist",
                "Meeting Coordinator schedules intro calls",
                "Document Workflow Manager provisions folders and docs",
                "Welcome message and schedule sent via Slack and Gmail"
            ]
        }
    },

    # ─── SCHEDULED RECIPES ───────────────────────────────────────
    {
        "name": "Morning Business Briefing",
        "description": "Runs daily at 7:00 AM. Aggregates overnight emails (Personal Assistant), CRM pipeline changes (CRM Pipeline Manager), and key metrics (Data Analyst) into a single executive briefing. Delivered as a formatted Slack message and optional email digest. Covers: unread priority emails, deals that moved stages, calendar for the day, and KPI snapshot. Start every day informed in 30 seconds.",
        "category": "Productivity",
        "tags": ["morning briefing", "daily", "executive summary", "scheduled", "digest"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["GMAIL", "SALESFORCE", "GOOGLESHEETS", "GOOGLECALENDAR", "SLACK"],
            "required_skills": ["Data Analysis", "pattern_recognition"],
            "suggested_agents": ["Personal Assistant", "CRM Pipeline Manager", "Data Analyst"],
            "schedule": "0 7 * * 1-5",
            "trigger_type": "cron",
            "estimated_time": "3 minutes",
            "cost_tier": "mid",
            "steps": [
                "Personal Assistant scans overnight emails and calendar",
                "CRM Pipeline Manager pulls deal stage changes",
                "Data Analyst generates KPI snapshot from Sheets",
                "All summaries compiled into briefing format",
                "Delivered via Slack and optional email at 7 AM"
            ]
        }
    },
    {
        "name": "Weekly Social Media Performance Report",
        "description": "Runs every Monday at 9:00 AM. The Social Media Manager pulls engagement metrics from Twitter and LinkedIn for the past week. The Data Analyst crunches the numbers — impressions, clicks, follower growth, top posts — and generates a formatted report in Google Sheets. Summary posted in Slack with highlights and recommendations for the coming week.",
        "category": "Social Media",
        "tags": ["social media", "analytics", "weekly report", "scheduled", "performance"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["TWITTER", "LINKEDIN", "GOOGLESHEETS", "SLACK"],
            "required_skills": ["Data Analysis", "pattern_recognition"],
            "suggested_agents": ["Social Media Manager", "Data Analyst"],
            "schedule": "0 9 * * 1",
            "trigger_type": "cron",
            "estimated_time": "10 minutes",
            "cost_tier": "mid",
            "steps": [
                "Social Media Manager exports weekly metrics",
                "Data Analyst processes engagement data",
                "Report generated in Google Sheets with charts",
                "Top posts and recommendations identified",
                "Summary delivered to Slack on Monday morning"
            ]
        }
    },
    {
        "name": "Invoice Chaser & AR Aging Report",
        "description": "Runs every Wednesday and Friday at 10:00 AM. The Invoice & Billing Manager checks QuickBooks for overdue invoices. The Bookkeeper cross-references with Stripe payment records to confirm status. Overdue accounts receive polite, escalating reminder emails via Gmail. A fresh AR aging report is generated in Google Sheets. Cash flow issues get flagged in Slack before they become crises.",
        "category": "Finance",
        "tags": ["invoices", "accounts receivable", "collections", "scheduled", "finance"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["QUICKBOOKS", "STRIPE", "GMAIL", "GOOGLESHEETS", "SLACK"],
            "required_skills": ["Data Analysis", "task_decomposition"],
            "suggested_agents": ["Invoice & Billing Manager", "Bookkeeper"],
            "schedule": "0 10 * * 3,5",
            "trigger_type": "cron",
            "estimated_time": "15 minutes",
            "cost_tier": "budget",
            "steps": [
                "Invoice Manager pulls overdue invoices from QuickBooks",
                "Bookkeeper cross-references with Stripe payments",
                "Escalating reminder emails sent to overdue accounts",
                "AR aging report updated in Google Sheets",
                "Cash flow alerts posted to Slack if thresholds breached"
            ]
        }
    },
    {
        "name": "Weekly Expense Reconciliation",
        "description": "Runs every Friday at 4:00 PM. The Expense Tracker scans the week's Gmail receipts and categorizes expenses. The Bookkeeper reconciles against QuickBooks transactions, flags mismatches, and updates the weekly expense sheet. Team spend summary posted to Slack. Perfect for keeping finances clean without the Friday afternoon grind.",
        "category": "Finance",
        "tags": ["expenses", "reconciliation", "weekly", "scheduled", "bookkeeping"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["GMAIL", "QUICKBOOKS", "GOOGLESHEETS", "SLACK"],
            "required_skills": ["Data Analysis", "pattern_recognition"],
            "suggested_agents": ["Expense Tracker", "Bookkeeper"],
            "schedule": "0 16 * * 5",
            "trigger_type": "cron",
            "estimated_time": "10 minutes",
            "cost_tier": "budget",
            "steps": [
                "Expense Tracker scans Gmail for week's receipts",
                "Expenses categorized and logged to Google Sheets",
                "Bookkeeper reconciles against QuickBooks",
                "Mismatches flagged for manual review",
                "Weekly spend summary posted to Slack"
            ]
        }
    },
    {
        "name": "Daily Competitor Pulse Monitor",
        "description": "Runs daily at 8:00 AM. The Competitive Intelligence Agent scans Twitter and LinkedIn for competitor mentions, product launches, and pricing changes. The Research Analyst digs deeper on significant findings using web search. Key insights logged in Notion with trend analysis. Slack alert for any major competitive moves. Stay ahead without manually checking competitor feeds.",
        "category": "Marketing",
        "tags": ["competitive intelligence", "monitoring", "daily", "scheduled", "strategy"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["TWITTER", "LINKEDIN", "NOTION", "SLACK", "TAVILY"],
            "required_skills": ["Data Analysis", "pattern_recognition"],
            "suggested_agents": ["Competitive Intelligence Agent", "Research Analyst"],
            "schedule": "0 8 * * 1-5",
            "trigger_type": "cron",
            "estimated_time": "10 minutes",
            "cost_tier": "premium",
            "steps": [
                "Competitive Intel Agent scans social for competitor mentions",
                "Research Analyst deep-dives on significant findings",
                "Insights categorized and logged in Notion",
                "Trend analysis updated with historical data",
                "Slack alert for major competitive moves"
            ]
        }
    },
    {
        "name": "Monthly Revenue & Financial Summary",
        "description": "Runs on the 1st of every month at 8:00 AM. The Bookkeeper closes the prior month in QuickBooks and reconciles with Stripe. The Data Analyst generates revenue, expense, and margin analysis in Google Sheets. The E-commerce Manager (if applicable) adds sales channel breakdown. Final report shared via Gmail to stakeholders and summarized in Slack. Your month-end close on autopilot.",
        "category": "Finance",
        "tags": ["monthly report", "revenue", "financial summary", "scheduled", "month-end"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["QUICKBOOKS", "STRIPE", "GOOGLESHEETS", "GMAIL", "SLACK"],
            "required_skills": ["Data Analysis", "pattern_recognition"],
            "suggested_agents": ["Bookkeeper", "Data Analyst", "E-commerce Manager"],
            "schedule": "0 8 1 * *",
            "trigger_type": "cron",
            "estimated_time": "30 minutes",
            "cost_tier": "mid",
            "steps": [
                "Bookkeeper reconciles QuickBooks and Stripe for month",
                "Data Analyst generates revenue and margin analysis",
                "E-commerce Manager adds sales channel breakdown",
                "Report compiled in Google Sheets with charts",
                "Summary emailed to stakeholders and posted to Slack"
            ]
        }
    },

    # ─── MANUAL RECIPES ──────────────────────────────────────────
    {
        "name": "Content Calendar & Auto-Publish Pipeline",
        "description": "Run manually when planning your content week. The Content Creator brainstorms topics and drafts posts in Notion. The SEO Strategist optimizes headlines and suggests keywords. The Social Media Manager formats for each platform and schedules publication across Twitter and LinkedIn at optimal engagement times. A full week of content planned and queued in one sitting.",
        "category": "Content Creation",
        "tags": ["content", "publishing", "social media", "seo", "manual", "planning"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["NOTION", "GOOGLESHEETS", "TWITTER", "LINKEDIN", "GOOGLEDOCS"],
            "required_skills": ["Data Analysis", "pattern_recognition"],
            "suggested_agents": ["Content Creator", "SEO Strategist", "Social Media Manager"],
            "trigger_type": "manual",
            "estimated_time": "45 minutes for a week of content",
            "cost_tier": "premium",
            "steps": [
                "Content Creator brainstorms topics and drafts in Notion",
                "SEO Strategist optimizes for search and suggests keywords",
                "Social Media Manager formats for each platform",
                "Posts scheduled at optimal engagement times",
                "Content calendar updated in Google Sheets"
            ]
        }
    },
    {
        "name": "Customer Churn Risk Assessment",
        "description": "Run manually or scheduled weekly. The Client Success Manager pulls customer health metrics from HubSpot — usage trends, support ticket volume, NPS scores. The Data Analyst builds a churn risk model scoring each account. At-risk accounts get flagged in Slack with recommended save actions. Personalized re-engagement emails queued for CSM review. Catch churn signals before customers leave.",
        "category": "Customer Support",
        "tags": ["churn", "retention", "customer success", "risk", "manual", "health score"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["HUBSPOT", "INTERCOM", "GOOGLESHEETS", "SLACK", "GMAIL"],
            "required_skills": ["Data Analysis", "pattern_recognition", "Machine Learning"],
            "suggested_agents": ["Client Success Manager", "Data Analyst"],
            "trigger_type": "manual",
            "estimated_time": "20 minutes",
            "cost_tier": "premium",
            "steps": [
                "Client Success Manager pulls health metrics from HubSpot",
                "Data Analyst scores accounts on churn risk factors",
                "At-risk accounts flagged with recommended actions",
                "Re-engagement emails drafted for CSM review",
                "Churn dashboard updated in Google Sheets"
            ]
        }
    },
    {
        "name": "Recruitment Pipeline Sprint",
        "description": "Run manually when you need to fill a role fast. The Recruitment Sourcer searches LinkedIn for candidates matching your job criteria and drafts personalized outreach. The HR Assistant tracks candidates through the pipeline, schedules interviews via Gmail, and manages offer workflows. Pipeline status posted to Slack daily until the role is filled. Compress a 4-week hiring process into days.",
        "category": "HR",
        "tags": ["recruiting", "hiring", "pipeline", "sourcing", "manual", "talent"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["LINKEDIN", "GMAIL", "SLACK", "BAMBOOHR"],
            "required_skills": ["pattern_recognition", "task_decomposition"],
            "suggested_agents": ["Recruitment Sourcer", "HR Assistant"],
            "trigger_type": "manual",
            "estimated_time": "30 minutes initial, then daily updates",
            "cost_tier": "mid",
            "steps": [
                "Recruitment Sourcer searches LinkedIn for matching candidates",
                "Personalized outreach drafted and sent via Gmail",
                "HR Assistant tracks responses and manages pipeline",
                "Interviews scheduled and prep docs generated",
                "Daily pipeline status updates posted to Slack"
            ]
        }
    },
    {
        "name": "Inbox Zero Email Processor",
        "description": "Run manually when your inbox is overwhelming, or schedule 3x daily. The Personal Assistant categorizes every unread email by priority and action type (reply needed, FYI, delegate, archive). The Communication Hub routes action items to the right Slack channels and creates Google Tasks for follow-ups. Drafts quick replies for routine emails. Takes a 200-email inbox to zero in minutes.",
        "category": "Productivity",
        "tags": ["email", "inbox zero", "productivity", "triage", "manual", "gmail"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["GMAIL", "SLACK", "GOOGLETASKS"],
            "required_skills": ["pattern_recognition", "task_decomposition"],
            "suggested_agents": ["Personal Assistant", "Communication Hub"],
            "trigger_type": "manual",
            "schedule_hint": "Can also be scheduled: 0 9,13,17 * * 1-5",
            "estimated_time": "5 minutes per 100 emails",
            "cost_tier": "mid",
            "steps": [
                "Personal Assistant scans and categorizes all unread emails",
                "Emails sorted: reply needed, FYI, delegate, archive",
                "Communication Hub routes action items to Slack channels",
                "Google Tasks created for follow-up items",
                "Draft replies generated for routine emails"
            ]
        }
    },
]


def seed_recipes_v2(force: bool = False):
    """Seed marketplace with 15 new workflow recipes (v2 expansion).

    By default only deletes Automatos-owned v2 recipes.
    Pass force=True (or set SEED_FORCE=true env var) to delete ALL marketplace recipes.
    """
    import os
    force = force or os.environ.get("SEED_FORCE", "").lower() in ("true", "1", "yes")

    print("🌱 Seeding marketplace with v2 recipes (15 new workflows)...")

    engine = create_engine(get_database_url())

    with engine.connect() as db:
        trans = db.begin()

        try:
            recipe_names = tuple(r["name"] for r in MARKETPLACE_RECIPES_V2)

            if force:
                db.execute(text("DELETE FROM marketplace_items WHERE type = 'recipe'"))
                print("✓ Cleared ALL existing recipes (force mode)\n")
            else:
                # Only clear v2 recipes by name match
                for name in recipe_names:
                    db.execute(text(
                        "DELETE FROM marketplace_items WHERE type = 'recipe' "
                        "AND creator_name = 'Automatos Team' AND name = :name"
                    ), {"name": name})
                print("✓ Cleared existing v2 recipes (if any)\n")

            created = 0
            for recipe in MARKETPLACE_RECIPES_V2:
                print(f"📦 Creating recipe: {recipe['name']}...")

                metadata_json = json.dumps(recipe.get("metadata", {}))
                tags_json = json.dumps(recipe.get("tags", []))

                result = db.execute(
                    text("""
                        INSERT INTO marketplace_items (
                            type, name, description, creator_name, category, tags,
                            install_count, is_featured, is_approved, version, metadata,
                            created_at, updated_at
                        ) VALUES (
                            'recipe', :name, :description, 'Automatos Team', :category, :tags,
                            0, true, true, '1.0.0', :metadata,
                            NOW(), NOW()
                        )
                        RETURNING id
                    """),
                    {
                        "name": recipe["name"],
                        "description": recipe["description"],
                        "category": recipe["category"],
                        "tags": tags_json,
                        "metadata": metadata_json,
                    }
                )

                new_id = result.scalar()
                created += 1
                trigger = recipe["metadata"].get("trigger_type", "manual")
                agents = ", ".join(recipe["metadata"].get("suggested_agents", []))
                print(f"   ✅ ID: {new_id} | Trigger: {trigger} | Agents: {agents}\n")

            trans.commit()
            print(f"✨ Successfully seeded {created} new marketplace recipes (v2)!")

        except Exception as e:
            trans.rollback()
            print(f"❌ Error seeding v2 recipes: {e}")
            raise


if __name__ == "__main__":
    seed_recipes_v2()
