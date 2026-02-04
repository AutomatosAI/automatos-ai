"""
Seed marketplace with workflow recipes that reference REAL skills and agents.

Recipes are reusable workflows combining agents, skills, and tools
for common automation tasks.
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from core.database.database import get_database_url

# Workflow recipes that reference REAL skills and agents
RECIPES = [
    {
        "name": "Daily Email Digest & Task Creation",
        "description": "Automated workflow that analyzes your Gmail inbox each morning, categorizes emails by priority, summarizes important messages, and automatically creates tasks in Google Tasks. Combines data analysis skills with email and task management tools. Perfect for busy professionals who receive 50+ emails daily.",
        "category": "Productivity",
        "tags": ["email", "automation", "digest", "tasks", "daily"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["GMAIL", "GOOGLETASKS"],
            "required_skills": ["Data Analysis", "pattern_recognition"],
            "suggested_agent": "Personal Assistant",
            "schedule": "daily",
            "estimated_time": "5 minutes",
            "steps": [
                "Connect Gmail and Google Tasks",
                "Configure email filters and priority keywords",
                "Set daily execution time",
                "Review generated task list"
            ]
        }
    },
    {
        "name": "Code Review Automation",
        "description": "Streamlines code review process by automatically analyzing GitHub pull requests for code quality, security vulnerabilities, and best practices. Uses code analysis skills to check for common issues, suggests improvements, and posts findings as PR comments. Reduces manual review time by 60%. Essential for development teams maintaining code quality.",
        "category": "DevOps",
        "tags": ["code review", "github", "automation", "quality", "development"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["GITHUB", "SLACK"],
            "required_skills": ["code_analysis"],
            "suggested_agent": "DevOps Engineer",
            "trigger": "new_pr",
            "estimated_time": "3 minutes per PR",
            "steps": [
                "Connect GitHub repository",
                "Configure code quality rules",
                "Set up Slack notifications",
                "Review automation reports"
            ]
        }
    },
    {
        "name": "Social Media Content Pipeline",
        "description": "End-to-end content workflow from idea to publication. Brainstorm topics using pattern recognition, draft posts with content skills, schedule across Twitter and LinkedIn at optimal times based on engagement analysis. Tracks performance metrics and suggests content improvements. Ideal for marketers managing multiple social channels.",
        "category": "Social Media",
        "tags": ["social media", "content", "marketing", "automation", "scheduling"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["TWITTER", "LINKEDIN", "NOTION"],
            "required_skills": ["pattern_recognition", "Data Analysis"],
            "suggested_agent": "Social Media Manager",
            "schedule": "custom",
            "estimated_time": "15 minutes per content batch",
            "steps": [
                "Set up content calendar in Notion",
                "Connect social media accounts",
                "Configure posting schedule",
                "Monitor engagement metrics"
            ]
        }
    },
    {
        "name": "Customer Support Ticket Triage",
        "description": "Intelligently routes incoming support tickets using machine learning skills to analyze content, urgency, and customer sentiment. Automatically categorizes tickets, assigns to appropriate team members via Slack, and suggests relevant help articles. Reduces first response time by 70%. Perfect for growing support teams.",
        "category": "Customer Support",
        "tags": ["support", "tickets", "automation", "triage", "zendesk"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["ZENDESK", "SLACK", "INTERCOM"],
            "required_skills": ["Machine Learning", "pattern_recognition"],
            "suggested_agent": "Customer Support Agent",
            "trigger": "new_ticket",
            "estimated_time": "Instant",
            "steps": [
                "Connect support platforms",
                "Configure triage rules",
                "Set up team routing in Slack",
                "Train ML model on historical tickets"
            ]
        }
    },
    {
        "name": "Sprint Planning Assistant",
        "description": "Automates agile sprint planning by analyzing Linear tasks, breaking down epics using task decomposition skills, estimating effort, and suggesting optimal sprint composition. Generates sprint reports and syncs with GitHub milestones. Saves 2+ hours per sprint planning session. Essential for agile teams.",
        "category": "DevOps",
        "tags": ["agile", "sprint", "planning", "tasks", "automation"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["LINEAR", "GITHUB", "SLACK"],
            "required_skills": ["task_decomposition", "Data Analysis"],
            "suggested_agent": "DevOps Engineer",
            "schedule": "biweekly",
            "estimated_time": "30 minutes",
            "steps": [
                "Connect Linear and GitHub",
                "Review backlog items",
                "Run decomposition analysis",
                "Approve sprint composition"
            ]
        }
    },
    {
        "name": "E-commerce Sales Intelligence",
        "description": "Comprehensive sales analytics combining Shopify order data with Stripe payment insights. Uses data analysis and pattern recognition to identify trends, predict inventory needs, and optimize pricing. Generates daily reports on best-selling products and revenue forecasts. Critical for data-driven e-commerce decisions.",
        "category": "E-commerce",
        "tags": ["ecommerce", "analytics", "sales", "insights", "automation"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["SHOPIFY", "STRIPE", "GOOGLESHEETS"],
            "required_skills": ["Data Analysis", "pattern_recognition"],
            "suggested_agent": "E-commerce Manager",
            "schedule": "daily",
            "estimated_time": "10 minutes",
            "steps": [
                "Connect Shopify and Stripe",
                "Configure Google Sheets dashboard",
                "Set alert thresholds",
                "Review daily insights"
            ]
        }
    },
    {
        "name": "Content Research & Outlining",
        "description": "Accelerates content creation by researching topics, analyzing competitor content using pattern recognition, and generating detailed outlines. Stores research in Notion, tracks content performance in Google Sheets. Reduces research time by 80%. Perfect for content teams and bloggers.",
        "category": "Content Creation",
        "tags": ["content", "research", "writing", "automation", "seo"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["NOTION", "GOOGLESHEETS"],
            "required_skills": ["Data Analysis", "pattern_recognition"],
            "suggested_agent": "Content Creator",
            "trigger": "new_topic",
            "estimated_time": "15 minutes",
            "steps": [
                "Define content topics",
                "Configure research parameters",
                "Review generated outlines",
                "Export to Notion"
            ]
        }
    },
    {
        "name": "HR Onboarding Automation",
        "description": "Streamlines employee onboarding from offer acceptance to first day. Coordinates tasks across BambooHR for HR data, Slack for team introductions, and automated welcome sequences. Uses task decomposition to ensure all onboarding steps are completed. Saves 5+ hours per new hire.",
        "category": "HR",
        "tags": ["hr", "onboarding", "automation", "employees", "workflow"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["BAMBOOHR", "SLACK"],
            "required_skills": ["task_decomposition"],
            "suggested_agent": "HR Assistant",
            "trigger": "new_hire",
            "estimated_time": "30 minutes setup",
            "steps": [
                "Connect BambooHR and Slack",
                "Configure onboarding checklist",
                "Set up automated welcome messages",
                "Monitor completion status"
            ]
        }
    },
    {
        "name": "Data Dashboard Automation",
        "description": "Creates and maintains real-time data dashboards by syncing Google Sheets with Airtable, applying data analysis skills to generate insights, and updating visualizations. Automatically refreshes every hour. Perfect for teams needing live KPI monitoring without manual data entry.",
        "category": "Data Analysis",
        "tags": ["data", "dashboards", "automation", "analytics", "kpi"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["GOOGLESHEETS", "AIRTABLE"],
            "required_skills": ["Data Analysis"],
            "suggested_agent": "Data Analyst",
            "schedule": "hourly",
            "estimated_time": "Automatic",
            "steps": [
                "Connect data sources",
                "Define KPI calculations",
                "Configure dashboard layout",
                "Set refresh schedule"
            ]
        }
    },
    {
        "name": "Project Status Reporting",
        "description": "Generates comprehensive project status reports by aggregating data from Asana tasks, GitHub commits, and Slack discussions. Uses pattern recognition to identify risks and blockers. Produces formatted reports in Google Sheets for stakeholders. Saves 3+ hours per week of manual reporting.",
        "category": "Productivity",
        "tags": ["project management", "reporting", "automation", "status"],
        "recipe_type": "workflow",
        "metadata": {
            "required_tools": ["ASANA", "GITHUB", "GOOGLESHEETS", "SLACK"],
            "required_skills": ["Data Analysis", "pattern_recognition"],
            "suggested_agent": "Project Manager",
            "schedule": "weekly",
            "estimated_time": "20 minutes",
            "steps": [
                "Connect all project tools",
                "Define report template",
                "Configure stakeholder distribution",
                "Review generated reports"
            ]
        }
    },
]


def seed_recipes(force: bool = False):
    """Seed marketplace with workflow recipes.

    By default only deletes Automatos-owned seed recipes.
    Pass force=True (or set SEED_FORCE=true env var) to delete ALL marketplace recipes.
    """
    import os
    force = force or os.environ.get("SEED_FORCE", "").lower() in ("true", "1", "yes")

    print("🌱 Seeding workflow recipes...")

    engine = create_engine(get_database_url())

    with engine.connect() as db:
        trans = db.begin()

        try:
            # Only delete Automatos-owned seed recipes unless force flag is set
            if force:
                db.execute(text("DELETE FROM marketplace_items WHERE type = 'recipe'"))
                print("✓ Cleared ALL existing recipes (force mode)\n")
            else:
                db.execute(text("DELETE FROM marketplace_items WHERE type = 'recipe' AND creator_name = 'Automatos Team'"))
                print("✓ Cleared Automatos-owned seed recipes\n")

            for recipe in RECIPES:
                metadata_json = json.dumps(recipe.get("metadata", {}))
                tags_json = json.dumps(recipe.get("tags", []))

                db.execute(
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

            trans.commit()
            print(f"✅ Seeded {len(RECIPES)} workflow recipes with real skill references!")

        except Exception as e:
            trans.rollback()
            print(f"❌ Error seeding recipes: {e}")
            raise


if __name__ == "__main__":
    seed_recipes()
