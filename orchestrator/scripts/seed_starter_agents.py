"""
Seed marketplace with 10 professional starter agents.

Creates pre-configured agents with REAL tools, detailed descriptions,
and proper LLM selections. All agents are marked as featured and
created by "Automatos Team".
"""

import sys
import os
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from core.database.database import get_database_url

# Professional starter agents with REAL tools and detailed descriptions
STARTER_AGENTS = [
    {
        "name": "Personal Assistant",
        "agent_type": "custom",
        "description": "Your intelligent personal assistant that streamlines daily productivity. Manages your Gmail inbox with smart filtering and auto-responses, syncs Google Calendar to coordinate meetings and events, and organizes Google Tasks to keep you on track. Perfect for busy professionals who need help staying organized across multiple channels. Uses Llama 3.3 70B for fast, accurate natural language understanding of your requests.",
        "category": "Personal Assistant",
        "tags": ["productivity", "email", "calendar", "tasks", "scheduling", "organization"],
        "tools": ["GMAIL", "GOOGLECALENDAR", "GOOGLETASKS"],
        "model_id": "llama-3.3-70b-instruct",
        "skills": ["pattern_recognition", "task_decomposition"],
    },
    {
        "name": "Communication Hub",
        "agent_type": "custom",
        "description": "Centralizes all your team communication channels into one intelligent interface. Monitors Slack conversations for important mentions and action items, manages Gmail for external communications, and coordinates across messaging platforms. Ideal for team leads and managers who juggle multiple communication tools. Powered by Meta-Llama 3.1 405B for superior context retention across long conversations.",
        "category": "Personal Assistant",
        "tags": ["communication", "slack", "email", "messaging", "collaboration"],
        "tools": ["SLACK", "GMAIL"],
        "model_id": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "skills": ["pattern_recognition"],
    },
    {
        "name": "Customer Support Agent",
        "agent_type": "custom",
        "description": "Handles customer inquiries across multiple support platforms with speed and empathy. Integrates Zendesk for ticket management, Intercom for live chat support, and Slack for internal team coordination. Automatically categorizes tickets, suggests responses, and escalates complex issues. Best for support teams looking to reduce response times. Uses Claude 3.5 Sonnet for nuanced, empathetic customer interactions.",
        "category": "Customer Support",
        "tags": ["support", "customers", "tickets", "helpdesk", "zendesk", "service"],
        "tools": ["ZENDESK", "INTERCOM", "SLACK"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": ["pattern_recognition", "Machine Learning"],
    },
    {
        "name": "DevOps Engineer",
        "agent_type": "custom",
        "description": "Automates your development workflow from code to deployment. Monitors GitHub repositories for issues and pull requests, manages Linear tasks for sprint planning, and coordinates team updates via Slack. Helps with code reviews, deployment status updates, and incident management. Perfect for engineering teams practicing agile development. Powered by Claude 3.5 Sonnet for technical accuracy and code understanding.",
        "category": "DevOps",
        "tags": ["devops", "github", "development", "deployment", "automation", "ci/cd"],
        "tools": ["GITHUB", "LINEAR", "SLACK"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": ["code_analysis", "task_decomposition"],
    },
    {
        "name": "Social Media Manager",
        "agent_type": "custom",
        "description": "Manages your social media presence across Twitter and LinkedIn. Schedules posts for optimal engagement times, monitors brand mentions, analyzes audience sentiment, and suggests content improvements. Ideal for marketers and brand managers who need consistent social presence. Uses Meta-Llama 3.2 90B Vision for analyzing images and creating engaging visual content strategies.",
        "category": "Social Media",
        "tags": ["social media", "marketing", "twitter", "linkedin", "content", "engagement"],
        "tools": ["TWITTER", "LINKEDIN"],
        "model_id": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "skills": ["Data Analysis", "pattern_recognition"],
    },
    {
        "name": "Project Manager",
        "agent_type": "custom",
        "description": "Streamlines project management across your organization. Syncs Asana tasks with team schedules, maintains Google Sheets for project tracking and reporting, and coordinates updates via Slack. Generates status reports, identifies bottlenecks, and keeps stakeholders informed. Essential for PMs managing multiple concurrent projects. Powered by Llama 3.3 70B for quick task analysis and prioritization.",
        "category": "Productivity",
        "tags": ["project management", "asana", "tasks", "coordination", "reporting"],
        "tools": ["ASANA", "GOOGLESHEETS", "SLACK"],
        "model_id": "llama-3.3-70b-instruct",
        "skills": ["Data Analysis", "task_decomposition"],
    },
    {
        "name": "E-commerce Manager",
        "agent_type": "custom",
        "description": "Manages your online store operations end-to-end. Integrates Shopify for inventory management and order processing, Stripe for payment analytics and customer insights. Monitors sales trends, alerts on low inventory, and helps optimize pricing strategies. Perfect for e-commerce businesses scaling their operations. Uses Meta-Llama 3.1 70B for analyzing sales patterns and customer behavior.",
        "category": "E-commerce",
        "tags": ["ecommerce", "shopify", "stripe", "sales", "inventory", "payments"],
        "tools": ["SHOPIFY", "STRIPE"],
        "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "skills": ["Data Analysis", "pattern_recognition"],
    },
    {
        "name": "Content Creator",
        "agent_type": "custom",
        "description": "Streamlines content creation and publishing workflows. Manages Notion for content planning and drafts, syncs with Google Sheets for content calendars and analytics tracking. Helps brainstorm topics, maintains editorial calendars, and tracks content performance. Ideal for content teams and creators. Powered by Claude 3.5 Sonnet for creative writing assistance and editorial suggestions.",
        "category": "Content Creation",
        "tags": ["content", "writing", "blogging", "publishing", "notion"],
        "tools": ["NOTION", "GOOGLESHEETS"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": ["pattern_recognition"],
    },
    {
        "name": "HR Assistant",
        "agent_type": "custom",
        "description": "Automates HR workflows from onboarding to employee engagement. Integrates BambooHR for employee data management, Slack for team communications and announcements. Handles time-off requests, tracks onboarding progress, and coordinates HR events. Perfect for HR teams managing growing organizations. Uses Llama 3.1 8B for efficient processing of routine HR tasks.",
        "category": "HR",
        "tags": ["hr", "human resources", "onboarding", "employees", "bamboohr", "benefits"],
        "tools": ["BAMBOOHR", "SLACK"],
        "model_id": "llama-3.1-8b-instruct",
        "skills": ["task_decomposition"],
    },
    {
        "name": "Data Analyst",
        "agent_type": "custom",
        "description": "Transforms raw data into actionable insights. Works with Google Sheets for data collection and analysis, Airtable for structured databases and workflow automation. Creates visualizations, identifies trends, and generates automated reports. Essential for data-driven decision making. Powered by DeepSeek R1 for advanced analytical reasoning and pattern recognition in complex datasets.",
        "category": "Data Analysis",
        "tags": ["data", "analytics", "insights", "reports", "visualization", "spreadsheets"],
        "tools": ["GOOGLESHEETS", "AIRTABLE"],
        "model_id": "deepseek-ai/DeepSeek-R1",
        "skills": ["Data Analysis", "pattern_recognition"],
    },
]


def seed_starter_agents():
    """Seed the marketplace with 10 professional starter agents."""
    print("🌱 Seeding marketplace with starter agents...")

    database_url = get_database_url()
    engine = create_engine(database_url)

    with engine.connect() as db:
        trans = db.begin()

        try:
            # First, clear existing agents
            db.execute(text("DELETE FROM marketplace_items WHERE type = 'agent' AND creator_name = 'Automatos Team'"))
            print("✓ Cleared existing Automatos Team agents\n")

            for agent_config in STARTER_AGENTS:
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
                        print(f"   ⚠️  Tool '{tool_name}' not found in database")

                # Build metadata
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

                # Insert marketplace item
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
                print(f"   ✅ Created ID: {new_id}")
                print(f"      - Tools: {', '.join(tool_names)} ({len(tool_ids)} mapped)")
                print(f"      - LLM: {agent_config['model_id']}")
                print(f"      - Category: {agent_config['category']}\n")

            trans.commit()
            print(f"✨ Successfully seeded {len(STARTER_AGENTS)} professional starter agents!")

        except Exception as e:
            trans.rollback()
            print(f"❌ Error seeding agents: {e}")
            raise


if __name__ == "__main__":
    seed_starter_agents()
