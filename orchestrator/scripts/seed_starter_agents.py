"""
Seed marketplace with 10 starter agents for US-008.

Creates pre-configured agents with tools and LLMs for new users to get started quickly.
All agents are marked as featured and created by "Automatos Team".
"""

import sys
import os
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sqlalchemy import create_engine, text
from core.database.database import get_database_url

# Starter agents configuration
STARTER_AGENTS = [
    {
        "name": "Personal Assistant",
        "agent_type": "custom",
        "description": "Your intelligent personal assistant that manages emails, calendar events, and tasks. Stays on top of your schedule and helps you stay organized.",
        "category": "Personal Assistant",
        "tags": ["productivity", "email", "calendar", "tasks", "scheduling"],
        "tools": ["GMAIL", "GOOGLECALENDAR", "GOOGLETASKS"],
        "model_id": "llama-3.3-70b-instruct",
        "skills": [],
    },
    {
        "name": "Communication Manager",
        "agent_type": "custom",
        "description": "Centralizes all your communication channels. Manages Slack messages, emails, and SMS coordination across your entire team.",
        "category": "Personal Assistant",
        "tags": ["communication", "slack", "email", "sms", "messaging"],
        "tools": ["SLACK", "GMAIL", "TWILIO"],
        "model_id": "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        "skills": [],
    },
    {
        "name": "Customer Support Agent",
        "agent_type": "custom",
        "description": "Handles customer inquiries across multiple support platforms. Integrates with Zendesk, Intercom, and Slack to provide seamless customer service.",
        "category": "Customer Support",
        "tags": ["support", "customers", "tickets", "helpdesk", "zendesk"],
        "tools": ["ZENDESK", "INTERCOM", "SLACK"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": [],
    },
    {
        "name": "DevOps Engineer",
        "agent_type": "custom",
        "description": "Automates development workflows and manages deployments. Integrates GitHub, Linear project management, and Slack notifications.",
        "category": "DevOps",
        "tags": ["devops", "github", "ci/cd", "deployment", "automation"],
        "tools": ["GITHUB", "LINEAR", "SLACK"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": [],
    },
    {
        "name": "Social Media Manager",
        "agent_type": "custom",
        "description": "Manages your social media presence across Twitter and LinkedIn. Creates engaging content and schedules posts for maximum reach.",
        "category": "Social Media",
        "tags": ["social media", "twitter", "linkedin", "content", "marketing"],
        "tools": ["TWITTER", "LINKEDIN"],
        "model_id": "meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
        "skills": [],
    },
    {
        "name": "Accounting Assistant",
        "agent_type": "custom",
        "description": "Streamlines financial workflows with QuickBooks integration and spreadsheet automation. Handles invoicing, expenses, and financial reporting.",
        "category": "Accounting",
        "tags": ["accounting", "finance", "quickbooks", "invoicing", "expenses"],
        "tools": ["QUICKBOOKS", "GOOGLESHEETS"],
        "model_id": "Qwen/QwQ-32B-Preview",
        "skills": [],
    },
    {
        "name": "E-commerce Manager",
        "agent_type": "custom",
        "description": "Manages your online store operations. Integrates Shopify for inventory and Stripe for payment processing and analytics.",
        "category": "E-commerce",
        "tags": ["ecommerce", "shopify", "stripe", "sales", "inventory"],
        "tools": ["SHOPIFY", "STRIPE"],
        "model_id": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
        "skills": [],
    },
    {
        "name": "Content Creator",
        "agent_type": "custom",
        "description": "Helps create and publish content across WordPress, Medium, and Notion. Streamlines your content workflow from draft to publication.",
        "category": "Content Creation",
        "tags": ["content", "writing", "blogging", "wordpress", "medium"],
        "tools": ["WORDPRESS", "MEDIUM", "NOTION"],
        "model_id": "anthropic/claude-3.5-sonnet",
        "skills": [],
    },
    {
        "name": "HR Assistant",
        "agent_type": "custom",
        "description": "Automates HR workflows with BambooHR integration. Manages employee onboarding, time-off requests, and team communication.",
        "category": "HR",
        "tags": ["hr", "human resources", "onboarding", "employees", "bamboohr"],
        "tools": ["BAMBOOHR", "SLACK"],
        "model_id": "llama-3.1-8b-instruct",
        "skills": [],
    },
    {
        "name": "Data Analyst",
        "agent_type": "custom",
        "description": "Analyzes data from Google Sheets and SQL databases. Generates insights, creates visualizations, and automates reporting workflows.",
        "category": "Data Analysis",
        "tags": ["data", "analytics", "sql", "reports", "insights"],
        "tools": ["GOOGLESHEETS", "POSTGRESQL"],
        "model_id": "deepseek-ai/DeepSeek-R1",
        "skills": [],
    },
]


def seed_starter_agents():
    """Seed the marketplace with 10 starter agents."""
    print("🌱 Seeding marketplace with starter agents...")

    # Create database connection
    database_url = get_database_url()
    engine = create_engine(database_url)

    with engine.connect() as db:
        # Start transaction
        trans = db.begin()

        try:
            for agent_config in STARTER_AGENTS:
                print(f"\n📦 Creating {agent_config['name']}...")

                # Store tool names in metadata (tool IDs will be resolved at install time)
                tool_names = agent_config['tools']

                # Build metadata
                metadata = {
                    "agent_type": agent_config['agent_type'],
                    "model_id": agent_config['model_id'],
                    "tools": [],  # Will be populated when tools are connected
                    "tool_names": tool_names,
                    "tool_icons": [],  # Will be populated at display time
                    "skills": agent_config['skills'],
                    "priority_level": "medium",
                    "max_concurrent_tasks": 3,
                    "auto_start": False,
                    "tags": agent_config['tags'],
                    "configuration": {}
                }

                # Check if agent already exists
                check_query = text("""
                    SELECT id FROM marketplace_items
                    WHERE name = :name AND type = 'agent'
                """)
                existing = db.execute(check_query, {"name": agent_config['name']})

                if existing.first():
                    print(f"   ⏭️  Agent already exists, skipping...")
                    continue

                # Insert marketplace item
                insert_query = text("""
                    INSERT INTO marketplace_items
                    (type, name, description, creator_name, creator_user_id, category, tags,
                     is_featured, is_approved, version, metadata, created_at, updated_at)
                    VALUES
                    ('agent', :name, :description, 'Automatos Team', NULL, :category, CAST(:tags AS jsonb),
                     true, true, '1.0.0', CAST(:metadata AS jsonb), NOW(), NOW())
                    RETURNING id
                """)

                result = db.execute(insert_query, {
                    "name": agent_config['name'],
                    "description": agent_config['description'],
                    "category": agent_config['category'],
                    "tags": json.dumps(agent_config['tags']),
                    "metadata": json.dumps(metadata)
                })

                agent_id = result.scalar()
                print(f"   ✅ Created marketplace agent ID: {agent_id}")
                print(f"      - Tools: {', '.join(tool_names) if tool_names else 'None'}")
                print(f"      - LLM: {agent_config['model_id']}")
                print(f"      - Category: {agent_config['category']}")

            # Commit transaction
            trans.commit()
            print("\n✨ Successfully seeded 10 starter agents!")

        except Exception as e:
            trans.rollback()
            print(f"\n❌ Error seeding agents: {str(e)}")
            raise


if __name__ == "__main__":
    seed_starter_agents()
