"""Unify marketplace_category values to lowercase system.

Maps legacy Title Case marketplace categories to the unified lowercase category IDs
used by the icon system and agent-constants.ts.

Old → New:
  Personal Assistant → general
  Customer Support → support
  DevOps → development
  Social Media → marketing
  Accounting → business
  E-commerce → sales
  Content Creation → writing
  Data Analysis → analytics
  Operations → productivity
  Productivity → productivity
  Research → research
  Custom → custom
"""

from alembic import op

revision = "unify_marketplace_categories"
down_revision = None  # Standalone — safe to run anytime
branch_labels = None
depends_on = None

# Old Title Case → new lowercase
CATEGORY_MAP = {
    "Personal Assistant": "general",
    "Customer Support": "support",
    "DevOps": "development",
    "Social Media": "marketing",
    "Accounting": "business",
    "E-commerce": "sales",
    "Content Creation": "writing",
    "HR": "hr",
    "Data Analysis": "analytics",
    "Custom": "custom",
    "Operations": "productivity",
    "Productivity": "productivity",
    "Research": "research",
}

# Reverse map for downgrade
REVERSE_MAP = {
    "general": "Personal Assistant",
    "support": "Customer Support",
    "development": "DevOps",
    "marketing": "Social Media",
    "business": "Accounting",
    "sales": "E-commerce",
    "writing": "Content Creation",
    "hr": "HR",
    "analytics": "Data Analysis",
    "custom": "Custom",
    "productivity": "Productivity",
    "research": "Research",
}


def upgrade():
    # Update agents table
    for old, new in CATEGORY_MAP.items():
        op.execute(
            f"UPDATE agents SET marketplace_category = '{new}' "
            f"WHERE marketplace_category = '{old}'"
        )

    # Update workflow_recipes table
    for old, new in CATEGORY_MAP.items():
        op.execute(
            f"UPDATE workflow_recipes SET marketplace_category = '{new}' "
            f"WHERE marketplace_category = '{old}'"
        )

    # Also update configuration.category in agents JSON for round-trip support
    for old, new in CATEGORY_MAP.items():
        op.execute(
            f"UPDATE agents SET configuration = jsonb_set(configuration, '{{category}}', '\"{new}\"') "
            f"WHERE marketplace_category = '{new}' AND configuration IS NOT NULL "
            f"AND (configuration->>'category' IS NULL OR configuration->>'category' = '{old}')"
        )


def downgrade():
    for new, old in REVERSE_MAP.items():
        op.execute(
            f"UPDATE agents SET marketplace_category = '{old}' "
            f"WHERE marketplace_category = '{new}'"
        )
    for new, old in REVERSE_MAP.items():
        op.execute(
            f"UPDATE workflow_recipes SET marketplace_category = '{old}' "
            f"WHERE marketplace_category = '{new}'"
        )
