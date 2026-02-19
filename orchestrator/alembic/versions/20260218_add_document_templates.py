"""Stub: document_templates already applied from main repo

Revision ID: 20260218_document_templates
Revises: 20260215_heartbeat_channels
Create Date: 2026-02-18

This is a stub migration. The actual document_templates table and
artifacts constraint update were applied from the main automatos-ai repo.
This file exists only to satisfy Alembic's revision chain resolution.
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '20260218_document_templates'
down_revision = '20260215_heartbeat_channels'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Tables already created by main repo migration
    pass


def downgrade() -> None:
    # Managed by main repo migration
    pass
