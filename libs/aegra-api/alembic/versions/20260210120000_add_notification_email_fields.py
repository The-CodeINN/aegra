"""add notification email fields and action item source

Revision ID: a1b2c3d4e5f6
Revises: 8a9c2d4e5f1b
Create Date: 2026-02-10 12:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision = "a1b2c3d4e5f6"
down_revision = "8a9c2d4e5f1b"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Notification model: add email/engagement tracking fields
    op.add_column("notifications", sa.Column("short_message", sa.Text(), nullable=True))
    op.add_column("notifications", sa.Column("full_message", sa.Text(), nullable=True))
    op.add_column("notifications", sa.Column("subject_line", sa.Text(), nullable=True))
    op.add_column("notifications", sa.Column("advisor_persona", sa.Text(), nullable=True))
    op.add_column("notifications", sa.Column("dismissed_at", sa.TIMESTAMP(timezone=True), nullable=True))
    op.add_column("notifications", sa.Column("clicked_at", sa.TIMESTAMP(timezone=True), nullable=True))
    op.add_column("notifications", sa.Column("delivered_at", sa.TIMESTAMP(timezone=True), nullable=True))

    # ActionItem model: add source and dependencies
    op.add_column("action_items", sa.Column("source", sa.Text(), nullable=True))
    op.add_column("action_items", sa.Column("dependencies", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("action_items", "dependencies")
    op.drop_column("action_items", "source")
    op.drop_column("notifications", "delivered_at")
    op.drop_column("notifications", "clicked_at")
    op.drop_column("notifications", "dismissed_at")
    op.drop_column("notifications", "advisor_persona")
    op.drop_column("notifications", "subject_line")
    op.drop_column("notifications", "full_message")
    op.drop_column("notifications", "short_message")
