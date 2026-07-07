"""add action item task tracking fields (evidence, advisor_note, miss_count)

Revision ID: b3c4d5e6f7a8
Revises: a9b0c1d2e3f4
Create Date: 2026-07-03 00:00:00.000000

"""

import sqlalchemy as sa
from sqlalchemy import text

from alembic import op

# revision identifiers, used by Alembic.
revision = "b3c4d5e6f7a8"
down_revision = "a9b0c1d2e3f4"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Agent optimisation spec Item 2 (TaskMemory): the agent needs to record
    # proof of completion, a private note on how to handle follow-ups, and a
    # consecutive-miss counter to drive escalation logic — none of which
    # existed on this table before the agent had any write path into it.
    op.add_column("action_items", sa.Column("evidence", sa.Text(), nullable=True))
    op.add_column("action_items", sa.Column("advisor_note", sa.Text(), nullable=True))
    op.add_column(
        "action_items",
        sa.Column("miss_count", sa.Integer(), server_default=text("0"), nullable=False),
    )


def downgrade() -> None:
    op.drop_column("action_items", "miss_count")
    op.drop_column("action_items", "advisor_note")
    op.drop_column("action_items", "evidence")
