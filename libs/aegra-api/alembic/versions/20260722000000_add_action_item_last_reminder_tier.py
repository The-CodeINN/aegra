"""add action_items.last_reminder_tier

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-07-22 00:00:00.000000

"""

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = "e5f6a7b8c9d0"
down_revision = "d4e5f6a7b8c9"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # check_deadlines previously gated re-sends on elapsed time only, with no
    # memory of which tier was last notified — the same tier ("7d", "3d", ...)
    # kept resending every ~4h for as long as a task sat in it, producing a
    # dozen identical "Due This Week" emails before a task ever progressed to
    # the next tier. This column lets _should_send_reminder send once per
    # tier crossing instead.
    op.add_column("action_items", sa.Column("last_reminder_tier", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("action_items", "last_reminder_tier")
