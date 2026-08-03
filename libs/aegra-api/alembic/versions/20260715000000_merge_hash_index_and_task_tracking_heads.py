"""Merge hash-index branch and task-tracking branch into a single head.

The migration graph diverged again after the June 13 merge
(``a9b0c1d2e3f4``): ``b88bb61be638`` (hash_assistant_config_index, created
2026-06-12) branches off ``c7d1f2a4b6e8`` in parallel with the work that
merge collapsed, and was never folded back in. ``b3c4d5e6f7a8``
(add_action_item_task_tracking_fields, created 2026-07-03) then extended the
other branch, leaving two heads:
  - b88bb61be638 (hash_assistant_config_index)
  - b3c4d5e6f7a8 (add_action_item_task_tracking_fields)

``alembic upgrade head`` fails with "Multiple head revisions" until these
are collapsed, which is what crash-loops the app on startup — see deploy
run on 2026-07-15. This no-op merge collapses them to a single head.

Revision ID: d4e5f6a7b8c9
Revises: b88bb61be638, b3c4d5e6f7a8
Create Date: 2026-07-15 00:00:00.000000

"""

revision = "d4e5f6a7b8c9"
down_revision = ("b88bb61be638", "b3c4d5e6f7a8")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
