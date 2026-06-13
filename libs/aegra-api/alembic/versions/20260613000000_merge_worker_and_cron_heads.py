"""Merge worker/rag branch and cron/gin-index branch into a single head.

The migration graph has two heads:
  - d6e7f8a9b0c1 (merge_upstream_worker_and_aiservice_branches)
  - c7d1f2a4b6e8 (cron_cascade_and_claim)

Both descend from a1b2c3d4e5f6 via different paths.  Production DBs that
had both paths applied hold two rows in alembic_version, causing
``alembic upgrade head`` to fail with "Multiple head revisions".  This
no-op merge collapses them to a single head so the precheck and upgrade
work correctly.

Revision ID: a9b0c1d2e3f4
Revises: d6e7f8a9b0c1, c7d1f2a4b6e8
Create Date: 2026-06-13 00:00:00.000000

"""

revision = "a9b0c1d2e3f4"
down_revision = ("d6e7f8a9b0c1", "c7d1f2a4b6e8")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
