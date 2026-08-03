"""Merge upstream worker branch with ai-service branch

Merges:
- b7d9f1c2a345: replace_rag_with_local_course_content (ai-service)
- c4d5e6f7a8b9: add_execution_params_and_lease_columns (upstream worker)

Revision ID: d6e7f8a9b0c1
Revises: b7d9f1c2a345, c4d5e6f7a8b9
Create Date: 2026-03-15 00:00:00.000000

"""

# revision identifiers, used by Alembic.
revision = "d6e7f8a9b0c1"
down_revision = ("b7d9f1c2a345", "c4d5e6f7a8b9")
branch_labels = None
depends_on = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
