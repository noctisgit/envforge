"""spend controls: builder extraction approvals and a monthly workspace limit

Revision ID: 7b0f4d2e5a43
Revises: 6a9e3c1d4f32
Create Date: 2026-10-21 22:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "7b0f4d2e5a43"
down_revision: str | Sequence[str] | None = "6a9e3c1d4f32"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("env_projects", sa.Column("extraction_quote", sa.JSON(), nullable=True))
    op.add_column(
        "env_projects",
        sa.Column("extraction_allowance", sa.Integer(), server_default="0", nullable=False),
    )
    op.add_column("workspaces", sa.Column("monthly_spend_limit_usd", sa.Float(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("workspaces", "monthly_spend_limit_usd")
    op.drop_column("env_projects", "extraction_allowance")
    op.drop_column("env_projects", "extraction_quote")
