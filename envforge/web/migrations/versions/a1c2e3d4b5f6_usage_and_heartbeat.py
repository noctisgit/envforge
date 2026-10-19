"""run usage + heartbeat, user session epoch

Revision ID: a1c2e3d4b5f6
Revises: f6af3cc35f31
Create Date: 2026-10-19 16:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a1c2e3d4b5f6"
down_revision: str | Sequence[str] | None = "f6af3cc35f31"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "runs", sa.Column("input_tokens", sa.Integer(), server_default="0", nullable=False)
    )
    op.add_column(
        "runs", sa.Column("output_tokens", sa.Integer(), server_default="0", nullable=False)
    )
    op.add_column("runs", sa.Column("cost_usd", sa.Float(), nullable=True))
    op.add_column("runs", sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True))
    op.add_column(
        "users", sa.Column("session_epoch", sa.Integer(), server_default="0", nullable=False)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("users", "session_epoch")
    op.drop_column("runs", "heartbeat_at")
    op.drop_column("runs", "cost_usd")
    op.drop_column("runs", "output_tokens")
    op.drop_column("runs", "input_tokens")
