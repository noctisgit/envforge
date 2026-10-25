"""DPO and GRPO: preference datasets, training methods, GRPO state, repeated rollouts

Revision ID: 9d2f6b1c3e84
Revises: 8c1e5f2a9d07
Create Date: 2026-10-25 23:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "9d2f6b1c3e84"
down_revision: str | Sequence[str] | None = "8c1e5f2a9d07"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "datasets", sa.Column("kind", sa.String(length=16), server_default="sft", nullable=False)
    )
    op.add_column(
        "training_jobs",
        sa.Column("method", sa.String(length=8), server_default="sft", nullable=False),
    )
    op.add_column("training_jobs", sa.Column("init_adapter", sa.String(length=64), nullable=True))
    op.add_column("training_jobs", sa.Column("state", sa.JSON(), nullable=True))
    op.add_column("runs", sa.Column("repeats", sa.Integer(), server_default="1", nullable=False))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("runs", "repeats")
    op.drop_column("training_jobs", "state")
    op.drop_column("training_jobs", "init_adapter")
    op.drop_column("training_jobs", "method")
    op.drop_column("datasets", "kind")
