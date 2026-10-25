"""GPU boxes without Tailscale: a per-workspace SSH key and vLLM over an SSH tunnel

Revision ID: 8c1e5f2a9d07
Revises: 7b0f4d2e5a43
Create Date: 2026-10-25 20:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8c1e5f2a9d07"
down_revision: str | Sequence[str] | None = "7b0f4d2e5a43"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("workspaces", sa.Column("ssh_key_ciphertext", sa.Text(), nullable=True))
    op.add_column("workspaces", sa.Column("ssh_public_key", sa.Text(), nullable=True))
    op.add_column(
        "workspaces", sa.Column("ssh_key_created_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column(
        "gpu_boxes",
        sa.Column("vllm_access", sa.String(length=8), server_default="direct", nullable=False),
    )
    with op.batch_alter_table("gpu_boxes") as batch:
        batch.alter_column("ssh_key_ciphertext", existing_type=sa.Text(), nullable=True)
        batch.alter_column("ssh_key_fingerprint", existing_type=sa.String(length=80), nullable=True)


def downgrade() -> None:
    """Downgrade schema. Refuses while boxes use the workspace key (their key column is
    NULL, which the old schema can't hold): remove those boxes first."""
    n = op.get_bind().scalar(
        sa.text("SELECT count(*) FROM gpu_boxes WHERE ssh_key_ciphertext IS NULL")
    )
    if n:
        raise RuntimeError(
            f"{n} GPU box(es) use the workspace SSH key; remove them before downgrading"
        )
    with op.batch_alter_table("gpu_boxes") as batch:
        batch.alter_column(
            "ssh_key_fingerprint", existing_type=sa.String(length=80), nullable=False
        )
        batch.alter_column("ssh_key_ciphertext", existing_type=sa.Text(), nullable=False)
    op.drop_column("gpu_boxes", "vllm_access")
    op.drop_column("workspaces", "ssh_key_created_at")
    op.drop_column("workspaces", "ssh_public_key")
    op.drop_column("workspaces", "ssh_key_ciphertext")
