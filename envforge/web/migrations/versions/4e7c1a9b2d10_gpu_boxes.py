"""gpu_boxes; jobs.workspace_id (compute jobs); runs.gpu_box_id (runs on a box endpoint)

Revision ID: 4e7c1a9b2d10
Revises: b8c9d0e1f2a3
Create Date: 2026-10-21 10:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "4e7c1a9b2d10"
down_revision: str | Sequence[str] | None = "b8c9d0e1f2a3"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "gpu_boxes",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("workspace_id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("host", sa.String(length=255), nullable=False),
        sa.Column("port", sa.Integer(), nullable=False),
        sa.Column("user", sa.String(length=64), nullable=False),
        sa.Column("runtime", sa.String(length=8), nullable=False),
        sa.Column("detected_runtime", sa.String(length=8), nullable=True),
        sa.Column("ssh_key_ciphertext", sa.Text(), nullable=False),
        sa.Column("ssh_key_fingerprint", sa.String(length=80), nullable=False),
        sa.Column("vllm_key_ciphertext", sa.Text(), nullable=True),
        sa.Column("vllm_key_masked", sa.String(length=16), nullable=True),
        sa.Column("vllm_key_generated", sa.Boolean(), server_default="true", nullable=False),
        sa.Column("hf_token_ciphertext", sa.Text(), nullable=True),
        sa.Column("hf_token_masked", sa.String(length=16), nullable=True),
        sa.Column("vllm_port", sa.Integer(), nullable=False),
        sa.Column("endpoint_host", sa.String(length=255), nullable=True),
        sa.Column("bind", sa.String(length=64), nullable=True),
        sa.Column("hourly_usd", sa.Float(), nullable=True),
        sa.Column("status", sa.String(length=16), server_default="unknown", nullable=False),
        sa.Column("check_result", sa.JSON(), nullable=True),
        sa.Column("checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("serve_state", sa.String(length=16), server_default="stopped", nullable=False),
        sa.Column("served_model", sa.String(length=200), nullable=True),
        sa.Column("served_adapters", sa.JSON(), nullable=True),
        sa.Column("endpoint_url", sa.String(length=500), nullable=True),
        sa.Column("serve_error", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "name"),
    )
    op.create_index(op.f("ix_gpu_boxes_workspace_id"), "gpu_boxes", ["workspace_id"])
    with op.batch_alter_table("jobs") as batch:
        batch.add_column(sa.Column("workspace_id", sa.String(length=32), nullable=True))
        batch.create_index("ix_jobs_workspace_id", ["workspace_id"])
        batch.create_foreign_key("fk_jobs_workspace_id", "workspaces", ["workspace_id"], ["id"])
    op.add_column("runs", sa.Column("gpu_box_id", sa.String(length=32), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("runs", "gpu_box_id")
    with op.batch_alter_table("jobs") as batch:
        batch.drop_constraint("fk_jobs_workspace_id", type_="foreignkey")
        batch.drop_index("ix_jobs_workspace_id")
        batch.drop_column("workspace_id")
    op.drop_index(op.f("ix_gpu_boxes_workspace_id"), table_name="gpu_boxes")
    op.drop_table("gpu_boxes")
