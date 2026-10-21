"""training_jobs and adapters (LoRA SFT from the browser)

Revision ID: 6a9e3c1d4f32
Revises: 5f8d2b0c3e21
Create Date: 2026-10-21 10:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6a9e3c1d4f32"
down_revision: str | Sequence[str] | None = "5f8d2b0c3e21"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "training_jobs",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("workspace_id", sa.String(length=32), nullable=False),
        sa.Column("dataset_id", sa.String(length=32), nullable=True),
        sa.Column("box_id", sa.String(length=32), nullable=True),
        sa.Column("job_id", sa.String(length=32), nullable=True),
        sa.Column("adapter_name", sa.String(length=64), nullable=False),
        sa.Column("base_model", sa.String(length=200), nullable=False),
        sa.Column("hyperparams", sa.JSON(), nullable=False),
        sa.Column("runtime", sa.String(length=8), nullable=True),
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("cancel_requested", sa.Boolean(), server_default="false", nullable=False),
        sa.Column("step", sa.Integer(), server_default="0", nullable=False),
        sa.Column("total_steps", sa.Integer(), nullable=True),
        sa.Column("epoch", sa.Float(), nullable=True),
        sa.Column("losses", sa.JSON(), nullable=True),
        sa.Column("log_cursor", sa.Integer(), server_default="0", nullable=False),
        sa.Column("log_tail", sa.Text(), nullable=True),
        sa.Column("poll_failures", sa.Integer(), server_default="0", nullable=False),
        sa.Column("polled_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("hourly_usd", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_by", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["box_id"], ["gpu_boxes.id"]),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_training_jobs_workspace_id"), "training_jobs", ["workspace_id"])
    op.create_index(op.f("ix_training_jobs_status"), "training_jobs", ["status"])
    op.create_table(
        "adapters",
        sa.Column("id", sa.String(length=32), nullable=False),
        sa.Column("workspace_id", sa.String(length=32), nullable=False),
        sa.Column("name", sa.String(length=64), nullable=False),
        sa.Column("base_model", sa.String(length=200), nullable=False),
        sa.Column("pack", sa.String(length=80), nullable=True),
        sa.Column("dataset_id", sa.String(length=32), nullable=True),
        sa.Column("training_job_id", sa.String(length=32), nullable=True),
        sa.Column("box_id", sa.String(length=32), nullable=True),
        sa.Column("path", sa.String(length=500), nullable=False),
        sa.Column("lora_rank", sa.Integer(), nullable=True),
        sa.Column("metrics", sa.JSON(), nullable=True),
        sa.Column("test_run_id", sa.String(length=80), nullable=True),
        sa.Column("created_by", sa.String(length=32), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["created_by"], ["users.id"]),
        sa.ForeignKeyConstraint(["dataset_id"], ["datasets.id"]),
        sa.ForeignKeyConstraint(["training_job_id"], ["training_jobs.id"]),
        sa.ForeignKeyConstraint(["workspace_id"], ["workspaces.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("workspace_id", "name"),
    )
    op.create_index(op.f("ix_adapters_workspace_id"), "adapters", ["workspace_id"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(op.f("ix_adapters_workspace_id"), table_name="adapters")
    op.drop_table("adapters")
    op.drop_index(op.f("ix_training_jobs_status"), table_name="training_jobs")
    op.drop_index(op.f("ix_training_jobs_workspace_id"), table_name="training_jobs")
    op.drop_table("training_jobs")
