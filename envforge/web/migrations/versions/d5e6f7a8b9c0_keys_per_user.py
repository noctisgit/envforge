"""provider keys belong to users, not workspaces

Revision ID: d5e6f7a8b9c0
Revises: c3d4e5f6a7b8
Create Date: 2026-10-19 20:00:00.000000

Existing keys move to their workspace's owner. If an owner had a key for the same
provider in several workspaces, the most recently updated one is kept.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "d5e6f7a8b9c0"
down_revision: str | Sequence[str] | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("provider_keys", sa.Column("user_id", sa.String(length=32), nullable=True))
    op.execute(
        "UPDATE provider_keys SET user_id = "
        "(SELECT owner_id FROM workspaces WHERE workspaces.id = provider_keys.workspace_id)"
    )
    op.execute(
        "DELETE FROM provider_keys WHERE id NOT IN (SELECT id FROM ("
        "  SELECT id, ROW_NUMBER() OVER ("
        "    PARTITION BY user_id, provider ORDER BY updated_at DESC) AS rn"
        "  FROM provider_keys) ranked WHERE rn = 1)"
    )
    with op.batch_alter_table("provider_keys", schema=None) as batch_op:
        batch_op.drop_index("ix_provider_keys_workspace_id")
        batch_op.drop_constraint("provider_keys_workspace_id_provider_key", type_="unique")
        batch_op.drop_constraint("provider_keys_workspace_id_fkey", type_="foreignkey")
        batch_op.drop_column("workspace_id")
        batch_op.alter_column("user_id", nullable=False)
        batch_op.create_foreign_key("provider_keys_user_id_fkey", "users", ["user_id"], ["id"])
        batch_op.create_index("ix_provider_keys_user_id", ["user_id"], unique=False)
        batch_op.create_unique_constraint(
            "provider_keys_user_id_provider_key", ["user_id", "provider"]
        )


def downgrade() -> None:
    """Downgrade schema: keys go back to each user's first owned workspace."""
    op.add_column("provider_keys", sa.Column("workspace_id", sa.String(length=32), nullable=True))
    op.execute(
        "UPDATE provider_keys SET workspace_id = (SELECT w.id FROM workspaces w "
        "WHERE w.owner_id = provider_keys.user_id ORDER BY w.created_at LIMIT 1)"
    )
    with op.batch_alter_table("provider_keys", schema=None) as batch_op:
        batch_op.drop_constraint("provider_keys_user_id_provider_key", type_="unique")
        batch_op.drop_index("ix_provider_keys_user_id")
        batch_op.drop_constraint("provider_keys_user_id_fkey", type_="foreignkey")
        batch_op.drop_column("user_id")
        batch_op.alter_column("workspace_id", nullable=False)
        batch_op.create_foreign_key(
            "provider_keys_workspace_id_fkey", "workspaces", ["workspace_id"], ["id"]
        )
        batch_op.create_index("ix_provider_keys_workspace_id", ["workspace_id"], unique=False)
        batch_op.create_unique_constraint(
            "provider_keys_workspace_id_provider_key", ["workspace_id", "provider"]
        )
