"""provider key verification: model list, verified_at, note

Revision ID: b7d8e9f0a1b2
Revises: a1c2e3d4b5f6
Create Date: 2026-10-19 18:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "b7d8e9f0a1b2"
down_revision: str | Sequence[str] | None = "a1c2e3d4b5f6"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("provider_keys", sa.Column("models", sa.JSON(), nullable=True))
    op.add_column(
        "provider_keys", sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True)
    )
    op.add_column("provider_keys", sa.Column("verify_note", sa.Text(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("provider_keys", "verify_note")
    op.drop_column("provider_keys", "verified_at")
    op.drop_column("provider_keys", "models")
