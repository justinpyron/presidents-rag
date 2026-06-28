"""add document to chunk schema

Revision ID: 720ed6dc040e
Revises: 73e3b4337b7b
Create Date: 2026-06-28 14:03:03.246784

"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "720ed6dc040e"
down_revision: Union[str, Sequence[str], None] = "73e3b4337b7b"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Existing rows stored the filename in ``source``; all current data is Wikipedia.
_LEGACY_SOURCE = "wikipedia"


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "chunks_dim_384",
        sa.Column("document", sa.Text(), nullable=True),
    )
    op.execute("UPDATE chunks_dim_384 SET document = source")

    op.drop_constraint(
        "uq_chunks_dim_384_vector_store_config_source_start",
        "chunks_dim_384",
        type_="unique",
    )
    op.execute(f"UPDATE chunks_dim_384 SET source = '{_LEGACY_SOURCE}'")

    op.alter_column("chunks_dim_384", "document", nullable=False)
    op.create_unique_constraint(
        "uq_chunks_dim_384_vector_store_config_source_document_start",
        "chunks_dim_384",
        ["vector_store_config_id", "source", "document", "start_index"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "uq_chunks_dim_384_vector_store_config_source_document_start",
        "chunks_dim_384",
        type_="unique",
    )
    op.execute("UPDATE chunks_dim_384 SET source = document")
    op.drop_column("chunks_dim_384", "document")
    op.create_unique_constraint(
        "uq_chunks_dim_384_vector_store_config_source_start",
        "chunks_dim_384",
        ["vector_store_config_id", "source", "start_index"],
    )
