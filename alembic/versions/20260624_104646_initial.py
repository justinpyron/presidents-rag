"""Initial

Revision ID: 73e3b4337b7b
Revises:
Create Date: 2026-06-24 10:46:46.572398

"""

from typing import Sequence, Union

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "73e3b4337b7b"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.create_table(
        "vector_store_configs",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("model_name", sa.String(length=255), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("chunk_size", sa.Integer(), nullable=False),
        sa.Column("chunk_overlap", sa.Integer(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "model_name",
            "chunk_size",
            "chunk_overlap",
            name="uq_vector_store_configs_model_chunk_params",
        ),
    )
    op.create_table(
        "chunks_dim_384",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("vector_store_config_id", sa.Integer(), nullable=False),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("start_index", sa.Integer(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384), nullable=False),
        sa.ForeignKeyConstraint(
            ["vector_store_config_id"],
            ["vector_store_configs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "vector_store_config_id",
            "source",
            "start_index",
            name="uq_chunks_dim_384_vector_store_config_source_start",
        ),
    )
    op.create_index(
        op.f("ix_chunks_dim_384_vector_store_config_id"),
        "chunks_dim_384",
        ["vector_store_config_id"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        op.f("ix_chunks_dim_384_vector_store_config_id"),
        table_name="chunks_dim_384",
    )
    op.drop_table("chunks_dim_384")
    op.drop_table("vector_store_configs")
