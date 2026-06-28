"""SQLAlchemy models for the vector store."""

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class VectorStoreConfig(Base):
    """Configuration for an embedding vector store."""

    __tablename__ = "vector_store_configs"
    __table_args__ = (
        UniqueConstraint(
            "model_name",
            "chunk_size",
            "chunk_overlap",
            name="uq_vector_store_configs_model_chunk_params",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    model_name: Mapped[str] = mapped_column(String(255), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_size: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_overlap: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )


class ChunkDim384(Base):
    """Document chunks stored as 384-dimensional embedding vectors."""

    __tablename__ = "chunks_dim_384"
    __table_args__ = (
        UniqueConstraint(
            "vector_store_config_id",
            "source",
            "document",
            "start_index",
            name="uq_chunks_dim_384_vector_store_config_source_document_start",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vector_store_config_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("vector_store_configs.id"),
        nullable=False,
        index=True,
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    document: Mapped[str] = mapped_column(Text, nullable=False)
    start_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(384),
        nullable=False,
    )
