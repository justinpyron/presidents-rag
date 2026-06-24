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

MINI_L6_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MINI_L6_DIM = 384

ALLOWED_MODEL_NAMES = [
    MINI_L6_MODEL,
]

MODEL_WEIGHTS_PATHS: dict[str, str] = {
    MINI_L6_MODEL: "weights/sentence-transformers_all-MiniLM-L6-v2",
}


def get_model_config(model_name: str) -> tuple[type, int]:
    if model_name == MINI_L6_MODEL:
        return ChunkMiniL6, MINI_L6_DIM
    raise ValueError(
        f"Model {model_name!r} is not allowed. "
        f"Choose from: {ALLOWED_MODEL_NAMES}"
    )


class Base(DeclarativeBase):
    pass


class StoreRegistry(Base):
    """Registry of configurations for various embedding vector stores."""

    __tablename__ = "store_registry"
    __table_args__ = (
        UniqueConstraint(
            "model_name",
            "chunk_size",
            "chunk_overlap",
            name="uq_store_registry_model_chunk_params",
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


class ChunkMiniL6(Base):
    """Document chunks embedded with the all-MiniLM-L6-v2 sentence-transformers model."""

    __tablename__ = "chunks_mini_l6"
    __table_args__ = (
        UniqueConstraint(
            "store_registry_id",
            "source",
            "start_index",
            name="uq_chunks_mini_l6_store_registry_source_start",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    store_registry_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("store_registry.id"),
        nullable=False,
        index=True,
    )
    source: Mapped[str] = mapped_column(Text, nullable=False)
    start_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(MINI_L6_DIM),
        nullable=False,
    )
