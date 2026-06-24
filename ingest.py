"""Chunk documents, embed them, and load into the vector store database."""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from db.models import (
    ALLOWED_MODEL_NAMES,
    MODEL_WEIGHTS_PATHS,
    StoreRegistry,
    get_model_config,
)
from db.session import get_session


@dataclass(frozen=True)
class ChunkRecord:
    source: str
    start_index: int
    text: str


def chunk_txt_files(
    directory: str,  # TODO: Rename to documents_folder
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks: list[ChunkRecord] = []
    for path in sorted(Path(directory).glob("*.txt")):
        documents = TextLoader(str(path)).load()
        for document in text_splitter.split_documents(documents):
            chunks.append(
                ChunkRecord(
                    source=path.name,
                    start_index=document.metadata["start_index"],
                    text=document.page_content,
                )
            )
    return chunks


def assert_registry_not_exists(
    session,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> None:
    existing = session.execute(
        select(StoreRegistry).where(
            StoreRegistry.model_name == model_name,
            StoreRegistry.chunk_size == chunk_size,
            StoreRegistry.chunk_overlap == chunk_overlap,
        )
    ).scalar_one_or_none()
    if existing is not None:
        raise ValueError(
            f"A store registry already exists for model_name={model_name!r}, "
            f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}."
        )


def load_embedding_model(model_name: str) -> SentenceTransformer:
    weights_path = MODEL_WEIGHTS_PATHS.get(model_name, model_name)
    if os.path.isdir(weights_path):
        return SentenceTransformer(weights_path)
    return SentenceTransformer(model_name)


def ingest(
    directory: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> int:
    # TODO: Add docstring
    chunk_model, dim = get_model_config(model_name)
    chunks = chunk_txt_files(directory, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(f"No .txt files found in {directory!r}")

    session = get_session()
    try:
        assert_registry_not_exists(
            session,
            model_name=model_name,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        registry = StoreRegistry(
            model_name=model_name,
            dim=dim,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        session.add(registry)
        session.flush()

        embedder = load_embedding_model(model_name)
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        rows = [
            {
                "store_registry_id": registry.id,
                "source": chunk.source,
                "start_index": chunk.start_index,
                "text": chunk.text,
                "embedding": embedding.tolist(),
            }
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        session.execute(insert(chunk_model), rows)
        session.commit()
        return len(rows)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ingest text files into the vector store."
    )
    parser.add_argument(
        "--directory",
        required=True,
        help="Directory containing .txt files to ingest.",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=ALLOWED_MODEL_NAMES,
        help="Sentence-transformers model name.",
    )
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    num_chunks = ingest(
        directory=args.directory,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    print(f"Ingested {num_chunks} chunks from {args.directory}")


if __name__ == "__main__":
    main()
