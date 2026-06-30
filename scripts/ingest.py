"""Chunk documents, embed them, and load into the vector store database."""

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from db.models import ChunkDim384, VectorStoreConfig
from db.session import get_session


@dataclass(frozen=True)
class EmbeddingModelConfig:
    weights_path: str
    data_model: type
    dim: int


@dataclass(frozen=True)
class ChunkRecord:
    source: str
    document: str
    start_index: int
    text: str


MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": EmbeddingModelConfig(
        weights_path="weights/sentence-transformers_all-MiniLM-L6-v2",
        data_model=ChunkDim384,
        dim=384,
    ),
    "sentence-transformers/all-MiniLM-L12-v2": EmbeddingModelConfig(
        weights_path="weights/sentence-transformers_all-MiniLM-L12-v2",
        data_model=ChunkDim384,
        dim=384,
    ),
    "BAAI/bge-small-en-v1.5": EmbeddingModelConfig(
        weights_path="weights/BAAI_bge-small-en-v1.5",
        data_model=ChunkDim384,
        dim=384,
    ),
}
PROJECT_ROOT = Path(__file__).parent.parent
SOURCES: dict[str, Path] = {
    "wikipedia": PROJECT_ROOT / "text" / "wikipedia",
    "miller_center": PROJECT_ROOT / "text" / "miller_center",
}


def chunk_txt_files(
    source: str,
    documents_folder: Path,
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks: list[ChunkRecord] = []
    for path in sorted(documents_folder.glob("*.txt")):
        text = path.read_text(encoding="utf-8")
        for document in text_splitter.create_documents([text]):
            chunks.append(
                ChunkRecord(
                    source=source,
                    document=path.name,
                    start_index=document.metadata["start_index"],
                    text=document.page_content,
                )
            )
    return chunks


def save_chunks(
    chunks: list[ChunkRecord],
    chunk_size: int,
    chunk_overlap: int,
) -> Path:
    out_dir = PROJECT_ROOT / "chunks"
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"chunks_cs{chunk_size}_co{chunk_overlap}.json"
    path.write_text(
        json.dumps(
            [asdict(chunk) for chunk in chunks], indent=2, ensure_ascii=False
        ),
        encoding="utf-8",
    )
    return path


def ingest(
    chunks: list[ChunkRecord],
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> int:
    """Embed ``chunks`` and load them into the vector store."""
    if not chunks:
        raise ValueError("No chunks to ingest.")

    model = MODELS[model_name]
    session = get_session()
    try:
        existing = session.execute(
            select(VectorStoreConfig).where(
                VectorStoreConfig.model_name == model_name,
                VectorStoreConfig.chunk_size == chunk_size,
                VectorStoreConfig.chunk_overlap == chunk_overlap,
            )
        ).scalar_one_or_none()
        if existing is not None:
            raise ValueError(
                f"A vector store config already exists for model_name={model_name!r}, "
                f"chunk_size={chunk_size}, chunk_overlap={chunk_overlap}."
            )
        vector_store_config = VectorStoreConfig(
            model_name=model_name,
            dim=model.dim,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        session.add(vector_store_config)
        session.flush()

        embedder = SentenceTransformer(model.weights_path)
        texts = [chunk.text for chunk in chunks]
        embeddings = embedder.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
        )
        chunk_records = [
            model.data_model(
                vector_store_config_id=vector_store_config.id,
                source=chunk.source,
                document=chunk.document,
                start_index=chunk.start_index,
                text=chunk.text,
                embedding=embedding.tolist(),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
        session.add_all(chunk_records)
        session.commit()
        return len(chunk_records)
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ingest all configured text sources into the vector store."
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        choices=MODELS,
        help="Sentence-transformers model name.",
    )
    parser.add_argument(
        "-cs",
        "--chunk-size",
        type=int,
        required=True,
        help="Chunk size for splitting documents.",
    )
    parser.add_argument(
        "-co",
        "--chunk-overlap",
        type=int,
        required=True,
        help="Number of overlapping characters between chunks.",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for embedding chunks.",
    )
    parser.add_argument(
        "-s",
        "--save-chunks",
        action="store_true",
        help="Save all chunks to a local JSON file before ingesting.",
    )
    args = parser.parse_args()

    if args.model not in MODELS:
        raise ValueError(
            f"Model {args.model!r} is not allowed. "
            f"Choose from: {list(MODELS)}"
        )

    start_time = time.time()
    chunks: list[ChunkRecord] = []
    for source, documents_folder in SOURCES.items():
        source_chunks = chunk_txt_files(
            source,
            documents_folder,
            args.chunk_size,
            args.chunk_overlap,
        )
        if not source_chunks:
            raise ValueError(f"No .txt files found in {documents_folder!r}")
        chunks.extend(source_chunks)
        print(
            f"Chunked {len(source_chunks)} chunks from {source!r} "
            f"({documents_folder})"
        )

    if args.save_chunks:
        path = save_chunks(chunks, args.chunk_size, args.chunk_overlap)
        print(f"Saved {len(chunks)} chunks to {path}")

    num_chunks = ingest(
        chunks=chunks,
        model_name=args.model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    elapsed = time.time() - start_time
    print(f"Ingested {num_chunks} total chunks in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
