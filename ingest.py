"""Chunk documents, embed them, and load into the vector store database."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
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


MODELS = {
    "sentence-transformers/all-MiniLM-L6-v2": EmbeddingModelConfig(
        weights_path="weights/sentence-transformers_all-MiniLM-L6-v2",
        data_model=ChunkDim384,
        dim=384,
    ),
}


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


def ingest(
    directory: str,
    model_name: str,
    chunk_size: int,
    chunk_overlap: int,
    batch_size: int,
) -> int:
    # TODO: Add docstring
    if model_name not in MODELS:
        raise ValueError(
            f"Model {model_name!r} is not allowed. "
            f"Choose from: {list(MODELS)}"
        )
    model = MODELS[model_name]
    chunks = chunk_txt_files(directory, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError(f"No .txt files found in {directory!r}")

    session = get_session()
    try:
        # 1. Create vector store config record
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

        # 2. Create embedding records
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
        description="Ingest text files into the vector store."
    )
    parser.add_argument(
        "-d",
        "--directory",
        required=True,
        help="Directory containing .txt files to ingest.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        choices=MODELS,
        help="Sentence-transformers model name.",
    )
    parser.add_argument("-cs", "--chunk-size", type=int, default=900)
    parser.add_argument("-co", "--chunk-overlap", type=int, default=300)
    parser.add_argument("-b", "--batch-size", type=int, default=64)
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
