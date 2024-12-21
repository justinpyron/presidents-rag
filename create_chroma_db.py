import os

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm

DOCUMENTS_ROOT = "text"
VECTOR_STORE_PATH = "vector_store"
VECTOR_STORE_NAME = "presidents"


def init_vector_store() -> Collection:
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    embedding_model = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = client.get_or_create_collection(
        name=VECTOR_STORE_NAME,
        embedding_function=embedding_model,
        metadata={"hnsw:space": "cosine"},  # Distance function
    )
    return vector_store


def chunk_documents() -> list[Document]:
    filenames = sorted(
        [os.path.join(DOCUMENTS_ROOT, file) for file in os.listdir(DOCUMENTS_ROOT)]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=300,
        add_start_index=True,
    )  # With chunk_size=900, 99%+ of chunks fit inside model context window
    chunks = list()
    for file in filenames:
        doc = TextLoader(file).load()
        chunks += text_splitter.split_documents(doc)
    return chunks


def create_vector_store() -> None:
    vector_store = init_vector_store()
    chunks = chunk_documents()
    for chunk in tqdm(chunks):
        metadata = chunk.metadata
        vector_store.add(
            ids=[f'{metadata["source"]}_{metadata["start_index"]}'],
            documents=[chunk.page_content],
            metadatas=[metadata],
        )


if __name__ == "__main__":
    create_vector_store()
