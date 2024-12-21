import os
import pickle

import numpy as np
import torch
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DOCUMENTS_ROOT = "text"


def write_pickle(
    obj: any,
    filename: str,
) -> None:
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle)


def read_pickle(filename: str) -> any:
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def chunk_documents(documents_folder: str) -> list[Document]:
    filenames = sorted(
        [os.path.join(documents_folder, file) for file in os.listdir(documents_folder)]
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=300,
        add_start_index=True,
    )  # With chunk_size=900, 99%+ of chunks fit inside all-MiniLM-L6-v2 context window
    chunks = list()
    for file in filenames:
        doc = TextLoader(file).load()
        chunks += text_splitter.split_documents(doc)
    return chunks


def create_vector_store(
    documents_folder: str,
    filename: str,
) -> None:
    sentence_transformer = SentenceTransformer(MODEL_NAME)
    chunks = chunk_documents(documents_folder)
    ids = list()
    texts = list()
    embeddings = list()
    for chunk in tqdm(chunks[:50]):
        id = (
            f'{chunk.metadata["source"].split("/")[-1]}:{chunk.metadata["start_index"]}'
        )
        text = chunk.page_content
        with torch.no_grad():
            embedding = sentence_transformer.encode(text)
        ids.append(id)
        texts.append(text)
        embeddings.append(embedding)
    vector_store = {
        "ids": ids,
        "texts": texts,
        "embeddings": np.vstack(embeddings),
    }
    write_pickle(vector_store, filename)


class KnowledgeBase:
    def __init__(self):
        self.filenames = read_pickle("artifact_filenames.pickle")
        self.text = read_pickle("artifact_text.pickle")
        self.embeddings = np.load("artifact_embeddings.npy")
        self.sentence_transformer = SentenceTransformer(MODEL_NAME)

    def embed_query(
        self,
        query: str,
    ) -> np.array:
        with torch.no_grad():
            query_embedding = self.sentence_transformer.encode([query])
        return query_embedding

    def fetch_similar_documents(
        self,
        query: str,
        top_k: int,
    ) -> dict:
        query_embedding = self.embed_query(query)
        similarity = 1 - cdist(self.embeddings, query_embedding, metric="cosine")[:, 0]
        idx_top = np.flip(np.argsort(similarity))[:top_k]
        respose = {
            "filename": [self.filenames[i] for i in idx_top],
            "text": [self.text[i] for i in idx_top],
            "embedding": self.embeddings[idx_top],
            "cosine_similarity": similarity[idx_top],
        }
        return respose
