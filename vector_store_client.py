import pickle

import numpy as np
import torch
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

MODEL_WEIGHTS = "weights/all-MiniLM-L6-v2"
VECTOR_STORE_PATH = "vector_store.pickle"


def read_pickle(filename: str) -> any:
    with open(filename, "rb") as handle:
        return pickle.load(handle)


class VectorStore:
    def __init__(self):
        self.vector_store = read_pickle(VECTOR_STORE_PATH)
        self.sentence_transformer = SentenceTransformer(MODEL_WEIGHTS)

    def query(
        self,
        query: str,
        n_results: int,
    ) -> dict:
        with torch.no_grad():
            query_embedding = self.sentence_transformer.encode([query])
        distance = cdist(
            self.vector_store["embeddings"], query_embedding, metric="cosine"
        )[:, 0]
        idx_top = np.argsort(distance)[:n_results]
        return {
            "ids": [self.vector_store["ids"][i] for i in idx_top],
            "texts": [self.vector_store["texts"][i] for i in idx_top],
            "distances": distance[idx_top],
        }
