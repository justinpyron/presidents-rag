import numpy as np
import torch
from create_artifacts import MODEL_NAME, read_pickle
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer


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
