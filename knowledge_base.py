import numpy as np
import torch
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer

from create_artifacts import MODEL_NAME, read_pickle


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
    ) -> tuple[list[str], list[str], np.array, np.array]:
        query_embedding = self.embed_query(query)
        distance = cdist(self.embeddings, query_embedding, metric="cosine")[:, 0]
        top_index = np.argsort(distance)[:top_k]
        top_filenames = [self.filenames[i] for i in top_index]
        top_text = [self.text[i] for i in top_index]
        top_embeddings = self.embeddings[top_index]
        top_cosine_similarities = 1 - distance[top_index]
        return top_filenames, top_text, top_embeddings, top_cosine_similarities
