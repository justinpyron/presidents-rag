"""Modal-based inference server for Presidents RAG."""

import pickle

import modal

# Modal configuration
MODAL_APP_NAME = "presidents-rag"
MODAL_VOLUME_NAME = "presidents-rag"

# Model and data paths (in Modal volume)
SENTENCE_TRANSFORMER_PATH = "/data/weights/sentence-transformers_all-MiniLM-L6-v2"
CROSS_ENCODER_PATH = "/data/weights/cross-encoder_ms-marco-MiniLM-L-6-v2"
VECTOR_STORE_PATH = "/data/vector_store.pickle"

app = modal.App(MODAL_APP_NAME)
image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "torch==2.5.1",
    "sentence-transformers==3.3.0",
    "numpy==2.1.3",
    "scipy==1.14.1",
    "pydantic==2.10.4",
    "fastapi==0.115.0",
)
volume = modal.Volume.from_name(MODAL_VOLUME_NAME)


@app.cls(
    image=image,
    volumes={"/data": volume},
    cpu=2,
    scaledown_window=600,
)
class Server:
    """Modal class for serving vector store queries and reranking."""

    @modal.enter()
    def load_models(self):
        """Load models and vector store on container startup."""
        from sentence_transformers import CrossEncoder, SentenceTransformer

        # Load sentence transformer for embeddings
        self.sentence_transformer = SentenceTransformer(SENTENCE_TRANSFORMER_PATH)

        # Load cross encoder for reranking
        self.cross_encoder = CrossEncoder(CROSS_ENCODER_PATH)

        # Load vector store
        with open(VECTOR_STORE_PATH, "rb") as handle:
            self.vector_store = pickle.load(handle)

    def query(self, query: str, n_results: int) -> dict:
        """
        Query the vector store for similar documents.

        Args:
            query: The query string
            n_results: Number of results to return

        Returns:
            Dictionary with ids, texts, and distances
        """
        import numpy as np
        import torch
        from scipy.spatial.distance import cdist

        # Generate query embedding
        with torch.no_grad():
            query_embedding = self.sentence_transformer.encode([query])

        # Compute distances
        distance = cdist(
            self.vector_store["embeddings"], query_embedding, metric="cosine"
        )[:, 0]

        # Get top results
        idx_top = np.argsort(distance)[:n_results]

        return {
            "ids": [self.vector_store["ids"][i] for i in idx_top],
            "texts": [self.vector_store["texts"][i] for i in idx_top],
            "distances": distance[idx_top].tolist(),
        }

    def rerank(
        self,
        query: str,
        ids: list[str],
        documents: list[str],
        top_k: int,
    ) -> dict:
        """
        Rerank documents using the cross encoder.

        Args:
            query: The query string
            ids: List of document IDs
            documents: List of document texts
            top_k: Number of top results to return

        Returns:
            Dictionary with ranked ids and documents
        """
        ranks = self.cross_encoder.rank(query, documents)
        ranked_ids = [ids[row["corpus_id"]] for row in ranks]
        ranked_docs = [documents[row["corpus_id"]] for row in ranks]

        return {
            "ids": ranked_ids[:top_k],
            "documents": ranked_docs[:top_k],
        }

    @modal.asgi_app()
    def fastapi_server(self):
        """Create and configure the FastAPI application."""
        from fastapi import FastAPI
        from pydantic import BaseModel

        server = FastAPI(title="Presidents RAG Vector Store API")

        class QueryRequest(BaseModel):
            query: str
            n_results: int

        class QueryResponse(BaseModel):
            ids: list[str]
            texts: list[str]
            distances: list[float]

        class RerankRequest(BaseModel):
            query: str
            ids: list[str]
            documents: list[str]
            top_k: int

        class RerankResponse(BaseModel):
            ids: list[str]
            documents: list[str]

        @server.post("/query", response_model=QueryResponse)
        def query_endpoint(request: QueryRequest) -> QueryResponse:
            """
            Query the vector store for similar documents.

            Args:
                request: Query request with query string and number of results

            Returns:
                Query response with ids, texts, and distances
            """
            result = self.query(
                query=request.query,
                n_results=request.n_results,
            )
            return QueryResponse.model_validate(result)

        @server.post("/rerank", response_model=RerankResponse)
        def rerank_endpoint(request: RerankRequest) -> RerankResponse:
            """
            Rerank documents using cross encoder.

            Args:
                request: Rerank request with query, documents, and top_k

            Returns:
                Rerank response with ranked ids and documents
            """
            result = self.rerank(
                query=request.query,
                ids=request.ids,
                documents=request.documents,
                top_k=request.top_k,
            )
            return RerankResponse.model_validate(result)

        @server.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy"}

        return server
