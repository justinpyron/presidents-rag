"""Modal-based inference server for Presidents RAG.

Hosts the embedding and cross-encoder models, and exposes retrieval/reranking
over HTTP. The actual logic lives in ``backend.retrieval``; this module only
handles model loading, the DB session, and the FastAPI wiring.
"""

import os

import modal

MODAL_APP_NAME = "presidents-rag"
MODAL_VOLUME_NAME = "presidents-rag"
MOUNT_PATH = "/data"

VECTOR_STORE_CONFIG_ID = 1
SENTENCE_TRANSFORMER_PATH = "weights/sentence-transformers_all-MiniLM-L6-v2"
CROSS_ENCODER_PATH = "weights/cross-encoder_ms-marco-MiniLM-L6-v2"

# The runtime queries Postgres through the pooled connection. The value is read
# from the deploy environment (GitHub secrets in CI, .env locally) and injected
# into the container, so GitHub remains the single source of truth.
db_secret = modal.Secret.from_dict(
    {"DATABASE_URL_POOLED": os.environ["DATABASE_URL_POOLED"]}
)

app = modal.App(MODAL_APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.12.1",
        "sentence-transformers==5.6.0",
        "transformers==5.12.1",
        "pydantic==2.13.4",
        "fastapi==0.138.0",
        "sqlalchemy==2.0.51",
        "pgvector==0.4.2",
        "psycopg[binary]==3.3.4",
    )
    .add_local_python_source("backend", "db")
)
volume = modal.Volume.from_name(MODAL_VOLUME_NAME)


@app.cls(
    image=image,
    volumes={MOUNT_PATH: volume},
    secrets=[db_secret],
    gpu="T4",
    scaledown_window=600,
    max_containers=1,
)
class Server:
    """Serves vector retrieval and cross-encoder reranking."""

    @modal.enter()
    def load_models(self):
        from sentence_transformers import CrossEncoder, SentenceTransformer

        embedder_path = os.path.join(MOUNT_PATH, SENTENCE_TRANSFORMER_PATH)
        cross_encoder_path = os.path.join(MOUNT_PATH, CROSS_ENCODER_PATH)
        print(f"Loading embedder from: {embedder_path}")
        print(f"Loading cross-encoder from: {cross_encoder_path}")
        self.embedder = SentenceTransformer(embedder_path)
        self.cross_encoder = CrossEncoder(cross_encoder_path)

    @modal.asgi_app()
    def fastapi_server(self):
        from fastapi import FastAPI

        from backend.retrieval import rerank, retrieve
        from backend.schemas import (
            RerankRequest,
            RetrievedChunk,
            RetrieveRequest,
        )
        from db.session import get_session

        server = FastAPI(title="Presidents RAG API")

        @server.post("/retrieve", response_model=list[RetrievedChunk])
        def retrieve_endpoint(
            request: RetrieveRequest,
        ) -> list[RetrievedChunk]:
            session = get_session()
            try:
                return retrieve(
                    session=session,
                    embedder=self.embedder,
                    vector_store_config_id=VECTOR_STORE_CONFIG_ID,
                    query=request.query,
                    top_k=request.top_k,
                    sources=request.sources,
                )
            finally:
                session.close()

        @server.post("/rerank", response_model=list[RetrievedChunk])
        def rerank_endpoint(request: RerankRequest) -> list[RetrievedChunk]:
            return rerank(
                cross_encoder=self.cross_encoder,
                query=request.query,
                chunks=request.chunks,
                top_k=request.top_k,
            )

        @server.get("/health")
        def health_check():
            return {"status": "healthy"}

        return server
