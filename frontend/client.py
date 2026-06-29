import os

import httpx
from dotenv import load_dotenv

from backend.schemas import RetrievedChunk

load_dotenv()

SERVER_URL = os.environ["SERVER_URL"]


class RAGClient:
    def __init__(self) -> None:
        self.http_client = httpx.Client(base_url=SERVER_URL, timeout=60.0)

    def is_healthy(self, timeout: float = 4.0) -> bool:
        """Whether the server's ``/health`` endpoint responds OK.

        Drives the warm-up indicator. A cold-starting server holds the request
        until its container is ready, so a timeout here means "still warming",
        not necessarily "down" — callers distinguish a genuine outage by how
        long the failures persist.
        """
        try:
            response = self.http_client.get("/health", timeout=timeout)
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def retrieve(
        self,
        query: str,
        top_k: int,
        sources: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        response = self.http_client.post(
            "/retrieve",
            json={
                "query": query,
                "top_k": top_k,
                "sources": sources,
            },
        )
        response.raise_for_status()
        return [
            RetrievedChunk.model_validate(chunk) for chunk in response.json()
        ]

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        response = self.http_client.post(
            "/rerank",
            json={
                "query": query,
                "chunks": [chunk.model_dump() for chunk in chunks],
                "top_k": top_k,
            },
        )
        response.raise_for_status()
        return [
            RetrievedChunk.model_validate(chunk) for chunk in response.json()
        ]
