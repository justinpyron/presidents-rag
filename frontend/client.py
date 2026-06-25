import os

import httpx
import jinja2
from dotenv import load_dotenv
from openai import OpenAI

from backend.schemas import RetrievedChunk

load_dotenv()

API_KEY = os.environ["OPENAI_API_KEY"]
SERVER_URL = os.environ["SERVER_URL"]
VECTOR_STORE_CONFIG_ID = int(os.environ["VECTOR_STORE_CONFIG_ID"])
OPENAI_MODEL = "gpt-5-mini-2025-08-07"
PROMPT_TEMPLATE = """
# ROLE
You are a skilled question-answering assistant.

# INSTRUCTIONS
- Answer the question below using the retrieved context between <context> tags.
- Return only the answer, no other text.
- The first sentence should be a concise, one-sentence answer to the question. Then, on a new line, follow up with another sentence or two to provide a richer response.

# GUIDANCE WHEN THE ANSWER IS NOT CLEAR
If you don't know the answer, just say that you don't know.
However, don't be overly cautious or too literal.
Your answer must be grounded in the context, but don't be too literal or fastidious.
If you can reasonably surmise the answer to the question from the context, provide an answer, then simply state your uncertaintly to the user transparently.
Above all, reponse to questions the way a common-sense, reasonable reader would.

# QUESTION
{{ query }}

# CONTEXT
<context>
{% for doc in documents %}
DOCUMENT {{ loop.index }}
{{ doc }}
{% endfor %}
</context>
"""


class RAGClient:
    def __init__(self) -> None:
        self.http_client = httpx.Client(base_url=SERVER_URL, timeout=30.0)
        self.prompt_template = jinja2.Template(PROMPT_TEMPLATE)
        self.openai_client = OpenAI(api_key=API_KEY)

    def retrieve(
        self,
        query: str,
        top_k: int,
        vector_store_config_id: int,
        source: str | None = None,
    ) -> list[RetrievedChunk]:
        response = self.http_client.post(
            "/retrieve",
            json={
                "query": query,
                "vector_store_config_id": vector_store_config_id,
                "top_k": top_k,
                "source": source,
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

    def ping_openai(self, prompt: str) -> str:
        response = self.openai_client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
            reasoning={"effort": "minimal"},
        )
        return response.output[1].content[0].text

    def ask(
        self,
        query: str,
        vector_store_config_id: int = VECTOR_STORE_CONFIG_ID,
        source: str | None = None,
        top_k_retrieval: int = 100,
        top_k_rerank: int = 20,
    ) -> tuple[str, list[RetrievedChunk]]:
        chunks = self.retrieve(
            query=query,
            top_k=top_k_retrieval,
            vector_store_config_id=vector_store_config_id,
            source=source,
        )
        top_chunks = self.rerank(query, chunks, top_k_rerank)
        prompt = self.prompt_template.render(
            query=query,
            documents=[chunk.text for chunk in top_chunks],
        )
        answer = self.ping_openai(prompt)
        return answer, top_chunks
