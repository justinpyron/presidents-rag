import os

import httpx
import jinja2
from openai import OpenAI

API_KEY = os.environ["OPENAI_API_KEY__PRESIDENTS_RAG"]
MODAL_APP_URL = os.environ["MODAL_APP_URL"]
OPENAI_MODEL = "gpt-5-mini-2025-08-07"
PROMPT_TEMPLATE = """
# ROLE
You are a skilled question-answering assistant.

# INSTRUCTIONS
Answer the question below using the retrieved context between <context> tags.
If you don't know the answer, just say that you don't know.
Keep the answer concise: use 3 sentences maximum.
Return only the answer, no other text.

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


class PresidentsRAG:
    def __init__(self) -> None:
        self.http_client = httpx.Client(base_url=MODAL_APP_URL, timeout=30.0)
        self.prompt_template = jinja2.Template(PROMPT_TEMPLATE)
        self.openai_client = OpenAI(api_key=API_KEY)

    def retrieve_documents(
        self,
        query: str,
        top_k: int,
    ) -> tuple[list[str], list[str]]:
        response = self.http_client.post(
            "/query",
            json={"query": query, "n_results": top_k},
        )
        response.raise_for_status()
        result = response.json()
        documents = result["texts"]
        ids = result["ids"]
        return ids, documents

    def rerank(
        self,
        query: str,
        ids: list[str],
        documents: list[str],
        top_k: int,
    ) -> tuple[list[str], list[str]]:
        response = self.http_client.post(
            "/rerank",
            json={
                "query": query,
                "ids": ids,
                "documents": documents,
                "top_k": top_k,
            },
        )
        response.raise_for_status()
        result = response.json()
        return result["ids"], result["documents"]

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
        top_k_retrieval: int = 100,
        top_k_rerank: int = 20,
    ) -> tuple[dict, str]:
        ids, docs = self.retrieve_documents(query, top_k_retrieval)
        top_ids, top_docs = self.rerank(query, ids, docs, top_k_rerank)
        prompt = self.prompt_template.render(query=query, documents=top_docs)
        answer = self.ping_openai(prompt)
        return answer, top_ids, top_docs
