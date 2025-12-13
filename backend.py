import os

import jinja2
from openai import OpenAI
from sentence_transformers import CrossEncoder

from vector_store import VectorStore

API_KEY = os.environ["OPENAI_API_KEY__PRESIDENTS_RAG"]
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
        self.vector_store = VectorStore()
        self.prompt_template = jinja2.Template(PROMPT_TEMPLATE)
        self.openai_client = OpenAI(api_key=API_KEY)

    def retrieve_documents(
        self,
        query: str,
        top_k: int,
    ) -> list[str]:
        result = self.vector_store.query(query=query, n_results=top_k)
        documents = result["texts"]
        ids = result["ids"]
        return ids, documents

    def rerank(
        self,
        query: str,
        ids: list[str],
        documents: list[str],
        top_k: int,
    ) -> list[str]:
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        ranks = model.rank(query, documents)
        ranked_ids = [ids[row["corpus_id"]] for row in ranks]
        ranked_docs = [documents[row["corpus_id"]] for row in ranks]
        return ranked_ids[:top_k], ranked_docs[:top_k]

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
        print(prompt)
        answer = self.ping_openai(prompt)
        return answer, top_ids, top_docs
