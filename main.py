import os

import chromadb
import jinja2
from openai import OpenAI
from sentence_transformers import CrossEncoder

from create_vector_store import VECTOR_STORE_NAME, VECTOR_STORE_PATH
from knowledge_base import KnowledgeBase

API_KEY = os.environ["OPENAI_API_KEY__PRESIDENTS_RAG"]
OPENAI_MODEL = "gpt-4o-mini"
PROMPT_SKELETON = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use 3 sentences maximum and keep the answer concise.
Question: {{ query }}

Context:
{% for doc in documents %}
DOCUMENT {{ loop.index }}
{{ doc }}
{% endfor %}

Answer:
"""


class PresidentsRAG:
    def __init__(self) -> None:
        client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
        self.vector_store = client.get_collection(name=VECTOR_STORE_NAME)
        self.prompt_template = jinja2.Template(PROMPT_SKELETON)
        self.openai_client = OpenAI(api_key=API_KEY)

    def ping_openai(self, prompt: str) -> str:
        completion = self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant with close attention to detail.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return completion.choices[0].message.content

    def retrieve_documents(
        self,
        query: str,
        top_k: int,
    ) -> list[str]:
        result = self.vector_store.query(query_texts=query, n_results=top_k)
        documents = result["documents"][0]
        ids = result["ids"][0]
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
