import os

import chromadb
import jinja2
from openai import OpenAI

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

    def ask(
        self,
        query: str,
        top_k: int = 5,
    ) -> tuple[dict, str]:
        result = self.vector_store.query(query_texts=query, n_results=top_k)
        documents = result["documents"][0]
        prompt = self.prompt_template.render(query=query, documents=documents)
        answer = self.ping_openai(prompt)
        return answer, documents
