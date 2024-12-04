import os

import jinja2
from openai import OpenAI

from knowledge_base import KnowledgeBase

API_KEY = os.environ["OPENAI_API_KEY__PRESIDENTS_RAG"]
OPENAI_MODEL = "gpt-4o-mini"
PROMPT_SKELETON = """
QUESTION:
{{ query }}

DOCUMENTS:
{% for doc in documents %}
DOCUMENT {{ loop.index }}
{{ doc }}
{% endfor %}

INSTRUCTIONS:
Answer the QUESTION above using the list of documents in DOCUMENTS. Your answer must only be based on the items in DOCUMENTS. It is imperative that your answer be grounded in DOCUMENTS.
"""


class PresidentsRAG:
    def __init__(self) -> None:
        self.knowledge_base = KnowledgeBase()
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
        knowledge = self.knowledge_base.fetch_similar_documents(query, top_k)
        prompt = self.prompt_template.render(query=query, documents=knowledge["text"])
        answer = self.ping_openai(prompt)
        return knowledge, answer
