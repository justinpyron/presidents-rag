import jinja2

from knowledge_base import KnowledgeBase

prompt_skeleton = """
QUESTION:
{{ query }}

DOCUMENTS:
{% for doc in documents %}
DOCUMENT {{ loop.index }}
{{ doc }}
{% endfor %}

INSTRUCTIONS:
Answer the QUESTION above using the list of documents in DOCUMENTS. Your answer must only be based on the items in DOCUMENTS. It is imperative that your response is grounded in documents in the DOCUMENTS section. If DOCUMENTS does not contain the answer to QUESTION, return "NONE".
"""


class PresidentsRAG:
    def __init__(self) -> None:
        self.knowledge_base = KnowledgeBase()
        self.prompt_template = jinja2.Template(prompt_skeleton)

    def ping_openai(self, prompt: str) -> str:
        pass

    def ask(
        self,
        query: str,
        top_k: int = 5,
    ):
        response = self.knowledge_base.fetch_similar_documents(query, top_k)
        prompt = self.prompt_template.render(query=query, documents=response["text"])
        return prompt
