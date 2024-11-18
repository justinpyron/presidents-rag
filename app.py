import streamlit as st

from main import PresidentsRAG

how_it_works = """
## 💫 Ask a question about a President 💫

An answer based on a knowledge base of documents is provided using Retrieval Augmented Generation (RAG).

## Knowledge Base
Answers are based on the Wikipedia articles of all US Presidents and Secretaries of State.
These articles were scraped using [Wikipedia's OpenSearch API](https://www.mediawiki.org/wiki/API:Opensearch).
The `text/` and `chunks/` directories in the GitHub contain the raw text files.

## Models
1. A model to create vector embeddings of documents and questions. I use a [Sentence Transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
2. A generative chatbot to answer context-enriched queries. I use [OpenAI's gpt-4o-mini](https://platform.openai.com/docs/models#gpt-4o-mini).

## Workflow
In simple terms, the RAG system transforms a user's query into a prompt enriched with context from the knowledge base.

###### Offline
1. Create a knowledge base of documents.
2. Create a vector embedding of each document in the knowledge base.

###### At inference time
1. Create a vector embedding of the question you want to ask.
2. Apply cosine similarity to the embeddings to find the documents most similar to the question.
3. Create a prompt that supplements the question with the most similar documents.
4. Submit the prompt to the generative chatbot to generate an answer.

## Full Details
See 👉 [GitHub](https://github.com/justinpyron/presidents-rag).
"""


@st.cache_resource
def load_rag() -> PresidentsRAG:
    return PresidentsRAG()


def build_sources_string(
    filenames: list[str],
    documents: list[str],
    similarities: list[float],
) -> str:
    sources = ""
    for i, (filename, text, similarity) in enumerate(
        zip(filenames, documents, similarities)
    ):
        sources += (
            f"### Document {i + 1}\n"
            f"##### `{filename}` (cosine similarity = {similarity:.3f})\n"
            f"{text}\n\n"
        )
    return sources


st.set_page_config(page_title="Presidents RAG", layout="centered", page_icon="🇺🇸")
rag = load_rag()
st.title("US Presidents RAG 🇺🇸")
with st.expander("How it works"):
    st.markdown(how_it_works)
query = st.text_area("Ask a question", "")
if st.button("Submit", type="primary", use_container_width=True):
    knowledge, answer = rag.ask(query)
    st.write(answer)
    with st.expander("Sources"):
        sources = build_sources_string(
            knowledge["filename"],
            knowledge["text"],
            knowledge["cosine_similarity"],
        )
        st.markdown(sources)
