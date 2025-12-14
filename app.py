import streamlit as st

from backend import PresidentsRAG

how_it_works = """
This app uses **Retrieval Augmented Generation (RAG)** to answer questions using a knowledge base of Wikipedia articles about US Presidents and Secretaries of State.

### ⚙️ How RAG Works

RAG transforms your question into a context-enriched prompt by retrieving relevant documents from the knowledge base.

**Offline:**
1. Split documents into chunks and create vector embeddings for each chunk

**At query time:**
1. Create a vector embedding of your question
2. Find the most similar document chunks using cosine similarity
3. Re-rank results using a cross-encoder model
4. Send your question + top documents to a generative model for the answer

### ⚡️ Models Used
- **Embeddings:** [SBERT Sentence Transformer](https://sbert.net/docs/sentence_transformer/usage/usage.html)
- **Re-ranking:** [SBERT Cross Encoder](https://sbert.net/docs/cross_encoder/usage/usage.html)
- **Generation:** OpenAI GPT-4o mini

### 🔍 Full Details
View the [full implementation on GitHub](https://github.com/justinpyron/presidents-rag).
"""


@st.cache_resource
def load_rag() -> PresidentsRAG:
    return PresidentsRAG()


st.set_page_config(page_title="Presidents RAG", layout="centered", page_icon="🇺🇸")
rag = load_rag()
st.title("US Presidents RAG 🇺🇸")
with st.expander("How it works"):
    st.markdown(how_it_works)
query = st.text_area("Ask a question", "")
if st.button("Submit", type="primary", use_container_width=True):
    with st.spinner("Searching docs + writing answer..."):
        answer, ids, documents = rag.ask(query)
    st.write(answer)
    with st.expander("Sources"):
        pretty_docs = [
            f"### Document {i+1}\n##### `{id.split(':')[0]}` -- `starting @ character {id.split(':')[-1]}`\n{doc}"
            for i, (id, doc) in enumerate(zip(ids, documents))
        ]
        st.markdown("\n\n".join(pretty_docs))
