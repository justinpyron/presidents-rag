import streamlit as st

from main import PresidentsRAG


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
            f"## Document {i + 1}\n"
            f"#### `{filename}` (cosine similarity = {similarity:.3f})\n"
            f"{text}\n\n"
        )
    return sources


st.set_page_config(page_title="Presidents RAG", layout="centered", page_icon="🇺🇸")
rag = load_rag()
st.title("Presidents RAG")
with st.expander("How it works"):
    st.markdown("TODO")
query = st.text_area("Ask a question", "")
# TODO: slider for minimum cosine similarity, in [0,1]
# TODO: slider for number of documents to consider
if st.button("Submit", type="primary", use_container_width=True):
    knowledge, answer = rag.ask(query)
    st.write(answer)
    with st.expander("Sources"):
        sources = build_sources_string(
            knowledge["filename"], knowledge["text"], knowledge["cosine_similarity"]
        )
        st.markdown(sources)
