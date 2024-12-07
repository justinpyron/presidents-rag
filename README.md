# presidents-rag
Use Wikipedia knowledge base and Retrieval-Augmented Generation (RAG) to answer questions about American presidents.

# Project Organization
```
├── README.md                  <- Overview
├── main.py                    <- Main RAG logic/backend
├── app.py                     <- Streamlit web app frontend
├── people.py                  <- List of people to include in the knowledge base
├── scrape_wikipedia.py        <- Extract text from Wikipedia articles
├── chunk_text.py              <- Split raw Wikipedia articles into smaller chunks
├── knowledge_base.py          <- Class for interfacing with knowledge base of embedding vectors
├── create_artifacts.py        <- Build the knowledge base by creating embedding vectors
├── artifact_embeddings.npy    <- Knowledge base of embedding vectors
├── artifact_text.pickle       <- Raw text of knowledge base elements
├── artifact_filenames.pickle  <- Filenames of raw text of knowledge base elements
├── pyproject.toml             <- Poetry config specifying Python environment dependencies
├── poetry.lock                <- Locked dependencies to ensure consistent installs
├── .pre-commit-config.yaml    <- Linting configs
```

# Installation
This project uses [Poetry](https://python-poetry.org/docs/) to manage its Python environment.

1. Install Poetry
```
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies
```
poetry install
```

# Usage
A Streamlit web app is the frontend for interacting with the model.

The app can be accessed at https://presidents-rag.streamlit.app.

Alternatively, the app can be run locally with
```
poetry run streamlit run app.py
```

# How it works
### Knowledge Base
Answers are based on the Wikipedia articles of all US Presidents and Secretaries of State.
These articles were scraped using [Wikipedia's OpenSearch API](https://www.mediawiki.org/wiki/API:Opensearch).
The `text/` and `chunks/` directories contain the raw text files.

### Models
1. A model to create vector embeddings of documents and questions. I use a [Sentence Transformer](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
2. A generative chatbot to answer context-enriched queries. I use [OpenAI's gpt-4o-mini](https://platform.openai.com/docs/models#gpt-4o-mini).

### Workflow
In simple terms, the RAG system transforms a user's query into a prompt enriched with context from the knowledge base.

###### Offline
1. Create a knowledge base of documents.
2. Create a vector embedding of each document in the knowledge base.

###### At inference time
1. Create a vector embedding of the question you want to ask.
2. Apply cosine similarity to the knowledge base embeddings to find the documents most similar to the question.
3. Create a prompt that supplements the question with the most similar documents.
4. Submit the prompt to the generative chatbot to generate an answer.
