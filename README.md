# presidents-rag
Use Wikipedia knowledge base and Retrieval-Augmented Generation (RAG) to answer questions about American presidents.

# Project Organization
```
├── README.md                  <- Overview
├── app.py                     <- Streamlit web app frontend
├── backend.py                 <- RAG logic used in the app
├── people.py                  <- List of people to include in the knowledge base
├── scrape_wikipedia.py        <- Extract text from Wikipedia articles
├── text/                      <- Folder with scraped wikipedia articles in .txt files
├── vector_store.py            <- Class for interfacing with vector store of embedded text chunks
├── vector_store.pickle        <- Saved vector store object
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
The `text/` directory contains the raw text files.

### Models
1. A model to create vector embeddings of documents and questions. I use an [SBERT Sentence Transformer](https://sbert.net/docs/sentence_transformer/usage/usage.html).
2. A model to compute similarity between a query and context document (a "re-ranker"). I use an [SBERT Cross Encoder](https://sbert.net/docs/cross_encoder/usage/usage.html).
3. A generative chatbot to answer context-enriched queries. I use [OpenAI's gpt-4o-mini](https://platform.openai.com/docs/models#gpt-4o-mini).

### Workflow
In simple terms, the RAG system transforms a user's query into a prompt enriched with context from the knowledge base.

###### Offline
1. Create a knowledge base of documents.
2. Create a vector embedding of each document in the knowledge base.

###### At inference time
1. Create a vector embedding of the question you want to ask.
2. Apply cosine similarity to the knowledge base embeddings to find the documents most similar to the question.
3. Refine the set of most similar documents using the re-ranker model.
4. Create a prompt that supplements the question with the most similar documents.
5. Submit the prompt to the generative chatbot to generate an answer.
