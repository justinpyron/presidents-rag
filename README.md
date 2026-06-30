# presidents-rag
**Presidential Archive** is an *agentic* Retrieval-Augmented Generation (RAG) system that answers
questions about U.S. presidents. Rather than running a fixed
`retrieve → rerank → generate` pipeline, an AI agent composes its own searches over a knowledge
base, reads what it finds, and writes an answer you can trace back to its sources.

The agent (built on [Pydantic AI](https://ai.pydantic.dev/)) is given a single `search_knowledge_base`
tool and decides *when* to search, *what* to search for, and *how many* searches to issue — searching
step by step for multi-hop questions and per-fact for comparisons. Its structured output cites the
specific chunks that support the answer, so citations can't drift from the source text.

# Architecture
The system is split into three independently deployed tiers:

```
┌──────────────────────┐     HTTP      ┌──────────────────────┐     SQL      ┌──────────────────────┐
│   Dash chat app      │ ───────────▶  │  Inference server    │ ──────────▶  │  Postgres + pgvector │
│   (Cloud Run)        │  retrieve/    │  (Modal, GPU)        │  vector      │  (Neon)              │
│                      │  rerank       │  embedder +          │  search      │  embedded chunks     │
│  + Pydantic AI agent │ ◀───────────  │  cross-encoder       │ ◀──────────  │                      │
└──────────┬───────────┘     chunks    └──────────────────────┘              └──────────────────────┘
           │
           ▼  generation
   OpenAI / Anthropic / Google
```

1. **Frontend** ([`frontend/`](frontend/)) — a [Dash](https://dash.plotly.com/) chat UI plus the
   agentic loop. The agent runs in the web process, calls the inference server over HTTP for
   retrieval/reranking, and calls a model provider directly for generation. Deployed to Cloud Run.
2. **Inference server** ([`backend/`](backend/)) — a FastAPI app on [Modal](https://modal.com/) that
   hosts the embedding and cross-encoder models on a GPU and exposes `/retrieve`, `/rerank`, and
   `/health`.
3. **Vector store** ([`db/`](db/)) — a Postgres database (Neon) with the
   [`pgvector`](https://github.com/pgvector/pgvector) extension storing embedded document chunks.
   Schema is managed with [Alembic](https://alembic.sqlalchemy.org/).

# Project Organization
```
├── README.md                  <- Overview
├── frontend/
│   ├── agent.py               <- Agentic RAG loop (Pydantic AI agent + retrieval tool)
│   ├── client.py              <- HTTP client for the inference server (retrieve/rerank/health)
│   └── dash_app/              <- Dash chat UI (layout, components, callbacks, services)
├── backend/
│   ├── server.py              <- Modal + FastAPI inference server (model loading, HTTP wiring)
│   ├── retrieval.py           <- Pure retrieve/rerank logic (shared by server and evals)
│   └── schemas.py             <- Request/response models
├── db/
│   ├── models.py              <- SQLAlchemy models (vector store config + embedded chunks)
│   └── session.py             <- Database session factory
├── alembic/                   <- Database migrations
├── scripts/
│   ├── scrape_wikipedia.py    <- Scrape Wikipedia articles
│   ├── scrape_miller_center.py<- Scrape Miller Center essays
│   ├── ingest.py              <- Chunk, embed, and load documents into the vector store
│   ├── download_weights.py    <- Download model weights locally
│   └── run_agent.py           <- Run the agent interactively from the terminal
├── evals/                     <- Retrieval and generation evals (pydantic-evals)
├── text/                      <- Scraped source articles (wikipedia/, miller_center/)
├── Dockerfile                 <- Container image for the Dash app (Cloud Run)
├── pyproject.toml             <- Project + dependency config (uv)
├── uv.lock                    <- Locked dependencies
└── .pre-commit-config.yaml    <- Linting configs
```

# Installation
This project uses [uv](https://docs.astral.sh/uv/) to manage its Python environment.

1. Install uv
```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies (the `dev` group includes the ingest/eval/migration tooling)
```
uv sync --group dev
```

3. Configure environment variables in a `.env` file:

| Variable | Used for |
| --- | --- |
| `SERVER_URL` | Base URL of the inference server |
| `OPENAI_API_KEY` | Default generation model (required) |
| `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | Claude / Gemini options in the model picker |
| `DATABASE_URL_POOLED` | Pooled Postgres connection (runtime queries) |
| `DATABASE_URL_DIRECT` | Direct Postgres connection (Alembic migrations) |
| `LOGFIRE_TOKEN` | Optional — exports agent traces to Logfire when set |

# Usage
### Run the chat app locally
```
uv run python -m frontend.dash_app.main
```
Then open http://127.0.0.1:8050. The app talks to the inference server at `SERVER_URL`, so a server
must be reachable (deploy one with `uv run modal deploy backend/server.py`).

### Try the agent in the terminal
```
uv run python scripts/run_agent.py
```

### Build the knowledge base
Scrape sources into `text/`, then chunk, embed, and load them into the vector store:
```
uv run python scripts/scrape_wikipedia.py
uv run python scripts/scrape_miller_center.py
uv run python scripts/ingest.py --model sentence-transformers/all-MiniLM-L6-v2 \
    --chunk-size 1000 --chunk-overlap 200
```

# How it works
### Knowledge base
Answers are grounded in articles about every U.S. president and secretary of state, drawn from two
source collections that can be toggled in the UI:
- **Wikipedia** — crowd-sourced encyclopedia articles.
- **Miller Center** — scholarly essays from the University of Virginia.

Each article is split into overlapping chunks, embedded, and stored in Postgres with `pgvector`.

### Models
1. **Embedder** — an [SBERT Sentence Transformer](https://sbert.net/docs/sentence_transformer/usage/usage.html)
   (`all-MiniLM-L6-v2`, 384-dim) embeds documents and queries for vector search.
2. **Re-ranker** — an [SBERT Cross Encoder](https://sbert.net/docs/cross_encoder/usage/usage.html)
   (`ms-marco-MiniLM-L-6-v2`) reorders retrieved chunks by query relevance.
3. **Agent** — a Pydantic AI agent drives the conversation and generation. It defaults to OpenAI, with
   Anthropic and Google models selectable in the UI.

### Agentic workflow
**Offline (ingestion):** scrape articles → chunk them → embed each chunk → load into the vector store.

**At inference time**, the agent runs a tool-calling loop:
1. The agent reads the question and decides whether (and how) to search.
2. Each `search_knowledge_base` call embeds the query, retrieves the nearest chunks by cosine
   similarity (`/retrieve`), and refines them with the cross-encoder re-ranker (`/rerank`).
3. The agent inspects the returned chunks and may search again — rewriting its query, following a
   multi-hop chain, or gathering independent facts — up to a request limit.
4. When it has enough evidence, it writes an answer grounded strictly in the retrieved chunks and
   cites the supporting chunks by id. If the knowledge base lacks the answer, it says so instead of
   guessing.

# Deployment
A GitHub Actions workflow ([`.github/workflows/build-and-deploy.yml`](.github/workflows/build-and-deploy.yml))
runs the full pipeline on demand: apply Alembic migrations → deploy the Modal inference server → build
the Docker image and deploy the Dash app to Cloud Run.

# Evals
The [`evals/`](evals/) directory contains retrieval and generation evals built on
[pydantic-evals](https://ai.pydantic.dev/evals/), reusing the same retrieval code path as the server.
