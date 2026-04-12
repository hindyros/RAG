# RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) backend built with FastAPI and Mistral AI.

## System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Lovable UI                                 │
└────────────────────────┬────────────────────────────────────────────┘
                         │ HTTP
          ┌──────────────▼──────────────┐
          │        FastAPI Backend       │
          │                              │
          │  POST /ingest   POST /query  │
          │  GET  /health               │
          └──────┬───────────┬──────────┘
                 │           │
    ┌────────────▼──┐   ┌────▼──────────────────────────────────┐
    │  Ingestion     │   │            Query Pipeline              │
    │  Pipeline      │   │                                        │
    │                │   │  1. Intent Detection                   │
    │  PDF Extract   │   │     ├─ conversational → direct reply  │
    │  ↓             │   │     └─ knowledge_seeking → retrieval   │
    │  Semantic      │   │                                        │
    │  Chunking      │   │  2. HyDE Query Transformation          │
    │  ↓             │   │     Generate hypothetical doc → embed  │
    │  Batch Embed   │   │                                        │
    │  (mistral-     │   │  3. Hybrid Retrieval                   │
    │   embed)       │   │     ├─ Dense: cosine similarity        │
    │  ↓             │   │     ├─ Sparse: BM25                    │
    │  Vector Store  │   │     └─ RRF fusion                      │
    └────────────────┘   │                                        │
                         │  4. LLM Reranking                      │
    ┌────────────────┐   │     Score each candidate 0–10          │
    │  Vector Store  │◄──┤                                        │
    │                │   │  5. Similarity Threshold Check         │
    │  Cosine Index  │   │     Refuse if best score < 0.35        │
    │  (numpy mat.)  │   │                                        │
    │  BM25 Index    │   │  6. Answer Shaping                     │
    │  Metadata      │   │     factual / list / comparison / table│
    │  ──────────── │   │                                        │
    │  Persist:      │   │  7. Generation (mistral-large)         │
    │  .npz + .json  │   │     Answer + inline [N] citations      │
    └────────────────┘   └────────────────────────────────────────┘
```

## Key Features

| Feature | Implementation |
|---|---|
| PDF extraction | PyMuPDF — font metadata for header detection |
| Chunking | 3-pass semantic (header → sentence boundary → overlap) |
| Embeddings | Mistral `mistral-embed` (1024-dim) |
| Dense retrieval | Cosine similarity — numpy matrix multiply |
| Sparse retrieval | BM25Okapi — from scratch, no external library |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Query expansion | HyDE (Hypothetical Document Embeddings) |
| Reranking | LLM-based pointwise scoring via Mistral |
| Vector store | In-memory numpy + atomic JSON/npz persistence |
| Intent detection | Mistral JSON mode — 2-level classification |
| Answer shaping | 4 templates: factual / list / comparison / table |
| Citation gate | Cosine threshold refusal ("insufficient evidence") |

## Project Structure

```
app/
├── main.py              # FastAPI factory + lifespan
├── config.py            # All settings (env vars)
├── dependencies.py      # FastAPI Depends() providers
├── api/
│   ├── schemas.py       # Pydantic request/response models
│   └── routes/
│       ├── ingest.py    # POST /ingest
│       ├── query.py     # POST /query
│       └── health.py    # GET /health
├── ingestion/
│   ├── pdf_extractor.py # PyMuPDF wrapper
│   ├── chunker.py       # 3-pass semantic chunker
│   └── pipeline.py      # Ingest orchestrator
├── retrieval/
│   ├── bm25.py          # BM25Okapi from scratch
│   ├── cosine.py        # Cosine index (numpy)
│   ├── hybrid.py        # RRF fusion
│   └── reranker.py      # LLM-based reranker
├── store/
│   ├── vector_store.py  # In-memory store + index wiring
│   └── persistence.py   # Atomic .npz + .json save/load
├── llm/
│   ├── client.py        # Async Mistral API client
│   └── prompts/         # System prompts for each component
│       ├── intent.py
│       ├── hyde.py
│       ├── rerank.py
│       └── answer/      # factual / list / comparison / table
├── intent/
│   └── detector.py      # Intent classification
└── query/
    ├── hyde.py          # HyDE transformer
    └── pipeline.py      # Query orchestrator
```

## How to Run

### 1. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env and set MISTRAL_API_KEY
```

### 3. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

The API is now available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

### 4. Ingest PDFs

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

### 5. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings of the paper?"}'
```

### 6. Run tests

```bash
pytest tests/ -v
```

## API Reference

### `POST /ingest`

**Request:** `multipart/form-data` with one or more `files` fields (PDF only, max 50MB each).

**Response:**
```json
{
  "message": "Processed 2 file(s), added 47 chunks.",
  "files_processed": 2,
  "chunks_added": 47,
  "total_chunks_in_store": 47
}
```

### `POST /query`

**Request:**
```json
{ "question": "What is the transformer architecture?" }
```

**Response (success):**
```json
{
  "answer": "The transformer architecture [1] uses self-attention...",
  "citations": [
    {
      "index": 1,
      "source_file": "attention_paper.pdf",
      "page_number": 3,
      "section_header": "Model Architecture",
      "chunk_index": 12,
      "similarity_score": 0.82,
      "excerpt": "The Transformer follows an encoder-decoder structure..."
    }
  ],
  "intent": {
    "primary": "knowledge_seeking",
    "sub": "factual",
    "confidence": 0.97,
    "reasoning": "Asks for a definition of a specific concept."
  },
  "refused": false,
  "refusal_reason": null
}
```

**Response (refusal):**
```json
{
  "answer": "The evidence I found does not meet the required confidence threshold...",
  "citations": [],
  "refused": true,
  "refusal_reason": "insufficient_evidence"
}
```

### `GET /health`

```json
{ "status": "ok", "total_chunks": 47, "total_documents": 2 }
```

## Configuration Reference

All settings are read from `.env` (see `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `MISTRAL_API_KEY` | required | Mistral AI API key |
| `MISTRAL_EMBED_MODEL` | `mistral-embed` | Embedding model |
| `MISTRAL_CHAT_MODEL` | `mistral-large-latest` | Chat model |
| `STORE_PATH` | `./data/store` | Persistence directory |
| `CHUNK_SIZE_TOKENS` | `512` | Target chunk size |
| `CHUNK_OVERLAP_TOKENS` | `64` | Overlap between chunks |
| `TOP_K_RETRIEVAL` | `10` | Candidates per retriever |
| `TOP_K_RERANK` | `5` | Chunks passed to the LLM |
| `RRF_K_CONSTANT` | `60` | RRF denominator constant |
| `MIN_SIMILARITY_THRESHOLD` | `0.35` | Refusal threshold |
| `INTENT_CONFIDENCE_MIN` | `0.60` | Min confidence for sub-intent |
| `MAX_EMBED_BATCH_SIZE` | `32` | Texts per embed API call |

## Libraries Used

| Library | Purpose | Link |
|---|---|---|
| FastAPI | Web framework | https://fastapi.tiangolo.com |
| Uvicorn | ASGI server | https://www.uvicorn.org |
| PyMuPDF | PDF extraction | https://pymupdf.readthedocs.io |
| NumPy | Vector math, BM25 | https://numpy.org |
| httpx | Async HTTP client | https://www.python-httpx.org |
| pydantic-settings | Configuration | https://docs.pydantic.dev/latest/concepts/pydantic_settings/ |
| python-dotenv | .env loading | https://github.com/theskumar/python-dotenv |
| python-multipart | File uploads | https://github.com/Kludex/python-multipart |

**No external vector database.  No external search library.  All retrieval logic is implemented from scratch using NumPy.**

## Design Decisions

See [DECISIONS.md](DECISIONS.md) for the reasoning behind every significant technical choice.
