# RAG Pipeline

A production-grade Retrieval-Augmented Generation (RAG) backend built with FastAPI. Supports **OpenAI** (default) and **Mistral AI** — switch via `LLM_PROVIDER` in `.env`.

**Demo:** https://www.loom.com/share/b3d5f9d581b84005baf36c1fe8148d1f

## System Design

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Built-in Dark UI (/ static)                     │
│          Chat · Document Manager · Semantic Map                     │
└────────────────────────┬────────────────────────────────────────────┘
                         │ HTTP
          ┌──────────────▼──────────────┐
          │        FastAPI Backend       │
          │                              │
          │  POST /ingest               │
          │  POST /query                │
          │  GET  /documents            │
          │  DELETE /documents/{id}     │
          │  GET  /visualize            │
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
    │  (OpenAI or    │   │  3. Hybrid Retrieval                   │
    │   Mistral)     │   │     ├─ Dense: cosine similarity        │
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
    │  Persist:      │   │  7. Generation (OpenAI or Mistral)     │
    │  .npz + .json  │   │     Answer + inline [N] citations      │
    └────────────────┘   │                                        │
                         │  8. Hallucination Detection            │
                         │     vectara/hallucination_evaluation   │
                         │     Per-sentence score (0–1)           │
                         │     overall = min(sentence scores)     │
                         └────────────────────────────────────────┘
```

## Key Features

| Feature | Implementation |
|---|---|
| PDF extraction | PyMuPDF — font metadata for header detection |
| Chunking | 3-pass semantic (header → sentence boundary → overlap) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) or Mistral `mistral-embed` (1024-dim) |
| Dense retrieval | Cosine similarity — numpy matrix multiply |
| Sparse retrieval | BM25Okapi — from scratch, no external library |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Query expansion | HyDE (Hypothetical Document Embeddings) |
| Reranking | LLM-based pointwise scoring |
| Vector store | In-memory numpy + atomic JSON/npz persistence |
| Intent detection | LLM JSON mode — 2-level classification |
| Answer shaping | 4 templates: factual / list / comparison / table |
| Citation gate | Cosine threshold refusal ("insufficient evidence") |
| Hallucination detection | `vectara/hallucination_evaluation_model` — per-sentence NLI scoring |
| Document management | List and delete ingested documents via API and UI |
| Semantic Map | Interactive 2D PCA scatter plot with k-means clusters and LLM-generated topic labels |
| Topic search & filter | Search clusters by topic; select clusters to ground queries in specific content |
| Dark UI | Single-file vanilla JS frontend served from FastAPI (`/`) |

## Project Structure

```
app/
├── main.py              # FastAPI factory + lifespan
├── config.py            # All settings (env vars, provider selection)
├── dependencies.py      # FastAPI Depends() providers
├── api/
│   ├── schemas.py       # Pydantic request/response models
│   └── routes/
│       ├── ingest.py    # POST /ingest
│       ├── query.py     # POST /query
│       ├── documents.py # GET /documents, DELETE /documents/{id}
│       ├── visualize.py # GET /visualize (PCA + clustering)
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
│   ├── base.py          # LLMClient protocol (provider-agnostic)
│   ├── openai_client.py # Async OpenAI API client
│   ├── client.py        # Async Mistral API client
│   └── prompts/         # System prompts for each component
│       ├── intent.py
│       ├── hyde.py
│       ├── rerank.py
│       └── answer/      # factual / list / comparison / table
├── hallucination/
│   └── checker.py       # HallucinationChecker (vectara model)
├── static/
│   └── index.html       # Dark UI — single self-contained file
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
# Set LLM_PROVIDER and the matching API key (see Configuration Reference below)
```

### 3. Start the server

```bash
uvicorn app.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`.
The UI is served at `http://localhost:8000` (open in a browser).
Interactive docs: `http://localhost:8000/docs`

> **Note:** First startup downloads the Vectara hallucination model (~300 MB) to `~/.cache/huggingface`.

> **Provider switch warning:** Changing `LLM_PROVIDER` after ingesting documents will cause dimension mismatches. Delete `./data/store` and re-ingest all documents when switching providers.

### 4. Ingest PDFs

```bash
curl -X POST http://localhost:8000/ingest \
  -F "files=@document1.pdf" \
  -F "files=@document2.pdf"
```

### 5. Query

```bash
# Basic query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'

# Query grounded in specific documents
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?", "document_ids": ["uuid-1", "uuid-2"]}'
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
  "total_chunks_in_store": 47,
  "document_ids": ["uuid-1", "uuid-2"]
}
```

### `POST /query`

**Request:**
```json
{
  "question": "What is the transformer architecture?",
  "document_ids": ["uuid-1"]
}
```
`document_ids` is optional. When provided, retrieval is restricted to those documents.

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
  "grounded": true,
  "hallucination": {
    "overall_score": 0.82,
    "consistent": true,
    "sentences": [
      { "sentence": "The transformer uses self-attention...", "score": 0.82, "flagged": false }
    ]
  }
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

### `GET /documents`

```json
{
  "documents": [
    { "document_id": "uuid-1", "source_file": "paper.pdf", "chunk_count": 23, "page_count": 8 }
  ],
  "total_documents": 1,
  "total_chunks": 23
}
```

### `DELETE /documents/{document_id}`

```json
{ "message": "Deleted ...", "chunks_removed": 23, "total_chunks_in_store": 0 }
```

### `GET /visualize`

Returns 2D PCA projection of all chunk embeddings with k-means cluster labels.

```json
{
  "total_chunks": 41,
  "points": [
    {
      "chunk_index": 0, "x": 1.23, "y": -0.45, "cluster_id": 0,
      "document_id": "uuid-1", "source_file": "paper.pdf",
      "page_number": 2, "section_header": "Introduction",
      "excerpt": "This paper presents..."
    }
  ],
  "document_ids": ["uuid-1", "uuid-2"],
  "clusters": [
    { "cluster_id": 0, "label": "Model Architecture", "centroid_x": 1.1, "centroid_y": -0.3, "chunk_count": 18 }
  ]
}
```

Response is cached by `total_chunks` count and invalidated automatically on ingest or delete.

### `GET /health`

```json
{ "status": "ok", "total_chunks": 47, "total_documents": 2 }
```

## Configuration Reference

All settings are read from `.env`:

| Variable | Default | Description |
|---|---|---|
| `LLM_PROVIDER` | `openai` | Provider: `openai` or `mistral` |
| `OPENAI_API_KEY` | required if OpenAI | OpenAI API key |
| `OPENAI_EMBED_MODEL` | `text-embedding-3-small` | Embedding model (1536-dim) |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Chat model |
| `MISTRAL_API_KEY` | required if Mistral | Mistral AI API key |
| `MISTRAL_EMBED_MODEL` | `mistral-embed` | Embedding model (1024-dim) |
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

| Library | Purpose |
|---|---|
| FastAPI | Web framework |
| Uvicorn | ASGI server |
| PyMuPDF | PDF extraction |
| NumPy | Vector math, BM25, PCA, k-means |
| httpx | Async HTTP client |
| pydantic-settings | Configuration |
| python-dotenv | .env loading |
| python-multipart | File uploads |
| PyTorch | Hallucination model inference |
| Transformers | Load vectara/hallucination_evaluation_model |
| sentencepiece | Tokenizer for the hallucination model |

**No external vector database. No external search library. All retrieval and visualization logic is implemented from scratch using NumPy.**

## Design Decisions

See [DECISIONS.md](DECISIONS.md) for the reasoning behind every significant technical choice.
