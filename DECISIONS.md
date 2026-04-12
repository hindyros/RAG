# Engineering Decisions

This document records the *why* behind every significant technical choice in this system.  Reading code tells you *what* was built; this file tells you *why* it was built that way.

---

## 1. PDF Extraction — PyMuPDF over pdfplumber

**Chosen:** `pymupdf` (fitz)
**Rejected:** pdfplumber, pdfminer.six

**Why:**
PyMuPDF exposes span-level font metadata (`get_text("dict")`).  Each text span carries its font size, bold/italic flags, and bounding box.  This lets the chunker detect section headers by font size comparison rather than fragile regex heuristics.  Without font metadata, you either lose header structure entirely or you write brittle patterns that break on different PDF generators.

PyMuPDF is also 5–10× faster on large documents and has a smaller memory footprint — both matter when ingesting many files in one request.

pdfplumber is excellent for table extraction but adds a Pillow dependency and is slower for pure text tasks.

---

## 2. Chunking Strategy — Semantic Three-Pass Approach

**Chosen:** Header-aware + sentence-boundary + sliding overlap (512 tokens, 64-token overlap)
**Rejected:** Fixed-size character splits, sliding window without boundary awareness

**Why:**
Fixed-size splits are the naive approach.  They routinely cut mid-sentence, which:
- Produces incoherent chunks that confuse the embedding model.
- Makes citations point to fragments that don't make sense as standalone excerpts.

The three-pass approach:
1. **Header splits** — every section header is a mandatory boundary.  A header buried at the end of a chunk would have its semantic weight averaged into an unrelated paragraph's embedding.  Making headers boundaries keeps each chunk topically focused.
2. **Sentence boundaries** — we never cut mid-sentence.  The embedding model encodes meaning from complete thoughts, not fragments.
3. **Sliding overlap** — 64 tokens (≈12.5% of chunk size) of overlap between adjacent chunks prevents the "boundary blindness" problem where the answer straddles two chunks and neither chunk has enough context to be retrieved.

**Token target (512):** Mistral-embed accepts up to 2048 tokens, but retrieval quality degrades with very long chunks because the embedding averages over too much content (multiple topics dilute the signal).  512 tokens is empirically optimal across BEIR/MTEB benchmarks.  It is also a safe margin below the 2048-token context limit, leaving room for long sentences that overshoot the target.

---

## 3. No External Search Library

**Chosen:** BM25 from scratch (numpy + stdlib), cosine similarity via numpy matmul
**Rejected:** rank-bm25, whoosh, Elasticsearch, Typesense

**Why (besides the challenge requirement):**
For a system of this scale (< 1M chunks), a pure-numpy implementation is fast enough and gives full transparency.  Adding a search library would introduce:
- A network dependency (Elasticsearch) or additional process (Typesense) that complicates deployment.
- An opaque scoring function that is harder to debug when retrieval quality drops.
- Dependency version churn and license concerns.

The numpy cosine implementation is three lines: normalize, normalize, dot product.  The BM25 implementation is ~100 lines of clearly commented arithmetic.  Any engineer on the team can read, test, and tune it without consulting external documentation.

---

## 4. No Third-Party Vector Database

**Chosen:** In-memory numpy array + JSON persistence
**Rejected:** Pinecone, Weaviate, Qdrant, ChromaDB, pgvector

**Why (besides the bonus requirement):**
A vector database adds an operational dependency (network, auth, billing, SLA) that is unnecessary at this scale.  The numpy matrix is:
- **Fast enough:** 100k chunks × 1024 dims = ~400MB float32.  A single cosine similarity pass takes < 100ms on a CPU with BLAS.
- **Simple to understand:** the data structure is a matrix.  No index configuration, no shard tuning.
- **Portable:** the store is two files (`.npz` + `.json`).  Copy them anywhere, the system picks up where it left off.

The persistence layer uses atomic `os.replace()` writes, so the store is never partially-written on crash.  This gives us the durability guarantee of a database without the operational complexity.

**When this would change:** Beyond ~1M chunks (≈4GB), numpy matmul latency exceeds acceptable query SLAs.  At that point, FAISS (IVF index) or a proper vector DB would be the right tool.

---

## 5. Hybrid Retrieval — BM25 + Dense + RRF Fusion

**Chosen:** Hybrid BM25 + cosine similarity fused with Reciprocal Rank Fusion
**Rejected:** Dense-only retrieval, BM25-only retrieval, score interpolation

**Why hybrid:**
Dense retrieval (embeddings) captures semantic similarity but misses exact keyword matches — especially for rare technical terms, proper nouns, and product names that appear infrequently in the embedding model's training data.

BM25 is the opposite: excellent for exact matches, blind to paraphrases.

Hybrid search consistently outperforms either alone on BEIR benchmarks (+3–8 nDCG@10).

**Why RRF over score interpolation:**
Score interpolation (`α × cosine + (1-α) × bm25`) requires:
1. Tuning α (another hyperparameter that needs labelled data to optimise).
2. Comparable score scales — cosine is bounded in [-1, 1]; BM25 scores are corpus-dependent and unbounded.

RRF uses rank positions, not scores.  It is scale-free and parameter-free except for the `k` constant (default 60, from the original Cormack et al., SIGIR 2009 paper, which showed k=60 is robust across a wide range of retrieval tasks without tuning).

---

## 6. HyDE Query Transformation

**Chosen:** Hypothetical Document Embeddings (HyDE, Gao et al. 2022)
**Rejected:** Raw query embedding, query expansion via synonym lookup

**Why:**
Short user queries and long document chunks have different distributional characteristics in embedding space, even when they cover the same topic.  A query like *"attention mechanism"* has a very different vector trajectory than a paragraph explaining attention mechanisms.

HyDE bridges this gap by generating a hypothetical document passage (same style/length as real chunks) and embedding *that* instead of the raw query.  The hypothetical passage's embedding lands closer to relevant real chunks in the space.

**Trade-off:** One extra LLM call per query (+0.5–1s latency).  For a demo with quality as the primary goal, this is acceptable.  BM25 still uses the original query tokens — HyDE only affects the dense retrieval vector.

---

## 7. LLM-Based Reranking

**Chosen:** Pointwise LLM scoring via Mistral (0–10 integer score)
**Rejected:** Cross-encoder model (BGE-reranker, Cohere Rerank), no reranking

**Why rerank at all:**
RRF fusion is rank-based and does not ask "does this chunk actually answer the question?".  The LLM reranker adds a cross-attention signal that considers the *specific question* against each candidate.  In practice, it moves the most directly helpful chunk to position 1, which matters because LLMs exhibit "lost in the middle" bias — they weight early context more heavily.

**Why pointwise over listwise:**
Listwise reranking puts all chunks in one prompt.  For `top_k=10` chunks of ~500 tokens each, that's ~5000 tokens of context in the reranker prompt — expensive and hitting the model's context limit.  Pointwise is N separate calls of ~600 tokens each; more expensive in API calls but more reliable.

**When this would change:** A specialised cross-encoder (e.g., BGE-reranker-v2-m3) would be faster (< 200ms for all candidates in one forward pass) and arguably more accurate.  It would add a dependency on torch/transformers, which is out of scope for this project.

---

## 8. Intent Detection — Binary + Sub-intent

**Chosen:** Two-level classification (primary: conversational/knowledge_seeking; sub: factual/list/comparison/table/chitchat) via Mistral JSON mode
**Rejected:** Rule-based keyword matching, single-level classification

**Why two levels:**
- The primary level decides *whether* to trigger retrieval.  A greeting should not hit the vector store — it wastes latency and may produce a confused response.
- The sub-intent level decides *how to format* the answer.  A list query gets a numbered list; a table query gets a markdown table.  Same content, radically different presentation.

**Why LLM classification over keyword rules:**
Keyword rules break on paraphrasing.  "Give me an enumeration of all X" and "What are the types of X?" both want a list but share no keywords.  An LLM generalises.

**Fallback strategy:**
- JSON parse failures → default to knowledge_seeking / factual (do retrieval, generic format).
- Low confidence (< 0.60) → keep primary intent but collapse sub-intent to factual.

This errs on the side of doing retrieval.  A false negative (skipping retrieval for a real question) returns no answer.  A false positive (doing retrieval for a greeting) costs one extra API call — acceptable.

---

## 9. Citation Quality Gate

**Chosen:** Minimum cosine similarity threshold (default 0.35) applied after reranking
**Rejected:** Always answer, keyword-based confidence, perplexity filtering

**Why a hard threshold:**
Without a quality gate, a RAG system will generate plausible-sounding answers for queries it cannot actually answer — this is the hallucination failure mode.  A hard threshold forces the system to say "I don't know" rather than fabricate.

**Why cosine similarity (not RRF score):**
RRF scores are rank-based and not comparable to an absolute threshold.  Cosine similarity is bounded in [0, 1] (for positive embeddings) and has an interpretable geometric meaning: the angle between the query and document vectors.

**Why 0.35:**
This is a starting point, not a magic number.  It should be calibrated on domain-specific data:
- Too high (e.g., 0.7) → too many false refusals for legitimate queries.
- Too low (e.g., 0.1) → allows low-quality matches through, defeating the purpose.

The threshold is configurable via `MIN_SIMILARITY_THRESHOLD` in `.env`.

**Where the threshold is applied:**
After reranking, on the best cosine score among all reranked chunks (not just rank-0).  This prevents a chunk promoted by BM25 from bypassing the quality gate because it scored well on keywords but poorly on semantics.

---

## 10. Answer Shaping via Template Dispatch

**Chosen:** `TEMPLATE_MAP` dict mapping sub-intent → prompt builder function
**Rejected:** Single universal prompt with format instructions, hard-coded if/else chain

**Why separate templates:**
A single prompt that says "if the question is about a list, format as a list; if it's a comparison, use headers..." produces inconsistent output — the model sometimes follows the instructions, sometimes not.  Separate system prompts that *only* describe the expected format for one intent are more reliable because the model has fewer competing instructions.

**Why a dict map:**
A dict (vs. an if/else chain) makes the template registry visible at a glance, easy to extend (add a new key), and testable (assert the dict contains the expected keys).

---

## 11. Atomic Disk Writes

**Chosen:** Write to temp file → `os.replace()` to final path
**Rejected:** Direct write to final path, SQLite (for metadata)

**Why atomic writes:**
A direct write to the final path leaves a window where the file exists but is partially written.  If the process crashes mid-write (OOM, power loss, OS kill), the next startup reads a corrupt file.  `os.replace()` is atomic on POSIX — the file either contains the complete new content or the complete old content, never a partial state.

**Why not SQLite:**
SQLite would be a reasonable choice for the metadata (it has atomic transactions built in).  We avoided it to keep the dependency footprint minimal and to keep the metadata human-readable (JSON) without requiring a query tool to inspect it.

---

## 12. Dependency Injection via FastAPI `Depends()`

**Chosen:** Singletons on `app.state`, accessed via `Depends()` functions
**Rejected:** Global module-level instances, thread-local storage

**Why:**
Global instances are invisible dependencies — any function anywhere can import and use them, making the call graph opaque.  `Depends()` makes dependencies explicit in function signatures and makes testing straightforward: override `app.state.vector_store` with a mock store, and all routes that depend on it automatically receive the mock.
