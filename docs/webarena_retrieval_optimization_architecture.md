# WebArena Retrieval Optimization Architecture

## What changed

The WebArena procedural memory now has a pluggable FAISS retrieval layer:

- `flat`: exact normalized inner-product baseline.
- `hnsw`: full-vector HNSW.
- `sq8`: scalar-quantized flat index.
- `hnsw_sq8`: HNSW + scalar quantization.
- `ivfpq`: IVF + product quantization.
- `opq_ivfpq`: OPQ rotation + IVF-PQ.
- `turboquant`: experimental signed-permutation rotation + binary quantization.
- `rabitq`: experimental dense random rotation + binary quantization.
- `binary_hnsw_rotation`: binary quantization with binary HNSW.

The implementation is in:

```text
webarena/advanced_vector_index.py
```

It is wired into:

```text
webarena/procedural_memory.py
```

The retrieval-only benchmark is:

```text
webarena/benchmark_procedural_retrieval.py
```

## Final retrieval architecture

```text
WebArena task + current page observation
-> same-site procedural memory store
-> advanced vector backend
-> lexical rerank
-> procedural graph score
-> task-family score
-> action-structure score
-> outcome score
-> negative-memory penalty
-> threshold gate
-> compact memory card injected into Gemini prompt
```

SQLite remains the source of truth:

```text
procedures table
  Stores full procedure_json, compact_text, success/failure counts.

procedure_edges table
  Stores the graph: family, strategy, steps, actions, checks, outcomes.

negative_memory table
  Stores failed patterns and avoid rules.

retrieval_events table
  Stores selected/rejected candidates and score breakdowns.
```

The FAISS index is a fast sidecar. It can always be rebuilt from SQLite.

## Embedding model

The embedding model is controlled by:

```bash
AWM_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
```

Recommended experiment order:

```text
1. sentence-transformers/all-MiniLM-L6-v2
2. BAAI/bge-small-en-v1.5
3. BAAI/bge-base-en-v1.5
4. mixedbread-ai/mxbai-embed-large-v1
```

`bge-small-en-v1.5` is the best practical first upgrade: much better than MiniLM while still fast.

## Run backend benchmark

```bash
cd ~/Downloads/agent-workflow-memory-main

AWM_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5" \
./.venv/bin/python webarena/benchmark_procedural_retrieval.py \
  --memory-dir webarena/memory/procedural \
  --output-dir webarena/retrieval_backend_benchmark_bge_small \
  --backends flat,hnsw,sq8,hnsw_sq8,ivfpq,opq_ivfpq,turboquant,rabitq,binary_hnsw_rotation \
  --max-queries 250 \
  --top-k 4 \
  --candidate-k 40
```

If the BGE model is not cached locally, cache it first:

```bash
./.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5'); print('cached')"
```

## Use a backend in a WebArena run

Example with HNSW + SQ8:

```bash
WEBARENA_FAST_MEMORY=0 \
WEBARENA_VECTOR_BACKEND=hnsw_sq8 \
AWM_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5" \
WA_SHOPPING="http://18.191.180.130" \
WA_SHOPPING_ADMIN="http://18.191.180.130:8083/admin" \
WA_REDDIT="http://18.191.180.130:8080/forums/all" \
WA_GITLAB="http://18.191.180.130:9001/explore" \
WA_WIKIPEDIA="http://18.191.180.130:8081/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing" \
WA_MAP="http://18.191.180.130:443" \
WA_HOMEPAGE="http://18.191.180.130" \
WA_FULL_RESET="http://18.191.180.130:7565" \
WEBARENA_EVAL_MODEL="google/gemini-2.5-pro" \
./.venv/bin/python webarena/pipeline.py \
  --website shopping \
  --start_index 158 \
  --end_index 198 \
  --memory_architecture procedural \
  --procedural_memory_dir memory/procedural \
  --browser_proxy http://192.168.140.15:3128
```

Swap this variable to compare retrieval backends:

```bash
WEBARENA_VECTOR_BACKEND=hnsw
WEBARENA_VECTOR_BACKEND=hnsw_sq8
WEBARENA_VECTOR_BACKEND=ivfpq
WEBARENA_VECTOR_BACKEND=opq_ivfpq
WEBARENA_VECTOR_BACKEND=turboquant
WEBARENA_VECTOR_BACKEND=rabitq
WEBARENA_VECTOR_BACKEND=binary_hnsw_rotation
```

## Expected behavior

For the current tiny memory store, embedding latency dominates, so all FAISS backends look similar. The important differences show up when the memory grows:

```text
hnsw
  Fast and accurate, but stores full vectors and graph links.

hnsw_sq8
  Best default for medium scale: good speed, much lower vector memory.

ivfpq / opq_ivfpq
  Best for large scale. Requires enough memories to train; falls back to hnsw_sq8 on small stores.

turboquant
  Very compact binary backend with fast signed-permutation rotation.

rabitq
  Dense-rotation binary experimental backend. More faithful to random-rotation quantization, but slower.

binary_hnsw_rotation
  Binary compressed vectors plus HNSW graph traversal.
```

## Current recommendation

Use this for most near-term WebArena experiments:

```bash
WEBARENA_VECTOR_BACKEND=hnsw_sq8
AWM_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
```

Use this when memory grows large enough for PQ training:

```bash
WEBARENA_VECTOR_BACKEND=opq_ivfpq
AWM_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5"
```

Use this for aggressive compression experiments:

```bash
WEBARENA_VECTOR_BACKEND=turboquant
```
