# Runtime Stats

| Metric | Value |
| --- | ---: |
| Total wall time | 325.34s |
| Build/retrieval-plan time | 165.43s |
| Prediction time | 108.25s |
| Output size | 61.16 MB |
| RSS start | 20.78 MB |
| RSS after build | 1068.83 MB |
| RSS end | 1192.71 MB |

## Memory Backend

```json
{
  "architecture": "procedural",
  "workflow_abstraction": "deterministic",
  "workflow_storage": "disk",
  "top_k": 5,
  "retrieval_candidate_k": 50,
  "procedural_manifest": {
    "backend": "procedural_graph_hybrid",
    "procedure_count": 500,
    "edge_count": 33346,
    "negative_memory_count": 548,
    "embedding_backend": "sentence_transformers_faiss_bm25",
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "candidate_k": 50,
    "scoring": {
      "semantic_weight": 0.5,
      "lexical_weight": 0.14,
      "activation_weight": 0.16,
      "graph_weight": 0.08,
      "outcome_weight": 0.08,
      "metadata_weight": 0.04,
      "acceptance_threshold": 0.34,
      "activation_threshold": 0.04
    }
  },
  "compressed_snapshot": {
    "backend": "compressed_faiss",
    "workflow_count": 500,
    "python_vector_count": 0,
    "python_vector_values": 0,
    "python_vector_mb": 0.0,
    "actual_index_kind": "sq8_flat",
    "candidate_k": 50,
    "faiss_code_size_bytes": 384,
    "estimated_faiss_code_mb": 0.18310546875,
    "avg_batch_add_ms": 5695.286686066538,
    "avg_batch_index_ms": 0.40043098852038383,
    "avg_retrieve_ms": 0.0,
    "avg_text_load_ms": 0.0,
    "workflow_text_cache_items": 0,
    "workflow_text_cache_hits": 0,
    "workflow_text_cache_misses": 0
  }
}
```
