# Workflow Memory Backend Benchmark

Same workload for both backends: same exemplar slice, same task/step limits, same abstraction mode.

| Metric | RAM FAISS/BM25 | LanceDB Disk ANN | Memmap Disk Exact |
| --- | ---: | ---: | ---: |
| Tasks | 500 | 0 | 500 |
| Steps | 1000 | 0 | 1000 |
| Accepted Retrievals | 990 | 0 | 1000 |
| Total Runtime ms | 89381.65 | 0.00 | 129955.06 |
| Online Runtime ms | 49329.50 | 0.00 | 115436.39 |
| Prep ms | 3844.87 | 0.00 | 3841.49 |
| Batch Add ms | 36206.81 | 0.00 | 10676.66 |
| Final Index ms | 0.00 | 0.00 | 0.00 |
| Avg Retrieval ms | 48.94 | 0.00 | 114.96 |
| P95 Retrieval ms | 64.32 | 0.00 | 133.55 |
| Avg Add/Store ms | 72.41 | 0.00 | 21.35 |
| P95 Add/Store ms | 72.41 | 0.00 | 21.35 |
| Start RSS MB | 869.89 | 0.00 | 869.95 |
| End RSS MB | 1040.34 | 0.00 | 1038.57 |
| Peak Sampled RSS MB | 1040.34 | 0.00 | 1038.57 |
| RSS Growth MB | 170.45 | 0.00 | 168.62 |
| Python Vector MB | 0.7324 | 0.0000 | 0.0000 |
| Disk Bytes | 16780807 | 0 | 13052399 |

## Interpretation

- RAM FAISS/BM25 keeps all workflow vectors and embedding text in Python memory.
- LanceDB keeps vector-search storage in the disk-backed Lance table and records `python_vector_mb` as zero for the workflow collection.
- Memmap keeps vectors in a disk-backed NumPy memmap and avoids database-engine overhead; it uses exact chunked vector search.
- LanceDB online writes append to the table; ANN refresh can be deferred to the final/background index phase.
- LanceDB may show higher fixed RSS on tiny runs because the database engine has startup/index overhead; the scaling benefit appears as workflow count grows.
- Retrieval latency includes query embedding and backend search/reranking.
