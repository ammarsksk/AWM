# Workflow Memory Backend Benchmark

Same workload for both backends: same exemplar slice, same task/step limits, same abstraction mode.

| Metric | RAM FAISS/BM25 | LanceDB Disk ANN | Memmap Disk Exact |
| --- | ---: | ---: | ---: |
| Tasks | 0 | 0 | 500 |
| Steps | 0 | 0 | 1000 |
| Accepted Retrievals | 0 | 0 | 1000 |
| Total Runtime ms | 0.00 | 0.00 | 134237.64 |
| Online Runtime ms | 0.00 | 0.00 | 119319.85 |
| Prep ms | 0.00 | 0.00 | 3338.84 |
| Batch Add ms | 0.00 | 0.00 | 11578.20 |
| Final Index ms | 0.00 | 0.00 | 0.00 |
| Avg Retrieval ms | 0.00 | 0.00 | 118.74 |
| P95 Retrieval ms | 0.00 | 0.00 | 130.99 |
| Avg Add/Store ms | 0.00 | 0.00 | 23.16 |
| P95 Add/Store ms | 0.00 | 0.00 | 23.16 |
| Start RSS MB | 0.00 | 0.00 | 870.41 |
| End RSS MB | 0.00 | 0.00 | 1018.12 |
| Peak Sampled RSS MB | 0.00 | 0.00 | 1018.12 |
| RSS Growth MB | 0.00 | 0.00 | 147.71 |
| Python Vector MB | 0.0000 | 0.0000 | 0.0000 |
| Disk Bytes | 0 | 0 | 13475445 |

## Interpretation

- RAM FAISS/BM25 keeps all workflow vectors and embedding text in Python memory.
- LanceDB keeps vector-search storage in the disk-backed Lance table and records `python_vector_mb` as zero for the workflow collection.
- Memmap keeps vectors in a disk-backed NumPy memmap and avoids database-engine overhead; it uses exact chunked vector search.
- LanceDB online writes append to the table; ANN refresh can be deferred to the final/background index phase.
- LanceDB may show higher fixed RSS on tiny runs because the database engine has startup/index overhead; the scaling benefit appears as workflow count grows.
- Retrieval latency includes query embedding and backend search/reranking.
