# Workflow Memory Backend Benchmark

Same workload for both backends: same exemplar slice, same task/step limits, same abstraction mode.

| Metric | RAM FAISS/BM25 | LanceDB Disk ANN | Memmap Disk Exact | Compressed FAISS |
| --- | ---: | ---: | ---: | ---: |
| Tasks | 500 | 0 | 0 | 500 |
| Steps | 1000 | 0 | 0 | 1000 |
| Accepted Retrievals | 990 | 0 | 0 | 1000 |
| Total Runtime ms | 76677.81 | 0.00 | 0.00 | 49018.73 |
| Online Runtime ms | 45494.32 | 0.00 | 0.00 | 35602.72 |
| Prep ms | 3208.57 | 0.00 | 0.00 | 3525.25 |
| Batch Add ms | 27974.53 | 0.00 | 0.00 | 9880.89 |
| Final Index ms | 0.00 | 0.00 | 0.00 | 8.81 |
| Avg Retrieval ms | 45.10 | 0.00 | 0.00 | 35.28 |
| P95 Retrieval ms | 67.05 | 0.00 | 0.00 | 43.54 |
| Avg Add/Store ms | 55.95 | 0.00 | 0.00 | 19.76 |
| P95 Add/Store ms | 55.95 | 0.00 | 0.00 | 19.76 |
| Start RSS MB | 870.66 | 0.00 | 0.00 | 870.87 |
| End RSS MB | 1043.32 | 0.00 | 0.00 | 1018.64 |
| Peak Sampled RSS MB | 1043.32 | 0.00 | 0.00 | 1018.64 |
| RSS Growth MB | 172.67 | 0.00 | 0.00 | 147.77 |
| Python Vector MB | 0.7324 | 0.0000 | 0.0000 | 0.0000 |
| Disk Bytes | 16780859 | 0 | 0 | 13008453 |

## Interpretation

- RAM FAISS/BM25 keeps all workflow vectors and embedding text in Python memory.
- LanceDB keeps vector-search storage in the disk-backed Lance table and records `python_vector_mb` as zero for the workflow collection.
- Memmap keeps vectors in a disk-backed NumPy memmap and avoids database-engine overhead; it uses exact chunked vector search.
- Compressed FAISS keeps quantized FAISS codes instead of Python vector lists, preserving FAISS-speed retrieval with lower vector memory.
- LanceDB online writes append to the table; ANN refresh can be deferred to the final/background index phase.
- LanceDB may show higher fixed RSS on tiny runs because the database engine has startup/index overhead; the scaling benefit appears as workflow count grows.
- Retrieval latency includes query embedding and backend search/reranking.
