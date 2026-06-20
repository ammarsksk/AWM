# Workflow Memory Backend Benchmark

Same workload for both backends: same exemplar slice, same task/step limits, same abstraction mode.

| Metric | RAM FAISS/BM25 | LanceDB Disk ANN | Memmap Disk Exact | Compressed FAISS |
| --- | ---: | ---: | ---: | ---: |
| Tasks | 500 | 0 | 0 | 500 |
| Steps | 1000 | 0 | 0 | 1000 |
| Accepted Retrievals | 990 | 0 | 0 | 998 |
| Total Runtime ms | 72569.91 | 0.00 | 0.00 | 39677.77 |
| Online Runtime ms | 40883.04 | 0.00 | 0.00 | 26477.92 |
| Prep ms | 3068.91 | 0.00 | 0.00 | 3099.21 |
| Batch Add ms | 28617.44 | 0.00 | 0.00 | 9808.44 |
| Final Index ms | 0.00 | 0.00 | 0.00 | 290.35 |
| Avg Retrieval ms | 40.50 | 0.00 | 0.00 | 26.16 |
| P95 Retrieval ms | 64.29 | 0.00 | 0.00 | 38.65 |
| Avg Add/Store ms | 57.23 | 0.00 | 0.00 | 19.62 |
| P95 Add/Store ms | 57.23 | 0.00 | 0.00 | 19.62 |
| Start RSS MB | 870.67 | 0.00 | 0.00 | 871.27 |
| End RSS MB | 1037.98 | 0.00 | 0.00 | 1026.78 |
| Peak Sampled RSS MB | 1037.98 | 0.00 | 0.00 | 1026.78 |
| RSS Growth MB | 167.31 | 0.00 | 0.00 | 155.51 |
| Python Vector MB | 0.7324 | 0.0000 | 0.0000 | 0.0000 |
| Disk Bytes | 16780891 | 0 | 0 | 13233091 |

## Interpretation

- RAM FAISS/BM25 keeps all workflow vectors and embedding text in Python memory.
- LanceDB keeps vector-search storage in the disk-backed Lance table and records `python_vector_mb` as zero for the workflow collection.
- Memmap keeps vectors in a disk-backed NumPy memmap and avoids database-engine overhead; it uses exact chunked vector search.
- Compressed FAISS keeps quantized FAISS codes instead of Python vector lists, preserving FAISS-speed retrieval with lower vector memory.
- LanceDB online writes append to the table; ANN refresh can be deferred to the final/background index phase.
- LanceDB may show higher fixed RSS on tiny runs because the database engine has startup/index overhead; the scaling benefit appears as workflow count grows.
- Retrieval latency includes query embedding and backend search/reranking.
