# Workflow Memory Backend Benchmark

Same workload for both backends: same exemplar slice, same task/step limits, same abstraction mode.

| Metric | RAM FAISS/BM25 | LanceDB Disk ANN | Memmap Disk Exact | Compressed FAISS |
| --- | ---: | ---: | ---: | ---: |
| Tasks | 500 | 0 | 0 | 500 |
| Steps | 1000 | 0 | 0 | 1000 |
| Accepted Retrievals | 990 | 0 | 0 | 998 |
| Total Runtime ms | 66750.70 | 0.00 | 0.00 | 40954.87 |
| Online Runtime ms | 39156.65 | 0.00 | 0.00 | 26816.60 |
| Prep ms | 3146.30 | 0.00 | 0.00 | 3271.32 |
| Batch Add ms | 24447.38 | 0.00 | 0.00 | 9977.92 |
| Final Index ms | 0.00 | 0.00 | 0.00 | 887.47 |
| Avg Retrieval ms | 38.80 | 0.00 | 0.00 | 26.51 |
| P95 Retrieval ms | 63.01 | 0.00 | 0.00 | 35.94 |
| Avg Add/Store ms | 48.89 | 0.00 | 0.00 | 19.96 |
| P95 Add/Store ms | 48.89 | 0.00 | 0.00 | 19.96 |
| Start RSS MB | 870.35 | 0.00 | 0.00 | 870.69 |
| End RSS MB | 1040.63 | 0.00 | 0.00 | 1038.55 |
| Peak Sampled RSS MB | 1040.63 | 0.00 | 0.00 | 1038.55 |
| RSS Growth MB | 170.28 | 0.00 | 0.00 | 167.86 |
| Python Vector MB | 0.7324 | 0.0000 | 0.0000 | 0.0000 |
| Disk Bytes | 16780991 | 0 | 0 | 13184931 |

## Interpretation

- RAM FAISS/BM25 keeps all workflow vectors and embedding text in Python memory.
- LanceDB keeps vector-search storage in the disk-backed Lance table and records `python_vector_mb` as zero for the workflow collection.
- Memmap keeps vectors in a disk-backed NumPy memmap and avoids database-engine overhead; it uses exact chunked vector search.
- Compressed FAISS keeps quantized FAISS codes instead of Python vector lists, preserving FAISS-speed retrieval with lower vector memory.
- LanceDB online writes append to the table; ANN refresh can be deferred to the final/background index phase.
- LanceDB may show higher fixed RSS on tiny runs because the database engine has startup/index overhead; the scaling benefit appears as workflow count grows.
- Retrieval latency includes query embedding and backend search/reranking.
