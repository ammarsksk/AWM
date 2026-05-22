# Workflow Memory Backend Benchmark

Same workload for both backends: same exemplar slice, same task/step limits, same abstraction mode.

| Metric | RAM FAISS/BM25 | LanceDB Disk ANN | Memmap Disk Exact | Compressed FAISS |
| --- | ---: | ---: | ---: | ---: |
| Tasks | 1009 | 0 | 0 | 1009 |
| Steps | 2018 | 0 | 0 | 2018 |
| Accepted Retrievals | 1985 | 0 | 0 | 2018 |
| Total Runtime ms | 214866.62 | 0.00 | 0.00 | 99193.74 |
| Online Runtime ms | 123699.77 | 0.00 | 0.00 | 71428.19 |
| Prep ms | 8227.50 | 0.00 | 0.00 | 6962.72 |
| Batch Add ms | 82938.79 | 0.00 | 0.00 | 20792.04 |
| Final Index ms | 0.00 | 0.00 | 0.00 | 9.56 |
| Avg Retrieval ms | 60.93 | 0.00 | 0.00 | 35.07 |
| P95 Retrieval ms | 94.31 | 0.00 | 0.00 | 42.94 |
| Avg Add/Store ms | 82.20 | 0.00 | 0.00 | 20.61 |
| P95 Add/Store ms | 82.20 | 0.00 | 0.00 | 20.61 |
| Start RSS MB | 871.52 | 0.00 | 0.00 | 870.66 |
| End RSS MB | 1060.49 | 0.00 | 0.00 | 1042.32 |
| Peak Sampled RSS MB | 1060.49 | 0.00 | 0.00 | 1042.32 |
| RSS Growth MB | 188.98 | 0.00 | 0.00 | 171.66 |
| Python Vector MB | 1.4780 | 0.0000 | 0.0000 | 0.0000 |
| Disk Bytes | 33845819 | 0 | 0 | 26211043 |

## Interpretation

- RAM FAISS/BM25 keeps all workflow vectors and embedding text in Python memory.
- LanceDB keeps vector-search storage in the disk-backed Lance table and records `python_vector_mb` as zero for the workflow collection.
- Memmap keeps vectors in a disk-backed NumPy memmap and avoids database-engine overhead; it uses exact chunked vector search.
- Compressed FAISS keeps quantized FAISS codes instead of Python vector lists, preserving FAISS-speed retrieval with lower vector memory.
- LanceDB online writes append to the table; ANN refresh can be deferred to the final/background index phase.
- LanceDB may show higher fixed RSS on tiny runs because the database engine has startup/index overhead; the scaling benefit appears as workflow count grows.
- Retrieval latency includes query embedding and backend search/reranking.
