# Workflow Memory Backend Benchmark

Same workload for both backends: same exemplar slice, same task/step limits, same abstraction mode.

| Metric | RAM FAISS/BM25 | LanceDB Disk ANN |
| --- | ---: | ---: |
| Tasks | 8 | 8 |
| Steps | 16 | 16 |
| Accepted Retrievals | 3 | 4 |
| Total Runtime ms | 990.85 | 1133.40 |
| Avg Retrieval ms | 42.37 | 36.31 |
| P95 Retrieval ms | 66.63 | 50.03 |
| Avg Add/Store ms | 36.42 | 46.66 |
| P95 Add/Store ms | 93.34 | 94.13 |
| Start RSS MB | 870.41 | 913.36 |
| End RSS MB | 960.45 | 1038.55 |
| Peak Sampled RSS MB | 960.45 | 1038.55 |
| RSS Growth MB | 90.04 | 125.18 |
| Python Vector MB | 0.0117 | 0.0000 |
| Disk Bytes | 238433 | 264091 |

## Interpretation

- RAM FAISS/BM25 keeps all workflow vectors and embedding text in Python memory.
- LanceDB keeps vector-search storage in the disk-backed Lance table and records `python_vector_mb` as zero for the workflow collection.
- LanceDB may show higher fixed RSS on tiny runs because the database engine has startup/index overhead; the scaling benefit appears as workflow count grows.
- Retrieval latency includes query embedding and backend search/reranking.
