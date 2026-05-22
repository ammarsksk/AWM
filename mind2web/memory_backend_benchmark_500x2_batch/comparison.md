# Workflow Memory Backend Benchmark

Same workload for both backends: same exemplar slice, same task/step limits, same abstraction mode.

| Metric | RAM FAISS/BM25 | LanceDB Disk ANN |
| --- | ---: | ---: |
| Tasks | 500 | 500 |
| Steps | 1000 | 1000 |
| Accepted Retrievals | 587 | 827 |
| Total Runtime ms | 223669.99 | 633253.37 |
| Online Runtime ms | 223669.97 | 632987.09 |
| Final Index ms | 0.00 | 265.85 |
| Avg Retrieval ms | 132.84 | 415.33 |
| P95 Retrieval ms | 1071.64 | 1327.39 |
| Avg Add/Store ms | 172.81 | 422.07 |
| P95 Add/Store ms | 1324.61 | 1347.11 |
| Start RSS MB | 869.68 | 913.22 |
| End RSS MB | 1033.30 | 1326.49 |
| Peak Sampled RSS MB | 1033.30 | 1326.49 |
| RSS Growth MB | 163.62 | 413.27 |
| Python Vector MB | 0.7324 | 0.0000 |
| Disk Bytes | 16827474 | 28922490 |

## Interpretation

- RAM FAISS/BM25 keeps all workflow vectors and embedding text in Python memory.
- LanceDB keeps vector-search storage in the disk-backed Lance table and records `python_vector_mb` as zero for the workflow collection.
- LanceDB online writes append to the table; ANN refresh can be deferred to the final/background index phase.
- LanceDB may show higher fixed RSS on tiny runs because the database engine has startup/index overhead; the scaling benefit appears as workflow count grows.
- Retrieval latency includes query embedding and backend search/reranking.
