# Synthetic Vector Index Sweep Report
Database Size: 5000 procedures, Queries: 100

| Backend | Actual Index Kind | Avg Retrieval (ms) | p95 Retrieval (ms) | Build Time (ms) | Index Size | Disk Size | RAM Delta (MB) |
|---|---|---:|---:|---:|---:|---:|---:|
| flat | flat_ip | 1.624 | 1.796 | 492.4 | 7,680,045 B | 8,077,843 B | 895.04 |
| hnsw | hnsw_m32 | 1.096 | 1.420 | 613.3 | 9,037,250 B | 9,435,049 B | 2.30 |
| sq8 | sq8_flat | 1.413 | 1.455 | 244.5 | 1,923,153 B | 2,320,950 B | 9.86 |
| hnsw_sq8 | hnsw_sq8_m32 | 1.371 | 1.753 | 495.7 | 3,280,358 B | 3,678,165 B | 7.44 |
| ivfpq | ivfpq_nlist64_m16_bits8 | 0.431 | 0.455 | 1695.8 | 612,212 B | 1,010,024 B | 0.51 |
| opq_ivfpq | opq_ivfpq_nlist64_m16_bits8 | 0.507 | 0.544 | 29826.0 | 1,202,107 B | 1,599,930 B | -9.17 |
| turboquant | turboquant_signperm | 0.513 | 0.529 | 184.7 | 240,033 B | 639,367 B | 14.59 |
| rabitq | rabitq_dense | 3.001 | 3.090 | 367.9 | 240,033 B | 1,184,535 B | 17.81 |
| binary_hnsw_rotation | turboquant_signperm_hnsw | 0.955 | 1.286 | 372.4 | 1,597,226 B | 1,996,576 B | 5.61 |
