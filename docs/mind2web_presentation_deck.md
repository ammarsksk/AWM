# Mind2Web Agent Workflow Memory Presentation

## Slide 1: Title

**Production-Oriented Agent Workflow Memory for Mind2Web**

Subtitle:

**Workflow reuse, procedural memory, compressed vector retrieval, and 500-task evaluation**

Talk track:

This project started from the Agent Workflow Memory paper idea: agents should not solve every task from scratch. They should remember useful workflows from previous tasks, retrieve them for similar future tasks, and use them as procedural guidance.

---

## Slide 2: Problem

LLM web agents are expensive and fragile when every task is solved from zero.

The core problems:

| Problem | Why it matters |
| --- | --- |
| No memory | The agent repeats reasoning for similar tasks |
| Long prompts | Full trajectories and page text inflate context |
| Slow retrieval | Naive vector search does not scale well |
| Weak transfer | Exact page actions do not generalize |
| Hard evaluation | Need paper-style metrics, not only anecdotes |

Talk track:

Mind2Web tasks often share procedural structure. A flight search, hotel search, restaurant booking, or product filter task has a reusable sequence even when the exact website and element ids change.

---

## Slide 3: Paper Baseline Idea

The AWM paper stores successful workflows and retrieves them for future tasks.

Paper-style flow:

```text
Task attempt
  -> trajectory
  -> workflow induction
  -> workflow memory
  -> retrieve relevant workflow
  -> add workflow to agent prompt
  -> solve new task
```

What the paper contributes:

- Procedural memory for web agents.
- Workflow induction from prior trajectories.
- Retrieval-augmented prompting.
- Evaluation on Mind2Web/WebArena-style environments.

Talk track:

The paper is conceptually strong, but the implementation style is research-oriented. My work focuses on making the memory layer more practical, measurable, and efficient.

---

## Slide 4: My Implementation

I implemented an end-to-end local Mind2Web AWM-style system.

Main modules:

| Component | File / artifact |
| --- | --- |
| Step evaluator | `mind2web/llm_step_eval.py` |
| Procedural memory | `mind2web/procedural_memory.py` |
| Compressed FAISS memory | `mind2web/compressed_faiss_memory.py` |
| Backend benchmark | `mind2web/benchmark_memory_backends.py` |
| Metrics/report builder | `mind2web/build_paper_comparison.py` |
| Best procedural run | `mind2web/step_eval_vertex_500_procedural_gemini25pro/` |
| Best memory benchmark | `mind2web/memory_backend_benchmark_500x2_ivfpq_nlist32/` |

Talk track:

The implementation is not just a prompt demo. It includes workflow extraction, procedural memory construction, retrieval, reranking, LLM prediction, paper-style metrics, and storage/retrieval benchmarks.

---

## Slide 5: Data And Evaluation Setup

The runs use real Mind2Web exemplar trajectories.

| Item | Value |
| --- | --- |
| Data source | `mind2web/data/memory/exemplars.json` |
| Accuracy run tasks | 500 |
| Accuracy run steps | 3790 |
| Model | Gemini 2.5 Pro through Vertex/OpenAI-compatible adapter |
| Parallel workers | 9 |
| Memory reuse policy | Same-website |
| Metrics | Paper-style strict element/action/step/task metrics |

Important qualification:

This is a local exemplar-based evaluation, not an official reproduction of the paper's official Mind2Web split protocol.

Talk track:

I compare against paper reference numbers, but I should present this honestly as a local implementation comparison, not as an official leaderboard claim.

---

## Slide 6: Accuracy Results

Best final Mind2Web procedural run:

`mind2web/step_eval_vertex_500_procedural_gemini25pro`

| Metric | Result |
| --- | ---: |
| Tasks | 500 |
| Steps | 3790 |
| Element Accuracy | 59.76% |
| Relaxed Element Accuracy | 95.78% |
| Operation Accuracy | 91.13% |
| Action F1 | 88.07% |
| Step Success Rate | 54.59% |
| Relaxed Step Success Rate | 84.67% |
| Task Success Rate | 11.60% |
| Exact Sequence Rate | 11.00% |

Talk track:

The strongest part is operation and action-value understanding. The main remaining weakness is exact element id selection, especially where the gold target is a nested or unlabeled element.

---

## Slide 7: Comparison Against Paper References

Strongest paper reference in the local report:

`Mind2Web cross-task / AWM GPT-4 offline`

| Metric | My Gemini 2.5 Pro Procedural Run | AWM Paper Reference | Difference |
| --- | ---: | ---: | ---: |
| Element Accuracy | 59.76% | 50.60% | +9.16 pts |
| Action F1 | 88.07% | 57.30% | +30.77 pts |
| Step Success Rate | 54.59% | 45.10% | +9.49 pts |
| Task Success Rate | 11.60% | 4.80% | +6.80 pts |

Relative improvements:

| Metric | Relative lift |
| --- | ---: |
| Element Accuracy | 1.18x |
| Action F1 | 1.54x |
| Step Success Rate | 1.21x |
| Task Success Rate | 2.42x |

Talk track:

The largest improvement is task success: 11.6% versus 4.8%, which is about 2.4x the paper reference. The caveat is that the evaluation setups differ, so I should phrase this as "under my local evaluation setup."

---

## Slide 8: My Run Versus My Earlier Baseline

Earlier local Gemini run:

`mind2web/step_eval_vertex_500`

Final procedural Gemini 2.5 Pro run:

`mind2web/step_eval_vertex_500_procedural_gemini25pro`

| Metric | Earlier Gemini Run | Procedural Gemini 2.5 Pro Run | Change |
| --- | ---: | ---: | ---: |
| Element Accuracy | 60.11% | 59.76% | -0.35 pts |
| Action F1 | 88.73% | 88.07% | -0.66 pts |
| Step Success Rate | 55.12% | 54.59% | -0.53 pts |
| Task Success Rate | 10.40% | 11.60% | +1.20 pts |
| Exact Sequence Rate | 9.80% | 11.00% | +1.20 pts |

Interpretation:

The procedural memory run slightly reduced step-level strict metrics but improved full-task completion. This suggests the memory helps task-level consistency even when individual strict element ids remain noisy.

Talk track:

This is important because web-agent success is not only about one isolated step. A memory-guided system can improve full sequence completion even if some step-level metrics move slightly.

---

## Slide 9: Architecture Overview

```text
Mind2Web exemplar
  -> structured workflow extraction
  -> workflow abstraction
  -> workflow text
  -> embedding
  -> procedural / vector memory
  -> retrieve top-k workflows
  -> rerank/filter
  -> prompt Gemini
  -> parse action
  -> compare against gold action
  -> update memory and traces
```

Key design choice:

Separate the memory payload from the searchable representation.

| Layer | Purpose |
| --- | --- |
| Full workflow JSON | Complete procedural evidence |
| Compact workflow text | Search and rerank text |
| Embedding vector | Semantic retrieval |
| Metadata | Website/domain/task-family filtering |
| Metrics trace | Reproducibility and analysis |

Talk track:

This is a more production-like design than storing everything as raw text in the prompt or as Python vectors in memory.

---

## Slide 10: Procedural Memory Design

The procedural memory stores reusable workflows, not just individual examples.

Stored information:

| Memory item | Purpose |
| --- | --- |
| Task goal | What the user wanted |
| Website/domain/subdomain | Where the procedure worked |
| Step skeleton | Reusable action sequence |
| Element/action metadata | How each step was grounded |
| Outcome data | Whether the procedure worked |
| Negative memories | Failed patterns to avoid |
| Graph edges | Relationships between procedures |

Talk track:

The goal is not to blindly replay old actions. The goal is to give the LLM a compact procedural plan that can be adapted to the current task.

---

## Slide 11: Retrieval And Scoring

The procedural run used a hybrid memory configuration:

| Component | Value |
| --- | --- |
| Procedure count | 500 |
| Graph edges | 33,346 |
| Negative memories | 1,205 |
| Embedding backend | SentenceTransformers + FAISS + BM25 |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` |
| Candidate retrieval | top 50 |
| Final workflows | top 5 |
| Acceptance threshold | 0.34 |

Scoring weights:

| Signal | Weight |
| --- | ---: |
| Semantic similarity | 0.50 |
| Lexical overlap | 0.14 |
| Page/task activation | 0.16 |
| Graph score | 0.08 |
| Past outcome | 0.08 |
| Metadata match | 0.04 |

Talk track:

This is better than pure embedding search because it combines meaning, words, website metadata, graph connectivity, and past success.

---

## Slide 12: Memory Optimization Result

Best memory benchmark:

`mind2web/memory_backend_benchmark_500x2_ivfpq_nlist32/compressed_faiss`

| Metric | Result |
| --- | ---: |
| Tasks | 500 |
| Retrieval steps | 1000 |
| Accepted retrievals | 998 / 1000 |
| Total runtime | 39.68 s |
| Online retrieval runtime | 26.48 s |
| Avg retrieval latency | 26.16 ms |
| P95 retrieval latency | 38.65 ms |
| Workflow count | 500 |
| FAISS index | IVFPQ, nlist 22, m 16, 8 bits |
| FAISS code size | 16 bytes/vector |
| Estimated compressed vector storage | 0.0076 MB |
| Disk bytes | 13.23 MB |

Talk track:

This is the best engineering result: fast retrieval plus extremely compact vector storage.

---

## Slide 13: Why Compressed FAISS Matters

Normal embedding storage:

```text
384 dimensions x 4 bytes = 1536 bytes per vector
```

Compressed IVFPQ storage:

```text
16 bytes per vector
```

Compression effect:

| Storage type | Bytes/vector |
| --- | ---: |
| Float32 embedding | 1536 |
| FAISS IVFPQ code | 16 |

That is a **96x smaller vector code** before Python overhead.

Talk track:

This is why the memory layer can scale. The system keeps the searchable semantic index compact and keeps full workflow payloads on disk.

---

## Slide 14: Backend Comparison

Same workload: 500 tasks, 1000 retrieval steps.

| Metric | RAM FAISS/BM25 | Compressed FAISS IVFPQ | Improvement |
| --- | ---: | ---: | ---: |
| Accepted retrievals | 990 | 998 | +8 |
| Total runtime | 72.57 s | 39.68 s | 1.83x faster |
| Online runtime | 40.88 s | 26.48 s | 1.54x faster |
| Avg retrieval latency | 40.50 ms | 26.16 ms | 35.4% lower |
| P95 retrieval latency | 64.29 ms | 38.65 ms | 39.9% lower |
| Add/store time | 57.23 ms | 19.62 ms | 65.7% lower |
| RSS growth | 167.31 MB | 155.51 MB | 7.1% lower |
| Disk bytes | 16.78 MB | 13.23 MB | 21.2% lower |
| Python vector memory | 0.7324 MB | 0 MB | removed |

Talk track:

The compressed backend is not just smaller. It is faster in total runtime, online retrieval, p95 latency, and ingestion/storage.

---

## Slide 15: Optimizations Implemented

| Optimization | Impact |
| --- | --- |
| SentenceTransformers embeddings | Semantic retrieval instead of exact word matching |
| FAISS vector search | Fast nearest-neighbor retrieval |
| BM25 / lexical reranking | Recovers exact keyword matches |
| Same-website reuse policy | Reduces misleading cross-site memory |
| Compact workflow text | Smaller prompt/context |
| Structured workflow JSON | Reusable procedure, not raw string |
| Deterministic abstraction for benchmarks | Removes LLM latency from memory timing |
| Parallel LLM prediction | Faster 500-task evaluation |
| Retry/output parsing fixes | Fewer failed model outputs |
| Compressed FAISS IVFPQ | 96x vector-code compression |
| Disk-backed payload storage | Full workflows do not need to live as Python objects |
| Text cache | Avoids repeated workflow text loads |

Talk track:

The main improvement is architectural: retrieval, storage, prompting, and evaluation are separated cleanly.

---

## Slide 16: My Implementation Versus Paper Implementation

| Area | AWM paper style | My implementation |
| --- | --- | --- |
| Environment | Official benchmark setup | Local Mind2Web exemplar evaluation |
| Model | GPT-4 / GPT-3.5 references | Vertex/Gemini 2.5 Pro |
| Workflow storage | Research workflow memory | Structured workflow JSON + compact text + metadata |
| Retrieval | Workflow retrieval for prompting | Hybrid semantic, lexical, metadata, graph, outcome scoring |
| Memory backend | Not optimized for storage benchmarking | RAM, LanceDB, memmap, SQ8 FAISS, IVFPQ FAISS tested |
| Vector storage | Not the focus | Compressed FAISS IVFPQ, 16 bytes/vector |
| Evaluation | Official split metrics | Paper-style metrics over local 500 exemplar tasks |
| Tracing | Paper results | Full prediction, retrieval, workflow, runtime traces |
| Production readiness | Research prototype | Modular storage, metrics, artifacts, reproducible run folders |

Talk track:

The paper focuses on proving that workflow memory helps agents. My work focuses on implementing that idea and pushing the memory layer toward production-grade speed, storage, and observability.

---

## Slide 17: Error Analysis

Largest remaining error category from the earlier 500-task strict run:

| Category | Percentage |
| --- | ---: |
| Exact step correct | 55.12% |
| Right operation, wrong element | 33.11% |
| Wrong operation | 8.28% |
| Right element, wrong value | 3.46% |
| Empty/unparsed output | 0.03% |

Main issue:

The model often understands the correct operation but chooses a nearby or parent/child element id.

Example issue:

Gold target may be a nested `svg` or `span`; model clicks the visible parent button. Strict scoring marks this wrong even when it is semantically close.

Talk track:

This explains why operation accuracy is above 91%, while strict step success is around 55%.

---

## Slide 18: What Is Actually Better?

Concrete wins:

1. Higher task success than the paper reference in local evaluation.
2. Much higher action F1 than the paper reference.
3. A complete workflow-memory implementation over real Mind2Web exemplars.
4. Compressed vector memory with 96x vector-code compression.
5. Faster retrieval than RAM FAISS/BM25 baseline.
6. Full reproducibility artifacts: traces, metrics, workflow JSON, embeddings, benchmark outputs.

Best headline numbers:

| Claim | Evidence |
| --- | --- |
| Task SR is higher than AWM paper reference | 11.60% vs 4.80% |
| Step SR is higher than AWM paper reference | 54.59% vs 45.10% |
| Action F1 is higher than AWM paper reference | 88.07% vs 57.30% |
| Retrieval is fast | 26.16 ms avg |
| Vector codes are compact | 16 bytes/vector |
| Retrieval acceptance is high | 998 / 1000 |

Talk track:

The two strongest messages are: better local paper-style results and a much more engineered memory backend.

---

## Slide 19: Limitations

Important limitations:

- This is not an official Mind2Web split reproduction.
- The run uses local exemplar ordering.
- The evaluation uses saved observations, not live browser interaction.
- Strict element id scoring can undercount semantically correct parent/child clicks.
- LLM abstraction was not used in the fastest memory benchmark because it would dominate latency.
- Results may change with model version, task order, and candidate extraction.

Talk track:

These limitations do not invalidate the result. They define exactly what the result proves: the memory architecture is strong, fast, compact, and works on local Mind2Web-style evaluation.

---

## Slide 20: Conclusion

Final summary:

I implemented a full AWM-style Mind2Web system with:

- Workflow extraction.
- Procedural memory.
- Hybrid retrieval.
- Gemini 2.5 Pro step prediction.
- Paper-style evaluation.
- Full tracing and metrics.
- Compressed FAISS IVFPQ memory backend.

Main result:

```text
Gemini 2.5 Pro procedural run:
Task SR: 11.60%
Step SR: 54.59%
Action F1: 88.07%

Best paper reference:
Task SR: 4.80%
Step SR: 45.10%
Action F1: 57.30%
```

Memory result:

```text
Compressed FAISS IVFPQ:
998 / 1000 accepted retrievals
26.16 ms average retrieval
38.65 ms p95 retrieval
16 bytes per vector
```

Talk track:

The project moves AWM from a paper idea into a measurable, optimized, reproducible implementation over real Mind2Web trajectories.

---

## Appendix: Exact Artifact Locations

| Artifact | Location |
| --- | --- |
| Best procedural Gemini run | `mind2web/step_eval_vertex_500_procedural_gemini25pro/` |
| Procedural run metrics | `mind2web/step_eval_vertex_500_procedural_gemini25pro/paper_metrics.json` |
| Procedural runtime stats | `mind2web/step_eval_vertex_500_procedural_gemini25pro/runtime_stats.json` |
| Earlier Gemini baseline | `mind2web/step_eval_vertex_500/` |
| Baseline final report | `mind2web/step_eval_vertex_500/final_report.md` |
| No-LLM procedural retrieval run | `mind2web/step_eval_500_procedural_no_llm/` |
| Best compressed FAISS benchmark | `mind2web/memory_backend_benchmark_500x2_ivfpq_nlist32/` |
| Current best memory report | `docs/current_best_run_report.md` |
| Project evolution report | `docs/project_evolution_report.md` |

