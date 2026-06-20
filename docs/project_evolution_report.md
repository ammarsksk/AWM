# Project Evolution Report

## Executive Summary

The project started as an attempt to understand and reproduce the Agent Workflow Memory idea from the paper. The initial goal was to make the core idea run locally, even without full WebArena infrastructure. Over time, the project evolved from a simple smoke test into a much more complete offline memory and retrieval system over Mind2Web trajectories.

The final architecture includes workflow extraction, deterministic and LLM-capable workflow abstraction, semantic embeddings, hybrid retrieval, benchmark metrics, memory traces, and multiple long-term storage backends. The strongest current result is the compressed FAISS IVFPQ workflow memory backend, which stores workflow vectors compactly and retrieves relevant workflows quickly.

The main engineering progression was:

1. Understand the paper and codebase.
2. Make a local smoke test to demonstrate core functionality.
3. Add LLM wiring for real model calls.
4. Move from fake examples to real Mind2Web exemplar data.
5. Add workflow extraction and workflow memory generation.
6. Add semantic retrieval using embeddings.
7. Improve retrieval using FAISS and lexical reranking.
8. Add paper-style metrics and comparison reports.
9. Add parallel execution to make evaluation faster.
10. Add detailed traces for prompts, retrieval, embeddings, and memory.
11. Improve workflow abstraction.
12. Redesign long-term memory away from pure RAM.
13. Test LanceDB, memmap, scalar-quantized FAISS, and IVFPQ FAISS.
14. Select compressed FAISS IVFPQ as the current best backend.

## Initial Problem

The original request was to implement the ideas from the Agent Workflow Memory paper in the local codebase. The full WebArena setup was too heavy to run directly because it required multiple websites, services, Docker infrastructure, credentials, and substantial compute.

The immediate challenge was therefore not simply implementation. It was deciding how to demonstrate the core mechanism without requiring the entire original benchmark infrastructure.

The project had to answer:

- Can we show workflow memory working locally?
- Can we show LLM traces?
- Can we show where workflows are stored?
- Can we retrieve workflows for new tasks?
- Can we compare metrics with the paper?
- Can we make the memory system more realistic and scalable?

## Phase 1: Understanding The Paper And Framework

The first phase was conceptual. The key idea from the paper is that agents should not solve each task from scratch. They should store successful workflows and retrieve them later when similar tasks appear.

The important memory concepts were:

- Short-term memory: context from the current task or current episode.
- Episodic memory: records of past task executions.
- Long-term workflow memory: reusable procedural knowledge extracted from previous tasks.
- Retrieval: selecting relevant workflows for a new problem.
- Induction: turning completed task trajectories into reusable workflows.

The core architecture from the paper can be summarized as:

1. Agent attempts a task.
2. The trajectory is recorded.
3. The trajectory is converted into a workflow.
4. The workflow is stored in memory.
5. A future task retrieves relevant workflows.
6. The retrieved workflows are included in the agent's context.
7. The agent uses them to act more effectively.

The main challenge was that the original environment expected full web servers and browser execution. That was not practical locally, so the first implementation focused on reproducing the memory logic.

## Phase 2: Local Smoke Test

The first working milestone was a local smoke test. This was designed to prove that the core flow could work without full WebArena infrastructure.

The smoke test demonstrated:

- A task could be represented locally.
- A workflow could be generated.
- A memory file could be written.
- A retrieval trace could be produced.
- A report could be generated.

At this stage, deterministic outputs were acceptable because the main goal was to show the structure of the system.

This helped establish the first complete loop:

1. Start with a task.
2. Produce actions.
3. Store a workflow.
4. Retrieve a workflow.
5. Write trace files.
6. Produce a report.

## Phase 3: LLM Integration

The next requirement was to involve a real LLM. Several options were explored, including local models, NVIDIA NIM, Gemini API, and Vertex AI.

The main issue was access and reliability:

- Local model download was blocked by DNS/network issues.
- Some cloud API options had quota or rate-limit problems.
- Vertex AI was eventually configured as a more reliable cloud route.

The LLM integration allowed the demo system to produce actual model outputs instead of only deterministic placeholder actions.

The LLM trace made the system easier to explain because it showed:

- Prompt construction.
- Task context.
- Candidate elements.
- Retrieved workflows.
- Model output.
- Parsed action.

This was important for presentation because it showed the agentic loop rather than only final metrics.

## Phase 4: Moving To Real Mind2Web Data

The next major improvement was using real Mind2Web exemplar data.

The exemplars are recorded task trajectories. They contain real benchmark examples with:

- Website metadata.
- User task goals.
- Observations.
- Candidate elements.
- Gold actions.

Using exemplars was important because it moved the project away from toy examples.

This phase made the project more credible because workflows were no longer invented manually. They were extracted from real benchmark trajectories.

The system could now:

1. Load real trajectories.
2. Extract workflows from them.
3. Store those workflows.
4. Retrieve workflows for step-level task queries.
5. Compute metrics over many tasks.

## Phase 5: Workflow Extraction

Workflow extraction turns a completed trajectory into a reusable workflow.

The raw trajectory contains step-level details:

- Observation.
- Candidate elements.
- Gold action.
- Action type.
- Action value.
- Element information.

The raw workflow preserves the evidence from the trajectory.

However, raw workflows are often too specific. If a workflow says "click element 136", that is not useful for a different page where the corresponding button has a different element ID.

So the system needed abstraction.

## Phase 6: Workflow Abstraction

Workflow abstraction was added to make stored workflows more reusable.

The project supports multiple abstraction modes:

- Raw workflow storage.
- Deterministic abstraction.
- LLM-based abstraction.

Raw workflow storage keeps the original details. It is useful for debugging but weak for generalization.

Deterministic abstraction is fast and stable. It converts trajectories into reusable summaries without an LLM call.

LLM-based abstraction is richer but slower and more expensive. It can describe the intent of each step more naturally, but it introduces latency, cost, and possible formatting failures.

For backend benchmarking, deterministic abstraction was used. This kept the memory and retrieval timing clean.

The abstraction goal was to move from specific actions like:

- Select element 136.
- Type value into element 753.

Toward reusable workflow steps like:

- Select the requested service type.
- Enter the requested location.
- Submit the search.

This is one of the most important conceptual steps in the project.

## Phase 7: First Retrieval System

The initial retrieval system was simpler and more deterministic. It could retrieve workflows using available workflow text and local matching.

This was enough for early tests, but it had a major limitation: retrieval quality depended too much on exact words matching.

The user correctly identified the problem:

If retrieval only matches words, it may miss workflows that are semantically similar but phrased differently.

For example:

- "book a flight"
- "search airline tickets"

These are semantically close, but exact-word matching may not recognize that.

This led to the next major change: embeddings.

## Phase 8: Semantic Embeddings

Embeddings were added so workflows and queries could be compared semantically.

An embedding model converts text into a vector. Similar meanings should produce nearby vectors.

The system embeds:

- Workflow text during memory construction.
- Query text during retrieval.

The query vector is compared against workflow vectors. This lets the system retrieve workflows based on meaning rather than exact wording.

This was a major upgrade from deterministic retrieval.

## Phase 9: Cosine Similarity

The system normalizes embeddings and uses inner-product search. Once vectors are normalized, inner product is equivalent to cosine similarity.

Cosine similarity is useful because it measures semantic direction rather than raw vector magnitude.

This means the system compares the meaning of a query against the meaning of stored workflows.

The retrieval logic became:

1. Convert workflow text into vector.
2. Store vector.
3. Convert query text into vector.
4. Compare query vector with workflow vectors.
5. Retrieve nearest workflows.

## Phase 10: FAISS Retrieval

FAISS was introduced to make vector search faster and more scalable.

FAISS is a vector search library designed for nearest-neighbor retrieval. It is much better suited for embedding search than manually comparing every vector in Python.

The first FAISS-based system still kept vectors in RAM. This was fast for small runs but not ideal for long-term memory.

The RAM system helped prove:

- Embeddings worked.
- Semantic retrieval worked.
- FAISS retrieval worked.
- Hybrid scoring could improve workflow selection.

But it also revealed a scaling problem:

- Raw vectors in Python memory grow with the number of workflows.
- Workflow text and metadata also grow.
- A million-workflow memory would not fit comfortably in a simple Python list.

This pushed the project toward long-term memory storage redesign.

## Phase 11: Hybrid Retrieval

Semantic similarity alone is powerful but not always enough.

The system added hybrid reranking with:

- Semantic score.
- Lexical overlap.
- Same-website boost.
- Same-domain boost.

Semantic score handles meaning.

Lexical overlap helps when important task terms match directly.

Website boost helps prefer workflows from the same website.

Domain boost helps prefer workflows from the same task family.

The current formula gives the largest weight to semantic similarity, which is correct because the main retrieval mechanism should be meaning-based.

The metadata boosts are deliberately small. They guide the ranking without completely blocking cross-website reuse.

## Phase 12: Step-Level Evaluation

The system then added step-level evaluation over Mind2Web examples.

The purpose was to measure whether the predicted action at each step matched the gold action.

The project computed paper-style metrics such as:

- Element accuracy.
- Action F1.
- Step success.

This helped compare the local implementation with paper-style evaluation.

However, a key limitation became clear:

The offline evaluator does not open a real website. It asks the model or heuristic to select from recorded candidate elements. Therefore, the model is not truly visually interacting with a page. It is predicting over a static representation.

That limitation does not make the benchmark useless, but it changes the interpretation. The benchmark is useful for measuring memory, retrieval, and step prediction over recorded trajectories. It is not the same as live browser execution.

## Phase 13: Parallel Execution

The full evaluation was slow, especially when LLM calls were involved.

Parallel execution was added to run multiple tasks concurrently.

The purpose was practical:

- Reduce total evaluation time.
- Make 500-step or larger runs feasible.
- Allow repeated experiments while tuning retrieval.

The parallel run did not simulate multiple independent agents with shared planning. It mainly parallelized task evaluation work. So it was a speed optimization, not a multi-agent architecture.

## Phase 14: Metrics Reports

Reports were added to summarize:

- Run configuration.
- Evaluation metrics.
- Paper-style comparisons.
- Architecture used.
- Limitations.
- Interpretation of results.

These reports were important because raw JSON traces are hard to present. A report makes it easier to explain what was tested and what the numbers mean.

The project also separated stricter paper-style metrics from relaxed internal metrics. This was important because relaxed metrics can make results look better but are not directly comparable to the paper.

## Phase 15: Embedding Trace Demo

A detailed embedding trace demo was added so the internal workflow retrieval process could be shown in the terminal.

The trace demo shows:

- Query text.
- Query embedding summary.
- Candidate retrievals.
- Semantic scores.
- Lexical scores.
- Metadata boosts.
- Accepted workflow.
- Workflow storage behavior.
- RAM and process memory measurements.

This was useful for explaining the system to someone unfamiliar with the codebase.

It also made the memory mechanism visible:

1. Workflow text is embedded.
2. Query is embedded.
3. Semantic comparison happens.
4. Top candidates are reranked.
5. The accepted workflow is chosen.

## Phase 16: Long-Term Memory Redesign

Once semantic retrieval worked, the next concern was scalability.

Keeping all embeddings in RAM is acceptable for small demos but weak as a long-term memory design.

The project explored long-term memory options:

1. Keep everything in RAM.
2. Use a disk-backed vector database.
3. Use memory-mapped vectors.
4. Use compressed FAISS indexes.

The goal was to keep retrieval fast while avoiding storing all raw vectors as Python objects.

## Phase 17: LanceDB Experiment

LanceDB was tested as a disk-backed ANN database.

The idea was attractive:

- Store vectors in a database.
- Keep workflow JSON separate.
- Use ANN indexes for retrieval.
- Avoid raw Python vector memory.

However, the benchmark showed poor results for the local workload.

The LanceDB run had higher latency and higher memory overhead than expected. It also had database-engine overhead that did not pay off at this scale.

The conclusion was:

LanceDB is a valid architecture for some production systems, but it was not the best backend for this local benchmark.

## Phase 18: Memmap Experiment

A memory-mapped vector backend was tested next.

The idea was:

- Store vectors in a disk-backed array.
- Avoid Python vector lists.
- Search vectors in chunks.

This improved storage behavior but retrieval remained slower because it still performed exact vector scanning across chunks.

The conclusion was:

Memmap is simple and storage-efficient, but it is not the best retrieval backend for this workload.

## Phase 19: Scalar-Quantized FAISS

Compressed FAISS was then introduced.

The first successful compressed FAISS backend used scalar quantization.

Scalar quantization reduced each vector dimension from float32 to an 8-bit representation. For a 384-dimensional vector, that reduced the vector code to around 384 bytes.

This was a major improvement over raw vectors.

The scalar-quantized FAISS run improved:

- Total runtime.
- Retrieval latency.
- Storage usage.
- Python vector memory.

It also preserved very high retrieval acceptance.

This became the first clearly strong long-term memory backend.

## Phase 20: IVFPQ FAISS

The final improvement was IVFPQ.

Scalar quantization still scans all vectors. IVFPQ adds clustering and product quantization.

IVF clusters the vector space so the system only searches relevant regions.

PQ compresses vectors into very small codes.

The best run used:

- 22 coarse clusters.
- 16 product-quantization chunks.
- 8 bits per chunk.
- 16 bytes per workflow vector code.

This produced the best runtime and retrieval latency so far.

The tradeoff was a tiny drop in accepted retrieval count compared with scalar quantization, but the speed improvement made IVFPQ the current best backend.

## Why The Final Backend Is The Best Current Choice

The compressed FAISS IVFPQ backend is the strongest because it balances speed, memory, and storage.

It is better than pure RAM because:

- It avoids raw Python vector storage.
- It compresses vectors heavily.
- It stores workflow payloads on disk.

It is better than memmap because:

- It uses FAISS ANN search instead of chunked exact scanning.

It is better than LanceDB in this local benchmark because:

- It has lower overhead.
- It is faster for the tested workload.
- It integrates directly with the existing retrieval code.

It is better than scalar-quantized FAISS because:

- It adds clustering.
- It reduces retrieval latency further.
- It compresses vector codes more aggressively.

## Current Final System

The current system can be summarized as:

1. Load real benchmark trajectories.
2. Extract task workflows.
3. Abstract workflows.
4. Convert workflows into embedding text.
5. Embed workflow text.
6. Normalize vectors.
7. Build compressed FAISS IVFPQ index.
8. Store full workflows on disk.
9. Store embedding texts separately.
10. Store metadata mapping FAISS rows to workflow records.
11. Embed incoming task-step queries.
12. Retrieve semantic candidates.
13. Rerank candidates using hybrid scoring.
14. Accept the best workflow if it passes threshold.
15. Record latency, memory, and retrieval metrics.

This is a complete workflow-memory retrieval architecture.

## Main Concepts Used

The project now uses several important concepts:

### Workflow Memory

Workflow memory stores reusable task procedures. Instead of remembering only individual actions, the system remembers structured action sequences.

### Abstraction

Abstraction converts specific trajectories into reusable patterns. This is necessary because exact element IDs and page-specific details do not transfer well.

### Embeddings

Embeddings convert text into vectors so semantic similarity can be measured.

### Cosine Similarity

Cosine similarity compares the direction of vectors. It is used to measure semantic closeness.

### FAISS

FAISS provides fast nearest-neighbor search over embeddings.

### IVF

IVF clusters vectors so search can focus on relevant regions.

### Product Quantization

Product quantization compresses high-dimensional vectors into compact codes.

### Hybrid Reranking

Hybrid reranking combines semantic, lexical, and metadata signals.

### Metadata Mapping

Metadata connects FAISS row IDs to workflow records and reranking text.

### LRU Cache

The text cache reduces repeated disk reads during reranking.

### Batch Indexing

Batch indexing builds the long-term index efficiently after workflows are prepared.

## Key Engineering Decisions

The most important engineering decisions were:

1. Use Mind2Web exemplars before attempting full WebArena.
2. Build a local smoke test before full evaluation.
3. Separate deterministic tests from LLM tests.
4. Use embeddings for semantic retrieval.
5. Use FAISS for vector search.
6. Add lexical and metadata reranking.
7. Move workflow payloads out of vector storage.
8. Avoid raw Python vector memory.
9. Benchmark multiple storage backends.
10. Select compressed FAISS IVFPQ as the best current backend.

Each decision made the system more realistic, scalable, or easier to evaluate.

## Current Limitations

The current project still has limitations:

- The best benchmark is offline, not live browser execution.
- The best benchmark does not use LLM action prediction.
- Deterministic abstraction is less powerful than LLM abstraction.
- The lexical reranker is not full BM25.
- The workflow memory may contain duplicate or overlapping workflows.
- The system has not yet been tested at million-workflow scale.
- Retrieval quality depends heavily on workflow text quality.

These are clear next directions rather than failures.

## Best Next Steps

The next best improvements are:

1. Improve workflow abstraction using LLM-generated abstract workflows.
2. Add workflow deduplication and merging.
3. Replace lexical overlap with real BM25.
4. Add a stronger reranker for top candidate workflows.
5. Test larger workflow banks.
6. Add staged online ingestion with periodic index rebuilds.
7. Connect retrieved workflows into a full LLM action-generation loop.
8. Evaluate on live or semi-live web tasks.

The memory backend is now strong enough. The next major gains will come from better workflow quality and better final workflow selection.

## Final Interpretation

The project successfully moved from a basic local demonstration to a strong compressed semantic workflow memory system.

The final backend is not just a toy memory array. It is a compressed long-term retrieval system with:

- Workflow induction.
- Workflow abstraction.
- Semantic embeddings.
- FAISS IVFPQ indexing.
- Disk-backed workflow payloads.
- Metadata mapping.
- Hybrid reranking.
- Text caching.
- Detailed benchmark metrics.

The current best result shows that the system can retrieve workflows quickly and compactly over a real benchmark-derived workflow set. This is a meaningful implementation of the core Agent Workflow Memory idea, adapted to a practical local benchmark environment.

