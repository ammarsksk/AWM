# Current Best Run Report

## Executive Summary

The current best run uses a compressed FAISS long-term workflow memory with an IVFPQ index. The run evaluated 500 Mind2Web tasks with 2 steps per task, producing 1000 retrieval queries. It achieved 998 accepted workflow retrievals out of 1000 queries, an average retrieval latency of 26.16 milliseconds, and a 95th percentile retrieval latency of 38.65 milliseconds.

The key architectural achievement is that workflow memory is no longer stored as raw Python vectors in RAM. Instead, full workflow records are stored on disk, compact workflow texts are stored separately for reranking, metadata maps vector rows to workflow records, and the semantic vector index is stored as a compressed FAISS index. This gives the system fast semantic retrieval while keeping the long-term memory representation compact.

The best run used an IVFPQ FAISS index with 22 coarse clusters, 16 product-quantization sub-vectors, and 8 bits per sub-vector. Each workflow vector was represented by a 16-byte FAISS code. This is a major compression improvement over a normal 384-dimensional float32 embedding, which would require 1536 bytes per vector before Python overhead.

## Benchmark Configuration

The run used the following configuration:

| Setting | Value |
| --- | --- |
| Number of tasks | 500 |
| Steps per task | 2 |
| Total retrieval steps | 1000 |
| Workflow abstraction mode | Deterministic |
| Ingestion mode | Batch |
| Retrieval backend | Compressed FAISS |
| FAISS index type | IVFPQ |
| Actual index layout | 22 coarse clusters, 16 PQ chunks, 8 bits per chunk |
| Workflow candidates retrieved before reranking | 50 |
| Final workflows returned after reranking | 5 |
| Workflow count in memory | 500 |

The benchmark was designed to test the workflow memory system, not live browser control. The benchmark does not open websites, does not execute actions in a browser, and does not use an LLM for action prediction in this specific run. Instead, it uses real Mind2Web trajectories to build workflows and then evaluates how efficiently the memory system retrieves relevant workflows.

## Main Results

| Metric | Result |
| --- | ---: |
| Total runtime | 39.68 seconds |
| Retrieval phase runtime | 26.48 seconds |
| Workflow preparation time | 3.10 seconds |
| Batch add/index preparation time | 9.81 seconds |
| Final FAISS index construction time | 0.29 seconds |
| Average retrieval latency | 26.16 ms |
| 95th percentile retrieval latency | 38.65 ms |
| Accepted retrievals | 998 / 1000 |
| Start RSS memory | 871.27 MB |
| End RSS memory | 1026.78 MB |
| RSS growth | 155.51 MB |
| Python vector memory | 0 MB |
| FAISS code size per workflow vector | 16 bytes |
| Estimated compressed FAISS vector storage | 0.0076 MB |
| Text cache size | 128 items |
| Text cache hits | 36,938 |
| Text cache misses | 13,062 |

The important result is not only that retrieval became fast. The more important result is that the architecture now separates semantic indexing, workflow payload storage, metadata, and reranking text. This makes the system much closer to a real long-term workflow memory system.

## What The System Is Solving

The goal of Agent Workflow Memory is to avoid solving every task from scratch. When a new task appears, the agent should remember similar workflows it has seen before and reuse them as guidance.

For example, if the system has seen a flight-booking workflow before, then for a new flight-booking task it should retrieve a related workflow such as:

1. Select trip type.
2. Enter origin.
3. Enter destination.
4. Select departure date.
5. Submit search.

The exact website elements and values may differ, but the structure of the task is reusable. This is the core reason for workflow memory: tasks that look different at the surface level can share an underlying procedural structure.

## Offline Benchmark Scope

This run is an offline retrieval benchmark. That means it measures the memory architecture and retrieval system using recorded benchmark trajectories.

It does test:

- Workflow construction from task trajectories.
- Workflow abstraction.
- Workflow text generation.
- Embedding generation.
- Compressed vector indexing.
- Semantic workflow retrieval.
- Lexical and metadata reranking.
- Long-term workflow storage.
- Latency and memory usage.

It does not test:

- Live website interaction.
- Browser automation.
- Real-time visual grounding.
- LLM action generation.
- End-to-end task success in a browser.

This distinction is important. The current best run proves that the memory layer is efficient and practical. It does not by itself prove full browser-agent success.

## End-To-End Architecture

The architecture has seven major stages:

1. Load recorded task trajectories.
2. Convert each trajectory into a structured workflow.
3. Abstract the workflow into a reusable memory representation.
4. Convert the workflow into embedding text.
5. Build compressed semantic memory using FAISS IVFPQ.
6. Retrieve candidate workflows for each query step.
7. Rerank, accept, and report workflow retrievals.

Each stage is described below.

## Stage 1: Loading Recorded Task Trajectories

The benchmark starts from recorded Mind2Web exemplars. Each exemplar represents a task trajectory from a website. A trajectory contains the task goal, website metadata, page observations, candidate elements, and gold actions.

A task trajectory contains information such as:

- Website name.
- Domain.
- Subdomain.
- User goal.
- Step observations.
- Candidate elements visible at each step.
- Gold action taken at each step.

The benchmark selected 500 trajectories and kept the first 2 steps from each trajectory. That produced 1000 step-level retrieval queries.

## Stage 2: Structured Workflow Construction

Each trajectory is converted into a structured workflow. A structured workflow is a machine-readable representation of how a task was completed.

A structured workflow contains:

- Workflow name.
- Website.
- Domain.
- Subdomain.
- Original task goal.
- Ordered steps.
- Action types.
- Element descriptions.
- Values used in actions.

The raw structured workflow remains close to the original trajectory. It is useful as evidence, but it is too specific to be maximally reusable. If a workflow only remembers exact element IDs or exact page-specific strings, it will not transfer well to new tasks.

## Stage 3: Workflow Abstraction

Workflow abstraction converts a specific recorded trajectory into a more reusable pattern.

The current best run used deterministic abstraction. This means the abstraction was produced by code rather than by an LLM. Deterministic abstraction was chosen because this benchmark is about memory performance and storage behavior. If LLM abstraction were used during the benchmark, API latency and rate limits would dominate the timing.

The abstraction process tries to preserve:

- The task intent.
- The action sequence.
- The role of each step.
- Website/domain context.
- Useful labels and values.

It tries to reduce dependence on:

- Exact element IDs.
- Overly specific page snapshots.
- Full HTML observations.
- Unnecessary low-level details.

The purpose is to produce workflows that can be reused for similar tasks, not only identical tasks.

## Stage 4: Workflow Text Generation

After abstraction, each workflow is converted into a compact text representation. This text is the representation used for embedding and reranking.

The workflow text usually contains:

- Workflow name.
- Website metadata.
- Domain and subdomain.
- Goal pattern.
- Abstract step sequence.
- Action types.
- Important labels or values.

This text is not the full workflow JSON. It is a compact textual summary designed to represent the meaning of the workflow.

This distinction matters:

- The full workflow JSON is the complete memory payload.
- The workflow text is the lightweight searchable representation.
- The vector embedding is the numerical semantic representation of that text.

## Stage 5: Embedding Generation

An embedding is a vector representation of text. The embedding model converts workflow text into a 384-dimensional numerical vector.

The idea is that semantically similar texts should have vectors that are close together. For example:

- "book a flight from Boston to Chicago"
- "search for airline tickets between two cities"

These should produce vectors closer to each other than unrelated tasks such as:

- "filter a shopping website by shoe size"
- "reserve a restaurant table"

Each workflow text is embedded once during index construction. Candidate workflow texts are not re-embedded during retrieval. They already have vectors stored inside the FAISS index.

## Stage 6: Vector Normalization And Cosine Similarity

Before insertion into FAISS, all workflow vectors are L2-normalized. The query vector is also L2-normalized at retrieval time.

After normalization, inner product search becomes equivalent to cosine similarity.

Cosine similarity measures the angle between two vectors. It is commonly used for semantic search because it focuses on direction rather than magnitude.

The retrieval logic is:

1. Embed query text.
2. Normalize query vector.
3. Search the FAISS index.
4. Retrieve vectors with the highest similarity scores.

Because both workflow vectors and query vectors are normalized, the FAISS inner-product score acts like a semantic similarity score.

## Stage 7: Compressed FAISS IVFPQ Indexing

The best run used FAISS IVFPQ.

IVFPQ combines two ideas:

1. IVF: inverted file indexing.
2. PQ: product quantization.

Together, they make vector search fast and compact.

## IVF: Inverted File Indexing

IVF clusters the vector space. Instead of searching every vector directly, the index groups vectors into clusters.

In the best run, the index used 22 clusters. With 500 workflows, this gives about 22 to 23 workflows per cluster on average.

At query time, FAISS first finds the nearest clusters. It then searches only the most relevant clusters instead of scanning the whole collection.

The number of clusters searched is controlled by nprobe. The run used a maximum of 8 searched clusters. So the search process is:

1. Find the closest coarse clusters to the query.
2. Search the top 8 clusters.
3. Return the best candidate vectors from those clusters.

This reduces retrieval work while keeping accuracy high enough for the benchmark.

## PQ: Product Quantization

Product quantization compresses each vector into a small code.

The original embedding has 384 dimensions. A normal float32 representation would require:

384 dimensions x 4 bytes = 1536 bytes per vector.

The best run used 16 product-quantization chunks with 8 bits per chunk.

That means:

16 chunks x 1 byte = 16 bytes per vector.

So each workflow vector code is compressed from 1536 bytes to 16 bytes. This is roughly a 96x compression of the vector representation.

This is why the benchmark reports a FAISS code size of 16 bytes per workflow vector.

## Long-Term Memory Storage Design

The long-term memory is split into four logical stores:

1. Full workflow records.
2. Embedding/reranking texts.
3. Metadata rows.
4. Compressed FAISS vector index.

This separation is intentional.

Full workflow records are larger and only needed when the agent wants to inspect or use a workflow in detail.

Embedding texts are smaller and useful for reranking, traces, and debugging.

Metadata connects vector-search results to workflow records.

The FAISS index stores compressed vector codes for semantic search.

This avoids putting large workflow payloads directly inside the vector index. It also avoids keeping all raw vectors as Python objects.

## Metadata Mapping

FAISS returns integer row IDs. It does not directly return workflow objects.

The metadata table maps each FAISS row ID to:

- Workflow name.
- Website.
- Domain.
- Subdomain.
- Goal pattern.
- Workflow text location.
- Full workflow record location.
- Creation order.

The retrieval path is:

1. FAISS returns a row ID.
2. The metadata row for that ID is read.
3. The workflow text is loaded for reranking.
4. The full workflow record can be loaded if needed.

This is a simple and efficient lookup design.

## Query-Time Retrieval

For each task step, the system constructs a query text using:

- Website.
- Domain.
- Subdomain.
- User task.
- Current observation.

This query text represents the current problem state.

The query text is embedded and normalized. FAISS then searches the compressed index and returns candidate workflows.

The first stage returns 50 candidate workflows. This is controlled by candidate_k.

The final returned list contains 5 workflows. This is controlled by top_k.

So:

- candidate_k controls retrieval breadth.
- top_k controls the final candidate list size.

## Semantic Scores

Semantic scores come directly from FAISS.

Workflow texts are embedded during index construction. Query text is embedded during retrieval. FAISS compares the query embedding against the stored compressed workflow embeddings.

Candidate texts are not embedded again during retrieval.

The semantic retrieval flow is:

1. Workflow text was embedded during build time.
2. Workflow embedding was inserted into FAISS.
3. Query text is embedded at retrieval time.
4. FAISS compares query vector to stored workflow vectors.
5. FAISS returns semantic scores and row IDs.

The candidate text is loaded after FAISS search, mainly for reranking and inspection.

## Reranking

FAISS gives strong semantic candidates, but raw vector similarity alone is not always enough. The system reranks candidates using multiple signals:

1. Semantic similarity.
2. Lexical overlap.
3. Same-website boost.
4. Same-domain boost.

The scoring formula is:

0.78 x semantic score
+ 0.12 x lexical score
+ 0.07 if same website
+ 0.03 if same domain.

Semantic similarity has the highest weight because the system should retrieve workflows by meaning, not just exact word matches.

Lexical overlap helps when important words match directly.

Website and domain boosts help prefer workflows from the same environment when multiple workflows are semantically similar.

## Lexical Reranking

The current lexical component is a lightweight token-overlap score. It is not a full BM25 engine.

The system compares words in the query against words in the candidate workflow text. If there is more overlap, the lexical score is higher.

This helps distinguish workflows that are semantically similar but differ in important surface details.

For example, multiple travel workflows may be semantically close:

- airline search
- hotel search
- car rental search

Lexical overlap can help identify which one shares the most concrete task terms.

## Metadata Boosts

Metadata boosts are small score additions for matching website or domain.

These boosts are useful because web workflows are often site-specific. A flight search workflow on one airline site may be more relevant to another step on the same airline site than a generic travel workflow from a different site.

The system does not strictly filter to the same website. It boosts same-website matches but still allows cross-website retrieval. This is important because similar workflows can transfer across websites.

The design avoids being too rigid.

## Acceptance Logic

After reranking, the top workflow is accepted if either:

- The combined score is at least 0.30.
- The semantic score is at least 0.28.

This gives two ways to accept a workflow:

1. Strong overall reranked match.
2. Strong semantic match even if metadata or lexical overlap is weaker.

In the best run, 998 of 1000 step queries accepted a workflow.

## Text Cache Optimization

Reranking requires loading candidate workflow texts. Loading text from disk repeatedly can become expensive, so the system uses an LRU cache.

LRU means least recently used. The cache keeps recently accessed workflow texts and evicts older entries when it reaches capacity.

The best run used 128 cache items and recorded:

- 36,938 cache hits.
- 13,062 cache misses.

This means a large number of candidate text reads were served from memory instead of disk.

The cache improves latency without changing the long-term memory design. The system still retrieves globally from the full FAISS index. The cache only speeds up repeated text loading.

## Why Batch Ingestion Was Used

Batch ingestion builds the memory first, then benchmarks retrieval.

This models a long-term memory system where workflows are accumulated and periodically indexed in the background.

Batch indexing has several advantages:

- It avoids rebuilding the vector index after every individual workflow.
- It allows FAISS to train IVFPQ once on the whole workflow set.
- It gives stable benchmark timing.
- It separates storage/indexing cost from retrieval cost.

For a production system, online ingestion could still exist, but it should likely write new workflows to a staging area and periodically merge them into the compressed long-term index.

## Why This Run Is Better Than The Earlier RAM Backend

The earlier RAM backend kept workflow vectors and embedding text as Python data structures. That made it simple, but not scalable.

The compressed FAISS IVFPQ backend improves this by:

- Removing raw Python vector storage.
- Compressing each vector to 16 bytes.
- Searching only relevant clusters.
- Keeping full workflow payloads on disk.
- Loading only candidate texts needed for reranking.
- Using cache for repeated candidate texts.

The result is lower latency and a more scalable memory architecture.

## Why This Run Is Better Than The SQ8 Compressed Run

The previous compressed FAISS run used scalar quantization with a flat scan. That compressed vectors but still searched all vectors.

The IVFPQ run adds clustering. Instead of scanning the entire compressed index, it searches only the most relevant clusters.

This improved retrieval latency:

- SQ8 average retrieval: 35.28 ms.
- IVFPQ average retrieval: 26.16 ms.

The tradeoff is that approximate search can miss a few matches:

- SQ8 accepted retrievals: 1000 / 1000.
- IVFPQ accepted retrievals: 998 / 1000.

For this run, the speed gain is worth the tiny drop in accepted retrieval count.

## Key Optimizations Used

The best run combines several optimizations:

1. Deterministic abstraction to avoid LLM latency during backend testing.
2. Batch ingestion to build the index efficiently.
3. Sentence embeddings for semantic retrieval.
4. L2 normalization so inner product behaves like cosine similarity.
5. FAISS IVFPQ for compressed approximate nearest-neighbor search.
6. Product quantization to reduce vector code size.
7. IVF clustering to avoid scanning every vector.
8. Candidate reranking to improve final workflow selection.
9. Metadata boosts to prefer contextually relevant workflows.
10. Lightweight lexical overlap for surface-term correction.
11. Disk-backed workflow payloads to keep large records out of RAM.
12. Metadata mapping to connect FAISS row IDs to workflow records.
13. LRU text cache to reduce repeated disk reads.
14. Separate benchmark child processes to keep memory measurements clean.

## Limitations

The current best run has important limitations:

- It is not live browser execution.
- It does not test visual grounding.
- It does not test an LLM agent making final actions.
- The abstraction is deterministic, not fully semantic.
- The lexical reranker is lightweight, not true BM25.
- The benchmark uses recorded trajectories.
- Workflow reuse quality depends heavily on abstraction quality.

These limitations do not invalidate the result. They define what the result proves. The run proves that the workflow memory and retrieval architecture is efficient. It does not yet prove full autonomous web task success.

## Best Interpretation

The current system has a strong long-term workflow memory backend.

The memory system can:

- Store hundreds of workflows compactly.
- Retrieve relevant workflows quickly.
- Avoid raw Python vector memory.
- Use semantic similarity instead of exact keyword matching.
- Keep full workflow payloads outside the vector index.
- Rerank candidates using semantic, lexical, and metadata signals.

The most important achievement is the move from a RAM-heavy prototype to a compressed, disk-backed, FAISS-indexed long-term memory system.

## Next Technical Priorities

The storage and retrieval architecture is now strong enough that the next improvements should focus on quality:

1. Improve workflow abstraction using LLM-based offline abstraction.
2. Add true BM25 or another stronger lexical retrieval signal.
3. Add duplicate workflow merging.
4. Add a stronger reranker for the top candidate workflows.
5. Test larger workflow banks.
6. Connect the retrieved workflows back into an LLM action-generation loop.
7. Evaluate on live or semi-live browser tasks.

The best next step is not another storage backend. The best next step is better workflow quality and better workflow selection.

