<div align="center">
  <h1>Agent Workflow Memory: Local Mind2Web Evaluation</h1>
  <a href="https://arxiv.org/abs/2409.07429">
    <img src="https://img.shields.io/badge/arXiv-2409.07429-b31b1b.svg" alt="arXiv">
  </a>
</div>

This repository contains a local implementation and evaluation harness for **Agent Workflow Memory (AWM)**, based on the paper [Agent Workflow Memory](https://arxiv.org/abs/2409.07429).

The implementation adds a practical local pipeline around the original codebase:

- workflow extraction from real Mind2Web exemplar trajectories,
- long-term workflow memory,
- short-term per-task action memory,
- hybrid workflow retrieval with SentenceTransformers, FAISS, and BM25,
- Vertex/Gemini step prediction,
- paper-style metrics,
- detailed agent traces and workflow artifacts.

The main completed evaluation is a **500-task Mind2Web exemplar run** using Vertex/Gemini.

## Headline Result

Run folder:

```text
mind2web/step_eval_vertex_500
```

Strict paper-style metrics:

| Metric | Value |
| --- | ---: |
| Tasks evaluated | 500 |
| Steps evaluated | 3790 |
| Element Accuracy | 60.11% |
| Operation Accuracy | 91.69% |
| Action F1 | 88.73% |
| Step Success Rate | 55.12% |
| Task Success Rate | 10.40% |
| Exact Sequence Rate | 9.80% |

Comparison with paper reference metrics:

| Setting | Method | Element Accuracy | Action F1 | Step SR | Task SR |
| --- | --- | ---: | ---: | ---: | ---: |
| Local 500-task run | Vertex/Gemini AWM runner | 60.11% | 88.73% | 55.12% | 10.40% |
| Mind2Web cross-task | MindAct GPT-4 baseline | 41.60% | 60.60% | 36.20% | 2.00% |
| Mind2Web cross-task | AWM GPT-4 offline | 50.60% | 57.30% | 45.10% | 4.80% |
| Mind2Web cross-website | AWM GPT-4 online | 42.10% | 45.10% | 33.90% | 1.60% |
| Mind2Web cross-domain | AWM GPT-4 online | 40.90% | 46.30% | 35.50% | 1.70% |

Against the strongest paper reference in the table:

| Metric | Local Run | Best Paper Reference | Difference |
| --- | ---: | ---: | ---: |
| Step Success Rate | 55.12% | 45.10% | +10.02 |
| Task Success Rate | 10.40% | 4.80% | +5.60 |

Important qualification: this is a **local exemplar-based evaluation**, not an official reproduction of the paper's exact Mind2Web split protocol.

## What Was Implemented

### Provider Integration

Provider configuration was added for OpenAI-compatible APIs and Vertex/Gemini:

```text
mind2web/utils/provider_config.py
webarena/provider_config.py
```

Supported provider styles include:

- Vertex AI through Google Application Default Credentials or access token,
- Gemini API,
- NVIDIA NIM,
- OpenAI-compatible base URLs,
- local `.env.local` loading.

### Hybrid Workflow Retrieval

Workflow retrieval now uses a hybrid semantic and lexical index:

```text
SentenceTransformers embeddings
+ FAISS vector search
+ BM25 lexical scoring
```

Default embedding model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Scoring:

```text
combined_score = 0.8 * semantic_score + 0.2 * bm25_score
```

Implementation:

```text
webarena/local_awm_full_demo.py
```

### Real Mind2Web Workflow Replay

Scripts were added to use the real Mind2Web exemplar trajectories:

```text
mind2web/real_data_awm_smoke.py
mind2web/real_data_awm_full_run.py
```

These scripts load real trajectories from:

```text
mind2web/data/memory/exemplars.json
```

and produce:

- workflow memory,
- episodic memory,
- retrieval traces,
- workflow embeddings,
- human-readable reports.

### Structured Workflow Extraction

The step evaluator extracts structured workflows from solved trajectories.

Each workflow step stores:

- operation type,
- target element id,
- target role,
- target label,
- example value,
- original action string,
- source observation.

Implementation:

```text
mind2web/llm_step_eval.py
```

### Agentic Step Prediction

For each task step, the runner:

1. Builds a query from website, domain, subdomain, task, and current observation.
2. Retrieves relevant long-term workflows.
3. Applies a same-website reuse policy.
4. Parses candidate elements from the current observation.
5. Sends the task, observation, candidate elements, and retrieved workflow to Vertex/Gemini.
6. Parses the predicted action.
7. Compares the prediction with the ground-truth Mind2Web action.
8. Stores the completed trajectory as a new workflow for future tasks.

Main runner:

```text
mind2web/llm_step_eval.py
```

### Parallel Evaluation

The evaluator supports task-level parallelism:

```text
--parallel-workers 9
```

The memory/retrieval plan is built sequentially because online AWM memory grows task by task. Prediction is then parallelized across tasks. Steps inside each task remain sequential.

## Memory Architecture

The implementation uses three practical memory layers:

| Memory Type | Meaning | Where To Inspect |
| --- | --- | --- |
| Short-term memory | Previous predicted actions inside the current task | `prediction_trace.json` |
| Long-term workflow memory | Workflows induced from previous tasks | `workflow_memory.txt`, `structured_workflows.json` |
| Retrieval memory | Embeddings and lexical index for workflow search | `workflow_embeddings.json` |

The agent does not manually choose between short-term and long-term memory. The pipeline retrieves long-term workflow candidates and includes both retrieved workflows and previous actions in the prompt. The LLM then uses that context to predict the next action.

## Main Artifacts

The 500-task run is stored in:

```text
mind2web/step_eval_vertex_500
```

Important files:

| File | Purpose |
| --- | --- |
| `final_report.md` | Detailed report for the 500-task run |
| `paper_metrics.md` | Strict paper-style metric table |
| `paper_metrics.json` | Machine-readable metrics |
| `paper_comparison.md` | Comparison against paper reference metrics |
| `prediction_trace.json` | Full per-step agentic trace |
| `workflow_memory.txt` | Human-readable workflow memory |
| `structured_workflows.json` | Structured workflow memory |
| `workflow_embeddings.json` | Workflow embedding index |

The trace records, for each step:

- current observation,
- candidate elements,
- retrieved workflow candidates,
- accepted workflow,
- gold action,
- predicted action,
- raw LLM output,
- parsed LLM output.

## Running The 500-Task Evaluation

From the repository root:

```bash
python3 mind2web/llm_step_eval.py \
  --mode llm \
  --model google/gemini-2.5-pro \
  --output-dir mind2web/step_eval_vertex_500 \
  --max-tasks 500 \
  --parallel-workers 9 \
  --max-output-tokens 1024 \
  --reuse-policy same-website \
  --save-every 25 \
  --llm-retries 3 \
  --retry-sleep 10
```

Generate the comparison report:

```bash
python3 mind2web/build_paper_comparison.py \
  --run-dir mind2web/step_eval_vertex_500 \
  --label "Vertex Gemini 2.5 Pro 500-task run"
```

Inspect metrics:

```bash
cat mind2web/step_eval_vertex_500/paper_metrics.md
cat mind2web/step_eval_vertex_500/paper_comparison.md
```

## Error Analysis

Strict error breakdown for the 500-task run:

| Category | Steps | Percentage |
| --- | ---: | ---: |
| Exact step correct | 2089 | 55.12% |
| Right operation, wrong element | 1255 | 33.11% |
| Wrong operation | 314 | 8.28% |
| Right element, wrong value | 131 | 3.46% |
| Empty or unparsed output | 1 | 0.03% |

The main remaining issue is exact element id selection. The model usually understands the correct operation and value, but sometimes chooses a nearby or alternative element id.

Breakdown by operation:

| Operation | Steps | Step Success Rate |
| --- | ---: | ---: |
| CLICK | 3194 | 55.92% |
| TYPE | 436 | 48.62% |
| SELECT | 160 | 56.88% |

Breakdown by domain:

| Domain | Steps | Step Success Rate |
| --- | ---: | ---: |
| Travel | 1916 | 55.85% |
| Shopping | 1132 | 52.92% |
| Entertainment | 742 | 56.60% |

## Differences From The Paper

This repository implements the AWM idea locally, but it is not an exact reproduction of the paper's full benchmark setup.

Key differences:

1. The paper uses official Mind2Web splits: `train`, `test_task`, `test_website`, and `test_domain`.
2. This evaluation uses the available local `exemplars.json` trajectories.
3. The paper uses GPT-4 and GPT-3.5 variants; this run uses Vertex/Gemini.
4. The paper's workflow induction prompts and environment filtering differ from this implementation.
5. This run performs step prediction over saved observations rather than live browser interaction.
6. This implementation uses hybrid retrieval with SentenceTransformers, FAISS, and BM25.

The reported comparison should therefore be read as a local implementation comparison against paper reference metrics, not as an official leaderboard result.

## Repository Layout

```text
mind2web/
  llm_step_eval.py                 # main Vertex/Gemini step evaluator
  build_paper_comparison.py        # paper comparison report generator
  real_data_awm_smoke.py           # small real-data retrieval smoke test
  real_data_awm_full_run.py        # full exemplar memory replay
  evaluate_full_run_metrics.py     # diagnostic evaluator
  data/memory/exemplars.json       # real Mind2Web exemplar trajectories
  step_eval_vertex_500/            # main 500-task run artifacts

webarena/
  local_awm_full_demo.py           # local AWM demo and hybrid retriever
  provider_config.py               # provider adapter
  pipeline.py                      # WebArena pipeline wrapper
  induce_rule.py                   # workflow induction utilities
```

## Limitations

- The 500-task evaluation is not the official Mind2Web split protocol.
- The system evaluates saved observations, not live browser states.
- Exact element id matching can penalize semantically close parent/child element choices.
- Candidate extraction is simpler than the original Mind2Web preprocessing.
- Results can vary with model version, task order, and provider settings.

## Citation

```bibtex
@inproceedings{awm2024wang,
  title = {Agent Workflow Memory},
  author = {Wang, Zhiruo and Mao, Jiayuan and Fried, Daniel and Neubig, Graham},
  journal = {arXiv preprint arXiv:2409.07429},
  year = {2024}
}
```

