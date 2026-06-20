# Agent Workflow Memory: Local Mind2Web 500-Task Evaluation Report

## Summary

This report documents the local Agent Workflow Memory implementation and the 500-task Mind2Web evaluation run using Vertex/Gemini. The system implements workflow memory extraction, hybrid workflow retrieval, structured workflow prompting, LLM-based step prediction, and paper-style metric evaluation.

The 500-task run produced strong strict paper-style metrics:

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

The main result is that the local Vertex-backed AWM runner achieves higher strict Step Success Rate and Task Success Rate than the AWM paper reference numbers listed below, under our local exemplar-based evaluation setup.

## Evaluation Setup

### Data

The evaluation uses real Mind2Web exemplar trajectories stored in:

```text
mind2web/data/memory/exemplars.json
```

Each exemplar contains:

- Website, domain, and subdomain metadata.
- A natural-language task instruction.
- A sequence of HTML-like observations.
- A sequence of ground-truth Mind2Web actions.

For this run, the first 500 exemplar tasks were evaluated:

```text
Tasks: 500
Steps: 3790
```

### Model

The step-prediction runner was configured to use a Vertex/Gemini model through the OpenAI-compatible provider adapter. The run label is:

```text
Vertex Gemini 2.5 Pro 500-task run
```

### Command

The run was launched with:

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

The comparison report was generated with:

```bash
python3 mind2web/build_paper_comparison.py \
  --run-dir mind2web/step_eval_vertex_500 \
  --label "Vertex Gemini 2.5 Pro 500-task run"
```

## Architecture Implemented

### 1. Provider Integration

We added provider configuration utilities so the code can use OpenAI-compatible APIs and Vertex/Gemini without hard-coding credentials.

Implemented in:

```text
mind2web/utils/provider_config.py
webarena/provider_config.py
```

This supports:

- Vertex AI via Google Application Default Credentials or access token.
- Gemini API.
- NVIDIA NIM.
- OpenAI-compatible base URLs.
- Local `.env.local` loading.

### 2. Local AWM Demonstration Harness

We implemented a local AWM demo for WebArena-style workflows:

```text
webarena/local_awm_full_demo.py
```

This introduced:

- Working memory in the prompt.
- Short-term action history.
- Episodic memory.
- Long-term workflow memory.
- LLM trace recording.
- Workflow induction from successful trajectories.
- Workflow embedding storage.

This was used to validate the AWM loop before moving to real Mind2Web data.

### 3. Hybrid Workflow Retrieval

Workflow retrieval was upgraded from deterministic matching to hybrid semantic/lexical retrieval.

Implemented in:

```text
webarena/local_awm_full_demo.py
```

Retrieval components:

| Component | Purpose |
| --- | --- |
| SentenceTransformers | Encodes workflows and task queries into dense vectors |
| FAISS | Fast vector search over workflow embeddings |
| BM25 | Lexical similarity backup |
| Combined score | Weighted ranking of semantic and lexical scores |

Scoring:

```text
combined_score = 0.8 * semantic_score + 0.2 * bm25_score
```

The default embedding model is:

```text
sentence-transformers/all-MiniLM-L6-v2
```

The system also loads the embedding model from the local Hugging Face cache when available, avoiding repeated network downloads.

### 4. Real Mind2Web Workflow Replay

We added scripts to use the real Mind2Web exemplar trajectories:

```text
mind2web/real_data_awm_smoke.py
mind2web/real_data_awm_full_run.py
```

These scripts:

- Load real Mind2Web exemplars.
- Extract workflows from solved trajectories.
- Store workflows in long-term memory.
- Build workflow embeddings.
- Retrieve workflows for later tasks.
- Store episodic memory and traces.

This validated the memory and retrieval pipeline over real benchmark-style data.

### 5. Structured Workflow Extraction

We then moved from plain workflow text to structured workflows.

Implemented in:

```text
mind2web/llm_step_eval.py
```

Each workflow step stores:

- Operation type, such as `CLICK`, `TYPE`, or `SELECT`.
- Original element id.
- Target role.
- Target label.
- Example value.
- Original action string.
- Source observation.

This allows workflows to act as reusable procedural guidance rather than literal copied action scripts.

### 6. Step-Level LLM Prediction

The main evaluator is:

```text
mind2web/llm_step_eval.py
```

For each task step, it:

1. Builds a task query from website, domain, subdomain, task, and current observation.
2. Retrieves relevant workflows from workflow memory.
3. Applies a same-website reuse policy.
4. Parses candidate elements from the current observation.
5. Sends task, observation, candidate elements, and retrieved workflow to Vertex/Gemini.
6. Parses the predicted action.
7. Compares the prediction to the ground-truth action.
8. Adds the completed exemplar trajectory as a new workflow for later tasks.

### 7. Same-Website Reuse Policy

The same-website policy accepts a retrieved workflow only if it came from the same website as the current task.

Example:

```text
Current task website: united
Retrieved workflow: united / Airlines workflow 10
Accepted: yes
```

```text
Current task website: enterprise
Retrieved workflow: exploretock / Restaurant workflow 2
Accepted: no
```

This keeps workflow reuse conservative and avoids injecting misleading cross-site workflows.

### 8. Candidate Element Parsing

The evaluator parses candidate elements from the current HTML-like observation.

For each candidate, it stores:

- Element id.
- Tag.
- Role hint.
- Label.
- Nearby text/context.

Nearby text was added because many Mind2Web gold actions target elements such as `svg`, `span`, or unlabeled controls. Without nearby context, the model sees many candidates as blank elements.

### 9. Output Parsing And Reliability Fixes

Several fixes were needed for stable LLM evaluation:

- Normalize `CLICK 123` to `CLICK [123]`.
- Normalize `TYPE 11031 Allan` to `TYPE [11031] [Allan]`.
- Increase model output budget to avoid truncated JSON.
- Put the `action` field first in the requested JSON output.
- Add LLM retry and sleep settings.
- Add parallel task-level prediction.

These fixes were important because earlier runs failed mainly due to incomplete or malformed outputs rather than reasoning errors.

### 10. Parallel Evaluation

The runner supports task-level parallelism:

```text
--parallel-workers 9
```

The online memory/retrieval plan is still built sequentially, because task N depends on workflows learned from earlier tasks. After that, prediction is parallelized across tasks. Steps inside a task remain sequential so previous predicted actions are still available as context.

## Artifacts Produced

The 500-task run output is stored in:

```text
mind2web/step_eval_vertex_500
```

Important files:

| File | Purpose |
| --- | --- |
| `paper_metrics.md` | Main strict metric table |
| `paper_metrics.json` | Machine-readable metric output |
| `paper_comparison.md` | Comparison against paper reference metrics |
| `prediction_trace.json` | Full per-step trace with observations, candidates, retrieved workflows, raw LLM outputs, predictions, and gold actions |
| `workflow_memory.txt` | Human-readable induced workflow memory |
| `structured_workflows.json` | Structured workflow memory |
| `workflow_embeddings.json` | Embedding index metadata and workflow vectors |

## Results

### Overall Strict Metrics

| Metric | Value |
| --- | ---: |
| Tasks | 500 |
| Steps | 3790 |
| Element Accuracy | 60.11% |
| Operation Accuracy | 91.69% |
| Action F1 | 88.73% |
| Step Success Rate | 55.12% |
| Task Success Rate | 10.40% |
| Exact Sequence Rate | 9.80% |

### Error Breakdown

| Category | Steps | Percentage |
| --- | ---: | ---: |
| Exact step correct | 2089 | 55.12% |
| Right operation, wrong element | 1255 | 33.11% |
| Wrong operation | 314 | 8.28% |
| Right element, wrong value | 131 | 3.46% |
| Empty or unparsed output | 1 | 0.03% |

The largest remaining error category is exact element selection. The model usually understands the correct action type, but it sometimes chooses a nearby or alternative element id.

### Breakdown By Operation

| Operation | Steps | Step Success Rate |
| --- | ---: | ---: |
| CLICK | 3194 | 55.92% |
| TYPE | 436 | 48.62% |
| SELECT | 160 | 56.88% |

The lowest strict success rate is for `TYPE`, mostly because values often differ in formatting or specificity. For example, the gold value may be `Boston`, while the model predicts `Boston, NY`.

### Breakdown By Domain

| Domain | Steps | Step Success Rate |
| --- | ---: | ---: |
| Travel | 1916 | 55.85% |
| Shopping | 1132 | 52.92% |
| Entertainment | 742 | 56.60% |

Performance is relatively balanced across the three domains.

## Comparison With The AWM Paper

The table below compares our strict metrics with the paper reference metrics. This section intentionally does not include relaxed local diagnostics.

| Setting | Method | Element Accuracy | Action F1 | Step SR | Task SR |
| --- | --- | ---: | ---: | ---: | ---: |
| Local 500-task run | Vertex/Gemini AWM runner | 60.11% | 88.73% | 55.12% | 10.40% |
| Mind2Web cross-task | MindAct GPT-4 baseline | 41.60% | 60.60% | 36.20% | 2.00% |
| Mind2Web cross-task | AWM GPT-4 offline | 50.60% | 57.30% | 45.10% | 4.80% |
| Mind2Web cross-website | AWM GPT-4 online | 42.10% | 45.10% | 33.90% | 1.60% |
| Mind2Web cross-domain | AWM GPT-4 online | 40.90% | 46.30% | 35.50% | 1.70% |

Against the strongest paper reference in this table:

| Metric | Local Run | Best Paper Reference | Difference |
| --- | ---: | ---: | ---: |
| Step Success Rate | 55.12% | 45.10% | +10.02 |
| Task Success Rate | 10.40% | 4.80% | +5.60 |

## Interpretation

The 500-task run substantially improves over the paper reference numbers in this local evaluation setup. The model is especially strong on operation selection and action value prediction:

```text
Operation Accuracy: 91.69%
Action F1: 88.73%
```

This means the model usually understands what type of action should happen next and what value should be typed or selected.

The main weakness is strict element id selection:

```text
Right operation, wrong element: 33.11%
```

This is expected in part because this evaluation is performed over static HTML-like observations rather than a live browser UI. Many Mind2Web actions target nested ids such as `svg` or `span`, while the model often chooses the surrounding button, link, or visible control. Strict scoring treats these as wrong even when the semantic click target is close.

## Differences From The Paper

Although we compare against the paper metrics, the setup is not an exact reproduction.

Key differences:

1. The paper uses official Mind2Web splits: `train`, `test_task`, `test_website`, and `test_domain`.
2. This run uses the available local `exemplars.json` trajectories.
3. The paper uses GPT-4 and GPT-3.5 variants; this run uses Vertex/Gemini.
4. The paper's workflow induction prompts and environment filtering differ from our implementation.
5. This run performs step prediction over saved observations, not live browser interaction.
6. The local runner uses our hybrid retrieval implementation with SentenceTransformers, FAISS, and BM25.

Therefore, the comparison should be presented as a local implementation comparison against paper reference numbers, not as an official benchmark reproduction.

## What Worked Well

- Vertex/Gemini integration is functional.
- The output parsing issues from earlier runs were resolved.
- Workflow retrieval and same-website memory reuse are stable.
- Structured workflows provide useful procedural guidance.
- Hybrid semantic and lexical workflow retrieval works at scale.
- Parallel evaluation with 9 workers completed the 500-task run.
- Strict Step SR and Task SR exceed the paper reference numbers in this local setup.

## Remaining Limitations

- The evaluation is not the official Mind2Web split protocol.
- The environment is static observations, not live websites.
- Strict element id matching can penalize semantically close parent/child element choices.
- Candidate element extraction is still simpler than the paper's full Mind2Web preprocessing.
- The run covers 500 tasks, not all available exemplars.
- Results may vary with task order, model version, and Vertex configuration.

## Recommended Next Steps

1. Run a second 500-task sample with a different start index to test stability.
2. Add DOM parent-child equivalence tracking for better analysis of element mismatches.
3. Improve candidate extraction for nested `svg`, `span`, and unlabeled controls.
4. Add workflow reranking before prompting.
5. If official comparison is required, download the official Mind2Web splits and run the same evaluator on `test_task`, `test_website`, and `test_domain`.

## Bottom Line

The implemented system demonstrates a working AWM-style memory architecture over Mind2Web trajectories with strong 500-task strict metrics:

```text
Element Accuracy: 60.11%
Action F1: 88.73%
Step SR: 55.12%
Task SR: 10.40%
```

These numbers exceed the paper reference metrics included in the report, while still requiring careful qualification because this is a local exemplar-based evaluation rather than an official reproduction of the paper's benchmark protocol.

