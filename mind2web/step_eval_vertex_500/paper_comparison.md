# Local Mind2Web vs AWM Paper Comparison

Local run: `Vertex Gemini 2.5 Pro 500-task run`

## Local Metrics

| Metric | Value |
| --- | ---: |
| Element Accuracy | 60.11% |
| Relaxed Element Accuracy | 95.99% |
| Operation Accuracy | 91.69% |
| Action F1 | 88.73% |
| Step Success Rate | 55.12% |
| Relaxed Step Success Rate | 85.41% |
| Task Success Rate | 10.40% |
| Relaxed Task Success Rate | 45.40% |
| Exact Sequence Rate | 9.80% |

## Paper Reference Results

| Setting | Method | Elem Acc | Action F1 | Step SR | Task SR |
| --- | --- | ---: | ---: | ---: | ---: |
| Mind2Web cross-task | MindAct GPT-4 baseline | 41.6% | 60.6% | 36.2% | 2.0% |
| Mind2Web cross-task | AWM GPT-4 offline | 50.6% | 57.3% | 45.1% | 4.8% |
| Mind2Web cross-website | AWM GPT-4 online | 42.1% | 45.1% | 33.9% | 1.6% |
| Mind2Web cross-domain | AWM GPT-4 online | 40.9% | 46.3% | 35.5% | 1.7% |

## Difference From Best Paper Reference

| Metric | Local | Best Paper Reference | Difference |
| --- | ---: | ---: | ---: |
| Step SR | 55.12% | 45.10% | +10.02 |
| Task SR | 10.40% | 4.80% | +5.60 |
| Relaxed Step SR | 85.41% | 45.10% | +40.31 |
| Relaxed Task SR | 45.40% | 4.80% | +40.60 |

## Important Notes

- This comparison is only meaningful if the local run used `--mode llm` on held-out observations.
- Strict metrics are closest to the paper; relaxed metrics are local diagnostics for no-browser/static-observation runs.
- `--mode oracle` is a sanity check and should not be presented as model performance.
- `--mode heuristic` is a no-LLM baseline; it is useful for debugging element matching, but it is not the paper's AWM agent.
- For a defensible comparison, run enough tasks with the same model family and split assumptions clearly stated.
