# Local Mind2Web vs AWM Paper Comparison

Local run: `Vertex full relaxed diagnostics`

## Local Metrics

| Metric | Value |
| --- | ---: |
| Element Accuracy | 29.85% |
| Relaxed Element Accuracy | 44.72% |
| Operation Accuracy | 43.15% |
| Action F1 | 41.54% |
| Step Success Rate | 27.31% |
| Relaxed Step Success Rate | 41.07% |
| Task Success Rate | 2.18% |
| Relaxed Task Success Rate | 4.76% |
| Exact Sequence Rate | 0.99% |

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
| Step SR | 27.31% | 45.10% | -17.79 |
| Task SR | 2.18% | 4.80% | -2.62 |
| Relaxed Step SR | 41.07% | 45.10% | -4.03 |
| Relaxed Task SR | 4.76% | 4.80% | -0.04 |

## Important Notes

- This comparison is only meaningful if the local run used `--mode llm` on held-out observations.
- Strict metrics are closest to the paper; relaxed metrics are local diagnostics for no-browser/static-observation runs.
- `--mode oracle` is a sanity check and should not be presented as model performance.
- `--mode heuristic` is a no-LLM baseline; it is useful for debugging element matching, but it is not the paper's AWM agent.
- For a defensible comparison, run enough tasks with the same model family and split assumptions clearly stated.
