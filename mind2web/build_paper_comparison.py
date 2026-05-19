"""Build a Markdown comparison between a local Mind2Web run and AWM paper results."""

from __future__ import annotations

from pathlib import Path
import argparse
import json


PAPER_RESULTS = [
    {
        "setting": "Mind2Web cross-task",
        "method": "MindAct GPT-4 baseline",
        "element_accuracy": 41.6,
        "action_f1": 60.6,
        "step_success_rate": 36.2,
        "task_success_rate": 2.0,
    },
    {
        "setting": "Mind2Web cross-task",
        "method": "AWM GPT-4 offline",
        "element_accuracy": 50.6,
        "action_f1": 57.3,
        "step_success_rate": 45.1,
        "task_success_rate": 4.8,
    },
    {
        "setting": "Mind2Web cross-website",
        "method": "AWM GPT-4 online",
        "element_accuracy": 42.1,
        "action_f1": 45.1,
        "step_success_rate": 33.9,
        "task_success_rate": 1.6,
    },
    {
        "setting": "Mind2Web cross-domain",
        "method": "AWM GPT-4 online",
        "element_accuracy": 40.9,
        "action_f1": 46.3,
        "step_success_rate": 35.5,
        "task_success_rate": 1.7,
    },
]


def metric(metrics: dict, name: str) -> float:
    return float(metrics.get(name + "_pct", metrics.get(name, 0.0)))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--label", type=str, default="Local run")
    args = parser.parse_args()

    metrics_path = args.run_dir / "paper_metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())

    local = {
        "element_accuracy": metric(metrics, "element_accuracy"),
        "relaxed_element_accuracy": metric(metrics, "relaxed_element_accuracy"),
        "operation_accuracy": metric(metrics, "operation_accuracy"),
        "action_f1": metric(metrics, "action_f1"),
        "step_success_rate": metric(metrics, "step_success_rate"),
        "relaxed_step_success_rate": metric(metrics, "relaxed_step_success_rate"),
        "task_success_rate": metric(metrics, "task_success_rate"),
        "relaxed_task_success_rate": metric(metrics, "relaxed_task_success_rate"),
        "exact_sequence_rate": metric(metrics, "exact_sequence_rate"),
    }

    best_paper_step = max(item["step_success_rate"] for item in PAPER_RESULTS)
    best_paper_task = max(item["task_success_rate"] for item in PAPER_RESULTS)

    lines = [
        "# Local Mind2Web vs AWM Paper Comparison",
        "",
        f"Local run: `{args.label}`",
        "",
        "## Local Metrics",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Element Accuracy | {local['element_accuracy']:.2f}% |",
        f"| Relaxed Element Accuracy | {local['relaxed_element_accuracy']:.2f}% |",
        f"| Operation Accuracy | {local['operation_accuracy']:.2f}% |",
        f"| Action F1 | {local['action_f1']:.2f}% |",
        f"| Step Success Rate | {local['step_success_rate']:.2f}% |",
        f"| Relaxed Step Success Rate | {local['relaxed_step_success_rate']:.2f}% |",
        f"| Task Success Rate | {local['task_success_rate']:.2f}% |",
        f"| Relaxed Task Success Rate | {local['relaxed_task_success_rate']:.2f}% |",
        f"| Exact Sequence Rate | {local['exact_sequence_rate']:.2f}% |",
        "",
        "## Paper Reference Results",
        "",
        "| Setting | Method | Elem Acc | Action F1 | Step SR | Task SR |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for item in PAPER_RESULTS:
        lines.append(
            "| {setting} | {method} | {ea:.1f}% | {af1:.1f}% | {step:.1f}% | {task:.1f}% |".format(
                setting=item["setting"],
                method=item["method"],
                ea=item["element_accuracy"],
                af1=item["action_f1"],
                step=item["step_success_rate"],
                task=item["task_success_rate"],
            )
        )

    lines.extend(
        [
            "",
            "## Difference From Best Paper Reference",
            "",
            "| Metric | Local | Best Paper Reference | Difference |",
            "| --- | ---: | ---: | ---: |",
            f"| Step SR | {local['step_success_rate']:.2f}% | {best_paper_step:.2f}% | {local['step_success_rate'] - best_paper_step:+.2f} |",
            f"| Task SR | {local['task_success_rate']:.2f}% | {best_paper_task:.2f}% | {local['task_success_rate'] - best_paper_task:+.2f} |",
            f"| Relaxed Step SR | {local['relaxed_step_success_rate']:.2f}% | {best_paper_step:.2f}% | {local['relaxed_step_success_rate'] - best_paper_step:+.2f} |",
            f"| Relaxed Task SR | {local['relaxed_task_success_rate']:.2f}% | {best_paper_task:.2f}% | {local['relaxed_task_success_rate'] - best_paper_task:+.2f} |",
            "",
            "## Important Notes",
            "",
            "- This comparison is only meaningful if the local run used `--mode llm` on held-out observations.",
            "- Strict metrics are closest to the paper; relaxed metrics are local diagnostics for no-browser/static-observation runs.",
            "- `--mode oracle` is a sanity check and should not be presented as model performance.",
            "- `--mode heuristic` is a no-LLM baseline; it is useful for debugging element matching, but it is not the paper's AWM agent.",
            "- For a defensible comparison, run enough tasks with the same model family and split assumptions clearly stated.",
        ]
    )

    output_path = args.run_dir / "paper_comparison.md"
    output_path.write_text("\n".join(lines) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
