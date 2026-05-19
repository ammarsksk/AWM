"""Compute paper-style metrics for the local Mind2Web full run.

The official Mind2Web metrics require model-predicted actions. Our full run is
an offline replay over ground-truth exemplars, so this evaluator reports:

- oracle: predicted actions are exactly the exemplar actions. This should be
  100% and only verifies that the metric code and parsed data are coherent.
- accepted_workflow_proxy: predicted actions are copied from the accepted reused
  workflow, if one exists.
- top_workflow_proxy: predicted actions are copied from the top retrieved
  workflow, even if the reuse policy rejected it.

The proxy modes are not official Mind2Web benchmark scores. They are useful
diagnostics for how directly a retrieved workflow's action sequence matches a
new task trajectory before an LLM adapts ids and values.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import re
from collections import Counter


ACTION_BLOCK_RE = re.compile(r"<action>\s*(.*?)\s*</action>", re.DOTALL)
ACTION_RE = re.compile(
    r"^(?P<op>[A-Z_]+)(?:\s+\[(?P<element>[^\]]+)\])?(?:\s+\[(?P<value>[^\]]*)\])?"
)


def parse_workflow_actions(path: Path) -> dict[str, list[str]]:
    text = path.read_text()
    workflows = {}
    for block in re.split(r"\n(?=## )", text):
        block = block.strip()
        if not block:
            continue
        name = block.splitlines()[0].lstrip("#").strip()
        actions = [
            " ".join(match.group(1).split())
            for match in ACTION_BLOCK_RE.finditer(block)
        ]
        workflows[name] = actions
    return workflows


def parse_action(action: str | None) -> dict[str, str]:
    if not action:
        return {"op": "", "element": "", "value": ""}
    match = ACTION_RE.match(action.strip())
    if not match:
        return {"op": "", "element": "", "value": ""}
    return {
        "op": (match.group("op") or "").strip(),
        "element": (match.group("element") or "").strip(),
        "value": (match.group("value") or "").strip(),
    }


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = prediction.lower().split()
    gold_tokens = gold.lower().split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def action_value_f1(pred: dict[str, str], gold: dict[str, str]) -> float:
    if pred["op"] != gold["op"]:
        return 0.0
    if gold["op"] in {"TYPE", "SELECT"}:
        return token_f1(pred["value"], gold["value"])
    return 1.0


def evaluate_predictions(trace: list[dict], predictions_by_task: list[list[str]]) -> dict:
    total_steps = 0
    element_correct = 0
    op_correct = 0
    action_f1_sum = 0.0
    step_success = 0
    task_success = 0
    exact_sequence = 0
    predicted_steps = 0
    length_matches = 0

    for item, predicted_actions in zip(trace, predictions_by_task):
        gold_actions = item.get("ground_truth_actions", [])
        predicted_steps += len(predicted_actions)
        length_matches += int(len(predicted_actions) == len(gold_actions))
        task_ok = bool(gold_actions) and len(predicted_actions) == len(gold_actions)
        exact_ok = predicted_actions == gold_actions

        for index, gold_action in enumerate(gold_actions):
            pred_action = predicted_actions[index] if index < len(predicted_actions) else None
            gold = parse_action(gold_action)
            pred = parse_action(pred_action)

            total_steps += 1
            element_match = bool(gold["element"]) and pred["element"] == gold["element"]
            op_match = bool(gold["op"]) and pred["op"] == gold["op"]
            act_f1 = action_value_f1(pred, gold)
            step_ok = element_match and act_f1 == 1.0

            element_correct += int(element_match)
            op_correct += int(op_match)
            action_f1_sum += act_f1
            step_success += int(step_ok)
            task_ok = task_ok and step_ok

        task_success += int(task_ok)
        exact_sequence += int(exact_ok)

    task_count = len(trace)
    return {
        "tasks": task_count,
        "gold_steps": total_steps,
        "predicted_steps": predicted_steps,
        "length_match_rate": length_matches / task_count if task_count else 0.0,
        "element_accuracy": element_correct / total_steps if total_steps else 0.0,
        "operation_accuracy": op_correct / total_steps if total_steps else 0.0,
        "action_f1": action_f1_sum / total_steps if total_steps else 0.0,
        "step_success_rate": step_success / total_steps if total_steps else 0.0,
        "task_success_rate": task_success / task_count if task_count else 0.0,
        "exact_sequence_rate": exact_sequence / task_count if task_count else 0.0,
    }


def pct_metrics(metrics: dict) -> dict:
    converted = dict(metrics)
    for key in [
        "length_match_rate",
        "element_accuracy",
        "operation_accuracy",
        "action_f1",
        "step_success_rate",
        "task_success_rate",
        "exact_sequence_rate",
    ]:
        converted[key + "_pct"] = round(100 * converted.pop(key), 2)
    return converted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=Path("mind2web/real_data_full_output"))
    args = parser.parse_args()

    trace_path = args.run_dir / "llm_trace.json"
    workflow_path = args.run_dir / "workflow_memory.txt"
    output_path = args.run_dir / "paper_style_metrics.json"
    report_path = args.run_dir / "paper_style_metrics.md"

    trace = json.loads(trace_path.read_text())
    workflows = parse_workflow_actions(workflow_path)

    oracle_predictions = [item.get("ground_truth_actions", []) for item in trace]
    accepted_predictions = [
        workflows.get(item.get("reused_workflow"), [])
        for item in trace
    ]
    top_predictions = [
        workflows.get(item.get("top_retrieved_workflow"), [])
        for item in trace
    ]

    payload = {
        "important_note": (
            "Only oracle is a sanity check. accepted_workflow_proxy and "
            "top_workflow_proxy are not official Mind2Web benchmark scores, "
            "because retrieved workflows are not adapted by an LLM to current "
            "element ids and values."
        ),
        "oracle_ground_truth_replay": pct_metrics(evaluate_predictions(trace, oracle_predictions)),
        "accepted_workflow_proxy": pct_metrics(evaluate_predictions(trace, accepted_predictions)),
        "top_workflow_proxy": pct_metrics(evaluate_predictions(trace, top_predictions)),
    }
    output_path.write_text(json.dumps(payload, indent=2))

    lines = [
        "# Paper-Style Metric Diagnostics",
        "",
        "These metrics are computed from the local Mind2Web full run.",
        "",
        "**Important:** the official paper metrics require model-predicted actions on held-out tasks. This run replays solved exemplars, so only the oracle row is guaranteed to be 100%. The workflow-proxy rows measure how well retrieved workflow action sequences match the new task before any LLM adaptation.",
        "",
        "| Mode | Element Acc | Operation Acc | Action F1 | Step SR | Task SR | Exact Sequence |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    labels = [
        ("Oracle ground-truth replay", payload["oracle_ground_truth_replay"]),
        ("Accepted workflow proxy", payload["accepted_workflow_proxy"]),
        ("Top workflow proxy", payload["top_workflow_proxy"]),
    ]
    for label, metrics in labels:
        lines.append(
            "| {label} | {ea:.2f}% | {op:.2f}% | {af1:.2f}% | {step:.2f}% | {task:.2f}% | {exact:.2f}% |".format(
                label=label,
                ea=metrics["element_accuracy_pct"],
                op=metrics["operation_accuracy_pct"],
                af1=metrics["action_f1_pct"],
                step=metrics["step_success_rate_pct"],
                task=metrics["task_success_rate_pct"],
                exact=metrics["exact_sequence_rate_pct"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Oracle ground-truth replay is a sanity check: because predictions equal the exemplar actions, it should be 100%.",
            "- Accepted workflow proxy uses only workflows accepted by the same-website policy. Missing or unadapted steps count as wrong.",
            "- Top workflow proxy uses the top retrieved workflow even when the reuse policy rejected it.",
            "- Low proxy element accuracy is expected: workflow element ids usually do not transfer directly across different pages/tasks. In real AWM, the LLM is supposed to adapt the workflow to the current observation.",
            "",
            f"Raw JSON: `{output_path}`",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
