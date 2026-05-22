"""Mind2Web step-prediction runner with workflow retrieval and adaptation.

This script is the bridge between the offline memory replay and paper-style
Mind2Web evaluation. It can run in three modes:

- oracle: uses ground-truth actions to verify metric code.
- heuristic: adapts retrieved workflow steps to current observations with local
  element matching. No API calls.
- llm: sends task, observation, candidate elements, and retrieved workflows to
  an OpenAI-compatible LLM such as Vertex/Gemini.

The runner writes predictions, retrieval traces, structured workflows, and
paper-style metrics: Element Accuracy, Operation Accuracy, Action F1, Step SR,
and Task SR.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


ROOT = Path(__file__).resolve().parents[1]
WEB_ARENA_DIR = ROOT / "webarena"
sys.path.insert(0, str(WEB_ARENA_DIR))

from local_awm_full_demo import WorkflowEmbeddingIndex  # noqa: E402
from real_data_awm_smoke import load_trajectories  # noqa: E402
from utils.provider_config import get_openai_compatible_kwargs  # noqa: E402
from workflow_abstraction import (  # noqa: E402
    abstract_workflow_to_text,
    abstract_workflow_with_llm,
    deterministic_abstract_workflow,
)
from workflow_store import DiskWorkflowStore  # noqa: E402


ACTION_RE = re.compile(
    r"Action:\s*`(?P<action>[^`]+)`(?:\s*\((?P<desc>.*?)\))?",
    re.DOTALL,
)
ACTION_PART_RE = re.compile(
    r"^(?P<op>[A-Z_]+)(?:\s+\[(?P<element>[^\]]+)\])?(?:\s+\[(?P<value>[^\]]*)\])?"
)
ACTION_PART_LOOSE_RE = re.compile(
    r"^(?P<op>CLICK|TYPE|SELECT|HOVER|PRESS|SCROLL)\s+(?P<element>\d+)(?:\s+(?P<value>.*))?$",
    re.IGNORECASE,
)
TAG_RE = re.compile(r"<(?P<tag>[a-zA-Z][\w:-]*)(?P<attrs>[^<>]*?\bid=(?P<id>\d+)[^<>]*?)(?:>|/>)")
JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class ElementCandidate:
    element_id: str
    tag: str
    role_hint: str
    label: str
    context: str
    raw: str


def parse_action_text(action: str | None) -> dict[str, str]:
    if not action:
        return {"op": "", "element": "", "value": ""}
    action = action.strip()
    match = ACTION_PART_RE.match(action)
    if match and match.group("element"):
        return {
            "op": (match.group("op") or "").strip().upper(),
            "element": (match.group("element") or "").strip(),
            "value": (match.group("value") or "").strip(),
        }
    match = ACTION_PART_LOOSE_RE.match(action)
    if not match:
        return {"op": "", "element": "", "value": ""}
    return {
        "op": (match.group("op") or "").strip().upper(),
        "element": (match.group("element") or "").strip(),
        "value": (match.group("value") or "").strip(),
    }


def normalize_action_text(action: str | None) -> str:
    parsed = parse_action_text(action)
    if not parsed["op"] or not parsed["element"]:
        return (action or "").strip()
    return render_action(parsed["op"], parsed["element"], parsed["value"])


def parse_assistant_action(content: str) -> dict[str, str] | None:
    match = ACTION_RE.search(content)
    if not match:
        return None
    action = " ".join(match.group("action").split())
    parsed = parse_action_text(action)
    desc = " ".join((match.group("desc") or "").split())
    role = ""
    label = ""
    desc_match = re.search(r"\(\[(?P<role>[^\]]+)\]\s*(?P<label>.*)", f"({desc}")
    if desc_match:
        role = desc_match.group("role").strip()
        label = desc_match.group("label").split("->")[0].strip()
    return {
        "action": action,
        "op": parsed["op"],
        "element": parsed["element"],
        "value": parsed["value"],
        "description": desc,
        "role": role,
        "label": label,
    }


def parse_elements(observation: str) -> list[ElementCandidate]:
    candidates = []
    for match in TAG_RE.finditer(observation):
        element_id = match.group("id")
        tag = match.group("tag")
        attrs = " ".join(match.group("attrs").split())
        role_words = [
            token
            for token in re.findall(r"[a-zA-Z_:-]+", attrs)
            if token.lower() not in {"id", "true", "false"}
        ]
        label = " ".join(token for token in role_words if not token.isdigit())
        context_window = observation[
            max(0, match.start() - 180) : min(len(observation), match.end() + 320)
        ]
        context = html_to_text(context_window)
        candidates.append(
            ElementCandidate(
                element_id=element_id,
                tag=tag,
                role_hint=" ".join(role_words[:4]),
                label=label,
                context=context,
                raw=f"<{tag} {attrs}>",
            )
        )
    return candidates


def html_to_text(text: str) -> str:
    text = re.sub(r"[<>/=\"']", " ", text)
    text = re.sub(r"\bid\s+\d+\b", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()[:300]


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def compatible_element(op: str, candidate: ElementCandidate) -> bool:
    text = f"{candidate.tag} {candidate.role_hint} {candidate.label}".lower()
    if op == "CLICK":
        return any(word in text for word in ["button", "link", "a", "svg", "li", "span", "tab"])
    if op == "TYPE":
        return any(word in text for word in ["input", "textbox", "search", "email", "text", "combobox"])
    if op == "SELECT":
        return any(word in text for word in ["select", "option", "combobox", "listbox"])
    return True


def choose_element(step: dict, observation: str) -> ElementCandidate | None:
    candidates = parse_elements(observation)
    if not candidates:
        return None

    best = None
    best_score = -1.0
    target_text = " ".join(
        [
            step.get("step_intent", ""),
            step.get("target_type", ""),
            step.get("target_description", ""),
            step.get("selection_rule", ""),
            step.get("role", ""),
            step.get("label", ""),
            step.get("description", ""),
            step.get("value", ""),
        ]
    )
    for candidate in candidates:
        candidate_text = (
            f"{candidate.tag} {candidate.role_hint} {candidate.label} "
            f"{candidate.context} {candidate.raw}"
        )
        score = lexical_overlap(target_text, candidate_text)
        operation = step.get("op") or step.get("operation", "")
        if compatible_element(operation, candidate):
            score += 0.35
        if step.get("role") and step["role"].lower() in candidate_text.lower():
            score += 0.25
        if step.get("value") and step["value"].lower() in candidate_text.lower():
            score += 0.15
        if score > best_score:
            best = candidate
            best_score = score
    return best


def render_action(op: str, element_id: str, value: str = "") -> str:
    if op in {"TYPE", "SELECT"}:
        return f"{op} [{element_id}] [{value}]"
    return f"{op} [{element_id}]"


def extract_observation_from_message(content: str) -> str:
    match = re.search(r"Observation:\s*`(.+?)`", content, flags=re.DOTALL)
    return " ".join(match.group(1).split()) if match else ""


def structured_steps_from_trajectory(traj) -> list[dict]:
    steps = []
    observations = []
    for message in traj.messages:
        if message.get("role") == "user":
            observations.append(extract_observation_from_message(message.get("content", "")))
        elif message.get("role") == "assistant":
            action_info = parse_assistant_action(message.get("content", ""))
            if action_info:
                index = len(steps)
                action_info["observation"] = observations[index] if index < len(observations) else ""
                steps.append(action_info)
    return steps


def make_structured_workflow(traj, index: int) -> dict:
    return {
        "name": f"{traj.website} / {traj.subdomain} workflow {index}",
        "source": "Mind2Web exemplar",
        "website": traj.website,
        "domain": traj.domain,
        "subdomain": traj.subdomain,
        "task": traj.task,
        "steps": structured_steps_from_trajectory(traj),
    }


def workflow_to_text(workflow: dict) -> str:
    if workflow.get("abstraction_type"):
        return abstract_workflow_to_text(workflow)
    lines = [
        f"## {workflow['name']}",
        f"Source: {workflow['source']}",
        f"Website: {workflow['website']}",
        f"Domain: {workflow['domain']}",
        f"Subdomain: {workflow['subdomain']}",
        f"Goal pattern: {workflow['task']}",
        "Structured steps:",
    ]
    for step in workflow["steps"]:
        lines.append(
            " - {op} target_role={role} target_label={label} value={value} example={action}".format(
                op=step.get("op", ""),
                role=step.get("role", ""),
                label=step.get("label", ""),
                value=step.get("value", ""),
                action=step.get("action", ""),
            )
        )
        lines.append("<action>")
        lines.append(step.get("action", ""))
        lines.append("</action>")
    return "\n".join(lines)


def make_memory_workflow(
    raw_workflow: dict,
    args,
    abstraction_client=None,
) -> tuple[dict, str | None]:
    """Return the workflow representation used for retrieval/prompting."""
    if args.workflow_abstraction == "raw":
        return raw_workflow, None
    if args.workflow_abstraction == "deterministic":
        return deterministic_abstract_workflow(raw_workflow), None
    workflow, raw_llm = abstract_workflow_with_llm(
        client=abstraction_client,
        model=args.abstraction_model or args.model,
        raw_workflow=raw_workflow,
        max_output_tokens=args.abstraction_max_output_tokens,
        max_steps=args.abstraction_max_steps,
        retries=args.llm_retries,
        retry_sleep=args.retry_sleep,
    )
    return workflow, raw_llm


def make_workflow_container(output_dir: Path, storage: str):
    if storage == "disk":
        return DiskWorkflowStore(output_dir / "workflow_json")
    return {}


def add_workflow_to_container(workflows, workflow: dict):
    if isinstance(workflows, DiskWorkflowStore):
        return workflows.add(workflow)
    workflows[workflow["name"]] = workflow
    return None


def workflows_to_dict(workflows) -> dict[str, dict]:
    if isinstance(workflows, DiskWorkflowStore):
        return workflows.to_dict()
    return dict(workflows)


def query_text(traj, observation: str) -> str:
    return "\n".join(
        [
            f"Website: {traj.website}",
            f"Domain: {traj.domain}",
            f"Subdomain: {traj.subdomain}",
            f"Task: {traj.task}",
            f"Observation: {observation}",
        ]
    )


def workflow_metadata(workflow_name: str | None) -> dict[str, str]:
    if not workflow_name or " / " not in workflow_name:
        return {}
    website, rest = workflow_name.split(" / ", 1)
    return {"website": website, "subdomain": rest.split(" workflow ")[0]}


def accept_reuse(policy: str, traj, workflow_name: str | None) -> bool:
    if not workflow_name:
        return False
    if policy == "threshold":
        return True
    meta = workflow_metadata(workflow_name)
    if policy == "same-website":
        return meta.get("website") == traj.website
    if policy == "same-subdomain":
        return meta.get("website") == traj.website and meta.get("subdomain") == traj.subdomain
    return False


def get_workflow_step(workflows, name: str | None, step_index: int) -> dict | None:
    if not name or name not in workflows:
        return None
    steps = workflows[name].get("steps", [])
    return steps[step_index] if step_index < len(steps) else None


def predict_heuristic(
    workflows,
    workflow_name: str | None,
    step_index: int,
    observation: str,
) -> str:
    step = get_workflow_step(workflows, workflow_name, step_index)
    if step is None:
        return ""
    candidate = choose_element(step, observation)
    if candidate is None:
        return ""
    operation = step.get("op") or step.get("operation", "")
    return render_action(operation, candidate.element_id, step.get("value", ""))


def compact_candidates(observation: str, limit: int = 60) -> list[dict]:
    return [
        {
            "id": item.element_id,
            "tag": item.tag,
            "role_hint": item.role_hint,
            "label": item.label,
            "nearby_text": item.context,
        }
        for item in parse_elements(observation)[:limit]
    ]


def workflow_for_prompt(workflows, name: str | None) -> dict | None:
    if not name or name not in workflows:
        return None
    workflow = dict(workflows[name])
    compact_steps = []
    for step in workflow.get("steps", [])[:12]:
        if workflow.get("abstraction_type"):
            compact_steps.append(
                {
                    "step_intent": step.get("step_intent", ""),
                    "operation": step.get("operation", step.get("op", "")),
                    "target_type": step.get("target_type", ""),
                    "target_description": step.get("target_description", ""),
                    "value_policy": step.get("value_policy", ""),
                    "selection_rule": step.get("selection_rule", ""),
                    "success_signal": step.get("success_signal", ""),
                }
            )
        else:
            compact_steps.append(
                {
                    "op": step.get("op", ""),
                    "target_role": step.get("role", ""),
                    "target_label": step.get("label", ""),
                    "example_value": step.get("value", ""),
                    "example_action": step.get("action", ""),
                }
            )
    workflow["steps"] = compact_steps
    workflow.pop("evidence", None)
    workflow.pop("llm_raw_abstraction", None)
    return workflow


def parse_llm_action(raw: str) -> tuple[str, dict | None]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = JSON_OBJECT_RE.search(text)
    if match:
        try:
            data = json.loads(match.group(0))
            action = normalize_action_text(str(data.get("action", "")).strip())
            return action, data
        except json.JSONDecodeError:
            pass
    action_match = re.search(r"(CLICK|TYPE|SELECT|HOVER|PRESS|SCROLL)\s+\[[^\]]+\](?:\s+\[[^\]]*\])?", text)
    if action_match:
        return normalize_action_text(action_match.group(0)), None
    loose_match = re.search(r"(CLICK|TYPE|SELECT|HOVER|PRESS|SCROLL)\s+\d+(?:\s+[^\n\r}`]+)?", text)
    return (normalize_action_text(loose_match.group(0)) if loose_match else ""), None


def predict_llm(
    client,
    model: str,
    traj,
    observation: str,
    previous_actions: list[str],
    retrieval_candidates: list[dict],
    workflows,
    accepted_workflow: str | None,
    max_output_tokens: int = 1024,
    retries: int = 2,
    retry_sleep: float = 5.0,
) -> tuple[str, str, dict | None]:
    compact_retrieval = [
        {
            key: candidate.get(key)
            for key in ["workflow_name", "combined_score", "semantic_score", "bm25_score"]
        }
        for candidate in retrieval_candidates[:5]
    ]
    workflow = workflow_for_prompt(workflows, accepted_workflow)
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Mind2Web action prediction agent. Return strict JSON only. "
                "Choose exactly one next action using only ids visible in the current observation. "
                "Do not copy ids from workflows unless that id appears in the current candidates. "
                "Put the action first in the JSON."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "task": traj.task,
                    "website": traj.website,
                    "domain": traj.domain,
                    "subdomain": traj.subdomain,
                    "current_observation": observation,
                    "candidate_elements": compact_candidates(observation),
                    "previous_actions": previous_actions,
                    "retrieval_candidates": compact_retrieval,
                    "accepted_workflow": workflow,
                    "valid_action_formats": [
                        "CLICK [element_id]",
                        "TYPE [element_id] [text]",
                        "SELECT [element_id] [option_text]",
                    ],
                    "return_json_schema": {
                        "action": "one action string in bracket format, e.g. CLICK [123] or TYPE [123] [text]",
                        "reason": "brief reason",
                    },
                },
                indent=2,
            ),
        },
    ]
    last_exc = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_output_tokens,
            )
            break
        except Exception as exc:
            last_exc = exc
            if attempt >= retries:
                raise
            time.sleep(retry_sleep * (attempt + 1))
    else:
        raise last_exc  # pragma: no cover
    raw = response.choices[0].message.content or ""
    action, parsed = parse_llm_action(raw)
    return action, raw, parsed


def make_llm_client():
    from openai import OpenAI

    return OpenAI(**get_openai_compatible_kwargs())


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


def candidate_by_id(step: dict, element_id: str) -> dict:
    for candidate in step.get("candidate_elements", []):
        if str(candidate.get("id")) == str(element_id):
            return candidate
    return {}


def relaxed_element_match(step: dict, pred: dict[str, str], gold: dict[str, str]) -> bool:
    if not pred["element"] or not gold["element"]:
        return False
    if pred["element"] == gold["element"]:
        return True

    pred_candidate = candidate_by_id(step, pred["element"])
    gold_candidate = candidate_by_id(step, gold["element"])
    if not pred_candidate or not gold_candidate:
        return False

    pred_text = " ".join(
        str(pred_candidate.get(key, ""))
        for key in ["tag", "role_hint", "label", "nearby_text"]
    )
    gold_text = " ".join(
        str(gold_candidate.get(key, ""))
        for key in ["tag", "role_hint", "label", "nearby_text"]
    )

    if pred_candidate.get("tag") == gold_candidate.get("tag") and lexical_overlap(pred_text, gold_text) >= 0.45:
        return True
    if lexical_overlap(pred_text, gold_text) >= 0.65:
        return True
    if pred["op"] == "CLICK" and pred_candidate.get("nearby_text") == gold_candidate.get("nearby_text"):
        return True
    return False


def compute_metrics(records: list[dict]) -> dict:
    total_steps = 0
    element_correct = 0
    op_correct = 0
    action_f1_sum = 0.0
    step_success = 0
    task_success = 0
    exact_sequence = 0
    relaxed_element_correct = 0
    relaxed_step_success = 0
    relaxed_task_success = 0

    for record in records:
        task_ok = True
        relaxed_task_ok = True
        exact_ok = True
        for step in record["steps"]:
            gold = parse_action_text(step.get("gold_action"))
            pred = parse_action_text(step.get("predicted_action"))
            total_steps += 1
            element_match = bool(gold["element"]) and pred["element"] == gold["element"]
            op_match = bool(gold["op"]) and pred["op"] == gold["op"]
            act_f1 = action_value_f1(pred, gold)
            step_ok = element_match and act_f1 == 1.0
            relaxed_match = relaxed_element_match(step, pred, gold)
            relaxed_step_ok = relaxed_match and act_f1 >= 0.8
            element_correct += int(element_match)
            relaxed_element_correct += int(relaxed_match)
            op_correct += int(op_match)
            action_f1_sum += act_f1
            step_success += int(step_ok)
            relaxed_step_success += int(relaxed_step_ok)
            task_ok = task_ok and step_ok
            relaxed_task_ok = relaxed_task_ok and relaxed_step_ok
            exact_ok = exact_ok and step.get("predicted_action") == step.get("gold_action")
        task_success += int(task_ok)
        relaxed_task_success += int(relaxed_task_ok)
        exact_sequence += int(exact_ok)

    tasks = len(records)
    return {
        "tasks": tasks,
        "steps": total_steps,
        "element_accuracy": element_correct / total_steps if total_steps else 0.0,
        "relaxed_element_accuracy": relaxed_element_correct / total_steps if total_steps else 0.0,
        "operation_accuracy": op_correct / total_steps if total_steps else 0.0,
        "action_f1": action_f1_sum / total_steps if total_steps else 0.0,
        "step_success_rate": step_success / total_steps if total_steps else 0.0,
        "relaxed_step_success_rate": relaxed_step_success / total_steps if total_steps else 0.0,
        "task_success_rate": task_success / tasks if tasks else 0.0,
        "relaxed_task_success_rate": relaxed_task_success / tasks if tasks else 0.0,
        "exact_sequence_rate": exact_sequence / tasks if tasks else 0.0,
    }


def percent_payload(metrics: dict) -> dict:
    output = dict(metrics)
    for key in [
        "element_accuracy",
        "relaxed_element_accuracy",
        "operation_accuracy",
        "action_f1",
        "step_success_rate",
        "relaxed_step_success_rate",
        "task_success_rate",
        "relaxed_task_success_rate",
        "exact_sequence_rate",
    ]:
        output[key + "_pct"] = round(output.pop(key) * 100, 2)
    return output


def save_outputs(
    output_dir: Path,
    records: list[dict],
    workflows,
    workflow_texts: list[str],
    embedding_index: WorkflowEmbeddingIndex | None = None,
    raw_workflows=None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prediction_trace.json").write_text(json.dumps(records, indent=2))
    (output_dir / "structured_workflows.json").write_text(json.dumps(workflows_to_dict(workflows), indent=2))
    if raw_workflows is not None:
        (output_dir / "raw_structured_workflows.json").write_text(
            json.dumps(workflows_to_dict(raw_workflows), indent=2)
        )
    (output_dir / "workflow_memory.txt").write_text("\n\n".join(workflow_texts) + "\n")
    if embedding_index is not None:
        embedding_index.save()
    metrics = percent_payload(compute_metrics(records))
    (output_dir / "paper_metrics.json").write_text(json.dumps(metrics, indent=2))
    lines = [
        "# Mind2Web Step Prediction Metrics",
        "",
        "These are paper-style metrics computed from generated predictions.",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Tasks | {metrics['tasks']} |",
        f"| Steps | {metrics['steps']} |",
        f"| Element Accuracy | {metrics['element_accuracy_pct']:.2f}% |",
        f"| Relaxed Element Accuracy | {metrics['relaxed_element_accuracy_pct']:.2f}% |",
        f"| Operation Accuracy | {metrics['operation_accuracy_pct']:.2f}% |",
        f"| Action F1 | {metrics['action_f1_pct']:.2f}% |",
        f"| Step Success Rate | {metrics['step_success_rate_pct']:.2f}% |",
        f"| Relaxed Step Success Rate | {metrics['relaxed_step_success_rate_pct']:.2f}% |",
        f"| Task Success Rate | {metrics['task_success_rate_pct']:.2f}% |",
        f"| Relaxed Task Success Rate | {metrics['relaxed_task_success_rate_pct']:.2f}% |",
        f"| Exact Sequence Rate | {metrics['exact_sequence_rate_pct']:.2f}% |",
        "",
        "Relaxed metrics count semantically equivalent nearby elements as matches, which is useful for local no-browser analysis where parent/child ids and unlabeled SVGs are often interchangeable.",
    ]
    (output_dir / "paper_metrics.md").write_text("\n".join(lines) + "\n")


def build_retrieval_plan(selected: list, args):
    embedding_index = WorkflowEmbeddingIndex(args.output_dir / "workflow_embeddings.json")
    workflows = make_workflow_container(args.output_dir, args.workflow_storage)
    raw_workflows = make_workflow_container(args.output_dir / "raw", args.workflow_storage)
    workflow_texts: list[str] = []
    records: list[dict] = []
    abstraction_client = make_llm_client() if args.workflow_abstraction == "llm" else None

    for local_index, traj in enumerate(selected, start=1):
        global_index = args.start_index + local_index
        structured_gold_steps = structured_steps_from_trajectory(traj)
        if args.max_steps_per_task is not None:
            structured_gold_steps = structured_gold_steps[: args.max_steps_per_task]

        task_steps = []
        for step_index, gold_step in enumerate(structured_gold_steps):
            observation = gold_step.get("observation") or ""
            retrieval = embedding_index.retrieve(query_text(traj, observation), top_k=args.top_k)
            top_workflow = retrieval.workflow_name
            accepted_workflow = top_workflow if accept_reuse(args.reuse_policy, traj, top_workflow) else None
            task_steps.append(
                {
                    "step": step_index + 1,
                    "observation": observation,
                    "candidate_elements": compact_candidates(observation),
                    "top_retrieved_workflow": top_workflow,
                    "accepted_workflow": accepted_workflow,
                    "retrieval_candidates": retrieval.candidates,
                    "gold_action": gold_step["action"],
                    "predicted_action": "",
                    "raw_llm_output": None,
                    "parsed_llm_output": None,
                }
            )

        raw_workflow = make_structured_workflow(traj, global_index)
        workflow, abstraction_raw_output = make_memory_workflow(
            raw_workflow,
            args,
            abstraction_client=abstraction_client,
        )
        workflow_text = workflow_to_text(workflow)
        add_workflow_to_container(raw_workflows, raw_workflow)
        add_workflow_to_container(workflows, workflow)
        workflow_texts.append(workflow_text)
        embedding_index.add_workflow(workflow_text, save=False)

        records.append(
            {
                "task_index": global_index,
                "website": traj.website,
                "domain": traj.domain,
                "subdomain": traj.subdomain,
                "goal": traj.task,
                "mode": args.mode,
                "reuse_policy": args.reuse_policy,
                "workflow_abstraction": args.workflow_abstraction,
                "workflow_storage": args.workflow_storage,
                "steps": task_steps,
                "added_workflow": workflow["name"],
                "abstraction_type": workflow.get("abstraction_type", "raw"),
                "abstraction_raw_output": abstraction_raw_output,
                "_trajectory": traj,
            }
        )

        if local_index % 100 == 0:
            print(f"planned {local_index}/{len(selected)} tasks")

    return records, workflows, workflow_texts, embedding_index, raw_workflows


def predict_record(record: dict, workflows, args) -> dict:
    traj = record["_trajectory"]
    record = dict(record)
    record.pop("_trajectory", None)
    steps = []
    previous_actions = []
    client = make_llm_client() if args.mode == "llm" else None

    for step in record["steps"]:
        step = dict(step)
        if args.mode == "oracle":
            predicted_action = step["gold_action"]
            raw_llm = None
            parsed_llm = None
        elif args.mode == "heuristic":
            predicted_action = predict_heuristic(
                workflows,
                step.get("accepted_workflow"),
                step["step"] - 1,
                step.get("observation", ""),
            )
            raw_llm = None
            parsed_llm = None
        else:
            predicted_action, raw_llm, parsed_llm = predict_llm(
                client=client,
                model=args.model,
                traj=traj,
                observation=step.get("observation", ""),
                previous_actions=previous_actions,
                retrieval_candidates=step.get("retrieval_candidates", []),
                workflows=workflows,
                accepted_workflow=step.get("accepted_workflow"),
                max_output_tokens=args.max_output_tokens,
                retries=args.llm_retries,
                retry_sleep=args.retry_sleep,
            )
        step["predicted_action"] = predicted_action
        step["raw_llm_output"] = raw_llm
        step["parsed_llm_output"] = parsed_llm
        previous_actions.append(predicted_action)
        steps.append(step)

    record["steps"] = steps
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=ROOT / "mind2web" / "data" / "memory" / "exemplars.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "mind2web" / "llm_step_eval_output")
    parser.add_argument("--mode", choices=["oracle", "heuristic", "llm"], default="heuristic")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--max-tasks", type=int, default=25)
    parser.add_argument("--max-steps-per-task", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--reuse-policy", choices=["same-website", "same-subdomain", "threshold"], default="same-website")
    parser.add_argument(
        "--workflow-abstraction",
        choices=["raw", "deterministic", "llm"],
        default="raw",
        help=(
            "raw keeps the previous gold-action workflow format; deterministic "
            "builds local abstract workflow fields; llm calls the provider once "
            "per completed task to induce reusable abstract workflows."
        ),
    )
    parser.add_argument(
        "--workflow-storage",
        choices=["ram", "disk"],
        default="ram",
        help=(
            "ram keeps workflow JSON objects in a Python dict; disk writes each "
            "workflow JSON under output_dir/workflow_json while FAISS vectors stay in RAM."
        ),
    )
    parser.add_argument(
        "--abstraction-model",
        type=str,
        default=None,
        help="Model used for LLM workflow abstraction. Defaults to --model.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--abstraction-max-output-tokens",
        type=int,
        default=4096,
        help="LLM output token budget for workflow abstraction calls.",
    )
    parser.add_argument(
        "--abstraction-max-steps",
        type=int,
        default=18,
        help="Maximum trajectory steps included in each LLM workflow abstraction prompt.",
    )
    parser.add_argument(
        "--max-output-tokens",
        type=int,
        default=1024,
        help="LLM output token budget. Gemini/Vertex may spend part of this before visible JSON appears.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "Parallelize task-level prediction after building the online retrieval plan. "
            "Useful for LLM runs; steps inside each task remain sequential."
        ),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trajectories = load_trajectories(args.data_path)
    selected = trajectories[args.start_index :]
    if args.max_tasks is not None:
        selected = selected[: args.max_tasks]

    planned_records, workflows, workflow_texts, embedding_index, raw_workflows = build_retrieval_plan(selected, args)
    save_outputs(
        args.output_dir,
        [],
        workflows,
        workflow_texts,
        embedding_index=embedding_index,
        raw_workflows=raw_workflows,
    )

    records = []
    if args.parallel_workers <= 1:
        for index, record in enumerate(planned_records, start=1):
            records.append(predict_record(record, workflows, args))
            if index % args.save_every == 0:
                save_outputs(
                    args.output_dir,
                    records,
                    workflows,
                    workflow_texts,
                    embedding_index=embedding_index,
                    raw_workflows=raw_workflows,
                )
                print(f"processed {index}/{len(planned_records)} tasks")
    else:
        completed: dict[int, dict] = {}
        with ThreadPoolExecutor(max_workers=args.parallel_workers) as executor:
            futures = {
                executor.submit(predict_record, record, workflows, args): record["task_index"]
                for record in planned_records
            }
            for done_count, future in enumerate(as_completed(futures), start=1):
                task_index = futures[future]
                completed[task_index] = future.result()
                if done_count % args.save_every == 0:
                    records = [completed[key] for key in sorted(completed)]
                    save_outputs(
                        args.output_dir,
                        records,
                        workflows,
                        workflow_texts,
                        embedding_index=embedding_index,
                        raw_workflows=raw_workflows,
                    )
                    print(f"processed {done_count}/{len(planned_records)} tasks")
        records = [completed[key] for key in sorted(completed)]

    save_outputs(
        args.output_dir,
        records,
        workflows,
        workflow_texts,
        embedding_index=embedding_index,
        raw_workflows=raw_workflows,
    )
    print(json.dumps(percent_payload(compute_metrics(records)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
