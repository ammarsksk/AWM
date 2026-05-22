"""Workflow abstraction helpers for Mind2Web memory.

The evaluator can still use deterministic gold-action workflows, but this
module adds an LLM-induced abstraction path that removes old element ids from
the reusable workflow shown to the prediction model.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any


JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def deterministic_abstract_workflow(raw_workflow: dict[str, Any]) -> dict[str, Any]:
    """Create a shallow abstract workflow without calling an LLM.

    This is used as a fallback if the abstraction LLM returns malformed output.
    It deliberately keeps raw example actions only under evidence, not in the
    prompt-facing abstract step fields.
    """
    abstract_steps = []
    evidence = []
    for index, step in enumerate(raw_workflow.get("steps", []), start=1):
        op = step.get("op", "")
        role = step.get("role") or step.get("description", "").split("]", 1)[0].strip("[() ")
        label = step.get("label") or step.get("description") or step.get("value") or "target element"
        value = step.get("value", "")
        abstract_steps.append(
            {
                "step": index,
                "step_intent": f"{op.lower()} the {label}".strip(),
                "operation": op,
                "target_type": role or "visible element",
                "target_description": label,
                "value_policy": (
                    "Use the task-specific value for this field."
                    if op in {"TYPE", "SELECT"} and value
                    else "No value required."
                ),
                "selection_rule": (
                    "Choose the current visible candidate whose label, role, "
                    "or nearby text best matches the target description."
                ),
                "success_signal": "The page advances or the selected field reflects the intended value.",
            }
        )
        evidence.append(
            {
                "step": index,
                "gold_action": step.get("action", ""),
                "gold_element_id": step.get("element", ""),
                "gold_value": value,
            }
        )

    return {
        "name": raw_workflow["name"],
        "source": raw_workflow.get("source", "Mind2Web exemplar"),
        "abstraction_type": "deterministic_fallback",
        "website": raw_workflow.get("website", ""),
        "domain": raw_workflow.get("domain", ""),
        "subdomain": raw_workflow.get("subdomain", ""),
        "task": raw_workflow.get("task", ""),
        "goal_pattern": raw_workflow.get("task", ""),
        "applicability": {
            "when_to_use": "Use for tasks with a similar goal on the same site area.",
            "when_not_to_use": "Do not use if the current task asks for a different flow or page area.",
        },
        "steps": abstract_steps,
        "evidence": evidence,
        "raw_workflow_name": raw_workflow["name"],
    }


def workflow_abstraction_prompt(raw_workflow: dict[str, Any], max_steps: int = 18) -> list[dict[str, str]]:
    """Build a strict JSON abstraction prompt."""
    evidence_steps = []
    for index, step in enumerate(raw_workflow.get("steps", [])[:max_steps], start=1):
        observation = " ".join((step.get("observation") or "").split())
        evidence_steps.append(
            {
                "step": index,
                "gold_operation": step.get("op", ""),
                "gold_value": step.get("value", ""),
                "gold_target_role": step.get("role", ""),
                "gold_target_label": step.get("label", ""),
                "gold_action_for_evidence_only": step.get("action", ""),
                "observation_excerpt": observation[:900],
            }
        )

    payload = {
        "workflow_name": raw_workflow["name"],
        "website": raw_workflow.get("website", ""),
        "domain": raw_workflow.get("domain", ""),
        "subdomain": raw_workflow.get("subdomain", ""),
        "task_goal": raw_workflow.get("task", ""),
        "evidence_steps": evidence_steps,
        "required_output_schema": {
            "name": raw_workflow["name"],
            "goal_pattern": "general reusable goal, with task-specific values abstracted",
            "applicability": {
                "when_to_use": "when this workflow should be retrieved",
                "when_not_to_use": "when this workflow is misleading",
            },
            "steps": [
                {
                    "step": 1,
                    "step_intent": "what this step accomplishes",
                    "operation": "CLICK | TYPE | SELECT",
                    "target_type": "button/input/select/link/etc",
                    "target_description": "semantic description, no element ids",
                    "value_policy": "how to derive typed/selected value from the current task",
                    "selection_rule": "how to choose among current candidate elements",
                    "success_signal": "what indicates this step worked",
                }
            ],
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "You induce abstract reusable web workflows. Return strict JSON only. "
                "Do not include old element ids in abstract step fields. Old gold actions "
                "are evidence only; generalize them into semantic target descriptions and "
                "value policies. Keep every field concise. Include exactly one step object "
                "for each evidence step provided."
            ),
        },
        {"role": "user", "content": json.dumps(payload, indent=2)},
    ]


def parse_json_object(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    match = JSON_OBJECT_RE.search(text)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def normalize_llm_abstraction(raw_workflow: dict[str, Any], data: dict[str, Any]) -> dict[str, Any]:
    steps = data.get("steps") if isinstance(data.get("steps"), list) else []
    cleaned_steps = []
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            continue
        cleaned_steps.append(
            {
                "step": int(step.get("step") or index),
                "step_intent": str(step.get("step_intent", "")).strip(),
                "operation": str(step.get("operation", step.get("op", ""))).strip().upper(),
                "target_type": str(step.get("target_type", "")).strip(),
                "target_description": strip_element_ids(str(step.get("target_description", "")).strip()),
                "value_policy": strip_element_ids(str(step.get("value_policy", "")).strip()),
                "selection_rule": strip_element_ids(str(step.get("selection_rule", "")).strip()),
                "success_signal": strip_element_ids(str(step.get("success_signal", "")).strip()),
            }
        )

    if not cleaned_steps:
        return deterministic_abstract_workflow(raw_workflow)

    return {
        "name": raw_workflow["name"],
        "source": raw_workflow.get("source", "Mind2Web exemplar"),
        "abstraction_type": "llm",
        "website": raw_workflow.get("website", ""),
        "domain": raw_workflow.get("domain", ""),
        "subdomain": raw_workflow.get("subdomain", ""),
        "task": raw_workflow.get("task", ""),
        "goal_pattern": strip_element_ids(str(data.get("goal_pattern", raw_workflow.get("task", ""))).strip()),
        "applicability": data.get("applicability", {})
        if isinstance(data.get("applicability"), dict)
        else {},
        "steps": cleaned_steps,
        "evidence": [
            {
                "step": index,
                "gold_action": step.get("action", ""),
                "gold_element_id": step.get("element", ""),
                "gold_value": step.get("value", ""),
            }
            for index, step in enumerate(raw_workflow.get("steps", []), start=1)
        ],
        "raw_workflow_name": raw_workflow["name"],
        "llm_raw_abstraction": data,
    }


def strip_element_ids(text: str) -> str:
    text = re.sub(r"\[[0-9]+\]", "[current_element]", text)
    text = re.sub(r"\bid\s*[=:]\s*[0-9]+\b", "current element", text, flags=re.IGNORECASE)
    return text


def abstract_workflow_with_llm(
    client,
    model: str,
    raw_workflow: dict[str, Any],
    max_output_tokens: int = 2048,
    max_steps: int = 18,
    retries: int = 2,
    retry_sleep: float = 5.0,
) -> tuple[dict[str, Any], str | None]:
    """Call an LLM to create an abstract workflow; fallback deterministically."""
    messages = workflow_abstraction_prompt(raw_workflow, max_steps=max_steps)
    last_raw = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                max_tokens=max_output_tokens,
            )
            last_raw = response.choices[0].message.content or ""
            parsed = parse_json_object(last_raw)
            if parsed is not None:
                return normalize_llm_abstraction(raw_workflow, parsed), last_raw
        except Exception:
            if attempt >= retries:
                break
        time.sleep(retry_sleep * (attempt + 1))

    fallback = deterministic_abstract_workflow(raw_workflow)
    fallback["abstraction_type"] = "llm_failed_fallback"
    return fallback, last_raw


def abstract_workflow_to_text(workflow: dict[str, Any]) -> str:
    """Render abstract workflow text for embedding and human memory files."""
    lines = [
        f"## {workflow['name']}",
        f"Source: {workflow.get('source', '')}",
        f"Abstraction: {workflow.get('abstraction_type', '')}",
        f"Website: {workflow.get('website', '')}",
        f"Domain: {workflow.get('domain', '')}",
        f"Subdomain: {workflow.get('subdomain', '')}",
        f"Goal pattern: {workflow.get('goal_pattern') or workflow.get('task', '')}",
    ]
    applicability = workflow.get("applicability", {})
    if isinstance(applicability, dict):
        lines.append(f"When to use: {applicability.get('when_to_use', '')}")
        lines.append(f"When not to use: {applicability.get('when_not_to_use', '')}")
    lines.append("Abstract steps:")
    for step in workflow.get("steps", []):
        lines.append(
            "Abstract action pattern: {operation} intent={intent} target_type={target_type} "
            "target={target} value_policy={value_policy} selection_rule={selection_rule} "
            "success={success}".format(
                operation=step.get("operation", step.get("op", "")),
                intent=step.get("step_intent", ""),
                target_type=step.get("target_type", ""),
                target=step.get("target_description", ""),
                value_policy=step.get("value_policy", ""),
                selection_rule=step.get("selection_rule", ""),
                success=step.get("success_signal", ""),
            )
        )
    return "\n".join(lines)
