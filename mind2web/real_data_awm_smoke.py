"""Mini AWM smoke test on real Mind2Web exemplar data.

This script is intentionally small and offline. It uses the Mind2Web exemplar
trajectories already stored in this repository, turns a handful of successful
trajectories into workflow memory, and tests semantic workflow retrieval on
held-out tasks from the same website/domain slice.

It does not open live websites. The goal is to start using the same benchmark
family as the AWM paper without downloading the full web environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import random
import re
import sys
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
WEB_ARENA_DIR = ROOT / "webarena"
sys.path.insert(0, str(WEB_ARENA_DIR))

from local_awm_full_demo import WorkflowEmbeddingIndex  # noqa: E402


@dataclass
class Mind2WebTrajectory:
    specifier: str
    website: str
    domain: str
    subdomain: str
    task: str
    messages: list[dict]
    actions: list[str]
    observations: list[str]


def parse_specifier(specifier: str) -> dict[str, str]:
    parsed = {}
    for key in ["Website", "Domain", "Subdomain", "Task"]:
        match = re.search(rf"{key}: ([^\n]+)", specifier)
        parsed[key.lower()] = match.group(1).strip() if match else ""
    return parsed


def extract_action(content: str) -> str | None:
    match = re.search(r"Action:\s*`([^`]+)`", content)
    return match.group(1).strip() if match else None


def extract_observation(content: str) -> str | None:
    match = re.search(r"Observation:\s*`(.+?)`", content, flags=re.DOTALL)
    if not match:
        return None
    return " ".join(match.group(1).split())


def load_trajectories(path: Path) -> list[Mind2WebTrajectory]:
    conversations = json.loads(path.read_text())
    trajectories = []
    for conversation in conversations:
        if not conversation:
            continue
        specifier = conversation[0].get("specifier", "")
        meta = parse_specifier(specifier)
        actions = [
            action
            for message in conversation
            if message.get("role") == "assistant"
            for action in [extract_action(message.get("content", ""))]
            if action
        ]
        observations = [
            observation
            for message in conversation
            if message.get("role") == "user"
            for observation in [extract_observation(message.get("content", ""))]
            if observation
        ]
        if actions and meta["website"] and meta["task"]:
            trajectories.append(
                Mind2WebTrajectory(
                    specifier=specifier,
                    website=meta["website"],
                    domain=meta["domain"],
                    subdomain=meta["subdomain"],
                    task=meta["task"],
                    messages=conversation,
                    actions=actions,
                    observations=observations,
                )
            )
    return trajectories


def action_pattern(actions: list[str]) -> str:
    verbs = []
    for action in actions:
        match = re.match(r"([A-Z_]+)", action)
        verbs.append(match.group(1) if match else "ACTION")
    return " -> ".join(verbs)


def make_workflow(traj: Mind2WebTrajectory, index: int) -> str:
    action_lines = "\n".join(f"<action>\n{action}\n</action>" for action in traj.actions)
    observation_hint = traj.observations[0] if traj.observations else ""
    if len(observation_hint) > 900:
        observation_hint = observation_hint[:900].rstrip() + " ..."
    return f"""\
## {traj.website} / {traj.subdomain} workflow {index}
Source: Mind2Web exemplar
Website: {traj.website}
Domain: {traj.domain}
Subdomain: {traj.subdomain}
Goal pattern: {traj.task}
Abstract action pattern: {action_pattern(traj.actions)}
First observation hint: {observation_hint}

{action_lines}
"""


def choose_slice(
    trajectories: list[Mind2WebTrajectory],
    website: str | None,
    domain: str | None,
    subdomain: str | None,
    source_count: int,
    query_count: int,
) -> tuple[list[Mind2WebTrajectory], str]:
    no_filter_requested = website is None and domain is None and subdomain is None
    filtered = [
        traj
        for traj in trajectories
        if (website is None or traj.website == website)
        and (domain is None or traj.domain == domain)
        and (subdomain is None or traj.subdomain == subdomain)
    ]
    if not no_filter_requested and len(filtered) >= source_count + query_count:
        label = website or subdomain or domain or "requested slice"
        return filtered, label

    website_counts = Counter(traj.website for traj in trajectories)
    for candidate, _ in website_counts.most_common():
        by_website = [traj for traj in trajectories if traj.website == candidate]
        if len(by_website) >= source_count + query_count:
            return by_website, candidate

    raise RuntimeError("Could not find enough trajectories for a mini split.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "mind2web" / "data" / "memory" / "exemplars.json",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT / "mind2web" / "real_data_output")
    parser.add_argument("--website", type=str, default=None)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--subdomain", type=str, default=None)
    parser.add_argument("--source-count", type=int, default=8)
    parser.add_argument("--query-count", type=int, default=6)
    parser.add_argument("--distractor-count", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(args.data_path)
    selected, selected_label = choose_slice(
        trajectories=trajectories,
        website=args.website,
        domain=args.domain,
        subdomain=args.subdomain,
        source_count=args.source_count,
        query_count=args.query_count,
    )
    selected = list(selected)
    random.shuffle(selected)
    sources = selected[: args.source_count]
    queries = selected[args.source_count : args.source_count + args.query_count]

    selected_specifiers = {traj.specifier for traj in sources + queries}
    distractors = [
        traj
        for traj in trajectories
        if traj.specifier not in selected_specifiers and traj.website != sources[0].website
    ]
    random.shuffle(distractors)
    sources_with_distractors = sources + distractors[: args.distractor_count]

    workflow_blocks = [
        make_workflow(traj, index)
        for index, traj in enumerate(sources_with_distractors, start=1)
    ]
    workflow_memory_path = args.output_dir / "workflow_memory.txt"
    workflow_memory_path.write_text("\n\n".join(block.strip() for block in workflow_blocks) + "\n")

    embedding_index = WorkflowEmbeddingIndex(args.output_dir / "workflow_embeddings.json")
    embedding_index.rebuild(workflow_blocks)

    trace = []
    correct_website = 0
    correct_domain = 0
    for query in queries:
        query_text = "\n".join(
            [
                f"Website: {query.website}",
                f"Domain: {query.domain}",
                f"Subdomain: {query.subdomain}",
                f"Task: {query.task}",
                f"Observation: {query.observations[0] if query.observations else ''}",
            ]
        )
        result = embedding_index.retrieve(query_text, top_k=5)
        top = result.candidates[0] if result.candidates else {}
        top_name = top.get("workflow_name", "")
        top_text = top.get("text_for_embedding", "")
        website_match = f"Website: {query.website}" in top_text or top_name.startswith(f"{query.website} /")
        domain_match = f"Domain: {query.domain}" in top_text
        correct_website += int(website_match)
        correct_domain += int(domain_match)
        trace.append(
            {
                "query_task": query.task,
                "query_website": query.website,
                "query_domain": query.domain,
                "query_subdomain": query.subdomain,
                "retrieved_workflow": result.workflow_name,
                "website_match": website_match,
                "domain_match": domain_match,
                "candidates": result.candidates,
                "ground_truth_actions": query.actions,
            }
        )

    trace_path = args.output_dir / "retrieval_trace.json"
    trace_path.write_text(json.dumps(trace, indent=2))

    summary = {
        "data_path": str(args.data_path),
        "total_real_trajectories_loaded": len(trajectories),
        "selected_slice": selected_label,
        "workflow_sources_from_selected_slice": len(sources),
        "distractor_workflows": min(args.distractor_count, len(distractors)),
        "queries": len(queries),
        "website_match_rate": correct_website / len(queries) if queries else 0.0,
        "domain_match_rate": correct_domain / len(queries) if queries else 0.0,
        "workflow_memory_path": str(workflow_memory_path),
        "workflow_embedding_index_path": str(embedding_index.path),
        "retrieval_trace_path": str(trace_path),
        "retrieval_backend": embedding_index.backend,
    }

    report_lines = [
        "# Real Mind2Web AWM Mini Smoke Test",
        "",
        "This run uses real Mind2Web exemplar trajectories from the AWM benchmark family.",
        "It builds long-term workflow memory from a small source split and retrieves workflows for held-out tasks.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Example Retrievals",
        "",
    ]
    for item in trace[:5]:
        report_lines.extend(
            [
                f"### {item['query_website']} / {item['query_subdomain']}",
                "",
                f"Task: {item['query_task']}",
                f"Retrieved workflow: {item['retrieved_workflow']}",
                f"Website match: {item['website_match']}",
                "",
                "Top candidates:",
                "",
                "```json",
                json.dumps(item["candidates"][:3], indent=2),
                "```",
                "",
            ]
        )
    report_path = args.output_dir / "report.md"
    report_path.write_text("\n".join(report_lines).rstrip() + "\n")

    summary["report_path"] = str(report_path)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
