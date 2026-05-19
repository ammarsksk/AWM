"""Online-style AWM run over real Mind2Web exemplar trajectories.

This is still offline with respect to websites: it does not open browsers or
execute actions. It uses the real Mind2Web trajectory data already present in
this repo to simulate the AWM memory loop:

1. Read the next real task trajectory.
2. Retrieve a reusable workflow from long-term workflow memory.
3. Record a trace/episode with the retrieval candidates and ground-truth actions.
4. Induce a workflow from the successful exemplar trajectory.
5. Store that workflow and update the semantic FAISS/BM25 index.

The output mirrors the local WebArena demo artifacts: trace, episodic memory,
workflow memory, workflow embeddings, and a report.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import random
import sys
from collections import Counter


ROOT = Path(__file__).resolve().parents[1]
WEB_ARENA_DIR = ROOT / "webarena"
sys.path.insert(0, str(WEB_ARENA_DIR))

from local_awm_full_demo import WorkflowEmbeddingIndex  # noqa: E402
from real_data_awm_smoke import load_trajectories, make_workflow  # noqa: E402


def query_text(traj) -> str:
    return "\n".join(
        [
            f"Website: {traj.website}",
            f"Domain: {traj.domain}",
            f"Subdomain: {traj.subdomain}",
            f"Task: {traj.task}",
            f"Observation: {traj.observations[0] if traj.observations else ''}",
        ]
    )


def build_prompt(traj, retrieval) -> str:
    return f"""\
# Mind2Web Task
Website: {traj.website}
Domain: {traj.domain}
Subdomain: {traj.subdomain}
Goal: {traj.task}

# Current Observation
{traj.observations[0] if traj.observations else "(no observation parsed)"}

# Retrieved Workflow Candidates
{json.dumps(retrieval.candidates, indent=2)}

# Instruction
Use a retrieved workflow only if it matches the task. Adapt element ids and values
to the current observation. The ground-truth action sequence is logged separately
because this full run uses real exemplar trajectories rather than live browser execution.
"""


def trace_steps(traj, reused_workflow: str | None) -> list[dict]:
    steps = []
    short_term = []
    max_len = max(len(traj.actions), len(traj.observations))
    for index in range(max_len):
        action = traj.actions[index] if index < len(traj.actions) else None
        observation = traj.observations[index] if index < len(traj.observations) else None
        before = list(short_term)
        if action is not None:
            short_term.append(action)
        steps.append(
            {
                "step": index + 1,
                "observation": observation,
                "ground_truth_action": action,
                "reused_workflow": reused_workflow,
                "short_term_before": before,
                "short_term_after": list(short_term),
            }
        )
    return steps


def workflow_metadata(workflow_name: str | None) -> dict:
    if not workflow_name:
        return {}
    parts = workflow_name.split(" / ")
    if len(parts) < 2:
        return {}
    return {"website": parts[0], "subdomain": parts[1].split(" workflow ")[0]}


def accept_reuse(policy: str, traj, workflow_name: str | None) -> bool:
    if workflow_name is None:
        return False
    if policy == "threshold":
        return True
    metadata = workflow_metadata(workflow_name)
    if policy == "same-website":
        return metadata.get("website") == traj.website
    if policy == "same-subdomain":
        return (
            metadata.get("website") == traj.website
            and metadata.get("subdomain") == traj.subdomain
        )
    return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "mind2web" / "data" / "memory" / "exemplars.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "mind2web" / "real_data_full_output",
    )
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--reuse-policy",
        choices=["same-website", "same-subdomain", "threshold"],
        default="same-website",
        help=(
            "Controls when the top retrieved workflow is accepted as reused. "
            "Candidates are always logged either way."
        ),
    )
    args = parser.parse_args()

    random.seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(args.data_path)
    if args.shuffle:
        random.shuffle(trajectories)
    if args.max_tasks is not None:
        trajectories = trajectories[: args.max_tasks]

    workflow_memory_path = args.output_dir / "workflow_memory.txt"
    workflow_embedding_path = args.output_dir / "workflow_embeddings.json"
    trace_path = args.output_dir / "llm_trace.json"
    episodic_path = args.output_dir / "episodic_memory.json"
    report_path = args.output_dir / "report.md"

    embedding_index = WorkflowEmbeddingIndex(workflow_embedding_path)
    workflow_blocks = []
    trace = []
    episodes = []
    reuse_count = 0
    same_website_reuse = 0
    same_subdomain_reuse = 0

    for index, traj in enumerate(trajectories, start=1):
        retrieval = embedding_index.retrieve(query_text(traj), top_k=args.top_k)
        retrieved_workflow = retrieval.workflow_name
        reused_workflow = (
            retrieved_workflow
            if accept_reuse(args.reuse_policy, traj, retrieved_workflow)
            else None
        )
        metadata = workflow_metadata(reused_workflow)
        reuse_count += int(reused_workflow is not None)
        same_website_reuse += int(metadata.get("website") == traj.website)
        same_subdomain_reuse += int(
            metadata.get("website") == traj.website
            and metadata.get("subdomain") == traj.subdomain
        )

        workflow = make_workflow(traj, index)
        workflow_blocks.append(workflow.strip())
        embedding_index.add_workflow(workflow, save=False)

        memory_stores = {
            "working_memory": "The prompt field for this task; rebuilt before each trajectory.",
            "short_term_memory": "The per-step ground-truth action history inside this trajectory.",
            "episodic_memory": str(episodic_path),
            "long_term_workflow_memory": str(workflow_memory_path),
            "workflow_embedding_index": str(workflow_embedding_path),
        }
        episode = {
            "task_index": index,
            "specifier": traj.specifier,
            "website": traj.website,
            "domain": traj.domain,
            "subdomain": traj.subdomain,
            "goal": traj.task,
            "success": True,
            "top_retrieved_workflow": retrieved_workflow,
            "reused_workflow": reused_workflow,
            "reuse_policy": args.reuse_policy,
            "same_website_reuse": metadata.get("website") == traj.website,
            "same_subdomain_reuse": (
                metadata.get("website") == traj.website
                and metadata.get("subdomain") == traj.subdomain
            ),
            "added_workflow": workflow.splitlines()[0].lstrip("#").strip(),
            "semantic_retrieval_candidates": retrieval.candidates,
            "ground_truth_actions": traj.actions,
        }
        episodes.append(episode)
        trace.append(
            {
                **episode,
                "memory_stores": memory_stores,
                "long_term_workflow_count_before_task": index - 1,
                "long_term_workflow_count_after_task": index,
                "prompt": build_prompt(traj, retrieval),
                "steps": trace_steps(traj, reused_workflow),
            }
        )

        if index % 100 == 0:
            print(f"processed {index}/{len(trajectories)} tasks")

    workflow_memory_path.write_text("\n\n".join(workflow_blocks) + "\n")
    embedding_index.save()
    trace_path.write_text(json.dumps(trace, indent=2))
    episodic_path.write_text(json.dumps(episodes, indent=2))

    website_counts = Counter(traj.website for traj in trajectories)
    domain_counts = Counter(traj.domain for traj in trajectories)
    summary = {
        "data_path": str(args.data_path),
        "tasks_processed": len(trajectories),
        "workflow_count": len(workflow_blocks),
        "episodes": len(episodes),
        "retrieval_backend": embedding_index.backend,
        "embedding_model": embedding_index.model_name if embedding_index.model is not None else None,
        "reuse_policy": args.reuse_policy,
        "reuse_count": reuse_count,
        "reuse_rate": reuse_count / len(trajectories) if trajectories else 0.0,
        "same_website_reuse_count": same_website_reuse,
        "same_website_reuse_rate": same_website_reuse / reuse_count if reuse_count else 0.0,
        "same_subdomain_reuse_count": same_subdomain_reuse,
        "same_subdomain_reuse_rate": same_subdomain_reuse / reuse_count if reuse_count else 0.0,
        "top_websites": website_counts.most_common(15),
        "domains": domain_counts.most_common(),
        "workflow_memory_path": str(workflow_memory_path),
        "workflow_embedding_index_path": str(workflow_embedding_path),
        "trace_path": str(trace_path),
        "episodic_memory_path": str(episodic_path),
        "report_path": str(report_path),
    }

    report_lines = [
        "# Real Mind2Web AWM Full Run",
        "",
        "This is an offline, online-style AWM run over real Mind2Web exemplar trajectories.",
        "Each task retrieves from current workflow memory, records an episode, then stores a new workflow.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Memory Stores",
        "",
        f"- Working memory: `{trace_path}` under each task's `prompt` field",
        f"- Short-term memory: `{trace_path}` under each step's `short_term_before` / `short_term_after`",
        f"- Episodic memory: `{episodic_path}`",
        f"- Long-term workflow memory: `{workflow_memory_path}`",
        f"- Workflow embedding index: `{workflow_embedding_path}`",
        "",
        "## First Five Episodes",
        "",
    ]
    for item in trace[:5]:
        report_lines.extend(
            [
                f"### Task {item['task_index']}: {item['website']} / {item['subdomain']}",
                "",
                f"Goal: {item['goal']}",
                f"Reused workflow: {item['reused_workflow'] or 'None'}",
                f"Top retrieved workflow: {item['top_retrieved_workflow'] or 'None'}",
                f"Added workflow: {item['added_workflow']}",
                f"Workflow memory count before/after: {item['long_term_workflow_count_before_task']} -> {item['long_term_workflow_count_after_task']}",
                "",
                "Top retrieval candidates:",
                "",
                "```json",
                json.dumps(item["semantic_retrieval_candidates"][:3], indent=2),
                "```",
                "",
            ]
        )
    report_path.write_text("\n".join(report_lines).rstrip() + "\n")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
