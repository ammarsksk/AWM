"""Verbose workflow-memory retrieval trace for Mind2Web exemplars.

This is a debug/demo runner, not the main evaluator. It shows exactly when
workflow embeddings are created, what query text is embedded, and how FAISS
semantic scores, BM25 scores, and combined scores rank stored workflows.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import json
import math
import resource
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mind2web"))
sys.path.insert(0, str(ROOT / "webarena"))

from mind2web.llm_step_eval import (  # noqa: E402
    add_workflow_to_container,
    accept_reuse,
    compact_candidates,
    make_memory_workflow,
    make_llm_client,
    make_structured_workflow,
    make_workflow_container,
    predict_llm,
    query_text,
    structured_steps_from_trajectory,
    workflow_for_prompt,
    workflow_to_text,
    workflows_to_dict,
)
from mind2web.real_data_awm_smoke import load_trajectories  # noqa: E402
from mind2web.lancedb_memory import LanceWorkflowMemory  # noqa: E402
from webarena.local_awm_full_demo import WorkflowEmbeddingIndex  # noqa: E402


def vector_summary(vector: list[float], preview: int = 8) -> str:
    norm = math.sqrt(sum(value * value for value in vector))
    values = ", ".join(f"{value:+.4f}" for value in vector[:preview])
    return f"dim={len(vector)} norm={norm:.4f} first_{preview}=[{values}]"


def short(text: str, limit: int = 220) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[: limit - 4] + " ..."


def block(text: str, indent: str = "    ", width: int = 112) -> str:
    return "\n".join(
        textwrap.wrap(
            " ".join(str(text).split()),
            width=width,
            initial_indent=indent,
            subsequent_indent=indent,
            break_long_words=False,
            break_on_hyphens=False,
        )
    )


def concise_step_list(workflow: dict, limit: int = 6) -> list[str]:
    lines = []
    for step in workflow.get("steps", [])[:limit]:
        if workflow.get("abstraction_type"):
            lines.append(
                "{step}. {op} | {intent} | target={target}".format(
                    step=step.get("step", len(lines) + 1),
                    op=step.get("operation", ""),
                    intent=short(step.get("step_intent", ""), 90),
                    target=short(step.get("target_description", ""), 90),
                )
            )
        else:
            lines.append(
                "{step}. {op} | role={role} label={label} value={value}".format(
                    step=len(lines) + 1,
                    op=step.get("op", ""),
                    role=short(step.get("role", ""), 40),
                    label=short(step.get("label", ""), 70),
                    value=short(step.get("value", ""), 50),
                )
            )
    if len(workflow.get("steps", [])) > limit:
        lines.append(f"... {len(workflow.get('steps', [])) - limit} more steps")
    return lines


def truncate_workflow_steps(workflow: dict, max_steps: int | None) -> dict:
    if max_steps is None:
        return workflow
    workflow = dict(workflow)
    workflow["steps"] = list(workflow.get("steps", []))[:max_steps]
    return workflow


def rss_mb() -> float:
    """Return max resident set size in MiB on Linux."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def entry_memory_summary(index) -> str:
    if hasattr(index, "stats") and not hasattr(index, "entries"):
        stats = index.stats()
        return (
            f"entries={stats['workflow_count']} "
            f"python_vector_values={stats['python_vector_values']} "
            f"python_vector_mb={stats['python_vector_mb']:.4f} "
            f"json_cache_items={stats['workflow_json_cache_items']} "
            f"process_rss_mb={rss_mb():.2f}"
        )
    vector_values = sum(len(entry.get("vector", [])) for entry in index.entries)
    vector_mb = vector_values * 4 / (1024 * 1024)
    text_bytes = sum(
        len(entry.get("text_for_embedding", "").encode("utf-8"))
        for entry in index.entries
    )
    return (
        f"entries={len(index.entries)} "
        f"vector_values={vector_values} "
        f"approx_vector_mb={vector_mb:.4f} "
        f"text_kb={text_bytes / 1024:.2f} "
        f"process_rss_mb={rss_mb():.2f}"
    )


def workflow_count(index) -> int:
    if hasattr(index, "entries"):
        return len(index.entries)
    if hasattr(index, "stats"):
        return int(index.stats().get("workflow_count", 0))
    return 0


def workflow_text_for_embedding(index, workflow_text: str) -> str:
    if hasattr(index, "workflow_text_for_embedding"):
        return index.workflow_text_for_embedding(workflow_text)
    if hasattr(index, "embedder"):
        return index.embedder.workflow_text_for_embedding(workflow_text)
    return workflow_text


def print_candidates(candidates: list[dict], indent: str = "    ") -> None:
    if not candidates:
        print(f"{indent}no retrieval candidates yet")
        return
    for rank, candidate in enumerate(candidates, start=1):
        print(
            f"{indent}{rank}. {candidate['workflow_name']} | "
            f"combined={candidate['combined_score']:.4f} "
            f"semantic={candidate['semantic_score']:.4f} "
            f"bm25={candidate['bm25_score']:.4f}"
        )
        print(f"{indent}   embedding text: {short(candidate.get('text_for_embedding', ''))}")


def print_retrieval_table(candidates: list[dict], accepted_workflow: str | None) -> None:
    if not candidates:
        print("    no stored workflows yet")
        return
    print("    rank  status    combined  semantic  bm25    workflow")
    print("    ----  --------  --------  --------  ------  ------------------------------")
    for rank, candidate in enumerate(candidates, start=1):
        name = candidate["workflow_name"]
        status = "accepted" if name == accepted_workflow else "candidate"
        print(
            f"    {rank:<4}  {status:<8}  "
            f"{candidate['combined_score']:<8.4f}  "
            f"{candidate['semantic_score']:<8.4f}  "
            f"{candidate['bm25_score']:<6.4f}  "
            f"{name}"
        )


def print_element_candidates(candidates: list[dict], limit: int = 8) -> None:
    if not candidates:
        print("    no parsed candidate elements")
        return
    for item in candidates[:limit]:
        print(
            "    "
            f"id={item.get('id')} tag={item.get('tag')} "
            f"role={short(item.get('role_hint', ''), 45)!r} "
            f"label={short(item.get('label', ''), 45)!r} "
            f"text={short(item.get('nearby_text', ''), 90)!r}"
        )
    if len(candidates) > limit:
        print(f"    ... {len(candidates) - limit} more candidates omitted")


def print_element_summary(candidates: list[dict]) -> None:
    tags: dict[str, int] = {}
    examples = []
    for item in candidates:
        tags[item.get("tag", "")] = tags.get(item.get("tag", ""), 0) + 1
        label = item.get("label") or item.get("role_hint") or item.get("nearby_text", "")
        if label and len(examples) < 4:
            examples.append(f"{item.get('id')}:{short(label, 35)}")
    tag_summary = ", ".join(f"{tag}={count}" for tag, count in sorted(tags.items()))
    example_summary = "; ".join(examples) if examples else "none"
    print(f"    count={len(candidates)} tags=[{tag_summary}] examples=[{example_summary}]")


def print_query_summary(traj, observation: str, candidates: list[dict], args) -> None:
    print(f"    website={traj.website} | domain={traj.domain} | subdomain={traj.subdomain}")
    print(f"    task={traj.task}")
    print(f"    observation_chars={len(observation)} | parsed_candidates={len(candidates)}")
    if args.show_observation:
        print("    observation:")
        print(block(observation, width=args.terminal_width))
    else:
        print(f"    observation_preview={short(observation, args.observation_preview_chars)}")
        print("    full observation hidden; add --show-observation to print raw page snapshot")


def run(args: argparse.Namespace) -> None:
    if args.retrieval_backend == "lancedb" and args.workflow_storage != "disk":
        print("forcing workflow_storage=disk because LanceDB stores json_path pointers")
        args.workflow_storage = "disk"
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_path = args.data_path
    trajectories = load_trajectories(data_path)
    selected = trajectories[args.start_index : args.start_index + args.num_tasks]

    if args.retrieval_backend == "lancedb":
        embedding_index = LanceWorkflowMemory(
            db_path=output_dir / "lancedb",
            workflow_json_root=output_dir / "workflow_json",
            candidate_k=args.lancedb_candidate_k,
            cache_size=args.workflow_cache_size,
            index_type=args.lancedb_index_type,
            reindex_every=args.lancedb_reindex_every,
            min_index_rows=args.lancedb_min_index_rows,
        )
    else:
        embedding_index = WorkflowEmbeddingIndex(output_dir / "workflow_embeddings.json")
    llm_client = make_llm_client() if args.use_llm else None
    abstraction_client = make_llm_client() if args.workflow_abstraction == "llm" else None
    workflows = make_workflow_container(output_dir, args.workflow_storage)
    raw_workflows = make_workflow_container(output_dir / "raw", args.workflow_storage)
    workflow_texts: list[str] = []
    trace: list[dict] = []

    print("=" * 88)
    print("AWM embedding/retrieval trace demo")
    print("=" * 88)
    print(f"data_path: {data_path}")
    print(f"output_dir: {output_dir}")
    print(f"tasks selected: {len(selected)}")
    print(f"top_k: {args.top_k}")
    print(f"reuse_policy: {args.reuse_policy}")
    print(f"use_llm: {args.use_llm}")
    if args.use_llm:
        print(f"llm_model: {args.model}")
    print(f"workflow_abstraction: {args.workflow_abstraction}")
    if args.workflow_abstraction == "llm":
        print(f"abstraction_model: {args.abstraction_model or args.model}")
    print(f"workflow_storage: {args.workflow_storage}")
    print(f"retrieval_backend: {args.retrieval_backend}")
    if args.retrieval_backend == "lancedb":
        print(f"lancedb_candidate_k: {args.lancedb_candidate_k}")
        print(f"lancedb_index_type: {args.lancedb_index_type}")
        print(f"workflow_cache_size: {args.workflow_cache_size}")
    print(f"show_candidates: {args.show_candidates}")
    print(f"show_prompt_json: {args.show_prompt_json}")
    print(f"embedding backend: {embedding_index.backend}")
    print(f"embedding model/source: {embedding_index.model_source or embedding_index.model_name}")
    print(f"initial RAM/index: {entry_memory_summary(embedding_index)}")
    print("=" * 88)

    for local_index, traj in enumerate(selected, start=1):
        task_index = args.start_index + local_index
        print()
        print("#" * 88)
        print(f"TASK {task_index}: {traj.task}")
        print(f"website={traj.website} domain={traj.domain} subdomain={traj.subdomain}")
        print(f"memory size before task: {workflow_count(embedding_index)} workflows")
        print(f"RAM/index before task: {entry_memory_summary(embedding_index)}")
        print("#" * 88)

        structured_steps = structured_steps_from_trajectory(traj)
        if args.max_steps_per_task is not None:
            structured_steps = structured_steps[: args.max_steps_per_task]

        task_trace = {
            "task_index": task_index,
            "website": traj.website,
            "domain": traj.domain,
            "subdomain": traj.subdomain,
            "goal": traj.task,
            "steps": [],
            "added_workflow": None,
        }
        previous_actions: list[str] = []

        for step_index, gold_step in enumerate(structured_steps, start=1):
            observation = gold_step.get("observation", "")
            query = query_text(traj, observation)
            query_vector = embedding_index.embed(query)
            metadata = {
                "website": traj.website,
                "domain": traj.domain,
                "subdomain": traj.subdomain,
            }
            if args.retrieval_backend == "lancedb":
                retrieval = embedding_index.retrieve(
                    query,
                    top_k=args.top_k,
                    metadata=metadata,
                    candidate_k=args.lancedb_candidate_k,
                )
            else:
                retrieval = embedding_index.retrieve(query, top_k=args.top_k)
            top_workflow = retrieval.workflow_name
            accepted_workflow = (
                top_workflow
                if accept_reuse(args.reuse_policy, traj, top_workflow)
                else None
            )
            prompt_workflows = workflows
            if args.retrieval_backend == "lancedb" and accepted_workflow:
                accepted_candidate = next(
                    (
                        candidate
                        for candidate in retrieval.candidates
                        if candidate.get("workflow_name") == accepted_workflow
                    ),
                    None,
                )
                if accepted_candidate is not None:
                    loaded_workflow = embedding_index.load_workflow_json(accepted_candidate)
                    if loaded_workflow is not None:
                        prompt_workflows = {accepted_workflow: loaded_workflow}
            candidates = compact_candidates(observation)

            print()
            print(f"  STEP {step_index}")
            print("  " + "-" * 84)
            print("  query summary:")
            print_query_summary(traj, observation, candidates, args)
            print(f"  query embedding: {vector_summary(query_vector)}")
            print("  candidate elements:")
            if args.show_candidates:
                print_element_candidates(candidates, limit=args.element_preview_count)
            else:
                print_element_summary(candidates)
                print("    details hidden; add --show-candidates to print every candidate preview")
            print("  retrieval:")
            print_retrieval_table(retrieval.candidates, accepted_workflow)
            print(f"  top_retrieved_workflow: {top_workflow}")
            print(f"  accepted_workflow: {accepted_workflow}")
            print(f"  gold action, shown for trace only: {gold_step.get('action', '')}")

            raw_llm_output = None
            parsed_llm_output = None
            predicted_action = None
            if args.use_llm:
                prompt_payload = {
                    "task": traj.task,
                    "website": traj.website,
                    "domain": traj.domain,
                    "subdomain": traj.subdomain,
                    "current_observation": short(observation, args.query_preview_chars),
                    "candidate_elements_preview": candidates[: args.element_preview_count],
                    "previous_actions": previous_actions,
                    "retrieval_candidates": [
                        {
                            key: candidate.get(key)
                            for key in [
                                "workflow_name",
                                "combined_score",
                                "semantic_score",
                                "bm25_score",
                            ]
                        }
                        for candidate in retrieval.candidates[:5]
                    ],
                    "accepted_workflow": workflow_for_prompt(prompt_workflows, accepted_workflow),
                }
                print("  LLM prompt summary:")
                if args.show_prompt_json:
                    print(block(json.dumps(prompt_payload, indent=2), width=args.terminal_width))
                else:
                    print(f"    task: {traj.task}")
                    print(f"    previous_actions: {previous_actions}")
                    print(f"    retrieval_candidates: {len(retrieval.candidates)}")
                    print(f"    accepted_workflow_present: {accepted_workflow is not None}")
                    print("    prompt JSON hidden; add --show-prompt-json to print it")
                predicted_action, raw_llm_output, parsed_llm_output = predict_llm(
                    client=llm_client,
                    model=args.model,
                    traj=traj,
                    observation=observation,
                    previous_actions=previous_actions,
                    retrieval_candidates=retrieval.candidates,
                    workflows=prompt_workflows,
                    accepted_workflow=accepted_workflow,
                    max_output_tokens=args.max_output_tokens,
                    retries=args.llm_retries,
                    retry_sleep=args.retry_sleep,
                )
                previous_actions.append(predicted_action)
                print(f"  LLM predicted_action: {predicted_action}")
                print("  LLM raw output:")
                print(block(raw_llm_output or "", width=args.terminal_width))

            task_trace["steps"].append(
                {
                    "step": step_index,
                    "query_text": query,
                    "query_embedding_preview": query_vector[: args.embedding_preview_count],
                    "candidate_elements": candidates,
                    "retrieval_candidates": retrieval.candidates,
                    "top_retrieved_workflow": top_workflow,
                    "accepted_workflow": accepted_workflow,
                    "gold_action_for_trace_only": gold_step.get("action", ""),
                    "predicted_action": predicted_action,
                    "raw_llm_output": raw_llm_output,
                    "parsed_llm_output": parsed_llm_output,
                }
            )

        raw_workflow = truncate_workflow_steps(
            make_structured_workflow(traj, task_index),
            args.workflow_max_steps,
        )
        workflow, abstraction_raw_output = make_memory_workflow(
            raw_workflow,
            args,
            abstraction_client=abstraction_client,
        )
        workflow_text = workflow_to_text(workflow)
        workflow_embedding_text = workflow_text_for_embedding(embedding_index, workflow_text)
        workflow_vector = embedding_index.embed(workflow_embedding_text)

        print()
        print("  WORKFLOW STORAGE AFTER TASK")
        print("  " + "-" * 84)
        print(f"  raw workflow name: {raw_workflow['name']}")
        print(f"  raw workflow steps stored in this demo: {len(raw_workflow.get('steps', []))}")
        print(f"  memory workflow name: {workflow['name']}")
        print(f"  abstraction type: {workflow.get('abstraction_type', 'raw')}")
        print(f"  memory workflow steps: {len(workflow.get('steps', []))}")
        if abstraction_raw_output is not None:
            print("  abstraction LLM raw output:")
            print(block(abstraction_raw_output, width=args.terminal_width))
        print("  abstract workflow steps:")
        for line in concise_step_list(workflow, limit=args.workflow_step_preview_count):
            print(f"    {line}")
        print("  text used for embedding:")
        print(block(workflow_embedding_text, width=args.terminal_width))
        print(f"  workflow embedding: {vector_summary(workflow_vector)}")
        print(f"  RAM/index before storing workflow: {entry_memory_summary(embedding_index)}")

        add_workflow_to_container(raw_workflows, raw_workflow)
        workflow_json_path = add_workflow_to_container(workflows, workflow)
        workflow_texts.append(workflow_text)
        if args.retrieval_backend == "lancedb":
            embedding_index.add_workflow(
                workflow_text,
                workflow=workflow,
                json_path=workflow_json_path,
                save=False,
            )
        else:
            embedding_index.add_workflow(workflow_text, save=False)
        task_trace["added_workflow"] = workflow["name"]
        task_trace["abstraction_type"] = workflow.get("abstraction_type", "raw")
        task_trace["abstraction_raw_output"] = abstraction_raw_output
        task_trace["workflow_embedding_preview"] = workflow_vector[: args.embedding_preview_count]
        trace.append(task_trace)

        print(f"  memory size after task: {workflow_count(embedding_index)} workflows")
        if args.workflow_storage == "disk":
            print(f"  abstract workflow JSON written under: {output_dir / 'workflow_json'}")
            print(f"  raw workflow JSON written under: {output_dir / 'raw' / 'workflow_json'}")
        print(f"  RAM/index after storing workflow: {entry_memory_summary(embedding_index)}")

    (output_dir / "embedding_retrieval_trace.json").write_text(json.dumps(trace, indent=2))
    (output_dir / "structured_workflows.json").write_text(json.dumps(workflows_to_dict(workflows), indent=2))
    (output_dir / "raw_structured_workflows.json").write_text(json.dumps(workflows_to_dict(raw_workflows), indent=2))
    (output_dir / "workflow_memory.txt").write_text("\n\n".join(workflow_texts) + "\n")
    embedding_index.save()
    if args.retrieval_backend == "lancedb" and args.lancedb_final_index:
        final_index_ms = embedding_index.finalize_index()
        print(f"final LanceDB ANN index refresh: {final_index_ms:.2f} ms")
        embedding_index.save()
    if hasattr(embedding_index, "stats"):
        (output_dir / "memory_perf.json").write_text(json.dumps(embedding_index.stats(), indent=2))

    print()
    print("=" * 88)
    print("done")
    print(f"wrote: {output_dir / 'embedding_retrieval_trace.json'}")
    print(f"wrote: {output_dir / 'structured_workflows.json'}")
    print(f"wrote: {output_dir / 'raw_structured_workflows.json'}")
    print(f"wrote: {output_dir / 'workflow_memory.txt'}")
    if args.retrieval_backend == "ram":
        print(f"wrote: {output_dir / 'workflow_embeddings.json'}")
    else:
        print(f"wrote LanceDB long-term memory under: {output_dir / 'lancedb'}")
    if hasattr(embedding_index, "stats"):
        print(f"wrote: {output_dir / 'memory_perf.json'}")
    if args.workflow_storage == "disk":
        print(f"wrote individual abstract workflow JSON files under: {output_dir / 'workflow_json'}")
        print(f"wrote individual raw workflow JSON files under: {output_dir / 'raw' / 'workflow_json'}")
    print(f"final RAM/index: {entry_memory_summary(embedding_index)}")
    print("=" * 88)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print a verbose Mind2Web workflow embedding and retrieval trace."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=ROOT / "mind2web" / "data" / "memory" / "exemplars.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "mind2web" / "embedding_trace_demo_output",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=4)
    parser.add_argument("--max-steps-per-task", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--use-llm", action="store_true")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--retrieval-backend", choices=["ram", "lancedb"], default="ram")
    parser.add_argument("--lancedb-candidate-k", type=int, default=50)
    parser.add_argument("--lancedb-index-type", type=str, default="IVF_SQ")
    parser.add_argument("--lancedb-reindex-every", type=int, default=0)
    parser.add_argument("--lancedb-min-index-rows", type=int, default=32)
    parser.add_argument("--lancedb-final-index", action="store_true")
    parser.add_argument("--workflow-cache-size", type=int, default=128)
    parser.add_argument(
        "--workflow-abstraction",
        choices=["raw", "deterministic", "llm"],
        default="raw",
    )
    parser.add_argument("--workflow-storage", choices=["ram", "disk"], default="ram")
    parser.add_argument("--abstraction-model", type=str, default=None)
    parser.add_argument("--abstraction-max-output-tokens", type=int, default=4096)
    parser.add_argument("--abstraction-max-steps", type=int, default=18)
    parser.add_argument(
        "--workflow-max-steps",
        type=int,
        default=None,
        help=(
            "Limit how many raw trajectory steps are stored/abstracted in this demo. "
            "Use this for clean presentation traces."
        ),
    )
    parser.add_argument("--llm-retries", type=int, default=2)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument("--max-output-tokens", type=int, default=1024)
    parser.add_argument(
        "--reuse-policy",
        choices=["same-website", "same-subdomain", "threshold"],
        default="same-website",
    )
    parser.add_argument("--terminal-width", type=int, default=112)
    parser.add_argument("--query-preview-chars", type=int, default=700)
    parser.add_argument("--workflow-preview-chars", type=int, default=900)
    parser.add_argument("--prompt-preview-chars", type=int, default=2500)
    parser.add_argument("--llm-output-preview-chars", type=int, default=1200)
    parser.add_argument("--element-preview-count", type=int, default=8)
    parser.add_argument("--embedding-preview-count", type=int, default=16)
    parser.add_argument("--workflow-step-preview-count", type=int, default=6)
    parser.add_argument("--show-candidates", action="store_true")
    parser.add_argument("--show-prompt-json", action="store_true")
    parser.add_argument("--show-observation", action="store_true")
    parser.add_argument("--observation-preview-chars", type=int, default=220)
    return parser.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
