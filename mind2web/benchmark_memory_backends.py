"""Compare RAM FAISS workflow memory against LanceDB disk-backed memory.

The parent command launches each backend in a fresh Python process so RSS and
latency measurements are not polluted by whichever backend ran first.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import time
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mind2web"))
sys.path.insert(0, str(ROOT / "webarena"))

from mind2web.lancedb_memory import LanceWorkflowMemory  # noqa: E402
from mind2web.memmap_memory import MemmapWorkflowMemory  # noqa: E402
from mind2web.compressed_faiss_memory import CompressedFaissWorkflowMemory  # noqa: E402
from mind2web.llm_step_eval import (  # noqa: E402
    add_workflow_to_container,
    accept_reuse,
    make_memory_workflow,
    make_structured_workflow,
    make_workflow_container,
    query_text,
    structured_steps_from_trajectory,
    workflow_to_text,
    workflows_to_dict,
)
from mind2web.real_data_awm_smoke import load_trajectories  # noqa: E402
from webarena.local_awm_full_demo import WorkflowEmbeddingIndex  # noqa: E402


def current_rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    except Exception:
        return 0.0
    return 0.0


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def average(values: list[float]) -> float:
    return mean(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def backend_stats(backend, output_dir: Path) -> dict:
    if isinstance(backend, WorkflowEmbeddingIndex):
        vector_values = sum(len(entry.get("vector", [])) for entry in backend.entries)
        text_bytes = sum(
            len(entry.get("text_for_embedding", "").encode("utf-8"))
            for entry in backend.entries
        )
        return {
            "backend": "ram_faiss_bm25",
            "workflow_count": len(backend.entries),
            "python_vector_count": len(backend.entries),
            "python_vector_values": vector_values,
            "python_vector_mb": vector_values * 4 / (1024 * 1024),
            "embedding_text_kb": text_bytes / 1024,
            "disk_bytes": directory_bytes(output_dir),
        }
    stats = backend.stats()
    stats["disk_bytes"] = directory_bytes(output_dir)
    return stats


def make_backend(args, output_dir: Path):
    if args.backend == "ram":
        return WorkflowEmbeddingIndex(output_dir / "workflow_embeddings.json")
    if args.backend == "memmap":
        return MemmapWorkflowMemory(
            root=output_dir / "memmap",
            workflow_json_root=output_dir / "workflow_json",
            candidate_k=args.candidate_k,
            chunk_size=args.memmap_chunk_size,
            vector_dtype=args.memmap_dtype,
            cache_size=args.workflow_cache_size,
        )
    if args.backend == "compressed_faiss":
        return CompressedFaissWorkflowMemory(
            root=output_dir / "compressed_faiss",
            workflow_json_root=output_dir / "workflow_json",
            candidate_k=args.candidate_k,
            cache_size=args.workflow_cache_size,
            index_kind=args.compressed_faiss_kind,
            ivf_nlist=args.compressed_faiss_nlist,
            pq_m=args.compressed_faiss_pq_m,
            pq_bits=args.compressed_faiss_pq_bits,
        )
    return LanceWorkflowMemory(
        db_path=output_dir / "lancedb",
        workflow_json_root=output_dir / "workflow_json",
        candidate_k=args.candidate_k,
        cache_size=args.workflow_cache_size,
        index_type=args.lancedb_index_type,
        reindex_every=args.lancedb_reindex_every,
        min_index_rows=args.lancedb_min_index_rows,
    )


def retrieve(backend, query: str, traj, args):
    metadata = {
        "website": traj.website,
        "domain": traj.domain,
        "subdomain": traj.subdomain,
    }
    if isinstance(backend, (LanceWorkflowMemory, MemmapWorkflowMemory, CompressedFaissWorkflowMemory)):
        return backend.retrieve(
            query,
            top_k=args.top_k,
            metadata=metadata,
            candidate_k=args.candidate_k,
        )
    return backend.retrieve(query, top_k=args.top_k)


def add_to_backend(backend, workflow_text: str, workflow: dict, workflow_json_path: Path | None) -> None:
    if isinstance(backend, LanceWorkflowMemory):
        backend.add_workflow(workflow_text, workflow=workflow, json_path=workflow_json_path, save=False)
    elif isinstance(backend, (MemmapWorkflowMemory, CompressedFaissWorkflowMemory)):
        # Memmap is intentionally batch-oriented for long-term memory builds.
        backend.add_workflows_batch([(workflow_text, workflow, workflow_json_path)], build_index=False)
    else:
        backend.add_workflow(workflow_text, save=False)


def run_child(args) -> int:
    output_dir = args.output_dir / args.backend
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = load_trajectories(args.data_path)
    selected = trajectories[args.start_index : args.start_index + args.num_tasks]

    backend = make_backend(args, output_dir)
    workflows = make_workflow_container(output_dir, "disk")
    raw_workflows = make_workflow_container(output_dir / "raw", "disk")
    workflow_texts: list[str] = []
    events: list[dict] = []
    retrieval_ms: list[float] = []
    add_ms: list[float] = []
    rss_samples: list[float] = [current_rss_mb()]
    accepted_count = 0
    steps = 0
    run_start = time.perf_counter()

    abstraction_args = Namespace(
        workflow_abstraction=args.workflow_abstraction,
        abstraction_model=None,
        model="",
        abstraction_max_output_tokens=4096,
        abstraction_max_steps=args.workflow_max_steps or 18,
        llm_retries=0,
        retry_sleep=0,
    )

    if args.ingest_mode == "batch":
        return run_child_batch(args, output_dir, selected, backend, workflows, raw_workflows)

    for local_index, traj in enumerate(selected, start=1):
        task_index = args.start_index + local_index
        structured_steps = structured_steps_from_trajectory(traj)
        if args.max_steps_per_task is not None:
            structured_steps = structured_steps[: args.max_steps_per_task]

        task_events = []
        for step_index, gold_step in enumerate(structured_steps, start=1):
            query = query_text(traj, gold_step.get("observation", ""))
            start = time.perf_counter()
            result = retrieve(backend, query, traj, args)
            elapsed = (time.perf_counter() - start) * 1000.0
            retrieval_ms.append(elapsed)
            top_workflow = result.workflow_name
            accepted_workflow = (
                top_workflow if accept_reuse(args.reuse_policy, traj, top_workflow) else None
            )
            if accepted_workflow and isinstance(backend, LanceWorkflowMemory):
                accepted_candidate = next(
                    (
                        candidate
                        for candidate in result.candidates
                        if candidate.get("workflow_name") == accepted_workflow
                    ),
                    None,
                )
                if accepted_candidate is not None:
                    backend.load_workflow_json(accepted_candidate)
            accepted_count += int(bool(accepted_workflow))
            steps += 1
            rss_samples.append(current_rss_mb())
            task_events.append(
                {
                    "step": step_index,
                    "retrieval_ms": elapsed,
                    "top_retrieved_workflow": top_workflow,
                    "accepted_workflow": accepted_workflow,
                    "candidate_count": len(result.candidates),
                    "top_candidates": result.candidates,
                    "rss_mb_after_retrieval": rss_samples[-1],
                }
            )

        raw_workflow = make_structured_workflow(traj, task_index)
        if args.workflow_max_steps is not None:
            raw_workflow = dict(raw_workflow)
            raw_workflow["steps"] = raw_workflow["steps"][: args.workflow_max_steps]
        workflow, _ = make_memory_workflow(raw_workflow, abstraction_args)
        workflow_text = workflow_to_text(workflow)
        add_workflow_to_container(raw_workflows, raw_workflow)
        workflow_json_path = add_workflow_to_container(workflows, workflow)
        start = time.perf_counter()
        add_to_backend(backend, workflow_text, workflow, workflow_json_path)
        elapsed = (time.perf_counter() - start) * 1000.0
        add_ms.append(elapsed)
        workflow_texts.append(workflow_text)
        rss_samples.append(current_rss_mb())
        events.append(
            {
                "task_index": task_index,
                "website": traj.website,
                "domain": traj.domain,
                "subdomain": traj.subdomain,
                "goal": traj.task,
                "steps": task_events,
                "added_workflow": workflow["name"],
                "add_ms": elapsed,
                "rss_mb_after_add": rss_samples[-1],
            }
        )

    online_total_ms = (time.perf_counter() - run_start) * 1000.0
    final_index_ms = 0.0
    if isinstance(backend, LanceWorkflowMemory) and args.lancedb_final_index:
        index_start = time.perf_counter()
        final_index_ms = backend.finalize_index()
        # Keep wall-clock accounting robust even if backend returns 0.
        if final_index_ms == 0.0:
            final_index_ms = (time.perf_counter() - index_start) * 1000.0
        rss_samples.append(current_rss_mb())
    total_ms = (time.perf_counter() - run_start) * 1000.0
    backend.save()
    (output_dir / "events.json").write_text(json.dumps(events, indent=2))
    (output_dir / "structured_workflows.json").write_text(json.dumps(workflows_to_dict(workflows), indent=2))
    (output_dir / "raw_structured_workflows.json").write_text(json.dumps(workflows_to_dict(raw_workflows), indent=2))
    (output_dir / "workflow_memory.txt").write_text("\n\n".join(workflow_texts) + "\n")

    summary = {
        "backend": args.backend,
        "tasks": len(selected),
        "steps": steps,
        "accepted_retrievals": accepted_count,
        "total_ms": total_ms,
        "online_total_ms": online_total_ms,
        "final_index_ms": final_index_ms,
        "avg_retrieval_ms": average(retrieval_ms),
        "p95_retrieval_ms": percentile(retrieval_ms, 95),
        "avg_add_ms": average(add_ms),
        "p95_add_ms": percentile(add_ms, 95),
        "start_rss_mb": rss_samples[0],
        "end_rss_mb": rss_samples[-1] if rss_samples else 0.0,
        "peak_sampled_rss_mb": max(rss_samples) if rss_samples else 0.0,
        "rss_growth_mb": (rss_samples[-1] - rss_samples[0]) if rss_samples else 0.0,
        **backend_stats(backend, output_dir),
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def make_workflow_pair(traj, task_index: int, args, abstraction_args):
    raw_workflow = make_structured_workflow(traj, task_index)
    if args.workflow_max_steps is not None:
        raw_workflow = dict(raw_workflow)
        raw_workflow["steps"] = raw_workflow["steps"][: args.workflow_max_steps]
    workflow, _ = make_memory_workflow(raw_workflow, abstraction_args)
    workflow_text = workflow_to_text(workflow)
    return raw_workflow, workflow, workflow_text


def run_child_batch(args, output_dir: Path, selected: list, backend, workflows, raw_workflows) -> int:
    """Batch-ingest workflows, then benchmark retrieval.

    This models a long-term memory database that is built/compacted in the
    background instead of appending and querying after every individual task.
    """
    rss_samples = [current_rss_mb()]
    run_start = time.perf_counter()
    abstraction_args = Namespace(
        workflow_abstraction=args.workflow_abstraction,
        abstraction_model=None,
        model="",
        abstraction_max_output_tokens=4096,
        abstraction_max_steps=args.workflow_max_steps or 18,
        llm_retries=0,
        retry_sleep=0,
    )

    items = []
    workflow_texts = []
    prep_start = time.perf_counter()
    for local_index, traj in enumerate(selected, start=1):
        task_index = args.start_index + local_index
        raw_workflow, workflow, workflow_text = make_workflow_pair(
            traj,
            task_index,
            args,
            abstraction_args,
        )
        add_workflow_to_container(raw_workflows, raw_workflow)
        workflow_json_path = add_workflow_to_container(workflows, workflow)
        workflow_texts.append(workflow_text)
        items.append((workflow_text, workflow, workflow_json_path))
    prep_ms = (time.perf_counter() - prep_start) * 1000.0

    if isinstance(backend, LanceWorkflowMemory):
        batch = backend.add_workflows_batch(items, build_index=args.lancedb_final_index)
        add_ms_total = float(batch["batch_add_ms"])
        final_index_ms = float(batch["batch_index_ms"])
    elif isinstance(backend, (MemmapWorkflowMemory, CompressedFaissWorkflowMemory)):
        batch = backend.add_workflows_batch(items)
        add_ms_total = float(batch["batch_add_ms"])
        final_index_ms = float(batch.get("batch_index_ms", 0.0))
    else:
        add_start = time.perf_counter()
        for workflow_text, workflow, workflow_json_path in items:
            add_to_backend(backend, workflow_text, workflow, workflow_json_path)
        add_ms_total = (time.perf_counter() - add_start) * 1000.0
        final_index_ms = 0.0
    rss_samples.append(current_rss_mb())

    events = []
    retrieval_ms = []
    accepted_count = 0
    steps = 0
    retrieval_start = time.perf_counter()
    for local_index, traj in enumerate(selected, start=1):
        task_index = args.start_index + local_index
        structured_steps = structured_steps_from_trajectory(traj)
        if args.max_steps_per_task is not None:
            structured_steps = structured_steps[: args.max_steps_per_task]
        task_events = []
        for step_index, gold_step in enumerate(structured_steps, start=1):
            query = query_text(traj, gold_step.get("observation", ""))
            start = time.perf_counter()
            result = retrieve(backend, query, traj, args)
            elapsed = (time.perf_counter() - start) * 1000.0
            retrieval_ms.append(elapsed)
            top_workflow = result.workflow_name
            accepted_workflow = (
                top_workflow if accept_reuse(args.reuse_policy, traj, top_workflow) else None
            )
            if accepted_workflow and isinstance(backend, LanceWorkflowMemory):
                accepted_candidate = next(
                    (
                        candidate
                        for candidate in result.candidates
                        if candidate.get("workflow_name") == accepted_workflow
                    ),
                    None,
                )
                if accepted_candidate is not None:
                    backend.load_workflow_json(accepted_candidate)
            accepted_count += int(bool(accepted_workflow))
            steps += 1
            rss_samples.append(current_rss_mb())
            task_events.append(
                {
                    "step": step_index,
                    "retrieval_ms": elapsed,
                    "top_retrieved_workflow": top_workflow,
                    "accepted_workflow": accepted_workflow,
                    "candidate_count": len(result.candidates),
                    "top_candidates": result.candidates,
                    "rss_mb_after_retrieval": rss_samples[-1],
                }
            )
        events.append(
            {
                "task_index": task_index,
                "website": traj.website,
                "domain": traj.domain,
                "subdomain": traj.subdomain,
                "goal": traj.task,
                "steps": task_events,
            }
        )
    retrieval_total_ms = (time.perf_counter() - retrieval_start) * 1000.0
    total_ms = (time.perf_counter() - run_start) * 1000.0

    backend.save()
    (output_dir / "events.json").write_text(json.dumps(events, indent=2))
    (output_dir / "structured_workflows.json").write_text(json.dumps(workflows_to_dict(workflows), indent=2))
    (output_dir / "raw_structured_workflows.json").write_text(json.dumps(workflows_to_dict(raw_workflows), indent=2))
    (output_dir / "workflow_memory.txt").write_text("\n\n".join(workflow_texts) + "\n")

    summary = {
        "backend": args.backend,
        "ingest_mode": "batch",
        "tasks": len(selected),
        "steps": steps,
        "accepted_retrievals": accepted_count,
        "total_ms": total_ms,
        "online_total_ms": retrieval_total_ms,
        "prep_ms": prep_ms,
        "batch_add_ms": add_ms_total,
        "final_index_ms": final_index_ms,
        "avg_retrieval_ms": average(retrieval_ms),
        "p95_retrieval_ms": percentile(retrieval_ms, 95),
        "avg_add_ms": add_ms_total / len(selected) if selected else 0.0,
        "p95_add_ms": add_ms_total / len(selected) if selected else 0.0,
        "start_rss_mb": rss_samples[0],
        "end_rss_mb": rss_samples[-1] if rss_samples else 0.0,
        "peak_sampled_rss_mb": max(rss_samples) if rss_samples else 0.0,
        "rss_growth_mb": (rss_samples[-1] - rss_samples[0]) if rss_samples else 0.0,
        **backend_stats(backend, output_dir),
    }
    (output_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


def run_parent(args) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    backend_order = args.backends.split(",")
    for backend in backend_order:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--backend",
            backend,
            "--data-path",
            str(args.data_path),
            "--output-dir",
            str(args.output_dir),
            "--start-index",
            str(args.start_index),
            "--num-tasks",
            str(args.num_tasks),
            "--max-steps-per-task",
            str(args.max_steps_per_task),
            "--workflow-max-steps",
            str(args.workflow_max_steps),
            "--top-k",
            str(args.top_k),
            "--candidate-k",
            str(args.candidate_k),
            "--workflow-abstraction",
            args.workflow_abstraction,
            "--reuse-policy",
            args.reuse_policy,
            "--lancedb-index-type",
            args.lancedb_index_type,
            "--lancedb-reindex-every",
            str(args.lancedb_reindex_every),
            "--lancedb-min-index-rows",
            str(args.lancedb_min_index_rows),
            "--workflow-cache-size",
            str(args.workflow_cache_size),
            "--ingest-mode",
            args.ingest_mode,
            "--memmap-dtype",
            args.memmap_dtype,
            "--memmap-chunk-size",
            str(args.memmap_chunk_size),
            "--compressed-faiss-kind",
            args.compressed_faiss_kind,
            "--compressed-faiss-nlist",
            str(args.compressed_faiss_nlist),
            "--compressed-faiss-pq-m",
            str(args.compressed_faiss_pq_m),
            "--compressed-faiss-pq-bits",
            str(args.compressed_faiss_pq_bits),
        ]
        if args.lancedb_final_index:
            cmd.append("--lancedb-final-index")
        print(f"\n=== running {backend} backend ===")
        subprocess.run(cmd, check=True)
        summary_path = args.output_dir / backend / "benchmark_summary.json"
        summaries.append(json.loads(summary_path.read_text()))

    comparison = {
        "description": "RAM FAISS vs LanceDB disk-backed workflow memory benchmark",
        "runs": summaries,
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(comparison, indent=2))
    (args.output_dir / "comparison.md").write_text(render_markdown(summaries))
    print(f"\nwrote {args.output_dir / 'comparison.md'}")
    return 0


def render_markdown(summaries: list[dict]) -> str:
    lines = [
        "# Workflow Memory Backend Benchmark",
        "",
        "Same workload for both backends: same exemplar slice, same task/step limits, same abstraction mode.",
        "",
        "| Metric | RAM FAISS/BM25 | LanceDB Disk ANN | Memmap Disk Exact | Compressed FAISS |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    ram = next(
        (item for item in summaries if item.get("backend") in {"ram", "ram_faiss_bm25"}),
        {},
    )
    lance = next(
        (item for item in summaries if item.get("backend") in {"lancedb", "lancedb_disk_ann"}),
        {},
    )
    memmap = next(
        (item for item in summaries if item.get("backend") in {"memmap", "memmap_exact"}),
        {},
    )
    compressed = next(
        (item for item in summaries if item.get("backend") in {"compressed_faiss"}),
        {},
    )
    rows = [
        ("Tasks", "tasks", "{:.0f}"),
        ("Steps", "steps", "{:.0f}"),
        ("Accepted Retrievals", "accepted_retrievals", "{:.0f}"),
        ("Total Runtime ms", "total_ms", "{:.2f}"),
        ("Online Runtime ms", "online_total_ms", "{:.2f}"),
        ("Prep ms", "prep_ms", "{:.2f}"),
        ("Batch Add ms", "batch_add_ms", "{:.2f}"),
        ("Final Index ms", "final_index_ms", "{:.2f}"),
        ("Avg Retrieval ms", "avg_retrieval_ms", "{:.2f}"),
        ("P95 Retrieval ms", "p95_retrieval_ms", "{:.2f}"),
        ("Avg Add/Store ms", "avg_add_ms", "{:.2f}"),
        ("P95 Add/Store ms", "p95_add_ms", "{:.2f}"),
        ("Start RSS MB", "start_rss_mb", "{:.2f}"),
        ("End RSS MB", "end_rss_mb", "{:.2f}"),
        ("Peak Sampled RSS MB", "peak_sampled_rss_mb", "{:.2f}"),
        ("RSS Growth MB", "rss_growth_mb", "{:.2f}"),
        ("Python Vector MB", "python_vector_mb", "{:.4f}"),
        ("Disk Bytes", "disk_bytes", "{:.0f}"),
    ]
    for label, key, fmt in rows:
        lines.append(
            f"| {label} | {fmt.format(float(ram.get(key, 0)))} | "
            f"{fmt.format(float(lance.get(key, 0)))} | "
            f"{fmt.format(float(memmap.get(key, 0)))} | "
            f"{fmt.format(float(compressed.get(key, 0)))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- RAM FAISS/BM25 keeps all workflow vectors and embedding text in Python memory.",
            "- LanceDB keeps vector-search storage in the disk-backed Lance table and records `python_vector_mb` as zero for the workflow collection.",
            "- Memmap keeps vectors in a disk-backed NumPy memmap and avoids database-engine overhead; it uses exact chunked vector search.",
            "- Compressed FAISS keeps quantized FAISS codes instead of Python vector lists, preserving FAISS-speed retrieval with lower vector memory.",
            "- LanceDB online writes append to the table; ANN refresh can be deferred to the final/background index phase.",
            "- LanceDB may show higher fixed RSS on tiny runs because the database engine has startup/index overhead; the scaling benefit appears as workflow count grows.",
            "- Retrieval latency includes query embedding and backend search/reranking.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--backend", choices=["ram", "lancedb", "memmap", "compressed_faiss"], default="ram")
    parser.add_argument(
        "--backends",
        default="ram,lancedb,memmap,compressed_faiss",
        help="Comma-separated backend list for parent mode.",
    )
    parser.add_argument("--data-path", type=Path, default=ROOT / "mind2web" / "data" / "memory" / "exemplars.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "mind2web" / "memory_backend_benchmark")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--num-tasks", type=int, default=8)
    parser.add_argument("--max-steps-per-task", type=int, default=2)
    parser.add_argument("--workflow-max-steps", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--workflow-abstraction", choices=["raw", "deterministic"], default="deterministic")
    parser.add_argument("--reuse-policy", choices=["same-website", "same-subdomain", "threshold"], default="same-website")
    parser.add_argument("--lancedb-index-type", type=str, default="IVF_SQ")
    parser.add_argument(
        "--lancedb-reindex-every",
        type=int,
        default=0,
        help="Online ANN refresh interval. 0 disables per-add indexing.",
    )
    parser.add_argument("--lancedb-min-index-rows", type=int, default=32)
    parser.add_argument(
        "--lancedb-final-index",
        action="store_true",
        help="Build/refresh LanceDB ANN index once after online ingestion.",
    )
    parser.add_argument("--workflow-cache-size", type=int, default=128)
    parser.add_argument("--memmap-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--memmap-chunk-size", type=int, default=8192)
    parser.add_argument("--compressed-faiss-kind", choices=["auto", "sq8", "ivfpq"], default="auto")
    parser.add_argument("--compressed-faiss-nlist", type=int, default=64)
    parser.add_argument("--compressed-faiss-pq-m", type=int, default=16)
    parser.add_argument("--compressed-faiss-pq-bits", type=int, default=8)
    parser.add_argument(
        "--ingest-mode",
        choices=["online", "batch"],
        default="online",
        help="online interleaves retrieve/add; batch ingests all workflows then benchmarks retrieval.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.child:
        return run_child(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
