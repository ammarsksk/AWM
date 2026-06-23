from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "webarena"


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def safe_name(text: str) -> str:
    return text.strip().lower().replace("/", "_").replace("-", "_")


def run_one(args: argparse.Namespace, backend: str, out_dir: Path) -> int:
    cmd = [
        sys.executable,
        "webarena/run_procedural_eval_with_metrics.py",
        "--website",
        args.website,
        "--output-dir",
        str(out_dir.relative_to(WEB) if out_dir.is_relative_to(WEB) else out_dir),
        "--procedural-memory-dir",
        args.procedural_memory_dir,
        "--vector-backend",
        backend,
        "--embedding-model",
        args.embedding_model,
        "--reranker",
        args.reranker,
        "--reranker-model",
        args.reranker_model,
        "--reranker-train-limit",
        str(args.reranker_train_limit),
        "--model-name",
        args.model_name,
        "--procedural-abstraction-model",
        args.procedural_abstraction_model,
        "--procedural-top-k",
        str(args.procedural_top_k),
        "--procedural-min-score",
        str(args.procedural_min_score),
        "--max-steps",
        str(args.max_steps),
        "--llm-retries",
        str(args.llm_retries),
        "--pre-observation-delay",
        str(args.pre_observation_delay),
        "--extract-obs-retries",
        str(args.extract_obs_retries),
    ]
    if args.task_ids:
        cmd.extend(["--task-ids", args.task_ids])
    else:
        cmd.extend(["--start-index", str(args.start_index), "--num-tasks", str(args.num_tasks)])
    if args.browser_proxy:
        cmd.extend(["--browser-proxy", args.browser_proxy])
    if args.headless:
        cmd.append("--headless")
    if args.skip_run:
        cmd.append("--skip-run")
    log_path = out_dir / "sweep_invocation.log"
    out_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n\n")
        log.flush()
        process = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    return int(process.returncode)


def summarize_run(backend: str, out_dir: Path) -> dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    payload = read_json(metrics_path, {})
    metrics = payload.get("metrics", {})
    retrieval = payload.get("retrieval_delta", {})
    storage = payload.get("memory_storage", {})
    vector = payload.get("vector_index_manifest", {})
    experiment = payload.get("experiment", {})
    return {
        "backend": backend,
        "returncode": experiment.get("returncode"),
        "success_rate": metrics.get("success_rate"),
        "error_rate": metrics.get("error_rate"),
        "tasks": metrics.get("tasks"),
        "successful_tasks": metrics.get("successful_tasks"),
        "failed_tasks": metrics.get("failed_tasks"),
        "avg_steps_per_task": metrics.get("avg_steps_per_task"),
        "avg_agent_elapsed_sec_per_task": metrics.get("avg_agent_elapsed_sec_per_task"),
        "avg_step_elapsed_sec_per_task": metrics.get("avg_step_elapsed_sec_per_task"),
        "retrieval_events_delta": retrieval.get("retrieval_events_delta"),
        "retrieval_selection_rate": retrieval.get("retrieval_selection_rate"),
        "avg_retrieved_memories_per_event": retrieval.get("avg_retrieved_memories_per_event"),
        "memory_dir_bytes_after": storage.get("memory_dir_bytes_after"),
        "db_bytes_after": storage.get("db_bytes_after"),
        "actual_index_kind": vector.get("actual_index_kind"),
        "index_bytes": vector.get("index_bytes"),
        "embedding_model": experiment.get("embedding_model"),
        "reranker": experiment.get("reranker"),
    }


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Backend | Success | Errors | Avg steps | Avg agent sec | Retrieval events | Select rate | Memories/event | Index kind | Index bytes | Memory dir bytes |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {backend} | {success:.1f}% | {error:.1f}% | {steps:.2f} | {agent:.2f} | {events} | {select:.2f} | {mems:.2f} | {kind} | {index} | {memory} |".format(
                backend=row["backend"],
                success=float(row.get("success_rate") or 0.0),
                error=float(row.get("error_rate") or 0.0),
                steps=float(row.get("avg_steps_per_task") or 0.0),
                agent=float(row.get("avg_agent_elapsed_sec_per_task") or 0.0),
                events=row.get("retrieval_events_delta") or 0,
                select=float(row.get("retrieval_selection_rate") or 0.0),
                mems=float(row.get("avg_retrieved_memories_per_event") or 0.0),
                kind=row.get("actual_index_kind") or "",
                index=row.get("index_bytes") or 0,
                memory=row.get("memory_dir_bytes_after") or 0,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", default="shopping")
    parser.add_argument("--start-index", type=int, default=158)
    parser.add_argument("--num-tasks", type=int, default=50)
    parser.add_argument("--task-ids", default=None)
    parser.add_argument("--output-dir", default="metrics/procedural_backend_sweep_50_full")
    parser.add_argument("--procedural-memory-dir", default="memory/procedural")
    parser.add_argument("--backends", default="hnsw_sq8,binary_hnsw_rotation,turboquant,sq8,hnsw,flat,opq_ivfpq,ivfpq,rabitq")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--reranker", default="full", choices=["none", "feature", "ml", "cross_encoder", "full"])
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-large")
    parser.add_argument("--reranker-train-limit", type=int, default=1000)
    parser.add_argument("--model-name", default="openai/google/gemini-2.5-pro")
    parser.add_argument("--procedural-abstraction-model", default="openai/google/gemini-2.5-pro")
    parser.add_argument("--procedural-top-k", type=int, default=4)
    parser.add_argument("--procedural-min-score", type=float, default=0.42)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--llm-retries", type=int, default=6)
    parser.add_argument("--pre-observation-delay", type=float, default=1.25)
    parser.add_argument("--extract-obs-retries", type=int, default=8)
    parser.add_argument("--browser-proxy", default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    sweep_dir = WEB / args.output_dir
    sweep_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for backend in [item.strip() for item in args.backends.split(",") if item.strip()]:
        backend_dir = sweep_dir / safe_name(backend)
        print(f"running_backend:{backend}", flush=True)
        rc = run_one(args, backend, backend_dir)
        row = summarize_run(backend, backend_dir)
        row["returncode"] = rc
        rows.append(row)
        (sweep_dir / "sweep_partial.json").write_text(json.dumps(rows, indent=2, sort_keys=True))

    summary = {
        "website": args.website,
        "task_ids": args.task_ids,
        "start_index": args.start_index,
        "num_tasks": args.num_tasks,
        "embedding_model": args.embedding_model,
        "reranker": args.reranker,
        "reranker_model": args.reranker_model,
        "rows": rows,
    }
    (sweep_dir / "sweep_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (sweep_dir / "sweep_summary.md").write_text(markdown_table(rows))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
