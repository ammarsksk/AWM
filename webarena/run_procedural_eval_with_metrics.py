from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from aggregate_webarena_metrics import aggregate, load_results, write_markdown
from pipeline import load_task_ids


ROOT = Path(__file__).resolve().parent


def directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def db_scalar(db_path: Path, query: str, default: int = 0) -> int:
    if not db_path.exists():
        return default
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(query).fetchone()
        return int(row[0]) if row and row[0] is not None else default
    finally:
        conn.close()


def retrieval_delta_stats(db_path: Path, before_id: int) -> dict[str, Any]:
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT selected, candidates_json, raw_candidates_json, rejected_candidates_json
            FROM retrieval_events
            WHERE id > ?
            ORDER BY id
            """,
            (before_id,),
        ).fetchall()
    finally:
        conn.close()

    def json_len(text: str) -> int:
        try:
            value = json.loads(text or "[]")
            return len(value) if isinstance(value, list) else 0
        except Exception:
            return 0

    selected = sum(1 for row in rows if row["selected"])
    candidate_counts = [json_len(row["candidates_json"]) for row in rows]
    raw_counts = [json_len(row["raw_candidates_json"]) for row in rows]
    rejected_counts = [json_len(row["rejected_candidates_json"]) for row in rows]
    return {
        "retrieval_events_delta": len(rows),
        "retrieval_selected_delta": selected,
        "retrieval_selection_rate": round(selected / len(rows) * 100, 2) if rows else 0.0,
        "avg_retrieved_memories_per_event": round(sum(candidate_counts) / len(candidate_counts), 3)
        if candidate_counts
        else 0.0,
        "avg_raw_candidates_per_event": round(sum(raw_counts) / len(raw_counts), 3)
        if raw_counts
        else 0.0,
        "avg_rejected_candidates_per_event": round(sum(rejected_counts) / len(rejected_counts), 3)
        if rejected_counts
        else 0.0,
    }


def task_subset_rows(website: str, task_ids: list[int], fresh_after: float | None = None) -> list[dict[str, Any]]:
    wanted = set(task_ids)
    rows = load_results(ROOT / "results", website)
    if fresh_after is None:
        return [row for row in rows if row["task_id"] in wanted]
    fresh = []
    for row in rows:
        if row["task_id"] not in wanted:
            continue
        summary_path = ROOT / "results" / f"webarena.{row['task_id']}" / "summary_info.json"
        if summary_path.exists() and summary_path.stat().st_mtime >= fresh_after:
            fresh.append(row)
    return fresh


def run_pipeline(args: argparse.Namespace, task_start: int, task_end: int, log_path: Path) -> int:
    cmd = [
        sys.executable,
        "pipeline.py",
        "--website",
        args.website,
        "--start_index",
        str(task_start),
        "--end_index",
        str(task_end),
        "--model_name",
        args.model_name,
        "--memory_architecture",
        "procedural",
        "--procedural_memory_dir",
        args.procedural_memory_dir,
        "--procedural_top_k",
        str(args.procedural_top_k),
        "--procedural_min_score",
        str(args.procedural_min_score),
        "--procedural_abstraction_model",
        args.procedural_abstraction_model,
        "--max_steps",
        str(args.max_steps),
        "--llm_retries",
        str(args.llm_retries),
        "--pre_observation_delay",
        str(args.pre_observation_delay),
        "--extract_obs_retries",
        str(args.extract_obs_retries),
    ]
    if args.headless:
        cmd.append("--headless")
    if args.browser_proxy:
        cmd.extend(["--browser_proxy", args.browser_proxy])

    env = os.environ.copy()
    env["WEBARENA_FAST_MEMORY"] = "0"
    env["WEBARENA_VECTOR_BACKEND"] = args.vector_backend
    env["AWM_EMBEDDING_MODEL"] = args.embedding_model

    with log_path.open("w") as log:
        log.write("COMMAND: " + " ".join(cmd) + "\n")
        log.write(
            "ENV: "
            + json.dumps(
                {
                    "WEBARENA_FAST_MEMORY": env["WEBARENA_FAST_MEMORY"],
                    "WEBARENA_VECTOR_BACKEND": env["WEBARENA_VECTOR_BACKEND"],
                    "AWM_EMBEDDING_MODEL": env["AWM_EMBEDDING_MODEL"],
                },
                sort_keys=True,
            )
            + "\n\n"
        )
        log.flush()
        process = subprocess.run(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return int(process.returncode)


def run_direct_task_ids(args: argparse.Namespace, task_ids: list[int], log_path: Path) -> int:
    env = os.environ.copy()
    env["WEBARENA_FAST_MEMORY"] = "0"
    env["WEBARENA_VECTOR_BACKEND"] = args.vector_backend
    env["AWM_EMBEDDING_MODEL"] = args.embedding_model
    memory_path = str(ROOT / args.procedural_memory_dir)

    with log_path.open("w") as log:
        log.write(
            "ENV: "
            + json.dumps(
                {
                    "WEBARENA_FAST_MEMORY": env["WEBARENA_FAST_MEMORY"],
                    "WEBARENA_VECTOR_BACKEND": env["WEBARENA_VECTOR_BACKEND"],
                    "AWM_EMBEDDING_MODEL": env["AWM_EMBEDDING_MODEL"],
                },
                sort_keys=True,
            )
            + "\n\n"
        )
        for tid in task_ids:
            run_cmd = [
                sys.executable,
                "run.py",
                "--task_name",
                f"webarena.{tid}",
                "--workflow_path",
                f"workflow/{args.website}.txt",
                "--model_name",
                args.model_name,
                "--headless",
                str(args.headless),
                "--max_steps",
                str(args.max_steps),
                "--llm_retries",
                str(args.llm_retries),
                "--pre_observation_delay",
                str(args.pre_observation_delay),
                "--extract_obs_retries",
                str(args.extract_obs_retries),
                "--procedural_memory_path",
                memory_path,
                "--procedural_site",
                args.website,
                "--procedural_top_k",
                str(args.procedural_top_k),
                "--procedural_min_score",
                str(args.procedural_min_score),
            ]
            if args.browser_proxy:
                run_cmd.extend(["--browser_proxy", args.browser_proxy])
            log.write("COMMAND: " + " ".join(run_cmd) + "\n")
            log.flush()
            process = subprocess.run(
                run_cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if process.returncode != 0:
                return int(process.returncode)

            ingest_cmd = [
                sys.executable,
                "procedural_memory.py",
                "ingest-result",
                "--memory-dir",
                memory_path,
                "--result-dir",
                f"results/webarena.{tid}",
                "--config-dir",
                "config_files",
                "--abstraction-model",
                args.procedural_abstraction_model,
            ]
            log.write("COMMAND: " + " ".join(ingest_cmd) + "\n")
            log.flush()
            process = subprocess.run(
                ingest_cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
            if process.returncode != 0:
                return int(process.returncode)
    return 0


def main() -> None:
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("--website", default="shopping")
    parser.add_argument("--start-index", type=int, default=158)
    parser.add_argument("--num-tasks", type=int, default=10)
    parser.add_argument(
        "--task-ids",
        default=None,
        help="Comma-separated literal WebArena task ids. If set, overrides --start-index/--num-tasks.",
    )
    parser.add_argument("--output-dir", default="metrics/procedural_eval_10_hnsw_sq8")
    parser.add_argument("--procedural-memory-dir", default="memory/procedural")
    parser.add_argument("--vector-backend", default="hnsw_sq8")
    parser.add_argument("--embedding-model", default="BAAI/bge-small-en-v1.5")
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

    if args.task_ids:
        task_ids = [int(item.strip()) for item in args.task_ids.split(",") if item.strip()]
        task_end = args.start_index + len(task_ids)
    else:
        task_ids_all = load_task_ids(args.website)
        task_end = min(args.start_index + args.num_tasks, len(task_ids_all))
        task_ids = task_ids_all[args.start_index:task_end]
    out_dir = ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    memory_dir = ROOT / args.procedural_memory_dir
    db_path = memory_dir / "procedural_memory.sqlite3"
    before_event_id = db_scalar(db_path, "SELECT COALESCE(MAX(id), 0) FROM retrieval_events")
    before_db_bytes = db_path.stat().st_size if db_path.exists() else 0
    before_memory_dir_bytes = directory_bytes(memory_dir)
    run_log = out_dir / "run.log"

    fresh_after = time.time() - 1.0
    start = time.perf_counter()
    returncode = 0
    if not args.skip_run:
        if args.task_ids:
            returncode = run_direct_task_ids(args, task_ids, run_log)
        else:
            returncode = run_pipeline(args, args.start_index, task_end, run_log)
    wall_sec = time.perf_counter() - start

    rows = task_subset_rows(args.website, task_ids, None if args.skip_run else fresh_after)
    metrics = aggregate(rows, memory_dir)
    after_event_id = db_scalar(db_path, "SELECT COALESCE(MAX(id), 0) FROM retrieval_events")
    after_db_bytes = db_path.stat().st_size if db_path.exists() else 0
    after_memory_dir_bytes = directory_bytes(memory_dir)
    manifest = read_json(memory_dir / "procedural_manifest.json", {})
    vector_manifest = read_json(memory_dir / "advanced_vector_index" / "advanced_manifest.json", {})
    retrieval_stats = retrieval_delta_stats(db_path, before_event_id)
    payload = {
        "experiment": {
            "website": args.website,
            "task_start_index": args.start_index,
            "task_end_index": task_end,
            "task_ids": task_ids,
            "num_tasks_requested": args.num_tasks,
            "returncode": returncode,
            "wall_clock_sec": round(wall_sec, 3),
            "vector_backend": args.vector_backend,
            "embedding_model": args.embedding_model,
            "model_name": args.model_name,
            "skip_run": args.skip_run,
        },
        "metrics": metrics,
        "retrieval_delta": {
            "before_event_id": before_event_id,
            "after_event_id": after_event_id,
            **retrieval_stats,
        },
        "memory_storage": {
            "db_bytes_before": before_db_bytes,
            "db_bytes_after": after_db_bytes,
            "db_bytes_delta": after_db_bytes - before_db_bytes,
            "memory_dir_bytes_before": before_memory_dir_bytes,
            "memory_dir_bytes_after": after_memory_dir_bytes,
            "memory_dir_bytes_delta": after_memory_dir_bytes - before_memory_dir_bytes,
        },
        "procedural_manifest": manifest,
        "vector_index_manifest": vector_manifest,
        "tasks": rows,
    }
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True))
    write_markdown(metrics, rows, out_dir / "metrics.md")
    (out_dir / "experiment_config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    print(json.dumps(payload["experiment"], indent=2, sort_keys=True))
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))
    print(json.dumps(payload["retrieval_delta"], indent=2, sort_keys=True))
    print(json.dumps(payload["memory_storage"], indent=2, sort_keys=True))
    print(f"wrote={out_dir / 'metrics.json'}")
    print(f"wrote={out_dir / 'metrics.md'}")
    print(f"wrote={run_log}")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
