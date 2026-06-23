from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webarena"))

from advanced_vector_index import AdvancedIndexConfig, AdvancedProcedureVectorIndex  # noqa: E402
from procedural_reranker import ProceduralReranker  # noqa: E402


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


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def load_rows(db_path: Path) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            "SELECT name, compact_text FROM procedures ORDER BY created_at_ms, name"
        ).fetchall()
    finally:
        conn.close()


def load_queries(db_path: Path, max_queries: int) -> list[dict[str, Any]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT goal, selected, raw_candidates_json
            FROM retrieval_events
            ORDER BY id DESC
            LIMIT ?
            """,
            (max_queries,),
        ).fetchall()
    finally:
        conn.close()

    queries = []
    for row in rows:
        selected = row["selected"]
        if not selected:
            try:
                raw = json.loads(row["raw_candidates_json"] or "[]")
                selected = raw[0]["name"] if raw else None
            except Exception:
                selected = None
        queries.append({"goal": row["goal"], "selected": selected})
    return list(reversed(queries))


def run_backend(
    args: argparse.Namespace,
    backend: str,
    rows: list[sqlite3.Row],
    queries: list[dict[str, Any]],
    reranker: ProceduralReranker | None,
) -> dict[str, Any]:
    output_dir = args.output_dir / backend
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    before_rss = current_rss_mb()
    index = AdvancedProcedureVectorIndex(
        output_dir,
        AdvancedIndexConfig(
            index_kind=backend,
            candidate_k=args.candidate_k,
            hnsw_m=args.hnsw_m,
            hnsw_ef_search=args.hnsw_ef_search,
            hnsw_ef_construction=args.hnsw_ef_construction,
            ivf_nlist=args.ivf_nlist,
            ivf_nprobe=args.ivf_nprobe,
            pq_m=args.pq_m,
            pq_bits=args.pq_bits,
            rotation_seed=args.rotation_seed,
        ),
    )
    build_start = time.perf_counter()
    index.rebuild_from_rows(rows)
    build_ms = (time.perf_counter() - build_start) * 1000.0
    after_build_rss = current_rss_mb()

    retrieval_ms = []
    stage1_ms = []
    rerank_ms = []
    top1_matches = 0
    topk_matches = 0
    events = []
    for query in queries:
        start = time.perf_counter()
        stage1_start = time.perf_counter()
        result = index.retrieve(query["goal"], top_k=max(args.top_k, args.rerank_pool))
        stage1_elapsed = (time.perf_counter() - stage1_start) * 1000.0
        rerank_start = time.perf_counter()
        candidates = result.candidates
        if reranker is not None:
            candidates = reranker.rerank(query["goal"], candidates, top_k=args.top_k)
        else:
            candidates = candidates[: args.top_k]
        rerank_elapsed = (time.perf_counter() - rerank_start) * 1000.0
        elapsed = (time.perf_counter() - start) * 1000.0
        retrieval_ms.append(elapsed)
        stage1_ms.append(stage1_elapsed)
        rerank_ms.append(rerank_elapsed)
        names = [candidate["workflow_name"] for candidate in candidates]
        selected = query.get("selected")
        top1_matches += int(bool(selected and names and names[0] == selected))
        topk_matches += int(bool(selected and selected in names))
        events.append(
            {
                "goal": query["goal"],
                "selected_reference": selected,
                "retrieved": names,
                "retrieval_ms": elapsed,
                "stage1_ms": stage1_elapsed,
                "rerank_ms": rerank_elapsed,
                "top_candidates": candidates[: args.top_k],
            }
        )

    stats = index.stats()
    summary = {
        "backend": backend,
        "actual_index_kind": stats.get("actual_index_kind"),
        "embedding_model": stats.get("embedding_model"),
        "embedding_backend": stats.get("embedding_backend"),
        "backend_error": stats.get("backend_error"),
        "procedures": len(rows),
        "queries": len(queries),
        "build_ms": build_ms,
        "avg_retrieval_ms": mean(retrieval_ms) if retrieval_ms else 0.0,
        "avg_stage1_ms": mean(stage1_ms) if stage1_ms else 0.0,
        "avg_rerank_ms": mean(rerank_ms) if rerank_ms else 0.0,
        "p50_retrieval_ms": percentile(retrieval_ms, 50),
        "p95_retrieval_ms": percentile(retrieval_ms, 95),
        "p99_retrieval_ms": percentile(retrieval_ms, 99),
        "top1_reference_match_rate": top1_matches / len(queries) if queries else 0.0,
        "topk_reference_match_rate": topk_matches / len(queries) if queries else 0.0,
        "rss_mb_before": before_rss,
        "rss_mb_after_build": after_build_rss,
        "rss_mb_delta": after_build_rss - before_rss,
        "disk_bytes": directory_bytes(output_dir),
        "reranker": reranker.stats_dict() if reranker is not None else {"mode": "none"},
        "index_stats": stats,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    (output_dir / "events.json").write_text(json.dumps(events, indent=2))
    index.save()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory-dir", default="webarena/memory/procedural")
    parser.add_argument("--output-dir", default="webarena/retrieval_backend_benchmark")
    parser.add_argument(
        "--backends",
        default="flat,hnsw,sq8,hnsw_sq8,ivfpq,opq_ivfpq,binary_rotation,binary_hnsw_rotation",
    )
    parser.add_argument("--max-queries", type=int, default=250)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--candidate-k", type=int, default=40)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--hnsw-ef-construction", type=int, default=80)
    parser.add_argument("--ivf-nlist", type=int, default=64)
    parser.add_argument("--ivf-nprobe", type=int, default=8)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--pq-bits", type=int, default=8)
    parser.add_argument("--rotation-seed", type=int, default=1337)
    parser.add_argument("--reranker", default="none", choices=["none", "feature", "ml", "cross_encoder", "full"])
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-large")
    parser.add_argument("--reranker-train-limit", type=int, default=1000)
    parser.add_argument("--rerank-pool", type=int, default=16)
    args = parser.parse_args()

    args.memory_dir = Path(args.memory_dir)
    args.output_dir = Path(args.output_dir)
    db_path = args.memory_dir / "procedural_memory.sqlite3"
    rows = load_rows(db_path)
    queries = load_queries(db_path, args.max_queries)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reranker = None
    if args.reranker != "none":
        reranker = ProceduralReranker(
            db_path=db_path,
            mode=args.reranker,
            cross_encoder_model=args.reranker_model,
            train_limit=args.reranker_train_limit,
        )

    summaries = []
    for backend in [item.strip() for item in args.backends.split(",") if item.strip()]:
        print(f"benchmarking:{backend}", flush=True)
        summaries.append(run_backend(args, backend, rows, queries, reranker))

    result = {
        "memory_dir": str(args.memory_dir),
        "procedures": len(rows),
        "queries": len(queries),
        "reranker": reranker.stats_dict() if reranker is not None else {"mode": "none"},
        "summaries": summaries,
    }
    (args.output_dir / "benchmark_summary.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
