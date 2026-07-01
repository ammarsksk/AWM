import argparse
import json
import shutil
import sys
import time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webarena"))

from advanced_vector_index import AdvancedIndexConfig, AdvancedProcedureVectorIndex


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Advanced Vector Index with Synthetic Data")
    parser.add_argument("--num-procedures", type=int, default=5000, help="Number of procedures/vectors to insert")
    parser.add_argument("--num-queries", type=int, default=100, help="Number of queries to perform")
    parser.add_argument("--output-dir", default="webarena/retrieval_synthetic_benchmark_5000", help="Directory to save indices and summaries")
    parser.add_argument(
        "--backends",
        default="flat,hnsw,sq8,hnsw_sq8,ivfpq,opq_ivfpq,turboquant,rabitq,binary_hnsw_rotation",
        help="Comma-separated vector index backends to sweep",
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--hnsw-ef-construction", type=int, default=80)
    parser.add_argument("--ivf-nlist", type=int, default=64)
    parser.add_argument("--ivf-nprobe", type=int, default=8)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--pq-bits", type=int, default=8)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {args.num_procedures} synthetic database vectors and {args.num_queries} query vectors (dimension: 384)...")
    np.random.seed(42)
    db_vectors = np.random.normal(size=(args.num_procedures, 384)).astype("float32")
    db_vectors /= np.linalg.norm(db_vectors, axis=1, keepdims=True)

    query_vectors = np.random.normal(size=(args.num_queries, 384)).astype("float32")
    query_vectors /= np.linalg.norm(query_vectors, axis=1, keepdims=True)

    # Define a Mock Embedder that returns pre-generated vectors
    class MockEmbedder:
        model = "mock_model"
        model_name = "mock_model_name"
        model_source = "mock_model_source"
        backend = "mock_backend"

        def embed(self, text: str) -> list[float]:
            if text.startswith("query_"):
                idx = int(text.split("_")[1])
                return query_vectors[idx].tolist()
            elif text.startswith("proc_"):
                idx = int(text.split("_")[1])
                return db_vectors[idx].tolist()
            else:
                return np.zeros(384).tolist()

    rows = [{"name": f"proc_{i}", "compact_text": f"proc_{i}"} for i in range(args.num_procedures)]
    backends_list = [b.strip() for b in args.backends.split(",") if b.strip()]

    summaries = []
    for backend in backends_list:
        print(f"\n==========================================")
        print(f"Benchmarking Backend: {backend}")
        print(f"==========================================")

        backend_dir = out_dir / backend
        if backend_dir.exists():
            shutil.rmtree(backend_dir)
        backend_dir.mkdir(parents=True, exist_ok=True)

        before_rss = current_rss_mb()
        
        # Instantiate index and inject mock embedder
        index = AdvancedProcedureVectorIndex(
            backend_dir,
            AdvancedIndexConfig(
                index_kind=backend,
                hnsw_m=args.hnsw_m,
                hnsw_ef_search=args.hnsw_ef_search,
                hnsw_ef_construction=args.hnsw_ef_construction,
                ivf_nlist=args.ivf_nlist,
                ivf_nprobe=args.ivf_nprobe,
                pq_m=args.pq_m,
                pq_bits=args.pq_bits,
            ),
        )
        index.embedder = MockEmbedder()

        # Build index
        build_start = time.perf_counter()
        index.rebuild_from_rows(rows)
        build_time_ms = (time.perf_counter() - build_start) * 1000.0
        after_build_rss = current_rss_mb()

        # Retrieve queries
        retrieval_times_ms = []
        for q_idx in range(args.num_queries):
            q_start = time.perf_counter()
            # Disable lexical/freshness re-ranking components to benchmark pure vector search
            # (By modifying config weights to only use semantic vector search)
            index.config.semantic_weight = 1.0
            index.config.lexical_weight = 0.0
            index.config.freshness_weight = 0.0
            
            _ = index.retrieve(f"query_{q_idx}", top_k=4)
            q_elapsed = (time.perf_counter() - q_start) * 1000.0
            retrieval_times_ms.append(q_elapsed)

        stats = index.stats()
        avg_ret_ms = np.mean(retrieval_times_ms)
        p95_ret_ms = np.percentile(retrieval_times_ms, 95)
        
        idx_size = stats.get("index_bytes", 0)
        disk_size = directory_bytes(backend_dir)
        rss_delta = after_build_rss - before_rss

        print(f"Build Time: {build_time_ms:.2f} ms")
        print(f"Avg Retrieval Time: {avg_ret_ms:.4f} ms (p95: {p95_ret_ms:.4f} ms)")
        print(f"Index Storage Size: {idx_size:,} bytes")
        print(f"Total Disk Storage: {disk_size:,} bytes")
        print(f"RAM RSS Delta: {rss_delta:.2f} MB")
        print(f"Actual Index Configured: {stats.get('actual_index_kind')}")

        summaries.append({
            "backend": backend,
            "actual_index_kind": stats.get("actual_index_kind"),
            "avg_retrieval_ms": float(avg_ret_ms),
            "p95_retrieval_ms": float(p95_ret_ms),
            "build_ms": float(build_time_ms),
            "index_bytes": int(idx_size),
            "disk_bytes": int(disk_size),
            "rss_mb_delta": float(rss_delta),
        })

    # Save summary report
    report_path = out_dir / "synthetic_sweep_summary.json"
    report_path.write_text(json.dumps({
        "num_procedures": args.num_procedures,
        "num_queries": args.num_queries,
        "summaries": summaries
    }, indent=2))

    # Generate Markdown Table
    md_lines = [
        f"# Synthetic Vector Index Sweep Report",
        f"Database Size: {args.num_procedures} procedures, Queries: {args.num_queries}",
        "",
        "| Backend | Actual Index Kind | Avg Retrieval (ms) | p95 Retrieval (ms) | Build Time (ms) | Index Size | Disk Size | RAM Delta (MB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        md_lines.append(
            f"| {s['backend']} | {s['actual_index_kind']} | {s['avg_retrieval_ms']:.3f} | {s['p95_retrieval_ms']:.3f} | {s['build_ms']:.1f} | {s['index_bytes']:,} B | {s['disk_bytes']:,} B | {s['rss_mb_delta']:.2f} |"
        )
    
    (out_dir / "synthetic_sweep_summary.md").write_text("\n".join(md_lines) + "\n")
    print(f"\nSweep completed. Summary markdown table saved to {out_dir / 'synthetic_sweep_summary.md'}")


if __name__ == "__main__":
    main()
