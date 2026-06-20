import argparse
import json
import sqlite3
from pathlib import Path


def load_results(results_dir: Path, website: str | None = None) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("webarena.*"), key=lambda p: int(p.name.split(".")[1])):
        summary_path = path / "summary_info.json"
        if not summary_path.exists():
            continue
        task_id = int(path.name.split(".")[1])
        config_path = Path("config_files") / f"{task_id}.json"
        if not config_path.exists():
            continue
        config = json.loads(config_path.read_text())
        site = config.get("sites", ["unknown"])[0]
        if website and site != website:
            continue
        summary = json.loads(summary_path.read_text())
        rows.append(
            {
                "task_id": task_id,
                "site": site,
                "intent_template_id": config.get("intent_template_id"),
                "reward": float(summary.get("cum_reward") or 0),
                "raw_reward": float(summary.get("cum_raw_reward") or 0),
                "n_steps": int(summary.get("n_steps") or 0),
                "errored": bool(summary.get("err_msg")),
                "terminated": bool(summary.get("terminated")),
                "truncated": bool(summary.get("truncated")),
                "agent_elapsed": float(summary.get("stats.cum_agent_elapsed") or 0),
                "step_elapsed": float(summary.get("stats.cum_step_elapsed") or 0),
                "agent_tokens": int(summary.get("stats.cum_n_token_agent_messages") or 0),
                "dom_tokens": int(summary.get("stats.cum_n_token_dom_txt") or 0),
                "axtree_tokens": int(summary.get("stats.cum_n_token_axtree_txt") or 0),
                "pruned_html_tokens": int(summary.get("stats.cum_n_token_pruned_html") or 0),
            }
        )
    return rows


def pct(value: float) -> float:
    return round(value * 100, 2)


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def procedural_stats(memory_dir: Path) -> dict:
    db_path = memory_dir / "procedural_memory.sqlite3"
    if not db_path.exists():
        return {}
    conn = sqlite3.connect(db_path)
    try:
        return {
            "procedures": conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0],
            "procedure_edges": conn.execute("SELECT COUNT(*) FROM procedure_edges").fetchone()[0],
            "negative_memories": conn.execute("SELECT COUNT(*) FROM negative_memory").fetchone()[0],
            "retrieval_events": conn.execute("SELECT COUNT(*) FROM retrieval_events").fetchone()[0],
            "memory_db_bytes": db_path.stat().st_size,
        }
    finally:
        conn.close()


def aggregate(rows: list[dict], memory_dir: Path | None = None) -> dict:
    total = len(rows)
    successful = sum(1 for row in rows if row["reward"] > 0 and not row["errored"])
    errors = sum(1 for row in rows if row["errored"])
    truncated = sum(1 for row in rows if row["truncated"])
    total_steps = sum(row["n_steps"] for row in rows)
    total_agent_elapsed = sum(row["agent_elapsed"] for row in rows)
    total_step_elapsed = sum(row["step_elapsed"] for row in rows)
    total_agent_tokens = sum(row["agent_tokens"] for row in rows)
    metrics = {
        "tasks": total,
        "successful_tasks": successful,
        "failed_tasks": total - successful,
        "errored_tasks": errors,
        "truncated_tasks": truncated,
        "success_rate": pct(successful / total) if total else 0.0,
        "error_rate": pct(errors / total) if total else 0.0,
        "total_steps": total_steps,
        "avg_steps_per_task": round(mean([row["n_steps"] for row in rows]), 3),
        "total_agent_elapsed_sec": round(total_agent_elapsed, 3),
        "total_step_elapsed_sec": round(total_step_elapsed, 3),
        "avg_agent_elapsed_sec_per_task": round(mean([row["agent_elapsed"] for row in rows]), 3),
        "avg_step_elapsed_sec_per_task": round(mean([row["step_elapsed"] for row in rows]), 3),
        "total_agent_message_tokens": total_agent_tokens,
        "avg_agent_message_tokens_per_task": round(
            mean([row["agent_tokens"] for row in rows]), 1
        ),
        "avg_dom_tokens_per_task": round(mean([row["dom_tokens"] for row in rows]), 1),
        "avg_axtree_tokens_per_task": round(mean([row["axtree_tokens"] for row in rows]), 1),
        "avg_pruned_html_tokens_per_task": round(
            mean([row["pruned_html_tokens"] for row in rows]), 1
        ),
    }
    if memory_dir is not None:
        metrics.update(procedural_stats(memory_dir))
    return metrics


def write_markdown(metrics: dict, rows: list[dict], path: Path) -> None:
    lines = [
        "# WebArena Metrics",
        "",
        f"- Tasks: {metrics['tasks']}",
        f"- Successful tasks: {metrics['successful_tasks']}",
        f"- Success rate: {metrics['success_rate']}%",
        f"- Errored tasks: {metrics['errored_tasks']}",
        f"- Total steps: {metrics['total_steps']}",
        f"- Avg steps/task: {metrics['avg_steps_per_task']}",
        f"- Total agent elapsed: {metrics['total_agent_elapsed_sec']} sec",
        f"- Avg agent elapsed/task: {metrics['avg_agent_elapsed_sec_per_task']} sec",
        f"- Avg agent-message tokens/task: {metrics['avg_agent_message_tokens_per_task']}",
        "",
        "| Task | Reward | Steps | Error | Agent sec | Agent tokens |",
        "|---:|---:|---:|:---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['task_id']} | {row['reward']} | {row['n_steps']} | "
            f"{'yes' if row['errored'] else 'no'} | {row['agent_elapsed']:.2f} | "
            f"{row['agent_tokens']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--website", default="shopping")
    parser.add_argument("--out-dir", default="metrics")
    parser.add_argument("--procedural-memory-dir", default="memory/procedural")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_results(Path(args.results_dir), args.website)
    metrics = aggregate(rows, Path(args.procedural_memory_dir))
    payload = {"metrics": metrics, "tasks": rows}
    (out_dir / f"{args.website}_metrics.json").write_text(json.dumps(payload, indent=2))
    write_markdown(metrics, rows, out_dir / f"{args.website}_metrics.md")
    print(json.dumps(metrics, indent=2))
    print(f"wrote={out_dir / f'{args.website}_metrics.json'}")
    print(f"wrote={out_dir / f'{args.website}_metrics.md'}")


if __name__ == "__main__":
    main()
