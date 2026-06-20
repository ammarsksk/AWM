import argparse
import json
import re
import shutil
import sqlite3
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


ACTION_RE = re.compile(r"^\s*([a-zA-Z_]+)\((.*)\)\s*$")


def action_name(raw: str) -> str:
    match = ACTION_RE.match(raw)
    return match.group(1) if match else raw.strip().split("(", 1)[0]


def iter_result_dirs(results_dir: Path):
    for path in sorted(results_dir.glob("webarena.*"), key=lambda p: int(p.name.split(".")[1])):
        summary = path / "summary_info.json"
        log = path / "experiment.log"
        if summary.exists() and log.exists():
            yield path


def extract_actions(log_path: Path) -> list[str]:
    actions = []
    capture = False
    for line in log_path.read_text(errors="ignore").splitlines():
        stripped = line.strip()
        if stripped == "action:":
            capture = True
            continue
        if capture:
            if not stripped:
                capture = False
                continue
            if stripped.startswith("2026-"):
                capture = False
                continue
            if "(" in stripped and ")" in stripped:
                actions.append(stripped)
    return actions


def workflow_blocks(workflow_path: Path) -> int:
    if not workflow_path.exists():
        return 0
    return workflow_path.read_text(errors="ignore").count("\nQuery:")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--workflow-path", default="workflow/shopping.txt")
    parser.add_argument("--procedural-memory-dir", default="memory/procedural")
    parser.add_argument("--out-dot", default="memory_graph.dot")
    parser.add_argument("--out-svg", default="memory_graph.svg")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    workflow_path = Path(args.workflow_path)
    reward_counts = Counter()
    action_counts = Counter()
    edges = Counter()
    successful = []

    for result_dir in iter_result_dirs(results_dir):
        summary = json.loads((result_dir / "summary_info.json").read_text())
        reward = float(summary.get("cum_reward") or 0)
        errored = bool(summary.get("err_msg"))
        reward_counts[(reward > 0, errored)] += 1
        actions = extract_actions(result_dir / "experiment.log")
        for action in actions:
            action_counts[action_name(action)] += 1
        if reward > 0 and not errored:
            successful.append(result_dir.name)
            names = ["START"] + [action_name(a) for a in actions] + ["SUCCESS"]
            for src, dst in zip(names, names[1:]):
                edges[(src, dst)] += 1

    dot_lines = [
        "digraph WebArenaMemory {",
        "  rankdir=LR;",
        "  node [shape=box, style=rounded];",
    ]
    for (src, dst), count in sorted(edges.items()):
        dot_lines.append(f'  "{src}" -> "{dst}" [label="{count}"];')
    dot_lines.append("}")
    Path(args.out_dot).write_text("\n".join(dot_lines) + "\n")

    if shutil.which("dot"):
        subprocess.run(["dot", "-Tsvg", args.out_dot, "-o", args.out_svg], check=False)

    total = sum(reward_counts.values())
    wins = reward_counts[(True, False)]
    errors = sum(count for (_, errored), count in reward_counts.items() if errored)
    print(f"results={total}")
    print(f"successful={wins}")
    print(f"errors={errors}")
    print(f"success_rate={(wins / total * 100) if total else 0:.1f}%")
    print(f"workflow_examples={workflow_blocks(workflow_path)}")
    print(f"workflow_bytes={workflow_path.stat().st_size if workflow_path.exists() else 0}")
    proc_db = Path(args.procedural_memory_dir) / "procedural_memory.sqlite3"
    if proc_db.exists():
        conn = sqlite3.connect(proc_db)
        try:
            procedures = conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0]
            edges_count = conn.execute("SELECT COUNT(*) FROM procedure_edges").fetchone()[0]
            negatives = conn.execute("SELECT COUNT(*) FROM negative_memory").fetchone()[0]
            retrievals = conn.execute("SELECT COUNT(*) FROM retrieval_events").fetchone()[0]
            print(f"procedural_db={proc_db.resolve()}")
            print(f"procedures={procedures}")
            print(f"procedure_edges={edges_count}")
            print(f"negative_memories={negatives}")
            print(f"retrieval_events={retrievals}")
        finally:
            conn.close()
    print(f"top_actions={dict(action_counts.most_common(8))}")
    print(f"graph_dot={Path(args.out_dot).resolve()}")
    if Path(args.out_svg).exists():
        print(f"graph_svg={Path(args.out_svg).resolve()}")


if __name__ == "__main__":
    main()
