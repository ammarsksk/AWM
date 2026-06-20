import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_task_ids(website: str) -> list[int]:
    """Return WebArena task ids for a website split."""
    numbered_configs = [
        path for path in (ROOT / "config_files").glob("*.json")
        if path.stem.isdigit()
    ]

    if numbered_configs:
        config_files = sorted(numbered_configs, key=lambda path: int(path.stem))
        configs = [json.loads(path.read_text()) for path in config_files]
    else:
        test_path = ROOT / "config_files" / "test.json"
        if not test_path.exists():
            raise FileNotFoundError(
                "No numbered WebArena config files found. Run "
                "`cd webarena/config_files && python generate_test_data.py` first."
            )
        configs = json.loads(test_path.read_text())

    return [
        config["task_id"]
        for config in configs
        if config.get("sites", [None])[0] == website
    ]


def run_step(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def main():
    workflow_dir = ROOT / "workflow"
    workflow_dir.mkdir(exist_ok=True)
    workflow_path = workflow_dir / f"{args.website}.txt"
    workflow_path.touch(exist_ok=True)
    procedural_memory_path = ROOT / args.procedural_memory_dir

    task_ids = load_task_ids(args.website)
    if args.end_index is None:
        args.end_index = len(task_ids)

    for tid in task_ids[args.start_index: args.end_index]:
        run_cmd = [
            sys.executable, "run.py",
            "--task_name", f"webarena.{tid}",
            "--workflow_path", str(workflow_path.relative_to(ROOT)),
            "--model_name", args.model_name,
            "--headless", str(args.headless),
            "--max_steps", str(args.max_steps),
            "--llm_retries", str(args.llm_retries),
            "--pre_observation_delay", str(args.pre_observation_delay),
            "--extract_obs_retries", str(args.extract_obs_retries),
        ]
        if args.memory_architecture == "procedural":
            run_cmd.extend(
                [
                    "--procedural_memory_path",
                    str(procedural_memory_path),
                    "--procedural_site",
                    args.website,
                    "--procedural_top_k",
                    str(args.procedural_top_k),
                    "--procedural_min_score",
                    str(args.procedural_min_score),
                ]
            )
        run_step(run_cmd + (["--browser_proxy", args.browser_proxy] if args.browser_proxy else []))

        if args.memory_architecture == "procedural":
            run_step(
                [
                    sys.executable,
                    "procedural_memory.py",
                    "ingest-result",
                    "--memory-dir",
                    str(procedural_memory_path),
                    "--result-dir",
                    f"results/webarena.{tid}",
                    "--config-dir",
                    "config_files",
                    "--abstraction-model",
                    args.procedural_abstraction_model,
                ]
            )
            continue

        if args.criteria == "autoeval":
            run_step([
                sys.executable, "-m", "autoeval.evaluate_trajectory",
                "--result_dir", f"results/webarena.{tid}",
                "--model", args.eval_model,
            ])

        induction_script = "induce_prompt.py" if args.induction == "prompt" else "induce_rule.py"
        induction_cmd = [
            sys.executable, induction_script,
            "--result_dir", "results",
            "--output_path", str(workflow_path.relative_to(ROOT)),
            "--criteria", args.criteria,
        ]
        if args.induction == "prompt":
            induction_cmd.extend(["--model", args.eval_model])
        else:
            induction_cmd.append("--auto")
        run_step(induction_cmd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--website",
        type=str,
        required=True,
        choices=["shopping", "shopping_admin", "gitlab", "reddit", "map"],
    )
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--end_index", type=int, default=None)
    parser.add_argument("--model_name", type=str, default="openai/google/gemini-2.5-pro")
    parser.add_argument("--max_steps", type=int, default=30)
    parser.add_argument("--llm_retries", type=int, default=6)
    parser.add_argument("--pre_observation_delay", type=float, default=1.25)
    parser.add_argument("--extract_obs_retries", type=int, default=8)
    parser.add_argument(
        "--criteria",
        type=str,
        default="autoeval",
        choices=["autoeval", "gt"],
        help="'gt' uses WebArena environment reward and avoids LLM auto-eval.",
    )
    parser.add_argument(
        "--induction",
        type=str,
        default="prompt",
        choices=["prompt", "rule"],
        help="'rule' avoids LLM workflow induction.",
    )
    parser.add_argument(
        "--eval_model",
        type=str,
        default="google/gemini-2.5-pro",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--browser_proxy", type=str, default=None)
    parser.add_argument(
        "--memory_architecture",
        choices=["awm", "procedural"],
        default="awm",
        help="awm uses workflow text files; procedural uses SQLite/graph hybrid procedural memory.",
    )
    parser.add_argument("--procedural_memory_dir", default="memory/procedural")
    parser.add_argument("--procedural_top_k", type=int, default=4)
    parser.add_argument("--procedural_min_score", type=float, default=0.42)
    parser.add_argument("--procedural_abstraction_model", default="openai/google/gemini-2.5-pro")
    args = parser.parse_args()

    main()
