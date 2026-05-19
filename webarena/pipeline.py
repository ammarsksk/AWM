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

    task_ids = load_task_ids(args.website)
    if args.end_index is None:
        args.end_index = len(task_ids)

    for tid in task_ids[args.start_index: args.end_index]:
        run_step([
            sys.executable, "run.py",
            "--task_name", f"webarena.{tid}",
            "--workflow_path", str(workflow_path.relative_to(ROOT)),
            "--model_name", args.model_name,
            "--headless", str(args.headless),
        ])

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
    parser.add_argument("--model_name", type=str, default="openai/gpt-4o")
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
        default="gpt-4o",
    )
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()

    main()
