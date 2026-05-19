"""Online Induction and Workflow Utilization Pipeline."""

import argparse
import subprocess
import sys
from pathlib import Path
from utils.data import load_json


ROOT = Path(__file__).resolve().parent


def run_step(cmd: list[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)

def offline():
    workflow_dir = ROOT / "workflow"
    workflow_dir.mkdir(exist_ok=True)

    # workflow induction
    run_step([
        sys.executable, 'offline_induction.py',
        '--mode', 'auto', '--website', args.website,
        '--domain', args.domain, '--subdomain', args.subdomain,
        '--model_name', args.model, '--output_dir', "workflow",
        '--instruction_path', args.instruction_path,
        '--one_shot_path', args.one_shot_path,
    ])

    # test inference
    run_step([
        sys.executable, 'run_mind2web.py',
        '--website', args.website,
        '--workflow_path', f"workflow/{args.website}.txt",
        '--model', args.model,
        '--benchmark', args.benchmark,
        '--domain', args.domain,
        '--subdomain', args.subdomain,
    ])


def online():
    workflow_path = Path(args.workflow_path)
    if not workflow_path.is_absolute():
        workflow_path = ROOT / workflow_path
    workflow_path.parent.mkdir(parents=True, exist_ok=True)
    workflow_path.touch(exist_ok=True)

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = str(Path("results") / args.model / args.benchmark / args.website / args.run_suffix)

    # load all examples for streaming
    samples = load_json(args.data_dir, args.benchmark)
    print(f"Loaded #{len(samples)} test examples")
    if args.website is not None:
        samples = [s for s in samples if s["website"] == args.website]
        print(f"Filtering down to #{len(samples)} examples on website [{args.website}]")
    n = len(samples)
    
    for i in range(0, n, args.induce_steps):
        j = min(n, i + args.induce_steps)
        print(f"Running inference on {i}-{j} th example..")

        inference_cmd = [
            sys.executable, 'run_mind2web.py',
            '--benchmark', args.benchmark,
            '--workflow_path', str(workflow_path.relative_to(ROOT)),
            '--website', args.website, 
            '--start_idx', f'{i}', '--end_idx', f'{j}',
            '--model', args.model,
            '--suffix', args.run_suffix,
        ]
        if args.domain is not None:
            inference_cmd.extend(['--domain', args.domain])
        if args.subdomain is not None:
            inference_cmd.extend(['--subdomain', args.subdomain])
        run_step(inference_cmd)
        print(f"Finished inference on {i}-{j} th example!\n")

        if j < len(samples):
            run_step([
                sys.executable, 'online_induction.py',
                '--benchmark', args.benchmark,
                '--website', args.website,
                '--results_dir', results_dir,
                '--output_path', str(workflow_path.relative_to(ROOT)),
                '--model_name', args.model,
                '--temperature', str(args.temperature),
                '--instruction_path', args.instruction_path,
                '--one_shot_path', args.one_shot_path,
                '--suffix', args.suffix,
            ])
            print(f"Finished workflow induction with examples up to {j - 1}!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # examples
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--benchmark", type=str, default="test_task",
        choices=["test_task", "test_website", "test_domain", "train"])
    parser.add_argument("--website", type=str, required=True)
    parser.add_argument("--domain", type=str, default=None)
    parser.add_argument("--subdomain", type=str, default=None)

    # results and workflows
    parser.add_argument("--results_dir", type=str, default=None)
    parser.add_argument("--workflow_path", type=str, default=None)
    parser.add_argument("--run_suffix", type=str, default="workflow")

    # prompt
    parser.add_argument("--instruction_path", type=str, default="prompt/instruction_action.txt")
    parser.add_argument("--one_shot_path", type=str, default="prompt/one_shot_action.txt")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="# Summary Workflows")

    # gpt
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)

    # induction frequency
    parser.add_argument("--induce_steps", type=int, default=1)

    # setup
    parser.add_argument("--setup", type=str, required=True,
                        choices=["online", "offline"])

    args = parser.parse_args()

    if args.setup == "online":
        assert args.workflow_path is not None
        online()
    elif args.setup == "offline":
        assert (args.domain is not None) and (args.subdomain is not None)
        offline()
