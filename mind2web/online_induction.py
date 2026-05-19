"""Induce Workflows from Past Agent Experiences."""

import os
import json
import argparse
from utils.data import load_json, format_examples, filter_workflows
from utils.provider_config import get_openai_compatible_kwargs



def is_io_dict(item: dict | str) -> bool:
    if isinstance(item, dict) and ("input" in item) and ("output" in item): return True
    return False

def get_trajectory(path: str):
    trajectory = []
    result = json.load(open(path, 'r'))
    for item in result:
        if not is_io_dict(item): continue
        step = {
            "env": "# " + item["input"][-1]["content"],
            "action": item["output"],
        }
        trajectory.append(step)
    return trajectory


def is_successful_result(path: str) -> bool:
    result = json.load(open(path, 'r'))
    for item in reversed(result):
        if isinstance(item, dict) and "success" in item:
            return bool(item["success"] and item["success"][-1])
    return False


def main():
    samples = load_json(args.data_dir, args.benchmark)
    print(f"Loaded #{len(samples)} test examples")
    samples = [s for s in samples if s["website"] == args.website]
    print(f"Filtering down to #{len(samples)} examples on website [{args.website}]")
    
    # load successful model predictions and format examples
    if not os.path.exists(args.results_dir):
        print(f"No result directory found at {args.results_dir}; leaving workflow memory unchanged.")
        return

    result_files = [
        os.path.join(args.results_dir, f)
        for f in os.listdir(args.results_dir)
        if f.endswith(".json") and f.split(".")[0].isdigit()
    ]
    result_files = sorted(result_files, key=lambda f: int(os.path.basename(f).split(".")[0]))
    result_files = [rf for rf in result_files if is_successful_result(rf)]
    if not result_files:
        print("No successful trajectories found; leaving workflow memory unchanged.")
        return

    result_list = [get_trajectory(rf) for rf in result_files]
    examples = []
    for rf, r in zip(result_files, result_list):
        sample_idx = int(os.path.basename(rf).split(".")[0])
        if sample_idx >= len(samples):
            continue
        s = samples[sample_idx]
        examples.append({
            "confirmed_task": s["confirmed_task"],
            "action_reprs": [step["env"] + '\n' + step["action"] for step in r],
        })
    if not examples:
        print("No result files matched loaded samples; leaving workflow memory unchanged.")
        return
    prompt = format_examples(examples, args.prefix, args.suffix)

    # transform to workflows
    INSTRUCTION = open(args.instruction_path, 'r').read()
    ONE_SHOT = open(args.one_shot_path, 'r').read()
    domain, subdomain, website = samples[0]["domain"], samples[0]["subdomain"], samples[0]["website"]
    prompt = '\n\n'.join([INSTRUCTION, ONE_SHOT, f"Website: {domain}, {subdomain}, {website}\n{prompt}"])
    from openai import OpenAI

    client = OpenAI(**get_openai_compatible_kwargs())
    response = client.chat.completions.create(
            model=args.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=args.temperature,
    ).choices[0].message.content
    response = filter_workflows(response, args.website)

    # save to file
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, 'w') as fw:
        fw.write(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--benchmark", type=str, default="test_task",
        choices=["test_task", "test_website", "test_domain", "train"])
    parser.add_argument("--website", type=str, required=True)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    # model
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.0)
    # prompt
    parser.add_argument("--instruction_path", type=str, default="prompt/instruction_action.txt")
    parser.add_argument("--one_shot_path", type=str, default="prompt/one_shot_action.txt")
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--suffix", type=str, default="# Summary Workflows")

    args = parser.parse_args()

    main()
