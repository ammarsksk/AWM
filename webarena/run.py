"""
WARNING DEPRECATED WILL BE REMOVED SOON
"""

import os
import argparse
import json
import sys
from types import MethodType
from pathlib import Path

import gymnasium as gym
from browsergym.experiments import ExpArgs, EnvArgs

from agents.legacy.agent import GenericAgentArgs
from agents.legacy.dynamic_prompting import Flags
from agents.legacy.utils.chat_api import ChatModelArgs


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser(description="Run experiment with hyperparameters.")
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/google/gemini-2.5-pro",
        help="Model name for the chat model.",
    )
    parser.add_argument(
        "--task_name",
        "--task",
        dest="task_name",
        type=str,
        default="openended",
        help="Name of the Browsergym task to run. If 'openended', you need to specify a 'start_url'",
    )
    parser.add_argument(
        "--start_url",
        type=str,
        default="https://www.google.com",
        help="Starting URL (only for the openended task).",
    )
    parser.add_argument(
        "--slow_mo", type=int, default=30, help="Slow motion delay for the playwright actions."
    )
    parser.add_argument(
        "--headless",
        type=str2bool,
        default=False,
        help="Run the experiment in headless mode (hides the browser windows).",
    )
    parser.add_argument(
        "--demo_mode",
        type=str2bool,
        default=True,
        help="Add visual effects when the agents performs actions.",
    )
    parser.add_argument(
        "--use_html", type=str2bool, default=False, help="Use HTML in the agent's observation space."
    )
    parser.add_argument(
        "--use_ax_tree",
        type=str2bool,
        default=True,
        help="Use AX tree in the agent's observation space.",
    )
    parser.add_argument(
        "--use_screenshot",
        type=str2bool,
        default=True,
        help="Use screenshot in the agent's observation space.",
    )
    parser.add_argument(
        "--multi_actions", type=str2bool, default=True, help="Allow multi-actions in the agent."
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="bid",
        choices=["python", "bid", "coord", "bid+coord", "bid+nav", "coord+nav", "bid+coord+nav"],
        help="",
    )
    parser.add_argument(
        "--use_history",
        type=str2bool,
        default=True,
        help="Use history in the agent's observation space.",
    )
    parser.add_argument(
        "--use_thinking",
        type=str2bool,
        default=True,
        help="Use thinking in the agent (chain-of-thought prompting).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=30,
        help="Maximum number of steps to take for each task.",
    )
    parser.add_argument("--llm_retries", type=int, default=6)
    parser.add_argument("--pre_observation_delay", type=float, default=1.25)
    parser.add_argument("--extract_obs_retries", type=int, default=8)
    parser.add_argument(
        "--workflow_path",
        type=str,
        default=None,
        help="Path to the memory file to load for the agent.",
    )
    parser.add_argument(
        "--browser_proxy",
        type=str,
        default=None,
        help="Proxy server for Playwright Chromium, e.g. http://host:port.",
    )
    parser.add_argument(
        "--procedural_memory_path",
        type=str,
        default=None,
        help="Path to the WebArena procedural memory directory.",
    )
    parser.add_argument("--procedural_site", type=str, default="shopping")
    parser.add_argument("--procedural_top_k", type=int, default=4)
    parser.add_argument("--procedural_min_score", type=float, default=0.42)

    return parser.parse_args()


def attach_browser_runtime_options(
    env_args: EnvArgs,
    proxy_server: str | None,
    pre_observation_delay: float,
) -> None:
    def make_env_with_proxy(self, action_mapping, exp_dir, exp_task_kwargs=None, use_raw_page_output=False):
        if self.task_name.startswith("webarena"):
            import browsergym.webarena  # noqa: F401

        exp_task_kwargs = exp_task_kwargs or {}
        extra_kwargs = {"pre_observation_delay": pre_observation_delay}
        if proxy_server:
            extra_kwargs["pw_chromium_kwargs"] = {"proxy": {"server": proxy_server}}
        if self.record_video:
            extra_kwargs["record_video_dir"] = exp_dir
        if self.viewport:
            extra_kwargs["viewport"] = self.viewport
        if self.slow_mo is not None:
            extra_kwargs["slow_mo"] = self.slow_mo
        if self.storage_state:
            extra_kwargs["pw_context_kwargs"] = {"storage_state": self.storage_state}
        if self.task_kwargs is not None:
            extra_kwargs["task_kwargs"] = self.task_kwargs
        if exp_task_kwargs:
            extra_kwargs["task_kwargs"] = extra_kwargs.get("task_kwargs", {}) | exp_task_kwargs

        return gym.make(
            f"browsergym/{self.task_name}",
            disable_env_checker=True,
            max_episode_steps=self.max_steps,
            headless=self.headless,
            wait_for_user_message=self.wait_for_user_message,
            action_mapping=action_mapping,
            use_raw_page_output=use_raw_page_output,
            **extra_kwargs,
        )

    env_args.make_env = MethodType(make_env_with_proxy, env_args)


def main():
    print(
        """\
WARNING this demo agent will soon be moved elsewhere. Expect it to be removed at some point."""
    )

    args = parse_args()
    if (args.workflow_path is not None) and (not os.path.exists(args.workflow_path)):
        open(args.workflow_path, "w").close()

    env_args = EnvArgs(
        task_name=args.task_name,
        task_seed=None,
        max_steps=args.max_steps,
        headless=args.headless,
        viewport={"width": 1500, "height": 1280},
        slow_mo=args.slow_mo,
    )

    if args.task_name == "openended":
        env_args.wait_for_user_message = True
        env_args.task_kwargs = {"start_url": args.start_url}
    try:
        import browsergym.core.env as browsergym_env

        browsergym_env.EXTRACT_OBS_MAX_TRIES = max(
            int(args.extract_obs_retries), browsergym_env.EXTRACT_OBS_MAX_TRIES
        )
    except Exception:
        pass
    attach_browser_runtime_options(
        env_args,
        proxy_server=args.browser_proxy,
        pre_observation_delay=args.pre_observation_delay,
    )

    exp_args = ExpArgs(
        env_args=env_args,
        agent_args=GenericAgentArgs(
            chat_model_args=ChatModelArgs(
                model_name=args.model_name,
                max_total_tokens=128_000,  # "Maximum total tokens for the chat model."
                max_input_tokens=126_000,  # "Maximum tokens for the input to the chat model."
                max_new_tokens=2_000,  # "Maximum total tokens for the chat model."
            ),
            flags=Flags(
                use_html=args.use_html,
                use_ax_tree=args.use_ax_tree,
                use_thinking=args.use_thinking,  # "Enable the agent with a memory (scratchpad)."
                use_error_logs=True,  # "Prompt the agent with the error logs."
                use_memory=False,  # "Enables the agent with a memory (scratchpad)."
                use_history=args.use_history,
                use_diff=False,  # "Prompt the agent with the difference between the current and past observation."
                use_past_error_logs=True,  # "Prompt the agent with the past error logs."
                use_action_history=True,  # "Prompt the agent with the action history."
                multi_actions=args.multi_actions,
                action_space=args.action_space,
                use_abstract_example=True,  # "Prompt the agent with an abstract example."
                use_concrete_example=True,  # "Prompt the agent with a concrete example."
                use_screenshot=args.use_screenshot,
                enable_chat=True,
                demo_mode="default" if args.demo_mode else "off",
                workflow_path=args.workflow_path,
                procedural_memory_path=args.procedural_memory_path,
                procedural_site=args.procedural_site,
                procedural_top_k=args.procedural_top_k,
                procedural_min_score=args.procedural_min_score,
            ),
            max_retry=args.llm_retries,
        ),
    )

    exp_args.prepare(Path("./results"))
    exp_args.run()

    final_dir = Path("results") / args.task_name
    if final_dir.exists():
        suffix = 0
        while True:
            archived = final_dir.with_name(
                f"_{final_dir.name}" if suffix == 0 else f"_{final_dir.name}_{suffix}"
            )
            if not archived.exists():
                final_dir.rename(archived)
                break
            suffix += 1
    Path(exp_args.exp_dir).rename(final_dir)

    summary_path = final_dir / "summary_info.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if summary.get("err_msg"):
            print(summary["err_msg"], file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
