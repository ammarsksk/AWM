"""Tiny local smoke test for Agent Workflow Memory.

This does not run a browser or a benchmark. It demonstrates the core AWM loop:

1. Start from a successful trajectory.
2. Clean/filter valid actions.
3. Convert the trajectory into workflow-memory text.
4. Save the workflow memory.
5. Verify that the agent prompt would load and expose that memory.
"""

from pathlib import Path
import sys
import tempfile

from induce_rule import format_trajectory, get_abstract_trajectory, remove_invalid_steps


def build_workflow_memory() -> str:
    query = "Find cafes near Carnegie Mellon University and report the closest one."
    raw_actions = [
        'fill("145", "cafes near Carnegie Mellon University")',
        'click("147")',
        "click(147)",  # invalid: element ids must be strings in this action space
        "noop()",
        'send_msg_to_user("The closest cafe is La Prima Espresso.")',
    ]
    cleaned_actions = remove_invalid_steps(raw_actions)

    think_list = [
        "I should search the map for cafes near Carnegie Mellon University.",
        "The search results are visible, so I will open the most relevant result.",
        "I found the closest cafe and should report it to the user.",
    ]
    action_list = [
        cleaned_actions[:2],
        [cleaned_actions[2]],
        [cleaned_actions[3]],
    ]

    abstract = get_abstract_trajectory(action_list)
    trajectory = format_trajectory(think_list, action_list)
    return (
        "## Concrete Examples\n\n"
        f"Query: {query}\n"
        f"# Abstract action pattern: {abstract}\n\n"
        f"{trajectory}\n"
    )


def build_agent_system_prompt(goal: str, workflow_path: Path) -> str:
    workflows = workflow_path.read_text().strip()
    return f"""\
# Instructions
Review the current state of the page and choose the best next action.

# Goal:
{goal}

# Agent Workflow Memory
The following workflows are reusable routines induced from previous successful
tasks. Use them as guidance when they match the current goal, adapting element
ids and variable values to the current page.

{workflows}
"""


def main() -> int:
    workflow_text = build_workflow_memory()

    with tempfile.TemporaryDirectory(prefix="awm-smoke-") as tmpdir:
        workflow_path = Path(tmpdir) / "map_smoke_workflow.txt"
        workflow_path.write_text(workflow_text)

        prompt = build_agent_system_prompt(
            "Find the closest cafe to CMU Hunt Library.",
            workflow_path,
        )

        checks = {
            "workflow file was written": workflow_path.exists(),
            "invalid numeric click was filtered": "click(147)" not in workflow_text,
            "valid search action is present": 'fill("145", "cafes near Carnegie Mellon University")' in workflow_text,
            "memory is injected into prompt": "# Agent Workflow Memory" in prompt,
            "workflow query is visible to agent": "Find cafes near Carnegie Mellon University" in prompt,
        }

        print("AWM smoke test")
        print("=" * 48)
        print(f"Workflow memory path: {workflow_path}")
        print()
        for name, passed in checks.items():
            print(f"[{'PASS' if passed else 'FAIL'}] {name}")

        print()
        print("Generated workflow memory preview:")
        print("-" * 48)
        print(workflow_text[:1200].rstrip())

        if not all(checks.values()):
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
