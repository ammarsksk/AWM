import argparse
import json
import sqlite3
import subprocess
import textwrap
from pathlib import Path


def esc(value: object) -> str:
    return str(value or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def wrap(value: object, width: int = 46, limit: int = 260) -> str:
    text = " ".join(str(value or "").split())
    if len(text) > limit:
        text = text[: limit - 3] + "..."
    return "\\n".join(textwrap.wrap(text, width=width, break_long_words=False)) or "N/A"


def node(lines: list[str], node_id: str, label: str, fill: str, color: str) -> None:
    lines.append(
        f'  "{esc(node_id)}" [label="{esc(label)}", fillcolor="{fill}", color="{color}"];'
    )


def edge(lines: list[str], source: str, target: str, label: str, color: str = "#475569") -> None:
    lines.append(f'  "{esc(source)}" -> "{esc(target)}" [label="{esc(label)}", color="{color}"];')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="webarena/memory/procedural/procedural_memory.sqlite3")
    parser.add_argument("--procedure", required=True)
    parser.add_argument("--out-dot", required=True)
    parser.add_argument("--out-png", required=True)
    parser.add_argument("--out-svg", required=True)
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM procedures WHERE name=?", (args.procedure,)).fetchone()
    if row is None:
        raise SystemExit(f"procedure not found: {args.procedure}")
    procedure = json.loads(row["procedure_json"])
    activation = procedure.get("activation", {})
    execution = procedure.get("execution", {})
    termination = procedure.get("termination", {})
    failure_recovery = procedure.get("failure_recovery", {})

    lines = [
        "digraph SingleProcedure {",
        "  rankdir=LR;",
        '  graph [labelloc=t, fontsize=18, fontname="Helvetica", bgcolor="white", '
        f'label="Single Stored Workflow Example: {esc(args.procedure)}"];',
        '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, margin="0.12,0.08"];',
        '  edge [fontname="Helvetica", fontsize=9, arrowsize=0.8];',
    ]

    proc_id = row["name"]
    family = activation.get("family", "unknown")
    skeleton = " -> ".join(execution.get("action_skeleton", []))
    procedure_label = (
        f"PROCEDURE\\n{proc_id}\\n"
        f"success={row['success_count']} failure={row['failure_count']}\\n"
        f"family={family}\\n"
        f"action skeleton={skeleton}\\n"
        f"goal={wrap(activation.get('goal_pattern') or row['goal_pattern'], 42, 180)}"
    )
    node(lines, proc_id, procedure_label, "#dbeafe", "#2563eb")

    family_id = f"family:{family}"
    intent_id = f"intent:{wrap(activation.get('goal_pattern') or row['goal_pattern'], 34, 80)}"
    strategy = execution.get("general_strategy", "")
    strategy_id = f"strategy:{wrap(strategy, 32, 90)}"
    answer_format = termination.get("answer_format", "")
    answer_id = f"answer_format:{wrap(answer_format, 32, 90)}"
    success_id = "outcome:success"

    node(lines, family_id, f"TASK FAMILY\\n{family}\\nUsed to prevent wrong-family reuse", "#fef3c7", "#d97706")
    node(lines, intent_id, f"GOAL PATTERN\\n{wrap(activation.get('goal_pattern') or row['goal_pattern'], 44, 220)}", "#ffedd5", "#ea580c")
    node(lines, strategy_id, f"GENERAL STRATEGY\\n{wrap(strategy, 44, 240)}", "#ecfccb", "#65a30d")
    node(lines, answer_id, f"ANSWER FORMAT\\n{wrap(answer_format, 44, 220)}", "#f5f3ff", "#7c3aed")
    node(lines, success_id, "SUCCESS OUTCOME\\nThis path previously solved tasks", "#dcfce7", "#16a34a")

    edge(lines, family_id, proc_id, "contains")
    edge(lines, intent_id, proc_id, "activates")
    edge(lines, proc_id, strategy_id, "uses_strategy")
    edge(lines, proc_id, answer_id, "formats_answer_as")

    previous = proc_id
    for step in execution.get("steps", []):
        step_id = f"{proc_id}:step:{step.get('step')}:{step.get('action_type')}"
        action_id = f"action:{step.get('action_type')}"
        check_id = f"check:{wrap(step.get('success_check'), 34, 80)}"
        step_label = (
            f"STEP {step.get('step')}: {step.get('action_type')}\\n"
            f"intent={wrap(step.get('intent'), 44, 260)}\\n"
            f"target policy={wrap(step.get('target_policy'), 44, 240)}\\n"
            f"value policy={wrap(step.get('value_policy'), 44, 160)}\\n"
            f"raw old action={wrap(step.get('raw_action_example'), 44, 120)}"
        )
        node(lines, step_id, step_label, "#ecfdf5", "#059669")
        node(lines, action_id, f"ACTION TYPE\\n{step.get('action_type')}\\nSkeleton-level operation", "#ede9fe", "#7c3aed")
        edge(lines, previous, step_id, "next_step / contains")
        edge(lines, step_id, action_id, "executes")
        if step.get("success_check"):
            node(lines, check_id, f"SUCCESS CHECK\\n{wrap(step.get('success_check'), 44, 200)}", "#f0f9ff", "#0284c7")
            edge(lines, step_id, check_id, "verifies")
        previous = step_id

    edge(lines, previous, success_id, "leads_to")

    avoid = failure_recovery.get("avoid") or []
    if avoid:
        avoid_id = "failure_recovery:avoid"
        node(lines, avoid_id, "AVOID / RECOVERY\\n" + "\\n".join(wrap(item, 46, 120) for item in avoid[:4]), "#fee2e2", "#dc2626")
        edge(lines, proc_id, avoid_id, "avoid")

    lines.append("}")
    Path(args.out_dot).write_text("\n".join(lines) + "\n")
    subprocess.run(["dot", "-Tpng", args.out_dot, "-o", args.out_png], check=True)
    subprocess.run(["dot", "-Tsvg", args.out_dot, "-o", args.out_svg], check=True)


if __name__ == "__main__":
    main()
