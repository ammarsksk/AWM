#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IP="${WEBARENA_IP:-18.225.67.57}"
BASE_DOMAIN="${IP}.nip.io"
STAMP="$(date +%Y%m%d_%H%M%S)"
FAILED_TASK_IDS="${FAILED_TASK_IDS:-21 25 26 49 50}"

SOURCE_MEMORY="webarena/memory/procedural"
OUT_DIR="metrics/failed_tasks_new_agent_frozen_memory_${STAMP}"
FULL_OUT_DIR="webarena/${OUT_DIR}"
SNAPSHOT_DIR="${FULL_OUT_DIR}/frozen_memory_snapshot"

if [ ! -d "${SOURCE_MEMORY}" ]; then
  echo "Missing source memory: ${SOURCE_MEMORY}" >&2
  exit 1
fi

mkdir -p "${FULL_OUT_DIR}"
cp -a "${SOURCE_MEMORY}" "${SNAPSHOT_DIR}"
printf '%s\n' "${FULL_OUT_DIR}" > webarena/metrics/latest_failed_tasks_frozen_eval_dir.txt

echo "Failed-task frozen-memory retest"
echo "IP: ${IP}"
echo "Tasks: ${FAILED_TASK_IDS}"
echo "Frozen memory snapshot: ${SNAPSHOT_DIR}"
echo "Output directory: ${FULL_OUT_DIR}"
echo

for TASK_ID in ${FAILED_TASK_IDS}; do
  TASK_MEMORY="${FULL_OUT_DIR}/memory_task_${TASK_ID}"
  TASK_OUTPUT="${OUT_DIR}/task_${TASK_ID}"
  rm -rf "${TASK_MEMORY}"
  cp -a "${SNAPSHOT_DIR}" "${TASK_MEMORY}"

  echo "=== Running webarena.${TASK_ID} with frozen memory copy ==="
  WA_SHOPPING="http://shopping.${BASE_DOMAIN}" \
  WA_SHOPPING_ADMIN="http://admin.${BASE_DOMAIN}/admin" \
  WA_REDDIT="http://reddit.${BASE_DOMAIN}/forums/all" \
  WA_GITLAB="http://gitlab.${BASE_DOMAIN}/explore" \
  WA_WIKIPEDIA="http://wiki.${BASE_DOMAIN}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing" \
  WA_MAP="http://map.${BASE_DOMAIN}" \
  WA_HOMEPAGE="http://shopping.${BASE_DOMAIN}" \
  WA_FULL_RESET="http://reset.${BASE_DOMAIN}" \
  WEBARENA_EVAL_MODEL="google/gemini-2.5-pro" \
  ./.venv/bin/python webarena/run_procedural_eval_with_metrics.py \
    --website shopping \
    --task-ids "${TASK_ID}" \
    --output-dir "${TASK_OUTPUT}" \
    --procedural-memory-dir "${OUT_DIR}/memory_task_${TASK_ID}" \
    --vector-backend hnsw_sq8 \
    --embedding-model BAAI/bge-small-en-v1.5 \
    --reranker ml \
    --model-name openai/google/gemini-2.5-pro \
    --procedural-abstraction-model openai/google/gemini-2.5-pro \
    --procedural-top-k 4 \
    --procedural-min-score 0.42 \
    --max-steps 30 \
    --llm-retries 6 \
    --task-retries 1 \
    --retry-sleep 15 \
    --use-thinking false \
    --skip-ingest \
    --pre-observation-delay 1.25 \
    --extract-obs-retries 8 \
    --browser-proxy http://192.168.140.15:3128 \
    --headless
done

./.venv/bin/python - <<PY
import json
from pathlib import Path

out = Path("${FULL_OUT_DIR}")
rows = []
for metrics_path in sorted(out.glob("task_*/metrics.json")):
    payload = json.loads(metrics_path.read_text())
    task_rows = payload.get("tasks", [])
    for row in task_rows:
        rows.append({
            "task_id": row.get("task_id"),
            "reward": row.get("reward"),
            "errored": row.get("errored"),
            "n_steps": row.get("n_steps"),
            "agent_tokens": row.get("agent_tokens"),
            "output_dir": str(metrics_path.parent),
        })
summary = {
    "tasks": len(rows),
    "successful_tasks": sum(1 for row in rows if row.get("reward") == 1.0),
    "failed_tasks": sum(1 for row in rows if row.get("reward") != 1.0),
    "errored_tasks": sum(1 for row in rows if row.get("errored")),
    "rows": rows,
}
(out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
lines = [
    "# Failed-Task Frozen-Memory Retest",
    "",
    f"- Tasks: {summary['tasks']}",
    f"- Successful tasks: {summary['successful_tasks']}",
    f"- Failed tasks: {summary['failed_tasks']}",
    f"- Errored tasks: {summary['errored_tasks']}",
    "",
    "| Task | Reward | Error | Steps | Agent tokens |",
    "|---:|---:|:---:|---:|---:|",
]
for row in rows:
    lines.append(
        f"| {row['task_id']} | {row['reward']} | {'yes' if row['errored'] else 'no'} | "
        f"{row['n_steps']} | {row['agent_tokens']} |"
    )
(out / "summary.md").write_text("\\n".join(lines) + "\\n")
print(json.dumps(summary, indent=2, sort_keys=True))
print(f"wrote={out / 'summary.json'}")
print(f"wrote={out / 'summary.md'}")
PY
