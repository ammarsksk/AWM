#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IP="${WEBARENA_IP:-18.225.67.57}"
BASE_DOMAIN="${IP}.nip.io"
STAMP="$(date +%Y%m%d_%H%M%S)"

MEMORY_DIR="webarena/memory/procedural"
BACKUP_DIR="webarena/memory/procedural_backup_before_clean_shopping_${STAMP}"
OUTPUT_DIR="metrics/full_shopping_192_clean_hardwareteam_ml_reranker_hnsw_sq8_${STAMP}"
TASK_IDS="$(
  cd webarena
  ../.venv/bin/python - <<'PY'
from pipeline import load_task_ids
print(",".join(str(task_id) for task_id in load_task_ids("shopping")))
PY
)"

echo "Clean WebArena shopping run"
echo "IP: ${IP}"
echo "Memory backup: ${BACKUP_DIR}"
echo "Output directory: webarena/${OUTPUT_DIR}"
echo "Task count: $(tr -cd ',' <<<"${TASK_IDS}" | wc -c | awk '{print $1 + 1}')"

if [ -e "${MEMORY_DIR}" ]; then
  mv "${MEMORY_DIR}" "${BACKUP_DIR}"
fi
mkdir -p "${MEMORY_DIR}"
mkdir -p "webarena/${OUTPUT_DIR}"
printf '%s\n' "webarena/${OUTPUT_DIR}" > webarena/metrics/latest_full_shopping_clean_ml_dir.txt

echo
echo "Run log will be written to:"
echo "  webarena/${OUTPUT_DIR}/run.log"
echo
echo "In another terminal, watch it with:"
echo "  tail -f ~/Downloads/agent-workflow-memory-main/webarena/${OUTPUT_DIR}/run.log"
echo

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
  --task-ids "${TASK_IDS}" \
  --output-dir "${OUTPUT_DIR}" \
  --procedural-memory-dir memory/procedural \
  --vector-backend hnsw_sq8 \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --reranker ml \
  --model-name openai/google/gemini-2.5-pro \
  --procedural-abstraction-model openai/google/gemini-2.5-pro \
  --procedural-top-k 4 \
  --procedural-min-score 0.42 \
  --max-steps 30 \
  --llm-retries 6 \
  --task-retries 2 \
  --retry-sleep 15 \
  --use-thinking false \
  --pre-observation-delay 1.25 \
  --extract-obs-retries 8 \
  --browser-proxy http://192.168.140.15:3128 \
  --headless
