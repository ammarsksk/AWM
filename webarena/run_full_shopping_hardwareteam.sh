#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IP="${WEBARENA_IP:-18.225.67.57}"
BASE_DOMAIN="${IP}.nip.io"

export WA_SHOPPING="http://shopping.${BASE_DOMAIN}"
export WA_SHOPPING_ADMIN="http://admin.${BASE_DOMAIN}/admin"
export WA_REDDIT="http://reddit.${BASE_DOMAIN}/forums/all"
export WA_GITLAB="http://gitlab.${BASE_DOMAIN}/explore"
export WA_WIKIPEDIA="http://wiki.${BASE_DOMAIN}/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
export WA_MAP="http://map.${BASE_DOMAIN}"
export WA_HOMEPAGE="${WA_SHOPPING}"
export WA_FULL_RESET="http://reset.${BASE_DOMAIN}"

export SHOPPING="${WA_SHOPPING}"
export SHOPPING_ADMIN="${WA_SHOPPING_ADMIN}"
export REDDIT="${WA_REDDIT}"
export GITLAB="${WA_GITLAB}"
export WIKIPEDIA="${WA_WIKIPEDIA}"
export MAP="${WA_MAP}"
export HOMEPAGE="${WA_HOMEPAGE}"
export WEBARENA_EVAL_MODEL="${WEBARENA_EVAL_MODEL:-google/gemini-2.5-pro}"

TASK_IDS="$(
  ./.venv/bin/python - <<'PY'
import json
import pathlib

ids = []
for p in sorted(pathlib.Path("webarena/config_files").glob("*.json"), key=lambda x: int(x.stem) if x.stem.isdigit() else 10**9):
    if not p.stem.isdigit():
        continue
    d = json.loads(p.read_text())
    sites = d.get("sites") or []
    if sites and sites[0] == "shopping":
        ids.append(str(d["task_id"]))
print(",".join(ids))
PY
)"

./.venv/bin/python webarena/run_procedural_eval_with_metrics.py \
  --website shopping \
  --task-ids "${TASK_IDS}" \
  --output-dir metrics/full_shopping_hardwareteam_nipio_hnsw_sq8_ml \
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
  --pre-observation-delay 1.25 \
  --extract-obs-retries 8 \
  --browser-proxy http://192.168.140.15:3128 \
  --headless
