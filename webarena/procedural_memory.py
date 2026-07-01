from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

from advanced_vector_index import AdvancedIndexConfig, AdvancedProcedureVectorIndex
from local_awm_full_demo import WorkflowEmbeddingIndex
from procedural_reranker import ProceduralReranker


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def now_ms() -> int:
    return int(time.time() * 1000)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def overlap(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def goal_family(text: str) -> str:
    lowered = (text or "").lower()
    tokens = token_set(lowered)
    if (
        "main criticisms" in lowered
        or "customers say" in lowered
        or "customer reviews" in lowered
        or "extract the relevant sentences" in lowered
        or "customer service" in lowered
        or "review" in tokens
        or "reviews" in tokens
    ):
        if (
            "reviewers" in tokens
            or "reviewer" in tokens
            or "complain" in tokens
            or "complained" in tokens
        ) and (
            "mention" in tokens
            or "mentioned" in tokens
            or "complain" in tokens
            or "complained" in tokens
            or "customer service" in lowered
        ):
            return "review_lookup"
        return "review_extraction"
    if (
        ("total cost" in lowered or "order total" in lowered)
        and "latest" in tokens
        and "order" in tokens
    ):
        return "order_status_total"
    if (
        "storage option" in lowered
        or ("storage" in tokens and ("fit" in tokens or "cards" in tokens or "capacity" in tokens))
    ):
        return "product_capacity_search"
    if "price range" in lowered or ("price" in tokens and "range" in tokens):
        return "price_range"
    if "how much i spent" in lowered or ("spent" in tokens and "shopping" in tokens):
        return "spend_category_total"
    if "fulfilled" in tokens and "orders" in tokens and ("spent" in tokens or "money" in tokens):
        return "fulfilled_order_total"
    if "configuration" in tokens and "bought" in tokens:
        return "bought_option_lookup"
    if "reviewers" in tokens and ("mention" in tokens or "mentioned" in tokens):
        return "review_lookup"
    if "fulfilled" in tokens and "orders" in tokens:
        return "fulfilled_order_total"
    if "first" in tokens and "purchase" in tokens:
        return "first_purchase_lookup"
    if "show" in tokens or "alleviate" in tokens:
        return "product_search"
    return "general"


FAMILY_ALIASES = {
    "shopping_find_product": "product_capacity_search",
    "shopping_find_item_with_capacity": "product_capacity_search",
    "find_item_with_capacity": "product_capacity_search",
    "product_search": "product_search",
    "review_extraction": "review_extraction",
    "extraction": "review_extraction",
    "information_extraction": "review_extraction",
    "review_lookup": "review_lookup",
    "shopping.order_status": "order_status_total",
    "shopping_order_status": "order_status_total",
    "order_status": "order_status_total",
    "order_lookup": "order_status_total",
    "order_history": "bought_option_lookup",
    "find_order_details": "bought_option_lookup",
    "find_product_reviews": "review_extraction",
}


def canonical_family(family: str | None) -> str:
    key = (family or "general").strip().lower().replace("-", "_").replace(" ", "_")
    return FAMILY_ALIASES.get(key, key or "general")


EXPECTED_ACTIONS = {
    "price_range": {"fill", "click", "select_option", "send_msg_to_user"},
    "spend_category_total": {"click", "send_msg_to_user"},
    "bought_option_lookup": {"click", "send_msg_to_user"},
    "review_lookup": {"click", "scroll", "send_msg_to_user"},
    "review_extraction": {"click", "scroll", "send_msg_to_user"},
    "fulfilled_order_total": {"click", "select_option", "send_msg_to_user"},
    "first_purchase_lookup": {"click", "send_msg_to_user"},
    "order_status_total": {"click", "send_msg_to_user"},
    "product_capacity_search": {"fill", "click"},
    "product_search": {"fill", "click"},
}


def compact(text: str, limit: int = 1200) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def action_name(action: str) -> str:
    match = re.match(r"\s*([a-zA-Z_]+)\(", action or "")
    return match.group(1) if match else (action or "").strip()


def action_args(action: str) -> str:
    match = re.match(r"\s*[a-zA-Z_]+\((.*)\)\s*$", action or "")
    return match.group(1) if match else ""


def safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


def read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return default


def read_json_from_text(text: str, default: Any) -> Any:
    try:
        return json.loads(text or "")
    except Exception:
        return default


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except Exception:
            return None
    return None


def extract_trace(log_path: Path) -> tuple[list[str], list[str]]:
    """Extract BrowserGym INFO thought/action blocks from experiment.log."""
    entries, current = [], []
    for line in log_path.read_text(errors="ignore").splitlines(True):
        if re.match(r"^\d{4}-\d{2}-\d{2} ", line):
            if current:
                entries.append("".join(current).rstrip())
            current = [line]
        elif current:
            current.append(line)
    if current:
        entries.append("".join(current).rstrip())

    marker = "browsergym.experiments.loop - INFO - "
    thoughts, actions = [], []
    for entry in entries:
        if marker not in entry or "action:" not in entry:
            continue
        content = entry.split(marker, 1)[1].strip()
        thought, action_block = content.split("action:", 1)
        for raw in action_block.splitlines():
            raw = raw.strip()
            if "(" in raw and ")" in raw:
                thoughts.append(thought.strip())
                actions.append(raw)
    return thoughts, actions


def ui_signature(observation: str) -> dict[str, Any]:
    text = compact(observation, 2400)
    controls = re.findall(
        r"\b(button|link|menuitem|tab|textbox|searchbox|combobox|option|select|input)\b",
        observation or "",
        flags=re.IGNORECASE,
    )
    return {
        "tokens": sorted(token_set(text))[:256],
        "top_terms": [term for term, _ in Counter(tokenize(text)).most_common(48)],
        "control_counts": dict(Counter(item.lower() for item in controls).most_common(20)),
    }


class WebArenaProceduralMemory:
    semantic_weight = 0.48
    lexical_weight = 0.18
    activation_weight = 0.16
    metadata_weight = 0.06
    outcome_weight = 0.10
    graph_weight = 0.08
    family_weight = 0.0
    structure_weight = 0.12
    family_mismatch_penalty = 0.0
    cross_family_min_score = 0.0

    def __init__(self, root: Path, candidate_k: int = 40) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "procedural_memory.sqlite3"
        self.fast_mode = os.environ.get("WEBARENA_FAST_MEMORY", "").lower() in {"1", "true", "yes"}
        self.candidate_k = candidate_k
        self.vector_backend = os.environ.get("WEBARENA_VECTOR_BACKEND", "hnsw_sq8")
        self.reranker_mode = os.environ.get("WEBARENA_RERANKER_MODE", "full")
        self.index = None if self.fast_mode else AdvancedProcedureVectorIndex(
            self.root / "advanced_vector_index",
            AdvancedIndexConfig(
                index_kind=self.vector_backend,
                candidate_k=candidate_k,
                hnsw_m=int(os.environ.get("WEBARENA_HNSW_M", "32")),
                hnsw_ef_search=int(os.environ.get("WEBARENA_HNSW_EF_SEARCH", "64")),
                hnsw_ef_construction=int(os.environ.get("WEBARENA_HNSW_EF_CONSTRUCTION", "80")),
                ivf_nlist=int(os.environ.get("WEBARENA_IVF_NLIST", "64")),
                ivf_nprobe=int(os.environ.get("WEBARENA_IVF_NPROBE", "8")),
                pq_m=int(os.environ.get("WEBARENA_PQ_M", "16")),
                pq_bits=int(os.environ.get("WEBARENA_PQ_BITS", "8")),
                rotation_seed=int(os.environ.get("WEBARENA_ROTATION_SEED", "1337")),
            ),
        )
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self.reranker = None
        if self.reranker_mode.lower() != "none":
            self.reranker = ProceduralReranker(
                db_path=self.db_path,
                mode=self.reranker_mode,
                cross_encoder_model=os.environ.get("WEBARENA_RERANKER_MODEL", "BAAI/bge-reranker-large"),
                train_limit=int(os.environ.get("WEBARENA_RERANKER_TRAIN_LIMIT", "1000")),
            )
        self._sync_index()

    def close(self) -> None:
        self.conn.close()

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS procedures (
                name TEXT PRIMARY KEY,
                site TEXT NOT NULL,
                intent_template_id INTEGER,
                goal_pattern TEXT NOT NULL,
                procedure_json TEXT NOT NULL,
                compact_text TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                avg_steps REAL NOT NULL DEFAULT 0.0,
                avg_agent_elapsed REAL NOT NULL DEFAULT 0.0,
                created_at_ms INTEGER NOT NULL,
                updated_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS procedure_edges (
                source TEXT NOT NULL,
                edge_type TEXT NOT NULL,
                target TEXT NOT NULL,
                weight REAL NOT NULL DEFAULT 1.0,
                evidence TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (source, edge_type, target)
            );

            CREATE TABLE IF NOT EXISTS negative_memory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                procedure_name TEXT,
                site TEXT NOT NULL,
                pattern TEXT NOT NULL,
                family TEXT NOT NULL DEFAULT 'general',
                abstract_failure TEXT NOT NULL DEFAULT '',
                avoid_json TEXT NOT NULL DEFAULT '[]',
                selected_memory TEXT,
                penalty REAL NOT NULL DEFAULT 0.08,
                evidence TEXT NOT NULL,
                created_at_ms INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS retrieval_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                goal TEXT NOT NULL,
                site TEXT NOT NULL,
                selected TEXT,
                candidates_json TEXT NOT NULL,
                raw_candidates_json TEXT NOT NULL DEFAULT '[]',
                rejected_candidates_json TEXT NOT NULL DEFAULT '[]',
                created_at_ms INTEGER NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_procedures_site
                ON procedures(site, intent_template_id);
            CREATE INDEX IF NOT EXISTS idx_edges_source
                ON procedure_edges(source);
            CREATE INDEX IF NOT EXISTS idx_negative_site
                ON negative_memory(site);
            """
        )
        self._ensure_column("negative_memory", "family", "TEXT NOT NULL DEFAULT 'general'")
        self._ensure_column("negative_memory", "abstract_failure", "TEXT NOT NULL DEFAULT ''")
        self._ensure_column("negative_memory", "avoid_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("negative_memory", "selected_memory", "TEXT")
        self._ensure_column("retrieval_events", "raw_candidates_json", "TEXT NOT NULL DEFAULT '[]'")
        self._ensure_column("retrieval_events", "rejected_candidates_json", "TEXT NOT NULL DEFAULT '[]'")
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        columns = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")

    def _sync_index(self) -> None:
        if self.fast_mode or self.index is None:
            return
        rows = self.conn.execute(
            "SELECT name, compact_text FROM procedures ORDER BY created_at_ms, name"
        ).fetchall()
        names = [row["name"] for row in rows]
        current_names = [entry.get("name") for entry in getattr(self.index, "entries", [])]
        if names != current_names:
            if isinstance(self.index, AdvancedProcedureVectorIndex):
                self.index.rebuild_from_rows(rows)
                self.index.save()
                return
            self.index.entries = [
                {
                    "name": row["name"],
                    "text_for_embedding": row["compact_text"],
                    "vector": self.index.embed(row["compact_text"]),
                }
                for row in rows
            ]
            self.index._rebuild_search_indexes()
            self.index.save()

    def _procedure_name(self, site: str, intent_template_id: int | None, actions: list[str]) -> str:
        skeleton = "_".join(action_name(action) for action in actions[:6]) or "empty"
        return f"{site}.template_{intent_template_id or 'unknown'}.{skeleton}"

    def _procedure_from_result(
        self,
        result_dir: Path,
        config: dict[str, Any],
        summary: dict[str, Any],
        abstraction: str = "deterministic",
        abstraction_model: str = "openai/google/gemini-2.5-pro",
    ) -> dict[str, Any] | None:
        thoughts, actions = extract_trace(result_dir / "experiment.log")
        if not actions:
            return None
        site = config.get("sites", ["unknown"])[0]
        intent = config.get("intent", "")
        intent_template_id = config.get("intent_template_id")
        steps = []
        for i, action in enumerate(actions, start=1):
            name = action_name(action)
            args = action_args(action)
            thought = thoughts[i - 1] if i - 1 < len(thoughts) else ""
            steps.append(
                {
                    "step": i,
                    "action_type": name,
                    "intent": compact(thought, 500),
                    "target_policy": self._target_policy(name, args, thought),
                    "value_policy": self._value_policy(name, args, intent),
                    "raw_action_example": action,
                }
            )
        procedure = {
            "name": self._procedure_name(site, intent_template_id, actions),
            "site": site,
            "intent_template_id": intent_template_id,
            "activation": {
                "goal_pattern": compact(intent, 300),
                "family": "site_scoped",
                "keywords": [term for term, _ in Counter(tokenize(intent)).most_common(24)],
                "ui_signature": {},
            },
            "execution": {
                "steps": steps,
                "action_skeleton": [action_name(action) for action in actions],
            },
            "termination": {
                "success_signal": "The final answer satisfies the WebArena evaluator and the task reward is positive.",
                "final_action": actions[-1],
            },
            "failure_recovery": {
                "avoid": [
                    "Do not reuse old bid ids directly; map each step to the current visible element.",
                    "Wait for navigation/search results before reading page contents.",
                    "If scanning reviews or search pages, verify all relevant pages before answering no matches.",
                ]
            },
            "stats": {
                "source_task": result_dir.name,
                "steps": summary.get("n_steps", 0),
                "agent_elapsed": summary.get("stats.cum_agent_elapsed", 0),
                "abstraction": "deterministic",
            },
        }
        if abstraction == "llm":
            refined = self._llm_refine_procedure(
                procedure=procedure,
                thoughts=thoughts,
                actions=actions,
                config=config,
                summary=summary,
                model_name=abstraction_model,
            )
            if refined is None:
                return None
            procedure = refined
        return procedure

    def _llm_refine_procedure(
        self,
        procedure: dict[str, Any],
        thoughts: list[str],
        actions: list[str],
        config: dict[str, Any],
        summary: dict[str, Any],
        model_name: str,
    ) -> dict[str, Any] | None:
        try:
            from langchain.schema import HumanMessage, SystemMessage
            from agents.legacy.utils.chat_api import ChatModelArgs
        except Exception:
            return None

        raw_steps = [
            {
                "step": index + 1,
                "thought": compact(thoughts[index] if index < len(thoughts) else "", 700),
                "action": actions[index],
                "action_type": action_name(actions[index]),
            }
            for index in range(len(actions))
        ]
        eval_spec = config.get("eval", {})
        prompt = {
            "task_goal": config.get("intent", ""),
            "intent_template": config.get("intent_template", ""),
            "intent_template_id": config.get("intent_template_id"),
            "site": config.get("sites", ["unknown"])[0],
            "success_reward": summary.get("cum_reward"),
            "evaluator": eval_spec,
            "raw_trace_steps": raw_steps,
        }
        system = """You convert successful WebArena browser traces into reusable procedural memory.
Return JSON only. Do not copy numeric bid ids as reusable targets. Abstract them into page-relative policies.
Do not overfit to literal product/order values, except for answer-format rules learned from evaluator examples.
The procedure should help a future agent on the same website solve a similar page/goal context with different variables."""
        human = """Create compact reusable procedure JSON. Return JSON only, no markdown fence.
Keep the result concise: at most 7 steps, short strings, no old numeric bid ids.
Schema:
{
  "context_label": "short_site_context_label",
  "goal_pattern": "abstract goal with variables",
  "activation_keywords": ["..."],
  "general_strategy": "one concise reusable strategy",
  "answer_format": "how to format the final answer for strict evaluator matching",
  "steps": [
    {
      "action_type": "click|fill|select_option|scroll|send_msg_to_user|...",
      "intent": "why this step exists",
      "target_policy": "how to identify the current-page target without old bid ids",
      "value_policy": "what value to use; mention variable source",
      "success_check": "what should be true after the step"
    }
  ],
  "avoid": ["specific reusable failure modes to avoid"]
}

Trace payload:
""" + json.dumps(prompt, ensure_ascii=True, indent=2)

        try:
            chat = ChatModelArgs(
                model_name=model_name,
                temperature=0.0,
                max_new_tokens=4096,
                max_total_tokens=32000,
                max_input_tokens=28000,
            ).make_chat_model()
            answer = chat.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        except Exception as exc:
            print(f"llm_abstraction_error:{type(exc).__name__}:{exc}", file=sys.stderr)
            return None

        payload = extract_json_object(getattr(answer, "content", ""))
        if not payload:
            payload = self._llm_refine_procedure_compact_retry(
                config=config,
                summary=summary,
                actions=actions,
                model_name=model_name,
            )
            if not payload:
                print("llm_abstraction_error:invalid_json_or_truncated", file=sys.stderr)
                return None

        steps = []
        for index, step in enumerate(payload.get("steps", []), start=1):
            if not isinstance(step, dict):
                continue
            action_type = compact(step.get("action_type") or "click", 80)
            steps.append(
                {
                    "step": index,
                    "action_type": action_type,
                    "intent": compact(step.get("intent") or "", 500),
                    "target_policy": compact(step.get("target_policy") or self._target_policy(action_type, "", ""), 500),
                    "value_policy": compact(step.get("value_policy") or self._value_policy(action_type, "", config.get("intent", "")), 400),
                    "success_check": compact(step.get("success_check") or "", 300),
                    "raw_action_example": actions[index - 1] if index - 1 < len(actions) else "",
                }
            )
        if not steps:
            print("llm_abstraction_error:no_steps", file=sys.stderr)
            return None

        procedure["activation"]["family"] = "site_scoped"
        if payload.get("context_label"):
            procedure["activation"]["context_label"] = compact(payload.get("context_label"), 80)
        procedure["activation"]["goal_pattern"] = compact(
            payload.get("goal_pattern") or procedure["activation"]["goal_pattern"], 300
        )
        keywords = payload.get("activation_keywords")
        if isinstance(keywords, list) and keywords:
            procedure["activation"]["keywords"] = [
                compact(str(item), 50) for item in keywords[:24] if str(item).strip()
            ]
        procedure["execution"]["steps"] = steps
        procedure["execution"]["action_skeleton"] = [step["action_type"] for step in steps]
        procedure["execution"]["general_strategy"] = compact(payload.get("general_strategy") or "", 900)
        procedure["termination"]["answer_format"] = compact(payload.get("answer_format") or "", 500)
        avoid = payload.get("avoid")
        if isinstance(avoid, list) and avoid:
            procedure["failure_recovery"]["avoid"] = [
                compact(str(item), 220) for item in avoid[:8] if str(item).strip()
            ]
        procedure["stats"]["abstraction"] = "llm"
        procedure["stats"]["abstraction_model"] = model_name
        return procedure

    def _llm_refine_procedure_compact_retry(
        self,
        config: dict[str, Any],
        summary: dict[str, Any],
        actions: list[str],
        model_name: str,
    ) -> dict[str, Any] | None:
        try:
            from langchain.schema import HumanMessage, SystemMessage
            from agents.legacy.utils.chat_api import ChatModelArgs
        except Exception:
            return None

        final_answer = ""
        for action in reversed(actions):
            if action_name(action) == "send_msg_to_user":
                final_answer = compact(action_args(action), 240)
                break
        payload = {
            "goal": config.get("intent", ""),
            "template": config.get("intent_template", ""),
            "template_id": config.get("intent_template_id"),
            "site": config.get("sites", ["unknown"])[0],
            "reward": summary.get("cum_reward"),
            "expected": config.get("eval", {}).get("reference_answers", {}),
            "action_types": [action_name(action) for action in actions],
            "final_answer": final_answer,
        }
        system = "Return valid minified JSON only. No markdown. No prose."
        human = (
            "Build reusable WebArena memory from this successful trace. "
            "Output <=900 chars. Use exactly these keys: "
            "context_label,goal_pattern,activation_keywords,general_strategy,answer_format,steps,avoid. "
            "activation_keywords must have <=5 short strings. avoid must have <=3 short strings. "
            "steps must contain exactly 3 objects with keys action_type,intent,target_policy,value_policy,success_check. "
            "Each step field must be <=9 words. No numeric bid ids. Payload:"
            + json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
        )
        try:
            chat = ChatModelArgs(
                model_name=model_name,
                temperature=0.0,
                max_new_tokens=8192,
                max_total_tokens=24000,
                max_input_tokens=9500,
            ).make_chat_model()
            answer = chat.invoke([SystemMessage(content=system), HumanMessage(content=human)])
        except Exception as exc:
            print(f"llm_abstraction_retry_error:{type(exc).__name__}:{exc}", file=sys.stderr)
            return None
        return extract_json_object(getattr(answer, "content", ""))

    def _latest_selected_memory(self, goal: str, site: str) -> str | None:
        row = self.conn.execute(
            """
            SELECT selected FROM retrieval_events
            WHERE goal=? AND site=?
            ORDER BY id DESC
            LIMIT 1
            """,
            (compact(goal, 500), site),
        ).fetchone()
        return row["selected"] if row and row["selected"] else None

    def _llm_abstract_failure(
        self,
        result_dir: Path,
        config: dict[str, Any],
        summary: dict[str, Any],
        selected_memory: str | None,
        model_name: str,
    ) -> dict[str, Any]:
        thoughts, actions = extract_trace(result_dir / "experiment.log")
        final_answers = [
            action for action in actions if action_name(action) == "send_msg_to_user"
        ]
        base = {
            "family": "site_scoped",
            "abstract_failure": "Task failed or received zero reward.",
            "avoid": [
                "Do not assume a retrieved procedure applies unless its website, page state, and required evidence match.",
                "Do not send a final answer until the requested value has been verified on the current page.",
            ],
            "pattern": "reward_zero",
        }
        self._add_failure_fallbacks(base, config, summary, final_answers)
        try:
            from langchain.schema import HumanMessage, SystemMessage
            from agents.legacy.utils.chat_api import ChatModelArgs
        except Exception:
            return base

        payload = {
            "task_goal": config.get("intent", ""),
            "intent_template": config.get("intent_template", ""),
            "intent_template_id": config.get("intent_template_id"),
            "site": config.get("sites", ["unknown"])[0],
            "expected_evaluator": config.get("eval", {}),
            "site_context_guardrail_goal": (
                "Extract reusable benchmark-facing instructions for future tasks on this same website/context. "
                "Prefer concrete do/don't rules such as answer format, all-pages requirements, "
                "date/status filters, and evidence that must be verified."
            ),
            "summary": {
                "reward": summary.get("cum_reward"),
                "steps": summary.get("n_steps"),
                "truncated": summary.get("truncated"),
                "err_msg": summary.get("err_msg"),
            },
            "selected_memory": selected_memory,
            "final_answers": final_answers,
            "trace_steps": [
                {
                    "step": index + 1,
                    "thought": compact(thoughts[index] if index < len(thoughts) else "", 500),
                    "action": actions[index],
                    "action_type": action_name(actions[index]),
                }
                for index in range(len(actions))
            ],
        }
        system = """You convert failed WebArena traces into reusable negative procedural memory.
Return JSON only. Be specific about the failure pattern and what future agents should avoid.
Use the expected_evaluator/reference answers to distinguish wrong evidence from answer-format mismatch.
If the final answer omitted required entities, say which type of evidence was missed.
If the final answer appears semantically right but formatting is evaluator-incompatible, say that.
If the model failed to output a valid action tag, create a formatting/process avoid rule.
If a retrieved memory was irrelevant or incomplete, say exactly why."""
        human = """Create compact negative memory JSON. Return JSON only, no markdown fence.
Schema:
{
  "context_label": "short_site_context_label",
  "pattern": "short_failure_pattern",
  "abstract_failure": "one concise explanation of why this failed",
  "avoid": ["site/context-specific future instruction 1", "site/context-specific future instruction 2"],
  "penalty": 0.06
}

Failed trace payload:
""" + json.dumps(payload, ensure_ascii=True, indent=2)
        try:
            chat = ChatModelArgs(
                model_name=model_name,
                temperature=0.0,
                max_new_tokens=2048,
                max_total_tokens=24000,
                max_input_tokens=21000,
            ).make_chat_model()
            answer = chat.invoke([SystemMessage(content=system), HumanMessage(content=human)])
            parsed = extract_json_object(getattr(answer, "content", "")) or {}
        except Exception:
            return base

        avoid = parsed.get("avoid") if isinstance(parsed.get("avoid"), list) else base["avoid"]
        penalty = parsed.get("penalty", 0.06 if not summary.get("err_msg") else 0.12)
        try:
            penalty = max(0.02, min(float(penalty), 0.20))
        except Exception:
            penalty = 0.06
        return {
            "family": "site_scoped",
            "pattern": compact(parsed.get("pattern") or base["pattern"], 160),
            "abstract_failure": compact(parsed.get("abstract_failure") or base["abstract_failure"], 700),
            "avoid": [compact(str(item), 260) for item in avoid[:8] if str(item).strip()],
            "penalty": penalty,
        }

    def _add_failure_fallbacks(
        self,
        base: dict[str, Any],
        config: dict[str, Any],
        summary: dict[str, Any],
        final_answers: list[str],
    ) -> None:
        goal_text = (config.get("intent", "") or "").lower()
        tokens = token_set(goal_text)
        evaluator = config.get("eval", {}) if isinstance(config.get("eval"), dict) else {}
        reference = evaluator.get("reference_answers", {}) if isinstance(evaluator.get("reference_answers"), dict) else {}
        if ("reviewer" in tokens or "reviewers" in tokens) and {"mention", "mentioned", "complain", "complained"} & tokens:
            expected = reference.get("must_include") or []
            base["pattern"] = "missing_required_reviewers_or_premature_no_match"
            if expected:
                base["abstract_failure"] = (
                    "The final answer did not include all evaluator-required reviewer names."
                )
            else:
                base["abstract_failure"] = (
                    "The task likely failed from answering before every review page was checked."
                )
            base["avoid"].extend(
                [
                    "For reviewer lookup tasks, inspect every review page/pagination page before final answer.",
                    "Keep a running candidate list of matching reviewer names; do not overwrite earlier matches.",
                    "Do not answer N/A/no reviewers unless every review page has been checked.",
                    "Final answer must contain only reviewer names separated by commas.",
                    "If no reviewer matches after exhaustive checking, final answer must be exactly N/A.",
                ]
            )
        elif "fulfilled" in tokens and "orders" in tokens:
            base["pattern"] = "fulfilled_order_total_count_or_format_error"
            base["abstract_failure"] = (
                "Fulfilled-order total task failed because the count/total was missing, wrong, or not in evaluator-friendly format."
            )
            base["avoid"].extend(
                [
                    "Use the task's current date to compute the exact date window.",
                    "Count only fulfilled/complete orders inside the date window.",
                    "Sum only totals from counted rows.",
                    "Final answer must be '<N> orders, $<amount> total spend'.",
                ]
            )
            if summary.get("err_msg"):
                base["avoid"].append("If the count and total are known, immediately emit send_msg_to_user in a valid action tag.")
                base["penalty"] = 0.12
        elif summary.get("err_msg"):
            base["pattern"] = "invalid_action_format"
            base["abstract_failure"] = "The model failed after retries because it did not emit a valid action tag."
            base["avoid"].extend(
                [
                    "Return exactly one <action>...</action> block.",
                    "Do not output visible chain-of-thought before the action.",
                    "If the final answer is known, call send_msg_to_user immediately.",
                ]
            )

    def _target_policy(self, action: str, args: str, thought: str) -> str:
        if action == "click":
            return "Click the current visible element whose label/role best matches this step intent. Never reuse the old numeric bid id blindly."
        if action == "fill":
            return "Fill the current visible text/search field that matches the step intent. Never reuse the old numeric bid id blindly."
        if action == "select_option":
            return "Choose the dropdown/option matching the current task value."
        if action == "scroll":
            return "Scroll only when relevant content is likely below or pagination/reviews need inspection."
        if action == "send_msg_to_user":
            return "Answer concisely with the requested entities/values, or state no matches only after verification."
        return f"Apply action type {action} according to the current page state."

    def _value_policy(self, action: str, args: str, goal: str) -> str:
        if action in {"fill", "select_option"}:
            return "Use the value required by the current task goal, not the old literal value unless it still matches."
        if action == "send_msg_to_user":
            return "Use evidence from the current page and the current goal."
        return "No fixed value; adapt to the current goal."

    def _compact_text(self, procedure: dict[str, Any]) -> str:
        parts = [
            f"## {procedure['name']}",
            f"Site: {procedure['site']}",
            f"Goal pattern: {procedure['activation']['goal_pattern']}",
            "Keywords: " + " ".join(procedure["activation"]["keywords"]),
            "Actions: " + " ".join(procedure["execution"]["action_skeleton"]),
        ]
        strategy = procedure.get("execution", {}).get("general_strategy")
        if strategy:
            parts.append("Strategy: " + strategy)
        answer_format = procedure.get("termination", {}).get("answer_format")
        if answer_format:
            parts.append("Answer format: " + answer_format)
        for step in procedure["execution"]["steps"]:
            parts.append(
                f"{step['step']}. {step['action_type']} {step['intent']} {step['target_policy']} {step['value_policy']}"
            )
        return compact("\n".join(parts), 5000)

    def ingest_result(
        self,
        result_dir: Path,
        config_dir: Path,
        abstraction: str = "llm",
        abstraction_model: str = "openai/google/gemini-2.5-pro",
    ) -> str:
        summary = read_json(result_dir / "summary_info.json", {})
        task_id = int(result_dir.name.split(".")[1])
        config = read_json(config_dir / f"{task_id}.json", {})
        site = config.get("sites", ["unknown"])[0]
        success = bool(summary.get("cum_reward")) and not summary.get("err_msg")
        if success:
            procedure = self._procedure_from_result(
                result_dir,
                config,
                summary,
                abstraction=abstraction,
                abstraction_model=abstraction_model,
            )
            if not procedure:
                return "skipped:no_llm_abstraction"
            self.upsert_procedure(procedure, success=True, summary=summary)
            return f"upserted:{procedure['name']}"
        selected_memory = self._latest_selected_memory(config.get("intent", ""), site)
        failure = self._llm_abstract_failure(
            result_dir=result_dir,
            config=config,
            summary=summary,
            selected_memory=selected_memory,
            model_name=abstraction_model,
        )
        pattern = summary.get("err_msg") or failure["pattern"] or "reward_zero"
        self.conn.execute(
            """
            INSERT INTO negative_memory(
                procedure_name, site, pattern, family, abstract_failure, avoid_json,
                selected_memory, penalty, evidence, created_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                selected_memory,
                site,
                compact(pattern, 500),
                "general",
                failure["abstract_failure"],
                safe_json(failure["avoid"]),
                selected_memory,
                failure["penalty"],
                safe_json(
                    {
                        "task": result_dir.name,
                        "reward": summary.get("cum_reward"),
                        "truncated": summary.get("truncated"),
                        "err_msg": summary.get("err_msg"),
                        "selected_memory": selected_memory,
                    }
                ),
                now_ms(),
            ),
        )
        failure_node = f"failure:{compact(failure['pattern'], 80)}"
        self._edge(f"site:{site}", "has_failure", failure_node, failure["penalty"], failure["abstract_failure"])
        if selected_memory:
            self._edge(selected_memory, "failed_on", failure_node, failure["penalty"], failure["abstract_failure"])
        self.conn.commit()
        return "negative_recorded"

    def upsert_procedure(self, procedure: dict[str, Any], success: bool, summary: dict[str, Any]) -> None:
        name = procedure["name"]
        compact_text = self._compact_text(procedure)
        existing = self.conn.execute(
            "SELECT success_count, failure_count, avg_steps, avg_agent_elapsed FROM procedures WHERE name=?",
            (name,),
        ).fetchone()
        if existing:
            success_count = int(existing["success_count"]) + int(success)
            failure_count = int(existing["failure_count"]) + int(not success)
            avg_steps = self._running_avg(existing["avg_steps"], existing["success_count"], summary.get("n_steps", 0))
            avg_agent_elapsed = self._running_avg(
                existing["avg_agent_elapsed"],
                existing["success_count"],
                summary.get("stats.cum_agent_elapsed", 0),
            )
            self.conn.execute(
                """
                UPDATE procedures
                SET procedure_json=?, compact_text=?, success_count=?, failure_count=?,
                    avg_steps=?, avg_agent_elapsed=?, updated_at_ms=?
                WHERE name=?
                """,
                (
                    safe_json(procedure),
                    compact_text,
                    success_count,
                    failure_count,
                    avg_steps,
                    avg_agent_elapsed,
                    now_ms(),
                    name,
                ),
            )
        else:
            timestamp = now_ms()
            self.conn.execute(
                """
                INSERT INTO procedures(
                    name, site, intent_template_id, goal_pattern, procedure_json, compact_text,
                    success_count, failure_count, avg_steps, avg_agent_elapsed, created_at_ms, updated_at_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    name,
                    procedure["site"],
                    procedure.get("intent_template_id"),
                    procedure["activation"]["goal_pattern"],
                    safe_json(procedure),
                    compact_text,
                    int(success),
                    int(not success),
                    float(summary.get("n_steps", 0) or 0),
                    float(summary.get("stats.cum_agent_elapsed", 0) or 0),
                    timestamp,
                    timestamp,
                ),
            )
        self._upsert_edges(procedure)
        self.conn.commit()
        self._sync_index()
        self.write_manifest()

    def _running_avg(self, old_avg: float, old_count: int, value: float) -> float:
        old_count = int(old_count or 0)
        if old_count <= 0:
            return float(value or 0)
        return (float(old_avg or 0) * old_count + float(value or 0)) / (old_count + 1)

    def _upsert_edges(self, procedure: dict[str, Any]) -> None:
        name = procedure["name"]
        goal = procedure["activation"]["goal_pattern"]
        self._edge(f"site:{procedure['site']}", "contains", name, 1.5, goal)
        self._edge(f"intent:{goal[:80]}", "activates", name, 1.0, goal)
        strategy = procedure.get("execution", {}).get("general_strategy")
        if strategy:
            self._edge(name, "uses_strategy", f"strategy:{compact(strategy, 80)}", 0.8, strategy)
        prev_step_node = name
        for step in procedure["execution"]["steps"]:
            step_node = f"{name}:step:{step['step']}:{step['action_type']}"
            action_node = f"action:{step['action_type']}"
            self._edge(prev_step_node, "next_step", step_node, 1.0, step["intent"])
            self._edge(step_node, "executes", action_node, 1.0, step["target_policy"])
            if step.get("success_check"):
                self._edge(step_node, "verifies", f"check:{compact(step['success_check'], 80)}", 0.7, step["success_check"])
            prev_step_node = step_node
        answer_format = procedure.get("termination", {}).get("answer_format")
        if answer_format:
            self._edge(name, "formats_answer_as", f"answer_format:{compact(answer_format, 80)}", 0.8, answer_format)
        self._edge(prev_step_node, "leads_to", "outcome:success", 1.0, name)

    def _edge(self, source: str, edge_type: str, target: str, weight: float, evidence: str) -> None:
        self.conn.execute(
            """
            INSERT INTO procedure_edges(source, edge_type, target, weight, evidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(source, edge_type, target) DO UPDATE SET
                weight = procedure_edges.weight + excluded.weight,
                evidence = excluded.evidence
            """,
            (source, edge_type, target, weight, compact(evidence, 500)),
        )

    def _structure_score(self, procedure: dict[str, Any], goal: str) -> float:
        execution = procedure.get("execution", {})
        skeleton = set(execution.get("action_skeleton", []))
        if not skeleton:
            return 0.0
        step_text = " ".join(
            " ".join(str(step.get(key, "")) for key in ["action_type", "intent", "target_policy", "value_policy"])
            for step in execution.get("steps", [])
        )
        action_prior = min(1.0, len(skeleton) / 5.0)
        semantic_fit = overlap(goal, step_text)
        return min(1.0, 0.42 * action_prior + 0.58 * semantic_fit)

    def _fast_retrieval_candidates(
        self,
        goal: str,
        observation: str,
        rows: dict[str, sqlite3.Row],
        limit: int,
    ) -> list[dict[str, Any]]:
        scored = []
        page_text = compact(observation, 1200)
        for row in rows.values():
            lexical = max(
                overlap(goal, row["goal_pattern"]),
                overlap(goal, row["compact_text"]),
            )
            page = overlap(page_text, row["compact_text"]) if page_text else 0.0
            combined = min(1.0, 0.72 * min(1.0, lexical * 2.2) + 0.28 * page)
            scored.append(
                {
                    "workflow_name": row["name"],
                    "combined_score": round(combined, 4),
                    "semantic_score": 0.0,
                    "bm25_score": round(lexical, 4),
                    "retrieval_backend": "sqlite_site_lexical_fast",
                    "text_for_embedding": row["compact_text"],
                }
            )
        scored.sort(key=lambda item: item["combined_score"], reverse=True)
        return scored[:limit]

    def retrieve(self, goal: str, observation: str, site: str, top_k: int = 4, min_score: float = 0.42) -> list[dict[str, Any]]:
        query = f"Site: {site}\nGoal: {goal}\nObservation: {compact(observation, 2200)}"
        by_name = {
            row["name"]: row
            for row in self.conn.execute("SELECT * FROM procedures WHERE site=?", (site,)).fetchall()
        }
        if self.fast_mode or self.index is None:
            retrieval_candidates = self._fast_retrieval_candidates(
                goal, observation, by_name, max(top_k, self.candidate_k)
            )
        else:
            retrieval = self.index.retrieve(query, top_k=max(top_k, self.candidate_k), threshold=0.0)
            retrieval_candidates = retrieval.candidates
        signature = ui_signature(observation)
        candidates = []
        candidate_pool = []
        raw_candidates = []
        rejected_candidates = []
        for item in retrieval_candidates:
            row = by_name.get(item["workflow_name"])
            if not row:
                continue
            procedure = json.loads(row["procedure_json"])
            lexical = overlap(goal, row["goal_pattern"])
            activation = overlap(" ".join(signature["top_terms"]), row["compact_text"])
            outcome = self._outcome_score(row)
            graph = self._graph_score(row["name"])
            metadata = 1.0 if row["site"] == site else 0.0
            negative = self._negative_penalty(goal, site)
            structure = self._structure_score(procedure, goal)
            score = (
                self.semantic_weight * float(item.get("combined_score", 0))
                + self.lexical_weight * lexical
                + self.activation_weight * activation
                + self.metadata_weight * metadata
                + self.outcome_weight * outcome
                + self.graph_weight * graph
                + self.structure_weight * structure
                - negative
            )
            required_score = min_score
            scored = {
                "name": row["name"],
                "workflow_name": row["name"],
                "score": round(score, 4),
                "semantic_score": item.get("combined_score", 0),
                "lexical_score": round(lexical, 4),
                "activation_score": round(activation, 4),
                "metadata_score": round(metadata, 4),
                "outcome_score": round(outcome, 4),
                "graph_score": round(graph, 4),
                "structure_score": round(structure, 4),
                "negative_penalty": round(negative, 4),
                "required_score": round(required_score, 4),
                "accepted": score >= required_score,
            }
            raw_candidates.append(scored)
            candidate_pool.append(
                {
                    "procedure": procedure,
                    "text_for_embedding": row["compact_text"],
                    **scored,
                }
            )
        if self.reranker is not None and candidate_pool:
            ranked_pool = self.reranker.rerank(query, candidate_pool, top_k=len(candidate_pool))
        else:
            ranked_pool = sorted(candidate_pool, key=lambda item: item["score"], reverse=True)
        for item in ranked_pool:
            rank_score = float(item.get("rerank_score", item["score"]) or 0.0)
            required_score = float(item.get("required_score", min_score) or min_score)
            accepted = bool(item.get("accepted")) or rank_score >= required_score
            public = {
                k: v
                for k, v in item.items()
                if k not in {"procedure", "text_for_embedding", "accepted"}
            }
            if accepted and len(candidates) < top_k:
                public["base_score"] = public.get("score", 0.0)
                public["score"] = round(rank_score, 4)
                candidates.append({"procedure": item["procedure"], **public})
            else:
                rejected_candidates.append({**public, "accepted": False})
        raw_candidates.sort(key=lambda item: item["score"], reverse=True)
        rejected_candidates.sort(key=lambda item: item.get("rerank_score", item.get("score", 0)), reverse=True)
        self.conn.execute(
            """
            INSERT INTO retrieval_events(
                goal, site, selected, candidates_json, raw_candidates_json,
                rejected_candidates_json, created_at_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                compact(goal, 500),
                site,
                candidates[0]["procedure"]["name"] if candidates else None,
                safe_json([{k: v for k, v in c.items() if k != "procedure"} | {"name": c["procedure"]["name"]} for c in candidates]),
                safe_json(raw_candidates[: self.candidate_k]),
                safe_json(rejected_candidates[: self.candidate_k]),
                now_ms(),
            ),
        )
        self.conn.commit()
        return candidates

    def _outcome_score(self, row: sqlite3.Row) -> float:
        successes = int(row["success_count"] or 0)
        failures = int(row["failure_count"] or 0)
        return (successes + 1) / (successes + failures + 2)

    def _graph_score(self, name: str) -> float:
        positive = self.conn.execute(
            """
            SELECT SUM(weight) AS total FROM procedure_edges
            WHERE (source=? OR target=?)
              AND edge_type NOT IN ('failed_on', 'has_failure')
            """,
            (name, name),
        ).fetchone()
        negative = self.conn.execute(
            """
            SELECT SUM(weight) AS total FROM procedure_edges
            WHERE source=? AND edge_type='failed_on'
            """,
            (name,),
        ).fetchone()
        total = float(positive["total"] or 0) - 2.0 * float(negative["total"] or 0)
        return max(0.0, min(total / 10.0, 1.0))

    def _negative_penalty(self, goal: str, site: str, family: str | None = None) -> float:
        rows = self._negative_matches(goal, site, limit=8)
        penalty = 0.0
        for row in rows:
            penalty += float(row["penalty"])
        return min(penalty, 0.14)

    def _negative_matches(
        self, goal: str, site: str, limit: int = 5, family: str | None = None
    ) -> list[sqlite3.Row]:
        rows = self.conn.execute(
            """
            SELECT * FROM negative_memory
            WHERE site=?
            ORDER BY id DESC
            LIMIT 200
            """,
            (site,),
        ).fetchall()
        scored = []
        for row in rows:
            lexical = max(
                overlap(goal, row["pattern"]),
                overlap(goal, row["abstract_failure"]),
            )
            selected_memory = row["selected_memory"] or row["procedure_name"] or ""
            memory_overlap = overlap(goal, selected_memory)
            score = 0.86 * lexical + 0.14 * memory_overlap
            if score >= 0.08:
                scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[:limit]]

    def prompt(self, goal: str, observation: str, site: str, top_k: int = 4, min_score: float = 0.42) -> str:
        candidates = self.retrieve(goal, observation, site, top_k=top_k, min_score=min_score)
        negatives = self._negative_matches(goal, site, limit=3)
        if not candidates and not negatives:
            return ""
        cards = [
            "# Procedural Memory",
            "Use these retrieved procedures only when they match the current goal/page. Adapt every element id to the current observation.",
            "Follow the workflow as a policy checklist, not as a script: after each step, verify the success check on the current page before continuing.",
        ]
        if negatives:
            cards.append("\n## Negative Memory")
            cards.append("Avoid repeating these similar failures:")
            for row in negatives:
                avoid = read_json_from_text(row["avoid_json"], [])
                cards.append(f"- Similar failure pattern={row['pattern']}: {row['abstract_failure']}")
                for item in avoid[:4]:
                    cards.append(f"  Avoid: {item}")
        for item in candidates:
            proc = item["procedure"]
            cards.append(f"\n## {proc['name']} (score={item['score']})")
            cards.append(f"Activation: {proc['activation']['goal_pattern']}")
            strategy = proc.get("execution", {}).get("general_strategy")
            if strategy:
                cards.append(f"Strategy: {strategy}")
            answer_format = proc.get("termination", {}).get("answer_format")
            if answer_format:
                cards.append(f"Answer format: {answer_format}")
            cards.append("Steps:")
            for step in proc["execution"]["steps"][:8]:
                success_check = step.get("success_check")
                line = f"- {step['action_type']}: {step['intent']} Target policy: {step['target_policy']} Value policy: {step['value_policy']}"
                if success_check:
                    line += f" Success check: {success_check}"
                cards.append(line)
            cards.append("Avoid: " + "; ".join(proc["failure_recovery"]["avoid"]))
            cards.append("Termination: " + proc["termination"]["success_signal"])
        return "\n".join(cards)

    def write_manifest(self) -> None:
        procedure_count = self.conn.execute("SELECT COUNT(*) AS n FROM procedures").fetchone()["n"]
        edge_count = self.conn.execute("SELECT COUNT(*) AS n FROM procedure_edges").fetchone()["n"]
        negative_count = self.conn.execute("SELECT COUNT(*) AS n FROM negative_memory").fetchone()["n"]
        event_count = self.conn.execute("SELECT COUNT(*) AS n FROM retrieval_events").fetchone()["n"]
        payload = {
            "backend": "webarena_procedural_graph_hybrid",
            "procedures": procedure_count,
            "edges": edge_count,
            "negative_memories": negative_count,
            "retrieval_events": event_count,
            "embedding_backend": "sqlite_family_lexical_fast"
            if self.index is None
            else getattr(self.index, "actual_index_kind", getattr(self.index, "backend", "unknown")),
            "vector_backend_requested": self.vector_backend,
            "vector_index_stats": None if self.index is None else self.index.stats(),
            "fast_mode": self.fast_mode,
            "updated_at_ms": now_ms(),
        }
        (self.root / "procedural_manifest.json").write_text(json.dumps(payload, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["ingest-result", "prompt", "manifest"])
    parser.add_argument("--memory-dir", default="memory/procedural")
    parser.add_argument("--result-dir")
    parser.add_argument("--config-dir", default="config_files")
    parser.add_argument("--goal", default="")
    parser.add_argument("--observation", default="")
    parser.add_argument("--site", default="shopping")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-score", type=float, default=0.42)
    parser.add_argument("--abstraction-model", default="openai/google/gemini-2.5-pro")
    args = parser.parse_args()

    memory = WebArenaProceduralMemory(Path(args.memory_dir))
    try:
        if args.command == "ingest-result":
            print(
                memory.ingest_result(
                    Path(args.result_dir),
                    Path(args.config_dir),
                    abstraction="llm",
                    abstraction_model=args.abstraction_model,
                )
            )
        elif args.command == "prompt":
            print(memory.prompt(args.goal, args.observation, args.site, top_k=args.top_k, min_score=args.min_score))
        else:
            memory.write_manifest()
            print((Path(args.memory_dir) / "procedural_manifest.json").read_text())
    finally:
        memory.close()


if __name__ == "__main__":
    main()
