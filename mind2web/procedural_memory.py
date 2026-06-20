"""Typed procedural memory for Mind2Web/WebArena agents.

This module upgrades plain workflow retrieval into a small procedural-memory
system:

- workflows are converted into compact, typed procedures;
- procedures are stored durably in SQLite with graph edges and outcomes;
- dense/BM25 retrieval is reused from the existing AWM index;
- retrieved candidates are reranked with metadata, UI-state, graph, and
  outcome signals;
- prompts receive compact activation/execution/termination records instead of
  long raw trajectories.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import json
import re
import sqlite3
import time
from typing import Any

try:
    from webarena.local_awm_full_demo import RetrievalResult, WorkflowEmbeddingIndex
except ModuleNotFoundError:  # llm_step_eval.py runs with webarena/ directly on sys.path.
    from local_awm_full_demo import RetrievalResult, WorkflowEmbeddingIndex


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def now_ms() -> int:
    return int(time.time() * 1000)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def token_set(text: str) -> set[str]:
    return set(tokenize(text))


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def compact_text(text: str, limit: int = 900) -> str:
    text = re.sub(r"<[^>]+>", " ", text or "")
    text = re.sub(r"[\[\]{}()\"'=<>/]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


def top_terms(text: str, limit: int = 32) -> list[str]:
    stop = {
        "the",
        "and",
        "for",
        "with",
        "this",
        "that",
        "from",
        "into",
        "task",
        "click",
        "type",
        "select",
        "button",
        "input",
        "workflow",
        "element",
        "visible",
        "current",
    }
    counts = Counter(token for token in tokenize(text) if len(token) > 2 and token not in stop)
    return [term for term, _ in counts.most_common(limit)]


def ui_signature(observation: str) -> dict[str, Any]:
    """Extract a compact signature from the current DOM-like observation."""
    text = compact_text(observation, limit=2000)
    ids = re.findall(r"\bid\s*=?\s*(\d+)", observation or "", flags=re.IGNORECASE)
    tags = re.findall(r"<\s*([a-zA-Z][\w:-]*)", observation or "")
    controls = []
    for pattern in [
        r"<[^>]*(?:button|link|menuitem|tab|textbox|searchbox|combobox|option|select|input)[^>]*>",
        r"\b(?:button|link|menuitem|tab|textbox|searchbox|combobox|option|select|input)\b",
    ]:
        controls.extend(re.findall(pattern, observation or "", flags=re.IGNORECASE)[:40])
    return {
        "text": text,
        "tokens": sorted(token_set(text))[:256],
        "top_terms": top_terms(text, limit=40),
        "ids_count": len(ids),
        "tags": dict(Counter(tag.lower() for tag in tags).most_common(20)),
        "controls": [compact_text(item, limit=120) for item in controls[:40]],
    }


def safe_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, sort_keys=True)


def parse_json(text: str | None, default: Any) -> Any:
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


@dataclass
class ProcedureCandidate:
    procedure: dict[str, Any]
    candidate: dict[str, Any]


class ProceduralMemoryStore:
    """SQLite-backed procedural memory with hybrid retrieval and graph rerank."""

    semantic_weight = 0.50
    lexical_weight = 0.14
    activation_weight = 0.16
    graph_weight = 0.08
    outcome_weight = 0.08
    metadata_weight = 0.04
    acceptance_threshold = 0.34
    activation_threshold = 0.04

    def __init__(self, root: Path, candidate_k: int = 50) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "procedural_memory.sqlite3"
        self.index = WorkflowEmbeddingIndex(self.root / "procedure_embeddings.json")
        self.candidate_k = candidate_k
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_db()
        self._load_index_from_db_if_needed()

    @property
    def backend(self) -> str:
        return "procedural_graph_hybrid"

    def close(self) -> None:
        self.conn.close()

    def _init_db(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS procedures (
                name TEXT PRIMARY KEY,
                website TEXT,
                domain TEXT,
                subdomain TEXT,
                goal_pattern TEXT,
                procedure_json TEXT NOT NULL,
                compact_text TEXT NOT NULL,
                success_count INTEGER NOT NULL DEFAULT 0,
                failure_count INTEGER NOT NULL DEFAULT 0,
                avg_steps REAL NOT NULL DEFAULT 0.0,
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
                procedure_name TEXT NOT NULL,
                pattern TEXT NOT NULL,
                penalty REAL NOT NULL DEFAULT 0.08,
                evidence TEXT NOT NULL DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_procedures_site
                ON procedures(website, domain, subdomain);
            CREATE INDEX IF NOT EXISTS idx_edges_source
                ON procedure_edges(source);
            CREATE INDEX IF NOT EXISTS idx_edges_target
                ON procedure_edges(target);
            """
        )
        self.conn.commit()

    def _load_index_from_db_if_needed(self) -> None:
        if self.index.entries:
            return
        rows = self.conn.execute(
            "SELECT compact_text FROM procedures ORDER BY created_at_ms, name"
        ).fetchall()
        if rows:
            self.index.rebuild([row["compact_text"] for row in rows])

    def workflow_to_procedure(self, workflow: dict[str, Any]) -> dict[str, Any]:
        steps = []
        operation_counts: Counter[str] = Counter()
        target_text_parts = []
        source_steps = workflow.get("steps", [])
        for index, step in enumerate(source_steps, start=1):
            operation = str(step.get("operation") or step.get("op") or "").upper()
            operation_counts[operation] += 1
            target = (
                step.get("target_description")
                or step.get("target_label")
                or step.get("label")
                or step.get("description")
                or step.get("role")
                or ""
            )
            target_type = step.get("target_type") or step.get("target_role") or step.get("role") or ""
            value_policy = step.get("value_policy")
            if not value_policy:
                value_policy = (
                    "Use the task-specific value."
                    if operation in {"TYPE", "SELECT"} and step.get("value")
                    else "No value required."
                )
            steps.append(
                {
                    "step": index,
                    "intent": step.get("step_intent") or f"{operation.lower()} {target}".strip(),
                    "operation": operation,
                    "target_type": target_type,
                    "target": target,
                    "value_policy": value_policy,
                    "selection_rule": step.get("selection_rule")
                    or "Choose the current visible element whose role, label, or nearby text best matches the target.",
                    "success_signal": step.get("success_signal")
                    or "The page advances or the target field reflects the intended value.",
                }
            )
            target_text_parts.append(f"{operation} {target_type} {target} {value_policy}")

        first_observation = ""
        for step in workflow.get("evidence", []) + source_steps:
            first_observation = step.get("observation", "")
            if first_observation:
                break

        applicability = workflow.get("applicability", {})
        when_to_use = ""
        when_not_to_use = ""
        if isinstance(applicability, dict):
            when_to_use = str(applicability.get("when_to_use", ""))
            when_not_to_use = str(applicability.get("when_not_to_use", ""))

        goal_pattern = str(workflow.get("goal_pattern") or workflow.get("task") or "")
        activation_text = " ".join(
            [
                str(workflow.get("website", "")),
                str(workflow.get("domain", "")),
                str(workflow.get("subdomain", "")),
                goal_pattern,
                when_to_use,
                compact_text(first_observation, limit=900),
                " ".join(target_text_parts),
            ]
        )
        ui_terms = top_terms(activation_text, limit=48)
        operations = [op for op, _ in operation_counts.most_common() if op]

        procedure = {
            "schema_version": 1,
            "name": workflow["name"],
            "source": workflow.get("source", "Mind2Web exemplar"),
            "memory_type": "typed_procedure",
            "website": workflow.get("website", ""),
            "domain": workflow.get("domain", ""),
            "subdomain": workflow.get("subdomain", ""),
            "goal_pattern": goal_pattern,
            "activation": {
                "when_to_use": when_to_use
                or "Use when the current task and page state match the goal pattern and expected UI targets.",
                "when_not_to_use": when_not_to_use
                or "Do not use when the current page lacks the expected controls or the task intent differs.",
                "ui_terms": ui_terms,
                "operations": operations,
            },
            "execution": {"steps": steps[:18]},
            "termination": {
                "success_check": "Stop when the task-specific goal is satisfied or the expected confirmation/result is visible.",
                "max_steps_hint": len(steps),
            },
            "failure_recovery": {
                "guards": [
                    "Never copy old element ids; choose only ids visible in the current observation.",
                    "If the expected UI state is absent, ignore this procedure and reason from the page.",
                    "If a click does not advance the page, re-check visible labels before retrying.",
                ],
                "negative_patterns": [],
            },
            "stats": {
                "success_count": 1,
                "failure_count": 0,
                "avg_steps": float(len(steps)),
            },
            "raw_workflow_name": workflow["name"],
        }
        return procedure

    def procedure_to_text(self, procedure: dict[str, Any]) -> str:
        activation = procedure.get("activation", {})
        execution = procedure.get("execution", {})
        lines = [
            f"## {procedure['name']}",
            "Type: typed_procedure",
            f"Website: {procedure.get('website', '')}",
            f"Domain: {procedure.get('domain', '')}",
            f"Subdomain: {procedure.get('subdomain', '')}",
            f"Goal pattern: {procedure.get('goal_pattern', '')}",
            f"Activation: {activation.get('when_to_use', '')}",
            f"Do not use: {activation.get('when_not_to_use', '')}",
            "UI terms: " + " ".join(activation.get("ui_terms", [])),
            "Operations: " + " ".join(activation.get("operations", [])),
        ]
        for step in execution.get("steps", []):
            lines.append(
                "Procedure step: {operation} intent={intent} target_type={target_type} "
                "target={target} value_policy={value_policy} selection_rule={selection_rule} "
                "success={success_signal}".format(**step)
            )
        lines.append(
            "Termination: "
            + str(procedure.get("termination", {}).get("success_check", ""))
        )
        return "\n".join(lines)

    def add_workflow(self, workflow: dict[str, Any], save: bool = False) -> dict[str, Any]:
        procedure = self.workflow_to_procedure(workflow)
        self.add_procedure(procedure, save=save)
        return procedure

    def add_procedure(self, procedure: dict[str, Any], save: bool = False) -> None:
        text = self.procedure_to_text(procedure)
        stats = procedure.get("stats", {})
        timestamp = now_ms()
        self.conn.execute(
            """
            INSERT OR REPLACE INTO procedures (
                name, website, domain, subdomain, goal_pattern, procedure_json,
                compact_text, success_count, failure_count, avg_steps,
                created_at_ms, updated_at_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, COALESCE(
                (SELECT created_at_ms FROM procedures WHERE name = ?), ?
            ), ?)
            """,
            (
                procedure["name"],
                procedure.get("website", ""),
                procedure.get("domain", ""),
                procedure.get("subdomain", ""),
                procedure.get("goal_pattern", ""),
                safe_json(procedure),
                text,
                int(stats.get("success_count", 1)),
                int(stats.get("failure_count", 0)),
                float(stats.get("avg_steps", len(procedure.get("execution", {}).get("steps", [])))),
                procedure["name"],
                timestamp,
                timestamp,
            ),
        )
        self._upsert_edges(procedure)
        self.conn.commit()
        if not any(entry.get("name") == procedure["name"] for entry in self.index.entries):
            self.index.add_workflow(text, save=save)

    def _upsert_edges(self, procedure: dict[str, Any]) -> None:
        name = procedure["name"]
        edges = []
        for key in ["website", "domain", "subdomain"]:
            value = str(procedure.get(key, "")).strip().lower()
            if value:
                edges.append((f"{key}:{value}", "indexes", name, 1.0, key))
                edges.append((name, f"has_{key}", f"{key}:{value}", 1.0, key))
        for term in procedure.get("activation", {}).get("ui_terms", [])[:32]:
            edges.append((f"ui:{term}", "activates", name, 0.45, "activation.ui_terms"))
            edges.append((name, "expects_ui", f"ui:{term}", 0.45, "activation.ui_terms"))
        for operation in procedure.get("activation", {}).get("operations", []):
            op_node = f"op:{operation.lower()}"
            edges.append((op_node, "used_by", name, 0.35, "activation.operations"))
            edges.append((name, "uses_op", op_node, 0.35, "activation.operations"))
        for source, edge_type, target, weight, evidence in edges:
            self.conn.execute(
                """
                INSERT INTO procedure_edges(source, edge_type, target, weight, evidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(source, edge_type, target) DO UPDATE SET
                    weight = MAX(weight, excluded.weight),
                    evidence = excluded.evidence
                """,
                (source, edge_type, target, float(weight), evidence),
            )

    def get(self, name: str | None) -> dict[str, Any] | None:
        if not name:
            return None
        row = self.conn.execute(
            "SELECT procedure_json, success_count, failure_count, avg_steps FROM procedures WHERE name = ?",
            (name,),
        ).fetchone()
        if row is None:
            return None
        procedure = parse_json(row["procedure_json"], {})
        procedure.setdefault("stats", {})
        procedure["stats"].update(
            {
                "success_count": row["success_count"],
                "failure_count": row["failure_count"],
                "avg_steps": row["avg_steps"],
            }
        )
        return procedure

    def to_dict(self) -> dict[str, dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT name, procedure_json FROM procedures ORDER BY created_at_ms, name"
        ).fetchall()
        return {row["name"]: parse_json(row["procedure_json"], {}) for row in rows}

    def procedures(self) -> list[dict[str, Any]]:
        return list(self.to_dict().values())

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        metadata: dict[str, str] | None = None,
        observation: str = "",
        previous_actions: list[str] | None = None,
        threshold: float | None = None,
    ) -> RetrievalResult:
        if threshold is None:
            threshold = self.acceptance_threshold
        previous_actions = previous_actions or []
        base = self.index.retrieve(query, top_k=max(top_k, self.candidate_k), threshold=-1.0)
        ui = ui_signature(observation)
        reranked = []
        for candidate in base.candidates:
            procedure = self.get(candidate.get("workflow_name"))
            if not procedure:
                continue
            enriched = self._score_candidate(
                query=query,
                candidate=candidate,
                procedure=procedure,
                metadata=metadata or {},
                ui=ui,
                previous_actions=previous_actions,
            )
            reranked.append(enriched)
        reranked.sort(key=lambda item: item["combined_score"], reverse=True)
        reranked = reranked[:top_k]
        if reranked and reranked[0]["combined_score"] >= threshold and reranked[0]["activation_score"] >= self.activation_threshold:
            return RetrievalResult(
                workflow_name=reranked[0]["workflow_name"],
                score=reranked[0]["combined_score"],
                candidates=reranked,
            )
        return RetrievalResult(workflow_name=None, score=0.0, candidates=reranked)

    def _score_candidate(
        self,
        query: str,
        candidate: dict[str, Any],
        procedure: dict[str, Any],
        metadata: dict[str, str],
        ui: dict[str, Any],
        previous_actions: list[str],
    ) -> dict[str, Any]:
        semantic = float(candidate.get("semantic_score", 0.0))
        lexical = float(candidate.get("bm25_score", 0.0))
        activation = self._activation_score(procedure, query, ui)
        graph = self._graph_score(procedure, metadata, ui)
        outcome = self._outcome_score(procedure)
        meta = self._metadata_score(procedure, metadata)
        negative = self._negative_penalty(procedure, query, ui, previous_actions)
        combined = (
            self.semantic_weight * semantic
            + self.lexical_weight * lexical
            + self.activation_weight * activation
            + self.graph_weight * graph
            + self.outcome_weight * outcome
            + self.metadata_weight * meta
            - negative
        )
        output = dict(candidate)
        output.update(
            {
                "workflow_name": procedure["name"],
                "combined_score": round(max(0.0, combined), 4),
                "semantic_score": round(semantic, 4),
                "bm25_score": round(lexical, 4),
                "activation_score": round(activation, 4),
                "graph_score": round(graph, 4),
                "outcome_score": round(outcome, 4),
                "metadata_score": round(meta, 4),
                "negative_penalty": round(negative, 4),
                "retrieval_backend": self.backend,
                "compact_procedure": prompt_procedure(procedure),
            }
        )
        return output

    def _activation_score(self, procedure: dict[str, Any], query: str, ui: dict[str, Any]) -> float:
        activation = procedure.get("activation", {})
        activation_text = " ".join(
            [
                str(procedure.get("goal_pattern", "")),
                str(activation.get("when_to_use", "")),
                " ".join(activation.get("ui_terms", [])),
                " ".join(step.get("target", "") for step in procedure.get("execution", {}).get("steps", [])),
            ]
        )
        query_match = lexical_overlap(query, activation_text)
        ui_match = lexical_overlap(" ".join(ui.get("top_terms", [])), activation_text)
        control_match = lexical_overlap(" ".join(ui.get("controls", [])), activation_text)
        return min(1.0, 0.45 * query_match + 0.40 * ui_match + 0.15 * control_match)

    def _graph_score(self, procedure: dict[str, Any], metadata: dict[str, str], ui: dict[str, Any]) -> float:
        sources = []
        for key in ["website", "domain", "subdomain"]:
            value = str(metadata.get(key, "")).strip().lower()
            if value:
                sources.append(f"{key}:{value}")
        sources.extend(f"ui:{term}" for term in ui.get("top_terms", [])[:20])
        if not sources:
            return 0.0
        placeholders = ",".join("?" for _ in sources)
        rows = self.conn.execute(
            f"""
            SELECT SUM(weight) AS score
            FROM procedure_edges
            WHERE source IN ({placeholders})
              AND target = ?
            """,
            (*sources, procedure["name"]),
        ).fetchone()
        raw = float(rows["score"] or 0.0)
        return min(1.0, raw / 3.0)

    def _outcome_score(self, procedure: dict[str, Any]) -> float:
        stats = procedure.get("stats", {})
        success = float(stats.get("success_count", 0))
        failure = float(stats.get("failure_count", 0))
        total = success + failure
        if total <= 0:
            return 0.5
        success_rate = success / total
        step_efficiency = 1.0 / max(1.0, float(stats.get("avg_steps", 8.0)) / 8.0)
        return min(1.0, 0.75 * success_rate + 0.25 * step_efficiency)

    def _metadata_score(self, procedure: dict[str, Any], metadata: dict[str, str]) -> float:
        if not metadata:
            return 0.0
        score = 0.0
        if procedure.get("website") == metadata.get("website"):
            score += 0.50
        if procedure.get("domain") == metadata.get("domain"):
            score += 0.25
        if procedure.get("subdomain") == metadata.get("subdomain"):
            score += 0.25
        return min(1.0, score)

    def _negative_penalty(
        self,
        procedure: dict[str, Any],
        query: str,
        ui: dict[str, Any],
        previous_actions: list[str],
    ) -> float:
        rows = self.conn.execute(
            "SELECT pattern, penalty FROM negative_memory WHERE procedure_name = ?",
            (procedure["name"],),
        ).fetchall()
        if not rows:
            return 0.0
        context = " ".join([query, " ".join(ui.get("top_terms", [])), " ".join(previous_actions)])
        penalty = 0.0
        for row in rows:
            if lexical_overlap(context, row["pattern"]) >= 0.15:
                penalty += float(row["penalty"])
        return min(0.35, penalty)

    def record_outcome(
        self,
        procedure_name: str,
        success: bool,
        steps: int | None = None,
        negative_pattern: str | None = None,
        evidence: str = "",
    ) -> None:
        row = self.conn.execute(
            "SELECT success_count, failure_count, avg_steps FROM procedures WHERE name = ?",
            (procedure_name,),
        ).fetchone()
        if row is None:
            return
        success_count = int(row["success_count"]) + int(success)
        failure_count = int(row["failure_count"]) + int(not success)
        avg_steps = float(row["avg_steps"])
        if steps is not None and success:
            avg_steps = ((avg_steps * max(0, success_count - 1)) + steps) / max(1, success_count)
        self.conn.execute(
            """
            UPDATE procedures
            SET success_count = ?, failure_count = ?, avg_steps = ?, updated_at_ms = ?
            WHERE name = ?
            """,
            (success_count, failure_count, avg_steps, now_ms(), procedure_name),
        )
        if negative_pattern and not success:
            self.conn.execute(
                """
                INSERT INTO negative_memory(procedure_name, pattern, penalty, evidence)
                VALUES (?, ?, ?, ?)
                """,
                (procedure_name, negative_pattern, 0.08, evidence),
            )
        self.conn.commit()

    def save(self) -> None:
        self.index.save()
        manifest = {
            "backend": self.backend,
            "procedure_count": self.conn.execute("SELECT COUNT(*) FROM procedures").fetchone()[0],
            "edge_count": self.conn.execute("SELECT COUNT(*) FROM procedure_edges").fetchone()[0],
            "negative_memory_count": self.conn.execute("SELECT COUNT(*) FROM negative_memory").fetchone()[0],
            "embedding_backend": self.index.backend,
            "embedding_model": self.index.model_name,
            "candidate_k": self.candidate_k,
            "scoring": {
                "semantic_weight": self.semantic_weight,
                "lexical_weight": self.lexical_weight,
                "activation_weight": self.activation_weight,
                "graph_weight": self.graph_weight,
                "outcome_weight": self.outcome_weight,
                "metadata_weight": self.metadata_weight,
                "acceptance_threshold": self.acceptance_threshold,
                "activation_threshold": self.activation_threshold,
            },
        }
        (self.root / "procedural_manifest.json").write_text(json.dumps(manifest, indent=2))

    def export_compressed_faiss(
        self,
        root: Path | None = None,
        candidate_k: int = 50,
        index_kind: str = "auto",
        ivf_nlist: int = 64,
        pq_m: int = 16,
        pq_bits: int = 8,
    ) -> dict[str, Any]:
        """Build a compressed FAISS snapshot from the current procedure table.

        The online procedural retriever stays mutable and graph-aware; this
        export gives us an optimized vector artifact for latency/storage tests.
        """
        try:
            from mind2web.compressed_faiss_memory import CompressedFaissWorkflowMemory
        except ModuleNotFoundError:
            from compressed_faiss_memory import CompressedFaissWorkflowMemory

        snapshot_root = root or (self.root / "compressed_faiss_snapshot")
        json_root = snapshot_root / "procedure_json"
        json_root.mkdir(parents=True, exist_ok=True)
        rows = self.conn.execute(
            "SELECT name, procedure_json, compact_text FROM procedures ORDER BY created_at_ms, name"
        ).fetchall()
        items = []
        for row in rows:
            procedure = parse_json(row["procedure_json"], {})
            json_path = json_root / f"{re.sub(r'[^a-zA-Z0-9_.-]+', '_', row['name']).strip('_')}.json"
            json_path.write_text(json.dumps(procedure, indent=2))
            items.append((row["compact_text"], procedure, json_path))

        backend = CompressedFaissWorkflowMemory(
            root=snapshot_root,
            workflow_json_root=json_root,
            candidate_k=candidate_k,
            index_kind=index_kind,
            ivf_nlist=ivf_nlist,
            pq_m=pq_m,
            pq_bits=pq_bits,
        )
        backend.add_workflows_batch(items)
        backend.save()
        stats = backend.stats()
        (snapshot_root / "snapshot_stats.json").write_text(json.dumps(stats, indent=2))
        return stats


def prompt_procedure(procedure: dict[str, Any], max_steps: int = 10) -> dict[str, Any]:
    """Return the compact object that should be shown to the action model."""
    execution_steps = []
    for step in procedure.get("execution", {}).get("steps", [])[:max_steps]:
        execution_steps.append(
            {
                "intent": step.get("intent", ""),
                "operation": step.get("operation", ""),
                "target_type": step.get("target_type", ""),
                "target": step.get("target", ""),
                "value_policy": step.get("value_policy", ""),
                "selection_rule": step.get("selection_rule", ""),
                "success_signal": step.get("success_signal", ""),
            }
        )
    return {
        "name": procedure.get("name", ""),
        "memory_type": "typed_procedure",
        "goal_pattern": procedure.get("goal_pattern", ""),
        "applies_when": procedure.get("activation", {}).get("when_to_use", ""),
        "do_not_use_when": procedure.get("activation", {}).get("when_not_to_use", ""),
        "expected_ui_terms": procedure.get("activation", {}).get("ui_terms", [])[:16],
        "steps": execution_steps,
        "success_check": procedure.get("termination", {}).get("success_check", ""),
        "guards": procedure.get("failure_recovery", {}).get("guards", [])[:4],
        "stats": procedure.get("stats", {}),
    }
