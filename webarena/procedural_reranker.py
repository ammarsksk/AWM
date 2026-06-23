from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")
ACTION_RE = re.compile(
    r"\b(click|fill|select_option|scroll|send_msg_to_user|press|hover|noop)\b",
    re.IGNORECASE,
)


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


FEATURE_NAMES = [
    "base_score",
    "semantic_score",
    "lexical_score",
    "activation_score",
    "metadata_score",
    "outcome_score",
    "graph_score",
    "family_score",
    "structure_score",
    "freshness_score",
    "reliability_score",
    "goal_text_overlap",
    "skeleton_overlap",
    "negative_penalty",
    "required_gap",
]


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


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


def infer_task_family(text: str) -> str:
    lowered = (text or "").lower()
    tokens = token_set(lowered)
    if (
        "main criticisms" in lowered
        or "customers say" in lowered
        or "customer reviews" in lowered
        or "extract the relevant sentences" in lowered
        or "review" in tokens
        or "reviews" in tokens
    ):
        if "reviewers" in tokens and ("mention" in tokens or "mentioned" in tokens):
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


def canonical_family(family: str | None) -> str:
    key = (family or "general").strip().lower().replace("-", "_").replace(" ", "_")
    return FAMILY_ALIASES.get(key, key or "general")


def family_score(query_family: str, procedure_family: str) -> float:
    query_family = canonical_family(query_family)
    procedure_family = canonical_family(procedure_family)
    if query_family == procedure_family:
        return 1.0
    compatible = {
        ("spend_category_total", "fulfilled_order_total"),
        ("bought_option_lookup", "first_purchase_lookup"),
        ("price_range", "product_search"),
        ("product_capacity_search", "product_search"),
        ("review_extraction", "review_lookup"),
        ("order_status_total", "fulfilled_order_total"),
        ("order_status_total", "bought_option_lookup"),
    }
    if (query_family, procedure_family) in compatible:
        return 0.55
    if query_family == "general" or procedure_family == "general":
        return 0.05
    return 0.0


def action_skeleton_terms(text: str) -> set[str]:
    text = text or ""
    actions_match = re.search(r"\bActions:\s*(.*?)(?:\s+\d+\.\s+|$)", text, re.IGNORECASE)
    action_region = actions_match.group(1) if actions_match else text
    actions = {item.lower() for item in ACTION_RE.findall(action_region)}
    semantic_markers = {
        "review",
        "reviews",
        "reviewers",
        "order",
        "orders",
        "total",
        "price",
        "range",
        "search",
        "product",
        "capacity",
        "storage",
        "account",
        "configuration",
        "fulfilled",
        "purchase",
        "answer",
    }
    markers = token_set(text) & semantic_markers
    return actions | markers


def skeleton_fit(query: str, candidate_text: str) -> float:
    query_family = canonical_family(infer_task_family(query))
    expected = EXPECTED_ACTIONS.get(query_family, set())
    candidate_terms = action_skeleton_terms(candidate_text)
    action_fit = len(expected & candidate_terms) / len(expected) if expected else 0.0
    semantic_fit = lexical_overlap(query, " ".join(candidate_terms))
    return clamp(0.72 * action_fit + 0.28 * semantic_fit)


def candidate_name(candidate: dict[str, Any]) -> str:
    return str(candidate.get("name") or candidate.get("workflow_name") or "")


def candidate_text(candidate: dict[str, Any], text_by_name: dict[str, str] | None = None) -> str:
    text = str(candidate.get("text_for_embedding") or "")
    if text:
        return text
    name = candidate_name(candidate)
    return (text_by_name or {}).get(name, "")


def local_hf_snapshot(model_name: str) -> Path | None:
    path = Path(model_name).expanduser()
    if path.exists():
        return path
    cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    model_dir = cache_root / ("models--" + model_name.replace("/", "--"))
    snapshots = model_dir / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [item for item in snapshots.iterdir() if item.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda item: item.stat().st_mtime)


class OptionalCrossEncoder:
    def __init__(self, model_name: str, enabled: bool) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self.available = False
        self.error: str | None = None
        self.model = None
        self.model_source: str | None = None
        if enabled:
            self._load()

    def _load(self) -> None:
        source = local_hf_snapshot(self.model_name)
        if source is None:
            self.error = "model_not_cached"
            return
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(str(source), max_length=512)
            self.model_source = str(source)
            self.available = True
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"

    def score(self, query: str, candidates: list[dict[str, Any]], text_by_name: dict[str, str]) -> list[float]:
        if not self.available or self.model is None or not candidates:
            return [0.0 for _ in candidates]
        pairs = [(query, candidate_text(candidate, text_by_name)[:2400]) for candidate in candidates]
        try:
            values = self.model.predict(pairs)
        except Exception as exc:
            self.available = False
            self.error = f"predict_failed:{type(exc).__name__}: {exc}"
            return [0.0 for _ in candidates]
        return [clamp(sigmoid(float(value))) for value in values]


class LogisticReranker:
    def __init__(self) -> None:
        self.weights = np.zeros(len(FEATURE_NAMES), dtype="float32")
        self.bias = 0.0
        self.trained = False
        self.training_examples = 0
        self.positive_examples = 0

    def fit(self, features: list[list[float]], labels: list[int]) -> None:
        if not features or len(set(labels)) < 2:
            return
        x = np.asarray(features, dtype="float32")
        y = np.asarray(labels, dtype="float32")
        pos = float(y.sum())
        neg = float(len(y) - pos)
        pos_weight = min(12.0, max(1.0, neg / max(pos, 1.0)))
        sample_weights = np.where(y > 0.5, pos_weight, 1.0).astype("float32")
        self.weights = np.zeros(x.shape[1], dtype="float32")
        self.bias = math.log((pos + 1.0) / (neg + 1.0))
        lr = 0.18
        l2 = 0.018
        for _ in range(360):
            logits = x @ self.weights + self.bias
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
            error = (probs - y) * sample_weights
            denom = max(float(sample_weights.sum()), 1.0)
            grad_w = (x.T @ error) / denom + l2 * self.weights
            grad_b = float(error.sum() / denom)
            self.weights -= lr * grad_w.astype("float32")
            self.bias -= lr * grad_b
        self.trained = True
        self.training_examples = len(labels)
        self.positive_examples = int(pos)

    def predict(self, features: list[float]) -> float:
        if not self.trained:
            return 0.5
        return sigmoid(float(np.dot(self.weights, np.asarray(features, dtype="float32")) + self.bias))


@dataclass
class RerankerStats:
    mode: str
    cross_encoder_model: str
    cross_encoder_available: bool
    cross_encoder_error: str | None
    ml_trained: bool
    ml_training_examples: int
    ml_positive_examples: int
    procedures_with_reliability: int


class ProceduralReranker:
    def __init__(
        self,
        db_path: Path | None = None,
        mode: str = "feature",
        cross_encoder_model: str = "BAAI/bge-reranker-large",
        train_limit: int = 1000,
    ) -> None:
        self.mode = (mode or "feature").strip().lower()
        self.cross_encoder_model = cross_encoder_model
        self.train_limit = train_limit
        self.text_by_name: dict[str, str] = {}
        self.proc_successes: dict[str, int] = {}
        self.proc_failures: dict[str, int] = {}
        self.proc_families: dict[str, str] = {}
        self.selected_counts: Counter[str] = Counter()
        self.seen_counts: Counter[str] = Counter()
        self.ml = LogisticReranker()
        self.cross_encoder = OptionalCrossEncoder(
            cross_encoder_model,
            enabled=self.mode in {"cross_encoder", "full"},
        )
        if db_path and Path(db_path).exists():
            self._load_memory_stats(Path(db_path))
            if self.mode in {"ml", "full"}:
                self._train_from_events(Path(db_path), train_limit=train_limit)

    def _load_memory_stats(self, db_path: Path) -> None:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute(
                "SELECT name, compact_text, success_count, failure_count, goal_pattern FROM procedures"
            ):
                name = row["name"]
                self.text_by_name[name] = row["compact_text"]
                self.proc_successes[name] = int(row["success_count"] or 0)
                self.proc_failures[name] = int(row["failure_count"] or 0)
                self.proc_families[name] = canonical_family(infer_task_family(row["goal_pattern"]))
            rows = conn.execute(
                """
                SELECT selected, raw_candidates_json, candidates_json
                FROM retrieval_events
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(self.train_limit, 1),),
            ).fetchall()
        finally:
            conn.close()

        for row in rows:
            selected = row["selected"]
            if selected:
                self.selected_counts[selected] += 1
            for field in ("raw_candidates_json", "candidates_json"):
                try:
                    candidates = json.loads(row[field] or "[]")
                except Exception:
                    candidates = []
                for candidate in candidates:
                    name = candidate_name(candidate)
                    if name:
                        self.seen_counts[name] += 1

    def _train_from_events(self, db_path: Path, train_limit: int) -> None:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT goal, selected, raw_candidates_json, candidates_json
                FROM retrieval_events
                WHERE selected IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
                """,
                (max(train_limit, 1),),
            ).fetchall()
        finally:
            conn.close()

        features: list[list[float]] = []
        labels: list[int] = []
        for row in rows:
            selected = row["selected"]
            if not selected:
                continue
            try:
                raw_candidates = json.loads(row["raw_candidates_json"] or "[]")
            except Exception:
                raw_candidates = []
            try:
                selected_candidates = json.loads(row["candidates_json"] or "[]")
            except Exception:
                selected_candidates = []
            merged: dict[str, dict[str, Any]] = {}
            for candidate in raw_candidates + selected_candidates:
                name = candidate_name(candidate)
                if name:
                    merged[name] = candidate
            if selected not in merged and selected in self.text_by_name:
                merged[selected] = {"name": selected}
            for name, candidate in merged.items():
                features.append(self.feature_vector(row["goal"], candidate))
                labels.append(1 if name == selected else 0)
        self.ml.fit(features, labels)

    def reliability(self, name: str) -> float:
        successes = self.proc_successes.get(name, 0)
        failures = self.proc_failures.get(name, 0)
        outcome_prior = (successes + 1.0) / (successes + failures + 2.0)
        seen = self.seen_counts.get(name, 0)
        selected = self.selected_counts.get(name, 0)
        retrieval_prior = (selected + 1.0) / (seen + 2.0)
        return clamp(0.62 * outcome_prior + 0.38 * retrieval_prior)

    def feature_vector(self, query: str, candidate: dict[str, Any]) -> list[float]:
        name = candidate_name(candidate)
        text = candidate_text(candidate, self.text_by_name)
        base = float(candidate.get("score", candidate.get("combined_score", 0.0)) or 0.0)
        semantic = float(candidate.get("semantic_score", candidate.get("combined_score", base)) or 0.0)
        lexical = float(candidate.get("lexical_score", candidate.get("bm25_score", 0.0)) or 0.0)
        if lexical <= 0.0 and text:
            lexical = lexical_overlap(query, text)
        activation = float(candidate.get("activation_score", 0.0) or 0.0)
        metadata = float(candidate.get("metadata_score", 1.0) or 0.0)
        outcome = float(candidate.get("outcome_score", 0.0) or 0.0)
        graph = float(candidate.get("graph_score", 0.0) or 0.0)
        q_family = canonical_family(str(candidate.get("query_family") or infer_task_family(query)))
        p_family = canonical_family(str(candidate.get("procedure_family") or self.proc_families.get(name) or infer_task_family(text)))
        family = float(candidate.get("family_score", family_score(q_family, p_family)) or 0.0)
        structure = float(candidate.get("structure_score", skeleton_fit(query, text)) or 0.0)
        freshness = float(candidate.get("freshness_score", 0.0) or 0.0)
        reliability = self.reliability(name) if name else 0.5
        goal_text = lexical_overlap(query, text) if text else lexical
        skeleton = skeleton_fit(query, text) if text else structure
        negative = float(candidate.get("negative_penalty", 0.0) or 0.0)
        required = float(candidate.get("required_score", 0.0) or 0.0)
        required_gap = clamp((base - required + 1.0) / 2.0)
        return [
            clamp(base),
            clamp(semantic),
            clamp(lexical),
            clamp(activation),
            clamp(metadata),
            clamp(outcome),
            clamp(graph),
            clamp(family),
            clamp(structure),
            clamp(freshness),
            clamp(reliability),
            clamp(goal_text),
            clamp(skeleton),
            clamp(negative),
            clamp(required_gap),
        ]

    def feature_score(self, features: list[float]) -> float:
        values = dict(zip(FEATURE_NAMES, features))
        score = (
            0.20 * values["base_score"]
            + 0.12 * values["semantic_score"]
            + 0.12 * values["lexical_score"]
            + 0.08 * values["activation_score"]
            + 0.04 * values["metadata_score"]
            + 0.07 * values["outcome_score"]
            + 0.07 * values["graph_score"]
            + 0.14 * values["family_score"]
            + 0.11 * values["structure_score"]
            + 0.02 * values["freshness_score"]
            + 0.11 * values["reliability_score"]
            + 0.09 * values["goal_text_overlap"]
            + 0.10 * values["skeleton_overlap"]
            + 0.03 * values["required_gap"]
            - 0.20 * values["negative_penalty"]
        )
        return clamp(score)

    def rerank(self, query: str, candidates: list[dict[str, Any]], top_k: int | None = None) -> list[dict[str, Any]]:
        if not candidates:
            return []
        top_k = top_k or len(candidates)
        cross_scores = self.cross_encoder.score(query, candidates, self.text_by_name)
        reranked: list[dict[str, Any]] = []
        for candidate, cross_score in zip(candidates, cross_scores):
            features = self.feature_vector(query, candidate)
            feature_score = self.feature_score(features)
            ml_score = self.ml.predict(features)
            parts: list[tuple[float, float]] = []
            if self.mode == "none":
                parts.append((1.0, clamp(float(candidate.get("score", candidate.get("combined_score", 0.0)) or 0.0))))
            elif self.mode == "feature":
                parts.append((1.0, feature_score))
            elif self.mode == "ml":
                parts.extend([(0.55, feature_score), (0.45, ml_score)])
            elif self.mode == "cross_encoder":
                parts.append((0.50, feature_score))
                if self.cross_encoder.available:
                    parts.append((0.50, cross_score))
            else:
                parts.append((0.45, feature_score))
                if self.ml.trained:
                    parts.append((0.25, ml_score))
                if self.cross_encoder.available:
                    parts.append((0.30, cross_score))
            denom = sum(weight for weight, _ in parts) or 1.0
            final_score = sum(weight * score for weight, score in parts) / denom
            enriched = dict(candidate)
            enriched["rerank_score"] = round(clamp(final_score), 4)
            enriched["feature_rerank_score"] = round(feature_score, 4)
            enriched["ml_rerank_score"] = round(ml_score, 4)
            enriched["cross_encoder_score"] = round(cross_score, 4)
            enriched["reliability_score"] = round(features[FEATURE_NAMES.index("reliability_score")], 4)
            enriched["skeleton_score"] = round(features[FEATURE_NAMES.index("skeleton_overlap")], 4)
            reranked.append(enriched)
        reranked.sort(key=lambda item: (item["rerank_score"], item.get("score", item.get("combined_score", 0))), reverse=True)
        return reranked[:top_k]

    def stats(self) -> RerankerStats:
        return RerankerStats(
            mode=self.mode,
            cross_encoder_model=self.cross_encoder_model,
            cross_encoder_available=self.cross_encoder.available,
            cross_encoder_error=self.cross_encoder.error,
            ml_trained=self.ml.trained,
            ml_training_examples=self.ml.training_examples,
            ml_positive_examples=self.ml.positive_examples,
            procedures_with_reliability=len(self.proc_successes),
        )

    def stats_dict(self) -> dict[str, Any]:
        stats = self.stats()
        return {
            "mode": stats.mode,
            "cross_encoder_model": stats.cross_encoder_model,
            "cross_encoder_available": stats.cross_encoder_available,
            "cross_encoder_error": stats.cross_encoder_error,
            "ml_trained": stats.ml_trained,
            "ml_training_examples": stats.ml_training_examples,
            "ml_positive_examples": stats.ml_positive_examples,
            "procedures_with_reliability": stats.procedures_with_reliability,
        }
