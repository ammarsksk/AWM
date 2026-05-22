"""Memory-mapped workflow vector store.

This is a lean long-term memory backend for local research runs:

- full workflow JSON lives on disk
- embedding text lives in small per-workflow text files
- vectors live in a NumPy memmap file, not Python lists
- retrieval scans the memmap in chunks using vectorized dot products

It is exact search rather than ANN, but for the current embedding dimension and
dataset sizes it is often faster than a database engine while still avoiding
Python-side vector growth. For very large stores this can be combined with
sharding, centroids, or a later ANN layer.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import json
import math
import re
import time
from typing import Any

import numpy as np

from webarena.local_awm_full_demo import RetrievalResult, WorkflowEmbeddingIndex


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "workflow"


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class TextCache:
    def __init__(self, max_items: int = 256) -> None:
        self.max_items = max_items
        self._cache: OrderedDict[str, str] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, path: Path) -> str:
        key = str(path)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        text = path.read_text()
        self._cache[key] = text
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return text


@dataclass
class MemmapTiming:
    batch_add_ms: float = 0.0
    retrieve_ms: float = 0.0
    text_load_ms: float = 0.0


class MemmapWorkflowMemory:
    semantic_weight = 0.78
    bm25_weight = 0.12
    same_website_boost = 0.07
    same_domain_boost = 0.03
    combined_acceptance_threshold = 0.30
    semantic_acceptance_threshold = 0.28

    def __init__(
        self,
        root: Path,
        workflow_json_root: Path,
        candidate_k: int = 50,
        chunk_size: int = 8192,
        vector_dtype: str = "float16",
        cache_size: int = 256,
    ) -> None:
        self.root = root
        self.workflow_json_root = workflow_json_root
        self.candidate_k = candidate_k
        self.chunk_size = chunk_size
        self.vector_dtype = np.dtype(vector_dtype)
        self.workflow_json_root.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)
        self.text_root = self.root / "embedding_text"
        self.text_root.mkdir(parents=True, exist_ok=True)
        self.vector_path = self.root / "vectors.memmap"
        self.metadata_path = self.root / "metadata.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.embedder = WorkflowEmbeddingIndex(self.root / "_embedder_metadata.json")
        self.text_cache = TextCache(max_items=cache_size)
        self.timings: list[MemmapTiming] = []
        self.metadata: list[dict[str, Any]] = []
        self.vectors = None
        self.dimension = None
        self._load_existing()

    @property
    def backend(self) -> str:
        return "memmap_exact"

    @property
    def model_source(self) -> str | None:
        return self.embedder.model_source

    @property
    def model_name(self) -> str:
        return self.embedder.model_name

    def _load_existing(self) -> None:
        if self.metadata_path.exists():
            self.metadata = [
                json.loads(line)
                for line in self.metadata_path.read_text().splitlines()
                if line.strip()
            ]
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            self.dimension = manifest.get("dimension")
            self.vector_dtype = np.dtype(manifest.get("vector_dtype", str(self.vector_dtype)))
        if self.metadata and self.dimension and self.vector_path.exists():
            self.vectors = np.memmap(
                self.vector_path,
                dtype=self.vector_dtype,
                mode="r",
                shape=(len(self.metadata), self.dimension),
            )

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed(text)

    def add_workflows_batch(
        self,
        items: list[tuple[str, dict[str, Any], Path | None]],
        build_index: bool = True,
    ) -> dict[str, float | int]:
        start = time.perf_counter()
        if not items:
            return {"batch_add_ms": 0.0, "batch_index_ms": 0.0, "rows": 0}

        vectors = []
        metadata = []
        for index, (workflow_text, workflow, json_path) in enumerate(items, start=1):
            name = workflow_text.splitlines()[0].lstrip("#").strip()
            text = self.embedder.workflow_text_for_embedding(workflow_text)
            vector = np.asarray(self.embed(text), dtype="float32")
            if json_path is None:
                json_path = self.workflow_json_root / f"{safe_name(name)}.json"
            text_path = self.text_root / f"{safe_name(name)}.txt"
            text_path.write_text(text)
            vectors.append(vector)
            metadata.append(
                {
                    "workflow_name": name,
                    "website": str(workflow.get("website", "")),
                    "domain": str(workflow.get("domain", "")),
                    "subdomain": str(workflow.get("subdomain", "")),
                    "goal_pattern": str(workflow.get("goal_pattern") or workflow.get("task", "")),
                    "json_path": str(json_path),
                    "text_path": str(text_path),
                    "created_order": len(self.metadata) + index,
                }
            )

        matrix = np.vstack(vectors).astype(self.vector_dtype)
        self.dimension = int(matrix.shape[1])
        memmap = np.memmap(
            self.vector_path,
            dtype=self.vector_dtype,
            mode="w+",
            shape=matrix.shape,
        )
        memmap[:] = matrix[:]
        memmap.flush()
        self.metadata = metadata
        self.metadata_path.write_text(
            "\n".join(json.dumps(item) for item in self.metadata) + "\n"
        )
        self.manifest_path.write_text(
            json.dumps(
                {
                    "backend": self.backend,
                    "dimension": self.dimension,
                    "rows": len(self.metadata),
                    "vector_dtype": str(self.vector_dtype),
                    "vector_path": str(self.vector_path),
                },
                indent=2,
            )
        )
        self.vectors = np.memmap(
            self.vector_path,
            dtype=self.vector_dtype,
            mode="r",
            shape=matrix.shape,
        )
        batch_add_ms = elapsed_ms(start)
        self.timings.append(MemmapTiming(batch_add_ms=batch_add_ms))
        return {"batch_add_ms": batch_add_ms, "batch_index_ms": 0.0, "rows": len(self.metadata)}

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        threshold: float | None = None,
        metadata: dict[str, str] | None = None,
        candidate_k: int | None = None,
    ) -> RetrievalResult:
        start = time.perf_counter()
        if threshold is None:
            threshold = self.combined_acceptance_threshold
        if self.vectors is None or not self.metadata:
            return RetrievalResult(workflow_name=None, score=0.0, candidates=[])

        query_vector = np.asarray(self.embed(query), dtype="float32")
        limit = min(len(self.metadata), max(top_k, candidate_k or self.candidate_k))
        best_indices, best_scores = self._topk_dot(query_vector, limit)
        rows = []
        for idx, score in zip(best_indices, best_scores):
            row = dict(self.metadata[int(idx)])
            row["_semantic_score"] = float(score)
            rows.append(row)
        candidates = self._rerank_rows(query, rows, metadata=metadata)[:top_k]
        retrieve_ms = elapsed_ms(start)
        self.timings.append(MemmapTiming(retrieve_ms=retrieve_ms))
        if candidates and (
            candidates[0]["combined_score"] >= threshold
            or candidates[0]["semantic_score"] >= self.semantic_acceptance_threshold
        ):
            return RetrievalResult(
                workflow_name=candidates[0]["workflow_name"],
                score=candidates[0]["combined_score"],
                candidates=candidates,
            )
        return RetrievalResult(workflow_name=None, score=0.0, candidates=candidates)

    def _topk_dot(self, query_vector: np.ndarray, limit: int) -> tuple[np.ndarray, np.ndarray]:
        top_indices = []
        top_scores = []
        total = len(self.metadata)
        for start in range(0, total, self.chunk_size):
            end = min(total, start + self.chunk_size)
            chunk = np.asarray(self.vectors[start:end], dtype="float32")
            scores = chunk @ query_vector
            local_k = min(limit, len(scores))
            if local_k == len(scores):
                local_indices = np.arange(len(scores))
            else:
                local_indices = np.argpartition(scores, -local_k)[-local_k:]
            top_indices.extend((local_indices + start).tolist())
            top_scores.extend(scores[local_indices].tolist())
        scores = np.asarray(top_scores, dtype="float32")
        indices = np.asarray(top_indices, dtype="int64")
        final_k = min(limit, len(scores))
        if final_k == len(scores):
            order = np.argsort(-scores)
        else:
            rough = np.argpartition(scores, -final_k)[-final_k:]
            order = rough[np.argsort(-scores[rough])]
        return indices[order[:final_k]], scores[order[:final_k]]

    def _rerank_rows(
        self,
        query: str,
        rows: list[dict[str, Any]],
        metadata: dict[str, str] | None,
    ) -> list[dict[str, Any]]:
        texts = []
        text_start = time.perf_counter()
        for row in rows:
            texts.append(self.text_cache.get(Path(row["text_path"])))
        self.timings.append(MemmapTiming(text_load_ms=elapsed_ms(text_start)))
        lexical_scores = self._lexical_scores(query, texts)
        candidates = []
        for idx, row in enumerate(rows):
            semantic_score = float(row.get("_semantic_score", 0.0))
            lexical_score = lexical_scores[idx] if idx < len(lexical_scores) else 0.0
            same_website = bool(metadata and row.get("website") == metadata.get("website"))
            same_domain = bool(metadata and row.get("domain") == metadata.get("domain"))
            combined = (
                self.semantic_weight * semantic_score
                + self.bm25_weight * lexical_score
                + (self.same_website_boost if same_website else 0.0)
                + (self.same_domain_boost if same_domain else 0.0)
            )
            candidates.append(
                {
                    "workflow_name": row["workflow_name"],
                    "combined_score": round(combined, 4),
                    "semantic_score": round(semantic_score, 4),
                    "bm25_score": round(lexical_score, 4),
                    "same_website_boost": self.same_website_boost if same_website else 0.0,
                    "same_domain_boost": self.same_domain_boost if same_domain else 0.0,
                    "retrieval_backend": self.backend,
                    "text_for_embedding": texts[idx],
                    "json_path": row.get("json_path", ""),
                    "website": row.get("website", ""),
                    "domain": row.get("domain", ""),
                    "subdomain": row.get("subdomain", ""),
                }
            )
        candidates.sort(key=lambda item: item["combined_score"], reverse=True)
        return candidates

    def _lexical_scores(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        raw_scores = [lexical_overlap(query, text) for text in texts]
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        if max_score == min_score:
            return [0.0] * len(raw_scores)
        return [(score - min_score) / (max_score - min_score) for score in raw_scores]

    def stats(self) -> dict[str, Any]:
        batch_add_ms = [timing.batch_add_ms for timing in self.timings if timing.batch_add_ms]
        retrieve_ms = [timing.retrieve_ms for timing in self.timings if timing.retrieve_ms]
        text_load_ms = [timing.text_load_ms for timing in self.timings if timing.text_load_ms]
        return {
            "backend": self.backend,
            "workflow_count": len(self.metadata),
            "python_vector_count": 0,
            "python_vector_values": 0,
            "python_vector_mb": 0.0,
            "vector_dtype": str(self.vector_dtype),
            "candidate_k": self.candidate_k,
            "chunk_size": self.chunk_size,
            "avg_batch_add_ms": average(batch_add_ms),
            "avg_retrieve_ms": average(retrieve_ms),
            "avg_text_load_ms": average(text_load_ms),
            "workflow_text_cache_items": len(self.text_cache._cache),
            "workflow_text_cache_hits": self.text_cache.hits,
            "workflow_text_cache_misses": self.text_cache.misses,
        }

    def save(self) -> None:
        (self.root / "memory_stats.json").write_text(json.dumps(self.stats(), indent=2))
