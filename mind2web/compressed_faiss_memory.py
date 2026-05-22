"""Compressed FAISS workflow memory.

This backend is the practical middle ground:

- workflow JSON and embedding text live on disk
- FAISS keeps compressed vector codes, not raw Python vector lists
- retrieval remains FAISS-fast
- batch build supports scalar quantization for small/medium sets and IVF-PQ
  for larger sets
"""

from __future__ import annotations

from collections import OrderedDict
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
    def __init__(self, max_items: int = 512) -> None:
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


class CompressedFaissWorkflowMemory:
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
        cache_size: int = 512,
        index_kind: str = "auto",
        ivf_nlist: int = 64,
        pq_m: int = 16,
        pq_bits: int = 8,
    ) -> None:
        import faiss

        self.faiss = faiss
        self.root = root
        self.workflow_json_root = workflow_json_root
        self.candidate_k = candidate_k
        self.cache_size = cache_size
        self.index_kind = index_kind
        self.ivf_nlist = ivf_nlist
        self.pq_m = pq_m
        self.pq_bits = pq_bits
        self.root.mkdir(parents=True, exist_ok=True)
        self.workflow_json_root.mkdir(parents=True, exist_ok=True)
        self.text_root = self.root / "embedding_text"
        self.text_root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "compressed.index"
        self.metadata_path = self.root / "metadata.jsonl"
        self.manifest_path = self.root / "manifest.json"
        self.embedder = WorkflowEmbeddingIndex(self.root / "_embedder_metadata.json")
        self.text_cache = TextCache(max_items=cache_size)
        self.index = None
        self.metadata: list[dict[str, Any]] = []
        self.dimension = None
        self.actual_index_kind = None
        self.timings: list[dict[str, float]] = []
        self._load_existing()

    @property
    def backend(self) -> str:
        return "compressed_faiss"

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
        if self.index_path.exists():
            self.index = self.faiss.read_index(str(self.index_path))
        if self.manifest_path.exists():
            manifest = json.loads(self.manifest_path.read_text())
            self.dimension = manifest.get("dimension")
            self.actual_index_kind = manifest.get("actual_index_kind")

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed(text)

    def add_workflows_batch(
        self,
        items: list[tuple[str, dict[str, Any], Path | None]],
        build_index: bool = True,
    ) -> dict[str, float | int]:
        start = time.perf_counter()
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
                    "created_order": index,
                }
            )
        if not vectors:
            return {"batch_add_ms": 0.0, "batch_index_ms": 0.0, "rows": 0}

        matrix = np.vstack(vectors).astype("float32")
        self.faiss.normalize_L2(matrix)
        self.dimension = int(matrix.shape[1])
        self.metadata = metadata
        self.metadata_path.write_text("\n".join(json.dumps(item) for item in metadata) + "\n")
        add_ms = elapsed_ms(start)

        index_start = time.perf_counter()
        self.index, self.actual_index_kind = self._build_index(matrix)
        index_ms = elapsed_ms(index_start)
        self.faiss.write_index(self.index, str(self.index_path))
        self.manifest_path.write_text(
            json.dumps(
                {
                    "backend": self.backend,
                    "dimension": self.dimension,
                    "rows": len(self.metadata),
                    "requested_index_kind": self.index_kind,
                    "actual_index_kind": self.actual_index_kind,
                    "index_path": str(self.index_path),
                    "candidate_k": self.candidate_k,
                },
                indent=2,
            )
        )
        self.timings.append({"batch_add_ms": add_ms, "batch_index_ms": index_ms})
        return {"batch_add_ms": add_ms, "batch_index_ms": index_ms, "rows": len(self.metadata)}

    def _build_index(self, matrix: np.ndarray):
        rows, dim = matrix.shape
        kind = self.index_kind
        if kind == "auto":
            kind = "ivfpq" if rows >= max(1024, self.pq_m * (2**self.pq_bits)) else "sq8"

        if kind == "ivfpq":
            nlist = max(1, min(self.ivf_nlist, int(math.sqrt(rows))))
            quantizer = self.faiss.IndexFlatIP(dim)
            index = self.faiss.IndexIVFPQ(
                quantizer,
                dim,
                nlist,
                self.pq_m,
                self.pq_bits,
                self.faiss.METRIC_INNER_PRODUCT,
            )
            index.train(matrix)
            index.add(matrix)
            index.nprobe = max(1, min(8, nlist))
            return index, f"ivfpq_nlist{nlist}_m{self.pq_m}_bits{self.pq_bits}"

        quantizer_type = self.faiss.ScalarQuantizer.QT_8bit
        index = self.faiss.IndexScalarQuantizer(
            dim,
            quantizer_type,
            self.faiss.METRIC_INNER_PRODUCT,
        )
        index.train(matrix)
        index.add(matrix)
        return index, "sq8_flat"

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
        if self.index is None or not self.metadata:
            return RetrievalResult(workflow_name=None, score=0.0, candidates=[])
        query_vector = np.asarray([self.embed(query)], dtype="float32")
        self.faiss.normalize_L2(query_vector)
        limit = min(len(self.metadata), max(top_k, candidate_k or self.candidate_k))
        scores, indices = self.index.search(query_vector, limit)
        rows = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            row = dict(self.metadata[int(idx)])
            row["_semantic_score"] = float(score)
            rows.append(row)
        candidates = self._rerank_rows(query, rows, metadata=metadata)[:top_k]
        retrieve_ms = elapsed_ms(start)
        self.timings.append({"retrieve_ms": retrieve_ms})
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

    def _rerank_rows(
        self,
        query: str,
        rows: list[dict[str, Any]],
        metadata: dict[str, str] | None,
    ) -> list[dict[str, Any]]:
        text_start = time.perf_counter()
        texts = [self.text_cache.get(Path(row["text_path"])) for row in rows]
        self.timings.append({"text_load_ms": elapsed_ms(text_start)})
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
        raw_scores = [lexical_overlap(query, text) for text in texts]
        if not raw_scores:
            return []
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        if max_score == min_score:
            return [0.0] * len(raw_scores)
        return [(score - min_score) / (max_score - min_score) for score in raw_scores]

    def stats(self) -> dict[str, Any]:
        batch_add_ms = [item["batch_add_ms"] for item in self.timings if item.get("batch_add_ms")]
        batch_index_ms = [item["batch_index_ms"] for item in self.timings if item.get("batch_index_ms")]
        retrieve_ms = [item["retrieve_ms"] for item in self.timings if item.get("retrieve_ms")]
        text_load_ms = [item["text_load_ms"] for item in self.timings if item.get("text_load_ms")]
        code_size = self.index.code_size if self.index is not None and hasattr(self.index, "code_size") else None
        estimated_code_mb = (
            len(self.metadata) * code_size / (1024 * 1024)
            if code_size is not None
            else 0.0
        )
        return {
            "backend": self.backend,
            "workflow_count": len(self.metadata),
            "python_vector_count": 0,
            "python_vector_values": 0,
            "python_vector_mb": 0.0,
            "actual_index_kind": self.actual_index_kind,
            "candidate_k": self.candidate_k,
            "faiss_code_size_bytes": code_size,
            "estimated_faiss_code_mb": estimated_code_mb,
            "avg_batch_add_ms": average(batch_add_ms),
            "avg_batch_index_ms": average(batch_index_ms),
            "avg_retrieve_ms": average(retrieve_ms),
            "avg_text_load_ms": average(text_load_ms),
            "workflow_text_cache_items": len(self.text_cache._cache),
            "workflow_text_cache_hits": self.text_cache.hits,
            "workflow_text_cache_misses": self.text_cache.misses,
        }

    def save(self) -> None:
        (self.root / "memory_stats.json").write_text(json.dumps(self.stats(), indent=2))
