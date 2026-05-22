"""LanceDB-backed long-term workflow memory.

This backend keeps full workflow JSON on disk and stores only vector-search
metadata in LanceDB. Unlike the existing RAM FAISS retriever, Python does not
hold every workflow vector in `entries`; LanceDB owns the on-disk table and ANN
index.
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

from webarena.local_awm_full_demo import RetrievalResult, WorkflowEmbeddingIndex


def token_set(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


class WorkflowJsonCache:
    """Small LRU cache for loaded workflow JSON files."""

    def __init__(self, max_items: int = 128) -> None:
        self.max_items = max_items
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, path: Path) -> dict[str, Any]:
        key = str(path)
        if key in self._cache:
            self.hits += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.misses += 1
        workflow = json.loads(path.read_text())
        self._cache[key] = workflow
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_items:
            self._cache.popitem(last=False)
        return workflow

    def stats(self) -> dict[str, int]:
        return {
            "workflow_json_cache_items": len(self._cache),
            "workflow_json_cache_hits": self.hits,
            "workflow_json_cache_misses": self.misses,
        }


@dataclass
class LanceTiming:
    add_ms: float = 0.0
    retrieve_ms: float = 0.0
    index_ms: float = 0.0
    final_index_ms: float = 0.0
    json_load_ms: float = 0.0


class LanceWorkflowMemory:
    """Disk-backed workflow retriever using LanceDB.

    LanceDB stores vectors and metadata on disk. We still keep the embedding
    model in memory, but not the whole workflow vector collection.
    """

    semantic_weight = 0.70
    bm25_weight = 0.15
    same_website_boost = 0.10
    same_domain_boost = 0.05
    combined_acceptance_threshold = 0.30
    semantic_acceptance_threshold = 0.28

    def __init__(
        self,
        db_path: Path,
        workflow_json_root: Path,
        table_name: str = "workflows",
        candidate_k: int = 50,
        cache_size: int = 128,
        index_type: str = "IVF_SQ",
        reindex_every: int = 0,
        min_index_rows: int = 2,
    ) -> None:
        try:
            import lancedb
        except ImportError as exc:
            raise RuntimeError(
                "LanceDB backend requires `lancedb`. Install with "
                "`python3 -m pip install lancedb`."
            ) from exc

        self.db_path = db_path
        self.workflow_json_root = workflow_json_root
        self.table_name = table_name
        self.candidate_k = candidate_k
        self.index_type = index_type
        self.reindex_every = reindex_every
        self.min_index_rows = min_index_rows
        self.workflow_json_root.mkdir(parents=True, exist_ok=True)
        self.db_path.mkdir(parents=True, exist_ok=True)

        self.embedder = WorkflowEmbeddingIndex(db_path / "_embedder_metadata.json")
        self.cache = WorkflowJsonCache(max_items=cache_size)
        self.timings: list[LanceTiming] = []
        self._count = 0
        self._table = None
        self._lancedb = lancedb
        self._db = lancedb.connect(str(db_path))
        self._open_existing_table()
        self._indexed_count = self._count

    @property
    def backend(self) -> str:
        return "lancedb_disk_ann"

    @property
    def model_source(self) -> str | None:
        return self.embedder.model_source

    @property
    def model_name(self) -> str:
        return self.embedder.model_name

    def _open_existing_table(self) -> None:
        try:
            table_names = set(self._db.table_names())
        except Exception:
            table_names = set()
        if self.table_name in table_names:
            self._table = self._db.open_table(self.table_name)
            self._count = self._table.count_rows()

    def embed(self, text: str) -> list[float]:
        return self.embedder.embed(text)

    def add_workflow(
        self,
        workflow_text: str,
        workflow: dict[str, Any] | None = None,
        json_path: Path | None = None,
        save: bool = True,
    ) -> bool:
        start = time.perf_counter()
        name = workflow_text.splitlines()[0].lstrip("#").strip()
        text = self.embedder.workflow_text_for_embedding(workflow_text)
        vector = self.embed(text)
        if json_path is None:
            json_path = self.workflow_json_root / f"{safe_name(name)}.json"
        relative_json_path = str(json_path)
        row = {
            "workflow_name": name,
            "vector": vector,
            "text_for_embedding": text,
            "website": str((workflow or {}).get("website", "")),
            "domain": str((workflow or {}).get("domain", "")),
            "subdomain": str((workflow or {}).get("subdomain", "")),
            "goal_pattern": str((workflow or {}).get("goal_pattern") or (workflow or {}).get("task", "")),
            "json_path": relative_json_path,
            "created_order": self._count + 1,
            "accepted_count": 0,
            "success_count": 0,
        }

        if self._table is None:
            self._table = self._db.create_table(self.table_name, data=[row], mode="overwrite")
        else:
            if self._workflow_exists(name):
                return False
            self._table.add([row])

        self._count += 1
        add_ms = elapsed_ms(start)
        index_ms = 0.0
        if self._should_reindex():
            index_start = time.perf_counter()
            self._create_ann_index()
            index_ms = elapsed_ms(index_start)
        self.timings.append(LanceTiming(add_ms=add_ms, index_ms=index_ms))
        return True

    def add_workflows_batch(
        self,
        items: list[tuple[str, dict[str, Any], Path | None]],
        build_index: bool = True,
    ) -> dict[str, float | int]:
        """Batch ingest workflows into LanceDB.

        This is the preferred long-term ingestion path for large memory builds:
        compute vectors, append all rows in one table write, then build ANN once.
        """
        start = time.perf_counter()
        rows = []
        for offset, (workflow_text, workflow, json_path) in enumerate(items, start=1):
            name = workflow_text.splitlines()[0].lstrip("#").strip()
            text = self.embedder.workflow_text_for_embedding(workflow_text)
            if json_path is None:
                json_path = self.workflow_json_root / f"{safe_name(name)}.json"
            rows.append(
                {
                    "workflow_name": name,
                    "vector": self.embed(text),
                    "text_for_embedding": text,
                    "website": str(workflow.get("website", "")),
                    "domain": str(workflow.get("domain", "")),
                    "subdomain": str(workflow.get("subdomain", "")),
                    "goal_pattern": str(workflow.get("goal_pattern") or workflow.get("task", "")),
                    "json_path": str(json_path),
                    "created_order": self._count + offset,
                    "accepted_count": 0,
                    "success_count": 0,
                }
            )

        if not rows:
            return {"batch_add_ms": 0.0, "batch_index_ms": 0.0, "rows": 0}

        if self._table is None:
            self._table = self._db.create_table(self.table_name, data=rows, mode="overwrite")
        else:
            self._table.add(rows)
        self._count += len(rows)
        add_ms = elapsed_ms(start)

        index_ms = 0.0
        if build_index:
            index_start = time.perf_counter()
            self._create_ann_index()
            index_ms = elapsed_ms(index_start)
            self.timings.append(LanceTiming(final_index_ms=index_ms))
        self.timings.append(LanceTiming(add_ms=add_ms, index_ms=index_ms))
        return {"batch_add_ms": add_ms, "batch_index_ms": index_ms, "rows": len(rows)}

    def _workflow_exists(self, name: str) -> bool:
        if self._table is None:
            return False
        try:
            return bool(self._table.search().where(f"workflow_name = {json.dumps(name)}").limit(1).to_list())
        except Exception:
            return False

    def _should_reindex(self) -> bool:
        if self._table is None or self._count < self.min_index_rows:
            return False
        if self.reindex_every <= 0:
            return False
        return self._count % self.reindex_every == 0

    def _create_ann_index(self) -> None:
        if self._table is None:
            return
        num_partitions = max(1, min(16, int(math.sqrt(max(self._count, 1)))))
        try:
            self._table.create_index(
                metric="cosine",
                vector_column_name="vector",
                index_type=self.index_type,
                num_partitions=num_partitions,
                replace=True,
                num_bits=8,
            )
        except Exception:
            # Tiny demo tables can fail index training depending on LanceDB
            # version. Search still works by scanning the disk table.
            return
        self._indexed_count = self._count

    def finalize_index(self) -> float:
        """Build/refresh ANN index once after online ingestion.

        This is the long-term-memory path we want for scale: online adds append
        rows cheaply, then a batch/background phase refreshes ANN state.
        """
        if self._table is None or self._count < self.min_index_rows:
            return 0.0
        start = time.perf_counter()
        self._create_ann_index()
        final_index_ms = elapsed_ms(start)
        self.timings.append(LanceTiming(final_index_ms=final_index_ms))
        return final_index_ms

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
        if self._table is None or self._count == 0:
            return RetrievalResult(workflow_name=None, score=0.0, candidates=[])

        query_vector = self.embed(query)
        limit = max(top_k, candidate_k or self.candidate_k)
        rows = self._table.search(query_vector).limit(limit).to_list()
        candidates = self._rerank_rows(query, rows, metadata=metadata)[:top_k]
        retrieve_ms = elapsed_ms(start)
        self.timings.append(LanceTiming(retrieve_ms=retrieve_ms))

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
        bm25_scores = self._bm25_scores(query, rows)
        candidates = []
        for idx, row in enumerate(rows):
            semantic_score = self._semantic_from_distance(float(row.get("_distance", 1.0)))
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
            same_website = bool(metadata and row.get("website") == metadata.get("website"))
            same_domain = bool(metadata and row.get("domain") == metadata.get("domain"))
            combined = (
                self.semantic_weight * semantic_score
                + self.bm25_weight * bm25_score
                + (self.same_website_boost if same_website else 0.0)
                + (self.same_domain_boost if same_domain else 0.0)
            )
            candidates.append(
                {
                    "workflow_name": row["workflow_name"],
                    "combined_score": round(combined, 4),
                    "semantic_score": round(semantic_score, 4),
                    "bm25_score": round(bm25_score, 4),
                    "same_website_boost": self.same_website_boost if same_website else 0.0,
                    "same_domain_boost": self.same_domain_boost if same_domain else 0.0,
                    "retrieval_backend": self.backend,
                    "text_for_embedding": row.get("text_for_embedding", ""),
                    "json_path": row.get("json_path", ""),
                    "website": row.get("website", ""),
                    "domain": row.get("domain", ""),
                    "subdomain": row.get("subdomain", ""),
                }
            )
        candidates.sort(key=lambda item: item["combined_score"], reverse=True)
        return candidates

    def _semantic_from_distance(self, distance: float) -> float:
        # LanceDB returns L2-style distances for normalized vectors in this API
        # path. For normalized vectors, dot/cosine ~= 1 - (L2^2 / 2). LanceDB's
        # reported distance for opposite orthogonal unit vectors is 2.0 in the
        # installed version, so this maps 0 -> 1 and 2 -> 0.
        return max(0.0, min(1.0, 1.0 - distance / 2.0))

    def _bm25_scores(self, query: str, rows: list[dict[str, Any]]) -> list[float]:
        texts = [row.get("text_for_embedding", "") for row in rows]
        if not texts:
            return []
        try:
            from rank_bm25 import BM25Okapi

            corpus = [WorkflowEmbeddingIndex.tokenize(text) for text in texts]
            raw_scores = list(BM25Okapi(corpus).get_scores(WorkflowEmbeddingIndex.tokenize(query)))
        except Exception:
            raw_scores = [lexical_overlap(query, text) for text in texts]
        if not raw_scores:
            return [0.0] * len(texts)
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        if max_score == min_score:
            return [0.0] * len(raw_scores)
        return [(score - min_score) / (max_score - min_score) for score in raw_scores]

    def load_workflow_json(self, candidate: dict[str, Any]) -> dict[str, Any] | None:
        json_path = candidate.get("json_path")
        if not json_path:
            return None
        start = time.perf_counter()
        workflow = self.cache.get(Path(json_path))
        self.timings.append(LanceTiming(json_load_ms=elapsed_ms(start)))
        return workflow

    def stats(self) -> dict[str, Any]:
        add_ms = [timing.add_ms for timing in self.timings if timing.add_ms]
        retrieve_ms = [timing.retrieve_ms for timing in self.timings if timing.retrieve_ms]
        index_ms = [timing.index_ms for timing in self.timings if timing.index_ms]
        final_index_ms = [
            timing.final_index_ms
            for timing in self.timings
            if timing.final_index_ms
        ]
        json_load_ms = [timing.json_load_ms for timing in self.timings if timing.json_load_ms]
        return {
            "backend": self.backend,
            "workflow_count": self._count,
            "python_vector_count": 0,
            "python_vector_values": 0,
            "python_vector_mb": 0.0,
            "lancedb_path": str(self.db_path),
            "ann_index_type": self.index_type,
            "candidate_k": self.candidate_k,
            "reindex_every": self.reindex_every,
            "indexed_count": getattr(self, "_indexed_count", 0),
            "avg_add_ms": average(add_ms),
            "avg_retrieve_ms": average(retrieve_ms),
            "avg_index_ms": average(index_ms),
            "final_index_ms": sum(final_index_ms),
            "avg_json_load_ms": average(json_load_ms),
            **self.cache.stats(),
        }

    def save(self) -> None:
        (self.db_path / "memory_stats.json").write_text(json.dumps(self.stats(), indent=2))


def safe_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_") or "workflow"


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
