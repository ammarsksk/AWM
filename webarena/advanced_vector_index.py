from __future__ import annotations

import json
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from local_awm_full_demo import RetrievalResult, WorkflowEmbeddingIndex


TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def average(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def token_set(text: str) -> set[str]:
    return set(TOKEN_RE.findall((text or "").lower()))


def lexical_overlap(left: str, right: str) -> float:
    left_tokens = token_set(left)
    right_tokens = token_set(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def normalized_matrix(vectors: list[list[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="float32")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def pack_binary(matrix: np.ndarray) -> np.ndarray:
    bits = (matrix >= 0).astype("uint8")
    return np.packbits(bits, axis=1)


def safe_index_kind(kind: str) -> str:
    return (kind or "hnsw_sq8").strip().lower().replace("-", "_")


@dataclass
class AdvancedIndexConfig:
    index_kind: str = "hnsw_sq8"
    candidate_k: int = 40
    hnsw_m: int = 32
    hnsw_ef_search: int = 64
    hnsw_ef_construction: int = 80
    ivf_nlist: int = 64
    ivf_nprobe: int = 8
    pq_m: int = 16
    pq_bits: int = 8
    rotation_seed: int = 1337
    semantic_weight: float = 0.78
    lexical_weight: float = 0.17
    freshness_weight: float = 0.05


class AdvancedProcedureVectorIndex:
    """Pluggable FAISS index layer for procedural memories.

    Supported index kinds:
    - flat: exact normalized inner product.
    - hnsw: HNSW over full vectors.
    - sq8: scalar-quantized flat index.
    - hnsw_sq8: HNSW graph over scalar-quantized vectors.
    - ivfpq: IVF + product quantization.
    - opq_ivfpq: OPQ rotation + IVF-PQ.
    - qsgngt: experimental Quantized Scalable Graph NGT-style backend:
      HNSW graph coarse router + IVF tree partitioning + SQ8 compressed vectors.
    - binary_rotation: TurboQuant/RaBitQ-like experimental random rotation + binary search.
    - binary_hnsw_rotation: same binary representation with binary HNSW.
    """

    def __init__(self, root: Path, config: AdvancedIndexConfig | None = None) -> None:
        self.root = root
        self.config = config or AdvancedIndexConfig()
        self.root.mkdir(parents=True, exist_ok=True)
        self.embedder = WorkflowEmbeddingIndex(self.root / "_embedder_metadata.json")
        self.faiss = None
        self.index = None
        self.entries: list[dict[str, Any]] = []
        self.dimension = 0
        self.actual_index_kind = "unbuilt"
        self.binary = False
        self.rotation: np.ndarray | None = None
        self.rotation_perm: np.ndarray | None = None
        self.rotation_signs: np.ndarray | None = None
        self.rotation_mode = "none"
        self.timings: list[dict[str, float]] = []
        self.backend_error: str | None = None
        self._load_faiss()

    @property
    def manifest_path(self) -> Path:
        return self.root / "advanced_manifest.json"

    @property
    def metadata_path(self) -> Path:
        return self.root / "advanced_metadata.jsonl"

    @property
    def index_path(self) -> Path:
        return self.root / "advanced.index"

    @property
    def rotation_path(self) -> Path:
        return self.root / "rotation.npz"

    def _load_faiss(self) -> None:
        try:
            import faiss

            self.faiss = faiss
        except Exception as exc:
            self.backend_error = f"{type(exc).__name__}: {exc}"
            self.faiss = None

    def rebuild_from_rows(self, rows: list[Any]) -> None:
        start = time.perf_counter()
        if self.root.exists():
            for path in [self.metadata_path, self.index_path, self.manifest_path, self.rotation_path]:
                if path.exists():
                    path.unlink()
        self.entries = []
        vectors = []
        for order, row in enumerate(rows):
            name = row["name"]
            text = row["compact_text"]
            vector = self.embedder.embed(text)
            vectors.append(vector)
            self.entries.append(
                {
                    "name": name,
                    "text_for_embedding": text,
                    "created_order": order,
                }
            )

        if not vectors:
            self.index = None
            self.dimension = 0
            self.actual_index_kind = "empty"
            self._write_metadata()
            return

        matrix = normalized_matrix(vectors)
        self.dimension = int(matrix.shape[1])
        self.index, self.actual_index_kind = self._build_index(matrix)
        self._write_index()
        self._write_metadata()
        self.timings.append({"rebuild_ms": elapsed_ms(start)})

    def _build_index(self, matrix: np.ndarray):
        if self.faiss is None:
            self.backend_error = self.backend_error or "faiss_unavailable"
            return None, "python_fallback"
        kind = safe_index_kind(self.config.index_kind)
        rows, dim = matrix.shape
        self.binary = False
        self.rotation = None

        try:
            if kind == "flat":
                index = self.faiss.IndexFlatIP(dim)
                index.add(matrix)
                return index, "flat_ip"

            if kind == "hnsw":
                index = self.faiss.IndexHNSWFlat(dim, self.config.hnsw_m, self.faiss.METRIC_INNER_PRODUCT)
                index.hnsw.efConstruction = self.config.hnsw_ef_construction
                index.hnsw.efSearch = self.config.hnsw_ef_search
                index.add(matrix)
                return index, f"hnsw_m{self.config.hnsw_m}"

            if kind == "sq8":
                index = self.faiss.IndexScalarQuantizer(
                    dim,
                    self.faiss.ScalarQuantizer.QT_8bit,
                    self.faiss.METRIC_INNER_PRODUCT,
                )
                index.train(matrix)
                index.add(matrix)
                return index, "sq8_flat"

            if kind == "hnsw_sq8":
                index = self.faiss.IndexHNSWSQ(
                    dim,
                    self.faiss.ScalarQuantizer.QT_8bit,
                    self.config.hnsw_m,
                    self.faiss.METRIC_INNER_PRODUCT,
                )
                index.hnsw.efConstruction = self.config.hnsw_ef_construction
                index.hnsw.efSearch = self.config.hnsw_ef_search
                index.train(matrix)
                index.add(matrix)
                return index, f"hnsw_sq8_m{self.config.hnsw_m}"

            if kind in {"ivfpq", "opq_ivfpq"}:
                return self._build_ivfpq(matrix, with_opq=(kind == "opq_ivfpq"))

            if kind in {"qsgngt", "qsg_ngt", "ngt_qg"}:
                return self._build_qsgngt(matrix)

            if kind in {"rabitq", "turboquant", "binary_rotation", "binary_hnsw_rotation"}:
                dense = kind == "rabitq"
                return self._build_binary_rotation(matrix, hnsw=kind == "binary_hnsw_rotation", dense=dense)
        except Exception as exc:
            self.backend_error = f"{kind}_failed:{type(exc).__name__}:{exc}"

        index = self.faiss.IndexFlatIP(dim)
        index.add(matrix)
        return index, f"flat_ip_fallback_from_{kind}"

    def _build_ivfpq(self, matrix: np.ndarray, with_opq: bool):
        rows, dim = matrix.shape
        # FAISS PQ training needs enough points per centroid. Fall back gracefully
        # for small WebArena memories while keeping the backend usable in larger runs.
        min_train = max(64, self.config.pq_m * (2**self.config.pq_bits))
        if rows < min_train:
            index = self.faiss.IndexHNSWSQ(
                dim,
                self.faiss.ScalarQuantizer.QT_8bit,
                self.config.hnsw_m,
                self.faiss.METRIC_INNER_PRODUCT,
            )
            index.train(matrix)
            index.add(matrix)
            return index, f"hnsw_sq8_fallback_rows{rows}_need{min_train}"

        nlist = max(1, min(self.config.ivf_nlist, int(math.sqrt(rows))))
        quantizer = self.faiss.IndexFlatIP(dim)
        ivfpq = self.faiss.IndexIVFPQ(
            quantizer,
            dim,
            nlist,
            self.config.pq_m,
            self.config.pq_bits,
            self.faiss.METRIC_INNER_PRODUCT,
        )
        ivfpq.nprobe = max(1, min(self.config.ivf_nprobe, nlist))
        if with_opq:
            opq = self.faiss.OPQMatrix(dim, self.config.pq_m)
            index = self.faiss.IndexPreTransform(opq, ivfpq)
            index.train(matrix)
            index.add(matrix)
            return index, f"opq_ivfpq_nlist{nlist}_m{self.config.pq_m}_bits{self.config.pq_bits}"
        ivfpq.train(matrix)
        ivfpq.add(matrix)
        return ivfpq, f"ivfpq_nlist{nlist}_m{self.config.pq_m}_bits{self.config.pq_bits}"

    def _build_qsgngt(self, matrix: np.ndarray):
        """Build a QSGNGT-style index.

        The public QSGNGT naming is not exposed as a standard Python package in
        this repo, so this backend implements the same design point with FAISS:
        a graph-based coarse router, a tree/IVF partitioning layer, and SQ8
        compressed vectors. This lets us compare the intended architecture
        against HNSW-SQ8 with the same memory and benchmark scripts.
        """
        rows, dim = matrix.shape
        if rows < 64:
            index = self.faiss.IndexHNSWSQ(
                dim,
                self.faiss.ScalarQuantizer.QT_8bit,
                self.config.hnsw_m,
                self.faiss.METRIC_INNER_PRODUCT,
            )
            index.hnsw.efConstruction = self.config.hnsw_ef_construction
            index.hnsw.efSearch = self.config.hnsw_ef_search
            index.train(matrix)
            index.add(matrix)
            return index, f"qsgngt_hnsw_sq8_small_rows{rows}"

        nlist = max(2, min(self.config.ivf_nlist, int(math.sqrt(rows))))
        if hasattr(self.faiss, "IndexHNSWFlat"):
            quantizer = self.faiss.IndexHNSWFlat(dim, self.config.hnsw_m, self.faiss.METRIC_INNER_PRODUCT)
            quantizer.hnsw.efConstruction = self.config.hnsw_ef_construction
            quantizer.hnsw.efSearch = self.config.hnsw_ef_search
            router = "hnsw_router"
        else:
            quantizer = self.faiss.IndexFlatIP(dim)
            router = "flat_router"
        index = self.faiss.IndexIVFScalarQuantizer(
            quantizer,
            dim,
            nlist,
            self.faiss.ScalarQuantizer.QT_8bit,
            self.faiss.METRIC_INNER_PRODUCT,
        )
        index.nprobe = max(1, min(self.config.ivf_nprobe, nlist))
        index.train(matrix)
        index.add(matrix)
        return index, f"qsgngt_{router}_ivf{nlist}_sq8_nprobe{index.nprobe}"

    def _build_binary_rotation(self, matrix: np.ndarray, hnsw: bool, dense: bool):
        rows, dim = matrix.shape
        rng = np.random.default_rng(self.config.rotation_seed)
        if dense:
            gaussian = rng.normal(size=(dim, dim)).astype("float32")
            q, _ = np.linalg.qr(gaussian)
            self.rotation = q.astype("float32")
            self.rotation_mode = "dense_random_orthogonal"
        else:
            self.rotation_perm = rng.permutation(dim).astype("int64")
            self.rotation_signs = rng.choice(np.asarray([-1.0, 1.0], dtype="float32"), size=dim)
            self.rotation_mode = "signed_permutation"
        rotated = self._apply_rotation(matrix)
        packed = pack_binary(rotated)
        self.binary = True
        if hnsw and hasattr(self.faiss, "IndexBinaryHNSW"):
            index = self.faiss.IndexBinaryHNSW(dim, self.config.hnsw_m)
        else:
            index = self.faiss.IndexBinaryFlat(dim)
        index.add(packed)
        prefix = "rabitq_dense" if dense else "turboquant_signperm"
        return index, (f"{prefix}_hnsw" if hnsw else prefix)

    def _apply_rotation(self, matrix: np.ndarray) -> np.ndarray:
        if self.rotation is not None:
            return matrix @ self.rotation
        if self.rotation_perm is not None and self.rotation_signs is not None:
            return matrix[:, self.rotation_perm] * self.rotation_signs
        return matrix

    def _write_index(self) -> None:
        if self.faiss is None or self.index is None:
            return
        if self.binary:
            self.faiss.write_index_binary(self.index, str(self.index_path))
        else:
            self.faiss.write_index(self.index, str(self.index_path))
        if self.rotation is not None:
            np.savez_compressed(self.rotation_path, mode="dense", rotation=self.rotation)
        elif self.rotation_perm is not None and self.rotation_signs is not None:
            np.savez_compressed(
                self.rotation_path,
                mode="signed_permutation",
                perm=self.rotation_perm,
                signs=self.rotation_signs,
            )

    def _write_metadata(self) -> None:
        self.metadata_path.write_text("\n".join(json.dumps(entry) for entry in self.entries) + "\n")
        self.manifest_path.write_text(json.dumps(self.stats(), indent=2, sort_keys=True))

    def retrieve(self, query: str, top_k: int = 4, threshold: float | None = None) -> RetrievalResult:
        start = time.perf_counter()
        if not self.entries:
            return RetrievalResult(workflow_name=None, score=0.0, candidates=[])
        limit = min(len(self.entries), max(top_k, self.config.candidate_k))
        scored = self._search(query, limit)
        candidates = self._rerank(query, scored)[:top_k]
        self.timings.append({"retrieve_ms": elapsed_ms(start)})
        if candidates:
            return RetrievalResult(
                workflow_name=candidates[0]["workflow_name"],
                score=candidates[0]["combined_score"],
                candidates=candidates,
            )
        return RetrievalResult(workflow_name=None, score=0.0, candidates=[])

    def _search(self, query: str, limit: int) -> list[dict[str, Any]]:
        vector = np.asarray([self.embedder.embed(query)], dtype="float32")
        vector = normalized_matrix(vector.tolist())
        if self.index is None or self.faiss is None:
            scores = [
                float(np.dot(vector[0], np.asarray(self.embedder.embed(entry["text_for_embedding"]), dtype="float32")))
                for entry in self.entries
            ]
            order = np.argsort(scores)[::-1][:limit]
            return [{"idx": int(idx), "semantic_score": float(scores[int(idx)])} for idx in order]

        if self.binary:
            if (
                self.rotation is None
                and self.rotation_perm is None
                and self.rotation_path.exists()
            ):
                payload = np.load(self.rotation_path)
                mode = str(payload["mode"])
                if mode == "dense":
                    self.rotation = payload["rotation"]
                    self.rotation_mode = "dense_random_orthogonal"
                else:
                    self.rotation_perm = payload["perm"]
                    self.rotation_signs = payload["signs"]
                    self.rotation_mode = "signed_permutation"
            rotated = self._apply_rotation(vector)
            packed = pack_binary(rotated)
            distances, indices = self.index.search(packed, limit)
            scored = []
            for dist, idx in zip(distances[0], indices[0]):
                if idx < 0:
                    continue
                similarity = 1.0 - (float(dist) / max(1.0, float(self.dimension)))
                scored.append({"idx": int(idx), "semantic_score": similarity})
            return scored

        scores, indices = self.index.search(vector, limit)
        scored = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:
                scored.append({"idx": int(idx), "semantic_score": float(score)})
        return scored

    def _rerank(self, query: str, scored: list[dict[str, Any]]) -> list[dict[str, Any]]:
        candidates = []
        max_order = max(1, len(self.entries) - 1)
        for item in scored:
            entry = self.entries[item["idx"]]
            semantic = float(item["semantic_score"])
            lexical = lexical_overlap(query, entry["text_for_embedding"])
            freshness = float(entry.get("created_order", 0)) / max_order
            combined = (
                self.config.semantic_weight * semantic
                + self.config.lexical_weight * lexical
                + self.config.freshness_weight * freshness
            )
            candidates.append(
                {
                    "workflow_name": entry["name"],
                    "combined_score": round(combined, 4),
                    "semantic_score": round(semantic, 4),
                    "bm25_score": round(lexical, 4),
                    "freshness_score": round(freshness, 4),
                    "retrieval_backend": f"advanced_{self.actual_index_kind}",
                    "text_for_embedding": entry["text_for_embedding"],
                }
            )
        candidates.sort(key=lambda item: item["combined_score"], reverse=True)
        return candidates

    def stats(self) -> dict[str, Any]:
        retrieve_ms = [item["retrieve_ms"] for item in self.timings if "retrieve_ms" in item]
        rebuild_ms = [item["rebuild_ms"] for item in self.timings if "rebuild_ms" in item]
        index_bytes = self.index_path.stat().st_size if self.index_path.exists() else 0
        metadata_bytes = self.metadata_path.stat().st_size if self.metadata_path.exists() else 0
        code_size = getattr(self.index, "code_size", None) if self.index is not None else None
        return {
            "backend": "advanced_procedural_vector_index",
            "requested_index_kind": self.config.index_kind,
            "actual_index_kind": self.actual_index_kind,
            "embedding_model": self.embedder.model_name if self.embedder.model is not None else None,
            "embedding_model_source": self.embedder.model_source,
            "embedding_backend": self.embedder.backend,
            "backend_error": self.backend_error,
            "entries": len(self.entries),
            "dimension": self.dimension,
            "binary": self.binary,
            "rotation_mode": self.rotation_mode,
            "candidate_k": self.config.candidate_k,
            "hnsw_m": self.config.hnsw_m,
            "hnsw_ef_search": self.config.hnsw_ef_search,
            "ivf_nlist": self.config.ivf_nlist,
            "ivf_nprobe": self.config.ivf_nprobe,
            "pq_m": self.config.pq_m,
            "pq_bits": self.config.pq_bits,
            "faiss_code_size_bytes": code_size,
            "index_bytes": index_bytes,
            "metadata_bytes": metadata_bytes,
            "avg_rebuild_ms": average(rebuild_ms),
            "avg_retrieve_ms": average(retrieve_ms),
            "p95_retrieve_ms": percentile(retrieve_ms, 95),
        }

    def save(self) -> None:
        self.manifest_path.write_text(json.dumps(self.stats(), indent=2, sort_keys=True))
