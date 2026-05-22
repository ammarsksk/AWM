"""Disk-backed workflow store.

The retrieval index keeps vectors in RAM for speed. Full workflow JSON objects
are persisted as individual files and loaded on demand for prompts/reports.
"""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Iterator, Mapping, Any


class DiskWorkflowStore(Mapping[str, dict[str, Any]]):
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._manifest: dict[str, str] = {}
        self._cache: dict[str, dict[str, Any]] = {}
        if self.manifest_path.exists():
            self._manifest = json.loads(self.manifest_path.read_text())

    def __getitem__(self, name: str) -> dict[str, Any]:
        workflow = self.get(name)
        if workflow is None:
            raise KeyError(name)
        return workflow

    def __iter__(self) -> Iterator[str]:
        return iter(self._manifest)

    def __len__(self) -> int:
        return len(self._manifest)

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._manifest

    def values(self):  # type: ignore[override]
        for name in self._manifest:
            workflow = self.get(name)
            if workflow is not None:
                yield workflow

    def items(self):  # type: ignore[override]
        for name in self._manifest:
            workflow = self.get(name)
            if workflow is not None:
                yield name, workflow

    def get(self, name: str | None, default=None):  # type: ignore[override]
        if not name or name not in self._manifest:
            return default
        if name in self._cache:
            return self._cache[name]
        workflow_path = self.root / self._manifest[name]
        workflow = json.loads(workflow_path.read_text())
        self._cache[name] = workflow
        return workflow

    def add(self, workflow: dict[str, Any]) -> Path:
        name = workflow["name"]
        filename = self._filename(name)
        path = self.root / filename
        path.write_text(json.dumps(workflow, indent=2))
        self._manifest[name] = filename
        self._cache[name] = workflow
        self.save_manifest()
        return path

    def path_for(self, name: str) -> Path:
        return self.root / self._filename(name)

    def to_dict(self) -> dict[str, dict[str, Any]]:
        return {name: workflow for name, workflow in self.items()}

    def save_manifest(self) -> None:
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2))

    @staticmethod
    def _filename(name: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9_.-]+", "_", name).strip("_")
        return f"{slug or 'workflow'}.json"
