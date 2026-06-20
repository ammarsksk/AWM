"""Elaborate local demo for Agent Workflow Memory.

This script is intentionally self-contained: no browser, no API key, no WebArena
servers. It demonstrates the AWM control flow with synthetic tasks:

1. Build working memory: an agent prompt with action docs and long-term workflow memory.
2. Produce deterministic "LLM" decisions and save a trace.
3. Track short-term memory: the within-task action history.
4. Store episodic memory: task trajectories and outcomes.
5. Induce long-term workflow memory from successful trajectories.
6. Reuse induced workflows on later tasks.
7. Write a human-readable report with prompts, actions, memory stores, reused
   workflows, and newly added workflows.

It is a demonstration harness, not a benchmark replacement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import re
import textwrap
from typing import Iterable

from induce_rule import format_trajectory, get_abstract_trajectory, remove_invalid_steps
from provider_config import get_openai_compatible_kwargs


ACTION_DOCS = """\
Available actions:
- fill("id", "text"): type text into an input element.
- click("id"): click a clickable page element.
- send_msg_to_user("message"): finish by reporting an answer.
Element ids must be strings.
"""


@dataclass
class LocalTask:
    task_id: str
    website: str
    goal: str
    observation: str
    target_answer: str
    workflow_name: str
    raw_plan: list[str]


@dataclass
class DemoStep:
    env: str
    reason: str
    action: str
    reused_workflow: str | None = None
    llm_raw_output: str | None = None
    short_term_before: list[str] = field(default_factory=list)
    short_term_after: list[str] = field(default_factory=list)
    long_term_memory_visible: list[str] = field(default_factory=list)
    retrieval_candidates: list[dict] = field(default_factory=list)


@dataclass
class DemoExperience:
    task: LocalTask
    prompt: str
    steps: list[DemoStep]
    success: bool
    long_term_before: list[str] = field(default_factory=list)
    long_term_after: list[str] = field(default_factory=list)
    added_workflow: str | None = None
    retrieval_candidates: list[dict] = field(default_factory=list)


@dataclass
class RetrievalResult:
    workflow_name: str | None
    score: float
    candidates: list[dict]


class WorkflowEmbeddingIndex:
    """Hybrid workflow retriever.

    Preferred path:
    - SentenceTransformers encodes workflows and queries.
    - FAISS indexes normalized vectors and returns cosine-like inner products.
    - BM25 supplies lexical backup.

    If optional retrieval dependencies or the embedding model are unavailable,
    this falls back to the deterministic hash embedding used by earlier demos.
    """

    fallback_dim = 384
    semantic_weight = 0.8
    bm25_weight = 0.2
    combined_acceptance_threshold = 0.30
    semantic_acceptance_threshold = 0.28
    default_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_aliases = {
        "cafe": ["coffee", "espresso", "restaurant", "nearby_place"],
        "cafes": ["coffee", "espresso", "restaurant", "nearby_place"],
        "supermarket": ["grocery", "market", "store", "nearby_place"],
        "hotel": ["inn", "lodging", "nearby_place"],
        "hilton": ["hotel", "inn", "lodging"],
        "nearest": ["closest", "nearby", "distance"],
        "closest": ["nearest", "nearby", "distance"],
        "find": ["search", "lookup", "retrieve"],
        "report": ["answer", "send", "tell"],
        "address": ["street", "city", "zip", "billing", "shipping", "account_form"],
        "update": ["change", "edit", "save"],
        "saved": ["account", "profile"],
    }

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.entries: list[dict] = []
        self.model_name = os.environ.get("AWM_EMBEDDING_MODEL", self.default_model_name)
        self.backend = "uninitialized"
        self.model = None
        self.model_source: str | None = None
        self.faiss_index = None
        self.bm25 = None
        self.corpus_tokens: list[list[str]] = []
        self._load_backend()
        self.save()

    def _load_backend(self) -> None:
        try:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            from sentence_transformers import SentenceTransformer
            import faiss

            model_source = self._local_sentence_transformer_path() or self.model_name
            self.model = SentenceTransformer(str(model_source), local_files_only=True)
            self.model_source = str(model_source)
            self.faiss = faiss
            self.backend = "sentence_transformers_faiss_bm25"
        except Exception as exc:
            self.model = None
            self.model_source = None
            self.faiss = None
            self.backend = "local_hash_fallback"
            self.backend_error = str(exc)

    def _local_sentence_transformer_path(self) -> Path | None:
        """Return a cached Hugging Face snapshot path without touching the network."""
        configured_path = Path(self.model_name).expanduser()
        if configured_path.exists():
            return configured_path

        if "/" not in self.model_name:
            return None

        cache_name = "models--" + self.model_name.replace("/", "--")
        cache_root = Path.home() / ".cache" / "huggingface" / "hub" / cache_name
        snapshots_root = cache_root / "snapshots"

        ref_path = cache_root / "refs" / "main"
        if ref_path.exists():
            snapshot_path = snapshots_root / ref_path.read_text().strip()
            if snapshot_path.exists():
                return snapshot_path

        if snapshots_root.exists():
            snapshots = sorted(path for path in snapshots_root.iterdir() if path.is_dir())
            if snapshots:
                return snapshots[-1]

        return None

    def rebuild(self, workflows: list[str]) -> None:
        entries = []
        for workflow in workflows:
            name = workflow.splitlines()[0].lstrip("#").strip()
            text = self.workflow_text_for_embedding(workflow)
            vector = self.embed(text)
            entries.append(
                {
                    "name": name,
                    "text_for_embedding": text,
                    "vector": vector,
                }
            )
        self.entries = entries
        self._rebuild_search_indexes()
        self.save()

    def add_workflow(self, workflow: str, save: bool = True) -> bool:
        name = workflow.splitlines()[0].lstrip("#").strip()
        if any(entry["name"] == name for entry in self.entries):
            return False
        text = self.workflow_text_for_embedding(workflow)
        self.entries.append(
            {
                "name": name,
                "text_for_embedding": text,
                "vector": self.embed(text),
            }
        )
        self._rebuild_search_indexes()
        if save:
            self.save()
        return True

    def _rebuild_search_indexes(self) -> None:
        self.corpus_tokens = [
            self.tokenize(entry["text_for_embedding"])
            for entry in self.entries
        ]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens) if self.corpus_tokens else None
        except Exception:
            self.bm25 = None

        if self.backend == "sentence_transformers_faiss_bm25" and self.entries:
            import numpy as np

            matrix = np.array([entry["vector"] for entry in self.entries], dtype="float32")
            self.faiss_index = self.faiss.IndexFlatIP(matrix.shape[1])
            self.faiss_index.add(matrix)
        else:
            self.faiss_index = None

    def retrieve(self, query: str, top_k: int = 3, threshold: float | None = None) -> RetrievalResult:
        if threshold is None:
            threshold = self.combined_acceptance_threshold
        semantic_scores = self._semantic_scores(query)
        bm25_scores = self._bm25_scores(query)
        candidates = []
        for idx, entry in enumerate(self.entries):
            semantic_score = semantic_scores[idx] if idx < len(semantic_scores) else 0.0
            bm25_score = bm25_scores[idx] if idx < len(bm25_scores) else 0.0
            combined = self.semantic_weight * semantic_score + self.bm25_weight * bm25_score
            candidates.append(
                {
                    "workflow_name": entry["name"],
                    "combined_score": round(combined, 4),
                    "semantic_score": round(semantic_score, 4),
                    "bm25_score": round(bm25_score, 4),
                    "retrieval_backend": self.backend,
                    "text_for_embedding": entry["text_for_embedding"],
                }
            )
        candidates.sort(key=lambda item: item["combined_score"], reverse=True)
        candidates = candidates[:top_k]
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

    def _semantic_scores(self, query: str) -> list[float]:
        if not self.entries:
            return []

        if self.backend == "sentence_transformers_faiss_bm25" and self.faiss_index is not None:
            import numpy as np

            query_vector = np.array([self.embed(query)], dtype="float32")
            scores, indices = self.faiss_index.search(query_vector, len(self.entries))
            ordered_scores = [0.0] * len(self.entries)
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0:
                    ordered_scores[int(idx)] = float(score)
            return ordered_scores

        query_vector = self.embed(query)
        return [cosine(query_vector, entry["vector"]) for entry in self.entries]

    def _bm25_scores(self, query: str) -> list[float]:
        if self.bm25 is None or not self.entries:
            return [0.0] * len(self.entries)

        raw_scores = list(self.bm25.get_scores(self.tokenize(query)))
        if not raw_scores:
            return [0.0] * len(self.entries)
        min_score = min(raw_scores)
        max_score = max(raw_scores)
        if max_score == min_score:
            return [0.0] * len(raw_scores)
        return [
            (score - min_score) / (max_score - min_score)
            for score in raw_scores
        ]

    def save(self) -> None:
        payload = {
            "embedding_type": self.backend,
            "embedding_model": self.model_name if self.model is not None else None,
            "embedding_model_source": self.model_source,
            "semantic_weight": self.semantic_weight,
            "bm25_weight": self.bm25_weight,
            "combined_acceptance_threshold": self.combined_acceptance_threshold,
            "semantic_acceptance_threshold": self.semantic_acceptance_threshold,
            "dimension": len(self.entries[0]["vector"]) if self.entries else None,
            "backend_error": getattr(self, "backend_error", None),
            "entries": self.entries,
        }
        self.path.write_text(json.dumps(payload, indent=2))

    @classmethod
    def workflow_text_for_embedding(cls, workflow: str) -> str:
        lines = []
        for line in workflow.splitlines():
            stripped = line.strip()
            if (
                stripped.startswith("## ")
                or stripped.startswith("Source:")
                or stripped.startswith("Website:")
                or stripped.startswith("Domain:")
                or stripped.startswith("Subdomain:")
                or stripped.startswith("Goal pattern:")
                or stripped.startswith("Abstract action pattern:")
                or stripped.startswith("First observation hint:")
                or stripped.startswith("<action>")
                or re.match(r"^(CLICK|TYPE|SELECT|HOVER|PRESS|SCROLL|NAVIGATE|ENTER)\b", stripped)
                or stripped.startswith("fill(")
                or stripped.startswith("click(")
                or stripped.startswith("send_msg_to_user(")
            ):
                lines.append(stripped)
        return " ".join(lines)

    def embed(self, text: str) -> list[float]:
        if self.model is not None:
            vector = self.model.encode(
                text,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            return [float(value) for value in vector.tolist()]
        return self.hash_embed(text)

    @classmethod
    def hash_embed(cls, text: str) -> list[float]:
        vector = [0.0] * cls.fallback_dim
        tokens = cls.tokenize(text)
        expanded_tokens = []
        for token in tokens:
            expanded_tokens.append(token)
            expanded_tokens.extend(cls.semantic_aliases.get(token, []))

        for token in expanded_tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % cls.fallback_dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vector[idx] += sign

        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_]+", text.lower())


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class WorkflowMemory:
    def __init__(self, path: Path, embedding_path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("")
        self.embedding_index = WorkflowEmbeddingIndex(embedding_path)

    def read(self) -> str:
        return self.path.read_text()

    def workflows(self) -> list[str]:
        blocks = [block.strip() for block in self.read().split("\n\n## ") if block.strip()]
        normalized = []
        for block in blocks:
            normalized.append(block if block.startswith("## ") else "## " + block)
        return normalized

    def names(self) -> list[str]:
        names = []
        for block in self.workflows():
            first = block.splitlines()[0].lstrip("#").strip()
            names.append(first)
        return names

    def workflow_by_name(self, name: str | None) -> str:
        if name is None:
            return ""
        for block in self.workflows():
            first = block.splitlines()[0].lstrip("#").strip()
            if first == name:
                return block
        return ""

    def find_reusable(self, task: LocalTask) -> RetrievalResult:
        query = f"{task.goal}\n{task.observation}"
        return self.embedding_index.retrieve(query=query)

    def add(self, workflow: str) -> bool:
        existing = self.read()
        workflow = workflow.strip()
        name = workflow.splitlines()[0].strip()
        if name in existing:
            return False
        self.path.write_text((existing.rstrip() + "\n\n" + workflow + "\n").lstrip())
        self.embedding_index.add_workflow(workflow)
        return True


class DeterministicDemoAgent:
    """A fake LLM-backed agent that emits traceable decisions.

    The decisions are deterministic so the demo is stable, but the trace is
    shaped like the information you would inspect from a real LLM run.
    """

    def __init__(
        self,
        memory: WorkflowMemory,
        use_real_llm: bool = False,
        model: str = "google/gemini-2.5-pro",
    ) -> None:
        self.memory = memory
        self.use_real_llm = use_real_llm
        self.model = model
        self._client = None

    def build_prompt(self, task: LocalTask, retrieval: RetrievalResult) -> str:
        memory_text = self.memory.workflow_by_name(retrieval.workflow_name).strip() or "(no workflow retrieved)"
        retrieval_text = json.dumps(retrieval.candidates, indent=2)
        return f"""\
# Instructions
You are an agent solving a web navigation task. Use reusable workflows when
they match the current goal, but adapt ids and values to the current page.

# Goal
{task.goal}

# Observation
{task.observation}

# Action Space
{ACTION_DOCS}

# Semantic Workflow Retrieval
Top retrieved workflow: {retrieval.workflow_name or "None"}
Candidate scores:
{retrieval_text}

# Retrieved Agent Workflow Memory
{memory_text}
"""

    def solve(self, task: LocalTask) -> DemoExperience:
        retrieval = self.memory.find_reusable(task)
        reused = retrieval.workflow_name
        prompt = self.build_prompt(task, retrieval)
        cleaned_plan = remove_invalid_steps(task.raw_plan)
        long_term_before = self.memory.names()
        short_term_memory: list[str] = []

        steps = []
        for index, action in enumerate(cleaned_plan):
            short_term_before = list(short_term_memory)
            if self.use_real_llm:
                reason, action, raw_output = self._llm_decision(
                    task=task,
                    proposed_action=action,
                    reused=reused,
                    retrieval_candidates=retrieval.candidates,
                    previous_actions=short_term_before,
                )
            else:
                reason = self._reason_for_action(task, action, reused, index)
                raw_output = None
            short_term_memory.append(action)
            steps.append(
                DemoStep(
                    env=task.observation if index == 0 else "The page reflects the previous action.",
                    reason=reason,
                    action=action,
                    reused_workflow=reused,
                    llm_raw_output=raw_output,
                    short_term_before=short_term_before,
                    short_term_after=list(short_term_memory),
                    long_term_memory_visible=long_term_before,
                    retrieval_candidates=retrieval.candidates,
                )
            )

        success = bool(steps and task.target_answer in steps[-1].action)
        return DemoExperience(
            task=task,
            prompt=prompt,
            steps=steps,
            success=success,
            long_term_before=long_term_before,
            retrieval_candidates=retrieval.candidates,
        )

    def _reason_for_action(
        self,
        task: LocalTask,
        action: str,
        reused: str | None,
        index: int,
    ) -> str:
        prefix = (
            f"I found a matching workflow, '{reused}', and will adapt it. "
            if reused
            else "No matching workflow exists yet, so I will solve from the task context. "
        )
        if action.startswith("fill("):
            return prefix + "I need to enter the task-specific search or form value."
        if action.startswith("click("):
            return prefix + "I need to activate the relevant page control."
        if action.startswith("send_msg_to_user("):
            return prefix + "I have enough information and should report the final answer."
        return prefix + f"This is step {index + 1} of the planned trajectory."

    def _llm_decision(
        self,
        task: LocalTask,
        proposed_action: str,
        reused: str | None,
        retrieval_candidates: list[dict],
        previous_actions: list[str],
    ) -> tuple[str, str, str]:
        if self._client is None:
            from openai import OpenAI

            kwargs = get_openai_compatible_kwargs()
            if not kwargs.get("api_key"):
                raise RuntimeError(
                    "No API key found. Set NVIDIA_NIM_API_KEY or OPENAI_API_KEY before "
                    "running with --use-real-llm."
                )
            self._client = OpenAI(**kwargs)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are simulating the decision trace of a web-navigation agent. "
                    "Return strict JSON only. Do not use markdown."
                ),
            },
            {
                "role": "user",
                "content": f"""\
Task goal:
{task.goal}

Observation:
{task.observation}

Workflow memory:
{self.memory.workflow_by_name(reused).strip() or "(no workflow retrieved)"}

Workflow matched for this task:
{reused or "None"}

Semantic retrieval candidates:
{json.dumps(retrieval_candidates, indent=2)}

Previous actions:
{json.dumps(previous_actions)}

For this local demo, choose exactly this next action:
{proposed_action}

Return JSON with exactly these keys:
{{
  "reason": "brief reason for the action, mentioning workflow reuse if applicable; no markdown",
  "action": "{proposed_action}"
}}

Return only the JSON object. No markdown fences. No commentary.
""",
            },
        ]
        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=800,
        )
        raw_output = response.choices[0].message.content or ""
        parsed = parse_llm_json(raw_output)
        if parsed is None:
            reason = self._reason_for_action(task, proposed_action, reused, len(previous_actions))
            reason += " The raw LLM output was preserved in the trace but was not valid JSON."
            action = proposed_action
        else:
            reason = str(parsed.get("reason", "")).strip() or "The model selected the next action."
            action = str(parsed.get("action", "")).strip()

        if action != proposed_action:
            reason += " The demo kept the planned action because the model did not return it exactly."
            action = proposed_action
        return reason, action, raw_output


def parse_llm_json(raw_output: str) -> dict | None:
    """Parse strict JSON or a JSON object wrapped in markdown fences."""
    text = raw_output.strip()
    if not text:
        return None

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def demo_tasks() -> list[LocalTask]:
    return [
        LocalTask(
            task_id="map-001",
            website="map",
            goal="Find cafes near Carnegie Mellon University and report the closest one.",
            observation='AXTree: [145] search textbox, [147] Go button, [301] result "La Prima Espresso"',
            target_answer="La Prima Espresso",
            workflow_name="Search and Report a Nearby Place",
            raw_plan=[
                'fill("145", "cafes near Carnegie Mellon University")',
                'click("147")',
                "click(147)",  # deliberately invalid; shows filtering
                'send_msg_to_user("The closest cafe is La Prima Espresso.")',
            ],
        ),
        LocalTask(
            task_id="map-002",
            website="map",
            goal="Find the nearest supermarket to Hilton Garden Inn and report it.",
            observation='AXTree: [145] search textbox, [147] Go button, [305] result "ALDI"',
            target_answer="ALDI",
            workflow_name="Search and Report a Nearby Place",
            raw_plan=[
                'fill("145", "supermarket near Hilton Garden Inn")',
                'click("147")',
                'send_msg_to_user("The nearest supermarket is ALDI.")',
            ],
        ),
        LocalTask(
            task_id="shopping-001",
            website="shopping",
            goal="Update my account address to 231 Willow Way, Chicago, IL 60601.",
            observation='AXTree: [227] My Account, [1668] Change Billing Address, [1694] Street, [2031] City, [2036] Zip, [2046] Save',
            target_answer="address updated",
            workflow_name="Update an Address Form",
            raw_plan=[
                'click("227")',
                'click("1668")',
                'fill("1694", "231 Willow Way")',
                'fill("2031", "Chicago")',
                'fill("2036", "60601")',
                'click("2046")',
                'send_msg_to_user("The address updated successfully.")',
            ],
        ),
        LocalTask(
            task_id="shopping-002",
            website="shopping",
            goal="Change my saved address to 10 Market Street, Pittsburgh, PA 15213.",
            observation='AXTree: [227] My Account, [1668] Change Billing Address, [1694] Street, [2031] City, [2036] Zip, [2046] Save',
            target_answer="address updated",
            workflow_name="Update an Address Form",
            raw_plan=[
                'click("227")',
                'click("1668")',
                'fill("1694", "10 Market Street")',
                'fill("2031", "Pittsburgh")',
                'fill("2036", "15213")',
                'click("2046")',
                'send_msg_to_user("The address updated successfully.")',
            ],
        ),
    ]


def induce_workflow(experience: DemoExperience) -> str | None:
    if not experience.success:
        return None

    think_list = [step.reason for step in experience.steps]
    action_list = [[step.action] for step in experience.steps]
    abstract = get_abstract_trajectory(action_list)
    trajectory = format_trajectory(think_list, action_list)

    return f"""\
## {experience.task.workflow_name}
Source task: {experience.task.task_id}
Website: {experience.task.website}
Goal pattern: {generalize_goal(experience.task)}
Abstract action pattern: {abstract}

{trajectory}
"""


def generalize_goal(task: LocalTask) -> str:
    if task.workflow_name == "Search and Report a Nearby Place":
        return "Search for {place-type} near {location}, inspect results, and report the answer."
    if task.workflow_name == "Update an Address Form":
        return "Navigate to account address settings, fill {address-fields}, save, and report completion."
    return task.goal


def experience_to_json(
    experience: DemoExperience,
    workflow_memory_path: Path,
    episodic_memory_path: Path,
    workflow_embedding_path: Path,
) -> dict:
    return {
        "task_id": experience.task.task_id,
        "website": experience.task.website,
        "goal": experience.task.goal,
        "success": experience.success,
        "memory_stores": {
            "working_memory": "The prompt field below; rebuilt for each task.",
            "short_term_memory": "Per-step previous actions; reset at the start of each task.",
            "episodic_memory": str(episodic_memory_path),
            "long_term_workflow_memory": str(workflow_memory_path),
            "workflow_embedding_index": str(workflow_embedding_path),
        },
        "long_term_memory_before_task": experience.long_term_before,
        "long_term_memory_after_task": experience.long_term_after,
        "added_workflow": experience.added_workflow,
        "semantic_retrieval_candidates": experience.retrieval_candidates,
        "prompt": experience.prompt,
        "steps": [
            {
                "env": step.env,
                "reason": step.reason,
                "action": step.action,
                "reused_workflow": step.reused_workflow,
                "llm_raw_output": step.llm_raw_output,
                "short_term_before": step.short_term_before,
                "short_term_after": step.short_term_after,
                "long_term_memory_visible": step.long_term_memory_visible,
                "retrieval_candidates": step.retrieval_candidates,
            }
            for step in experience.steps
        ],
    }


def render_report(
    experiences: Iterable[DemoExperience],
    added_workflows: list[str],
    memory: WorkflowMemory,
    trace_path: Path,
    episodic_memory_path: Path,
) -> str:
    llm_mode = "real external LLM calls" if any(
        step.llm_raw_output is not None for exp in experiences for step in exp.steps
    ) else "deterministic local decisions"
    lines = [
        "# Local AWM Full Demo Report",
        "",
        f"Decision mode: {llm_mode}.",
        "This report is generated without WebArena servers.",
        "It demonstrates workflow memory creation, hybrid retrieval, reuse, action filtering, and trace logging.",
        f"Workflow retrieval backend: `{memory.embedding_index.backend}`.",
        f"Retrieval scoring: `{memory.embedding_index.semantic_weight}` semantic + `{memory.embedding_index.bm25_weight}` BM25.",
        "",
        "## Memory Architecture",
        "",
        "| Memory Layer | What It Means In AWM | Where This Demo Stores It |",
        "| --- | --- | --- |",
        f"| Working memory | Current prompt: task goal, observation, action space, and visible workflows | `{trace_path}` under each task's `prompt` field |",
        f"| Short-term memory | Previous actions inside the current task trajectory | `{trace_path}` under each step's `short_term_before` and `short_term_after` |",
        f"| Episodic memory | Completed task experiences: goal, success, trajectory, and reuse outcome | `{episodic_memory_path}` |",
        f"| Long-term workflow memory | Reusable workflows induced from successful tasks | `{memory.path}` |",
        f"| Workflow embedding index | Vector representation of every stored workflow for semantic retrieval | `{memory.embedding_index.path}` |",
        "",
        "## Final Workflow Memory",
        "",
        "```text",
        memory.read().strip(),
        "```",
        "",
        "## Newly Added Workflows",
        "",
    ]
    if added_workflows:
        for workflow in added_workflows:
            lines.extend(["```text", workflow.strip(), "```", ""])
    else:
        lines.append("No workflows were added.")

    lines.extend(["", "## Task Traces", ""])
    for experience in experiences:
        lines.extend(
            [
                f"### {experience.task.task_id} ({experience.task.website})",
                "",
                f"Goal: {experience.task.goal}",
                f"Success: {experience.success}",
                f"Long-term memory before task: {experience.long_term_before or '[]'}",
                f"Long-term memory after task: {experience.long_term_after or '[]'}",
                f"New workflow stored after task: {experience.added_workflow or 'None'}",
                f"Semantic retrieval candidates: {experience.retrieval_candidates or '[]'}",
                "",
                "Prompt excerpt:",
                "",
                "```text",
                excerpt(experience.prompt, 1200),
                "```",
                "",
                "LLM-style trace:",
                "",
            ]
        )
        for i, step in enumerate(experience.steps, start=1):
            reused = step.reused_workflow or "None"
            lines.extend(
                [
                    f"{i}. Reused workflow: {reused}",
                    f"   Short-term before: {step.short_term_before}",
                    f"   Reason: {step.reason}",
                    f"   Action: `{step.action}`",
                    f"   Short-term after: {step.short_term_after}",
                ]
            )
            if step.llm_raw_output is not None:
                lines.extend(
                    [
                        "   Raw LLM output:",
                        "",
                        "   ```json",
                        textwrap.indent(step.llm_raw_output.strip(), "   "),
                        "   ```",
                    ]
                )
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def excerpt(text: str, max_chars: int) -> str:
    compact = textwrap.dedent(text).strip()
    if len(compact) <= max_chars:
        return compact
    return compact[:max_chars].rstrip() + "\n... [truncated]"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("local_demo_output"))
    parser.add_argument(
        "--use-real-llm",
        action="store_true",
        help="Call an OpenAI-compatible LLM such as NVIDIA NIM/Kimi for each trace step.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="google/gemini-2.5-pro",
        help="OpenAI-compatible model name to use with --use-real-llm.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Run only the first N synthetic tasks. Useful for free-tier LLM smoke tests.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    memory = WorkflowMemory(
        path=args.output_dir / "workflow_memory.txt",
        embedding_path=args.output_dir / "workflow_embeddings.json",
    )
    agent = DeterministicDemoAgent(
        memory,
        use_real_llm=args.use_real_llm,
        model=args.model,
    )

    experiences = []
    added_workflows = []
    tasks = demo_tasks()
    if args.max_tasks is not None:
        tasks = tasks[: args.max_tasks]

    for task in tasks:
        experience = agent.solve(task)
        experiences.append(experience)
        workflow = induce_workflow(experience)
        if workflow and memory.add(workflow):
            added_workflows.append(workflow)
            experience.added_workflow = workflow.splitlines()[0].lstrip("#").strip()
        experience.long_term_after = memory.names()

    trace_path = args.output_dir / "llm_trace.json"
    episodic_memory_path = args.output_dir / "episodic_memory.json"
    trace_payload = [
        experience_to_json(exp, memory.path, episodic_memory_path, memory.embedding_index.path)
        for exp in experiences
    ]
    trace_path.write_text(json.dumps(trace_payload, indent=2))

    episodic_payload = [
        {
            "task_id": exp.task.task_id,
            "website": exp.task.website,
            "goal": exp.task.goal,
            "success": exp.success,
            "reused_workflows": sorted(
                {step.reused_workflow for step in exp.steps if step.reused_workflow}
            ),
            "semantic_retrieval_candidates": exp.retrieval_candidates,
            "added_workflow": exp.added_workflow,
            "trajectory": [
                {"reason": step.reason, "action": step.action}
                for step in exp.steps
            ],
        }
        for exp in experiences
    ]
    episodic_memory_path.write_text(json.dumps(episodic_payload, indent=2))

    report_path = args.output_dir / "report.md"
    report_path.write_text(
        render_report(
            experiences=experiences,
            added_workflows=added_workflows,
            memory=memory,
            trace_path=trace_path,
            episodic_memory_path=episodic_memory_path,
        )
    )

    summary = {
        "tasks": len(experiences),
        "successes": sum(exp.success for exp in experiences),
        "workflow_count": len(memory.workflows()),
        "trace_path": str(trace_path),
        "report_path": str(report_path),
        "episodic_memory_path": str(episodic_memory_path),
        "workflow_memory_path": str(memory.path),
        "workflow_embedding_index_path": str(memory.embedding_index.path),
        "llm_mode": "real" if args.use_real_llm else "deterministic",
        "model": args.model if args.use_real_llm else None,
        "base_url_configured": bool(
            os.environ.get("OPENAI_BASE_URL")
            or os.environ.get("OPENAI_API_BASE")
            or os.environ.get("NVIDIA_NIM_BASE_URL")
            or os.environ.get("NVIDIA_BASE_URL")
        ),
        "reused_workflows": [
            {
                "task_id": exp.task.task_id,
                "reused": sorted({step.reused_workflow for step in exp.steps if step.reused_workflow}),
            }
            for exp in experiences
        ],
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
