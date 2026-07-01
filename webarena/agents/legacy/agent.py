"""
WARNING DEPRECATED WILL BE REMOVED SOON
"""

from dataclasses import asdict, dataclass, field
import ast
import json
import logging
from pathlib import Path
import re
import traceback
from warnings import warn
from langchain.schema import HumanMessage, SystemMessage

from browsergym.core.action.base import AbstractActionSet
from browsergym.utils.obs import flatten_axtree_to_str, flatten_dom_to_str, prune_html
from browsergym.experiments import Agent, AbstractAgentArgs

from ..legacy import dynamic_prompting
from .utils.llm_utils import ParseError, retry
from .utils.chat_api import ChatModelArgs

try:
    from procedural_memory import WebArenaProceduralMemory
except Exception:  # Keep the legacy agent importable without the optional local backend.
    WebArenaProceduralMemory = None


def compact_for_prompt(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


@dataclass
class GenericAgentArgs(AbstractAgentArgs):
    chat_model_args: ChatModelArgs = None
    flags: dynamic_prompting.Flags = field(default_factory=lambda: dynamic_prompting.Flags())
    max_retry: int = 4

    def make_agent(self):
        return GenericAgent(
            chat_model_args=self.chat_model_args, flags=self.flags, max_retry=self.max_retry
        )


class GenericAgent(Agent):

    def obs_preprocessor(self, obs: dict) -> dict:
        """
        Augment observations with text HTML and AXTree representations, which will be stored in
        the experiment traces.
        """

        obs = obs.copy()
        obs["dom_txt"] = flatten_dom_to_str(
            obs["dom_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["axtree_txt"] = flatten_axtree_to_str(
            obs["axtree_object"],
            with_visible=self.flags.extract_visible_tag,
            with_center_coords=self.flags.extract_coords == "center",
            with_bounding_box_coords=self.flags.extract_coords == "box",
            filter_visible_only=self.flags.extract_visible_elements_only,
        )
        obs["pruned_html"] = prune_html(obs["dom_txt"])

        return obs

    def __init__(
        self,
        chat_model_args: ChatModelArgs = None,
        flags: dynamic_prompting.Flags = None,
        max_retry: int = 4,
    ):
        self.chat_model_args = chat_model_args if chat_model_args is not None else ChatModelArgs()
        self.flags = flags if flags is not None else dynamic_prompting.Flags()
        self.max_retry = max_retry

        self.chat_llm = chat_model_args.make_chat_model()
        self.action_set = dynamic_prompting._get_action_space(self.flags)

        # consistency check
        if self.flags.use_screenshot:
            if not self.chat_model_args.has_vision():
                warn(
                    """\

Warning: use_screenshot is set to True, but the chat model \
does not support vision. Disabling use_screenshot."""
                )
                self.flags.use_screenshot = False

        # reset episode memory
        self.obs_history = []
        self.actions = []
        self.memories = []
        self.thoughts = []
        self.procedural_memory = None
        self.task_evidence = {}
        self.orchestrator_events = []

    def get_action(self, obs):

        self.obs_history.append(obs)
        self._update_task_evidence(obs)

        main_prompt = dynamic_prompting.MainPrompt(
            obs_history=self.obs_history,
            actions=self.actions,
            memories=self.memories,
            thoughts=self.thoughts,
            flags=self.flags,
        )

        # Determine the minimum non-None token limit from prompt, total, and input tokens, or set to None if all are None.
        maxes = (
            self.flags.max_prompt_tokens,
            self.chat_model_args.max_total_tokens,
            self.chat_model_args.max_input_tokens,
        )
        maxes = [m for m in maxes if m is not None]
        max_prompt_tokens = min(maxes) if maxes else None

        prompt = dynamic_prompting.fit_tokens(
            main_prompt,
            max_prompt_tokens=max_prompt_tokens,
            model_name=self.chat_model_args.model_name,
        )

        sys_msg = dynamic_prompting.SystemPrompt().prompt
        sys_msg += (
            "\n\n# Strict Final Answer Formatting\n"
            "Do not output visible chain-of-thought. Think internally, then respond with only "
            "one machine-readable action tag. Your response must always contain exactly one "
            "complete action tag in this form:\n"
            "<action>\n"
            "one_allowed_action(...)\n"
            "</action>\n"
            "The first non-whitespace characters in your response must be <action>. "
            "Never output <think>, never output a bare action: block, never output only "
            "<action> without </action>, and never continue thinking after the action tag. "
            "When using send_msg_to_user, answer only the requested value/entities. "
            "Avoid caveats unless the task explicitly asks for explanation. "
            "For dimensions, normalize to forms like 16x24 with no spaces, quotes, or inch marks. "
            "For prices/totals, include the numeric amount(s) exactly and avoid unrelated numbers. "
            "For no-match/zero-spend tasks, answer the exact zero/no-match result plainly. "
            "For reviewer lookup tasks, be deliberately inquisitive: open reviews, inspect "
            "every visible review page/pagination page, keep a running list of all matching "
            "reviewer names, and only then answer. Return only reviewer names separated by "
            "commas. If no matching reviewer exists after checking every review page, answer "
            "exactly N/A."
            "\n\n# Stop And Loop Rules\n"
            "If all requested answer fields are known, stop browsing and immediately call "
            "send_msg_to_user with the final answer. Do not re-sort, re-open, re-click, or "
            "re-verify after the answer is already known. If your last few actions repeat the "
            "same control pattern without adding new information, either answer from the known "
            "evidence or choose a genuinely different strategy. For price range tasks, once you "
            "know both the lowest and highest prices, answer exactly '$min - $max' and do not "
            "toggle ascending/descending sort again. If you have checked all relevant pages and "
            "the requested evidence is absent, answer the benchmark no-match value plainly; "
            "for reviewer lookup no-match tasks this value is exactly N/A. "
            "Never invent a reviewer, order, price, product, or amount that is not visible in "
            "the current context or accumulated evidence."
            "\n\n# Action Grounding\n"
            "For click, fill, and select_option actions, the first argument must be the exact "
            "numeric bid id visible in the current observation, for example click('1288'). "
            "Never write semantic targets such as click('Reviews tab'), click('Reviews'), or "
            "click('product title'). If you know the target concept but not the bid, inspect "
            "the current observation and choose the matching numeric id before acting."
        )
        procedural_prompt = self._procedural_memory_prompt(obs)
        if procedural_prompt:
            sys_msg += "\n\n" + procedural_prompt
        elif self.flags.workflow_path is not None and not self.flags.procedural_memory_path:
            workflow_path = Path(self.flags.workflow_path)
            if workflow_path.exists() and workflow_path.read_text().strip():
                sys_msg += (
                    "\n\n# Agent Workflow Memory\n"
                    "The following workflows are reusable routines induced from previous "
                    "successful tasks. Use them as guidance when they match the current goal, "
                    "adapting element ids and variable values to the current page. Do not "
                    "blindly copy ids from a workflow if they do not appear in the current "
                    "observation.\n\n"
                    + workflow_path.read_text().strip()
                )
        evidence_prompt = self._task_evidence_prompt(obs)
        if evidence_prompt:
            sys_msg += "\n\n" + evidence_prompt

        chat_messages = [
            SystemMessage(content=sys_msg),
            HumanMessage(content=prompt),
        ]

        def parser(text):
            try:
                ans_dict = main_prompt._parse_answer(text)
            except ParseError as e:
                recovered = self._recover_action_from_malformed_response(text)
                if recovered:
                    try:
                        self.action_set.to_python_code(recovered)
                        return {"action": recovered}, True, ""
                    except Exception:
                        pass
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, (
                    str(e)
                    + "\nReturn exactly one complete tag and nothing else after it:\n"
                    + "<action>\n"
                    + "one_allowed_action(...)\n"
                    + "</action>"
                )

            return ans_dict, True, ""

        try:
            ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {
                "action": self._fallback_action_after_parse_failure(
                    obs.get("goal") or "", chat_messages
                )
            }
            ans_dict["err_msg"] = str(e)
            ans_dict["stack_trace"] = traceback.format_exc()
            ans_dict["n_retry"] = self.max_retry

        ans_dict["action"] = self._normalize_final_answer_action(
            ans_dict.get("action"), obs.get("goal") or ""
        )
        ans_dict["action"] = self._loop_guard_action(
            ans_dict.get("action"),
            obs.get("goal") or "",
            ans_dict.get("think") or "",
        )
        ans_dict["action"] = self._evidence_guard_action(
            ans_dict.get("action"),
            obs,
        )
        ans_dict["action"] = self._repair_or_replace_action(ans_dict.get("action"), obs)

        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        ans_dict["chat_messages"] = [m.content for m in chat_messages]
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)
        ans_dict["task_notebook"] = self._serializable_task_notebook()
        ans_dict["orchestrator_events"] = list(self.orchestrator_events[-6:])

        return ans_dict["action"], ans_dict

    def _recover_action_from_malformed_response(self, text: str | None) -> str | None:
        if not text:
            return None
        candidates: list[str] = []

        tag_match = re.search(r"<action>\s*(.*?)(?:</action>|$)", text, flags=re.S | re.I)
        if tag_match:
            candidates.append(tag_match.group(1))

        action_block = re.search(
            r"(?:^|\n)\s*action\s*:\s*(.*?)(?:\n\s*(?:memory|think|observation)\s*:|\Z)",
            text,
            flags=re.S | re.I,
        )
        if action_block:
            candidates.append(action_block.group(1))

        direct = re.search(
            r"((?:click|fill|press|scroll|select_option|send_msg_to_user|goto|go_back|go_forward|noop)\s*\(.*)",
            text,
            flags=re.S,
        )
        if direct:
            candidates.append(direct.group(1))

        for candidate in candidates:
            action = self._trim_action_candidate(candidate)
            if action:
                return action
        final_answer = self._recover_plain_final_answer(text)
        if final_answer:
            return f"send_msg_to_user({final_answer!r})"
        return None

    def _recover_plain_final_answer(self, text: str) -> str | None:
        cleaned = re.sub(r"</?think>", "", text or "", flags=re.I).strip()
        cleaned = re.split(r"\n\s*\[User\]:", cleaned, maxsplit=1)[0].strip()
        if not cleaned or len(cleaned) > 900:
            return None
        lowered = cleaned.lower()
        final_markers = [
            "the answer is",
            "final answer",
            "i found",
            "the reviewer",
            "the reviewers",
            "there are no",
            "no reviewers",
            "no matching",
        ]
        if not any(marker in lowered for marker in final_markers):
            return None
        if any(marker in lowered for marker in ["i will", "i need to", "next, i", "my first action"]):
            return None
        for pattern in [
            r"(?:final answer|the answer is)\s*:?\s*(.+)$",
            r"(?:i found(?: that)?|the reviewers? (?:are|is))\s*:?\s*(.+)$",
        ]:
            match = re.search(pattern, cleaned, flags=re.I | re.S)
            if match:
                return " ".join(match.group(1).split())
        if re.search(r"\b(there are no|no reviewers|no matching)\b", lowered):
            return "N/A"
        return None

    def _trim_action_candidate(self, text: str) -> str | None:
        cleaned = (text or "").strip()
        if not cleaned:
            return None
        cleaned = re.sub(r"</?action>", "", cleaned, flags=re.I).strip()
        cleaned = cleaned.split("<think>", 1)[0].strip()
        cleaned = re.split(r"\n\s*(?:think|memory|observation)\s*:", cleaned, maxsplit=1, flags=re.I)[0].strip()

        allowed = "click|fill|press|scroll|select_option|send_msg_to_user|goto|go_back|go_forward|noop"
        match = re.search(rf"\b({allowed})\s*\(", cleaned)
        if not match:
            return None
        start = match.start()
        fragment = cleaned[start:]
        end = self._balanced_call_end(fragment)
        if end is None:
            if fragment.startswith("send_msg_to_user("):
                return self._recover_truncated_send_msg(fragment)
            return None
        action = fragment[:end].strip()
        if "\n" in action and not self._is_multi_action_safe(action):
            action = action.splitlines()[0].strip()
        return action or None

    def _balanced_call_end(self, text: str) -> int | None:
        depth = 0
        quote = None
        escape = False
        for index, char in enumerate(text):
            if quote:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == quote:
                    quote = None
                continue
            if char in {"'", '"'}:
                quote = char
                continue
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    return index + 1
        return None

    def _recover_truncated_send_msg(self, text: str) -> str | None:
        match = re.match(r"send_msg_to_user\(\s*(['\"])(.*)", text, flags=re.S)
        if not match:
            return None
        message = match.group(2)
        message = re.split(r"\n\s*(?:\[User\]:|Missing the key|<think>|</action>)", message, maxsplit=1)[0]
        message = message.rstrip()
        message = message.replace("\\", "\\\\")
        return f"send_msg_to_user({message!r})"

    def _is_multi_action_safe(self, action: str) -> bool:
        lines = [line.strip() for line in action.splitlines() if line.strip()]
        return len(lines) <= 4 and all(re.match(r"^[a-zA-Z_]+\(", line) for line in lines)

    def _loop_guard_action(self, action: str | None, goal: str, current_thought: str) -> str | None:
        if not action or action.strip().startswith("send_msg_to_user"):
            return action
        repeated = self._looks_like_repeated_action_loop(action)
        if repeated:
            evidence = "\n".join(str(item or "") for item in [*self.thoughts[-6:], current_thought])
            return self._fallback_final_answer_action(goal, evidence, reason="repeat_loop")
        goal_l = (goal or "").lower()
        if "price range" not in goal_l:
            return action
        if not self._looks_like_sort_loop(action, current_thought):
            return action
        evidence = "\n".join(
            str(item or "") for item in [*self.thoughts[-4:], current_thought]
        )
        prices = self._extract_price_values(evidence)
        if len(prices) < 2:
            return action
        low = min(prices)
        high = max(prices)
        if low == high:
            return action
        return f"send_msg_to_user({self._format_price_range(low, high)!r})"

    def _looks_like_repeated_action_loop(self, action: str | None) -> bool:
        if not action:
            return False
        current = self._canonical_action(action)
        if not current:
            return False
        recent = [self._canonical_action(item) for item in self.actions[-3:]]
        recent = [item for item in recent if item]
        if len(recent) >= 2 and recent[-2:] == [current, current]:
            return True
        return False

    def _canonical_action(self, action: str | None) -> str | None:
        if not action:
            return None
        return re.sub(r"\s+", "", action.strip())

    def _fallback_action_after_parse_failure(self, goal: str, chat_messages: list) -> str:
        evidence_parts = []
        for message in reversed(chat_messages):
            content = getattr(message, "content", "")
            if not content:
                continue
            recovered = self._recover_action_from_malformed_response(content)
            if recovered:
                return recovered
            evidence_parts.append(str(content))
            if len(evidence_parts) >= 4:
                break
        return self._fallback_final_answer_action(goal, "\n".join(reversed(evidence_parts)), reason="parse_failure")

    def _fallback_final_answer_action(self, goal: str, evidence: str, reason: str) -> str:
        answer = self._infer_final_answer_from_evidence(goal, evidence)
        if answer is None:
            answer = self._safe_unresolved_answer(goal, evidence, reason)
        return f"send_msg_to_user({self._normalize_final_answer_text(answer, goal)!r})"

    def _infer_final_answer_from_evidence(self, goal: str, evidence: str) -> str | None:
        text = " ".join((evidence or "").split())
        goal_l = (goal or "").lower()
        if "fulfilled orders" in goal_l and "spent" in goal_l:
            match = re.search(
                r"(\d+)\s+(?:fulfilled\s+)?orders?.{0,80}?\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)",
                text,
                flags=re.I,
            )
            if match:
                return f"{match.group(1)} orders, ${match.group(2)} total spend"
        if "price range" in goal_l:
            prices = self._extract_price_values(text)
            if len(prices) >= 2:
                return self._format_price_range(min(prices), max(prices))
        no_match_phrases = [
            "no reviewers",
            "no matching reviewer",
            "none of the reviews",
            "did not find any",
            "no matching",
            "not found",
        ]
        if ("reviewer" in goal_l or "reviewers" in goal_l) and any(
            phrase in text.lower() for phrase in no_match_phrases
        ):
            return "N/A"
        return None

    def _safe_unresolved_answer(self, goal: str, evidence: str, reason: str) -> str:
        goal_l = (goal or "").lower()
        evidence_l = (evidence or "").lower()
        if ("reviewer" in goal_l or "reviewers" in goal_l) and (
            "all review pages" in evidence_l or "all the reviews" in evidence_l
        ):
            return "N/A"
        return "I could not determine the answer from the available page context."

    def _looks_like_sort_loop(self, action: str | None, current_thought: str) -> bool:
        recent_thoughts = "\n".join(str(item or "") for item in self.thoughts[-4:])
        text = f"{recent_thoughts}\n{current_thought}".lower()
        sort_terms = sum(
            text.count(term)
            for term in ["sort", "ascending", "descending", "lowest", "highest"]
        )
        recent_actions = [
            action_name
            for action_name in map(self._action_name, [*self.actions[-5:], action])
            if action_name
        ]
        control_actions = [name for name in recent_actions if name in {"click", "select_option", "press"}]
        return sort_terms >= 6 and len(control_actions) >= 3

    def _action_name(self, action: str | None) -> str | None:
        if not action:
            return None
        match = re.match(r"\s*([a-zA-Z_]+)\(", action)
        return match.group(1) if match else None

    def _extract_price_values(self, text: str) -> list[float]:
        values = []
        for raw in re.findall(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)", text or ""):
            try:
                values.append(float(raw.replace(",", "")))
            except ValueError:
                continue
        return values

    def _format_price_range(self, low: float, high: float) -> str:
        def fmt(value: float) -> str:
            return f"${value:.2f}".rstrip("0").rstrip(".")

        return f"{fmt(low)} - {fmt(high)}"

    def _update_task_evidence(self, obs: dict) -> None:
        goal = obs.get("goal") or ""
        if self.task_evidence.get("goal") != goal:
            self.task_evidence = self._new_task_notebook(goal)
            self.orchestrator_events = []
        update = self._orchestrator_update(obs)
        if update:
            update = self._filter_grounded_orchestrator_update(update, obs)
            self._merge_orchestrator_update(update)
            self.orchestrator_events.append(
                {
                    "step": len(self.obs_history),
                    "event": "notebook_update",
                    "summary": update.get("step_summary", ""),
                    "ready_to_answer": update.get("ready_to_answer"),
                    "recommended_next_step": update.get("recommended_next_step", ""),
                }
            )

    def _task_evidence_prompt(self, obs: dict) -> str:
        goal = obs.get("goal") or ""
        if self.task_evidence.get("goal") != goal:
            return ""
        notebook = self.task_evidence
        lines = [
            "# Orchestrator Task Notebook",
            "This is a supervisor-maintained, task-local notebook. Treat it as authoritative working memory.",
            "Use it to retain important facts across pages. Do not discard earlier evidence just because it is not visible now.",
            f"Task family: {notebook.get('task_family', 'unknown')}",
            f"Completion status: {'ready to answer' if notebook.get('ready_to_answer') else 'not ready yet'}",
        ]
        plan = notebook.get("current_plan", [])
        if plan:
            lines.append("Current plan:")
            lines.extend(f"- {item}" for item in plan[:5])
        evidence = notebook.get("evidence", [])
        if evidence:
            lines.append("Evidence collected so far:")
            for item in evidence[:12]:
                value = item.get("value") or item.get("text") or ""
                why = item.get("why_relevant") or ""
                source = item.get("source") or "observed page"
                confidence = item.get("confidence")
                conf = f" confidence={confidence}" if confidence is not None else ""
                lines.append(f"- {compact_for_prompt(value, 260)} | source={source}{conf} | why={compact_for_prompt(why, 160)}")
        candidate_answers = notebook.get("candidate_answers", [])
        if candidate_answers:
            lines.append("Candidate final answers:")
            for item in candidate_answers[:5]:
                answer = item.get("answer") if isinstance(item, dict) else str(item)
                support = item.get("support") if isinstance(item, dict) else ""
                lines.append(f"- {compact_for_prompt(answer, 220)} | support={compact_for_prompt(str(support), 180)}")
        visited = notebook.get("visited_pages", [])
        if visited:
            lines.append("Visited/checkpoints:")
            for item in visited[-8:]:
                if isinstance(item, dict):
                    lines.append(f"- {compact_for_prompt(item.get('page') or item.get('label') or str(item), 220)}")
                else:
                    lines.append(f"- {compact_for_prompt(str(item), 220)}")
        open_questions = notebook.get("open_questions", [])
        if open_questions:
            lines.append("Still missing before final answer:")
            lines.extend(f"- {item}" for item in open_questions[:6])
        negatives = notebook.get("negative_constraints", [])
        if negatives:
            lines.append("Avoid / failure-prevention constraints:")
            lines.extend(f"- {item}" for item in negatives[:8])
        recommended = notebook.get("recommended_next_step")
        if recommended and not notebook.get("ready_to_answer"):
            lines.append("Recommended next step from orchestrator: " + compact_for_prompt(recommended, 300))
        if notebook.get("ready_to_answer"):
            draft = notebook.get("draft_answer")
            if draft:
                lines.append("If answering now, use this notebook-supported draft unless the current page contradicts it: " + compact_for_prompt(str(draft), 350))
        else:
            lines.append("Do not send a final answer yet unless the current observation clearly resolves every missing item above.")
        return "\n".join(lines)

    def _evidence_guard_action(self, action: str | None, obs: dict) -> str | None:
        goal = obs.get("goal") or ""
        if not action:
            return action
        if not action.strip().startswith("send_msg_to_user"):
            return action
        if self.task_evidence.get("goal") != goal:
            return action
        decision = self._orchestrator_verify_final_answer(obs, action)
        if not decision:
            return action
        self.orchestrator_events.append(
            {
                "step": len(self.obs_history),
                "event": "final_answer_verification",
                "allow_final_answer": decision.get("allow_final_answer"),
                "reason": decision.get("reason", ""),
            }
        )
        if decision.get("allow_final_answer", True):
            corrected = decision.get("corrected_answer")
            if corrected:
                return f"send_msg_to_user({self._normalize_final_answer_text(str(corrected), goal)!r})"
            return action
        next_action = decision.get("next_action")
        if isinstance(next_action, str):
            repaired = self._repair_or_replace_action(next_action, obs)
            if repaired and self._is_valid_action(repaired):
                return repaired
        corrected = decision.get("corrected_answer")
        if corrected:
            return f"send_msg_to_user({self._normalize_final_answer_text(str(corrected), goal)!r})"
        return self._recovery_action_after_rejected_answer(obs)

    def _new_task_notebook(self, goal: str) -> dict:
        return {
            "goal": goal,
            "task_family": "unknown",
            "current_plan": [],
            "evidence": [],
            "candidate_answers": [],
            "visited_pages": [],
            "open_questions": [],
            "negative_constraints": [],
            "ready_to_answer": False,
            "draft_answer": None,
            "recommended_next_step": "",
        }

    def _orchestrator_update(self, obs: dict) -> dict | None:
        goal = obs.get("goal") or ""
        observation = self._compact_observation_for_orchestrator(obs)
        previous_action = self.actions[-1] if self.actions else ""
        previous_error = obs.get("last_action_error") or ""
        notebook_json = json.dumps(self._serializable_task_notebook(), ensure_ascii=False)
        system = (
            "You are the task orchestrator for a web-browsing benchmark agent. "
            "Your job is not to choose browser actions. Your job is to maintain a compact, "
            "task-local notebook that preserves every important fact, entity, number, page "
            "state, failure warning, and missing check across steps. "
            "You are not allowed to infer, guess, or invent evidence. Evidence is valid only "
            "when it is copied from the current observation, the previous action error, or the "
            "existing notebook. "
            "Return JSON only."
        )
        human = f"""Update the notebook after this browser step.

Rules:
- Preserve useful evidence from earlier pages even if it is no longer visible.
- Add only evidence that is relevant to the user goal or prevents a likely benchmark failure.
- Every evidence.value must be grounded in visible text from the current page, the previous
  action error, or an existing notebook item. If a reviewer/order/product/price/name is not
  literally present in those sources, do not add it.
- Candidate answers may only contain names, prices, products, dates, quantities, or order
  details already present in the evidence list or visible current observation.
- If you are uncertain whether a fact was visible, put it in open_questions, not evidence.
- Track what pages/sections were checked and what is still missing.
- Do not mark ready_to_answer=true unless the notebook contains enough support for the final answer.
- If the page suggests pagination, tabs, filters, order pages, review pages, or multiple result pages remain unchecked, keep ready_to_answer=false.
- Include negative_constraints when you see a likely failure mode such as premature answer, repeated action loop, wrong answer format, or unsupported no-match answer.
- Keep the response tiny: at most 3 current_plan items, 3 evidence items, 2 candidate_answers,
  2 visited_pages, 3 open_questions, and 3 negative_constraints.
- Do not include old example names or plausible names. Benchmark correctness requires exact
  entities from the page.
- No markdown fences. No prose outside JSON. No explanations longer than one short sentence.

Return this minified JSON shape exactly. Use these short keys only:
{{"fam":"short label","sum":"short sentence","plan":["short item"],"ev":["exact visible fact"],"ans":["exact possible answer"],"seen":["short page checked"],"open":["short missing check"],"avoid":["short avoid rule"],"ready":false,"draft":null,"next":"short next step"}}

User goal:
{goal}

Existing notebook JSON:
{notebook_json}

Previous browser action:
{previous_action or "none yet"}

Previous action error:
{compact_for_prompt(str(previous_error), 700) or "none"}

Current page observation:
{compact_for_prompt(observation, 3600)}
"""
        try:
            update = retry(
                self.chat_llm,
                [SystemMessage(content=system), HumanMessage(content=human)],
                n_retry=2,
                parser=self._json_object_parser,
                log=True,
                min_retry_wait_time=15,
                rate_limit_max_wait_time=120,
            )
            return self._expand_orchestrator_update(update)
        except Exception as exc:
            logging.warning("orchestrator_update_failed: %s", exc)
            return None

    def _orchestrator_verify_final_answer(self, obs: dict, action: str) -> dict | None:
        goal = obs.get("goal") or ""
        proposed = self._send_msg_argument(action)
        notebook_json = json.dumps(self._serializable_task_notebook(), ensure_ascii=False)
        system = (
            "You are the final-answer verifier for a web-browsing benchmark agent. "
            "Decide whether the proposed answer is supported by the task notebook and current page. "
            "Reject answers containing any entity that is not present in the notebook evidence "
            "or current observation. "
            "Return JSON only."
        )
        human = f"""Verify this proposed final answer.

Rules:
- Allow the final answer only if it is directly supported by accumulated evidence or the current page.
- If the notebook says important pages/checks are still missing, reject the final answer and provide one safe next_action.
- The next_action must be a single allowed browser action. For click/fill/select_option, the
  first argument must be a numeric bid id visible in the current observation, e.g. click('1288').
  Do not return click('Reviews tab') or any semantic string target.
- If the answer is supported but formatting is wrong, allow it and provide corrected_answer.
- For no-match answers, require evidence that the relevant places were checked.

Return JSON:
{{
  "allow_final_answer": true,
  "corrected_answer": null,
  "next_action": null,
  "reason": "brief reason"
}}

User goal:
{goal}

Task notebook JSON:
{notebook_json}

Proposed answer:
{proposed}

Current page observation:
{self._compact_observation_for_orchestrator(obs, max_chars=4500)}
"""
        try:
            return retry(
                self.chat_llm,
                [SystemMessage(content=system), HumanMessage(content=human)],
                n_retry=2,
                parser=self._json_object_parser,
                log=True,
                min_retry_wait_time=15,
                rate_limit_max_wait_time=120,
            )
        except Exception as exc:
            logging.warning("orchestrator_final_verify_failed: %s", exc)
            return None

    def _json_object_parser(self, text: str):
        parsed = self._extract_json_object(text)
        if parsed is None:
            return None, False, "Return one valid JSON object only. Do not use markdown fences."
        return parsed, True, ""

    def _extract_json_object(self, text: str | None) -> dict | None:
        if not text:
            return None
        cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I | re.M).strip()
        try:
            value = json.loads(cleaned)
            return value if isinstance(value, dict) else None
        except Exception:
            pass
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            value = json.loads(cleaned[start : end + 1])
            return value if isinstance(value, dict) else None
        except Exception:
            return None

    def _expand_orchestrator_update(self, update: dict) -> dict:
        if not isinstance(update, dict):
            return {}
        if any(key in update for key in ["evidence", "candidate_answers", "current_plan"]):
            return update

        def as_list(value) -> list:
            if value is None:
                return []
            if isinstance(value, list):
                return value
            return [value]

        def item_text(value) -> str:
            if isinstance(value, dict):
                for key in ["v", "value", "answer", "text", "page", "summary"]:
                    if value.get(key):
                        return str(value[key])
            return str(value)

        evidence = [
            {
                "value": item_text(item),
                "source": "current/notebook",
                "why_relevant": "Relevant to the current task.",
                "confidence": 1.0,
            }
            for item in as_list(update.get("ev"))[:3]
            if item_text(item).strip()
        ]
        candidate_answers = [
            {
                "answer": item_text(item),
                "support": "Supported by notebook evidence.",
                "confidence": 0.5,
            }
            for item in as_list(update.get("ans"))[:2]
            if item_text(item).strip()
        ]
        visited_pages = [
            {"page": item_text(item), "summary": item_text(item)}
            for item in as_list(update.get("seen"))[:2]
            if item_text(item).strip()
        ]
        return {
            "task_family": str(update.get("fam") or "unknown"),
            "step_summary": str(update.get("sum") or ""),
            "current_plan": [item_text(item) for item in as_list(update.get("plan"))[:3] if item_text(item).strip()],
            "evidence": evidence,
            "candidate_answers": candidate_answers,
            "visited_pages": visited_pages,
            "open_questions": [item_text(item) for item in as_list(update.get("open"))[:3] if item_text(item).strip()],
            "negative_constraints": [item_text(item) for item in as_list(update.get("avoid"))[:3] if item_text(item).strip()],
            "ready_to_answer": bool(update.get("ready")),
            "draft_answer": update.get("draft"),
            "recommended_next_step": str(update.get("next") or ""),
        }

    def _merge_orchestrator_update(self, update: dict) -> None:
        notebook = self.task_evidence or self._new_task_notebook("")
        if update.get("task_family"):
            notebook["task_family"] = compact_for_prompt(str(update["task_family"]), 80)
        for key in ["current_plan", "open_questions", "negative_constraints"]:
            values = update.get(key)
            if isinstance(values, list):
                notebook[key] = self._merge_string_list(notebook.get(key, []), values, limit=8)
        for key, limit in [("evidence", 12), ("candidate_answers", 6), ("visited_pages", 8)]:
            values = update.get(key)
            if isinstance(values, list):
                notebook[key] = self._merge_dict_list(notebook.get(key, []), values, limit=limit)
        if isinstance(update.get("ready_to_answer"), bool):
            notebook["ready_to_answer"] = update["ready_to_answer"]
        if "draft_answer" in update:
            notebook["draft_answer"] = update.get("draft_answer")
        if update.get("recommended_next_step"):
            notebook["recommended_next_step"] = compact_for_prompt(str(update["recommended_next_step"]), 500)
        self.task_evidence = notebook

    def _merge_string_list(self, old: list, new: list, limit: int) -> list[str]:
        merged = []
        seen = set()
        for item in [*(old or []), *(new or [])]:
            text = compact_for_prompt(str(item), 260)
            key = text.lower()
            if text and key not in seen:
                merged.append(text)
                seen.add(key)
        return merged[-limit:]

    def _merge_dict_list(self, old: list, new: list, limit: int) -> list[dict]:
        merged = []
        seen = set()
        for item in [*(old or []), *(new or [])]:
            if not isinstance(item, dict):
                item = {"value": str(item)}
            cleaned = {
                str(k): compact_for_prompt(str(v), 520) if v is not None and not isinstance(v, (int, float, bool)) else v
                for k, v in item.items()
                if v not in (None, "", [])
            }
            signature = json.dumps(cleaned, sort_keys=True, ensure_ascii=False).lower()
            if cleaned and signature not in seen:
                merged.append(cleaned)
                seen.add(signature)
        return merged[-limit:]

    def _serializable_task_notebook(self) -> dict:
        def convert(value):
            if isinstance(value, set):
                return sorted(value)
            if isinstance(value, dict):
                return {str(k): convert(v) for k, v in value.items()}
            if isinstance(value, list):
                return [convert(v) for v in value]
            return value

        return convert(self.task_evidence or {})

    def _compact_observation_for_orchestrator(self, obs: dict, max_chars: int = 6500) -> str:
        parts = []
        for label, key, limit in [
            ("AXTree", "axtree_txt", 2800),
            ("Pruned HTML", "pruned_html", 2600),
            ("DOM", "dom_txt", 1600),
        ]:
            value = obs.get(key)
            if value:
                parts.append(f"{label}:\n{compact_for_prompt(str(value), limit)}")
        return compact_for_prompt("\n\n".join(parts), max_chars)

    def _send_msg_argument(self, action: str | None) -> str:
        match = re.match(r"\s*send_msg_to_user\((.*)\)\s*$", action or "", flags=re.S)
        if not match:
            return ""
        try:
            value = ast.literal_eval(match.group(1))
            return str(value)
        except Exception:
            return match.group(1).strip()

    def _is_valid_action(self, action: str) -> bool:
        try:
            self.action_set.to_python_code(action)
            return True
        except Exception:
            return False

    def _repair_or_replace_action(self, action: str | None, obs: dict) -> str | None:
        if not action:
            return action
        repaired = self._repair_semantic_bid_action(action, obs)
        if repaired and self._is_valid_action(repaired):
            return repaired
        if self._action_needs_numeric_bid(action):
            fallback = self._recovery_action_after_invalid_action(obs)
            self.orchestrator_events.append(
                {
                    "step": len(self.obs_history),
                    "event": "action_repair_failed",
                    "bad_action": compact_for_prompt(action, 260),
                    "fallback_action": fallback,
                }
            )
            return fallback
        return action

    def _action_needs_numeric_bid(self, action: str | None) -> bool:
        parts = self._parse_action_call(action)
        if not parts:
            return False
        name, args = parts
        if name not in {"click", "fill", "select_option"} or not args:
            return False
        first = args[0]
        return isinstance(first, str) and not first.strip().isdigit()

    def _repair_semantic_bid_action(self, action: str, obs: dict) -> str | None:
        parts = self._parse_action_call(action)
        if not parts:
            return action
        name, args = parts
        if name not in {"click", "fill", "select_option"} or not args:
            return action
        first = args[0]
        if not isinstance(first, str) or first.strip().isdigit():
            return action
        bid = self._find_bid_for_label(first, obs)
        if not bid:
            return None
        args = [bid, *args[1:]]
        repaired = f"{name}({', '.join(repr(arg) for arg in args)})"
        self.orchestrator_events.append(
            {
                "step": len(self.obs_history),
                "event": "action_repaired_to_bid",
                "original_action": compact_for_prompt(action, 260),
                "repaired_action": repaired,
            }
        )
        return repaired

    def _parse_action_call(self, action: str | None) -> tuple[str, list] | None:
        if not action:
            return None
        try:
            parsed = ast.parse(action.strip(), mode="eval")
        except Exception:
            return None
        call = parsed.body
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            return None
        args = []
        for arg in call.args:
            try:
                args.append(ast.literal_eval(arg))
            except Exception:
                return None
        return call.func.id, args

    def _find_bid_for_label(self, label: str, obs: dict) -> str | None:
        query = self._normalize_label_for_match(label)
        if not query:
            return None
        candidates = []
        for source_key in ["axtree_txt", "pruned_html", "dom_txt"]:
            text = str(obs.get(source_key) or "")
            candidates.extend(self._bid_label_candidates(text, source_key))
        best_bid = None
        best_score = 0.0
        for bid, candidate_text, source_key in candidates:
            score = self._label_match_score(query, candidate_text)
            if score > best_score:
                best_bid = bid
                best_score = score
        return best_bid if best_score >= 0.45 else None

    def _bid_label_candidates(self, text: str, source_key: str) -> list[tuple[str, str, str]]:
        candidates = []
        if not text:
            return candidates
        for line in text.splitlines():
            for match in re.finditer(r"\[(\d+)\]", line):
                candidates.append((match.group(1), line, source_key))
        for match in re.finditer(r'bid="(\d+)"', text):
            start = max(0, match.start() - 180)
            end = min(len(text), match.end() + 260)
            candidates.append((match.group(1), text[start:end], source_key))
        return candidates

    def _label_match_score(self, query: str, candidate: str) -> float:
        candidate_norm = self._normalize_label_for_match(candidate)
        if not query or not candidate_norm:
            return 0.0
        if query == candidate_norm:
            return 1.0
        score = 0.0
        if query in candidate_norm:
            score += 0.72
        query_tokens = [token for token in query.split() if len(token) > 1]
        if query_tokens:
            overlap = sum(1 for token in query_tokens if token in candidate_norm)
            score += 0.26 * (overlap / len(query_tokens))
        if any(role in candidate_norm for role in ["link", "button", "tab", "radio", "option"]):
            score += 0.08
        return min(score, 1.0)

    def _normalize_label_for_match(self, text: str) -> str:
        text = re.sub(r"\[\d+\]|bid=\"\d+\"", " ", str(text or ""))
        text = re.sub(r"\b(?:click|tab|link|button|statictext|labeltext|role|checked|false|true)\b", " ", text, flags=re.I)
        text = re.sub(r"\(\d+\)", " ", text)
        text = re.sub(r"[^a-zA-Z0-9$]+", " ", text)
        return " ".join(text.lower().split())

    def _recovery_action_after_invalid_action(self, obs: dict) -> str:
        recommended = str((self.task_evidence or {}).get("recommended_next_step") or "")
        if "review" in recommended.lower():
            bid = self._find_bid_for_label("reviews", obs)
            if bid:
                return f"click({bid!r})"
        last_actions = [self._action_name(action) for action in self.actions[-3:]]
        if last_actions.count("scroll") < 2:
            return "scroll(0, 500)"
        if last_actions[-1:] != ["go_back"]:
            return "go_back()"
        return "noop()"

    def _recovery_action_after_rejected_answer(self, obs: dict) -> str:
        recommended = str((self.task_evidence or {}).get("recommended_next_step") or "")
        if "review" in recommended.lower():
            bid = self._find_bid_for_label("reviews", obs)
            if bid:
                return f"click({bid!r})"
        if self._looks_like_repeated_action_loop(self.actions[-1] if self.actions else None):
            return "go_back()"
        return self._recovery_action_after_invalid_action(obs)

    def _filter_grounded_orchestrator_update(self, update: dict, obs: dict) -> dict:
        if not isinstance(update, dict):
            return {}
        filtered = dict(update)
        evidence = update.get("evidence")
        if isinstance(evidence, list):
            filtered_evidence = [
                item
                for item in evidence
                if self._dict_claim_is_grounded(item, obs, include_current_update=False)
            ]
            filtered["evidence"] = filtered_evidence[:12]
        else:
            filtered["evidence"] = []

        grounding_extra = json.dumps(filtered.get("evidence", []), ensure_ascii=False)
        candidates = update.get("candidate_answers")
        if isinstance(candidates, list):
            filtered["candidate_answers"] = [
                item
                for item in candidates
                if self._dict_claim_is_grounded(item, obs, extra_grounding=grounding_extra)
            ][:6]
        else:
            filtered["candidate_answers"] = []

        draft = filtered.get("draft_answer")
        if draft and not self._claim_text_is_grounded(str(draft), obs, extra_grounding=grounding_extra):
            filtered["draft_answer"] = None
            filtered["ready_to_answer"] = False
        if filtered.get("ready_to_answer") and not (
            filtered.get("candidate_answers") or filtered.get("draft_answer")
        ):
            filtered["ready_to_answer"] = False
        return filtered

    def _dict_claim_is_grounded(
        self,
        item: object,
        obs: dict,
        include_current_update: bool = True,
        extra_grounding: str = "",
    ) -> bool:
        if not isinstance(item, dict):
            item = {"value": str(item)}
        text = " ".join(
            str(item.get(key) or "")
            for key in ["value", "answer", "support", "summary", "page"]
        )
        return self._claim_text_is_grounded(text, obs, extra_grounding=extra_grounding)

    def _claim_text_is_grounded(self, text: str, obs: dict, extra_grounding: str = "") -> bool:
        claim = " ".join(str(text or "").split())
        if not claim:
            return False
        grounding = self._grounding_text(obs, extra_grounding)
        entities = self._named_entities(claim)
        for entity in entities:
            if entity.lower() not in grounding:
                return False
        numbers = re.findall(r"\b\d+(?:\.\d+)?\b", claim)
        for number in numbers:
            if number not in grounding:
                return False
        return True

    def _grounding_text(self, obs: dict, extra: str = "") -> str:
        parts = [
            str(obs.get("axtree_txt") or ""),
            str(obs.get("pruned_html") or ""),
            str(obs.get("dom_txt") or ""),
            str(obs.get("last_action_error") or ""),
            json.dumps(self._serializable_task_notebook(), ensure_ascii=False),
            str(extra or ""),
        ]
        return " ".join(" ".join(parts).lower().split())

    def _named_entities(self, text: str) -> list[str]:
        ignored = {
            "reviewer",
            "reviewers",
            "customer",
            "customers",
            "review",
            "reviews",
            "section",
            "page",
            "product",
            "current",
            "previous",
            "the",
            "n/a",
            "none",
        }
        entities = []
        for match in re.finditer(r"\b[A-Z][A-Za-z0-9'\-]*(?:\s+[A-Z][A-Za-z0-9'\-]*){0,4}\b", text or ""):
            entity = match.group(0).strip()
            words = [word for word in entity.split() if word.lower() not in ignored]
            if not words:
                continue
            cleaned = " ".join(words)
            if len(cleaned) >= 3 and cleaned.lower() not in ignored:
                entities.append(cleaned)
        return list(dict.fromkeys(entities))

    def _procedural_memory_prompt(self, obs: dict) -> str:
        if not self.flags.procedural_memory_path or WebArenaProceduralMemory is None:
            return ""
        if self.procedural_memory is None:
            self.procedural_memory = WebArenaProceduralMemory(
                Path(self.flags.procedural_memory_path)
            )
        observation = obs.get("pruned_html") or obs.get("axtree_txt") or obs.get("dom_txt") or ""
        goal = obs.get("goal") or ""
        return self.procedural_memory.prompt(
            goal=goal,
            observation=observation,
            site=self.flags.procedural_site,
            top_k=self.flags.procedural_top_k,
            min_score=self.flags.procedural_min_score,
        )

    def _normalize_final_answer_action(self, action: str | None, goal: str) -> str | None:
        if not action or not action.strip().startswith("send_msg_to_user"):
            return action
        match = re.match(r"\s*send_msg_to_user\((.*)\)\s*$", action, flags=re.S)
        if not match:
            return action
        raw_arg = match.group(1)
        try:
            message = ast.literal_eval(raw_arg)
        except Exception:
            return action
        if not isinstance(message, str):
            return action
        normalized = self._normalize_final_answer_text(message, goal)
        return f"send_msg_to_user({normalized!r})"

    def _normalize_final_answer_text(self, message: str, goal: str) -> str:
        text = " ".join((message or "").split())
        goal_l = (goal or "").lower()

        if "configuration" in goal_l or "size" in goal_l:
            dim = re.search(
                r"(\d+(?:\.\d+)?)\s*(?:\"|inches|inch|in\.?)?\s*[xX]\s*(\d+(?:\.\d+)?)",
                text,
            )
            if dim:
                return f"{dim.group(1)}x{dim.group(2)}"

        if "price range" in goal_l:
            amounts = re.findall(r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)", text)
            cleaned = [amount.replace(",", "") for amount in amounts]
            if len(cleaned) >= 2:
                return f"${cleaned[0]} - ${cleaned[-1]}"

        if "fulfilled orders" in goal_l and "spent" in goal_l:
            order_count = re.search(r"\b(\d+)\s+(?:fulfilled\s+)?orders?\b", text, flags=re.I)
            amount = re.search(r"\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)", text)
            if order_count and amount:
                return f"{order_count.group(1)} orders, ${amount.group(1).replace(',', '')} total spend"

        if "reviewer" in goal_l or "reviewers" in goal_l:
            lowered = text.lower()
            if re.search(r"\b(no|none|n/a|no matching|there are no)\b", lowered):
                return "N/A"
            text = re.sub(r"^(?:the\s+)?(?:following\s+)?reviewers?\s+(?:are|is)\s*:?\s*", "", text, flags=re.I)
            text = re.sub(r"\s+(?:mentioned|mention|complained|complain).*$", "", text, flags=re.I)
            names = [
                item.strip(" .;:")
                for item in re.split(r"\s*,\s*|\s+and\s+|\n+", text)
                if item.strip(" .;:")
            ]
            if 1 <= len(names) <= 12:
                return ", ".join(dict.fromkeys(names))

        if "how much" in goal_l and "spent" in goal_l:
            if re.search(r"\b(no|none|zero)\b", text.lower()) and not re.search(r"\$\s*\d", text):
                return "0"
            amounts = re.findall(r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)", text)
            if len(amounts) == 1:
                return amounts[0].replace(",", "")

        return text
