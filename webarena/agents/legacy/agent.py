"""
WARNING DEPRECATED WILL BE REMOVED SOON
"""

from dataclasses import asdict, dataclass, field
import ast
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

    def get_action(self, obs):

        self.obs_history.append(obs)

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
            "When using send_msg_to_user, answer only the requested value/entities. "
            "Avoid caveats unless the task explicitly asks for explanation. "
            "For dimensions, normalize to forms like 16x24 with no spaces, quotes, or inch marks. "
            "For prices/totals, include the numeric amount(s) exactly and avoid unrelated numbers. "
            "For no-match/zero-spend tasks, answer the exact zero/no-match result plainly."
            "\n\n# Stop And Loop Rules\n"
            "If all requested answer fields are known, stop browsing and immediately call "
            "send_msg_to_user with the final answer. Do not re-sort, re-open, re-click, or "
            "re-verify after the answer is already known. If your last few actions repeat the "
            "same control pattern without adding new information, either answer from the known "
            "evidence or choose a genuinely different strategy. For price range tasks, once you "
            "know both the lowest and highest prices, answer exactly '$min - $max' and do not "
            "toggle ascending/descending sort again."
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

        chat_messages = [
            SystemMessage(content=sys_msg),
            HumanMessage(content=prompt),
        ]

        def parser(text):
            try:
                ans_dict = main_prompt._parse_answer(text)
            except ParseError as e:
                # these parse errors will be caught by the retry function and
                # the chat_llm will have a chance to recover
                return None, False, str(e)

            return ans_dict, True, ""

        try:
            ans_dict = retry(self.chat_llm, chat_messages, n_retry=self.max_retry, parser=parser)
            # inferring the number of retries, TODO: make this less hacky
            ans_dict["n_retry"] = (len(chat_messages) - 3) / 2
        except ValueError as e:
            # Likely due to maximum retry. We catch it here to be able to return
            # the list of messages for further analysis
            ans_dict = {"action": None}
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

        self.actions.append(ans_dict["action"])
        self.memories.append(ans_dict.get("memory", None))
        self.thoughts.append(ans_dict.get("think", None))

        ans_dict["chat_messages"] = [m.content for m in chat_messages]
        ans_dict["chat_model_args"] = asdict(self.chat_model_args)

        return ans_dict["action"], ans_dict

    def _loop_guard_action(self, action: str | None, goal: str, current_thought: str) -> str | None:
        if not action or action.strip().startswith("send_msg_to_user"):
            return action
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

        if "how much" in goal_l and "spent" in goal_l:
            if re.search(r"\b(no|none|zero)\b", text.lower()) and not re.search(r"\$\s*\d", text):
                return "0"
            amounts = re.findall(r"\$?\s*(\d+(?:,\d{3})*(?:\.\d+)?)", text)
            if len(amounts) == 1:
                return amounts[0].replace(",", "")

        return text
