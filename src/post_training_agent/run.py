"""Post-Training agent entry point for Valkyrie."""

import argparse
import asyncio
import logging
import os
import re
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from model_library.agent import (
    Agent,
    AgentConfig,
    AgentHooks,
    AgentTurn,
    ErrorTurn,
    SerializableException,
    TurnLimit,
    TurnResult,
    truncate_oldest,
)
from model_library.base import LLM, LLMConfig
from model_library.base.input import InputItem, TextInput
from model_library.exceptions import (
    MaxContextWindowExceededError,
    MaxOutputTokensExceededError,
    RetryException,
    exception_http_status_code,
)
from model_library.registry_utils import get_raw_model, get_registry_model
from model_library.retriers.backoff import ExponentialBackoffRetrier
from model_library.retriers.base import retry_decorator
from pydantic import SecretStr

from post_training_agent._permissions import LOGS_DIR, secure_directory
from post_training_agent.tools import CppExecutor, Submission
from post_training_agent.utils import INSTRUCTIONS_PROMPT

MAX_TURNS = 100
MAX_OUTPUT_TOKENS = 16384
MAX_PARSE_FAILURES = 3
MAX_SAMPLE_FAILURES = 4
MALFORMED_TOOL_CALL_PROMPT = "Your last tool call was malformed and could not be parsed. Retry it with valid arguments."
CONTEXT_OVERFLOW_PROMPT = "Older turns were dropped because the context window was exceeded."
FREE_WINDOW_SHARE = 0.25
TINKER_REQUEST_TIMEOUT = 600
TINKER_RETRY_MAX_TIME = 1200
PROMPT_OVERFLOW = re.compile(r"(\d+) prompt tokens \+ \d+ max_tokens > (\d+)")
OUTPUT_CLAMPED = re.compile(r"in_tokens=(\d+).*?out_tokens=(\d+)", re.DOTALL)


def _post_training_should_stop(turn_result: TurnResult) -> bool:
    """Stop if agent says EXIT (only when there are no tool calls)."""
    if not turn_result.tool_calls and turn_result.response_text:
        if re.search(r"\bEXIT\b", turn_result.response_text, re.IGNORECASE):
            return True
    return False


def _post_training_determine_answer(
    state: dict[str, Any],
    turns: list[AgentTurn | ErrorTurn],
    final_error: SerializableException | None,
) -> str:
    if state.get("has_submissions"):
        return f"Saved a solution; it passed {state['samples_passed']} sample test(s)"
    return "Agent never saved a compiling solution"


class MalformedToolCallLimit(Exception):
    pass


class ContextExhausted(Exception):
    pass


class SampleConversionLimit(Exception):
    pass


def _tokens_to_free(error: Exception, max_output_tokens: int) -> tuple[int, float]:
    message = str(error)
    if overflow := PROMPT_OVERFLOW.search(message):
        prompt, window = int(overflow[1]), int(overflow[2])
    elif clamped := OUTPUT_CLAMPED.search(message):
        prompt, window = int(clamped[1]), int(clamped[1]) + int(clamped[2])
    else:
        return 0, 0
    return prompt, prompt + max_output_tokens - (1 - FREE_WINDOW_SHARE) * window


def _chars(history: list[InputItem]) -> int:
    return sum(len(str(item)) for item in history)


def drop_oldest_until_room(history: list[InputItem], error: Exception, max_output_tokens: int) -> list[InputItem]:
    prompt_tokens, excess = _tokens_to_free(error, max_output_tokens)
    tokens_per_char = prompt_tokens / _chars(history)
    freed = 0.0
    while True:
        try:
            shorter = truncate_oldest(history)
        except ValueError:
            if freed == 0:
                raise ContextExhausted(str(error)) from error
            return history
        freed += (_chars(history) - _chars(shorter)) * tokens_per_char
        history = shorter
        if freed >= excess:
            return history


class QueryErrorRecovery:
    def __init__(self, max_output_tokens: int) -> None:
        self.max_output_tokens = max_output_tokens
        self.parse_failures = 0
        self.sample_failures = 0

    def __call__(self, history: list[InputItem], last_error: Exception | None) -> list[InputItem]:
        if last_error is not None and "The model's output could not be converted to an OpenAI response" in str(last_error):
            self.sample_failures += 1
            if self.sample_failures >= MAX_SAMPLE_FAILURES:
                raise SampleConversionLimit(str(last_error)) from last_error
            return history
        self.sample_failures = 0
        if last_error is not None and "RendererOutputParseError" in str(last_error):
            self.parse_failures += 1
            if self.parse_failures >= MAX_PARSE_FAILURES:
                raise MalformedToolCallLimit(str(last_error))
            return [*history, TextInput(text=MALFORMED_TOOL_CALL_PROMPT)]
        self.parse_failures = 0
        if isinstance(last_error, MaxOutputTokensExceededError):
            clamped = OUTPUT_CLAMPED.search(str(last_error))
            if clamped is None or int(clamped[1]) == 0 or int(clamped[2]) == 0 or int(clamped[2]) >= self.max_output_tokens:
                return [
                    *history,
                    TextInput(
                        text=f"An error occurred: {last_error}. Please review what happened and try a different approach."
                    ),
                ]
        if isinstance(last_error, (MaxContextWindowExceededError, MaxOutputTokensExceededError)):
            shorter = drop_oldest_until_room(history, last_error, self.max_output_tokens)
            return [
                *[item for item in shorter if not (isinstance(item, TextInput) and item.text == CONTEXT_OVERFLOW_PROMPT)],
                TextInput(text=CONTEXT_OVERFLOW_PROMPT),
            ]
        if last_error:
            raise last_error
        return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-Training Agent")
    model = parser.add_mutually_exclusive_group(required=True)
    model.add_argument("--model", help="Registry model key")
    model.add_argument("--checkpoint", help="Tinker sampler checkpoint (tinker://.../sampler_weights/...)")
    parser.add_argument(
        "--reasoning-effort", default="high", choices=["high", "none"], help="Tinker thinking on or off"
    )
    parser.add_argument("--temperature", type=float, help="Tinker sampling temperature")
    parser.add_argument("--top-p", type=float, help="Tinker nucleus sampling top_p")
    parser.add_argument(
        "--max-output-tokens", type=int, default=MAX_OUTPUT_TOKENS, help="Tinker output token cap per turn"
    )
    parser.add_argument("--problem-file", required=True, help="Path to problem statement file")
    parser.add_argument("--problem-dir", required=True, help="Workspace holding samples/ and receiving solution.cpp")
    parser.add_argument("--task-id", required=True, help="Task identifier (e.g. median)")
    parser.add_argument("--output", required=True, help="Path to write result JSON")
    return parser.parse_args()


class TinkerRetrier(ExponentialBackoffRetrier):
    def _should_retry(self, exception: Exception) -> bool:
        status = exception_http_status_code(exception)
        if status is not None and 400 <= status < 500 and status != 429:
            return False
        return super()._should_retry(exception)


def tinker_retry(query: Callable[[], Awaitable[Any]]) -> Callable[[], Awaitable[Any]]:
    async def attempt() -> Any:
        try:
            return await asyncio.wait_for(query(), TINKER_REQUEST_TIMEOUT)
        except TimeoutError as e:
            raise RetryException(f"Tinker request timed out after {TINKER_REQUEST_TIMEOUT}s") from e

    retrier = TinkerRetrier(logging.getLogger(__name__), max_time=TINKER_RETRY_MAX_TIME)
    return retry_decorator(retrier)(attempt)


def build_llm(args: argparse.Namespace) -> LLM:
    if args.checkpoint is None:
        return get_registry_model(args.model)
    llm = get_raw_model(
        f"thinkingmachines/{args.checkpoint}",
        LLMConfig(
            custom_api_key=SecretStr(os.environ["TINKER_API_KEY"]),
            reasoning=args.reasoning_effort != "none",
            reasoning_effort=args.reasoning_effort != "none",
            supports_tools=True,
            max_tokens=args.max_output_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            custom_endpoint=os.environ.get("TINKER_BASE_URL"),
        ),
    )
    llm.custom_retrier = tinker_retry
    return llm


async def run(args: argparse.Namespace) -> None:
    os.umask(0o077)
    secure_directory(Path(args.problem_dir), private=False)
    secure_directory(LOGS_DIR, private=True)
    full_problem_statement = Path(args.problem_file).read_text()
    llm = build_llm(args)

    tools = [
        CppExecutor(),
        Submission(problem_path=args.problem_dir),
    ]

    agent = Agent(
        llm=llm,
        tools=tools,
        name="post_training",
        log_dir=LOGS_DIR,
        config=AgentConfig(
            turn_limit=TurnLimit(max_turns=MAX_TURNS),
            time_limit=None,
        ),
        hooks=AgentHooks(
            should_stop=_post_training_should_stop,
            determine_answer=_post_training_determine_answer,
            before_query=QueryErrorRecovery(args.max_output_tokens),
        ),
    )

    result = await agent.run(
        [TextInput(text=INSTRUCTIONS_PROMPT.format(question=full_problem_statement))],
        question_id=args.task_id.replace("/", "_"),
        atif_export=True,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2))

    if (
        result.stop_reason == "error"
        and result.final_error
        and result.final_error.type not in (
            MalformedToolCallLimit.__name__, ContextExhausted.__name__, SampleConversionLimit.__name__
        )
    ):
        print(f"Agent error: {result.final_error}", file=sys.stderr)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
