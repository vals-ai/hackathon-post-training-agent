import argparse
import sys

import httpx2
import pytest
from model_library.base.input import InputItem, RawResponse, TextInput
from model_library.exceptions import MaxContextWindowExceededError, MaxOutputTokensExceededError
from openai import APIError

import post_training_agent.run as run_module

CHECKPOINT = "tinker://run-1/sampler_weights/final"


def _parse(monkeypatch: pytest.MonkeyPatch, *flags: str) -> argparse.Namespace:
    monkeypatch.setenv("TINKER_API_KEY", "tinker-key")
    monkeypatch.delenv("TINKER_BASE_URL", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "post-training-agent",
            "--checkpoint",
            CHECKPOINT,
            *flags,
            "--problem-file",
            "p.txt",
            "--problem-dir",
            ".",
            "--task-id",
            "2024/nile",
            "--output",
            "r.json",
        ],
    )
    return run_module.parse_args()


@pytest.mark.parametrize(("effort", "expected"), [("high", True), ("none", False)])
def test_checkpoint_builds_tinker_model(monkeypatch: pytest.MonkeyPatch, effort: str, expected: bool) -> None:
    llm = run_module.build_llm(_parse(monkeypatch, "--reasoning-effort", effort))

    assert llm.provider == "thinkingmachines"
    assert llm.model_name == CHECKPOINT
    assert llm.reasoning_effort == expected
    assert llm.reasoning is expected
    assert llm.supports_tools
    assert llm.max_tokens == run_module.MAX_OUTPUT_TOKENS
    assert llm.temperature is None
    assert llm.top_p is None


def test_checkpoint_sampling_flags_reach_tinker(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _parse(monkeypatch, "--temperature", "0.6", "--top-p", "0.95", "--max-output-tokens", "8192")

    llm = run_module.build_llm(args)

    assert (llm.temperature, llm.top_p, llm.max_tokens) == (0.6, 0.95, 8192)


OVERFLOW = MaxContextWindowExceededError(
    "PromptTooLongException: 400: Prompt length plus max_tokens exceeds the model's context window: "
    "30000 prompt tokens + 4096 max_tokens > 32000."
)
CLAMPED = MaxOutputTokensExceededError(
    "{'finish_reason': 'length', 'metadata': QueryResultMetadata(\n    in_tokens=30000,\n    out_tokens=813,\n)}"
)


@pytest.mark.parametrize(("error", "kept_exchanges"), [(OVERFLOW, 1), (CLAMPED, 1)])
def test_overflow_frees_a_quarter_of_the_window(error: Exception, kept_exchanges: int) -> None:
    question = TextInput(text="q")
    exchanges = [[RawResponse(response=c * 1000), TextInput(text=c)] for c in "abc"]
    history: list[InputItem] = [question, *[item for exchange in exchanges for item in exchange]]

    kept = [item for exchange in exchanges[-kept_exchanges:] for item in exchange]
    assert run_module.QueryErrorRecovery(4096)(history, error) == [
        question, *kept, TextInput(text=run_module.CONTEXT_OVERFLOW_PROMPT)
    ]


def test_recovery_leaves_normal_history_unchanged_and_raises_unrelated_errors() -> None:
    recovery = run_module.QueryErrorRecovery(4096)
    history: list[InputItem] = [TextInput(text="q")]
    assert recovery(history, None) is history
    error = RuntimeError("unrelated error")
    with pytest.raises(RuntimeError) as raised:
        recovery(history, error)
    assert raised.value is error


def test_repeated_overflows_keep_one_notice_as_the_latest_message() -> None:
    recovery = run_module.QueryErrorRecovery(4096)
    question = TextInput(text="q")
    exchanges = [[RawResponse(response=c * 1000), TextInput(text=c)] for c in "abcdef"]
    history: list[InputItem] = [question, *[item for exchange in exchanges for item in exchange]]
    notice = TextInput(text=run_module.CONTEXT_OVERFLOW_PROMPT)

    for _ in range(2):
        history = recovery(history, OVERFLOW)
        assert history.count(notice) == 1
        assert history[-1] == notice
        assert history[0] is question
        assert exchanges[-1][0] in history


@pytest.mark.parametrize(
    "error",
    [
        MaxOutputTokensExceededError("{'finish_reason': 'length', 'metadata': in_tokens=1100, out_tokens=16384}"),
        MaxOutputTokensExceededError(),
        MaxOutputTokensExceededError("{'finish_reason': 'length', 'metadata': in_tokens=0, out_tokens=0}"),
    ],
)
@pytest.mark.parametrize("with_exchange", [False, True])
def test_output_limit_at_short_context_preserves_history_and_continues(
    error: MaxOutputTokensExceededError, with_exchange: bool
) -> None:
    recovery = run_module.QueryErrorRecovery(16384)
    history: list[InputItem] = [TextInput(text="q")]
    if with_exchange:
        history.extend([RawResponse(response="earlier response"), TextInput(text="tool result")])

    retry = recovery(history, error)

    assert retry == [
        *history,
        TextInput(text=f"An error occurred: {error}. Please review what happened and try a different approach."),
    ]
    assert TextInput(text=run_module.CONTEXT_OVERFLOW_PROMPT) not in retry
    assert recovery(retry, None) is retry


def test_overflow_with_nothing_to_drop_stops() -> None:
    with pytest.raises(run_module.ContextExhausted):
        run_module.QueryErrorRecovery(4096)([TextInput(text="q")], OVERFLOW)


def test_malformed_tool_calls_retry_then_stop() -> None:
    recovery = run_module.QueryErrorRecovery(4096)
    history: list[InputItem] = [TextInput(text="q")]
    parse_error = Exception("RendererOutputParseError: failed to parse model output")
    retry = [*history, TextInput(text=run_module.MALFORMED_TOOL_CALL_PROMPT)]

    assert recovery(history, parse_error) == retry
    assert recovery(history, parse_error) == retry
    assert recovery(history, None) == history
    assert recovery(history, parse_error) == retry
    assert recovery(history, parse_error) == retry
    with pytest.raises(run_module.MalformedToolCallLimit):
        recovery(history, parse_error)


def test_sample_conversion_failures_resample_then_stop() -> None:
    recovery = run_module.QueryErrorRecovery(4096)
    history: list[InputItem] = [TextInput(text="q"), RawResponse(response="earlier work")]
    error = APIError(
        "The model's output could not be converted to an OpenAI response",
        request=httpx2.Request("POST", "https://example.test/sample"),
        body=None,
    )

    for _ in range(3):
        assert recovery(history, error) is history
    assert recovery(history, None) is history
    for _ in range(3):
        assert recovery(history, error) is history
    with pytest.raises(run_module.SampleConversionLimit, match="The model's output could not be converted"):
        recovery(history, error)
