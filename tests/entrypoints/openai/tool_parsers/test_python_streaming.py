# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest

from tests.entrypoints.openai.tool_parsers.utils import (
    run_tool_extraction_nonstreaming,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager

pytestmark = pytest.mark.cpu_test


class FakeTokenizer:
    """Minimal tokenizer for streaming tests.

    The Python streamer ignores token IDs, so ``tokenize`` returns an
    empty list and ``encode``/``decode`` keep the streaming test util
    happy.
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next_token_id = 1

    def get_vocab(self):
        return self._vocab

    def encode(self, text: str, add_special_tokens: bool = False):
        if text not in self._vocab:
            self._vocab[text] = self._next_token_id
            self._next_token_id += 1
        return [self._vocab[text]]

    def decode(self, token_ids):
        reverse_vocab = {tid: tok for tok, tid in self._vocab.items()}
        return "".join(reverse_vocab[tid] for tid in token_ids)

    def tokenize(self, text: str):
        return []


def make_parser() -> ToolParser:
    return ToolParserManager.get_tool_parser("multi_format")(
        FakeTokenizer(),
        chat_template_kwargs={"tool_format": "python"},
    )


def make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="test-model", messages=[])


def _stream(parser: ToolParser, deltas):
    return run_tool_extraction_streaming(
        parser,
        deltas,
        make_request(),
        assert_one_tool_per_delta=False,
    )


def test_streaming_single_string_arg():
    parser = make_parser()
    output = '<tool_call>\nget_weather(city="SF")\n</tool_call>'

    reconstructor = _stream(parser, [output])

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 1
    tool_call = reconstructor.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {"city": "SF"}
    assert tool_call.id


def test_streaming_mixed_types():
    parser = make_parser()
    output = (
        "<tool_call>\n"
        "register_user("
        'name="John", '
        "age=37, "
        "active=True, "
        "manager=None, "
        'aliases=["Johnny", "JD"], '
        'meta={"team": "infra", "lead": False}'
        ")\n"
        "</tool_call>"
    )

    reconstructor = _stream(parser, [output])

    assert len(reconstructor.tool_calls) == 1
    tool_call = reconstructor.tool_calls[0]
    assert tool_call.function.name == "register_user"
    assert json.loads(tool_call.function.arguments) == {
        "name": "John",
        "age": 37,
        "active": True,
        "manager": None,
        "aliases": ["Johnny", "JD"],
        "meta": {"team": "infra", "lead": False},
    }


def test_streaming_multiple_calls_increments_index():
    parser = make_parser()
    output = (
        '<tool_call>\nget_weather(city="SF")\n</tool_call>'
        "\n"
        '<tool_call>\nget_time(timezone="UTC")\n</tool_call>'
    )

    reconstructor = _stream(parser, [output])

    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {"city": "SF"}
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {
        "timezone": "UTC"
    }


def test_streaming_emits_content_prefix():
    parser = make_parser()
    output = (
        "Here is the call I will make:\n"
        '<tool_call>\nget_weather(city="SF")\n</tool_call>'
    )

    reconstructor = _stream(parser, list(output))

    assert reconstructor.other_content == "Here is the call I will make:\n"
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {"city": "SF"}


def test_streaming_handles_single_char_chunks_across_markers():
    parser = make_parser()
    output = (
        "preamble "
        '<tool_call>\nget_weather(city="SF", unit="celsius")\n</tool_call>'
        " trailing"
        '<tool_call>\nget_time(timezone="UTC")\n</tool_call>'
    )

    reconstructor = _stream(parser, list(output))

    assert reconstructor.other_content == "preamble "
    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "SF",
        "unit": "celsius",
    }
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {
        "timezone": "UTC"
    }


def test_streaming_no_tool_call_emits_only_content():
    parser = make_parser()
    output = "Just chatting, no tools here."

    reconstructor = _stream(parser, list(output))

    assert reconstructor.other_content == output
    assert reconstructor.tool_calls == []


def test_streaming_does_not_leak_partial_start_marker_as_content():
    parser = make_parser()
    output = 'hi <tool_call>\nget_weather(city="SF")\n</tool_call>'

    reconstructor = _stream(parser, list(output))

    # The "<tool_call>" prefix must never be emitted as content even when
    # split character-by-character across deltas.
    assert reconstructor.other_content == "hi "
    assert len(reconstructor.tool_calls) == 1


@pytest.mark.parametrize(
    "model_output",
    [
        '<tool_call>\nget_weather(city="SF")\n</tool_call>',
        (
            "<tool_call>\n"
            'get_weather(city="SF", meta={"enabled": true, "missing": null})\n'
            "</tool_call>"
        ),
        (
            '<tool_call>\nget_weather(city="SF")\n</tool_call>'
            "\n"
            '<tool_call>\nget_time(timezone="UTC")\n</tool_call>'
        ),
        (
            "Here's the plan:\n"
            "<tool_call>\n"
            'get_weather(location="San Francisco, CA", unit="celsius")\n'
            "</tool_call>"
        ),
    ],
)
def test_streaming_matches_nonstreaming(model_output):
    expected = run_tool_extraction_nonstreaming(
        make_parser(), model_output, make_request()
    )
    reconstructor = _stream(make_parser(), list(model_output))

    assert len(reconstructor.tool_calls) == len(expected.tool_calls)
    for streamed, ref in zip(reconstructor.tool_calls, expected.tool_calls):
        assert streamed.function.name == ref.function.name
        assert json.loads(streamed.function.arguments) == json.loads(
            ref.function.arguments
        )
    assert (reconstructor.other_content or None) == expected.content
