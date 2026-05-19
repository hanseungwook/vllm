# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import pytest

from tests.entrypoints.openai.tool_parsers.utils import (
    run_tool_extraction_nonstreaming,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager

pytestmark = pytest.mark.cpu_test


class FakeTokenizer:
    """Streaming-friendly tokenizer matching the pattern in
    ``test_multi_format_tool_parser`` but with a ``tokenize`` method so
    ``run_tool_extraction_streaming`` can compute token deltas."""

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
        reverse = {tid: tok for tok, tid in self._vocab.items()}
        return "".join(reverse[tid] for tid in token_ids)

    def tokenize(self, text: str):
        # The minimax streamer does not consult token ids, so returning an
        # empty list keeps the streaming runner happy without forcing a
        # specific tokenization.
        return []


def make_parser(tool_format: str) -> ToolParser:
    return ToolParserManager.get_tool_parser("multi_format")(
        FakeTokenizer(),
        chat_template_kwargs={"tool_format": tool_format},
    )


def make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="test-model", messages=[])


def _stream(parser: ToolParser, deltas, assert_one_tool_per_delta: bool = True):
    return run_tool_extraction_streaming(
        parser,
        deltas,
        make_request(),
        assert_one_tool_per_delta=assert_one_tool_per_delta,
    )


def _args_of(tool_call) -> dict[str, Any]:
    return json.loads(tool_call.function.arguments)


# ---- minimax --------------------------------------------------------------


def test_minimax_streaming_single_invoke_mixed_types():
    parser = make_parser("minimax")
    text = (
        '<tool_calls><invoke name="get_weather">'
        '<parameter name="city">Tokyo</parameter>'
        '<parameter name="days">5</parameter>'
        '<parameter name="enabled">true</parameter>'
        '<parameter name="filters">{"units":"metric"}</parameter>'
        "</invoke></tool_calls>"
    )

    reconstructor = _stream(parser, list(text))

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 1
    tool = reconstructor.tool_calls[0]
    assert tool.id
    assert tool.function.name == "get_weather"
    assert _args_of(tool) == {
        "city": "Tokyo",
        "days": 5,
        "enabled": True,
        "filters": {"units": "metric"},
    }


def test_minimax_streaming_multiple_invokes_in_one_block():
    parser = make_parser("minimax")
    text = (
        "<tool_calls>"
        '<invoke name="get_weather">'
        '<parameter name="city">Tokyo</parameter>'
        "</invoke>"
        '<invoke name="get_time">'
        '<parameter name="timezone">UTC</parameter>'
        "</invoke>"
        "</tool_calls>"
    )

    reconstructor = _stream(parser, list(text))

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 2
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert _args_of(reconstructor.tool_calls[0]) == {"city": "Tokyo"}
    assert reconstructor.tool_calls[1].function.name == "get_time"
    assert _args_of(reconstructor.tool_calls[1]) == {"timezone": "UTC"}
    # Each tool call gets a distinct id.
    assert reconstructor.tool_calls[0].id != reconstructor.tool_calls[1].id


def test_minimax_streaming_content_prefix_is_emitted():
    parser = make_parser("minimax")
    text = (
        "Thinking it through.\n"
        '<tool_calls><invoke name="echo">'
        '<parameter name="value">hi</parameter>'
        "</invoke></tool_calls>"
    )

    reconstructor = _stream(parser, list(text))

    assert reconstructor.other_content == "Thinking it through.\n"
    assert len(reconstructor.tool_calls) == 1
    assert _args_of(reconstructor.tool_calls[0]) == {"value": "hi"}


def test_minimax_streaming_splits_markers_across_single_char_deltas():
    parser = make_parser("minimax")
    text = (
        '<tool_calls><invoke name="f">'
        '<parameter name="x">1</parameter>'
        '<parameter name="y">two</parameter>'
        "</invoke></tool_calls>"
    )

    # Pass single-character chunks to force every marker boundary to be split
    # across multiple feed() calls.
    reconstructor = _stream(parser, list(text))

    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 1
    assert _args_of(reconstructor.tool_calls[0]) == {"x": 1, "y": "two"}


def test_minimax_streaming_empty_parameter_value():
    parser = make_parser("minimax")
    text = (
        '<tool_calls><invoke name="f">'
        '<parameter name="x"></parameter>'
        "</invoke></tool_calls>"
    )

    reconstructor = _stream(parser, list(text))

    assert len(reconstructor.tool_calls) == 1
    assert _args_of(reconstructor.tool_calls[0]) == {"x": ""}


# ---- dsv32 ----------------------------------------------------------------


def test_dsv32_streaming_string_true_keeps_raw_value():
    parser = make_parser("dsv32")
    text = (
        '<tool_calls><invoke name="echo">'
        '<parameter name="text" string="true">  hello  </parameter>'
        '<parameter name="number" string="true">42</parameter>'
        "</invoke></tool_calls>"
    )

    reconstructor = _stream(parser, list(text))

    assert len(reconstructor.tool_calls) == 1
    # ``string="true"`` preserves whitespace and never coerces to int / bool.
    args = _args_of(reconstructor.tool_calls[0])
    assert args == {"text": "  hello  ", "number": "42"}
    assert isinstance(args["number"], str)


def test_dsv32_streaming_string_false_parses_value():
    parser = make_parser("dsv32")
    text = (
        '<tool_calls><invoke name="example">'
        '<parameter name="count" string="false">42</parameter>'
        '<parameter name="enabled" string="false">false</parameter>'
        '<parameter name="nested" string="false">{"k":"v"}</parameter>'
        "</invoke></tool_calls>"
    )

    reconstructor = _stream(parser, list(text))

    args = _args_of(reconstructor.tool_calls[0])
    assert args == {
        "count": 42,
        "enabled": False,
        "nested": {"k": "v"},
    }


def test_dsv32_streaming_missing_string_attr_parses_value():
    parser = make_parser("dsv32")
    text = (
        '<tool_calls><invoke name="example">'
        '<parameter name="count">7</parameter>'
        "</invoke></tool_calls>"
    )

    reconstructor = _stream(parser, list(text))

    assert _args_of(reconstructor.tool_calls[0]) == {"count": 7}


# ---- round-trip vs non-streaming ------------------------------------------


@pytest.mark.parametrize(
    "tool_format,text",
    [
        (
            "minimax",
            "Calling now.\n"
            '<tool_calls><invoke name="get_weather">'
            '<parameter name="city">Tokyo</parameter>'
            '<parameter name="days">5</parameter>'
            '<parameter name="enabled">true</parameter>'
            '<parameter name="filters">{"units":"metric"}</parameter>'
            "</invoke></tool_calls>",
        ),
        (
            "dsv32",
            "Prefix.\n"
            '<tool_calls><invoke name="get_weather">'
            '<parameter name="city" string="true">San Francisco, CA</parameter>'
            '<parameter name="days" string="false">5</parameter>'
            '<parameter name="enabled" string="false">false</parameter>'
            "</invoke></tool_calls>",
        ),
    ],
)
def test_streaming_round_trip_matches_non_streaming(tool_format: str, text: str):
    parser_stream = make_parser(tool_format)
    parser_full = make_parser(tool_format)

    streamed = _stream(parser_stream, list(text))
    extracted = run_tool_extraction_nonstreaming(parser_full, text, make_request())

    # Content prefix must round-trip.
    assert (streamed.other_content or None) == extracted.content

    assert extracted.tools_called
    assert len(streamed.tool_calls) == len(extracted.tool_calls)
    for streamed_call, extracted_call in zip(streamed.tool_calls, extracted.tool_calls):
        assert streamed_call.function.name == extracted_call.function.name
        # Concatenated streamed arguments must equal the non-streaming
        # arguments string byte-for-byte.
        assert streamed_call.function.arguments == extracted_call.function.arguments
