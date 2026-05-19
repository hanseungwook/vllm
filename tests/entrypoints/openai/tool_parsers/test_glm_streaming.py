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

    def tokenize(self, text: str) -> list[str]:
        # The GLM streamer does not use token ids, so we return an empty
        # token list to avoid registering arbitrary substrings in the vocab.
        return []


def make_parser() -> ToolParser:
    return ToolParserManager.get_tool_parser("multi_format")(
        FakeTokenizer(),
        chat_template_kwargs={"tool_format": "glm"},
    )


def make_request(tools: list[dict[str, Any]] | None = None) -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
        tools=tools,
    )


def _tokenized_deltas(text: str) -> list[str]:
    """Split text into deltas at each GLM marker boundary so each delta is
    either a marker or a chunk of payload text. Approximates a token stream
    without depending on a real tokenizer.
    """
    markers = [
        "<tool_call>",
        "</tool_call>",
        "<arg_key>",
        "</arg_key>",
        "<arg_value>",
        "</arg_value>",
    ]
    deltas: list[str] = []
    pos = 0
    while pos < len(text):
        best_idx = -1
        best_marker = ""
        for m in markers:
            i = text.find(m, pos)
            if i == -1:
                continue
            if best_idx == -1 or i < best_idx:
                best_idx = i
                best_marker = m
        if best_idx == -1 or not best_marker:
            deltas.append(text[pos:])
            break
        if best_idx > pos:
            deltas.append(text[pos:best_idx])
        deltas.append(best_marker)
        pos = best_idx + len(best_marker)
    return deltas


def test_single_string_arg():
    text = (
        "<tool_call>get_weather"
        "<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value>"
        "</tool_call>"
    )
    recon = run_tool_extraction_streaming(
        make_parser(), _tokenized_deltas(text), make_request()
    )
    assert recon.other_content == ""
    assert len(recon.tool_calls) == 1
    assert recon.tool_calls[0].function.name == "get_weather"
    assert json.loads(recon.tool_calls[0].function.arguments) == {"city": "Beijing"}
    assert recon.tool_calls[0].id


def test_mixed_argument_types_via_deserialize_glm_value():
    text = (
        "<tool_call>do_thing"
        "<arg_key>s</arg_key><arg_value>hello world</arg_value>"
        "<arg_key>i</arg_key><arg_value>42</arg_value>"
        "<arg_key>b</arg_key><arg_value>true</arg_value>"
        "<arg_key>l</arg_key><arg_value>[1, 2, 3]</arg_value>"
        '<arg_key>d</arg_key><arg_value>{"x": "y"}</arg_value>'
        "</tool_call>"
    )
    recon = run_tool_extraction_streaming(
        make_parser(), _tokenized_deltas(text), make_request()
    )
    assert len(recon.tool_calls) == 1
    args = json.loads(recon.tool_calls[0].function.arguments)
    assert args == {
        "s": "hello world",
        "i": 42,
        "b": True,
        "l": [1, 2, 3],
        "d": {"x": "y"},
    }
    assert isinstance(args["s"], str)
    assert isinstance(args["i"], int)
    assert isinstance(args["b"], bool)
    assert isinstance(args["l"], list)
    assert isinstance(args["d"], dict)


def test_multiple_sequential_tool_calls():
    text = (
        "<tool_call>foo<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        "<tool_call>bar<arg_key>y</arg_key>"
        "<arg_value>two words</arg_value></tool_call>"
    )
    recon = run_tool_extraction_streaming(
        make_parser(), _tokenized_deltas(text), make_request()
    )
    assert len(recon.tool_calls) == 2
    assert recon.tool_calls[0].function.name == "foo"
    assert json.loads(recon.tool_calls[0].function.arguments) == {"x": 1}
    assert recon.tool_calls[1].function.name == "bar"
    assert json.loads(recon.tool_calls[1].function.arguments) == {"y": "two words"}
    assert recon.tool_calls[0].id != recon.tool_calls[1].id


def test_content_prefix_emitted_as_content():
    text = (
        "Planning the call.\n"
        "<tool_call>foo<arg_key>x</arg_key>"
        "<arg_value>1</arg_value></tool_call>"
    )
    recon = run_tool_extraction_streaming(
        make_parser(), _tokenized_deltas(text), make_request()
    )
    assert recon.other_content == "Planning the call.\n"
    assert len(recon.tool_calls) == 1
    assert recon.tool_calls[0].function.name == "foo"
    assert json.loads(recon.tool_calls[0].function.arguments) == {"x": 1}


def test_split_markers_single_char_chunks():
    text = (
        "<tool_call>foo"
        "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
        '<arg_key>filters</arg_key><arg_value>{"unit": "c"}</arg_value>'
        "</tool_call>"
    )
    deltas = list(text)
    recon = run_tool_extraction_streaming(make_parser(), deltas, make_request())
    assert len(recon.tool_calls) == 1
    assert recon.tool_calls[0].function.name == "foo"
    assert json.loads(recon.tool_calls[0].function.arguments) == {
        "city": "Beijing",
        "filters": {"unit": "c"},
    }


def test_split_markers_with_content_prefix_single_char_chunks():
    text = (
        "Hello.\n<tool_call>foo<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
    )
    deltas = list(text)
    recon = run_tool_extraction_streaming(make_parser(), deltas, make_request())
    assert recon.other_content == "Hello.\n"
    assert len(recon.tool_calls) == 1
    assert json.loads(recon.tool_calls[0].function.arguments) == {"x": 1}


def test_string_schema_forces_string_coercion():
    schema_tools = [
        {
            "type": "function",
            "function": {
                "name": "do_thing",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_id": {"type": "string"},
                    },
                },
            },
        }
    ]
    text = (
        "<tool_call>do_thing"
        "<arg_key>user_id</arg_key>"
        "<arg_value>12345</arg_value>"
        "</tool_call>"
    )
    recon = run_tool_extraction_streaming(
        make_parser(), _tokenized_deltas(text), make_request(tools=schema_tools)
    )
    args = json.loads(recon.tool_calls[0].function.arguments)
    assert args == {"user_id": "12345"}
    assert isinstance(args["user_id"], str)


def test_streaming_round_trip_matches_non_streaming():
    text = (
        "Preamble.\n"
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>San Francisco, CA</arg_value>"
        "<arg_key>days</arg_key><arg_value>5</arg_value>"
        "<arg_key>enabled</arg_key><arg_value>true</arg_value>"
        '<arg_key>filters</arg_key><arg_value>{"units": "metric"}</arg_value>'
        "</tool_call>"
        "<tool_call>get_time"
        "<arg_key>tz</arg_key><arg_value>UTC</arg_value>"
        "</tool_call>"
    )
    extracted = run_tool_extraction_nonstreaming(make_parser(), text, make_request())
    recon = run_tool_extraction_streaming(
        make_parser(), _tokenized_deltas(text), make_request()
    )

    assert extracted.tools_called
    assert extracted.content == recon.other_content or extracted.content == (
        recon.other_content or None
    )
    assert len(extracted.tool_calls) == len(recon.tool_calls)
    for non_stream_call, stream_call in zip(extracted.tool_calls, recon.tool_calls):
        assert non_stream_call.function.name == stream_call.function.name
        assert non_stream_call.function.arguments == stream_call.function.arguments


def test_streaming_round_trip_single_char_chunks_matches_non_streaming():
    text = (
        "<tool_call>foo"
        "<arg_key>a</arg_key><arg_value>hi</arg_value>"
        "<arg_key>b</arg_key><arg_value>2</arg_value>"
        "<arg_key>c</arg_key><arg_value>[1, 2]</arg_value>"
        "</tool_call>"
    )
    extracted = run_tool_extraction_nonstreaming(make_parser(), text, make_request())
    recon = run_tool_extraction_streaming(make_parser(), list(text), make_request())
    assert len(extracted.tool_calls) == len(recon.tool_calls) == 1
    assert (
        extracted.tool_calls[0].function.arguments
        == recon.tool_calls[0].function.arguments
    )
