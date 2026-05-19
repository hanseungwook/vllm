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
    """Minimal tokenizer that registers each delta string as a single token."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next_token_id = 1

    def _ensure_token(self, text: str) -> int:
        if text not in self._vocab:
            self._vocab[text] = self._next_token_id
            self._next_token_id += 1
        return self._vocab[text]

    def get_vocab(self):
        return self._vocab

    def encode(self, text: str, add_special_tokens: bool = False):
        return [self._ensure_token(text)]

    def decode(self, token_ids):
        reverse_vocab = {token_id: token for token, token_id in self._vocab.items()}
        return "".join(reverse_vocab[token_id] for token_id in token_ids)

    def tokenize(self, text: str) -> list[str]:
        self._ensure_token(text)
        return [text]


def make_parser() -> ToolParser:
    return ToolParserManager.get_tool_parser("multi_format")(
        FakeTokenizer(),
        chat_template_kwargs={"tool_format": "gptoss"},
    )


def make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(model="test-model", messages=[])


def _stream(chunks: list[str]):
    return run_tool_extraction_streaming(make_parser(), chunks, make_request())


def test_single_tool_call_without_assistant_prefix():
    chunks = [
        "<tool_call>",
        "to=functions.get_weather json\n",
        '{"location":"SF"}',
        "\n</tool_call>",
    ]
    reconstructor = _stream(chunks)
    assert reconstructor.other_content == ""
    assert len(reconstructor.tool_calls) == 1
    tool_call = reconstructor.tool_calls[0]
    assert tool_call.id
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {"location": "SF"}


def test_single_tool_call_with_assistant_prefix():
    chunks = [
        "<tool_call>assistant to=functions.get_weather json\n",
        '{"location": "San Francisco, CA", "unit": "celsius"}',
        "\n</tool_call>",
    ]
    reconstructor = _stream(chunks)
    assert len(reconstructor.tool_calls) == 1
    tool_call = reconstructor.tool_calls[0]
    assert tool_call.function.name == "get_weather"
    assert json.loads(tool_call.function.arguments) == {
        "location": "San Francisco, CA",
        "unit": "celsius",
    }


def test_multiple_sequential_tool_calls():
    chunks = [
        "<tool_call>to=functions.get_weather json\n",
        '{"location":"SF"}',
        "\n</tool_call>",
        "<tool_call>to=functions.get_time json\n",
        '{"timezone":"UTC"}',
        "\n</tool_call>",
    ]
    reconstructor = _stream(chunks)
    assert len(reconstructor.tool_calls) == 2
    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "location": "SF",
    }
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {
        "timezone": "UTC",
    }
    # Each tool call should have its own id.
    ids = [tc.id for tc in reconstructor.tool_calls]
    assert len(set(ids)) == 2


def test_json_args_streamed_across_multiple_deltas():
    chunks = [
        "<tool_call>to=functions.foo json\n",
        "{",
        '"a"',
        ": 1, ",
        '"b": ',
        '"hello"',
        ", ",
        '"c": [1, 2, 3]',
        "}",
        "\n</tool_call>",
    ]
    reconstructor = _stream(chunks)
    assert len(reconstructor.tool_calls) == 1
    tool_call = reconstructor.tool_calls[0]
    assert tool_call.function.name == "foo"
    assert json.loads(tool_call.function.arguments) == {
        "a": 1,
        "b": "hello",
        "c": [1, 2, 3],
    }


def test_content_prefix_is_emitted_as_content():
    chunks = [
        "Planning...",
        "\n",
        "<tool_call>to=functions.get_weather json\n",
        '{"location":"SF"}',
        "\n</tool_call>",
    ]
    reconstructor = _stream(chunks)
    assert reconstructor.other_content == "Planning...\n"
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "location": "SF",
    }


def test_content_between_tool_calls_is_dropped():
    # Non-streaming includes only the prefix before the first call as content;
    # streaming must do the same and not leak text between/after calls.
    chunks = [
        "<tool_call>to=functions.a json\n",
        '{"x":1}',
        "\n</tool_call>",
        "ignored between calls",
        "<tool_call>to=functions.b json\n",
        '{"y":2}',
        "\n</tool_call>",
        "trailing ignored",
    ]
    reconstructor = _stream(chunks)
    assert reconstructor.other_content == ""
    assert [tc.function.name for tc in reconstructor.tool_calls] == ["a", "b"]


def test_split_markers_single_char_chunks():
    text = (
        "Hi.\n"
        "<tool_call>to=functions.do_thing json\n"
        '{"x":1,"y":[1,2,3],"z":"<not a tag>"}\n'
        "</tool_call>"
    )
    reconstructor = _stream(list(text))
    assert reconstructor.other_content == "Hi.\n"
    assert len(reconstructor.tool_calls) == 1
    tool_call = reconstructor.tool_calls[0]
    assert tool_call.function.name == "do_thing"
    assert json.loads(tool_call.function.arguments) == {
        "x": 1,
        "y": [1, 2, 3],
        "z": "<not a tag>",
    }


def test_round_trip_matches_non_streaming():
    text = (
        "Planning...\n"
        "<tool_call>to=functions.get_weather json\n"
        '{"location":"SF"}\n'
        "</tool_call>"
        "<tool_call>assistant to=functions.get_time json\n"
        '{"timezone":"UTC"}\n'
        "</tool_call>"
    )

    nonstream = run_tool_extraction_nonstreaming(make_parser(), text, make_request())
    reconstructor = _stream(list(text))

    assert nonstream.tools_called
    assert (nonstream.content or "") == reconstructor.other_content
    assert len(nonstream.tool_calls) == len(reconstructor.tool_calls)
    for ns, st in zip(nonstream.tool_calls, reconstructor.tool_calls):
        assert ns.function.name == st.function.name
        assert json.loads(ns.function.arguments) == json.loads(st.function.arguments)


def test_no_tool_call_emits_only_content():
    chunks = ["just plain text, no tool calls here."]
    reconstructor = _stream(chunks)
    assert reconstructor.other_content == "just plain text, no tool calls here."
    assert reconstructor.tool_calls == []


def test_state_resets_on_empty_previous_text():
    # Run two streaming sessions on the same parser instance and confirm the
    # second one starts fresh (tool index back to 0).
    parser = make_parser()
    first_session = run_tool_extraction_streaming(
        parser,
        [
            "<tool_call>to=functions.first json\n",
            '{"a":1}',
            "\n</tool_call>",
        ],
        make_request(),
    )
    assert len(first_session.tool_calls) == 1

    second_session = run_tool_extraction_streaming(
        parser,
        [
            "<tool_call>to=functions.second json\n",
            '{"b":2}',
            "\n</tool_call>",
        ],
        make_request(),
    )
    assert len(second_session.tool_calls) == 1
    assert second_session.tool_calls[0].function.name == "second"
    assert json.loads(second_session.tool_calls[0].function.arguments) == {"b": 2}
