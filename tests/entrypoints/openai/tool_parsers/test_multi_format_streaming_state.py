# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for ``serving_chat.py`` end-of-stream state on the
``MultiFormatToolParser``. After streaming completes, the parser instance's
inherited ``prev_tool_call_arr`` and ``streamed_args_for_tool`` fields MUST
reflect the streamed tool calls so ``serving_chat`` can:

  * set ``finish_reason="tool_calls"`` (``len(prev_tool_call_arr) > 0``)
  * compute a no-op unstreamed-args flush
    (``json.dumps(prev_tool_call_arr[i]['arguments'])`` matches
    ``streamed_args_for_tool[i]``).
"""

import json
from typing import Any

import pytest

from tests.entrypoints.openai.tool_parsers.utils import (
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager

pytestmark = pytest.mark.cpu_test


class _FakeTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._next = 1

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if text not in self._vocab:
            self._vocab[text] = self._next
            self._next += 1
        return [self._vocab[text]]

    def decode(self, token_ids: list[int]) -> str:
        rev = {v: k for k, v in self._vocab.items()}
        return "".join(rev[t] for t in token_ids)

    def tokenize(self, text: str) -> list[str]:
        return []


def _make_parser(tool_format: str) -> ToolParser:
    return ToolParserManager.get_tool_parser("multi_format")(
        _FakeTokenizer(),
        chat_template_kwargs={"tool_format": tool_format},
    )


def _run(parser: ToolParser, model_output: str) -> None:
    run_tool_extraction_streaming(
        parser,
        list(model_output),
        ChatCompletionRequest(model="test-model", messages=[]),
        assert_one_tool_per_delta=False,
    )


def _assert_streamed_state(
    parser: ToolParser,
    expected_calls: list[tuple[str, dict[str, Any]]],
) -> None:
    """``serving_chat.py`` invariants on the parser instance."""
    assert len(parser.prev_tool_call_arr) == len(expected_calls), (
        f"expected {len(expected_calls)} tool calls, got "
        f"{parser.prev_tool_call_arr!r}"
    )
    assert len(parser.streamed_args_for_tool) == len(expected_calls)
    for i, (name, args) in enumerate(expected_calls):
        entry = parser.prev_tool_call_arr[i]
        assert entry["name"] == name
        assert entry["arguments"] == args, (
            f"call {i}: expected args {args}, got {entry['arguments']}"
        )
        expected_json = json.dumps(args, ensure_ascii=False)
        actual_json = parser.streamed_args_for_tool[i]
        # serving_chat does ``expected_call.replace(actual_call, "", 1)`` and
        # expects a no-op for streamers that emit complete coerced JSON.
        assert actual_json == expected_json, (
            f"call {i}: streamed JSON {actual_json!r} must equal final "
            f"JSON {expected_json!r}"
        )


# ---------- IFM ----------


def test_ifm_xml_populates_state():
    parser = _make_parser("xml")
    _run(
        parser,
        "<ifm|tool_calls>"
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>SF</ifm|arg_value>"
        "</ifm|tool_call>"
        "</ifm|tool_calls>",
    )
    _assert_streamed_state(parser, [("get_weather", {"city": "SF"})])


def test_ifm_json_populates_state():
    parser = _make_parser("json")
    _run(
        parser,
        '<ifm|tool_call>{"name":"get_weather","arguments":{"city":"SF"}}'
        "</ifm|tool_call>",
    )
    _assert_streamed_state(parser, [("get_weather", {"city": "SF"})])


# ---------- GLM ----------


def test_glm_populates_state():
    parser = _make_parser("glm")
    _run(
        parser,
        "<tool_call>get_weather"
        "<arg_key>city</arg_key><arg_value>SF</arg_value>"
        "</tool_call>",
    )
    _assert_streamed_state(parser, [("get_weather", {"city": "SF"})])


# ---------- Minimax / DSV32 ----------


def test_minimax_populates_state():
    parser = _make_parser("minimax")
    _run(
        parser,
        "<tool_calls>"
        '<invoke name="get_weather">'
        '<parameter name="city">SF</parameter>'
        "</invoke>"
        "</tool_calls>",
    )
    _assert_streamed_state(parser, [("get_weather", {"city": "SF"})])


def test_dsv32_populates_state():
    parser = _make_parser("dsv32")
    _run(
        parser,
        "<tool_calls>"
        '<invoke name="get_weather">'
        '<parameter name="city" string="true">SF</parameter>'
        "</invoke>"
        "</tool_calls>",
    )
    _assert_streamed_state(parser, [("get_weather", {"city": "SF"})])


# ---------- GPT-OSS ----------


def test_gptoss_populates_state():
    parser = _make_parser("gptoss")
    _run(
        parser,
        "<tool_call>to=functions.get_weather json\n"
        '{"city": "SF"}\n'
        "</tool_call>",
    )
    _assert_streamed_state(parser, [("get_weather", {"city": "SF"})])


# ---------- Python ----------


def test_python_populates_state():
    parser = _make_parser("python")
    _run(
        parser,
        '<tool_call>\nget_weather(city="SF")\n</tool_call>',
    )
    _assert_streamed_state(parser, [("get_weather", {"city": "SF"})])


# ---------- Multi-call ----------


def test_multi_call_indices_align():
    parser = _make_parser("xml")
    _run(
        parser,
        "<ifm|tool_calls>"
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>SF</ifm|arg_value>"
        "</ifm|tool_call>"
        "<ifm|tool_call>get_time"
        "<ifm|arg_key>tz</ifm|arg_key>"
        "<ifm|arg_value>UTC</ifm|arg_value>"
        "</ifm|tool_call>"
        "</ifm|tool_calls>",
    )
    _assert_streamed_state(
        parser,
        [
            ("get_weather", {"city": "SF"}),
            ("get_time", {"tz": "UTC"}),
        ],
    )
