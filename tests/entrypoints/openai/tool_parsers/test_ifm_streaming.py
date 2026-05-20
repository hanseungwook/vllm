# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from typing import Any

import pytest

from tests.entrypoints.openai.tool_parsers.utils import (
    StreamingToolReconstructor,
    run_tool_extraction_streaming,
)
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaMessage,
)
from vllm.entrypoints.openai.tool_parsers import ToolParser, ToolParserManager

pytestmark = pytest.mark.cpu_test


class CharTokenizer:
    """Per-character tokenizer used for streaming tests.

    Each character is a separate token, so feeding a string to
    ``run_tool_extraction_streaming`` produces one ``feed()`` call per
    character. This exercises split-marker handling at the worst-case
    granularity without relying on a downloaded model tokenizer.
    """

    def __init__(self) -> None:
        self._vocab: dict[str, int] = {}
        self._next_id = 1

    def get_vocab(self) -> dict[str, int]:
        return self._vocab

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids: list[int] = []
        for ch in text:
            if ch not in self._vocab:
                self._vocab[ch] = self._next_id
                self._next_id += 1
            ids.append(self._vocab[ch])
        return ids

    def decode(self, token_ids: list[int]) -> str:
        reverse = {v: k for k, v in self._vocab.items()}
        return "".join(reverse[i] for i in token_ids)

    def tokenize(self, text: str) -> list[str]:
        return list(text)


def make_parser(tool_format: str) -> ToolParser:
    return ToolParserManager.get_tool_parser("multi_format")(
        CharTokenizer(),
        chat_template_kwargs={"tool_format": tool_format},
    )


def make_request(tools: list[dict[str, Any]] | None = None) -> ChatCompletionRequest:
    kwargs: dict[str, Any] = {"model": "test-model", "messages": []}
    if tools is not None:
        kwargs["tools"] = tools
    return ChatCompletionRequest(**kwargs)


def make_schema_request() -> ChatCompletionRequest:
    return make_request(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "study_args",
                    "description": "Study argument coercion.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "user_id": {"type": "string"},
                            "include_revoked": {"type": "boolean"},
                            "page": {"type": "integer"},
                            "filters": {"type": "object"},
                        },
                    },
                },
            }
        ]
    )


def assert_round_trip(
    tool_format: str,
    model_output: str,
    request: ChatCompletionRequest,
    *,
    assert_one_tool_per_delta: bool = True,
) -> StreamingToolReconstructor:
    """Run streaming and non-streaming on the same input and assert they
    produce identical tool calls (name + arguments JSON) and content."""
    streaming_parser = make_parser(tool_format)
    nonstreaming_parser = make_parser(tool_format)

    reconstructor = run_tool_extraction_streaming(
        streaming_parser,
        model_output,
        request=request,
        assert_one_tool_per_delta=assert_one_tool_per_delta,
    )
    expected = nonstreaming_parser.extract_tool_calls(model_output, request)

    assert len(reconstructor.tool_calls) == len(expected.tool_calls)
    for actual, want in zip(reconstructor.tool_calls, expected.tool_calls):
        assert actual.function.name == want.function.name
        assert json.loads(actual.function.arguments) == json.loads(
            want.function.arguments
        )
        # Exact string match: streaming must reconstruct the same JSON as
        # the non-streaming parser, not just an equivalent dict.
        assert actual.function.arguments == want.function.arguments

    streamed_content = reconstructor.other_content or None
    assert streamed_content == expected.content
    return reconstructor


def test_xml_single_tool_single_string_arg():
    model_output = (
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    reconstructor = assert_round_trip("xml", model_output, make_request())
    assert len(reconstructor.tool_calls) == 1
    tc = reconstructor.tool_calls[0]
    assert tc.function.name == "get_weather"
    assert json.loads(tc.function.arguments) == {"city": "Tokyo"}
    assert tc.id, "first delta must populate a tool-call id"


def test_xml_parallel_tool_calls_bump_index():
    model_output = (
        "<ifm|tool_calls>"
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
        "<ifm|tool_call>get_time"
        "<ifm|arg_key>timezone</ifm|arg_key>"
        "<ifm|arg_value>UTC</ifm|arg_value>"
        "</ifm|tool_call>"
        "</ifm|tool_calls>"
    )
    reconstructor = assert_round_trip("xml", model_output, make_request())
    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {
        "timezone": "UTC"
    }
    ids = [tc.id for tc in reconstructor.tool_calls]
    assert all(ids) and ids[0] != ids[1], "each tool call must have a unique id"


def test_xml_schema_coerces_value_types():
    model_output = (
        "<ifm|tool_calls>"
        "<ifm|tool_call>study_args"
        "<ifm|arg_key>user_id</ifm|arg_key>"
        "<ifm|arg_value>12345</ifm|arg_value>"
        "<ifm|arg_key>include_revoked</ifm|arg_key>"
        "<ifm|arg_value>true</ifm|arg_value>"
        "<ifm|arg_key>page</ifm|arg_key>"
        "<ifm|arg_value>2</ifm|arg_value>"
        "<ifm|arg_key>filters</ifm|arg_key>"
        '<ifm|arg_value>{"unit":"celsius"}</ifm|arg_value>'
        "</ifm|tool_call>"
        "</ifm|tool_calls>"
    )
    reconstructor = assert_round_trip("xml", model_output, make_schema_request())
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert args == {
        "user_id": "12345",
        "include_revoked": True,
        "page": 2,
        "filters": {"unit": "celsius"},
    }
    assert isinstance(args["user_id"], str)


def test_xml_typed_uses_arg_type_without_schema():
    model_output = (
        "<ifm|tool_calls>"
        "<ifm|tool_call>study_args"
        "<ifm|arg_key>user_id</ifm|arg_key>"
        "<ifm|arg_type>string</ifm|arg_type>"
        "<ifm|arg_value>12345</ifm|arg_value>"
        "<ifm|arg_key>include_revoked</ifm|arg_key>"
        "<ifm|arg_type>boolean</ifm|arg_type>"
        "<ifm|arg_value>true</ifm|arg_value>"
        "<ifm|arg_key>page</ifm|arg_key>"
        "<ifm|arg_type>integer</ifm|arg_type>"
        "<ifm|arg_value>2</ifm|arg_value>"
        "</ifm|tool_call>"
        "</ifm|tool_calls>"
    )
    reconstructor = assert_round_trip("xml_typed", model_output, make_request())
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert args == {
        "user_id": "12345",
        "include_revoked": True,
        "page": 2,
    }
    assert isinstance(args["user_id"], str)


def test_json_format_emits_full_coerced_arguments():
    model_output = (
        "<ifm|tool_calls>"
        '<ifm|tool_call>{"name": "study_args", "arguments": {'
        '"user_id": 12345, '
        '"include_revoked": "true", '
        '"page": "2", '
        '"filters": "{\\"unit\\":\\"celsius\\"}"'
        "}}</ifm|tool_call>"
        "</ifm|tool_calls>"
    )
    reconstructor = assert_round_trip("json", model_output, make_schema_request())
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert args == {
        "user_id": "12345",
        "include_revoked": True,
        "page": 2,
        "filters": {"unit": "celsius"},
    }


def test_json_format_handles_list_of_tool_calls_in_one_block():
    model_output = (
        '<ifm|tool_call>[{"name": "get_weather", '
        '"arguments": {"city": "Tokyo"}}, '
        '{"name": "get_time", '
        '"arguments": {"tz": "UTC"}}]</ifm|tool_call>'
    )
    # A single <ifm|tool_call> block holds a list of two calls. The
    # non-streaming parser walks the list and emits two ToolCalls; the
    # streamer mirrors that by bumping the index for the second call.
    reconstructor = assert_round_trip(
        "json",
        model_output,
        make_request(),
        assert_one_tool_per_delta=False,
    )
    assert [tc.function.name for tc in reconstructor.tool_calls] == [
        "get_weather",
        "get_time",
    ]


def test_content_prefix_before_tool_is_emitted_as_content():
    model_output = (
        "Planning before the call.\n"
        "<ifm|tool_calls>"
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
        "</ifm|tool_calls>"
    )
    reconstructor = assert_round_trip("xml", model_output, make_request())
    assert reconstructor.other_content == "Planning before the call.\n"
    assert reconstructor.tool_calls[0].function.name == "get_weather"


def test_content_prefix_before_bare_tool_call_marker():
    # No outer <ifm|tool_calls> wrapper; the prefix still streams as content.
    model_output = (
        "Direct call: "
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    reconstructor = assert_round_trip("xml", model_output, make_request())
    assert reconstructor.other_content == "Direct call: "
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }


def test_tool_with_no_arguments_emits_empty_object():
    model_output = "<ifm|tool_call>ping</ifm|tool_call>"
    reconstructor = assert_round_trip("xml", model_output, make_request())
    assert reconstructor.tool_calls[0].function.name == "ping"
    assert reconstructor.tool_calls[0].function.arguments == "{}"


def test_split_markers_per_character_feed():
    """Drive the streamer with single-character deltas to prove
    multi-character markers like ``<ifm|tool_call>`` are buffered correctly
    when they straddle delta boundaries."""
    parser = make_parser("xml")
    model_output = (
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    deltas = list(model_output)  # one character per feed
    reconstructor = run_tool_extraction_streaming(
        parser, deltas, request=make_request()
    )
    assert len(reconstructor.tool_calls) == 1
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }


def test_split_markers_chunked_at_marker_boundaries():
    """Feed the input chunked so that markers straddle deltas (the worst
    case for naive substring matching)."""
    parser = make_parser("xml")
    full = (
        "Before.<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    # Split inside several different markers to maximize the chance of
    # accidentally consuming a partial marker as content.
    deltas = [
        "Befo",
        "re.<ifm|tool_",
        "call>get_wea",
        "ther<ifm|arg",
        "_key>ci",
        "ty</ifm|arg_key>",
        "<ifm|arg_value>To",
        "kyo</ifm|arg",
        "_value></ifm|",
        "tool_call>",
    ]
    assert "".join(deltas) == full
    reconstructor = run_tool_extraction_streaming(
        parser, deltas, request=make_request()
    )
    assert reconstructor.other_content == "Before."
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }


def test_missing_tool_format_defaults_to_xml_for_streaming():
    parser = ToolParserManager.get_tool_parser("multi_format")(
        CharTokenizer(),
        chat_template_kwargs={},
    )
    model_output = (
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    reconstructor = run_tool_extraction_streaming(
        parser, model_output, request=make_request()
    )
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Tokyo"
    }


def test_first_delta_carries_id_and_name_subsequent_do_not():
    """Drive the streamer manually and inspect the raw DeltaMessages to
    verify the OpenAI streaming contract: only the first delta for a tool
    call carries ``id``/``name``; later deltas extend ``arguments`` only."""
    parser = make_parser("xml")
    request = make_request()
    deltas = list(
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    seen: list[DeltaMessage] = []
    previous_text = ""
    for delta in deltas:
        current_text = previous_text + delta
        msg = parser.extract_tool_calls_streaming(
            previous_text,
            current_text,
            delta,
            [],
            [],
            [],
            request,
        )
        previous_text = current_text
        if msg is not None and msg.tool_calls:
            seen.append(msg)

    assert seen, "expected at least one tool-call delta"
    first, *rest = seen
    assert first.tool_calls[0].id, "first tool delta must carry id"
    assert first.tool_calls[0].type == "function"
    assert first.tool_calls[0].function.name == "get_weather"
    assert first.tool_calls[0].index == 0
    for later in rest:
        assert later.tool_calls[0].id is None
        assert later.tool_calls[0].type is None
        assert later.tool_calls[0].function.name in (None, "")
        assert later.tool_calls[0].function.arguments is not None
        assert later.tool_calls[0].index == 0


def test_request_state_resets_between_streams():
    """vLLM reuses parser instances across requests. The streamer must
    reset all state when it sees ``previous_text == ""``."""
    parser = make_parser("xml")
    request = make_request()
    sample = (
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>Tokyo</ifm|arg_value>"
        "</ifm|tool_call>"
    )

    first = run_tool_extraction_streaming(parser, sample, request=request)
    assert len(first.tool_calls) == 1
    assert first.tool_calls[0].function.name == "get_weather"

    # Reuse the same parser for a new request.
    second = run_tool_extraction_streaming(parser, sample, request=request)
    assert len(second.tool_calls) == 1
    # Index resets to 0, not 1, because the streamer reset its state.
    assert second.tool_calls[0].function.name == "get_weather"


def test_xml_eos_mid_call_records_partial_state_for_flush():
    """Model stops mid-call (EOS without ``</ifm|tool_call>``) — ``serving_chat``
    must still see ``finish_reason="tool_calls"`` and flush the missing ``}``.
    """
    parser = make_parser("xml")
    request = make_request()
    # No `</ifm|tool_call>`: the stream ends mid-call.
    model_output = (
        "<ifm|tool_call>get_weather"
        "<ifm|arg_key>city</ifm|arg_key>"
        "<ifm|arg_value>SF</ifm|arg_value>"
    )
    run_tool_extraction_streaming(parser, model_output, request=request)
    streamer = parser._streamer
    assert streamer is not None
    mirror = streamer.prev_tool_call_arr
    assert len(mirror) == 1, (
        f"expected one in-progress tool call recorded, got {mirror!r}"
    )
    assert mirror[0]["name"] == "get_weather"
    assert mirror[0]["arguments"] == {"city": "SF"}
    streamed = streamer.streamed_args_for_tool[0]
    # No closing `}` — ``serving_chat``'s flush logic emits the diff:
    #   ``json.dumps({"city": "SF"}).replace(streamed, "", 1) == "}"``.
    assert streamed == '{"city": "SF"'


def test_streaming_mirror_records_tools_for_end_of_stream_flush():
    """``serving_chat.py`` reads ``prev_tool_call_arr`` /
    ``streamed_args_for_tool`` from the parser at end-of-stream. The
    streamer mirrors these so end-of-stream consumers see one entry per
    completed tool with concatenated arguments matching what was streamed.
    """
    parser = make_parser("xml")
    request = make_schema_request()
    model_output = (
        "<ifm|tool_call>study_args"
        "<ifm|arg_key>user_id</ifm|arg_key>"
        "<ifm|arg_value>12345</ifm|arg_value>"
        "<ifm|arg_key>page</ifm|arg_key>"
        "<ifm|arg_value>2</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    run_tool_extraction_streaming(parser, model_output, request=request)
    streamer = parser._streamer
    assert streamer is not None
    mirror = streamer.prev_tool_call_arr
    assert len(mirror) == 1
    assert mirror[0]["name"] == "study_args"
    assert mirror[0]["arguments"] == {"user_id": "12345", "page": 2}
    streamed = streamer.streamed_args_for_tool[0]
    # Reconstructed args from the streamed fragments must be valid JSON
    # that round-trips to the recorded arguments dict.
    assert json.loads(streamed) == {"user_id": "12345", "page": 2}


def test_json_format_streams_args_incrementally():
    """Char-by-char streaming of a JSON body should produce more than one
    argument fragment — args are emitted as they're parsed, not buffered
    until the close marker.
    """
    parser = make_parser("json")
    request = make_request()
    body = (
        '<ifm|tool_call>{"name":"foo","arguments":'
        '{"city":"SF","unit":"celsius"}}</ifm|tool_call>'
    )

    arg_fragments: list[str] = []
    name_seen: str | None = None
    previous_text = ""
    for ch in body:
        current_text = previous_text + ch
        msg = parser.extract_tool_calls_streaming(
            previous_text, current_text, ch, [], [], [], request
        )
        previous_text = current_text
        if msg is None or not msg.tool_calls:
            continue
        call = msg.tool_calls[0]
        if call.function and call.function.name:
            name_seen = call.function.name
        if call.function and call.function.arguments:
            arg_fragments.append(call.function.arguments)

    assert name_seen == "foo"
    assert "".join(arg_fragments) == '{"city": "SF", "unit": "celsius"}'
    # The whole point of using ``partial_json_parser``: at least two
    # argument deltas, not one big atomic emit at the close marker.
    assert len(arg_fragments) > 1, (
        f"expected incremental args emit, got one big fragment: {arg_fragments}"
    )


def test_json_format_eos_mid_args_records_partial_state_for_flush():
    """Model stops mid-call (EOS without ``</ifm|tool_call>``) — the streamer
    must still expose ``prev_tool_call_arr`` and ``streamed_args_for_tool``
    so ``serving_chat``'s flush emits the missing tail with the right index.
    """
    parser = make_parser("json")
    request = make_request()
    # No </ifm|tool_call>: stream ends mid-args.
    model_output = '<ifm|tool_call>{"name":"foo","arguments":{"city":"SF"}'

    run_tool_extraction_streaming(parser, model_output, request=request)
    streamer = parser._streamer
    assert streamer is not None
    mirror = streamer.prev_tool_call_arr
    assert len(mirror) == 1
    assert mirror[0]["name"] == "foo"
    assert mirror[0]["arguments"] == {"city": "SF"}
    streamed = streamer.streamed_args_for_tool[0]
    canonical = json.dumps(mirror[0]["arguments"], ensure_ascii=False)
    # Critical streaming invariant: at any partial point the streamed bytes
    # form a prefix of the canonical args JSON, so the serving_chat flush is
    # ``canonical[len(streamed):]`` — at worst a tail, never a duplication.
    assert canonical.startswith(streamed)


def test_json_format_list_body_emits_calls_with_distinct_indices():
    """The list-body case used to dump all calls into one DeltaMessage, which
    broke ``serving_chat``'s flush logic (it indexes by tool_calls[0] but
    reads prev_tool_call_arr[-1]). With incremental emit each list element
    gets its own ``index`` in distinct deltas.
    """
    parser = make_parser("json")
    request = make_request()
    body = (
        "<ifm|tool_call>["
        '{"name":"a","arguments":{"x":1}},'
        '{"name":"b","arguments":{"y":2}}'
        "]</ifm|tool_call>"
    )

    indices_seen_per_delta: list[set[int]] = []
    previous_text = ""
    for ch in body:
        current_text = previous_text + ch
        msg = parser.extract_tool_calls_streaming(
            previous_text, current_text, ch, [], [], [], request
        )
        previous_text = current_text
        if msg is None or not msg.tool_calls:
            continue
        indices_seen_per_delta.append({tc.index for tc in msg.tool_calls})

    # Every delta touches exactly one tool index — never both at once.
    for indices in indices_seen_per_delta:
        assert len(indices) == 1, (
            f"a single delta carried multiple tool indices: {indices}"
        )
    # Both tools' indices appeared, separately.
    all_indices = {idx for s in indices_seen_per_delta for idx in s}
    assert all_indices == {0, 1}

    streamer = parser._streamer
    assert streamer is not None
    assert len(streamer.prev_tool_call_arr) == 2
    assert streamer.prev_tool_call_arr[0]["name"] == "a"
    assert streamer.prev_tool_call_arr[1]["name"] == "b"


def test_json_format_function_wrapped_call_still_extracts_name():
    """The non-streaming parser accepts ``{"function":{"name":...}}`` as well
    as the flat form. The streamer should too.
    """
    model_output = (
        '<ifm|tool_call>{"function":{"name":"foo","arguments":'
        '{"city":"SF"}}}</ifm|tool_call>'
    )
    reconstructor = assert_round_trip("json", model_output, make_request())
    assert reconstructor.tool_calls[0].function.name == "foo"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {"city": "SF"}
