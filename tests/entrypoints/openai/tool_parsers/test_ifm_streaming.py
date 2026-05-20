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


# ----------------------------------------------------------------------
# Char-by-char streaming for string-typed args (the GLM-5 port feature).
# These tests prove that long string values arrive at the client as a
# sequence of args fragments rather than buffered until ``</arg_value>``.
# ----------------------------------------------------------------------


def _write_file_request(content_type: str = "string") -> ChatCompletionRequest:
    return make_request(
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "write_file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                            "content": {"type": content_type},
                        },
                    },
                },
            }
        ]
    )


def _drive_char_by_char(
    parser: ToolParser,
    text: str,
    request: ChatCompletionRequest,
) -> list[DeltaMessage]:
    """Feed ``text`` one character at a time and return all DeltaMessages
    that carry a tool-call args fragment (excludes name-only deltas and
    content-only deltas)."""
    seen: list[DeltaMessage] = []
    previous_text = ""
    for ch in text:
        current_text = previous_text + ch
        msg = parser.extract_tool_calls_streaming(
            previous_text, current_text, ch, [], [], [], request
        )
        previous_text = current_text
        if msg is None or not msg.tool_calls:
            continue
        call = msg.tool_calls[0]
        if call.function is None or not call.function.arguments:
            continue
        seen.append(msg)
    return seen


def test_xml_string_arg_streams_char_by_char_with_schema():
    """Schema declares ``content`` as string, so a multi-char value
    arrives at the client as one args fragment per (escaped) character,
    not buffered until ``</ifm|arg_value>``."""
    content = "line1\nline2\nline3"
    text = (
        "<ifm|tool_call>write_file"
        "<ifm|arg_key>content</ifm|arg_key>"
        f"<ifm|arg_value>{content}</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    parser = make_parser("xml")
    request = _write_file_request()
    args_deltas = _drive_char_by_char(parser, text, request)

    # The opener ``{"content": "`` arrives once, then each input char
    # produces its own escaped fragment, then the closer ``"`` and ``}``.
    # For a 17-char content this is ~20 fragments; we assert ``>= 5`` to
    # tolerate batching where ``</`` partial-marker handling holds back a
    # few chars until the next feed.
    assert len(args_deltas) >= 5, (
        "expected char-by-char emission for a string arg; got "
        f"{len(args_deltas)} fragments: "
        f"{[m.tool_calls[0].function.arguments for m in args_deltas]}"
    )

    joined = "".join(m.tool_calls[0].function.arguments for m in args_deltas)
    assert json.loads(joined) == {"content": content}
    assert joined == json.dumps({"content": content}, ensure_ascii=False)


def test_xml_string_arg_streams_4000_char_content_in_many_chunks():
    """Latency proof: a 4000-char string arg under schema=string emits
    many args fragments (not one), so the client renders content as the
    model produces it. Before this port, the entire 4000 chars buffered
    until ``</ifm|arg_value>`` was seen.
    """
    content = "x" * 4000
    text = (
        "<ifm|tool_call>write_file"
        "<ifm|arg_key>content</ifm|arg_key>"
        f"<ifm|arg_value>{content}</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    parser = make_parser("xml")
    request = _write_file_request()
    args_deltas = _drive_char_by_char(parser, text, request)

    # We expect roughly one fragment per char (the opener + ~4000 chars
    # + the close-quote + the brace-close). The exact count depends on
    # partial-marker hold-back behavior; ``>= 1000`` is a safe lower
    # bound that fails loudly if we accidentally re-introduce buffering.
    assert len(args_deltas) >= 1000, (
        f"expected many fragments for a 4000-char string arg; got {len(args_deltas)}"
    )

    joined = "".join(m.tool_calls[0].function.arguments for m in args_deltas)
    assert json.loads(joined) == {"content": content}


def test_xml_string_arg_escapes_quote_and_backslash_under_schema():
    """Embedded ``"`` and ``\\`` in a string value are JSON-escaped in
    each char's fragment. Concatenated, the streamed fragments form
    valid JSON whose parsed value equals the original raw string."""
    content = 'say "hi" and \\ test'
    text = (
        "<ifm|tool_call>write_file"
        "<ifm|arg_key>content</ifm|arg_key>"
        f"<ifm|arg_value>{content}</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    parser = make_parser("xml")
    request = _write_file_request()
    args_deltas = _drive_char_by_char(parser, text, request)

    joined = "".join(m.tool_calls[0].function.arguments for m in args_deltas)
    parsed = json.loads(joined)
    assert parsed == {"content": content}
    # Escapes must appear in the streamed text -- verify the quote and
    # backslash were emitted in their JSON-escaped form (\\" and \\\\)
    # somewhere in the concatenation.
    assert '\\"' in joined
    assert "\\\\" in joined


def test_xml_non_string_arg_buffered_until_close_marker():
    """Schema declares ``count`` as integer. The value buffers until
    ``</ifm|arg_value>`` then emits in one fragment with ``json.dumps``-
    of the coerced int. No char-by-char streaming for non-string args."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "incr",
                "parameters": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                },
            },
        }
    ]
    text = (
        "<ifm|tool_call>incr"
        "<ifm|arg_key>count</ifm|arg_key>"
        "<ifm|arg_value>42</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    parser = make_parser("xml")
    request = make_request(tools=tools)
    args_deltas = _drive_char_by_char(parser, text, request)

    fragments = [m.tool_calls[0].function.arguments for m in args_deltas]
    # Exactly two fragments: the value's ``{"count": 42`` (buffered) and
    # the close ``}``. No per-char emissions during ``42`` collection.
    assert fragments == ['{"count": 42', "}"], fragments
    joined = "".join(fragments)
    assert joined == '{"count": 42}'


def test_xml_mixed_schema_string_streams_int_buffers():
    """One string arg + one int arg in the same tool call. String
    streams char-by-char; int buffers and emits one fragment."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "write_chunk",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {"type": "string"},
                        "size": {"type": "integer"},
                    },
                },
            },
        }
    ]
    text = (
        "<ifm|tool_call>write_chunk"
        "<ifm|arg_key>data</ifm|arg_key>"
        "<ifm|arg_value>abc</ifm|arg_value>"
        "<ifm|arg_key>size</ifm|arg_key>"
        "<ifm|arg_value>3</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    parser = make_parser("xml")
    request = make_request(tools=tools)
    args_deltas = _drive_char_by_char(parser, text, request)

    fragments = [m.tool_calls[0].function.arguments for m in args_deltas]
    joined = "".join(fragments)
    # Final reconstruction equals the non-streaming JSON serialization.
    assert joined == json.dumps({"data": "abc", "size": 3}, ensure_ascii=False)
    # Each input string char produces its own fragment (in addition to
    # the opener and closer); the int produces exactly one buffered
    # fragment + the tool-close fragment.
    string_char_fragments = [f for f in fragments if f in {"a", "b", "c"}]
    assert len(string_char_fragments) == 3, (
        f"expected one fragment per input string char; got fragments {fragments!r}"
    )
    # The int arg's fragment includes its full coerced JSON value.
    assert ', "size": 3' in fragments


def test_xml_eos_mid_string_arg_records_partial_state():
    """The stream stops mid-way through a string arg's char-by-char
    streaming. ``prev_tool_call_arr`` should reflect the partial string
    seen so far, and ``streamed_args_for_tool`` should be a valid prefix
    of ``json.dumps`` of that partial dict."""
    # Truncate after "lin" -- the second arg_value never closes.
    text = (
        "<ifm|tool_call>write_file"
        "<ifm|arg_key>content</ifm|arg_key>"
        "<ifm|arg_value>line1\nlin"
    )
    parser = make_parser("xml")
    request = _write_file_request()
    run_tool_extraction_streaming(parser, list(text), request=request)
    streamer = parser._streamer
    assert streamer is not None

    mirror = streamer.prev_tool_call_arr
    assert len(mirror) == 1
    assert mirror[0]["name"] == "write_file"
    # The dict reflects the partial string up to the last streamed char.
    assert mirror[0]["arguments"] == {"content": "line1\nlin"}

    streamed = streamer.streamed_args_for_tool[0]
    # The streamed args are a valid prefix of json.dumps of the partial
    # dict: serving_chat's flush appends the missing ``"}`` tail.
    full = json.dumps({"content": "line1\nlin"}, ensure_ascii=False)
    assert full.startswith(streamed)
    assert full == streamed + '"}'


def test_xml_eos_mid_string_just_after_opener_seeds_empty_value():
    """The stream stops right after ``<ifm|arg_value>`` (before any
    char). The mirror seeds ``{"key": ""}`` and the streamed args end at
    the open quote, so serving_chat's flush can close cleanly."""
    text = "<ifm|tool_call>write_file<ifm|arg_key>content</ifm|arg_key><ifm|arg_value>"
    parser = make_parser("xml")
    request = _write_file_request()
    run_tool_extraction_streaming(parser, list(text), request=request)
    streamer = parser._streamer
    assert streamer is not None
    assert streamer.prev_tool_call_arr[0]["arguments"] == {"content": ""}
    streamed = streamer.streamed_args_for_tool[0]
    assert streamed == '{"content": "'


def test_xml_typed_string_arg_type_triggers_char_by_char_without_schema():
    """No schema, but the model emits ``<ifm|arg_type>string</ifm|arg_type>``
    -- that should still trigger char-by-char streaming so file content
    under xml_typed format gets the same latency benefit."""
    content = "hello world"
    text = (
        "<ifm|tool_call>write_file"
        "<ifm|arg_key>content</ifm|arg_key>"
        "<ifm|arg_type>string</ifm|arg_type>"
        f"<ifm|arg_value>{content}</ifm|arg_value>"
        "</ifm|tool_call>"
    )
    parser = make_parser("xml_typed")
    request = make_request()  # no schema
    args_deltas = _drive_char_by_char(parser, text, request)
    assert len(args_deltas) >= 5, (
        f"arg_type=string should opt this value into char-by-char; got "
        f"{[m.tool_calls[0].function.arguments for m in args_deltas]}"
    )
    joined = "".join(m.tool_calls[0].function.arguments for m in args_deltas)
    assert json.loads(joined) == {"content": content}


# ----------------------------------------------------------------------
# GLM-format coverage (the unified streamer also handles tool_format="glm").
# These tests carry forward the unique GLM grammar scenarios from the
# old test_glm_streaming.py: deserialize-coercion-without-schema, the
# outer-wrapperless markers, and parallel sequential calls.
# ----------------------------------------------------------------------


def test_glm_single_string_arg_round_trip():
    text = (
        "<tool_call>get_weather"
        "<arg_key>city</arg_key>"
        "<arg_value>Beijing</arg_value>"
        "</tool_call>"
    )
    reconstructor = assert_round_trip("glm", text, make_request())
    assert reconstructor.tool_calls[0].function.name == "get_weather"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {
        "city": "Beijing"
    }


def test_glm_mixed_argument_types_via_deserialize_glm_value():
    """No schema. Each arg is deserialized by ``_coerce_argument_value``:
    integer-looking values become ints, ``true`` becomes bool, JSON
    list/object literals become lists/dicts. The streamer buffers each
    value (the default for unknown-type args) and emits the coerced
    JSON, byte-exact to the non-streaming reference."""
    text = (
        "<tool_call>do_thing"
        "<arg_key>s</arg_key><arg_value>hello world</arg_value>"
        "<arg_key>i</arg_key><arg_value>42</arg_value>"
        "<arg_key>b</arg_key><arg_value>true</arg_value>"
        "<arg_key>l</arg_key><arg_value>[1, 2, 3]</arg_value>"
        '<arg_key>d</arg_key><arg_value>{"x": "y"}</arg_value>'
        "</tool_call>"
    )
    reconstructor = assert_round_trip("glm", text, make_request())
    args = json.loads(reconstructor.tool_calls[0].function.arguments)
    assert args == {
        "s": "hello world",
        "i": 42,
        "b": True,
        "l": [1, 2, 3],
        "d": {"x": "y"},
    }
    assert isinstance(args["i"], int)
    assert isinstance(args["b"], bool)
    assert isinstance(args["l"], list)
    assert isinstance(args["d"], dict)


def test_glm_multiple_sequential_tool_calls():
    text = (
        "<tool_call>foo<arg_key>x</arg_key><arg_value>1</arg_value></tool_call>"
        "<tool_call>bar<arg_key>y</arg_key>"
        "<arg_value>two words</arg_value></tool_call>"
    )
    reconstructor = assert_round_trip("glm", text, make_request())
    assert len(reconstructor.tool_calls) == 2
    assert reconstructor.tool_calls[0].function.name == "foo"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {"x": 1}
    assert reconstructor.tool_calls[1].function.name == "bar"
    assert json.loads(reconstructor.tool_calls[1].function.arguments) == {
        "y": "two words"
    }
    assert reconstructor.tool_calls[0].id != reconstructor.tool_calls[1].id


def test_glm_content_prefix_emitted_as_content():
    text = (
        "Planning the call.\n"
        "<tool_call>foo<arg_key>x</arg_key>"
        "<arg_value>1</arg_value></tool_call>"
    )
    reconstructor = assert_round_trip("glm", text, make_request())
    assert reconstructor.other_content == "Planning the call.\n"
    assert json.loads(reconstructor.tool_calls[0].function.arguments) == {"x": 1}


def test_glm_orphan_arg_key_dropped_and_tool_closed():
    """A key without a paired value is dropped (matching the non-
    streaming regex which only emits pairs). The remaining args close
    cleanly: ``{}`` if no args were collected, ``}`` if some were."""
    # Orphan key WITHOUT any prior valid args -> emit `{}`.
    text_orphan = "<tool_call>foo<arg_key>orphan</arg_key></tool_call>"
    parser = make_parser("glm")
    recon = run_tool_extraction_streaming(parser, text_orphan, make_request())
    assert len(recon.tool_calls) == 1
    assert recon.tool_calls[0].function.arguments == "{}"

    # Orphan key after a valid arg -> close with ``}``, keep valid arg.
    text_mixed = (
        "<tool_call>bar"
        "<arg_key>k</arg_key><arg_value>v</arg_value>"
        "<arg_key>orphan</arg_key>"
        "</tool_call>"
    )
    parser = make_parser("glm")
    recon = run_tool_extraction_streaming(parser, text_mixed, make_request())
    assert json.loads(recon.tool_calls[0].function.arguments) == {"k": "v"}


def test_glm_string_arg_streams_char_by_char_with_schema():
    """The schema-driven char-by-char streaming feature applies to the
    GLM format too -- this is the unified streamer's whole point."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "do_thing",
                "parameters": {
                    "type": "object",
                    "properties": {"prose": {"type": "string"}},
                },
            },
        }
    ]
    prose = "The quick brown fox."
    text = (
        "<tool_call>do_thing"
        "<arg_key>prose</arg_key>"
        f"<arg_value>{prose}</arg_value>"
        "</tool_call>"
    )
    parser = make_parser("glm")
    request = make_request(tools=tools)
    seen: list[DeltaMessage] = []
    previous_text = ""
    for ch in text:
        current_text = previous_text + ch
        msg = parser.extract_tool_calls_streaming(
            previous_text, current_text, ch, [], [], [], request
        )
        previous_text = current_text
        if msg and msg.tool_calls and msg.tool_calls[0].function.arguments:
            seen.append(msg)
    joined = "".join(m.tool_calls[0].function.arguments for m in seen)
    assert json.loads(joined) == {"prose": prose}
    assert len(seen) >= 5, (
        f"GLM schema=string should stream char-by-char; got {len(seen)} args fragments"
    )
