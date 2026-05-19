# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming tool-call parser for the IFM (LLM360) formats.

Handles three closely-related formats:
  * ``xml``       -- ``<ifm|tool_call>name<ifm|arg_key>K</ifm|arg_key>
                       <ifm|arg_value>V</ifm|arg_value></ifm|tool_call>``
                    with an optional ``<ifm|tool_calls>...</ifm|tool_calls>``
                    wrapper, and schema-driven type coercion.
  * ``xml_typed`` -- same as ``xml`` plus an
                    ``<ifm|arg_type>T</ifm|arg_type>`` between key and value
                    (used for type coercion when no JSON Schema is on the
                    request).
  * ``json``      -- ``<ifm|tool_call>{"name":..., "arguments":{...}}
                    </ifm|tool_call>`` (one or a list of JSON tool-calls per
                    block).

See ``MultiFormatToolParser._extract_ifm_xml_tool_calls`` and
``_extract_ifm_json_tool_calls`` for the non-streaming reference behavior.

``prev_tool_call_arr`` and ``streamed_args_for_tool`` are inherited from
``BaseToolCallStreamer`` and aliased to the owning parser instance, so
``serving_chat.py``'s end-of-stream flush logic sees the live streaming
state. Because every argument is emitted as complete coerced JSON as soon
as ``</ifm|arg_value>`` is observed, the flush is always a no-op.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from vllm.entrypoints.chat_utils import make_tool_call_id
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.base import (
    BaseToolCallStreamer,
)


@dataclass
class _ContentEmission:
    text: str


@dataclass
class _ToolNameEmission:
    index: int
    id: str
    name: str
    initial_arguments: str


@dataclass
class _ToolArgsEmission:
    index: int
    args: str


class _Sentinel:
    def __init__(self, name: str) -> None:
        self._name = name

    def __repr__(self) -> str:
        return self._name


_STALL = _Sentinel("STALL")
_PROGRESS = _Sentinel("PROGRESS")


class IFMToolCallStreamer(BaseToolCallStreamer):
    """Streamer for IFM ``xml`` / ``xml_typed`` / ``json`` formats."""

    _TOOL_CALLS_START = "<ifm|tool_calls>"
    _TOOL_CALLS_END = "</ifm|tool_calls>"
    _TOOL_CALL_START = "<ifm|tool_call>"
    _TOOL_CALL_END = "</ifm|tool_call>"
    _ARG_KEY_START = "<ifm|arg_key>"
    _ARG_KEY_END = "</ifm|arg_key>"
    _ARG_TYPE_START = "<ifm|arg_type>"
    _ARG_TYPE_END = "</ifm|arg_type>"
    _ARG_VALUE_START = "<ifm|arg_value>"
    _ARG_VALUE_END = "</ifm|arg_value>"

    _BEFORE_MARKERS = (_TOOL_CALLS_START, _TOOL_CALL_START)

    _BEFORE = "before"
    _BETWEEN = "between"
    _COLLECTING_NAME = "name"
    _COLLECTING_ARG_KEY = "arg_key"
    _AFTER_ARG_KEY = "after_arg_key"
    _COLLECTING_ARG_TYPE = "arg_type"
    _AFTER_ARG_TYPE = "after_arg_type"
    _COLLECTING_ARG_VALUE = "arg_value"
    _AFTER_ARG_VALUE = "after_arg_value"
    _SKIPPING_UNTIL_CLOSE = "skipping"
    _JSON_BODY = "json_body"
    _AFTER = "after"

    _MAX_BUFFER_BYTES = 1 << 20

    def __init__(
        self,
        tool_format: str,
        parser_cls,
    ) -> None:
        super().__init__(tool_format, parser_cls)
        self._reset_state()

    def _reset_state(self) -> None:
        self._buffer: str = ""
        self._phase: str = self._BEFORE
        self._has_outer_wrapper: bool = False
        self._current_tool_index: int = -1
        self._current_tool_name: str = ""
        self._args_count: int = 0
        self._args_so_far: dict[str, Any] = {}
        self._arg_key_buffer: str = ""
        self._arg_type_buffer: str = ""
        self._streamed_args_running: str = ""
        # Clear (not reassign) the inherited lists so the alias set in
        # MultiFormatToolParser.__init__ stays valid across resets.
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    def feed(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
        request: ChatCompletionRequest,
    ) -> DeltaMessage | None:
        if previous_text == "":
            self._reset_state()
        self._buffer += delta_text
        if len(self._buffer) > self._MAX_BUFFER_BYTES:
            self._buffer = self._buffer[-(self._MAX_BUFFER_BYTES // 2) :]
        emissions = self._collect_emissions(request)
        if not emissions:
            return None
        return self._build_delta_message(emissions)

    def _collect_emissions(self, request: ChatCompletionRequest) -> list:
        emissions: list = []
        while True:
            result = self._step(request)
            if result is _STALL:
                break
            if result is _PROGRESS:
                continue
            if isinstance(result, list):
                emissions.extend(result)
                continue
            emissions.append(result)
        return emissions

    def _step(self, request: ChatCompletionRequest):
        phase = self._phase
        if phase == self._BEFORE:
            return self._step_before()
        if phase == self._BETWEEN:
            return self._step_between()
        if phase == self._COLLECTING_NAME:
            return self._step_collecting_name()
        if phase == self._COLLECTING_ARG_KEY:
            return self._step_collecting_arg_key()
        if phase == self._AFTER_ARG_KEY:
            return self._step_after_arg_key()
        if phase == self._COLLECTING_ARG_TYPE:
            return self._step_collecting_arg_type()
        if phase == self._AFTER_ARG_TYPE:
            return self._step_after_arg_type()
        if phase == self._COLLECTING_ARG_VALUE:
            return self._step_collecting_arg_value(request)
        if phase == self._AFTER_ARG_VALUE:
            return self._step_after_arg_value()
        if phase == self._SKIPPING_UNTIL_CLOSE:
            return self._step_skipping_until_close()
        if phase == self._JSON_BODY:
            return self._step_json_body(request)
        return self._step_after()

    def _step_before(self):
        pos_calls = self._buffer.find(self._TOOL_CALLS_START)
        pos_call = self._buffer.find(self._TOOL_CALL_START)
        candidates: list[tuple[int, str, str]] = []
        if pos_calls != -1:
            candidates.append((pos_calls, self._TOOL_CALLS_START, "outer"))
        if pos_call != -1:
            candidates.append((pos_call, self._TOOL_CALL_START, "call"))

        if not candidates:
            safe, tail = self._split_at_partial_marker(
                self._buffer, self._BEFORE_MARKERS
            )
            if not safe:
                return _STALL
            self._buffer = tail
            return _ContentEmission(safe)

        pos, marker, kind = min(candidates, key=lambda x: x[0])
        if pos > 0:
            content = self._buffer[:pos]
            self._buffer = self._buffer[pos:]
            return _ContentEmission(content)

        self._buffer = self._buffer[len(marker) :]
        if kind == "outer":
            self._has_outer_wrapper = True
            self._phase = self._BETWEEN
        else:
            self._start_new_tool_call()
            if self.tool_format == "json":
                self._phase = self._JSON_BODY
            else:
                self._phase = self._COLLECTING_NAME
        return _PROGRESS

    def _step_between(self):
        pos_call = self._buffer.find(self._TOOL_CALL_START)
        pos_outer_end = (
            self._buffer.find(self._TOOL_CALLS_END) if self._has_outer_wrapper else -1
        )

        candidates: list[tuple[int, str, str]] = []
        if pos_call != -1:
            candidates.append((pos_call, self._TOOL_CALL_START, "call"))
        if pos_outer_end != -1:
            candidates.append((pos_outer_end, self._TOOL_CALLS_END, "outer_end"))

        if not candidates:
            markers: tuple[str, ...] = (self._TOOL_CALL_START,)
            if self._has_outer_wrapper:
                markers = markers + (self._TOOL_CALLS_END,)
            safe, tail = self._split_at_partial_marker(self._buffer, markers)
            if not safe:
                return _STALL
            self._buffer = tail
            return _PROGRESS

        pos, marker, kind = min(candidates, key=lambda x: x[0])
        self._buffer = self._buffer[pos + len(marker) :]
        if kind == "call":
            self._start_new_tool_call()
            if self.tool_format == "json":
                self._phase = self._JSON_BODY
            else:
                self._phase = self._COLLECTING_NAME
        else:
            self._phase = self._AFTER
        return _PROGRESS

    def _step_collecting_name(self):
        pos_arg = self._buffer.find(self._ARG_KEY_START)
        pos_close = self._buffer.find(self._TOOL_CALL_END)
        candidates: list[tuple[int, str, str]] = []
        if pos_arg != -1:
            candidates.append((pos_arg, self._ARG_KEY_START, "arg"))
        if pos_close != -1:
            candidates.append((pos_close, self._TOOL_CALL_END, "close"))

        if not candidates:
            return _STALL

        pos, marker, kind = min(candidates, key=lambda x: x[0])
        name = self._buffer[:pos].strip()
        self._buffer = self._buffer[pos + len(marker) :]

        if not name:
            # Non-streaming behavior skips tool calls with empty names.
            self._current_tool_index -= 1
            self._reset_per_tool_state()
            if kind == "close":
                self._goto_inter_tool_phase()
                return _PROGRESS
            self._phase = self._SKIPPING_UNTIL_CLOSE
            return _PROGRESS

        self._current_tool_name = name
        tool_id = make_tool_call_id()

        if kind == "arg":
            self._phase = self._COLLECTING_ARG_KEY
            self._streamed_args_running = "{"
            return _ToolNameEmission(self._current_tool_index, tool_id, name, "{")

        self._streamed_args_running = "{}"
        emission = _ToolNameEmission(self._current_tool_index, tool_id, name, "{}")
        self._record_completed_tool()
        self._reset_per_tool_state()
        self._goto_inter_tool_phase()
        return emission

    def _step_collecting_arg_key(self):
        pos = self._buffer.find(self._ARG_KEY_END)
        if pos == -1:
            return _STALL
        self._arg_key_buffer = self._buffer[:pos].strip()
        self._buffer = self._buffer[pos + len(self._ARG_KEY_END) :]
        self._phase = self._AFTER_ARG_KEY
        return _PROGRESS

    def _step_after_arg_key(self):
        pos_type = -1
        if self.tool_format == "xml_typed":
            pos_type = self._buffer.find(self._ARG_TYPE_START)
        pos_value = self._buffer.find(self._ARG_VALUE_START)

        candidates: list[tuple[int, str, str]] = []
        if pos_type != -1:
            candidates.append((pos_type, self._ARG_TYPE_START, "type"))
        if pos_value != -1:
            candidates.append((pos_value, self._ARG_VALUE_START, "value"))

        if not candidates:
            return _STALL

        pos, marker, kind = min(candidates, key=lambda x: x[0])
        self._buffer = self._buffer[pos + len(marker) :]
        if kind == "type":
            self._phase = self._COLLECTING_ARG_TYPE
        else:
            self._phase = self._COLLECTING_ARG_VALUE
        return _PROGRESS

    def _step_collecting_arg_type(self):
        pos = self._buffer.find(self._ARG_TYPE_END)
        if pos == -1:
            return _STALL
        self._arg_type_buffer = self._buffer[:pos].strip()
        self._buffer = self._buffer[pos + len(self._ARG_TYPE_END) :]
        self._phase = self._AFTER_ARG_TYPE
        return _PROGRESS

    def _step_after_arg_type(self):
        pos = self._buffer.find(self._ARG_VALUE_START)
        if pos == -1:
            return _STALL
        self._buffer = self._buffer[pos + len(self._ARG_VALUE_START) :]
        self._phase = self._COLLECTING_ARG_VALUE
        return _PROGRESS

    def _step_collecting_arg_value(self, request: ChatCompletionRequest):
        pos = self._buffer.find(self._ARG_VALUE_END)
        if pos == -1:
            return _STALL
        value_text = self._buffer[:pos].strip()
        self._buffer = self._buffer[pos + len(self._ARG_VALUE_END) :]

        arg_type = self._arg_type_buffer or None
        try:
            coerced = self._parser._coerce_argument_value(
                value_text,
                self._current_tool_name,
                self._arg_key_buffer,
                request.tools,
                arg_type=arg_type,
                from_text=True,
            )
        except Exception:
            coerced = value_text

        key_json = json.dumps(self._arg_key_buffer, ensure_ascii=False)
        val_json = json.dumps(coerced, ensure_ascii=False)
        if self._args_count == 0:
            fragment = f"{key_json}: {val_json}"
        else:
            fragment = f", {key_json}: {val_json}"

        self._args_so_far[self._arg_key_buffer] = coerced
        self._streamed_args_running += fragment
        self._args_count += 1
        self._arg_key_buffer = ""
        self._arg_type_buffer = ""
        self._phase = self._AFTER_ARG_VALUE
        return _ToolArgsEmission(self._current_tool_index, fragment)

    def _step_after_arg_value(self):
        pos_arg = self._buffer.find(self._ARG_KEY_START)
        pos_close = self._buffer.find(self._TOOL_CALL_END)
        candidates: list[tuple[int, str, str]] = []
        if pos_arg != -1:
            candidates.append((pos_arg, self._ARG_KEY_START, "arg"))
        if pos_close != -1:
            candidates.append((pos_close, self._TOOL_CALL_END, "close"))
        if not candidates:
            return _STALL

        pos, marker, kind = min(candidates, key=lambda x: x[0])
        self._buffer = self._buffer[pos + len(marker) :]

        if kind == "arg":
            self._phase = self._COLLECTING_ARG_KEY
            return _PROGRESS

        self._streamed_args_running += "}"
        emission = _ToolArgsEmission(self._current_tool_index, "}")
        self._record_completed_tool()
        self._reset_per_tool_state()
        self._goto_inter_tool_phase()
        return emission

    def _step_skipping_until_close(self):
        pos = self._buffer.find(self._TOOL_CALL_END)
        if pos == -1:
            safe, tail = self._split_at_partial_marker(
                self._buffer, (self._TOOL_CALL_END,)
            )
            if not safe:
                return _STALL
            self._buffer = tail
            return _PROGRESS
        self._buffer = self._buffer[pos + len(self._TOOL_CALL_END) :]
        self._goto_inter_tool_phase()
        return _PROGRESS

    def _step_json_body(self, request: ChatCompletionRequest):
        pos = self._buffer.find(self._TOOL_CALL_END)
        if pos == -1:
            return _STALL
        body = self._buffer[:pos].strip()
        self._buffer = self._buffer[pos + len(self._TOOL_CALL_END) :]

        try:
            parsed = json.loads(body)
        except Exception:
            self._current_tool_index -= 1
            self._reset_per_tool_state()
            self._goto_inter_tool_phase()
            return _PROGRESS

        raw_calls = parsed if isinstance(parsed, list) else [parsed]
        emissions: list = []
        emitted = 0
        for raw_call in raw_calls:
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function", raw_call)
            if not isinstance(function, dict):
                continue
            name = function.get("name")
            if not name or not isinstance(name, str):
                continue
            try:
                arguments = self._parser._json_arguments_to_dict(
                    function.get("arguments", {}),
                )
            except Exception:
                continue
            try:
                coerced = self._parser._coerce_arguments(name, arguments, request.tools)
            except Exception:
                coerced = arguments

            if emitted > 0:
                self._start_new_tool_call()
            emitted += 1

            args_json = json.dumps(coerced, ensure_ascii=False)
            tool_id = make_tool_call_id()

            self._current_tool_name = name
            self._args_so_far = dict(coerced)
            self._streamed_args_running = args_json
            self._record_completed_tool()

            emissions.append(
                _ToolNameEmission(self._current_tool_index, tool_id, name, args_json)
            )

        if emitted == 0:
            self._current_tool_index -= 1

        self._reset_per_tool_state()
        self._goto_inter_tool_phase()
        if not emissions:
            return _PROGRESS
        return emissions

    def _step_after(self):
        self._buffer = ""
        return _STALL

    def _split_at_partial_marker(
        self,
        s: str,
        markers: tuple[str, ...],
    ) -> tuple[str, str]:
        """Split ``s`` into ``(safe, tail)`` where ``tail`` is the longest
        suffix of ``s`` that is a strict prefix of one of ``markers``."""
        if not s:
            return "", ""
        max_k = 0
        for marker in markers:
            limit = min(len(marker) - 1, len(s))
            for k in range(limit, max_k, -1):
                if s.endswith(marker[:k]):
                    max_k = k
                    break
        if max_k == 0:
            return s, ""
        return s[:-max_k], s[-max_k:]

    def _start_new_tool_call(self) -> None:
        self._current_tool_index += 1
        self._current_tool_name = ""
        self._args_count = 0
        self._args_so_far = {}
        self._arg_key_buffer = ""
        self._arg_type_buffer = ""
        self._streamed_args_running = ""

    def _reset_per_tool_state(self) -> None:
        self._current_tool_name = ""
        self._args_count = 0
        self._args_so_far = {}
        self._arg_key_buffer = ""
        self._arg_type_buffer = ""
        self._streamed_args_running = ""

    def _record_completed_tool(self) -> None:
        self.prev_tool_call_arr.append(
            {
                "name": self._current_tool_name,
                "arguments": dict(self._args_so_far),
            }
        )
        self.streamed_args_for_tool.append(self._streamed_args_running)

    def _goto_inter_tool_phase(self) -> None:
        # Whether or not the outer wrapper was seen, look for the next tool
        # call (or wrapper end). Never re-enter content emission once any
        # tool marker has been observed -- this matches non-streaming, which
        # treats only the prefix before the first tool as content.
        self._phase = self._BETWEEN

    def _build_delta_message(self, emissions: list) -> DeltaMessage:
        content_parts: list[str] = []
        tool_calls: dict[int, DeltaToolCall] = {}
        for em in emissions:
            if isinstance(em, _ContentEmission):
                content_parts.append(em.text)
            elif isinstance(em, _ToolNameEmission):
                tool_calls[em.index] = DeltaToolCall(
                    index=em.index,
                    id=em.id,
                    type="function",
                    function=DeltaFunctionCall(
                        name=em.name,
                        arguments=em.initial_arguments,
                    ),
                )
            elif isinstance(em, _ToolArgsEmission):
                existing = tool_calls.get(em.index)
                if existing is None:
                    tool_calls[em.index] = DeltaToolCall(
                        index=em.index,
                        function=DeltaFunctionCall(arguments=em.args),
                    )
                else:
                    assert existing.function is not None
                    existing.function.arguments = (
                        existing.function.arguments or ""
                    ) + em.args

        content = "".join(content_parts) if content_parts else None
        tools_list = [tool_calls[k] for k in sorted(tool_calls)] if tool_calls else None

        if content is not None and tools_list:
            return DeltaMessage(content=content, tool_calls=tools_list)
        if content is not None:
            return DeltaMessage(content=content)
        return DeltaMessage(tool_calls=tools_list or [])
