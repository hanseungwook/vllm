# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming tool-call parser for the IFM (LLM360) XML formats.

Handles two closely-related formats (see
``MultiFormatToolParser._extract_ifm_xml_tool_calls``):

  * ``xml``       -- ``<ifm|tool_call>name<ifm|arg_key>K</ifm|arg_key>
                       <ifm|arg_value>V</ifm|arg_value></ifm|tool_call>``
                    with an optional ``<ifm|tool_calls>...</ifm|tool_calls>``
                    wrapper and schema-driven type coercion.
  * ``xml_typed`` -- as above plus an ``<ifm|arg_type>T</ifm|arg_type>``
                    between key and value for type coercion when no JSON
                    schema is on the request.

``prev_tool_call_arr`` and ``streamed_args_for_tool`` (aliased from the
owning parser instance) are populated as soon as the tool name is parsed,
so if the model stops mid-call (EOS without ``</ifm|tool_call>``) the
``serving_chat`` end-of-stream flush still emits ``finish_reason="tool_calls"``
plus the missing ``}`` tail.
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


_STALL = object()
_PROGRESS = object()


class IFMXMLToolCallStreamer(BaseToolCallStreamer):
    """Streamer for IFM ``xml`` / ``xml_typed`` formats."""

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
    _NAME = "name"
    _ARG_KEY = "arg_key"
    _AFTER_ARG_KEY = "after_arg_key"
    _ARG_TYPE = "arg_type"
    _AFTER_ARG_TYPE = "after_arg_type"
    _ARG_VALUE = "arg_value"
    _AFTER_ARG_VALUE = "after_arg_value"
    _SKIP = "skipping"
    _AFTER = "after"

    _MAX_BUFFER_BYTES = 1 << 20

    def __init__(self, tool_format: str, parser_cls) -> None:
        super().__init__(tool_format, parser_cls)
        self._reset_state()

    def _reset_state(self) -> None:
        self._buffer: str = ""
        self._phase: str = self._BEFORE
        self._has_outer_wrapper: bool = False
        self._current_tool_index: int = -1
        self._current_tool_name: str = ""
        self._current_args: dict[str, Any] = {}
        self._arg_key_buffer: str = ""
        self._arg_type_buffer: str = ""
        # Clear (not reassign) so the alias set in
        # MultiFormatToolParser.__init__ stays valid across resets.
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    def _reset_per_tool_state(self) -> None:
        self._current_tool_name = ""
        self._current_args = {}
        self._arg_key_buffer = ""
        self._arg_type_buffer = ""

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

        emissions: list = []
        while True:
            result = self._step(request)
            if result is _STALL:
                break
            if result is _PROGRESS:
                continue
            emissions.append(result)
        if not emissions:
            return None
        return self._build_delta_message(emissions)

    def _step(self, request: ChatCompletionRequest):
        p = self._phase
        if p == self._BEFORE:
            return self._step_before()
        if p == self._BETWEEN:
            return self._step_between()
        if p == self._NAME:
            return self._step_collecting_name()
        if p == self._ARG_KEY:
            return self._step_collecting_until(
                self._ARG_KEY_END, "_arg_key_buffer", self._AFTER_ARG_KEY
            )
        if p == self._AFTER_ARG_KEY:
            return self._step_after_arg_key()
        if p == self._ARG_TYPE:
            return self._step_collecting_until(
                self._ARG_TYPE_END, "_arg_type_buffer", self._AFTER_ARG_TYPE
            )
        if p == self._AFTER_ARG_TYPE:
            return self._consume_marker(self._ARG_VALUE_START, self._ARG_VALUE)
        if p == self._ARG_VALUE:
            return self._step_collecting_arg_value(request)
        if p == self._AFTER_ARG_VALUE:
            return self._step_after_arg_value()
        if p == self._SKIP:
            return self._step_skipping_until_close()
        # _AFTER — outer wrapper closed; the non-streaming reference
        # ``_IFM_BLOCK_REGEX.finditer`` matches every ``<ifm|tool_call>``
        # block in the response, so a second ``<ifm|tool_calls>`` wrapper
        # or a bare ``<ifm|tool_call>`` after the close must still be
        # parsed. Drop any intervening text (matches non-streaming, which
        # treats only the pre-first-tool prefix as content).
        return self._step_after()

    def _step_before(self):
        pos = self._first_marker((self._TOOL_CALLS_START, self._TOOL_CALL_START))
        if pos is None:
            safe, tail = self._split_at_partial_marker(
                self._buffer, self._BEFORE_MARKERS
            )
            if not safe:
                return _STALL
            self._buffer = tail
            return _ContentEmission(safe)

        idx, marker = pos
        if idx > 0:
            content = self._buffer[:idx]
            self._buffer = self._buffer[idx:]
            return _ContentEmission(content)

        self._buffer = self._buffer[len(marker) :]
        if marker == self._TOOL_CALLS_START:
            self._has_outer_wrapper = True
            self._phase = self._BETWEEN
        else:
            self._current_tool_index += 1
            self._reset_per_tool_state()
            self._phase = self._NAME
        return _PROGRESS

    def _step_between(self):
        markers: list[str] = [self._TOOL_CALL_START]
        if self._has_outer_wrapper:
            markers.append(self._TOOL_CALLS_END)
        pos = self._first_marker(markers)
        if pos is None:
            safe, tail = self._split_at_partial_marker(self._buffer, tuple(markers))
            if not safe:
                return _STALL
            self._buffer = tail
            return _PROGRESS

        idx, marker = pos
        self._buffer = self._buffer[idx + len(marker) :]
        if marker == self._TOOL_CALL_START:
            self._current_tool_index += 1
            self._reset_per_tool_state()
            self._phase = self._NAME
        else:
            self._phase = self._AFTER
        return _PROGRESS

    def _step_collecting_name(self):
        pos = self._first_marker((self._ARG_KEY_START, self._TOOL_CALL_END))
        if pos is None:
            return _STALL
        idx, marker = pos
        name = self._buffer[:idx].strip()
        self._buffer = self._buffer[idx + len(marker) :]

        if not name:
            # Non-streaming behavior skips tool calls with empty names.
            self._current_tool_index -= 1
            self._reset_per_tool_state()
            if marker == self._TOOL_CALL_END:
                self._phase = self._BETWEEN
            else:
                self._phase = self._SKIP
            return _PROGRESS

        self._current_tool_name = name
        index = self._current_tool_index
        tool_id = make_tool_call_id()
        self.record_tool_call_start(index, name)

        if marker == self._ARG_KEY_START:
            self.record_args_fragment(index, "{")
            self.record_args_final(index, {})
            self._phase = self._ARG_KEY
            return _ToolNameEmission(index, tool_id, name, "{")

        # name + immediate close: emit empty-args tool.
        self.record_args_fragment(index, "{}")
        self.record_args_final(index, {})
        emission = _ToolNameEmission(index, tool_id, name, "{}")
        self._reset_per_tool_state()
        self._phase = self._BETWEEN
        return emission

    def _step_collecting_until(self, end_marker: str, attr: str, next_phase: str):
        pos = self._buffer.find(end_marker)
        if pos == -1:
            return _STALL
        setattr(self, attr, self._buffer[:pos].strip())
        self._buffer = self._buffer[pos + len(end_marker) :]
        self._phase = next_phase
        return _PROGRESS

    def _consume_marker(self, marker: str, next_phase: str):
        pos = self._buffer.find(marker)
        if pos == -1:
            return _STALL
        self._buffer = self._buffer[pos + len(marker) :]
        self._phase = next_phase
        return _PROGRESS

    def _step_after_arg_key(self):
        # Always recognize <ifm|arg_type>: the non-streaming
        # _IFM_ARG_REGEX captures the optional type group regardless of
        # tool_format and forwards it to _coerce_argument_value. Limiting
        # this to xml_typed would silently drop the type tag if the model
        # emits one under plain xml (e.g., a schema-string field where the
        # model annotates the type) and produce a different coerced value
        # than the non-streaming path.
        markers = [self._ARG_TYPE_START, self._ARG_VALUE_START]
        pos = self._first_marker(markers)
        if pos is None:
            return _STALL
        idx, marker = pos
        self._buffer = self._buffer[idx + len(marker) :]
        self._phase = (
            self._ARG_TYPE if marker == self._ARG_TYPE_START else self._ARG_VALUE
        )
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
        prefix = "" if not self._current_args else ", "
        fragment = f"{prefix}{key_json}: {val_json}"

        self._current_args[self._arg_key_buffer] = coerced
        index = self._current_tool_index
        self.record_args_fragment(index, fragment)
        self.record_args_final(index, dict(self._current_args))
        self._arg_key_buffer = ""
        self._arg_type_buffer = ""
        self._phase = self._AFTER_ARG_VALUE
        return _ToolArgsEmission(index, fragment)

    def _step_after_arg_value(self):
        pos = self._first_marker((self._ARG_KEY_START, self._TOOL_CALL_END))
        if pos is None:
            return _STALL
        idx, marker = pos
        self._buffer = self._buffer[idx + len(marker) :]
        if marker == self._ARG_KEY_START:
            self._phase = self._ARG_KEY
            return _PROGRESS

        index = self._current_tool_index
        self.record_args_fragment(index, "}")
        emission = _ToolArgsEmission(index, "}")
        self._reset_per_tool_state()
        self._phase = self._BETWEEN
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
        self._phase = self._BETWEEN
        return _PROGRESS

    def _step_after(self):
        # After the outer ``</ifm|tool_calls>``, look for another outer
        # wrapper or a bare ``<ifm|tool_call>`` so the non-streaming
        # finditer semantics are preserved across multiple wrappers.
        markers = (self._TOOL_CALLS_START, self._TOOL_CALL_START)
        pos = self._first_marker(markers)
        if pos is None:
            safe, tail = self._split_at_partial_marker(self._buffer, markers)
            if not safe:
                return _STALL
            # Drop intervening content (matches non-streaming, which only
            # treats the pre-first-tool prefix as content).
            self._buffer = tail
            return _PROGRESS

        idx, marker = pos
        self._buffer = self._buffer[idx + len(marker) :]
        if marker == self._TOOL_CALLS_START:
            self._has_outer_wrapper = True
            self._phase = self._BETWEEN
        else:
            self._current_tool_index += 1
            self._reset_per_tool_state()
            self._phase = self._NAME
        return _PROGRESS

    def _first_marker(self, markers: Sequence[str]) -> tuple[int, str] | None:
        best_idx = -1
        best_marker: str | None = None
        for marker in markers:
            idx = self._buffer.find(marker)
            if idx == -1:
                continue
            if best_idx == -1 or idx < best_idx:
                best_idx, best_marker = idx, marker
        if best_marker is None:
            return None
        return best_idx, best_marker

    @staticmethod
    def _split_at_partial_marker(s: str, markers: tuple[str, ...]) -> tuple[str, str]:
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
            else:  # _ToolArgsEmission
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
