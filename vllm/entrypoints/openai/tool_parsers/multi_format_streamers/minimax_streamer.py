# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming tool-call parser for the Minimax / DSV3.2 formats.

Both formats share the outer grammar::

    <tool_calls>
      <invoke name="NAME">
        <parameter name="K"[ string="true|false"]>V</parameter>
        ...
      </invoke>
    </tool_calls>

The ``string=`` attribute is honored only in ``dsv32`` (see
``MultiFormatToolParser._extract_dsv32_tool_calls``). The ``minimax`` path
always runs ``_json_or_string`` on the value.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import regex as re

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

if TYPE_CHECKING:
    from vllm.entrypoints.openai.tool_parsers.multi_format_tool_parser import (
        MultiFormatToolParser,
    )


_TOOL_CALLS_OPEN = "<tool_calls>"
_TOOL_CALLS_CLOSE = "</tool_calls>"
_INVOKE_CLOSE = "</invoke>"
_PARAM_CLOSE = "</parameter>"
_INVOKE_OPEN_RE = re.compile(r'<invoke\s+name="([^"]+)"\s*>')
_PARAM_OPEN_RE = re.compile(
    r'<parameter\s+name="([^"]+)"(?:\s+string="(true|false)")?\s*>'
)

# State labels for the streaming state machine.
_STATE_BEFORE_FIRST = "before_first_tool_calls"
_STATE_AWAIT_NEXT = "await_tool_calls"
_STATE_IN_TOOL_CALLS = "in_tool_calls"
_STATE_IN_INVOKE = "in_invoke"
_STATE_IN_PARAMETER = "in_parameter"


class MinimaxToolCallStreamer(BaseToolCallStreamer):
    """Streamer for ``minimax`` and ``dsv32`` formats."""

    def __init__(
        self,
        tool_format: str,
        parser_cls: type[MultiFormatToolParser],
    ) -> None:
        super().__init__(tool_format, parser_cls)
        self._reset_state()

    def _reset_state(self) -> None:
        self._buffer: str = ""
        self._state: str = _STATE_BEFORE_FIRST
        self._tool_index: int = -1
        self._first_param_in_invoke: bool = True
        self._param_name: str | None = None
        self._param_string_flag: str | None = None
        # Tool indices for which the START delta (id + name) has already
        # been emitted in a prior feed() call.
        self._tool_name_sent: set[int] = set()
        # Accumulator for the current invoke's parsed arguments.
        self._current_args: dict[str, Any] = {}
        # Clear (not reassign) so the alias set in
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

        content_parts: list[str] = []
        tool_updates: dict[int, dict[str, Any]] = {}

        while True:
            if self._state == _STATE_BEFORE_FIRST:
                advanced = self._step_before_first(content_parts)
            elif self._state == _STATE_AWAIT_NEXT:
                advanced = self._step_await_next()
            elif self._state == _STATE_IN_TOOL_CALLS:
                advanced = self._step_in_tool_calls(tool_updates)
            elif self._state == _STATE_IN_INVOKE:
                advanced = self._step_in_invoke(tool_updates)
            elif self._state == _STATE_IN_PARAMETER:
                advanced = self._step_in_parameter(tool_updates)
            else:
                break
            if not advanced:
                break

        return self._build_delta(content_parts, tool_updates)

    # ---- State transitions ---------------------------------------------

    def _step_before_first(self, content_parts: list[str]) -> bool:
        idx = self._buffer.find(_TOOL_CALLS_OPEN)
        if idx != -1:
            if idx > 0:
                content_parts.append(self._buffer[:idx])
            self._buffer = self._buffer[idx + len(_TOOL_CALLS_OPEN) :]
            self._state = _STATE_IN_TOOL_CALLS
            return True
        safe = self._safe_emit_len(self._buffer, _TOOL_CALLS_OPEN)
        if safe:
            content_parts.append(self._buffer[:safe])
            self._buffer = self._buffer[safe:]
        return False

    def _step_await_next(self) -> bool:
        idx = self._buffer.find(_TOOL_CALLS_OPEN)
        if idx != -1:
            self._buffer = self._buffer[idx + len(_TOOL_CALLS_OPEN) :]
            self._state = _STATE_IN_TOOL_CALLS
            return True
        # Content between two ``<tool_calls>`` blocks is dropped (matches the
        # non-streaming reference, which only captures the prefix before the
        # first block). Trim the buffer down to a possible partial-marker
        # suffix so it does not grow unboundedly.
        safe = self._safe_emit_len(self._buffer, _TOOL_CALLS_OPEN)
        if safe:
            self._buffer = self._buffer[safe:]
        return False

    def _step_in_tool_calls(self, tool_updates: dict[int, dict[str, Any]]) -> bool:
        invoke_match = _INVOKE_OPEN_RE.search(self._buffer)
        close_idx = self._buffer.find(_TOOL_CALLS_CLOSE)

        if close_idx != -1 and (
            invoke_match is None or close_idx < invoke_match.start()
        ):
            self._buffer = self._buffer[close_idx + len(_TOOL_CALLS_CLOSE) :]
            self._state = _STATE_AWAIT_NEXT
            return True

        if invoke_match is not None:
            name = invoke_match.group(1)
            self._buffer = self._buffer[invoke_match.end() :]
            self._tool_index += 1
            self._first_param_in_invoke = True
            self._current_args = {}
            self._apply_tool_event(
                tool_updates,
                tool_index=self._tool_index,
                is_first=True,
                name=name,
                args_chunk="{",
            )
            self.record_tool_call_start(self._tool_index, name)
            self.record_args_fragment(self._tool_index, "{")
            self._state = _STATE_IN_INVOKE
            return True
        return False

    def _step_in_invoke(self, tool_updates: dict[int, dict[str, Any]]) -> bool:
        param_match = _PARAM_OPEN_RE.search(self._buffer)
        close_idx = self._buffer.find(_INVOKE_CLOSE)

        if close_idx != -1 and (param_match is None or close_idx < param_match.start()):
            self._buffer = self._buffer[close_idx + len(_INVOKE_CLOSE) :]
            self._apply_tool_event(
                tool_updates,
                tool_index=self._tool_index,
                args_chunk="}",
            )
            self.record_args_fragment(self._tool_index, "}")
            self.record_args_final(self._tool_index, dict(self._current_args))
            self._state = _STATE_IN_TOOL_CALLS
            return True

        if param_match is not None:
            self._param_name = param_match.group(1)
            self._param_string_flag = param_match.group(2)
            self._buffer = self._buffer[param_match.end() :]
            self._state = _STATE_IN_PARAMETER
            return True
        return False

    def _step_in_parameter(self, tool_updates: dict[int, dict[str, Any]]) -> bool:
        close_idx = self._buffer.find(_PARAM_CLOSE)
        if close_idx == -1:
            return False

        raw_value = self._buffer[:close_idx]
        self._buffer = self._buffer[close_idx + len(_PARAM_CLOSE) :]
        parsed_value = self._parse_value(raw_value)
        prefix = "" if self._first_param_in_invoke else ", "
        self._first_param_in_invoke = False
        args_chunk = (
            prefix
            + json.dumps(self._param_name, ensure_ascii=False)
            + ": "
            + json.dumps(parsed_value, ensure_ascii=False)
        )
        self._apply_tool_event(
            tool_updates,
            tool_index=self._tool_index,
            args_chunk=args_chunk,
        )
        if self._param_name is not None:
            self._current_args[self._param_name] = parsed_value
        self.record_args_fragment(self._tool_index, args_chunk)
        self._state = _STATE_IN_INVOKE
        return True

    # ---- Value parsing --------------------------------------------------

    def _parse_value(self, raw_value: str) -> Any:
        if self.tool_format == "dsv32" and self._param_string_flag == "true":
            return raw_value
        return self._parser._json_or_string(raw_value)

    # ---- Event aggregation ----------------------------------------------

    def _apply_tool_event(
        self,
        tool_updates: dict[int, dict[str, Any]],
        *,
        tool_index: int,
        args_chunk: str = "",
        is_first: bool = False,
        name: str | None = None,
    ) -> None:
        upd = tool_updates.setdefault(tool_index, {"args": ""})
        if is_first and tool_index not in self._tool_name_sent:
            upd["id"] = make_tool_call_id()
            upd["type"] = "function"
            upd["name"] = name
            self._tool_name_sent.add(tool_index)
        if args_chunk:
            upd["args"] = upd.get("args", "") + args_chunk

    # ---- Build delta ----------------------------------------------------

    def _build_delta(
        self,
        content_parts: list[str],
        tool_updates: dict[int, dict[str, Any]],
    ) -> DeltaMessage | None:
        content = "".join(content_parts)
        delta_tool_calls: list[DeltaToolCall] = []
        for tool_index in sorted(tool_updates.keys()):
            upd = tool_updates[tool_index]
            args_str = upd.get("args", "")
            function_kwargs: dict[str, Any] = {}
            if "name" in upd:
                function_kwargs["name"] = upd["name"]
            if args_str:
                function_kwargs["arguments"] = args_str
            function = DeltaFunctionCall(**function_kwargs) if function_kwargs else None
            delta_tool_calls.append(
                DeltaToolCall(
                    index=tool_index,
                    id=upd.get("id"),
                    type=upd.get("type"),
                    function=function,
                )
            )

        if not content and not delta_tool_calls:
            return None
        kwargs: dict[str, Any] = {}
        if content:
            kwargs["content"] = content
        if delta_tool_calls:
            kwargs["tool_calls"] = delta_tool_calls
        return DeltaMessage(**kwargs)

    # ---- Helpers --------------------------------------------------------

    @staticmethod
    def _safe_emit_len(buffer: str, marker: str) -> int:
        """Length of ``buffer`` that can be released without truncating a
        marker. The held-back tail is the longest suffix of ``buffer`` that
        is a non-empty prefix of ``marker``."""
        max_check = min(len(buffer), len(marker) - 1)
        for i in range(max_check, 0, -1):
            if marker.startswith(buffer[-i:]):
                return len(buffer) - i
        return len(buffer)
