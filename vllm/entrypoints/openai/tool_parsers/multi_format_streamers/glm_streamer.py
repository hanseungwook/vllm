# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming tool-call parser for the GLM format.

Grammar:
    <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>

See ``MultiFormatToolParser._extract_glm_tool_calls`` for the non-streaming
reference behavior, including value coercion via ``_deserialize_glm_value``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

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


class GLMToolCallStreamer(BaseToolCallStreamer):
    """Streamer for the GLM ``<tool_call><arg_key>...</tool_call>`` format."""

    _TOOL_CALL_OPEN = "<tool_call>"
    _TOOL_CALL_CLOSE = "</tool_call>"
    _ARG_KEY_OPEN = "<arg_key>"
    _ARG_KEY_CLOSE = "</arg_key>"
    _ARG_VALUE_OPEN = "<arg_value>"
    _ARG_VALUE_CLOSE = "</arg_value>"

    _OUTSIDE = "outside"
    _NAME = "name"
    _KEY = "key"
    _AFTER_KEY = "after_key"
    _VALUE = "value"
    _AFTER_VALUE = "after_value"

    def __init__(self, tool_format, parser_cls):
        super().__init__(tool_format, parser_cls)
        self._reset_state()

    def _reset_state(self) -> None:
        self._state: str = self._OUTSIDE
        self._buffer: str = ""
        self._current_tool_idx: int = -1
        self._current_tool_name: str = ""
        self._first_tool_seen: bool = False
        self._name_buffer: str = ""
        self._key_buffer: str = ""
        self._value_buffer: str = ""
        self._current_key: str = ""
        self._brace_opened: bool = False
        self._current_args: dict = {}
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
        tool_event: dict = {
            "present": False,
            "index": None,
            "id": None,
            "name": None,
            "args_parts": [],
        }

        while self._step(content_parts, tool_event, request) == "progress":
            pass

        return self._build_delta(content_parts, tool_event)

    @staticmethod
    def _find_first_marker(text: str, markers: Sequence[str]) -> tuple[int, str | None]:
        best_idx = -1
        best_marker: str | None = None
        for m in markers:
            idx = text.find(m)
            if idx == -1:
                continue
            if best_idx == -1 or idx < best_idx:
                best_idx = idx
                best_marker = m
        return best_idx, best_marker

    @staticmethod
    def _safe_emit_end(text: str, markers: Sequence[str]) -> int:
        """Length of the prefix of ``text`` that is safe to commit without
        risking that the unemitted tail is the start of one of ``markers``.
        """
        n = len(text)
        if n == 0:
            return 0
        max_partial = 0
        for m in markers:
            for k in range(min(len(m), n), 0, -1):
                if text.endswith(m[:k]):
                    if k > max_partial:
                        max_partial = k
                    break
        return n - max_partial

    def _step(self, content_parts, tool_event, request) -> str:
        s = self._state
        if s == self._OUTSIDE:
            return self._step_outside(content_parts, tool_event)
        if s == self._NAME:
            return self._step_name(tool_event)
        if s == self._KEY:
            return self._step_key()
        if s == self._AFTER_KEY:
            return self._step_after_key(tool_event)
        if s == self._VALUE:
            return self._step_value(tool_event, request)
        if s == self._AFTER_VALUE:
            return self._step_after_value(tool_event)
        return "stuck"

    def _step_outside(self, content_parts, tool_event) -> str:
        idx = self._buffer.find(self._TOOL_CALL_OPEN)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, [self._TOOL_CALL_OPEN])
            if safe_end <= 0:
                return "stuck"
            text = self._buffer[:safe_end]
            if text and not self._first_tool_seen:
                if tool_event["present"]:
                    return "stuck"
                content_parts.append(text)
            self._buffer = self._buffer[safe_end:]
            return "progress"

        text = self._buffer[:idx]
        new_idx = self._current_tool_idx + 1
        if tool_event["present"] and tool_event["index"] != new_idx:
            return "stuck"
        if text and not self._first_tool_seen:
            if tool_event["present"]:
                return "stuck"
            content_parts.append(text)

        self._buffer = self._buffer[idx + len(self._TOOL_CALL_OPEN) :]
        self._current_tool_idx = new_idx
        self._first_tool_seen = True
        self._state = self._NAME
        self._name_buffer = ""
        self._current_tool_name = ""
        self._brace_opened = False
        self._current_args = {}
        return "progress"

    def _step_name(self, tool_event) -> str:
        markers = [self._ARG_KEY_OPEN, self._TOOL_CALL_CLOSE]
        idx, marker = self._find_first_marker(self._buffer, markers)
        if idx == -1 or marker is None:
            safe_end = self._safe_emit_end(self._buffer, markers)
            if safe_end <= 0:
                return "stuck"
            self._name_buffer += self._buffer[:safe_end]
            self._buffer = self._buffer[safe_end:]
            return "progress"
        if tool_event["present"] and tool_event["index"] != self._current_tool_idx:
            return "stuck"

        self._name_buffer += self._buffer[:idx]
        name = self._name_buffer.strip()
        self._buffer = self._buffer[idx + len(marker) :]
        if not name:
            # Empty-name tool: skip without padding prev_tool_call_arr.
            # Decrement so the next valid tool gets the right slot index
            # (matches IFM XML's _step_collecting_name recovery).
            self._current_tool_idx -= 1
            self._state = self._OUTSIDE
            return "progress"

        self._current_tool_name = name
        tool_event["present"] = True
        tool_event["index"] = self._current_tool_idx
        tool_event["id"] = make_tool_call_id()
        tool_event["name"] = name
        self.record_tool_call_start(self._current_tool_idx, name)

        if marker == self._ARG_KEY_OPEN:
            self._state = self._KEY
            self._key_buffer = ""
        else:
            tool_event["args_parts"].append("{}")
            self.record_args_fragment(self._current_tool_idx, "{}")
            self.record_args_final(self._current_tool_idx, {})
            self._brace_opened = True
            self._state = self._OUTSIDE
        return "progress"

    def _step_key(self) -> str:
        idx = self._buffer.find(self._ARG_KEY_CLOSE)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, [self._ARG_KEY_CLOSE])
            if safe_end <= 0:
                return "stuck"
            self._key_buffer += self._buffer[:safe_end]
            self._buffer = self._buffer[safe_end:]
            return "progress"
        self._key_buffer += self._buffer[:idx]
        self._current_key = self._key_buffer.strip()
        self._buffer = self._buffer[idx + len(self._ARG_KEY_CLOSE) :]
        self._state = self._AFTER_KEY
        return "progress"

    def _step_after_key(self, tool_event) -> str:
        # Both ``<arg_value>`` (well-formed) and ``</tool_call>`` (malformed:
        # key without a value) are valid next markers. Without handling the
        # second case, a model that emits ``<arg_key>k</arg_key></tool_call>``
        # would stall the stream forever.
        markers = [self._ARG_VALUE_OPEN, self._TOOL_CALL_CLOSE]
        idx, marker = self._find_first_marker(self._buffer, markers)
        if idx == -1 or marker is None:
            safe_end = self._safe_emit_end(self._buffer, markers)
            if safe_end <= 0:
                return "stuck"
            self._buffer = self._buffer[safe_end:]
            return "progress"
        self._buffer = self._buffer[idx + len(marker) :]
        if marker == self._ARG_VALUE_OPEN:
            self._value_buffer = ""
            self._state = self._VALUE
        else:
            # Orphan key (key without value) — drop it. Non-streaming regex
            # requires ``<arg_key>K</arg_key><arg_value>V</arg_value>`` pairs,
            # so it also silently drops a key without a value. Close the tool
            # with whatever args were collected before this key.
            self._emit_tool_close(tool_event)
            self._state = self._OUTSIDE
        return "progress"

    def _emit_tool_close(self, tool_event) -> None:
        """Append the closing args fragment for the current tool and record
        the final args dict. Used by ``_step_after_value`` (normal path) and
        ``_step_after_key`` (orphan-key recovery)."""
        if not tool_event["present"]:
            tool_event["present"] = True
            tool_event["index"] = self._current_tool_idx
        if self._brace_opened:
            tool_event["args_parts"].append("}")
            self.record_args_fragment(self._current_tool_idx, "}")
        else:
            tool_event["args_parts"].append("{}")
            self.record_args_fragment(self._current_tool_idx, "{}")
        self.record_args_final(self._current_tool_idx, dict(self._current_args))

    def _step_value(self, tool_event, request) -> str:
        idx = self._buffer.find(self._ARG_VALUE_CLOSE)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, [self._ARG_VALUE_CLOSE])
            if safe_end <= 0:
                return "stuck"
            self._value_buffer += self._buffer[:safe_end]
            self._buffer = self._buffer[safe_end:]
            return "progress"
        if tool_event["present"] and tool_event["index"] != self._current_tool_idx:
            return "stuck"

        self._value_buffer += self._buffer[:idx]
        value_str = self._value_buffer.strip()
        self._buffer = self._buffer[idx + len(self._ARG_VALUE_CLOSE) :]

        coerced = self._parser._coerce_argument_value(
            value_str,
            self._current_tool_name,
            self._current_key,
            request.tools,
            from_text=True,
        )
        key_json = json.dumps(self._current_key, ensure_ascii=False)
        value_json = json.dumps(coerced, ensure_ascii=False)
        if self._brace_opened:
            fragment = ", " + key_json + ": " + value_json
        else:
            fragment = "{" + key_json + ": " + value_json
            self._brace_opened = True

        if not tool_event["present"]:
            tool_event["present"] = True
            tool_event["index"] = self._current_tool_idx
        tool_event["args_parts"].append(fragment)
        self._current_args[self._current_key] = coerced
        self.record_args_fragment(self._current_tool_idx, fragment)
        # Keep prev_tool_call_arr in sync after every arg so serving_chat's
        # end-of-stream flush stays a no-op even on EOS-mid-call (the
        # canonical json.dumps of the partial dict is a strict prefix of the
        # streamed args, missing only the closing "}").
        self.record_args_final(self._current_tool_idx, dict(self._current_args))
        self._state = self._AFTER_VALUE
        return "progress"

    def _step_after_value(self, tool_event) -> str:
        markers = [self._ARG_KEY_OPEN, self._TOOL_CALL_CLOSE]
        idx, marker = self._find_first_marker(self._buffer, markers)
        if idx == -1 or marker is None:
            safe_end = self._safe_emit_end(self._buffer, markers)
            if safe_end <= 0:
                return "stuck"
            self._buffer = self._buffer[safe_end:]
            return "progress"
        if tool_event["present"] and tool_event["index"] != self._current_tool_idx:
            return "stuck"

        self._buffer = self._buffer[idx + len(marker) :]
        if marker == self._ARG_KEY_OPEN:
            self._state = self._KEY
            self._key_buffer = ""
        else:
            self._emit_tool_close(tool_event)
            self._state = self._OUTSIDE
        return "progress"

    def _build_delta(
        self,
        content_parts: list[str],
        tool_event: dict,
    ) -> DeltaMessage | None:
        content_str = "".join(content_parts) if content_parts else None
        tool_calls: list[DeltaToolCall] = []
        if tool_event["present"]:
            args_str = (
                "".join(tool_event["args_parts"]) if tool_event["args_parts"] else None
            )
            function = DeltaFunctionCall(
                name=tool_event["name"],
                arguments=args_str,
            )
            tool_calls.append(
                DeltaToolCall(
                    index=tool_event["index"],
                    id=tool_event["id"],
                    type="function" if tool_event["id"] else None,
                    function=function,
                )
            )
        if content_str is None and not tool_calls:
            return None
        return DeltaMessage(content=content_str, tool_calls=tool_calls)
