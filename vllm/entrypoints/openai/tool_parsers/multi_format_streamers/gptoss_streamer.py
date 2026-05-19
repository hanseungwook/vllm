# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming tool-call parser for the GPT-OSS format.

Grammar (one or more blocks)::

    <tool_call>[assistant ]to=functions.NAME[ json]
    {... JSON arguments ...}
    </tool_call>

See ``MultiFormatToolParser._extract_gptoss_tool_calls`` for the
non-streaming reference. JSON arguments can be streamed directly once the
header line is consumed.
"""

from __future__ import annotations

from collections.abc import Sequence

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

_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_HEADER_RE = re.compile(r"\s*(?:assistant\s+)?to=functions\.(\S+?)(?:\s+json)?\s*\n")

_STATE_OUTSIDE = "outside"
_STATE_HEADER = "header"
_STATE_ARGS = "args"


class GPTOSSToolCallStreamer(BaseToolCallStreamer):
    """Streamer for the GPT-OSS ``to=functions.NAME`` format."""

    def __init__(self, tool_format, parser_cls):
        super().__init__(tool_format, parser_cls)
        self._reset()

    def _reset(self) -> None:
        self._buffer: str = ""
        self._state: str = _STATE_OUTSIDE
        self._tool_index: int = -1
        self._args_emitted_len: int = 0
        self._first_tool_seen: bool = False

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
            self._reset()
        self._buffer += delta_text

        contents: list[str] = []
        tool_call_deltas: list[DeltaToolCall] = []

        while True:
            if self._state == _STATE_OUTSIDE:
                if not self._step_outside(contents):
                    break
            elif self._state == _STATE_HEADER:
                if not self._step_header(tool_call_deltas):
                    break
            elif self._state == _STATE_ARGS:
                if not self._step_args(tool_call_deltas):
                    break
            else:
                break

        content = "".join(contents) if contents else None
        if content is None and not tool_call_deltas:
            return None
        return DeltaMessage(content=content, tool_calls=tool_call_deltas)

    def _step_outside(self, contents: list[str]) -> bool:
        idx = self._buffer.find(_TOOL_CALL_OPEN)
        if idx >= 0:
            prefix = self._buffer[:idx]
            if not self._first_tool_seen and prefix:
                contents.append(prefix)
            self._buffer = self._buffer[idx + len(_TOOL_CALL_OPEN) :]
            self._state = _STATE_HEADER
            self._first_tool_seen = True
            return True
        # No marker found. Hold back any suffix that could be the start of
        # ``<tool_call>``; emit (or drop) the rest.
        holdback = self._suffix_prefix_length(self._buffer, _TOOL_CALL_OPEN)
        safe = len(self._buffer) - holdback
        if safe > 0:
            if not self._first_tool_seen:
                contents.append(self._buffer[:safe])
            self._buffer = self._buffer[safe:]
        return False

    def _step_header(self, tool_call_deltas: list[DeltaToolCall]) -> bool:
        idx = self._buffer.find("\n")
        if idx < 0:
            return False
        header_line = self._buffer[: idx + 1]
        match = _HEADER_RE.match(header_line)
        if not match:
            # Malformed header — drop the line and recover to OUTSIDE.
            self._buffer = self._buffer[idx + 1 :]
            self._state = _STATE_OUTSIDE
            return True
        function_name = match.group(1)
        self._tool_index += 1
        self._args_emitted_len = 0
        tool_call_deltas.append(
            DeltaToolCall(
                index=self._tool_index,
                id=make_tool_call_id(),
                type="function",
                function=DeltaFunctionCall(name=function_name, arguments=""),
            )
        )
        self._buffer = self._buffer[idx + 1 :]
        self._state = _STATE_ARGS
        return True

    def _step_args(self, tool_call_deltas: list[DeltaToolCall]) -> bool:
        close_idx = self._buffer.find(_TOOL_CALL_CLOSE)
        if close_idx >= 0:
            args_portion = self._buffer[:close_idx].rstrip()
            new_to_emit = args_portion[self._args_emitted_len :]
            if new_to_emit:
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=self._tool_index,
                        function=DeltaFunctionCall(arguments=new_to_emit),
                    )
                )
            self._buffer = self._buffer[close_idx + len(_TOOL_CALL_CLOSE) :]
            self._state = _STATE_OUTSIDE
            self._args_emitted_len = 0
            return True
        safe_end = self._safe_args_end()
        if safe_end > self._args_emitted_len:
            new_to_emit = self._buffer[self._args_emitted_len : safe_end]
            tool_call_deltas.append(
                DeltaToolCall(
                    index=self._tool_index,
                    function=DeltaFunctionCall(arguments=new_to_emit),
                )
            )
            self._args_emitted_len = safe_end
        return False

    def _safe_args_end(self) -> int:
        # Returns the index up to which the args buffer is safe to emit.
        # Holds back: any suffix that could be a prefix of ``</tool_call>``,
        # plus any trailing whitespace (which the non-streaming path strips
        # via ``.strip()`` before the close tag).
        n = len(self._buffer)
        if n == 0:
            return 0
        holdback = self._suffix_prefix_length(self._buffer, _TOOL_CALL_CLOSE)
        end = n - holdback
        while end > 0 and self._buffer[end - 1].isspace():
            end -= 1
        return end

    @staticmethod
    def _suffix_prefix_length(buffer: str, tag: str) -> int:
        # Longest k such that ``buffer`` ends with the first k characters of
        # ``tag``. Used to hold back partial marker tokens at delta boundaries.
        n = len(buffer)
        for k in range(min(len(tag), n), 0, -1):
            if buffer[n - k :] == tag[:k]:
                return k
        return 0
