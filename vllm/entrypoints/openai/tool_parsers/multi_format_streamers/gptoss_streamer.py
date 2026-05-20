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

import json
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
        self._first_tool_seen: bool = False
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
        tool_call_deltas.append(
            DeltaToolCall(
                index=self._tool_index,
                id=make_tool_call_id(),
                type="function",
                function=DeltaFunctionCall(name=function_name, arguments=""),
            )
        )
        self.record_tool_call_start(self._tool_index, function_name)
        self._buffer = self._buffer[idx + 1 :]
        self._state = _STATE_ARGS
        return True

    def _step_args(self, tool_call_deltas: list[DeltaToolCall]) -> bool:
        # Atomic-per-block emit: buffer the entire args body, parse it once
        # ``</tool_call>`` arrives, then emit canonical ``json.dumps(parsed)``.
        # Two correctness wins over per-fragment streaming:
        #   1. EOS arriving mid-args silently drops the args (with the tool's
        #      name already on the wire from ``_step_header``); ``serving_chat``'s
        #      flush stays a no-op instead of emitting ``"{}"`` glued onto a
        #      partial JSON.
        #   2. Model output with non-canonical whitespace (e.g. ``{"a":1}``)
        #      is normalized to ``json.dumps`` form, so ``streamed_args_for_tool``
        #      stays byte-equal to ``json.dumps(prev_tool_call_arr[i]['arguments'])``
        #      and the flush never duplicates args at end-of-stream.
        close_idx = self._buffer.find(_TOOL_CALL_CLOSE)
        if close_idx < 0:
            return False

        args_portion = self._buffer[:close_idx].rstrip()
        try:
            parsed_args = json.loads(args_portion) if args_portion else {}
        except json.JSONDecodeError:
            # Match the non-streaming behavior: malformed JSON args degrade
            # to an empty dict rather than killing the stream.
            parsed_args = {}
        if not isinstance(parsed_args, dict):
            parsed_args = {}

        args_json = json.dumps(parsed_args, ensure_ascii=False)
        tool_call_deltas.append(
            DeltaToolCall(
                index=self._tool_index,
                function=DeltaFunctionCall(arguments=args_json),
            )
        )
        self.record_args_fragment(self._tool_index, args_json)
        self.record_args_final(self._tool_index, parsed_args)
        self._buffer = self._buffer[close_idx + len(_TOOL_CALL_CLOSE) :]
        self._state = _STATE_OUTSIDE
        return True

    @staticmethod
    def _suffix_prefix_length(buffer: str, tag: str) -> int:
        # Longest k such that ``buffer`` ends with the first k characters of
        # ``tag``. Used to hold back partial marker tokens at delta boundaries.
        n = len(buffer)
        for k in range(min(len(tag), n), 0, -1):
            if buffer[n - k :] == tag[:k]:
                return k
        return 0
