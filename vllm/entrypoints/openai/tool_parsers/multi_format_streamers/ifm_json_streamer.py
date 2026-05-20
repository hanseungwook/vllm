# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming tool-call parser for the IFM (LLM360) ``json`` format.

Grammar::

    <ifm|tool_call>{"name": "...", "arguments": {...}}</ifm|tool_call>

or a list of calls inside one block::

    <ifm|tool_call>[{"name": "fa", "arguments": {...}},
                    {"name": "fb", "arguments": {...}}]</ifm|tool_call>

The body is JSON and cannot be parsed incrementally, so this streamer
buffers everything between ``<ifm|tool_call>`` and ``</ifm|tool_call>``,
parses atomically once the close marker arrives, and emits one complete
``DeltaToolCall`` per call. See
``MultiFormatToolParser._extract_ifm_json_tool_calls`` for the
non-streaming reference behavior.

``prev_tool_call_arr`` and ``streamed_args_for_tool`` are inherited from
``BaseToolCallStreamer`` and aliased to the owning parser instance, so
``serving_chat.py``'s end-of-stream flush logic sees the live streaming
state. Because each call's full coerced JSON is emitted atomically, the
flush is always a no-op once the call closes; if EOS hits while the body
is still buffering we can't recover a partial JSON object, so nothing is
recorded for that aborted call.

Known limitation: when a single ``<ifm|tool_call>[...]</ifm|tool_call>``
list-body produces N>1 calls AND the closing marker arrives in the same
output chunk as ``finish_reason``, ``serving_chat.py``'s flush logic
indexes ``delta_message.tool_calls[0]`` against ``prev_tool_call_arr[-1]``
(the last call) and can mis-compute the remaining-args diff. In
practice this is rare because real tokenizers separate ``</ifm|tool_call>``
from ``<EOS>`` into distinct tokens (and therefore distinct ``feed()``
calls), so the final chunk for ``feed()`` sees only EOS and emits no
tool_calls. Multi-chunk streaming is unaffected. A complete fix
requires ``serving_chat`` to handle multi-tool-call deltas; tracked as
a follow-up.
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


class IFMJSONToolCallStreamer(BaseToolCallStreamer):
    """Streamer for IFM ``json`` format."""

    _OUTER_OPEN = "<ifm|tool_calls>"
    _OUTER_CLOSE = "</ifm|tool_calls>"
    _CALL_OPEN = "<ifm|tool_call>"
    _CALL_CLOSE = "</ifm|tool_call>"
    _ALL_MARKERS = (_OUTER_OPEN, _OUTER_CLOSE, _CALL_OPEN, _CALL_CLOSE)

    def __init__(self, tool_format: str, parser_cls) -> None:
        super().__init__(tool_format, parser_cls)
        self._reset_state()

    def _reset_state(self) -> None:
        self._buffer: str = ""
        self._in_block: bool = False
        self._saw_first_call: bool = False
        self._exhausted: bool = False
        self._next_index: int = 0
        # Clear (not reassign) so the alias set in
        # MultiFormatToolParser.__init__ stays valid across resets.
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    @staticmethod
    def _partial_marker_suffix_len(buffer: str, markers: Sequence[str]) -> int:
        """Longest non-empty suffix of ``buffer`` that is a prefix of any
        marker. Used to hold back potentially-partial markers."""
        n = len(buffer)
        if n == 0:
            return 0
        best = 0
        for marker in markers:
            for k in range(min(len(marker) - 1, n), best, -1):
                if buffer.endswith(marker[:k]):
                    best = k
                    break
        return best

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
        tool_deltas: list[DeltaToolCall] = []

        while True:
            if self._exhausted:
                self._buffer = ""
                break

            if not self._in_block:
                if not self._step_outside(content_parts):
                    break
                continue

            end_idx = self._buffer.find(self._CALL_CLOSE)
            if end_idx == -1:
                break
            body = self._buffer[:end_idx].strip()
            self._buffer = self._buffer[end_idx + len(self._CALL_CLOSE) :]
            self._in_block = False
            self._saw_first_call = True
            self._parse_block_body(body, request, tool_deltas)

        return self._build_delta(content_parts, tool_deltas)

    def _step_outside(self, content_parts: list[str]) -> bool:
        """Consume the buffer up to the next marker. Returns ``True`` if
        progress was made (caller should loop), ``False`` to break out."""
        idx_outer_open = self._buffer.find(self._OUTER_OPEN)
        idx_outer_close = self._buffer.find(self._OUTER_CLOSE)
        idx_call_open = self._buffer.find(self._CALL_OPEN)

        best = -1
        marker = ""
        kind = ""
        for i, m, k in (
            (idx_outer_open, self._OUTER_OPEN, "outer_open"),
            (idx_outer_close, self._OUTER_CLOSE, "outer_close"),
            (idx_call_open, self._CALL_OPEN, "call_open"),
        ):
            if i != -1 and (best == -1 or i < best):
                best, marker, kind = i, m, k

        if best == -1:
            hold = self._partial_marker_suffix_len(self._buffer, self._ALL_MARKERS)
            safe_end = len(self._buffer) - hold
            if safe_end > 0:
                emit = self._buffer[:safe_end]
                if emit and not self._saw_first_call:
                    content_parts.append(emit)
                self._buffer = self._buffer[safe_end:]
            return False

        prefix = self._buffer[:best]
        if prefix and not self._saw_first_call:
            content_parts.append(prefix)
        self._buffer = self._buffer[best + len(marker) :]
        if kind == "outer_close":
            self._exhausted = True
        elif kind == "call_open":
            self._in_block = True
        return True

    def _parse_block_body(
        self,
        body: str,
        request: ChatCompletionRequest,
        tool_deltas: list[DeltaToolCall],
    ) -> None:
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            # Match non-streaming behavior: silently skip malformed JSON.
            return
        raw_calls = parsed if isinstance(parsed, list) else [parsed]
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
                    function.get("arguments", {})
                )
            except Exception:
                continue
            try:
                coerced = self._parser._coerce_arguments(name, arguments, request.tools)
            except Exception:
                coerced = arguments

            args_json = json.dumps(coerced, ensure_ascii=False)
            index = self._next_index
            self._next_index += 1
            tool_id = make_tool_call_id()
            self.record_tool_call_start(index, name)
            self.record_args_fragment(index, args_json)
            self.record_args_final(index, dict(coerced))
            tool_deltas.append(
                DeltaToolCall(
                    index=index,
                    id=tool_id,
                    type="function",
                    function=DeltaFunctionCall(name=name, arguments=args_json),
                )
            )

    def _build_delta(
        self,
        content_parts: list[str],
        tool_deltas: list[DeltaToolCall],
    ) -> DeltaMessage | None:
        if tool_deltas:
            content = "".join(content_parts) if content_parts else None
            return DeltaMessage(content=content, tool_calls=tool_deltas)
        if content_parts:
            return DeltaMessage(content="".join(content_parts))
        return None
