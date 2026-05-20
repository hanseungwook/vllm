# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming tool-call parser for the Python (literal-call) format.

Grammar (one or more blocks)::

    <tool_call>
    fn_name(kw1="v1", kw2=2, kw3={"k": True})
    </tool_call>

The body is parsed with ``ast`` (see
``MultiFormatToolParser._extract_python_tool_calls``), so streaming has to
buffer each ``<tool_call>...</tool_call>`` body, then emit the full
``name`` + ``arguments`` JSON atomically once the closing tag arrives.
"""

from __future__ import annotations

import ast
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
from vllm.logger import init_logger

logger = init_logger(__name__)


class PythonToolCallStreamer(BaseToolCallStreamer):
    """Streamer for the Python (literal-call) format.

    A Python tool body cannot be parsed incrementally (``ast`` needs the
    full ``<tool_call>...</tool_call>`` body), so this streamer buffers
    each block atomically and, once the closing marker arrives, emits two
    OpenAI-compatible tool-call deltas in the same ``DeltaMessage``:

    1. ``id`` + ``type="function"`` + ``function.name`` + ``index``.
    2. ``function.arguments`` with the full JSON object produced by
       :meth:`MultiFormatToolParser._handle_python_tool`.
    """

    _START_MARKER = "<tool_call>"
    _END_MARKER = "</tool_call>"

    def __init__(self, tool_format, parser_cls):
        super().__init__(tool_format, parser_cls)
        self._reset_state()

    def _reset_state(self) -> None:
        self._buffer: str = ""
        self._in_block: bool = False
        self._saw_first_call: bool = False
        self._next_index: int = 0
        # Clear (not reassign) the inherited lists so the alias set in
        # MultiFormatToolParser.__init__ stays valid across resets.
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    @staticmethod
    def _partial_marker_suffix_len(buffer: str, marker: str) -> int:
        """Length of the longest non-empty suffix of ``buffer`` that is a
        prefix of ``marker``.

        Used to hold back text that might be a partially-streamed opening
        marker, so we never emit ``"<tool_"`` as content only to later
        learn it was the start of ``<tool_call>``.
        """
        max_check = min(len(buffer), len(marker) - 1)
        for i in range(max_check, 0, -1):
            if marker.startswith(buffer[-i:]):
                return i
        return 0

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
        tool_call_deltas: list[DeltaToolCall] = []

        while True:
            if not self._in_block:
                start_idx = self._buffer.find(self._START_MARKER)
                if start_idx >= 0:
                    prefix = self._buffer[:start_idx]
                    if not self._saw_first_call and prefix:
                        content_parts.append(prefix)
                    self._buffer = self._buffer[start_idx + len(self._START_MARKER) :]
                    self._in_block = True
                    continue

                hold_len = self._partial_marker_suffix_len(
                    self._buffer, self._START_MARKER
                )
                safe_end = len(self._buffer) - hold_len
                if safe_end > 0:
                    emit_text = self._buffer[:safe_end]
                    if not self._saw_first_call:
                        content_parts.append(emit_text)
                    self._buffer = self._buffer[safe_end:]
                break

            end_idx = self._buffer.find(self._END_MARKER)
            if end_idx < 0:
                break

            body = self._buffer[:end_idx]
            self._buffer = self._buffer[end_idx + len(self._END_MARKER) :]
            self._in_block = False
            self._saw_first_call = True

            try:
                module = ast.parse(body.strip())
                if not module.body:
                    raise ValueError("Empty Python tool call body.")
                statements = []
                for statement in module.body:
                    if not isinstance(statement, ast.Expr) or not isinstance(
                        statement.value, ast.Call
                    ):
                        raise ValueError(
                            "Expected Python function call(s) inside "
                            "<tool_call> tags."
                        )
                    statements.append(statement.value)
                tool_calls_for_block = [
                    self._parser._handle_python_tool(call) for call in statements
                ]
            except Exception:
                logger.exception("Failed to parse Python tool call body: %r", body)
                continue

            for tool_call in tool_calls_for_block:
                index = self._next_index
                self._next_index += 1

                tool_call_deltas.append(
                    DeltaToolCall(
                        id=make_tool_call_id(),
                        type="function",
                        index=index,
                        function=DeltaFunctionCall(name=tool_call.function.name),
                    )
                )
                tool_call_deltas.append(
                    DeltaToolCall(
                        index=index,
                        function=DeltaFunctionCall(
                            arguments=tool_call.function.arguments
                        ),
                    )
                )

                self.record_tool_call_start(index, tool_call.function.name)
                self.record_args_fragment(index, tool_call.function.arguments)
                self.record_args_final(
                    index, json.loads(tool_call.function.arguments)
                )

        if tool_call_deltas:
            content_text = "".join(content_parts) if content_parts else None
            return DeltaMessage(
                content=content_text,
                tool_calls=tool_call_deltas,
            )

        if content_parts:
            return DeltaMessage(content="".join(content_parts))

        return None
