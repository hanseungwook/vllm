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

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.base import (
    BaseToolCallStreamer,
)


class PythonToolCallStreamer(BaseToolCallStreamer):
    """Streamer for the Python (literal-call) format."""

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
        # TODO(VLL-32 child E): implement streaming state machine.
        return None
