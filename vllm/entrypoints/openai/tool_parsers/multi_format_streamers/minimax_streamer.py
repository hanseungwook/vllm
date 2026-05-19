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

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.base import (
    BaseToolCallStreamer,
)


class MinimaxToolCallStreamer(BaseToolCallStreamer):
    """Streamer for ``minimax`` and ``dsv32`` formats."""

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
        # TODO(VLL-32 child C): implement streaming state machine.
        return None
