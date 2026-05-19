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

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.base import (
    BaseToolCallStreamer,
)


class GPTOSSToolCallStreamer(BaseToolCallStreamer):
    """Streamer for the GPT-OSS ``to=functions.NAME`` format."""

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
        # TODO(VLL-32 child D): implement streaming state machine.
        return None
