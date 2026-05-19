# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming tool-call parser for the GLM format.

Grammar:
    <tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>...</tool_call>

See ``MultiFormatToolParser._extract_glm_tool_calls`` for the non-streaming
reference behavior, including value coercion via ``_deserialize_glm_value``.
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.base import (
    BaseToolCallStreamer,
)


class GLMToolCallStreamer(BaseToolCallStreamer):
    """Streamer for the GLM ``<tool_call><arg_key>...</tool_call>`` format."""

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
        # TODO(VLL-32 child B): implement streaming state machine.
        return None
