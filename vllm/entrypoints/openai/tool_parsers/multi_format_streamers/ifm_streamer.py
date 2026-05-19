# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming tool-call parser for the IFM (LLM360) formats.

Handles three closely-related formats:
  * ``xml``       — ``<ifm|tool_call>name<ifm|arg_key>K</ifm|arg_key>
                       <ifm|arg_value>V</ifm|arg_value></ifm|tool_call>``
                    with optional ``<ifm|tool_calls>...</ifm|tool_calls>``
                    wrapper, and schema-driven type coercion.
  * ``xml_typed`` — same as ``xml`` plus an ``<ifm|arg_type>T</ifm|arg_type>``
                    between key and value (used for type coercion when no
                    JSON Schema is available on the request).
  * ``json``      — ``<ifm|tool_call>{"name":..., "arguments":{...}}
                    </ifm|tool_call>`` (one JSON tool-call per block).

See ``MultiFormatToolParser._extract_ifm_xml_tool_calls`` and
``_extract_ifm_json_tool_calls`` for the non-streaming reference behavior.
"""

from __future__ import annotations

from collections.abc import Sequence

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage
from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.base import (
    BaseToolCallStreamer,
)


class IFMToolCallStreamer(BaseToolCallStreamer):
    """Streamer for IFM ``xml`` / ``xml_typed`` / ``json`` formats."""

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
        # TODO(VLL-32 child A): implement streaming state machine.
        return None
