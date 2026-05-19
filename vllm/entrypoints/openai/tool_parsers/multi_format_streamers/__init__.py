# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming helpers for MultiFormatToolParser.

Each non-delegated tool format has a dedicated ``BaseToolCallStreamer``
subclass that turns model deltas into OpenAI-compatible
``DeltaMessage`` updates. ``build_streamer`` selects the correct one
for a given ``tool_format``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.base import (
    BaseToolCallStreamer,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.tool_parsers.multi_format_tool_parser import (
        MultiFormatToolParser,
    )

__all__ = ["BaseToolCallStreamer", "build_streamer"]


def build_streamer(
    tool_format: str,
    parser_cls: "type[MultiFormatToolParser]",
) -> BaseToolCallStreamer | None:
    """Return a streamer instance for ``tool_format``, or ``None`` if
    no streamer is registered for that format (caller should fall back
    to the no-stream behavior).
    """
    if tool_format in ("xml", "xml_typed", "json"):
        from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.ifm_streamer import (  # noqa: E501
            IFMToolCallStreamer,
        )

        return IFMToolCallStreamer(tool_format, parser_cls)
    if tool_format == "glm":
        from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.glm_streamer import (  # noqa: E501
            GLMToolCallStreamer,
        )

        return GLMToolCallStreamer(tool_format, parser_cls)
    if tool_format in ("minimax", "dsv32"):
        from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.minimax_streamer import (  # noqa: E501
            MinimaxToolCallStreamer,
        )

        return MinimaxToolCallStreamer(tool_format, parser_cls)
    if tool_format == "gptoss":
        from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.gptoss_streamer import (  # noqa: E501
            GPTOSSToolCallStreamer,
        )

        return GPTOSSToolCallStreamer(tool_format, parser_cls)
    if tool_format == "python":
        from vllm.entrypoints.openai.tool_parsers.multi_format_streamers.python_streamer import (  # noqa: E501
            PythonToolCallStreamer,
        )

        return PythonToolCallStreamer(tool_format, parser_cls)
    return None
