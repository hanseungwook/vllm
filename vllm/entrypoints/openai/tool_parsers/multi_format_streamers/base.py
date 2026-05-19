# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, DeltaMessage

if TYPE_CHECKING:
    from vllm.entrypoints.openai.tool_parsers.multi_format_tool_parser import (
        MultiFormatToolParser,
    )


class BaseToolCallStreamer(ABC):
    """Per-request streaming extractor for one tool-format family.

    A new instance is created per parser instance (i.e., per request).
    Subclasses MUST hold all per-request state on ``self`` and reset it
    when ``feed`` is called with empty ``previous_text`` (the first call
    of a new streaming session).

    Subclasses MUST emit OpenAI-compatible deltas:
      * Plain content before any tool marker  -> ``DeltaMessage(content=...)``.
      * First delta for a given tool          -> include ``id``,
        ``type="function"``, and ``function.name`` (arguments optional).
      * Subsequent deltas for the same tool   -> only ``function.arguments``
        as incremental JSON fragments whose concatenation equals the
        non-streaming ``extract_tool_calls`` arguments string.
      * Multiple parallel tool calls          -> monotonically increasing
        ``index`` starting at 0.
    """

    def __init__(
        self,
        tool_format: str,
        parser_cls: "type[MultiFormatToolParser]",
    ) -> None:
        self.tool_format = tool_format
        self._parser = parser_cls

    @abstractmethod
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
        """Consume one streaming delta and return what to emit.

        Return ``None`` to suppress this chunk (vLLM skips it).
        Return ``DeltaMessage(content="")`` to keep the stream alive
        with no observable content.
        """
        raise NotImplementedError
