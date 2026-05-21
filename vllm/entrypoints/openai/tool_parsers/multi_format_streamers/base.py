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
        # End-of-stream flush state read by ``serving_chat.py``. ``serving_chat``
        # uses ``len(prev_tool_call_arr) > 0`` to decide ``finish_reason="tool_calls"``
        # and reads both lists to flush any unstreamed argument tail. Subclasses
        # MUST mutate these lists in place (``.append``, ``.clear``) — never
        # reassign — so the owning ``MultiFormatToolParser`` can alias them.
        self.prev_tool_call_arr: list[dict] = []
        self.streamed_args_for_tool: list[str] = []

    def record_tool_call_start(self, index: int, name: str) -> None:
        """Append a placeholder entry for a newly-started tool call.

        Subclasses MUST call this when emitting the first delta for each
        tool call (the delta that carries ``id`` + ``function.name``).
        Pads both lists so ``index`` is a valid slot.
        """
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({"name": "", "arguments": {}})
            self.streamed_args_for_tool.append("")
        self.prev_tool_call_arr[index]["name"] = name

    def record_args_fragment(self, index: int, args_fragment: str) -> None:
        """Record an emitted ``function.arguments`` fragment for a tool call.

        ``args_fragment`` is the raw JSON-string fragment that went into the
        emitted ``DeltaMessage``. ``serving_chat.py`` compares
        ``json.dumps(prev_tool_call_arr[i]['arguments'])`` with
        ``streamed_args_for_tool[i]`` to compute any remaining flush; for
        streamers that emit complete coerced JSON for every arg, these match
        once the tool call is closed and the flush is a no-op.
        """
        while len(self.streamed_args_for_tool) <= index:
            self.prev_tool_call_arr.append({"name": "", "arguments": {}})
            self.streamed_args_for_tool.append("")
        self.streamed_args_for_tool[index] += args_fragment

    def record_args_final(self, index: int, arguments: dict) -> None:
        """Record the fully-parsed arguments dict for a completed tool call.

        Used at end-of-tool emission so ``serving_chat.py``'s unstreamed-arg
        flush can compute ``json.dumps(arguments) - streamed_args_for_tool``
        and emit any tail bytes (a no-op when our streaming output already
        matches the canonical JSON serialization).
        """
        while len(self.prev_tool_call_arr) <= index:
            self.prev_tool_call_arr.append({"name": "", "arguments": {}})
            self.streamed_args_for_tool.append("")
        self.prev_tool_call_arr[index]["arguments"] = arguments

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
