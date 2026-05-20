# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming tool-call parser for the IFM (LLM360) ``json`` format.

Grammar (matches the non-streaming ``_extract_ifm_json_tool_calls``)::

    <ifm|tool_call>{"name": "X", "arguments": {...}}</ifm|tool_call>
    <ifm|tool_call>[{"name": "fa", ...}, {"name": "fb", ...}]</ifm|tool_call>
    <ifm|tool_call>{"function": {"name": "X", "arguments": {...}}}</ifm|tool_call>

The body is JSON. ``partial_json_parser`` decodes whatever portion has
arrived so far, so the function name is emitted as soon as it's complete
and ``arguments`` stream as a series of diffs (Hermes/Llama3 pattern).
The concatenation of streamed argument fragments always equals the
canonical coerced JSON the non-streaming parser would have produced.

Invariant after every emit::

    json.dumps(prev_tool_call_arr[i]['arguments'], ensure_ascii=False)
        .startswith(streamed_args_for_tool[i])

That is what makes ``serving_chat.py``'s end-of-stream flush a no-op
(or a small tail diff) rather than a duplication: ``serving_chat``
indexes by ``len(prev_tool_call_arr) - 1`` and computes the tail by
length-stripping ``streamed_args_for_tool[i]`` from the canonical args
JSON. Because we record per-index state and stream in canonical order,
the list-body case (``[{...},{...}]`` in a single block) no longer
emits multiple ``tool_calls`` in one ``DeltaMessage`` — each list
element becomes its own sequence of deltas with a distinct ``index``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import partial_json_parser
from partial_json_parser.core.options import Allow

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
from vllm.entrypoints.openai.tool_parsers.utils import find_common_prefix


@dataclass
class _ContentEmission:
    text: str


@dataclass
class _ToolNameEmission:
    index: int
    id: str
    name: str


@dataclass
class _ToolArgsEmission:
    index: int
    args: str


class IFMJSONToolCallStreamer(BaseToolCallStreamer):
    """Incremental streamer for IFM ``json`` format."""

    _OUTER_OPEN = "<ifm|tool_calls>"
    _OUTER_CLOSE = "</ifm|tool_calls>"
    _CALL_OPEN = "<ifm|tool_call>"
    _CALL_CLOSE = "</ifm|tool_call>"
    _ALL_MARKERS = (_OUTER_OPEN, _OUTER_CLOSE, _CALL_OPEN, _CALL_CLOSE)

    _BEFORE = "before"
    _BETWEEN = "between"
    _IN_BLOCK = "in_block"
    _AFTER_WRAPPER = "after_wrapper"

    _MAX_BUFFER_BYTES = 1 << 20

    def __init__(self, tool_format: str, parser_cls) -> None:
        super().__init__(tool_format, parser_cls)
        self._reset_state()

    def _reset_state(self) -> None:
        self._buffer: str = ""
        self._phase: str = self._BEFORE
        self._has_outer_wrapper: bool = False
        self._saw_first_call: bool = False
        # Per-block partial-parse state.
        self._block_buffer: str = ""
        # First tool index for the current ``<ifm|tool_call>`` block. Set
        # when the block opens and frozen for the lifetime of that block
        # so list-body indices stay consistent across feeds.
        self._block_start_id: int = 0
        # Per-tool incremental-stream state.
        self._current_tool_id: int = -1
        self._current_tool_name_sent: bool = False
        # Clear (not reassign) so the alias set in
        # MultiFormatToolParser.__init__ stays valid across resets.
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    @staticmethod
    def _partial_marker_suffix_len(buffer: str, markers: Sequence[str]) -> int:
        """Longest non-empty suffix of ``buffer`` that is a strict prefix of
        any marker. Used to hold back potentially-partial markers."""
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
        if len(self._buffer) > self._MAX_BUFFER_BYTES:
            self._buffer = self._buffer[-(self._MAX_BUFFER_BYTES // 2) :]

        emissions: list = []
        while True:
            if not self._step(emissions, request):
                break
        return self._build_delta(emissions)

    def _step(self, emissions: list, request: ChatCompletionRequest) -> bool:
        phase = self._phase
        if phase == self._BEFORE:
            return self._step_before(emissions)
        if phase == self._BETWEEN:
            return self._step_between()
        if phase == self._IN_BLOCK:
            return self._step_in_block(emissions, request)
        # _AFTER_WRAPPER: outer </ifm|tool_calls> seen. ``finditer`` in the
        # non-streaming parser still picks up later <ifm|tool_call> blocks,
        # so we keep scanning but drop intervening content.
        return self._step_after_wrapper()

    def _step_before(self, emissions: list) -> bool:
        markers = (self._OUTER_OPEN, self._CALL_OPEN)
        pos = self._first_marker(markers)
        if pos is None:
            hold = self._partial_marker_suffix_len(self._buffer, markers)
            safe_end = len(self._buffer) - hold
            if safe_end > 0:
                emit = self._buffer[:safe_end]
                self._buffer = self._buffer[safe_end:]
                if emit and not self._saw_first_call:
                    emissions.append(_ContentEmission(emit))
            return False

        idx, marker = pos
        prefix = self._buffer[:idx]
        if prefix and not self._saw_first_call:
            emissions.append(_ContentEmission(prefix))
        self._buffer = self._buffer[idx + len(marker) :]

        if marker == self._OUTER_OPEN:
            self._has_outer_wrapper = True
            self._phase = self._BETWEEN
        else:
            self._enter_block()
        return True

    def _step_between(self) -> bool:
        # Inside an outer wrapper: a new <ifm|tool_call> opens a block, or
        # </ifm|tool_calls> exits the wrapper. Bare-block mode (no wrapper)
        # only looks for <ifm|tool_call>.
        markers: tuple[str, ...] = (self._CALL_OPEN,)
        if self._has_outer_wrapper:
            markers = markers + (self._OUTER_CLOSE,)
        return self._consume_to_marker(markers, after_wrapper=False)

    def _step_after_wrapper(self) -> bool:
        # After </ifm|tool_calls>: finditer matches every <ifm|tool_call>
        # regardless of wrapper, so keep looking for another outer wrapper
        # or a bare block. Drop any intervening text.
        return self._consume_to_marker(
            (self._OUTER_OPEN, self._CALL_OPEN), after_wrapper=True
        )

    def _consume_to_marker(
        self, markers: tuple[str, ...], after_wrapper: bool
    ) -> bool:
        """Shared marker-scan + phase transition for ``_BETWEEN`` /
        ``_AFTER_WRAPPER``. Drops intervening text (non-streaming reference
        only treats the pre-first-tool prefix as content)."""
        pos = self._first_marker(markers)
        if pos is None:
            hold = self._partial_marker_suffix_len(self._buffer, markers)
            safe_end = len(self._buffer) - hold
            if safe_end > 0:
                self._buffer = self._buffer[safe_end:]
            return False
        idx, marker = pos
        self._buffer = self._buffer[idx + len(marker) :]
        if marker == self._CALL_OPEN:
            self._enter_block()
        elif marker == self._OUTER_OPEN:
            self._has_outer_wrapper = True
            self._phase = self._BETWEEN
        else:
            # _OUTER_CLOSE, only reachable from _BETWEEN (after_wrapper=False).
            assert not after_wrapper
            self._phase = self._AFTER_WRAPPER
        return True

    def _enter_block(self) -> None:
        self._phase = self._IN_BLOCK
        self._block_buffer = ""
        self._saw_first_call = True
        self._block_start_id = (
            self._current_tool_id + 1 if self._current_tool_id >= 0 else 0
        )

    def _step_in_block(
        self,
        emissions: list,
        request: ChatCompletionRequest,
    ) -> bool:
        close_idx = self._buffer.find(self._CALL_CLOSE)
        if close_idx != -1:
            self._block_buffer += self._buffer[:close_idx]
            self._buffer = self._buffer[close_idx + len(self._CALL_CLOSE) :]
            self._finalize_block(emissions, request)
            self._phase = self._BETWEEN
            return True

        # No close yet — feed as much as is safely past any partial close
        # marker into the block buffer and re-run partial parsing.
        hold = self._partial_marker_suffix_len(self._buffer, (self._CALL_CLOSE,))
        safe_end = len(self._buffer) - hold
        if safe_end > 0:
            self._block_buffer += self._buffer[:safe_end]
            self._buffer = self._buffer[safe_end:]
            self._update_partial(emissions, request)
        return False

    def _finalize_block(
        self,
        emissions: list,
        request: ChatCompletionRequest,
    ) -> None:
        body = self._block_buffer.strip()
        self._block_buffer = ""
        if not body:
            return
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            # Match non-streaming behavior: silently skip malformed JSON.
            return
        items = parsed if isinstance(parsed, list) else [parsed]
        for i, raw_call in enumerate(items):
            target_id = self._block_start_id + i
            self._process_item(
                emissions, raw_call, target_id, is_complete=True, request=request
            )

    def _update_partial(
        self,
        emissions: list,
        request: ChatCompletionRequest,
    ) -> None:
        body = self._block_buffer.lstrip()
        if not body:
            return
        # Always exclude partial strings: with ``Allow.ALL`` a list body like
        # ``[{...}, {"name":"ba`` parses to two items where the second has a
        # truncated name ("ba"), which would let us advance to a new tool
        # with the wrong name. Excluding partial strings drops the in-flight
        # element entirely until its name closes, so the advance is safe.
        # Hermes can use the looser flag because each call lives inside its
        # own ``<tool_call>...</tool_call>`` markers — IFM JSON packs
        # multiple calls into one block, so we need the stricter rule.
        flags = Allow.ALL & ~Allow.STR
        try:
            parsed = partial_json_parser.loads(body, flags)
        except (
            partial_json_parser.core.exceptions.MalformedJSON,
            json.JSONDecodeError,
        ):
            return
        if isinstance(parsed, list):
            items = parsed
        elif isinstance(parsed, dict):
            items = [parsed]
        else:
            return
        if not items:
            return
        last_index = len(items) - 1
        for i, raw_call in enumerate(items):
            target_id = self._block_start_id + i
            # Items before the partial tail are JSON-complete because they
            # were followed by a comma; the last item is treated as partial.
            item_complete = i < last_index
            self._process_item(
                emissions,
                raw_call,
                target_id,
                is_complete=item_complete,
                request=request,
            )

    def _process_item(
        self,
        emissions: list,
        raw_call,
        target_id: int,
        is_complete: bool,
        request: ChatCompletionRequest,
    ) -> None:
        if not isinstance(raw_call, dict):
            return
        function = raw_call.get("function", raw_call)
        if not isinstance(function, dict):
            return
        name = function.get("name")
        cur_arguments = function.get("arguments")
        if isinstance(cur_arguments, str):
            try:
                cur_arguments = (
                    json.loads(cur_arguments) if cur_arguments.strip() else {}
                )
            except json.JSONDecodeError:
                cur_arguments = None
        if not isinstance(cur_arguments, dict):
            cur_arguments = {}

        if target_id < self._current_tool_id:
            # Already moved past this tool — its args were flushed when we
            # advanced past it. Nothing to do.
            return

        if target_id > self._current_tool_id:
            if not isinstance(name, str) or not name:
                # Can't start a tool until its name is fully parsed.
                return
            if self._current_tool_id >= 0:
                self._flush_remaining_args(emissions)
            self._current_tool_id = target_id
            self._current_tool_name_sent = False
            self.record_tool_call_start(target_id, name)

        if not self._current_tool_name_sent:
            if not isinstance(name, str) or not name:
                return
            emissions.append(
                _ToolNameEmission(
                    index=self._current_tool_id,
                    id=make_tool_call_id(),
                    name=name,
                )
            )
            self._current_tool_name_sent = True

        self._emit_args(emissions, name, cur_arguments, request, is_complete)

    def _emit_args(
        self,
        emissions: list,
        name: str | None,
        cur_arguments: dict,
        request: ChatCompletionRequest,
        is_complete: bool,
    ) -> None:
        # ``name`` is guaranteed to be a non-empty string in the streaming
        # path that reaches here, but ``_coerce_arguments`` accepts any
        # ``tool_name`` (it just won't find a schema for a non-string), so
        # we keep the looser type instead of inserting an ``assert``.
        try:
            coerced = self._parser._coerce_arguments(name, cur_arguments, request.tools)
        except Exception:
            coerced = cur_arguments

        cur_args_json = json.dumps(coerced, ensure_ascii=False)
        idx = self._current_tool_id
        sent = len(self.streamed_args_for_tool[idx])
        prev_arguments = self.prev_tool_call_arr[idx].get("arguments")

        if is_complete:
            argument_diff = cur_args_json[sent:]
        elif prev_arguments is not None:
            prev_args_json = json.dumps(prev_arguments, ensure_ascii=False)
            if cur_args_json == prev_args_json:
                argument_diff = ""
            else:
                prefix = find_common_prefix(prev_args_json, cur_args_json)
                argument_diff = prefix[sent:] if len(prefix) > sent else ""
        else:
            argument_diff = ""

        # Always update the recorded coerced dict so EOS sees the latest
        # best-effort args even before any diff has been emitted.
        self.record_args_final(idx, dict(coerced))
        if argument_diff:
            self.record_args_fragment(idx, argument_diff)
            emissions.append(_ToolArgsEmission(idx, argument_diff))

    def _flush_remaining_args(self, emissions: list) -> None:
        """When advancing past the current tool to a new one, emit any
        diff between what's been streamed and the recorded coerced JSON.
        Past tools have been forced complete by the comma separating them
        from the next list element, so it's safe to commit the tail now.
        """
        idx = self._current_tool_id
        if idx < 0:
            return
        arguments = self.prev_tool_call_arr[idx].get("arguments")
        if arguments is None:
            return
        args_json = json.dumps(arguments, ensure_ascii=False)
        sent = len(self.streamed_args_for_tool[idx])
        diff = args_json[sent:]
        if diff:
            self.record_args_fragment(idx, diff)
            emissions.append(_ToolArgsEmission(idx, diff))

    def _first_marker(self, markers: Sequence[str]) -> tuple[int, str] | None:
        best_idx = -1
        best_marker: str | None = None
        for marker in markers:
            idx = self._buffer.find(marker)
            if idx == -1:
                continue
            if best_idx == -1 or idx < best_idx:
                best_idx, best_marker = idx, marker
        if best_marker is None:
            return None
        return best_idx, best_marker

    def _build_delta(self, emissions: list) -> DeltaMessage | None:
        content_parts: list[str] = []
        tool_calls: dict[int, DeltaToolCall] = {}
        for em in emissions:
            if isinstance(em, _ContentEmission):
                content_parts.append(em.text)
            elif isinstance(em, _ToolNameEmission):
                existing = tool_calls.get(em.index)
                if existing is None:
                    tool_calls[em.index] = DeltaToolCall(
                        index=em.index,
                        id=em.id,
                        type="function",
                        function=DeltaFunctionCall(name=em.name),
                    )
                else:
                    # Name emission for an already-touched tool index would
                    # break the OpenAI streaming contract (name must appear
                    # exactly once). This shouldn't happen but guard anyway.
                    existing.id = em.id
                    if existing.function is not None:
                        existing.function.name = em.name
            else:  # _ToolArgsEmission
                existing = tool_calls.get(em.index)
                if existing is None:
                    tool_calls[em.index] = DeltaToolCall(
                        index=em.index,
                        function=DeltaFunctionCall(arguments=em.args),
                    )
                else:
                    assert existing.function is not None
                    existing.function.arguments = (
                        existing.function.arguments or ""
                    ) + em.args

        content = "".join(content_parts) if content_parts else None
        tools_list = [tool_calls[k] for k in sorted(tool_calls)] if tool_calls else None
        if tools_list:
            return DeltaMessage(content=content, tool_calls=tools_list)
        if content is not None:
            return DeltaMessage(content=content)
        return None
