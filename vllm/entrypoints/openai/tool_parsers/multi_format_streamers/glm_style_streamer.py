# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unified streaming tool-call parser for GLM-style XML grammars.

Handles three closely-related ``tool_format`` values:

  * ``glm``       -- ``<tool_call>NAME<arg_key>K</arg_key>
                     <arg_value>V</arg_value></tool_call>``.
                     No outer wrapper, no ``<arg_type>`` tag.
  * ``xml``       -- ``<ifm|tool_call>NAME<ifm|arg_key>K</ifm|arg_key>
                     <ifm|arg_value>V</ifm|arg_value></ifm|tool_call>``,
                     with an optional ``<ifm|tool_calls>...</ifm|tool_calls>``
                     wrapper.
  * ``xml_typed`` -- same as ``xml`` plus an optional
                     ``<ifm|arg_type>T</ifm|arg_type>`` between key and
                     value for type coercion when no JSON schema is on
                     the request.

The ``<ifm|arg_type>`` tag is always recognized for both ``xml`` and
``xml_typed`` (matches the non-streaming ``_IFM_ARG_REGEX`` which captures
the optional type group regardless of format). Only ``glm`` skips it,
because the ``<arg_type>`` token does not exist in the GLM grammar.

Ported from vLLM upstream's ``Glm4MoeModelToolParser`` (``khluu/glm5``
branch), generalized to (a) IFM's ``<ifm|...>`` marker namespace + the
outer wrapper / arg_type extensions, and (b) byte-exact ``json.dumps``-
formatted streaming output (``: ``/``, `` separators with spaces) the
rest of vLLM's tool-call test suite expects.

Schema-aware streaming policy (the key feature):

  * **String-typed args** stream **character-by-character** with JSON
    escape. Long string values (file contents, code, prose) no longer
    block on ``</arg_value>`` -- clients see characters arrive as fast
    as the model emits them.
  * **Non-string args** (int/bool/object/array) buffer until
    ``</arg_value>`` then emit one fragment with ``json.dumps`` of the
    coerced value. This matches what ``_coerce_argument_value`` produces
    and avoids streaming intermediate text that would parse-error.

The string-vs-non-string decision uses (in order):
  1. The schema in ``request.tools``.
  2. The ``<ifm|arg_type>`` tag if present (xml/xml_typed only).
  3. Buffer by default -- preserves non-streaming's deserialization
     (``"42"`` -> int 42, ``"true"`` -> bool True, etc.).

``prev_tool_call_arr`` and ``streamed_args_for_tool`` (aliased from the
owning parser instance) are kept in lockstep with the emitted args so
``serving_chat.py``'s end-of-stream flush stays a no-op even on
truncated streams. After char-by-char streaming, the mirrored args
reflect the partial string up to the last streamed character.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

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


class GLMStyleToolCallStreamer(BaseToolCallStreamer):
    """Streamer for GLM-style tool-call grammars with incremental
    string-value streaming. Handles ``tool_format`` in
    ``{"glm", "xml", "xml_typed"}``.
    """

    _OUTSIDE = "outside"
    _BETWEEN = "between"
    _NAME = "name"
    _KEY = "key"
    _AFTER_KEY = "after_key"
    _ARG_TYPE = "arg_type"
    _AFTER_ARG_TYPE = "after_arg_type"
    _VALUE_STRING = "value_string"
    _VALUE_BUFFER = "value_buffer"
    _AFTER_VALUE = "after_value"
    _SKIP_TO_CLOSE = "skip_to_close"

    _MAX_BUFFER_BYTES = 1 << 20

    def __init__(self, tool_format: str, parser_cls) -> None:
        super().__init__(tool_format, parser_cls)
        if tool_format == "glm":
            self._tool_call_start = "<tool_call>"
            self._tool_call_end = "</tool_call>"
            self._arg_key_start = "<arg_key>"
            self._arg_key_end = "</arg_key>"
            self._arg_value_start = "<arg_value>"
            self._arg_value_end = "</arg_value>"
            self._arg_type_start: str | None = None
            self._arg_type_end: str | None = None
            self._outer_start: str | None = None
            self._outer_end: str | None = None
        else:
            # xml / xml_typed -- identical streaming logic. The optional
            # <ifm|arg_type> tag is always recognized to match the
            # non-streaming _IFM_ARG_REGEX, which captures the type group
            # regardless of which IFM tool_format the request declared.
            self._tool_call_start = "<ifm|tool_call>"
            self._tool_call_end = "</ifm|tool_call>"
            self._arg_key_start = "<ifm|arg_key>"
            self._arg_key_end = "</ifm|arg_key>"
            self._arg_value_start = "<ifm|arg_value>"
            self._arg_value_end = "</ifm|arg_value>"
            self._arg_type_start = "<ifm|arg_type>"
            self._arg_type_end = "</ifm|arg_type>"
            self._outer_start = "<ifm|tool_calls>"
            self._outer_end = "</ifm|tool_calls>"
        self._reset_state()

    def _reset_state(self) -> None:
        self._buffer = ""
        self._phase = self._OUTSIDE
        self._has_outer_wrapper = False
        self._first_tool_seen = False
        self._current_tool_idx = -1
        self._current_tool_name = ""
        self._name_buffer = ""
        self._key_buffer = ""
        self._arg_type_buffer = ""
        self._current_key = ""
        self._current_arg_type: str | None = None
        # Non-string value collection (buffer until </arg_value>).
        self._value_buffer = ""
        # Raw chars already streamed in string mode -- the mirror's
        # ``current_args[key]`` tracks this, byte-for-byte.
        self._streamed_value_buffer = ""
        self._current_args: dict[str, Any] = {}
        self._tool_call_ids: list[str] = []
        self._args_started: list[bool] = []
        # Clear in place so MultiFormatToolParser's alias stays valid.
        self.prev_tool_call_arr.clear()
        self.streamed_args_for_tool.clear()

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _json_escape_string_content(s: str) -> str:
        """JSON-escape content that goes INSIDE a JSON string (between
        quotes), not including the surrounding quotes themselves."""
        if not s:
            return ""
        return json.dumps(s, ensure_ascii=False)[1:-1]

    @staticmethod
    def _first_marker(text: str, markers: Sequence[str]) -> tuple[int, str]:
        """Return (idx, marker) of the leftmost match, or (-1, "")."""
        best_idx = -1
        best_marker = ""
        for m in markers:
            idx = text.find(m)
            if idx == -1:
                continue
            if best_idx == -1 or idx < best_idx:
                best_idx, best_marker = idx, m
        return best_idx, best_marker

    @staticmethod
    def _safe_emit_end(text: str, markers: Sequence[str]) -> int:
        """Length of the prefix of ``text`` that is safe to commit
        without risking the unemitted tail being the start of a marker.
        Returns the longest matching prefix held back across all markers.
        """
        if not text:
            return 0
        max_partial = 0
        for m in markers:
            for k in range(min(len(m), len(text)), 0, -1):
                if text.endswith(m[:k]):
                    if k > max_partial:
                        max_partial = k
                    break
        return len(text) - max_partial

    def _is_string_arg(
        self,
        key: str,
        arg_type: str | None,
        request: ChatCompletionRequest,
    ) -> bool:
        """Match what ``_coerce_argument_value`` would classify: schema
        wins, then arg_type, then default-buffer (preserves the non-
        streaming deserialization fallback for un-typed values)."""
        target_type = (
            self._parser._schema_arg_type(self._current_tool_name, key, request.tools)
            or arg_type
        )
        return self._parser._arg_type_is_string(target_type)

    def _post_tool_phase(self) -> str:
        """Phase to enter after closing a tool: ``BETWEEN`` inside the
        outer wrapper, ``OUTSIDE`` otherwise."""
        return self._BETWEEN if self._has_outer_wrapper else self._OUTSIDE

    def _ensure_tool_state(self, idx: int) -> None:
        while len(self._tool_call_ids) <= idx:
            self._tool_call_ids.append(make_tool_call_id())
        while len(self._args_started) <= idx:
            self._args_started.append(False)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

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
            # Keep tail half to preserve any in-progress marker.
            self._buffer = self._buffer[-(self._MAX_BUFFER_BYTES // 2) :]

        emissions: list = []
        while self._step(request, emissions):
            pass
        return self._build_delta_message(emissions)

    def _step(self, request: ChatCompletionRequest, emissions: list) -> bool:
        p = self._phase
        if p == self._OUTSIDE:
            return self._step_outside(emissions)
        if p == self._BETWEEN:
            return self._step_between()
        if p == self._NAME:
            return self._step_name(emissions)
        if p == self._KEY:
            return self._step_key()
        if p == self._AFTER_KEY:
            return self._step_after_key(request, emissions)
        if p == self._ARG_TYPE:
            return self._step_arg_type()
        if p == self._AFTER_ARG_TYPE:
            return self._step_after_arg_type(request, emissions)
        if p == self._VALUE_STRING:
            return self._step_value_string(emissions)
        if p == self._VALUE_BUFFER:
            return self._step_value_buffer(request, emissions)
        if p == self._AFTER_VALUE:
            return self._step_after_value(emissions)
        if p == self._SKIP_TO_CLOSE:
            return self._step_skip_to_close()
        return False

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _step_outside(self, emissions: list) -> bool:
        markers = [self._tool_call_start]
        if self._outer_start:
            markers.append(self._outer_start)
        idx, marker = self._first_marker(self._buffer, markers)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, markers)
            if safe_end <= 0:
                return False
            text = self._buffer[:safe_end]
            self._buffer = self._buffer[safe_end:]
            if text and not self._first_tool_seen:
                emissions.append(_ContentEmission(text))
            return True

        text = self._buffer[:idx]
        self._buffer = self._buffer[idx + len(marker) :]
        if text and not self._first_tool_seen:
            emissions.append(_ContentEmission(text))

        if marker == self._tool_call_start:
            self._first_tool_seen = True
            self._begin_tool_call()
            self._phase = self._NAME
        else:
            self._first_tool_seen = True
            self._has_outer_wrapper = True
            self._phase = self._BETWEEN
        return True

    def _step_between(self) -> bool:
        markers: list[str] = [self._tool_call_start]
        if self._outer_end:
            markers.append(self._outer_end)
        idx, marker = self._first_marker(self._buffer, markers)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, markers)
            if safe_end <= 0:
                return False
            # Drop intervening content; only the pre-first-tool prefix
            # is treated as content (matches non-streaming).
            self._buffer = self._buffer[safe_end:]
            return True
        self._buffer = self._buffer[idx + len(marker) :]
        if marker == self._tool_call_start:
            self._begin_tool_call()
            self._phase = self._NAME
        else:
            self._has_outer_wrapper = False
            self._phase = self._OUTSIDE
        return True

    def _begin_tool_call(self) -> None:
        self._current_tool_idx += 1
        self._current_tool_name = ""
        self._name_buffer = ""
        self._key_buffer = ""
        self._arg_type_buffer = ""
        self._current_key = ""
        self._current_arg_type = None
        self._value_buffer = ""
        self._streamed_value_buffer = ""
        self._current_args = {}
        self._ensure_tool_state(self._current_tool_idx)

    def _step_name(self, emissions: list) -> bool:
        markers = [self._arg_key_start, self._tool_call_end]
        idx, marker = self._first_marker(self._buffer, markers)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, markers)
            if safe_end <= 0:
                return False
            self._name_buffer += self._buffer[:safe_end]
            self._buffer = self._buffer[safe_end:]
            return True

        self._name_buffer += self._buffer[:idx]
        name = self._name_buffer.strip()
        self._buffer = self._buffer[idx + len(marker) :]
        self._name_buffer = ""

        if not name:
            # Empty-name tool: revert idx (no slot consumed by the
            # mirror; the per-tool arrays keep their preallocated slot,
            # matching upstream GLM's behavior).
            self._current_tool_idx -= 1
            if marker == self._tool_call_end:
                self._phase = self._post_tool_phase()
            else:
                self._phase = self._SKIP_TO_CLOSE
            return True

        self._current_tool_name = name
        idx_tc = self._current_tool_idx
        self.record_tool_call_start(idx_tc, name)
        emissions.append(_ToolNameEmission(idx_tc, self._tool_call_ids[idx_tc], name))

        if marker == self._arg_key_start:
            self._key_buffer = ""
            self._phase = self._KEY
        else:
            # ``<tool_call>name</tool_call>`` -- no args.
            self._emit_args_close(idx_tc, emissions)
            self._phase = self._post_tool_phase()
        return True

    def _step_key(self) -> bool:
        idx = self._buffer.find(self._arg_key_end)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, [self._arg_key_end])
            if safe_end <= 0:
                return False
            self._key_buffer += self._buffer[:safe_end]
            self._buffer = self._buffer[safe_end:]
            return True
        self._key_buffer += self._buffer[:idx]
        self._current_key = self._key_buffer.strip()
        self._buffer = self._buffer[idx + len(self._arg_key_end) :]
        self._key_buffer = ""
        self._current_arg_type = None
        self._phase = self._AFTER_KEY
        return True

    def _step_after_key(
        self,
        request: ChatCompletionRequest,
        emissions: list,
    ) -> bool:
        markers: list[str] = []
        if self._arg_type_start:
            markers.append(self._arg_type_start)
        markers.append(self._arg_value_start)
        markers.append(self._tool_call_end)
        idx, marker = self._first_marker(self._buffer, markers)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, markers)
            if safe_end <= 0:
                return False
            self._buffer = self._buffer[safe_end:]
            return True
        self._buffer = self._buffer[idx + len(marker) :]
        if self._arg_type_start and marker == self._arg_type_start:
            self._arg_type_buffer = ""
            self._phase = self._ARG_TYPE
        elif marker == self._arg_value_start:
            self._enter_value_phase(request, emissions)
        else:
            # Orphan key (key without value) -- drop and close. Matches
            # the non-streaming regex which only captures arg_key when
            # paired with an arg_value.
            self._close_current_tool(emissions)
        return True

    def _step_arg_type(self) -> bool:
        assert self._arg_type_end is not None
        idx = self._buffer.find(self._arg_type_end)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, [self._arg_type_end])
            if safe_end <= 0:
                return False
            self._arg_type_buffer += self._buffer[:safe_end]
            self._buffer = self._buffer[safe_end:]
            return True
        self._arg_type_buffer += self._buffer[:idx]
        self._current_arg_type = self._arg_type_buffer.strip() or None
        self._buffer = self._buffer[idx + len(self._arg_type_end) :]
        self._arg_type_buffer = ""
        self._phase = self._AFTER_ARG_TYPE
        return True

    def _step_after_arg_type(
        self,
        request: ChatCompletionRequest,
        emissions: list,
    ) -> bool:
        markers = [self._arg_value_start, self._tool_call_end]
        idx, marker = self._first_marker(self._buffer, markers)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, markers)
            if safe_end <= 0:
                return False
            self._buffer = self._buffer[safe_end:]
            return True
        self._buffer = self._buffer[idx + len(marker) :]
        if marker == self._arg_value_start:
            self._enter_value_phase(request, emissions)
        else:
            self._close_current_tool(emissions)
        return True

    def _enter_value_phase(
        self,
        request: ChatCompletionRequest,
        emissions: list,
    ) -> None:
        self._value_buffer = ""
        self._streamed_value_buffer = ""
        if self._is_string_arg(self._current_key, self._current_arg_type, request):
            self._open_string_arg(emissions)
            self._phase = self._VALUE_STRING
        else:
            self._phase = self._VALUE_BUFFER

    def _open_string_arg(self, emissions: list) -> None:
        """Emit the opener fragment for a string arg: ``{"k": "`` for
        the first arg, ``, "k": "`` for subsequent. Seeds the mirror's
        dict with an empty string so an EOS landing here still produces
        a coherent ``{key: ""}`` partial state."""
        idx_tc = self._current_tool_idx
        key_json = json.dumps(self._current_key, ensure_ascii=False)
        if self._args_started[idx_tc]:
            fragment = ", " + key_json + ': "'
        else:
            fragment = "{" + key_json + ': "'
            self._args_started[idx_tc] = True
        self.record_args_fragment(idx_tc, fragment)
        self._current_args[self._current_key] = ""
        self.record_args_final(idx_tc, dict(self._current_args))
        emissions.append(_ToolArgsEmission(idx_tc, fragment))

    def _step_value_string(self, emissions: list) -> bool:
        idx_tc = self._current_tool_idx
        val_end_pos = self._buffer.find(self._arg_value_end)
        if val_end_pos != -1:
            raw_chunk = self._buffer[:val_end_pos]
            self._buffer = self._buffer[val_end_pos + len(self._arg_value_end) :]
            self._emit_string_chars(idx_tc, raw_chunk, emissions)
            # Close the JSON string with ``"``.
            self.record_args_fragment(idx_tc, '"')
            emissions.append(_ToolArgsEmission(idx_tc, '"'))
            self._current_key = ""
            self._current_arg_type = None
            self._streamed_value_buffer = ""
            self._phase = self._AFTER_VALUE
            return True
        # No close marker yet -- emit safe prefix.
        safe_end = self._safe_emit_end(self._buffer, [self._arg_value_end])
        if safe_end <= 0:
            return False
        raw_chunk = self._buffer[:safe_end]
        self._buffer = self._buffer[safe_end:]
        self._emit_string_chars(idx_tc, raw_chunk, emissions)
        return True

    def _emit_string_chars(
        self,
        idx_tc: int,
        raw_chunk: str,
        emissions: list,
    ) -> None:
        if not raw_chunk:
            return
        self._streamed_value_buffer += raw_chunk
        self._current_args[self._current_key] = self._streamed_value_buffer
        self.record_args_final(idx_tc, dict(self._current_args))
        escaped = self._json_escape_string_content(raw_chunk)
        if escaped:
            self.record_args_fragment(idx_tc, escaped)
            emissions.append(_ToolArgsEmission(idx_tc, escaped))

    def _step_value_buffer(
        self,
        request: ChatCompletionRequest,
        emissions: list,
    ) -> bool:
        idx = self._buffer.find(self._arg_value_end)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, [self._arg_value_end])
            if safe_end <= 0:
                return False
            self._value_buffer += self._buffer[:safe_end]
            self._buffer = self._buffer[safe_end:]
            return True
        self._value_buffer += self._buffer[:idx]
        value_text = self._value_buffer.strip()
        self._buffer = self._buffer[idx + len(self._arg_value_end) :]
        self._value_buffer = ""

        try:
            coerced = self._parser._coerce_argument_value(
                value_text,
                self._current_tool_name,
                self._current_key,
                request.tools,
                arg_type=self._current_arg_type,
                from_text=True,
            )
        except Exception:
            coerced = value_text

        idx_tc = self._current_tool_idx
        key_json = json.dumps(self._current_key, ensure_ascii=False)
        val_json = json.dumps(coerced, ensure_ascii=False)
        if self._args_started[idx_tc]:
            fragment = ", " + key_json + ": " + val_json
        else:
            fragment = "{" + key_json + ": " + val_json
            self._args_started[idx_tc] = True

        self.record_args_fragment(idx_tc, fragment)
        self._current_args[self._current_key] = coerced
        self.record_args_final(idx_tc, dict(self._current_args))
        emissions.append(_ToolArgsEmission(idx_tc, fragment))
        self._current_key = ""
        self._current_arg_type = None
        self._phase = self._AFTER_VALUE
        return True

    def _step_after_value(self, emissions: list) -> bool:
        markers = [self._arg_key_start, self._tool_call_end]
        idx, marker = self._first_marker(self._buffer, markers)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, markers)
            if safe_end <= 0:
                return False
            self._buffer = self._buffer[safe_end:]
            return True
        self._buffer = self._buffer[idx + len(marker) :]
        if marker == self._arg_key_start:
            self._key_buffer = ""
            self._phase = self._KEY
        else:
            self._close_current_tool(emissions)
        return True

    def _close_current_tool(self, emissions: list) -> None:
        idx_tc = self._current_tool_idx
        self._emit_args_close(idx_tc, emissions)
        self._phase = self._post_tool_phase()
        self._current_key = ""
        self._current_arg_type = None
        self._value_buffer = ""
        self._streamed_value_buffer = ""

    def _emit_args_close(self, idx_tc: int, emissions: list) -> None:
        if self._args_started[idx_tc]:
            fragment = "}"
        else:
            fragment = "{}"
            self._args_started[idx_tc] = True
        self.record_args_fragment(idx_tc, fragment)
        self.record_args_final(idx_tc, dict(self._current_args))
        emissions.append(_ToolArgsEmission(idx_tc, fragment))

    def _step_skip_to_close(self) -> bool:
        idx = self._buffer.find(self._tool_call_end)
        if idx == -1:
            safe_end = self._safe_emit_end(self._buffer, [self._tool_call_end])
            if safe_end <= 0:
                return False
            self._buffer = self._buffer[safe_end:]
            return True
        self._buffer = self._buffer[idx + len(self._tool_call_end) :]
        self._phase = self._post_tool_phase()
        return True

    # ------------------------------------------------------------------
    # Delta assembly
    # ------------------------------------------------------------------

    def _build_delta_message(self, emissions: list) -> DeltaMessage | None:
        if not emissions:
            return None
        content_parts: list[str] = []
        tool_calls: dict[int, DeltaToolCall] = {}
        for em in emissions:
            if isinstance(em, _ContentEmission):
                content_parts.append(em.text)
            elif isinstance(em, _ToolNameEmission):
                tool_calls[em.index] = DeltaToolCall(
                    index=em.index,
                    id=em.id,
                    type="function",
                    function=DeltaFunctionCall(name=em.name, arguments=""),
                )
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
        if content is None and not tools_list:
            return None
        if content is not None and tools_list:
            return DeltaMessage(content=content, tool_calls=tools_list)
        if content is not None:
            return DeltaMessage(content=content)
        return DeltaMessage(tool_calls=tools_list or [])
