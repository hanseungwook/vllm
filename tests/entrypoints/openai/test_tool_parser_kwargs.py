# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import types
from unittest.mock import MagicMock

import pytest

llguidance = types.ModuleType("llguidance")
llguidance.LLMatcher = object
llguidance.LLTokenizer = object
sys.modules.setdefault("llguidance", llguidance)

llguidance_hf = types.ModuleType("llguidance.hf")
llguidance_hf.from_tokenizer = lambda *args, **kwargs: None
sys.modules.setdefault("llguidance.hf", llguidance_hf)

llguidance_torch = types.ModuleType("llguidance.torch")
llguidance_torch.allocate_token_bitmask = lambda *args, **kwargs: None
llguidance_torch.fill_next_token_bitmask = lambda *args, **kwargs: None
sys.modules.setdefault("llguidance.torch", llguidance_torch)

diskcache = types.ModuleType("diskcache")
diskcache.Cache = object
sys.modules.setdefault("diskcache", diskcache)

xgrammar = types.ModuleType("xgrammar")
xgrammar.TokenizerInfo = object
xgrammar.VocabType = types.SimpleNamespace(RAW="RAW", BYTE_FALLBACK="BYTE_FALLBACK")
xgrammar.GrammarCompiler = object
xgrammar.StructuralTagItem = object
xgrammar.GrammarMatcher = object
xgrammar.__file__ = "/tmp/xgrammar_stub.py"


def _xgrammar_getattr(name: str):
    if name.startswith("__"):
        raise AttributeError(name)
    return object


xgrammar.__getattr__ = _xgrammar_getattr
sys.modules.setdefault("xgrammar", xgrammar)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
    FunctionCall,
    ToolCall,
)
from vllm.entrypoints.openai.engine.serving import OpenAIServing
from vllm.tool_parsers import ToolParser

pytestmark = pytest.mark.cpu_test


class KwargAwareToolParser(ToolParser):
    def __init__(self, tokenizer, tools=None, **kwargs):
        super().__init__(tokenizer, tools, **kwargs)
        self.tool_format = kwargs.get("chat_template_kwargs", {}).get("tool_format")

    def extract_tool_calls(self, model_output, request):
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    function=FunctionCall(
                        name=self.tool_format or "missing",
                        arguments="{}",
                    )
                )
            ],
            content=None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        return None


class LegacyToolParser(ToolParser):
    def __init__(self, tokenizer, tools=None):
        super().__init__(tokenizer, tools)

    def extract_tool_calls(self, model_output, request):
        return ExtractedToolCallInformation(
            tools_called=True,
            tool_calls=[
                ToolCall(
                    function=FunctionCall(
                        name="legacy",
                        arguments="{}",
                    )
                )
            ],
            content=None,
        )

    def extract_tool_calls_streaming(
        self,
        previous_text,
        current_text,
        delta_text,
        previous_token_ids,
        current_token_ids,
        delta_token_ids,
        request,
    ):
        return None


def make_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        model="test-model",
        messages=[],
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "noop",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                    },
                },
            }
        ],
    )


def test_parse_tool_calls_from_content_passes_chat_template_kwargs():
    request = make_request()

    function_calls, content = OpenAIServing._parse_tool_calls_from_content(
        request=request,
        tokenizer=MagicMock(),
        enable_auto_tools=True,
        tool_parser_cls=KwargAwareToolParser,
        content="<function_calls>noop()</function_calls>",
        chat_template_kwargs={"tool_format": "python"},
    )

    assert content is None
    assert function_calls is not None
    assert len(function_calls) == 1
    assert function_calls[0].name == "python"
    assert function_calls[0].arguments == "{}"
    assert function_calls[0].id is not None


def test_parse_tool_calls_from_content_keeps_legacy_parsers_compatible():
    request = make_request()

    function_calls, content = OpenAIServing._parse_tool_calls_from_content(
        request=request,
        tokenizer=MagicMock(),
        enable_auto_tools=True,
        tool_parser_cls=LegacyToolParser,
        content="<function_calls>noop()</function_calls>",
        chat_template_kwargs={"tool_format": "python"},
    )

    assert content is None
    assert function_calls is not None
    assert len(function_calls) == 1
    assert function_calls[0].name == "legacy"
    assert function_calls[0].arguments == "{}"
    assert function_calls[0].id is not None
