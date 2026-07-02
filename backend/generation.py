"""Backward-compatible facade for the answer-generation layer.

The implementation was split for volume into focused modules:

  - prompts.py    : system prompt, sentinel messages, preamble stripping
  - answering.py  : rewrite_query, answer, answer_stream (one-shot path)
  - agentic.py    : the search tool-loop (agentic_chat[_events]) + its tools

Existing callers import from here; new code may import the modules directly.
"""
from prompts import CLARIFY_PREFIX, NO_INFO_MESSAGE, SYSTEM_PROMPT, strip_preamble
from answering import answer, answer_stream, rewrite_query
from agentic import (
    KEYWORD_TOOL,
    SEARCH_TOOL,
    agentic_chat,
    agentic_chat_events,
)

__all__ = [
    "CLARIFY_PREFIX",
    "NO_INFO_MESSAGE",
    "SYSTEM_PROMPT",
    "strip_preamble",
    "rewrite_query",
    "answer",
    "answer_stream",
    "agentic_chat",
    "agentic_chat_events",
    "SEARCH_TOOL",
    "KEYWORD_TOOL",
]
