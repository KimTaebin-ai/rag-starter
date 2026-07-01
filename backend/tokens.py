"""Process-wide runtime singletons shared by the retrieval and generation layers.

Holds the single Anthropic API client and the cumulative token-usage tracker.
Keeping them in the lowest-level module lets every other backend module import
them without creating an import cycle.
"""
import sys
import threading
from pathlib import Path

# Make the project root importable (config.py / indexer.py live one level up).
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from anthropic import Anthropic

from config import CONFIG  # importing config loads .env (ANTHROPIC_API_KEY + toggles)

client = Anthropic()


def _zero_usage() -> dict:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read": 0, "cache_write": 0}


class TokenTracker:
    """Accumulates Claude token usage across the process lifetime and logs it.

    Also supports per-request accumulation via `start_request()`/`end_request()`:
    every `record()` on the same thread adds to the active request accumulator,
    so a route can report the TRUE token cost of a question — including the
    auxiliary LLM calls (query rewrite, rerank, every agentic turn) that the
    answer call's own usage leaves out. The accumulator is thread-local so
    concurrent requests (Flask threaded=True) don't cross-contaminate.
    """

    def __init__(self) -> None:
        self.total_input = 0
        self.total_output = 0
        self.total_cache_read = 0
        self.total_cache_write = 0
        self.calls = 0
        self._local = threading.local()

    def start_request(self) -> None:
        """Begin per-request accumulation on this thread (resets any prior one)."""
        self._local.acc = _zero_usage()

    def end_request(self) -> dict:
        """Return this thread's accumulated per-request usage and clear it."""
        acc = getattr(self._local, "acc", None)
        self._local.acc = None
        return acc if acc is not None else _zero_usage()

    def record(self, usage, label: str) -> dict:
        """Record one API call's usage, log it, and return the per-call counts.

        `input_tokens` from the API excludes cache reads/writes; we surface those
        separately (cache_read is billed ~0.1x, cache_write ~1.25x) so the effect
        of prompt caching on the agentic loop is visible in the logs and payload.
        """
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        self.total_input += in_tok
        self.total_output += out_tok
        self.total_cache_read += cache_read
        self.total_cache_write += cache_write
        self.calls += 1
        acc = getattr(self._local, "acc", None)
        if acc is not None:  # fold this call into the active per-request total
            acc["input_tokens"] += in_tok
            acc["output_tokens"] += out_tok
            acc["cache_read"] += cache_read
            acc["cache_write"] += cache_write
        if CONFIG.track_tokens:
            cache_note = ""
            if cache_read or cache_write:
                cache_note = f" cache_read={cache_read} cache_write={cache_write}"
            print(
                f"[tokens] {label}: input={in_tok} output={out_tok}{cache_note} | "
                f"cumulative input={self.total_input} output={self.total_output} "
                f"cache_read={self.total_cache_read} (over {self.calls} calls)"
            )
        return {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "cache_read": cache_read,
            "cache_write": cache_write,
        }


TOKENS = TokenTracker()
