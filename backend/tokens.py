"""Process-wide runtime singletons shared by the retrieval and generation layers.

Holds the single Anthropic API client and the cumulative token-usage tracker.
Keeping them in the lowest-level module lets every other backend module import
them without creating an import cycle.
"""
import sys
from pathlib import Path

# Make the project root importable (config.py / indexer.py live one level up).
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from anthropic import Anthropic

from config import CONFIG  # importing config loads .env (ANTHROPIC_API_KEY + toggles)

client = Anthropic()


class TokenTracker:
    """Accumulates Claude token usage across the process lifetime and logs it."""

    def __init__(self) -> None:
        self.total_input = 0
        self.total_output = 0
        self.total_cache_read = 0
        self.total_cache_write = 0
        self.calls = 0

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
