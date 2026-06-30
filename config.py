"""Central feature configuration for the RAG chatbot.

Every improvement from the spec is gated behind an on/off toggle or a tunable
value so each feature can be measured independently (change before/after with
the same question). All values come from environment variables (.env), with
conservative defaults.

Usage:
    from config import CONFIG
    if CONFIG.enable_query_rewrite: ...

Print the active config at startup with CONFIG.summary().
"""
import os
from dataclasses import dataclass, fields


def _flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return int(raw) if raw not in (None, "") else default


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return float(raw) if raw not in (None, "") else default


def _str(name: str, default: str) -> str:
    raw = os.getenv(name)
    return raw if raw not in (None, "") else default


@dataclass(frozen=True)
class Config:
    # ── Model ─────────────────────────────────────────────────────
    claude_model: str = _str("CLAUDE_MODEL", "claude-sonnet-4-6")
    max_tokens: int = _int("MAX_TOKENS", 1024)

    # ── Phase 0 — measurement ─────────────────────────────────────
    debug_search: bool = _flag("DEBUG_SEARCH", True)        # 0-2 retrieval debug logs
    track_tokens: bool = _flag("TRACK_TOKENS", True)        # 0-1 token usage tracking

    # ── Retrieval basics ──────────────────────────────────────────
    top_k: int = _int("TOP_K", 5)                           # chunks fed to the LLM

    # ── Phase 2-1 — similarity threshold filter ───────────────────
    enable_threshold: bool = _flag("ENABLE_THRESHOLD", True)
    # Cosine SIMILARITY in [-1, 1]; drop chunks below this. With this multilingual
    # model in-corpus questions score ~0.6–0.76 and clearly-unrelated questions
    # ~0.13–0.22, so 0.35 filters junk without touching real hits. Conservative
    # default — raise it cautiously while watching the [search] score logs.
    similarity_threshold: float = _float("SIMILARITY_THRESHOLD", 0.35)

    # ── Phase 2-2 — query rewrite (extra LLM call) ────────────────
    enable_query_rewrite: bool = _flag("ENABLE_QUERY_REWRITE", False)

    # ── Phase 3-1 — group/sort chunks by document ────────────────
    enable_chunk_grouping: bool = _flag("ENABLE_CHUNK_GROUPING", True)

    # ── Phase 3-2 — inject source info into context ──────────────
    enable_source_in_context: bool = _flag("ENABLE_SOURCE_IN_CONTEXT", True)

    # ── Phase 4-1 — re-ranking (fetch many, keep best N) ─────────
    enable_rerank: bool = _flag("ENABLE_RERANK", False)
    rerank_fetch_k: int = _int("RERANK_FETCH_K", 30)        # 1st-pass candidates
    rerank_top_n: int = _int("RERANK_TOP_N", 5)             # kept after rerank

    # ── Phase 4-2 — agentic iterative search (tool loop) ─────────
    enable_agentic_search: bool = _flag("ENABLE_AGENTIC_SEARCH", False)
    max_search_iters: int = _int("MAX_SEARCH_ITERS", 3)     # loop guard

    def summary(self) -> str:
        lines = ["Active RAG config:"]
        for f in fields(self):
            lines.append(f"  {f.name} = {getattr(self, f.name)}")
        return "\n".join(lines)


CONFIG = Config()
