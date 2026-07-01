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

from dotenv import load_dotenv

# Load .env BEFORE the toggles below are read — CONFIG is built at import time,
# so any module that imports config picks up .env-based overrides regardless of
# import order.
load_dotenv()


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

    # ── Answer gate — refuse to answer (0 LLM tokens) off-topic questions ─
    # similarity_threshold (above) decides which chunks to INCLUDE; this decides
    # whether to ANSWER AT ALL. In-corpus questions score ~0.65–0.78 on their top
    # hit; clearly off-topic ones ("what is my name?", "내 이름이 뭐야") top out at
    # ~0.27–0.36 and can squeak a single weak chunk past 0.35 — enough to fire an
    # answer call. Requiring the BEST hit to clear this higher bar short-circuits
    # such questions to the no-info reply BEFORE any LLM call. Sits in the wide
    # 0.36–0.65 gap, so it drops off-topic questions without touching real hits.
    enable_answer_gate: bool = _flag("ENABLE_ANSWER_GATE", True)
    min_answer_similarity: float = _float("MIN_ANSWER_SIMILARITY", 0.45)

    # ── Phase 2-2 — query rewrite (extra LLM call) ────────────────
    enable_query_rewrite: bool = _flag("ENABLE_QUERY_REWRITE", False)

    # ── Drop near-duplicate / overlapping chunks (token saver, idea ②) ─
    enable_dedupe: bool = _flag("ENABLE_DEDUPE", True)

    # ── Phase 3-1 — group/sort chunks by document ────────────────
    enable_chunk_grouping: bool = _flag("ENABLE_CHUNK_GROUPING", True)

    # ── Phase 3-2 — inject source info into context ──────────────
    enable_source_in_context: bool = _flag("ENABLE_SOURCE_IN_CONTEXT", True)

    # ── Phase 3-4 — cross-reference expansion (follow § refs) ────
    # FAA rules lean on other sections ("in the areas of operation listed in
    # § 61.107(b)(1)", "except as provided in § 61.110"). Fetch the referenced
    # section's chunk so it becomes a real, citable source — the corpus's heavy
    # cross-referencing means single retrieval often can't complete an answer.
    # Bounded by max_xref (and no recursion) to keep token cost in check.
    enable_xref: bool = _flag("ENABLE_XREF", True)
    max_xref: int = _int("MAX_XREF", 3)

    # ── Phase 3-3 — neighbor chunk expansion (stitch split paragraphs) ─
    # A long § section is split across several chunks, so a retrieved chunk can
    # end mid-paragraph ("...3 hours of flight training on the control and
    # maneuvering of an"). Pull the adjacent chunks of the SAME section in so the
    # answer isn't truncated at a chunk boundary. radius = chunks on each side.
    enable_neighbor_expansion: bool = _flag("ENABLE_NEIGHBOR_EXPANSION", True)
    # Default 2: long § sections span 3+ chunks (e.g. § 61.109(a) = 4632→4634),
    # so radius 1 truncates the paragraph tail ("...maneuvering of an"). 2 reaches
    # it. This is a production default, not just a .env override — a fresh checkout
    # must ground/complete answers without depending on a gitignored .env.
    neighbor_radius: int = _int("NEIGHBOR_RADIUS", 2)
    # Only expand the top-N most relevant hits — truncation hits the answer
    # chunk, which reranking floats to the top, so expanding all of top_k just
    # bloats context. 2 keeps a margin (answer at rank 1 or 2). 0 = expand all.
    neighbor_expand_top: int = _int("NEIGHBOR_EXPAND_TOP", 2)

    # ── Phase 4-1 — re-ranking (fetch many, keep best N) ─────────
    # Default ON: many § sections share a generic title ("Aeronautical
    # experience"), so raw top-k collapses onto one section and mis-ranks the
    # right chunk → misattributed citations. Rerank over a wider fetch fixes
    # that. Production default (not just .env) so a fresh checkout cites correctly.
    enable_rerank: bool = _flag("ENABLE_RERANK", True)
    rerank_fetch_k: int = _int("RERANK_FETCH_K", 20)        # 1st-pass candidates
    rerank_top_n: int = _int("RERANK_TOP_N", 5)             # kept after rerank
    # Chars per candidate shown to the reranker. The model only needs enough to
    # judge relevance — a CFR chunk's section heading + opening fits in ~200, so
    # 200 cuts the rerank call's input ~60% vs 400 with no ranking change.
    rerank_passage_chars: int = _int("RERANK_PASSAGE_CHARS", 200)

    # ── Phase 4-2 — agentic iterative search (tool loop) ─────────
    enable_agentic_search: bool = _flag("ENABLE_AGENTIC_SEARCH", False)
    max_search_iters: int = _int("MAX_SEARCH_ITERS", 3)     # loop guard

    # ── Multi-turn — conversation history for follow-ups ─────────
    # Prior turns (text only, no re-sent context) passed to the answer call so
    # follow-ups like "그럼 야간 요건은?" resolve. Capped to bound token cost.
    max_history_messages: int = _int("MAX_HISTORY_MESSAGES", 6)

    def summary(self) -> str:
        lines = ["Active RAG config:"]
        for f in fields(self):
            lines.append(f"  {f.name} = {getattr(self, f.name)}")
        return "\n".join(lines)


CONFIG = Config()
