"""Retrieval pipeline over the in-memory CFR index.

    vector search → [threshold] → [rerank] → [neighbor expansion]
                  → [dedupe] → [group by document] → [cross-reference expansion]

Each stage is gated by a toggle in config.py so it can be measured on its own.
`retrieve()` is the single entry point; `corpus_gate()` and `keyword_search()`
are the agentic loop's gate + lexical-recall helpers. The shared index tables
live in index_store.py; the individual stages live in retrieval_stages.py.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CONFIG
from indexer import search
from index_store import INDEX, log_search
from retrieval_stages import dedupe_hits, expand_neighbors, expand_xrefs, rerank


# ════════════════════════════════════════════════════════════════
# Loop helpers — gate + lexical recall
# ════════════════════════════════════════════════════════════════

def corpus_gate(search_query: str) -> bool:
    """Top-level answer gate for the agentic loop: does the corpus cover this?

    One vector search, no LLM. True when the best hit clears min_answer_similarity
    — off-topic questions ("what is my name?") top out well below and get a
    0-token no-info reply before the loop runs. This lives at the TOP of the loop
    (on the resolved question) rather than inside `retrieve()` per sub-query: the
    loop decomposes a broad question into facets, and a real facet whose best hit
    is just under the bar must NOT be silently dropped — only a question the
    corpus can't touch at all should be refused. No-op when the toggle is off.
    """
    if not CONFIG.enable_answer_gate:
        return True
    raw = search(search_query, INDEX, k=1)
    top = raw[0]["similarity"] if raw else 0.0
    if top < CONFIG.min_answer_similarity:
        log_search(
            f"corpus gate: top sim {top:.3f} < {CONFIG.min_answer_similarity} → no-info",
            search_query, [],
        )
        return False
    return True


def keyword_search(term: str) -> list[dict]:
    """Lexical substring search — a recall path for enumeration questions.

    Returns ONE representative chunk per matching § section, up to
    keyword_search_limit. Vector search ranks by overall meaning and misses the
    long tail of sections that merely mention a term ("certificate", "oxygen");
    this surfaces that tail so the model can enumerate which sections govern a
    topic. Sections are ranked by how OFTEN the term occurs across the section
    (density), NOT corpus order — otherwise incidental early mentions (§ 1.1
    definitions) would crowd out the substantive section (§ 91.211 supplemental
    oxygen). The representative is the section's densest chunk. Chunks are tagged
    similarity 0.0 (lexical, not scored) and carry every field a vector hit does,
    so they slot into the same pool/citation path.
    """
    t = term.lower().strip()
    if not t:
        return []
    sections: dict[str, dict] = {}  # section key → running tally + best chunk
    for r in INDEX:
        c = r["text"].lower().count(t)
        if c == 0:
            continue
        key = r.get("section") or r["source"]
        s = sections.get(key)
        if s is None:
            sections[key] = {"count": c, "best": r, "best_count": c,
                             "order": (r["source"], r["chunk_index"])}
        else:
            s["count"] += c
            if c > s["best_count"]:  # densest chunk represents the section
                s["best_count"] = c
                s["best"] = r
    # Most mentions first; corpus order breaks ties for stable output.
    ranked = sorted(sections.values(), key=lambda s: (-s["count"], s["order"]))
    out: list[dict] = []
    for s in ranked[: CONFIG.keyword_search_limit]:
        h = {k: v for k, v in s["best"].items() if k != "embedding"}
        h["similarity"] = 0.0
        out.append(h)
    log_search(f"keyword {term!r} ({len(sections)} sections, kept {len(out)})",
               term, out)
    return out


# ════════════════════════════════════════════════════════════════
# Pipeline entry point
# ════════════════════════════════════════════════════════════════

def retrieve(question: str, search_query: str, recall: bool = False) -> tuple[list[dict], list[dict]]:
    """Run the full retrieval pipeline.

    Returns (final_hits, raw_hits): the chunks that go into the context, and the
    raw top-k from the vector search (with scores) so the caller can show the
    user what was retrieved and which chunks the threshold dropped.

    recall=True is the agentic-loop variant: skip the answer gate (the loop gates
    once, up front, via corpus_gate) and skip LLM rerank (the loop's model is the
    reranker), keeping the top recall_k by raw similarity instead. This stops
    per-sub-query rerank/gate from dropping real facets of a broad question. The
    default (recall=False) is the unchanged precision path used by one-shot.
    """
    fetch_k = CONFIG.rerank_fetch_k if (CONFIG.enable_rerank or recall) else CONFIG.top_k
    raw_hits = search(search_query, INDEX, k=fetch_k)
    log_search("raw", search_query, raw_hits)
    hits = raw_hits

    # Answer gate: if even the best hit is weak, the corpus doesn't cover this
    # question. Return no hits NOW — before rerank/expansion/answer — so the
    # caller skips every LLM call and the request costs zero tokens. Skipped in
    # recall mode: the loop gates once up front (corpus_gate), and gating each
    # facet here would drop real parts of a broad/enumeration question.
    if CONFIG.enable_answer_gate and not recall:
        top_sim = raw_hits[0]["similarity"] if raw_hits else 0.0
        if top_sim < CONFIG.min_answer_similarity:
            log_search(
                f"answer gate: top sim {top_sim:.3f} < {CONFIG.min_answer_similarity} → no-info",
                search_query, [],
            )
            return [], raw_hits

    if CONFIG.enable_threshold:
        hits = [h for h in hits if h["similarity"] >= CONFIG.similarity_threshold]
        log_search(f"after threshold>={CONFIG.similarity_threshold}", search_query, hits)

    if recall:
        # No LLM rerank — the loop's model filters when it composes the answer.
        # Keep more than top_k so broad/enumeration facets aren't pruned to 5.
        hits = hits[: CONFIG.recall_k]
        log_search(f"after recall trim (top {CONFIG.recall_k})", search_query, hits)
    elif CONFIG.enable_rerank and hits:
        hits = rerank(question, hits)
        log_search("after rerank", search_query, hits)
    else:
        hits = hits[: CONFIG.top_k]

    # (C) In recall mode these two expansions are skipped by default: each loop
    # search would add neighbors + referenced sections on top of recall_k, and
    # the loop re-sends the whole pool every turn. The agentic model searches
    # again for adjacent/referenced detail instead. RECALL_EXPAND=1 restores them.
    expand = (not recall) or CONFIG.recall_expand

    if CONFIG.enable_neighbor_expansion and hits and expand:
        # A retrieved chunk can end mid-paragraph because its § section was split
        # across chunks; pull adjacent same-section chunks so the answer isn't
        # truncated at a chunk boundary.
        before = len(hits)
        hits = expand_neighbors(hits)
        if len(hits) != before:
            log_search(f"after neighbor expansion ({before}->{len(hits)})", search_query, hits)

    if CONFIG.enable_dedupe and hits:
        # Drop chunks whose text is ~80%+ contained in a higher-ranked one
        # (near-duplicates / heavy overlap) — pure token-waste removal.
        before = len(hits)
        hits = dedupe_hits(hits)
        if len(hits) != before:
            log_search(f"after dedupe ({before}->{len(hits)})", search_query, hits)

    if CONFIG.enable_chunk_grouping and hits:
        # Cluster chunks from the same document and order them by chunk_index,
        # keeping documents in order of their best hit.
        best_rank: dict[str, int] = {}
        for rank, h in enumerate(hits):
            best_rank.setdefault(h["source"], rank)
        hits = sorted(hits, key=lambda h: (best_rank[h["source"]], h["chunk_index"]))
        log_search("after grouping", search_query, hits)

    if CONFIG.enable_xref and hits and expand:
        # Follow § cross-references in the kept chunks and append the referenced
        # sections as citable sources (after grouping, so they take the higher
        # [n] numbers and the primary answer chunks stay [1], [2]…).
        before = len(hits)
        hits = expand_xrefs(hits)
        if len(hits) != before:
            log_search(f"after xref (+{len(hits) - before})", search_query, hits)

    return hits, raw_hits
