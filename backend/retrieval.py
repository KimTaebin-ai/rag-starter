"""Retrieval pipeline over the in-memory CFR index.

    vector search → [threshold] → [rerank] → [neighbor expansion]
                  → [dedupe] → [group by document] → [cross-reference expansion]

Each stage is gated by a toggle in config.py so it can be measured on its own.
`retrieve()` is the single entry point; everything above it are its stages.
"""
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CONFIG
from indexer import embed, load_index, search
from tokens import TOKENS, client

# ── Index + lookup tables (built once at import) ─────────────────
# Fails fast if there is no index — run `python indexer.py` first.
INDEX = load_index()
# (source, chunk_index) → record, for O(1) neighbor lookup during expansion.
INDEX_BY_POS = {(r["source"], r["chunk_index"]): r for r in INDEX}


def _section_num(section: str | None) -> str | None:
    """'§ 61.109 Aeronautical experience' → '61.109' (None if no § number)."""
    m = re.match(r"§\s*(\d+\.\d+)", section or "")
    return m.group(1) if m else None


# section number (e.g. '61.107') → its chunks, ordered by chunk_index — lets
# cross-reference expansion fetch a referenced section without a vector search.
INDEX_BY_SECTION: dict[str, list[dict]] = {}
for _r in INDEX:
    _n = _section_num(_r.get("section"))
    if _n:
        INDEX_BY_SECTION.setdefault(_n, []).append(_r)
for _chunks in INDEX_BY_SECTION.values():
    _chunks.sort(key=lambda r: r["chunk_index"])

print(f"Loaded {len(INDEX)} chunks from disk")
# Warm up the embedding model so the FIRST query's [timing] reflects steady-state
# retrieval, not the one-time model graph warm-up (~4s otherwise).
embed(["warmup"])

# A § cross-reference inside a chunk, e.g. "§ 61.107(b)(1)" or "§ 61.110".
XREF_RE = re.compile(r"§\s*(\d+\.\d+)((?:\([a-z0-9]{1,3}\))*)")


# ════════════════════════════════════════════════════════════════
# Debug logging
# ════════════════════════════════════════════════════════════════

def log_search(stage: str, query: str, hits: list[dict]) -> None:
    if not CONFIG.debug_search:
        return
    print(f"[search] {stage} | query={query!r} | {len(hits)} hits")
    for i, h in enumerate(hits):
        print(
            f"    [{i + 1}] sim={h['similarity']:.3f} "
            f"src={h['source']} chunk#{h['chunk_index']} "
            f"title={h.get('title', '?')!r}"
        )


# ════════════════════════════════════════════════════════════════
# Stages
# ════════════════════════════════════════════════════════════════

def rerank(question: str, hits: list[dict]) -> list[dict]:
    """Keep the passages Claude judges actually needed (up to rerank_top_n).

    Sends numbered candidate snippets and asks for only the indices that
    contribute to the answer — fewer for a focused question, more for a broad
    one — which trims unused "on-topic but irrelevant" context. Falls back to
    the original (similarity) order on any error.
    """
    if len(hits) <= CONFIG.rerank_top_n:
        return hits
    listing = "\n\n".join(
        f"[{i + 1}] {h['text'][:CONFIG.rerank_passage_chars]}" for i, h in enumerate(hits)
    )
    try:
        resp = client.messages.create(
            model=CONFIG.claude_model,
            max_tokens=80,
            system=(
                "You select which candidate passages are actually needed to "
                "answer the question. Return ONLY the passages that contribute "
                f"to the answer — as FEW as possible, up to {CONFIG.rerank_top_n}, "
                "most relevant first. Omit passages that are merely on the same "
                "topic but not needed (a focused question may need just one). "
                "Comma-separated numbers only (e.g. '3, 1'). No other text."
            ),
            messages=[{
                "role": "user",
                "content": f"QUESTION:\n{question}\n\nPASSAGES:\n{listing}",
            }],
        )
        TOKENS.record(resp.usage, "rerank")
        order = [int(n) for n in re.findall(r"\d+", resp.content[0].text)]
        picked: list[dict] = []
        seen: set[int] = set()
        for n in order:
            if 1 <= n <= len(hits) and n not in seen:
                seen.add(n)
                picked.append(hits[n - 1])
            if len(picked) >= CONFIG.rerank_top_n:
                break
        return picked or hits[: CONFIG.rerank_top_n]
    except Exception as exc:  # noqa: BLE001
        print(f"[rerank] failed, keeping similarity order: {exc}")
        return hits[: CONFIG.rerank_top_n]


def expand_neighbors(hits: list[dict]) -> list[dict]:
    """Add adjacent same-section chunks so split paragraphs reunite.

    For the top-N hits (neighbor_expand_top), look up their chunk_index ± radius
    neighbors in the same source — truncation strikes the answer chunk, which
    reranking floats to the top, so expanding the long tail just bloats context.
    A neighbor is added only if it belongs to the SAME § section (so we complete
    the current provision, not bleed into the next one). Added chunks inherit
    their parent's similarity so the later grouping step slots them in by
    chunk_index, restoring reading order. Dedupe runs after this and removes any
    true overlap.
    """
    present = {(h["source"], h["chunk_index"]) for h in hits}
    additions: list[dict] = []
    # Expand only the top-N hits (0 = all); they're in relevance order here.
    to_expand = hits[: CONFIG.neighbor_expand_top] if CONFIG.neighbor_expand_top else hits
    for h in to_expand:
        for d in range(1, CONFIG.neighbor_radius + 1):
            for ci in (h["chunk_index"] - d, h["chunk_index"] + d):
                key = (h["source"], ci)
                if key in present:
                    continue
                rec = INDEX_BY_POS.get(key)
                if rec is None or rec.get("section") != h.get("section"):
                    continue
                present.add(key)
                neighbor = {k: v for k, v in rec.items() if k != "embedding"}
                neighbor["similarity"] = h["similarity"]
                additions.append(neighbor)
    return hits + additions


def dedupe_hits(hits: list[dict], thresh: float = 0.8) -> list[dict]:
    """Drop near-duplicate chunks.

    A chunk is a duplicate if its word set is >= `thresh` contained in an
    already-kept chunk (containment = |A∩B| / |smaller|). Keeps the first
    (higher-ranked) occurrence. Removes redundant/overlapping chunks so the
    context carries no repeated text.
    """
    kept: list[dict] = []
    kept_words: list[set] = []
    for h in hits:
        words = set(h["text"].lower().split())
        if not words:
            continue
        if any(len(words & kw) / min(len(words), len(kw)) >= thresh for kw in kept_words):
            continue
        kept.append(h)
        kept_words.append(words)
    return kept


def _lead_chunk_for(num: str, sub: str) -> dict | None:
    """Best chunk of section `num` for a reference, preferring its subsection.

    `sub` is the parenthesized tail, e.g. '(b)(1)'. Among the section's chunks,
    pick the earliest whose text contains the first subsection component (e.g.
    '(b)'); else the section's lead chunk. Returns an embedding-free copy tagged
    as a cross-reference.
    """
    chunks = INDEX_BY_SECTION.get(num)
    if not chunks:
        return None
    pick = chunks[0]
    first = re.match(r"\([a-z0-9]{1,3}\)", sub)
    if first:
        marker = first.group(0)
        pick = next((c for c in chunks if marker in c["text"]), pick)
    out = {k: v for k, v in pick.items() if k != "embedding"}
    out["similarity"] = 0.0
    out["xref"] = True
    return out


def expand_xrefs(hits: list[dict]) -> list[dict]:
    """Follow § cross-references in the retrieved chunks into citable sources.

    Scans the final hits' text for "§ X.Y(...)" references to OTHER sections and
    appends each referenced section's most relevant chunk (bounded by max_xref,
    no recursion) so a dangling in-text reference becomes a real, numbered source
    the answer can cite. Referenced-section lookup is by metadata, no vector search.
    """
    present = {_section_num(h.get("section")) for h in hits}
    additions: list[dict] = []
    seen: set[str] = set()
    for h in hits:
        for num, sub in XREF_RE.findall(h["text"]):
            if num in present or num in seen:
                continue
            seen.add(num)
            chunk = _lead_chunk_for(num, sub)
            if chunk is not None:
                additions.append(chunk)
                if len(additions) >= CONFIG.max_xref:
                    return hits + additions
    return hits + additions


# ════════════════════════════════════════════════════════════════
# Pipeline entry point
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
