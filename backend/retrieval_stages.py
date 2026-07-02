"""Retrieval pipeline stages — each gated by a config toggle and measurable on
its own. `retrieve()` (in retrieval.py) composes these; they read the shared
index tables from index_store.py.
"""
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CONFIG
from tokens import TOKENS, client
from index_store import INDEX_BY_POS, INDEX_BY_SECTION, XREF_RE, _section_num


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
