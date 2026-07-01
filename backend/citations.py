"""Formatting layer: turn retrieved chunks into the numbered CONTEXT block the
model reads, the citation list the answer's [n] markers resolve to, and the
retrieval-monitoring payload the frontend renders. Pure functions — no LLM or
index access.
"""
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CONFIG


def source_label(h: dict) -> str:
    """Full provenance for the UI: '14 CFR Part 61, § 61.109 …, p.49'."""
    label = h.get("title", h["source"])
    if h.get("section"):
        label += f", {h['section']}"
    if h.get("page"):
        label += f", p.{h['page']}"
    return label


def context_label(h: dict) -> str:
    """Compact in-context provenance for the LLM.

    Just the section (e.g. '§ 91.155'), or the title if there's no section. The
    full title/page still reach the frontend via the citations payload, so
    there's no point spending ~15 tokens/chunk repeating them to the model.
    """
    return h.get("section") or h.get("title") or h["source"]


def build_context(hits: list[dict]) -> str:
    """Numbered context block, each chunk prefixed with its source."""
    blocks = []
    for i, h in enumerate(hits):
        head = f"[{i + 1}] ({context_label(h)})" if CONFIG.enable_source_in_context else f"[{i + 1}]"
        blocks.append(f"{head}\n{h['text']}")
    return "\n\n".join(blocks)


def renumber_citations(answer: str, hits: list[dict]) -> tuple[str, list[dict], list[dict]]:
    """Renumber the answer's [n] markers to be contiguous by first appearance.

    The model sees context chunks [1..N] but often cites only some (e.g. it
    names an xref source in prose without attaching [n]), leaving gaps like
    [1][2][4]. Remap the cited numbers to [1][2][3] so the reader sees a clean
    sequence, and return the sources in that new order plus the fed-but-uncited
    ones, so the citation list and the monitoring panel stay aligned with the
    renumbered markers.
    """
    used: list[int] = []
    for tok in re.findall(r"\[(\d+)\]", answer):
        n = int(tok)
        if 1 <= n <= len(hits) and n not in used:
            used.append(n)
    remap = {old: i + 1 for i, old in enumerate(used)}
    new_answer = re.sub(
        r"\[(\d+)\]",
        lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else m.group(0),
        answer,
    )
    cited = [hits[old - 1] for old in used]
    uncited = [h for i, h in enumerate(hits, start=1) if i not in remap]
    return new_answer, cited, uncited


def build_citations(answer: str, hits: list[dict]) -> list[dict]:
    """One citation entry per unique valid [n] used in the answer."""
    used = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    seen: set[int] = set()
    citations: list[dict] = []
    for n in used:
        if n in seen or n < 1 or n > len(hits):
            continue
        seen.add(n)
        h = hits[n - 1]
        citations.append({
            "n": n,
            "source": h["source"],
            "title": h.get("title", h["source"]),
            "section": h.get("section"),
            "page": h.get("page"),
            "chunk_index": h["chunk_index"],
            "xref": h.get("xref", False),
        })
    return citations


def build_retrieval(final_hits: list[dict], raw_hits: list[dict]) -> dict:
    """Monitoring payload for the frontend: what was retrieved and what was used.

    `chunks` mirrors the numbered context the LLM saw. `dropped` lists raw hits
    the threshold filtered out, so the user can see the score cut. Text is
    truncated to a preview.
    """
    used_ids = {h["chunk_id"] for h in final_hits}
    chunks = [{
        "n": i + 1,
        "source": h["source"],
        "title": h.get("title", h["source"]),
        "section": h.get("section"),
        "page": h.get("page"),
        "chunk_index": h["chunk_index"],
        "similarity": round(h["similarity"], 4),
        "xref": h.get("xref", False),
        "preview": h["text"][:240],
    } for i, h in enumerate(final_hits)]
    dropped = [{
        "source": h["source"],
        "title": h.get("title", h["source"]),
        "section": h.get("section"),
        "page": h.get("page"),
        "similarity": round(h["similarity"], 4),
    } for h in raw_hits if h["chunk_id"] not in used_ids]
    return {
        "threshold": CONFIG.similarity_threshold if CONFIG.enable_threshold else None,
        "chunks": chunks,
        "dropped": dropped,
    }
