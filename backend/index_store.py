"""In-memory CFR index + shared lookup tables, loaded once at import.

Both the retrieval pipeline (retrieval.py) and its stages (retrieval_stages.py)
read this shared state, so it lives in its own module to avoid a circular import
between them. Importing this module loads the index from disk and warms up the
embedding model.
"""
import re
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CONFIG
from indexer import embed, load_index

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
