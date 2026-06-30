"""Context Management RAG starter — indexer.

Walks documents/, chunks each file, embeds chunks, persists the index to disk
so the chat backend can load it without re-indexing.

TODO: implement chunk_text(). The embedding and storage code is provided so
you can focus on the structure.
"""
import pickle
import re
from pathlib import Path

from sentence_transformers import SentenceTransformer

# Multilingual (50+ languages), 384-dim — same model as the /embedding project.
# Lets the corpus and the queries be in different languages and still match.
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
INDEX_PATH = Path(__file__).parent / "index.pkl"
DOCS_DIR = Path(__file__).parent / "documents"


# ════════════════════════════════════════════════════════════════
# TODO — implement chunk_text
#
# Split `text` into overlapping chunks. A reasonable default:
#   - ~1000 characters per chunk
#   - ~100 characters of overlap
#   - try to break on paragraph boundaries (\n\n) when possible
#
# Return a list of non-empty strings.
# See the lecture slide on chunking for one working implementation.
# ════════════════════════════════════════════════════════════════

def chunk_text(text: str, target_chars: int = 1000, overlap_chars: int = 100) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries.

    Greedily packs paragraphs (split on blank lines) into chunks of up to
    ~target_chars. Each chunk carries ~overlap_chars of trailing context from
    the previous chunk so retrieval doesn't lose information across cut points.
    Paragraphs longer than target_chars are split by character window.
    """
    # Split on blank lines, keeping non-empty paragraphs.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Break up any single paragraph that exceeds the target size.
    pieces: list[str] = []
    for para in paragraphs:
        if len(para) <= target_chars:
            pieces.append(para)
        else:
            for start in range(0, len(para), target_chars):
                pieces.append(para[start:start + target_chars])

    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + 2 + len(piece) > target_chars:
            chunks.append(current)
            # Carry the tail of the finished chunk as overlap context.
            tail = current[-overlap_chars:] if overlap_chars > 0 else ""
            current = (tail + "\n\n" + piece) if tail else piece
        else:
            current = (current + "\n\n" + piece) if current else piece

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ════════════════════════════════════════════════════════════════
# Provided: embedding (sentence-transformers, no API key required)
# ════════════════════════════════════════════════════════════════

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        print(f"Loading embedding model ({MODEL_NAME})... (one-time download ~470MB)")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns unit-normalized 384-dim vectors."""
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return vectors.tolist()


# ════════════════════════════════════════════════════════════════
# Provided: build / save / load / search
# ════════════════════════════════════════════════════════════════

SUPPORTED_SUFFIXES = (".md", ".txt", ".pdf")


def extract_title(text: str, fallback: str) -> str:
    """Best-effort human title for a text/markdown doc: first markdown '# '
    heading, else the first non-empty line, else the filename fallback."""
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        return line.lstrip("#").strip() or fallback
    return fallback


def humanize_source(path: Path) -> str:
    """Human title for a CFR PDF derived from its filename.

    CFR-2025-title14-vol2-part61.pdf → '14 CFR Part 61'
    CFR-2025-title14-vol1.pdf        → '14 CFR Title 14 (Vol 1, Parts 1–59)'
    """
    name = path.stem
    m = re.search(r"part(\d+)", name, re.IGNORECASE)
    if m:
        return f"14 CFR Part {int(m.group(1))}"
    if re.search(r"title14", name, re.IGNORECASE):
        vol = re.search(r"vol(\d+)", name, re.IGNORECASE)
        suffix = f" (Vol {vol.group(1)})" if vol else ""
        return f"14 CFR Title 14{suffix}"
    return path.stem


# CFR section heading, e.g. "§ 61.109 Aeronautical experience." — a section
# number followed by a Title-case title and a period. Cross-references like
# "§ 61.107(b)(1)" (paren after the number) or "§ 61.110 of this part"
# (lowercase after the number) deliberately do NOT match.
SECTION_RE = re.compile(r"§\s*(\d+\.\d+)\s+([A-Z][^.§\n]{2,80}?)\.")


def clean_pdf_text(text: str) -> str:
    """Normalize a page of extracted CFR text so it embeds cleanly.

    PDF extraction leaves line-break hyphenation ("single-\\nengine"), hard
    newlines mid-sentence, running page headers, and docket citations. Those
    artifacts wreck the embedding (a chunk that literally contains the answer
    scored 0.39 vs 0.80 once cleaned), so strip them here.
    """
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]*\n[ \t]*", " ", text)            # newlines → spaces
    text = re.sub(r"\d{1,4}\s+14 CFR Ch\. I \([^)]*Edition\)(\s*§?\s*[\d.]+)?", " ", text)
    text = re.sub(r"\d{1,4}\s+Federal Aviation Administration, DOT(\s*§?\s*[\d.]+)?", " ", text)
    text = re.sub(r"\[Docket[^\]]*\]", " ", text)          # docket/amendment citations
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def read_pdf_chunks(path: "Path") -> list[tuple[str, int | None, str | None]]:
    """Chunk a CFR PDF by SECTION (not by page), returning (text, page, section).

    Page-by-page chunking blends the tail of one section with the head of the
    next, diluting the embedding. Splitting on § section boundaries keeps each
    chunk topically coherent, and we prepend the section heading to every chunk
    so even later chunks of a long section carry their anchor ("§ 61.109
    Aeronautical experience"). Falls back to plain chunking if no sections parse.
    """
    from pypdf import PdfReader  # imported lazily so md/txt-only runs don't need it

    # Build one cleaned text stream while tracking which page each offset is on.
    full = ""
    page_starts: list[tuple[int, int]] = []
    for pno, page in enumerate(PdfReader(str(path)).pages, start=1):
        cleaned = clean_pdf_text(page.extract_text() or "")
        if not cleaned:
            continue
        page_starts.append((len(full), pno))
        full += cleaned + " "

    def page_at(pos: int) -> int | None:
        pg = page_starts[0][1] if page_starts else None
        for off, pno in page_starts:
            if off <= pos:
                pg = pno
            else:
                break
        return pg

    matches = list(SECTION_RE.finditer(full))
    if not matches:  # fallback — no recognizable sections
        return [(c, page_at(0), None) for c in chunk_text(full)]

    out: list[tuple[str, int | None, str | None]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full)
        label = f"§ {m.group(1)} {m.group(2).strip()}"
        page = page_at(start)
        for chunk in chunk_text(full[start:end].strip()):
            anchored = chunk if chunk.startswith(f"§ {m.group(1)}") else f"{label} — {chunk}"
            out.append((anchored, page, label))
    return out


def build_index() -> list[dict]:
    """Walk DOCS_DIR, chunk each file (by section for PDFs), embed, return records."""
    records: list[dict] = []
    chunk_id = 0
    for path in sorted(DOCS_DIR.glob("*")):
        if path.is_dir() or path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        if path.suffix.lower() == ".pdf":
            title = humanize_source(path)
            chunked = read_pdf_chunks(path)
        else:
            text = path.read_text()
            title = extract_title(text, path.stem)
            chunked = [(c, None, None) for c in chunk_text(text)]
        if not chunked:
            continue

        vectors = embed([c for c, _, _ in chunked])
        for i, ((chunk, page, section), vec) in enumerate(zip(chunked, vectors)):
            records.append({
                "chunk_id": chunk_id,
                "source": path.name,
                "title": title,          # Phase 2-3 metadata
                "page": page,            # page number for PDFs, else None
                "section": section,      # CFR § for PDFs, else None
                "chunk_index": i,
                "text": chunk,
                "embedding": vec,
            })
            chunk_id += 1
        n_sections = len({s for _, _, s in chunked if s})
        print(f"  {path.name}: {len(chunked)} chunks across {n_sections} sections")
    return records


def save_index(records: list[dict]) -> None:
    with INDEX_PATH.open("wb") as f:
        pickle.dump(records, f)


def load_index() -> list[dict]:
    if not INDEX_PATH.exists():
        raise FileNotFoundError(
            f"No index found at {INDEX_PATH}. Run `python indexer.py` from the project root first."
        )
    with INDEX_PATH.open("rb") as f:
        return pickle.load(f)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    # Both vectors are unit-normalized, so cosine similarity == dot product.
    return sum(x * y for x, y in zip(a, b))


def cosine_distance(a: list[float], b: list[float]) -> float:
    return 1.0 - cosine_similarity(a, b)


def search(query: str, records: list[dict], k: int = 5) -> list[dict]:
    """Embed the query, return top-k records by cosine similarity.

    Each returned item is a shallow copy of the record WITHOUT the heavy
    `embedding`, plus a `similarity` field (cosine, higher = closer) so callers
    can threshold/log without recomputing. The `title` metadata is backfilled
    from the source filename for indexes built before titles were stored.
    """
    [query_vec] = embed([query])
    scored = [(cosine_similarity(r["embedding"], query_vec), r) for r in records]
    scored.sort(key=lambda x: x[0], reverse=True)
    hits: list[dict] = []
    for sim, r in scored[:k]:
        hit = {key: val for key, val in r.items() if key != "embedding"}
        hit["similarity"] = sim
        hit.setdefault("title", Path(r["source"]).stem)
        hit.setdefault("page", None)
        hit.setdefault("section", None)
        hits.append(hit)
    return hits


def main() -> None:
    print(f"Indexing documents from {DOCS_DIR}/")
    records = build_index()
    save_index(records)
    print(f"\n✓ Indexed {len(records)} chunks → {INDEX_PATH.name}")


if __name__ == "__main__":
    main()
