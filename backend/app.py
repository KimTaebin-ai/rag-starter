"""Context Management RAG starter — chat backend.

Retrieval-augmented chat over the Apollo corpus. Every improvement from the
spec is gated behind a toggle in config.py so each feature can be measured
independently (same question, feature off vs. on).

Pipeline per request (toggles in brackets):
    question
      → [query rewrite]        Phase 2-2  rewrite to a cleaner search query
      → vector search          (fetch top_k, or rerank_fetch_k if reranking)
      → [threshold filter]     Phase 2-1  drop low-similarity chunks
      → [re-rank]              Phase 4-1  LLM keeps the best N
      → [group by document]    Phase 3-1  cluster same-doc chunks in order
      → build numbered context Phase 3-2  prefix each chunk with its source
      → Claude answer + [n] citations
All calls record token usage (Phase 0-1) and retrieval is logged (Phase 0-2).

Set ENABLE_AGENTIC_SEARCH=1 to instead let Claude drive search as a tool loop
(Phase 4-2).
"""
import re
import sys
from pathlib import Path

# Make the parent directory importable so we can use indexer.py and config.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from anthropic import Anthropic
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS

from config import CONFIG
from indexer import load_index, search

load_dotenv()  # ANTHROPIC_API_KEY from .env

app = Flask(__name__)
CORS(app)
client = Anthropic()

# Load the index once at startup. Fails fast if no index — run `python indexer.py` first.
INDEX = load_index()
print(f"Loaded {len(INDEX)} chunks from disk")
print(CONFIG.summary())

NO_INFO_MESSAGE = "관련 정보를 찾지 못했습니다."


SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the \
sources provided in the CONTEXT block of each message.

Rules:
- Base every statement strictly on the provided context. Do not use outside knowledge, \
and do not guess or infer beyond what the sources state.
- Cite each factual claim with a bracketed source number, e.g. [1] or [2][3], using the \
numbers shown in the context. Place the citation immediately after the claim it supports.
- Only use citation numbers that appear in the context. Never invent a number.
- If the context does not contain enough information to answer, say so explicitly \
(e.g. "The provided sources don't contain an answer to that.") and do not fabricate one.
- If only part of the question is supported, answer that part and clearly state what the \
sources do not cover.
- You may answer in the language of the question. Format answers in Markdown \
(use tables, lists, headings, and code blocks where helpful)."""


# ════════════════════════════════════════════════════════════════
# Phase 0-1 — token usage tracking
# ════════════════════════════════════════════════════════════════

class TokenTracker:
    """Accumulates Claude token usage across the process lifetime and logs it."""

    def __init__(self) -> None:
        self.total_input = 0
        self.total_output = 0
        self.calls = 0

    def record(self, usage, label: str) -> dict:
        """Record one API call's usage, log it, and return the per-call counts."""
        in_tok = getattr(usage, "input_tokens", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or 0
        self.total_input += in_tok
        self.total_output += out_tok
        self.calls += 1
        if CONFIG.track_tokens:
            print(
                f"[tokens] {label}: input={in_tok} output={out_tok} | "
                f"cumulative input={self.total_input} output={self.total_output} "
                f"(over {self.calls} calls)"
            )
        return {"input_tokens": in_tok, "output_tokens": out_tok}


TOKENS = TokenTracker()


# ════════════════════════════════════════════════════════════════
# Phase 0-2 — retrieval debug logging
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
# Phase 2-2 — query rewrite
# ════════════════════════════════════════════════════════════════

def rewrite_query(question: str) -> str:
    """Ask Claude to turn the user's question into a clear search query.

    Resolves vague pronouns / context-dependent phrasing into explicit terms.
    Falls back to the original question on any error.
    """
    try:
        resp = client.messages.create(
            model=CONFIG.claude_model,
            max_tokens=120,
            system=(
                "You rewrite a user's question into a concise, explicit search "
                "query for a vector search over the U.S. Federal Aviation "
                "Regulations (14 CFR — pilot certification, medical, airspace, "
                "operating rules). Resolve vague pronouns and implicit context "
                "into explicit regulatory terms (e.g. part/section names, "
                "airspace classes). Output ONLY the rewritten query, no quotes, "
                "no explanation."
            ),
            messages=[{"role": "user", "content": question}],
        )
        TOKENS.record(resp.usage, "query_rewrite")
        rewritten = resp.content[0].text.strip()
        return rewritten or question
    except Exception as exc:  # noqa: BLE001 — never let rewrite break the request
        print(f"[query_rewrite] failed, using original question: {exc}")
        return question


# ════════════════════════════════════════════════════════════════
# Phase 4-1 — LLM re-ranking
# ════════════════════════════════════════════════════════════════

def rerank(question: str, hits: list[dict]) -> list[dict]:
    """Keep only the rerank_top_n chunks Claude judges most relevant.

    Sends numbered candidate snippets and asks for the best indices. Falls back
    to the original (similarity) order on any error.
    """
    if len(hits) <= CONFIG.rerank_top_n:
        return hits
    listing = "\n\n".join(
        f"[{i + 1}] {h['text'][:400]}" for i, h in enumerate(hits)
    )
    try:
        resp = client.messages.create(
            model=CONFIG.claude_model,
            max_tokens=80,
            system=(
                "You re-rank candidate passages for relevance to a question. "
                f"Return the numbers of the {CONFIG.rerank_top_n} most relevant "
                "passages, most relevant first, as a comma-separated list of "
                "numbers only (e.g. '3, 1, 7'). No other text."
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


# ════════════════════════════════════════════════════════════════
# Retrieval pipeline (threshold → rerank → group)
# ════════════════════════════════════════════════════════════════

def retrieve(question: str, search_query: str) -> list[dict]:
    """Run the full retrieval pipeline.

    Returns (final_hits, raw_hits): the chunks that go into the context, and the
    raw top-k from the vector search (with scores) so the caller can show the
    user what was retrieved and which chunks the threshold dropped.
    """
    fetch_k = CONFIG.rerank_fetch_k if CONFIG.enable_rerank else CONFIG.top_k
    raw_hits = search(search_query, INDEX, k=fetch_k)
    log_search("raw", search_query, raw_hits)
    hits = raw_hits

    if CONFIG.enable_threshold:
        kept = [h for h in hits if h["similarity"] >= CONFIG.similarity_threshold]
        log_search(f"after threshold>={CONFIG.similarity_threshold}", search_query, kept)
        hits = kept

    if CONFIG.enable_rerank and hits:
        hits = rerank(question, hits)
        log_search("after rerank", search_query, hits)
    else:
        hits = hits[: CONFIG.top_k]

    if CONFIG.enable_chunk_grouping and hits:
        # Phase 3-1: cluster chunks from the same document and order them by
        # chunk_index, while keeping documents in order of their best hit.
        best_rank: dict[str, int] = {}
        for rank, h in enumerate(hits):
            best_rank.setdefault(h["source"], rank)
        hits = sorted(hits, key=lambda h: (best_rank[h["source"]], h["chunk_index"]))
        log_search("after grouping", search_query, hits)

    return hits, raw_hits


def _source_label(h: dict) -> str:
    """e.g. '14 CFR Part 61, § 61.109 Aeronautical experience, p.49'."""
    label = h.get("title", h["source"])
    if h.get("section"):
        label += f", {h['section']}"
    if h.get("page"):
        label += f", p.{h['page']}"
    return label


def build_context(hits: list[dict]) -> str:
    """Phase 3-2: numbered context block, each chunk prefixed with its source."""
    blocks = []
    for i, h in enumerate(hits):
        head = f"[{i + 1}] (출처: {_source_label(h)})" if CONFIG.enable_source_in_context else f"[{i + 1}]"
        blocks.append(f"{head}\n{h['text']}")
    return "\n\n".join(blocks)


def _build_retrieval(final_hits: list[dict], raw_hits: list[dict]) -> dict:
    """Monitoring payload for the frontend: what was retrieved and what was used.

    `chunks` mirrors the numbered context the LLM saw (used=True). `dropped`
    lists raw hits that the threshold filtered out, so the user can see the
    score cut. Text is truncated to a preview.
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


def _build_citations(answer: str, hits: list[dict]) -> list[dict]:
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
        })
    return citations


# ════════════════════════════════════════════════════════════════
# Phase 4-2 — agentic iterative search (search as a tool)
# ════════════════════════════════════════════════════════════════

SEARCH_TOOL = {
    "name": "search_corpus",
    "description": (
        "Search the 14 CFR (Federal Aviation Regulations) corpus for passages "
        "relevant to a query. Returns numbered passages with their source/part "
        "and page. Call again with a refined query if results are insufficient."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
        },
        "required": ["query"],
    },
}


def agentic_chat(question: str) -> tuple[str, list[dict], dict]:
    """Let Claude call search repeatedly until it can answer (bounded loop).

    Chunks from every search accumulate into a single pool with stable [n]
    numbering so the final answer's citations map back to real sources.
    """
    pool: list[dict] = []          # accumulated chunks, stable global numbering
    seen_chunks: set[int] = set()  # chunk_id dedupe across searches
    usage_total = {"input_tokens": 0, "output_tokens": 0}

    def add_usage(call: dict) -> None:
        usage_total["input_tokens"] += call["input_tokens"]
        usage_total["output_tokens"] += call["output_tokens"]

    messages = [{
        "role": "user",
        "content": (
            f"QUESTION:\n{question}\n\nUse the search_corpus tool to gather "
            "evidence, then answer. Cite claims with [n] using the passage "
            "numbers shown in the tool results."
        ),
    }]

    for _ in range(CONFIG.max_search_iters):
        resp = client.messages.create(
            model=CONFIG.claude_model,
            max_tokens=CONFIG.max_tokens,
            system=SYSTEM_PROMPT,
            tools=[SEARCH_TOOL],
            messages=messages,
        )
        add_usage(TOKENS.record(resp.usage, "agentic_turn"))

        if resp.stop_reason != "tool_use":
            answer = "".join(b.text for b in resp.content if b.type == "text")
            return answer, pool, usage_total

        messages.append({"role": "assistant", "content": resp.content})
        tool_results = []
        for block in resp.content:
            if block.type != "tool_use" or block.name != "search_corpus":
                continue
            sub_query = block.input.get("query", question)
            # Add new (deduped) chunks to the pool, then show them with their
            # stable global numbers so citations stay consistent.
            new_hits = []
            sub_hits, _ = retrieve(question, sub_query)
            for h in sub_hits:
                if h["chunk_id"] in seen_chunks:
                    continue
                seen_chunks.add(h["chunk_id"])
                pool.append(h)
                new_hits.append(h)
            if new_hits:
                start = len(pool) - len(new_hits)
                listing = "\n\n".join(
                    f"[{start + i + 1}] (출처: {_source_label(h)})\n{h['text']}"
                    for i, h in enumerate(new_hits)
                )
            else:
                listing = "(no new passages found)"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": listing,
            })
        messages.append({"role": "user", "content": tool_results})

    # Hit the iteration cap. The conversation ends on a tool_result (a user
    # turn), so call once more WITHOUT tools — the model must now answer using
    # whatever it has gathered. (Don't append another user message; that would
    # be two user turns in a row, which the API rejects.)
    resp = client.messages.create(
        model=CONFIG.claude_model,
        max_tokens=CONFIG.max_tokens,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    add_usage(TOKENS.record(resp.usage, "agentic_final"))
    answer = "".join(b.text for b in resp.content if b.type == "text")
    return answer, pool, usage_total


# ════════════════════════════════════════════════════════════════
# Route
# ════════════════════════════════════════════════════════════════

@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]

    if CONFIG.enable_agentic_search:
        answer, hits, usage = agentic_chat(user_message)
        citations = _build_citations(answer, hits)
        return jsonify({
            "reply": answer,
            "citations": citations,
            "usage": usage,
            "retrieval": _build_retrieval(hits, hits),
        })

    # Phase 2-2: optionally rewrite the question into a better search query.
    if CONFIG.enable_query_rewrite:
        search_query = rewrite_query(user_message)
        print(f"[query_rewrite] {user_message!r} → {search_query!r}")
    else:
        search_query = user_message

    hits, raw_hits = retrieve(user_message, search_query)
    retrieval = _build_retrieval(hits, raw_hits)
    retrieval["search_query"] = search_query

    # Phase 2-1: nothing cleared the threshold → don't call the LLM, say so.
    if not hits:
        print("[search] no chunks passed the threshold → returning no-info response")
        return jsonify({
            "reply": NO_INFO_MESSAGE,
            "citations": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "retrieval": retrieval,
        })

    context = build_context(hits)
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{user_message}"

    resp = client.messages.create(
        model=CONFIG.claude_model,
        max_tokens=CONFIG.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    answer = resp.content[0].text
    usage = TOKENS.record(resp.usage, "answer")

    citations = _build_citations(answer, hits)
    return jsonify({
        "reply": answer,
        "citations": citations,
        "usage": usage,
        "retrieval": retrieval,
    })


if __name__ == "__main__":
    app.run(port=5001, debug=True)
