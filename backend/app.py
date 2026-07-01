"""Context Management RAG starter — chat backend (Flask entry point).

Retrieval-augmented chat over the 14 CFR (Federal Aviation Regulations) corpus.
This module is only HTTP glue; the work lives in:

    retrieval.py   vector search + the ranking/expansion pipeline
    generation.py  the grounded answer, query rewrite, and agentic loop
    citations.py   context block + citation / monitoring payloads
    tokens.py      shared Anthropic client + token-usage tracker

Every improvement is gated behind a toggle in config.py so it can be measured
independently (same question, feature off vs. on). Run: `python backend/app.py`.
"""
import json
import sys
import time
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from flask import Flask, Response, jsonify, request, send_from_directory
from flask_cors import CORS

from config import CONFIG
from citations import build_citations, build_retrieval, renumber_citations
from generation import (
    CLARIFY_PREFIX, NO_INFO_MESSAGE, agentic_chat, answer, answer_stream, rewrite_query,
)
from retrieval import retrieve

# Corpus PDFs, served read-only so citations can deep-link to a page.
DOCS_DIR = Path(__file__).resolve().parent.parent / "documents"

app = Flask(__name__)
CORS(app)
print(CONFIG.summary())


@app.route("/api/pdf/<path:filename>")
def pdf(filename):
    """Serve a source PDF inline so a citation can jump to its page (#page=N).

    send_from_directory confines access to DOCS_DIR (no path traversal).
    """
    return send_from_directory(DOCS_DIR, filename)


def _ms(seconds: float) -> int:
    return round(seconds * 1000)


def _history_messages(history) -> list[dict]:
    """Sanitize client-supplied chat history into Claude messages (capped).

    Keeps only the last max_history_messages turns of visible Q&A text. Past
    retrieved CONTEXT is deliberately NOT re-sent (token saver) — the prior
    answer text is enough for the model to resolve a follow-up's references.
    Trims any leading assistant turn so the sequence starts with 'user'.
    """
    if not isinstance(history, list):
        return []
    msgs: list[dict] = []
    for turn in history[-CONFIG.max_history_messages:]:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role")
        text = (turn.get("text") or turn.get("content") or "").strip()
        if role in ("user", "assistant") and text:
            msgs.append({"role": role, "content": text})
    while msgs and msgs[0]["role"] != "user":
        msgs.pop(0)
    return msgs


@app.route("/api/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    history = _history_messages(request.json.get("history"))
    t_start = time.perf_counter()

    # Agentic path: Claude drives search as a tool loop (Phase 4-2).
    if CONFIG.enable_agentic_search:
        reply, hits, usage = agentic_chat(user_message)
        reply, cited, uncited = renumber_citations(reply, hits)
        timing = {"total_ms": _ms(time.perf_counter() - t_start)}
        print(f"[timing] agentic total={timing['total_ms']}ms")
        return jsonify({
            "reply": reply,
            "citations": build_citations(reply, cited),
            "usage": usage,
            "retrieval": build_retrieval(cited + uncited, cited + uncited),
            "timing": timing,
        })

    # Rewrite the question into a better search query when explicitly enabled,
    # OR whenever there's history: a follow-up like "How about for Class B?" is
    # meaningless to vector search until it's resolved against the prior turns.
    if CONFIG.enable_query_rewrite or history:
        search_query = rewrite_query(user_message, history)
        print(f"[query_rewrite] {user_message!r} → {search_query!r}")
    else:
        search_query = user_message

    # Too broad/vague to search → ask the user to narrow it (no retrieval/answer).
    if search_query.startswith(CLARIFY_PREFIX):
        clarification = search_query[len(CLARIFY_PREFIX):].strip()
        print("[clarify] question too vague → asking to narrow down")
        return jsonify({
            "reply": clarification,
            "citations": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "retrieval": None,
            "timing": {"total_ms": _ms(time.perf_counter() - t_start)},
        })

    t_retrieval = time.perf_counter()
    hits, raw_hits = retrieve(user_message, search_query)
    retrieval_ms = _ms(time.perf_counter() - t_retrieval)

    # Nothing cleared the threshold → don't call the LLM, say so.
    if not hits:
        print("[search] no chunks passed the threshold → returning no-info response")
        retrieval = build_retrieval(hits, raw_hits)
        retrieval["search_query"] = search_query
        timing = {"retrieval_ms": retrieval_ms, "llm_ms": 0,
                  "total_ms": _ms(time.perf_counter() - t_start)}
        print(f"[timing] retrieval={timing['retrieval_ms']}ms llm=0ms "
              f"total={timing['total_ms']}ms (no-info)")
        return jsonify({
            "reply": NO_INFO_MESSAGE,
            "citations": [],
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "retrieval": retrieval,
            "timing": timing,
        })

    t_llm = time.perf_counter()
    reply, usage = answer(user_message, hits, history)
    llm_ms = _ms(time.perf_counter() - t_llm)

    # Renumber [n] markers contiguously ([1][2][4] → [1][2][3]) and keep the
    # citation list + monitoring panel aligned with the renumbered markers.
    reply, cited, uncited = renumber_citations(reply, hits)
    retrieval = build_retrieval(cited + uncited, raw_hits)
    retrieval["search_query"] = search_query

    timing = {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms,
              "total_ms": _ms(time.perf_counter() - t_start)}
    print(f"[timing] retrieval={retrieval_ms}ms llm={llm_ms}ms total={timing['total_ms']}ms")

    return jsonify({
        "reply": reply,
        "citations": build_citations(reply, cited),
        "usage": usage,
        "retrieval": retrieval,
        "timing": timing,
    })


def _sse(event: str, **data) -> str:
    """Encode one Server-Sent Event line: `data: {"event": ..., ...}`."""
    return f"data: {json.dumps({'event': event, **data}, ensure_ascii=False)}\n\n"


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """Same pipeline as /api/chat, but streams the answer token-by-token (SSE).

    Emits `delta` events (incremental answer text) followed by one `done` event
    carrying the canonical reply + citations/usage/retrieval/timing — the same
    payload /api/chat returns. The done.reply is the citation-renumbered text, so
    the client replaces the streamed text with it to stay consistent. /api/chat
    is kept for eval.py, which needs the full JSON in one response.
    """
    user_message = request.json["message"]
    history = _history_messages(request.json.get("history"))

    def generate():
        t_start = time.perf_counter()

        # Agentic path can't stream incrementally (it's a tool loop); compute it
        # fully, then emit the answer as a single delta so the UX is uniform.
        if CONFIG.enable_agentic_search:
            reply, hits, usage = agentic_chat(user_message)
            reply, cited, uncited = renumber_citations(reply, hits)
            yield _sse("delta", text=reply)
            yield _sse("done", reply=reply,
                       citations=build_citations(reply, cited), usage=usage,
                       retrieval=build_retrieval(cited + uncited, cited + uncited),
                       timing={"total_ms": _ms(time.perf_counter() - t_start)})
            return

        if CONFIG.enable_query_rewrite or history:
            search_query = rewrite_query(user_message, history)
            print(f"[query_rewrite] {user_message!r} → {search_query!r}")
        else:
            search_query = user_message

        # Too broad/vague to search → stream a clarification request instead.
        if search_query.startswith(CLARIFY_PREFIX):
            clarification = search_query[len(CLARIFY_PREFIX):].strip()
            print("[clarify] question too vague → asking to narrow down")
            yield _sse("delta", text=clarification)
            yield _sse("done", reply=clarification, citations=[],
                       usage={"input_tokens": 0, "output_tokens": 0},
                       retrieval=None,
                       timing={"total_ms": _ms(time.perf_counter() - t_start)})
            return

        t_retrieval = time.perf_counter()
        hits, raw_hits = retrieve(user_message, search_query)
        retrieval_ms = _ms(time.perf_counter() - t_retrieval)

        # Nothing cleared the threshold → no LLM call, stream the no-info reply.
        if not hits:
            print("[search] no chunks passed the threshold → returning no-info response")
            retrieval = build_retrieval(hits, raw_hits)
            retrieval["search_query"] = search_query
            timing = {"retrieval_ms": retrieval_ms, "llm_ms": 0,
                      "total_ms": _ms(time.perf_counter() - t_start)}
            yield _sse("delta", text=NO_INFO_MESSAGE)
            yield _sse("done", reply=NO_INFO_MESSAGE, citations=[],
                       usage={"input_tokens": 0, "output_tokens": 0},
                       retrieval=retrieval, timing=timing)
            return

        t_llm = time.perf_counter()
        full_text = ""
        usage = {"input_tokens": 0, "output_tokens": 0}
        for piece in answer_stream(user_message, hits, history):
            if isinstance(piece, dict):  # terminal {"type": "final", ...}
                full_text = piece["text"]
                usage = piece["usage"]
            else:
                yield _sse("delta", text=piece)
        llm_ms = _ms(time.perf_counter() - t_llm)

        reply, cited, uncited = renumber_citations(full_text, hits)
        retrieval = build_retrieval(cited + uncited, raw_hits)
        retrieval["search_query"] = search_query
        timing = {"retrieval_ms": retrieval_ms, "llm_ms": llm_ms,
                  "total_ms": _ms(time.perf_counter() - t_start)}
        print(f"[timing] retrieval={retrieval_ms}ms llm={llm_ms}ms total={timing['total_ms']}ms (stream)")
        yield _sse("done", reply=reply, citations=build_citations(reply, cited),
                   usage=usage, retrieval=retrieval, timing=timing)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(port=5001, debug=True, threaded=True)
