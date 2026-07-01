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
    CLARIFY_PREFIX, NO_INFO_MESSAGE, agentic_chat, agentic_chat_events,
    answer, answer_stream, rewrite_query,
)
from retrieval import retrieve
from tokens import TOKENS

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
    # Accumulate EVERY LLM call this request makes (rewrite + rerank + answer, or
    # every agentic turn) so the reported usage is the question's true token cost,
    # not just the final answer call's.
    TOKENS.start_request()

    # Resolve a follow-up into a standalone query BEFORE dispatching, so BOTH the
    # agentic loop and one-shot retrieval get a self-contained question. A bare
    # follow-up like "그럼 야간 비행은?" is meaningless on its own — without this the
    # agentic path (which takes only the message) searched the fragment verbatim
    # and returned an empty answer. Runs only when there's history to resolve (or
    # rewrite is explicitly on); a first-turn question passes through unchanged.
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
            "usage": TOKENS.end_request(),  # the rewrite call still cost tokens
            "retrieval": None,
            "timing": {"total_ms": _ms(time.perf_counter() - t_start)},
        })

    # Agentic path: Claude drives search as a tool loop (Phase 4-2). Seeded with
    # the resolved standalone query so follow-ups keep their context.
    if CONFIG.enable_agentic_search:
        reply, hits, _ = agentic_chat(search_query)
        reply = reply.strip() or NO_INFO_MESSAGE  # never return a blank bubble
        reply, cited, uncited = renumber_citations(reply, hits)
        timing = {"total_ms": _ms(time.perf_counter() - t_start)}
        print(f"[timing] agentic total={timing['total_ms']}ms")
        return jsonify({
            "reply": reply,
            "citations": build_citations(reply, cited),
            "usage": TOKENS.end_request(),
            "retrieval": build_retrieval(cited + uncited, cited + uncited),
            "timing": timing,
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
            "usage": TOKENS.end_request(),  # rewrite (if any) still cost tokens
            "retrieval": retrieval,
            "timing": timing,
        })

    t_llm = time.perf_counter()
    # History was used only to resolve the search query (rewrite_query above);
    # the answer is grounded in THIS turn's retrieved context alone.
    reply, _ = answer(user_message, hits)
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
        "usage": TOKENS.end_request(),
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
        TOKENS.start_request()  # accumulate every LLM call this request makes

        # Resolve a follow-up into a standalone query BEFORE dispatching, so the
        # agentic loop and one-shot retrieval alike keep the follow-up's context
        # (a bare "그럼 야간 비행은?" is unanswerable on its own).
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
                       usage=TOKENS.end_request(),  # the rewrite call still cost tokens
                       retrieval=None,
                       timing={"total_ms": _ms(time.perf_counter() - t_start)})
            return

        # Agentic path: stream the search loop as `agent` events (live loop
        # monitoring) AND the answer token-by-token as `delta` events as the
        # model composes it inside the tool loop. `answer_reset` clears text the
        # model streamed on a turn that then decided to search instead. Seeded
        # with the resolved query so follow-ups keep their context.
        if CONFIG.enable_agentic_search:
            reply, hits = "", []
            for evt in agentic_chat_events(search_query):
                t = evt["type"]
                if t == "final":
                    reply, hits = evt["reply"], evt["pool"]
                elif t == "answer_delta":
                    yield _sse("delta", text=evt["text"])
                elif t == "answer_reset":
                    yield _sse("reset")
                else:
                    yield _sse("agent", **evt)
            # Streamed text carried the pool's raw [n]; the done.reply is the
            # citation-renumbered text the client swaps in (same as non-agentic).
            reply = reply.strip() or NO_INFO_MESSAGE  # never return a blank bubble
            reply, cited, uncited = renumber_citations(reply, hits)
            yield _sse("done", reply=reply,
                       citations=build_citations(reply, cited), usage=TOKENS.end_request(),
                       retrieval=build_retrieval(cited + uncited, cited + uncited),
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
                       usage=TOKENS.end_request(),  # rewrite (if any) still cost tokens
                       retrieval=retrieval, timing=timing)
            return

        t_llm = time.perf_counter()
        full_text = ""
        # History only shaped the search query; answer from this turn's context.
        for piece in answer_stream(user_message, hits):
            if isinstance(piece, dict):  # terminal {"type": "final", ...}
                full_text = piece["text"]
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
                   usage=TOKENS.end_request(), retrieval=retrieval, timing=timing)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


if __name__ == "__main__":
    app.run(port=5001, debug=True, threaded=True)
