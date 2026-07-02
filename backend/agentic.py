"""Agentic iterative search — Claude drives search as a tool loop, accumulating a
single citable pool, then composes the grounded answer.

  - agentic_chat_events : streaming generator of loop progress + result events
  - agentic_chat        : non-streaming wrapper (drains the loop, returns result)

Shared prompt text lives in prompts.py; the one-shot answer path in answering.py;
retrieval + its loop helpers (corpus_gate, keyword_search) in retrieval.py.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CONFIG
from citations import source_label
from tokens import TOKENS, client
from prompts import NO_INFO_MESSAGE, SYSTEM_PROMPT, strip_preamble
from answering import answer_stream
from retrieval import corpus_gate, keyword_search, retrieve


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

# Lexical recall tool for enumeration questions. Vector search ranks by overall
# meaning and misses the long tail of sections that merely mention a term; this
# returns one representative passage per § section whose text contains the exact
# term, so the model can enumerate which regulations govern a topic.
KEYWORD_TOOL = {
    "name": "keyword_search",
    "description": (
        "Find every 14 CFR section whose text contains an EXACT word or phrase, "
        "returning one representative passage per matching section. Use this for "
        "enumeration/coverage questions — 'which rules require a certificate', "
        "'list the regulations governing oxygen' — where you must find ALL "
        "sections mentioning a term, not just the most semantically similar ones. "
        "Search a single concrete term (e.g. 'endorsement', 'oxygen', 'ATC "
        "clearance'); call again with other terms to widen coverage. For a "
        "specific factual question, prefer search_corpus instead."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "term": {"type": "string",
                     "description": "Exact word or phrase to match, e.g. 'endorsement'."},
        },
        "required": ["term"],
    },
}


def agentic_chat_events(question: str):
    """Run the bounded search loop as a generator of progress events.

    Yields dicts describing each step as it happens so the caller can stream the
    loop to the UI (monitoring); the final dict is
    {"type": "final", "reply", "pool", "usage"} carrying the result. Event types:

      - {"type": "think",  "iter"}                       — calling the model
      - {"type": "search", "iter", "query", "found", "total"} — a search ran
      - {"type": "answer", "iter", "total", "forced"}    — composing the answer

    Chunks from every search accumulate into a single pool with stable [n]
    numbering so the final answer's citations map back to real sources.
    """
    pool: list[dict] = []          # accumulated chunks, stable global numbering
    seen_chunks: set[int] = set()  # chunk_id dedupe across searches
    usage_total = {"input_tokens": 0, "output_tokens": 0,
                   "cache_read": 0, "cache_write": 0}

    def add_usage(call: dict) -> None:
        for k in usage_total:
            usage_total[k] += call.get(k, 0)

    # Top-level answer gate: refuse a question the corpus can't touch at all
    # (off-topic / off-domain) BEFORE any LLM call — 0 tokens. One cheap vector
    # search. Unlike the old per-sub-query gate this fires once, on the whole
    # question, so it never drops a real facet of a broad question mid-loop.
    if not corpus_gate(question):
        yield {"type": "answer", "iter": 0, "total": 0, "forced": False}
        yield {"type": "final", "reply": NO_INFO_MESSAGE, "pool": [], "usage": usage_total}
        return

    # search_corpus (vector, semantic) is always available; keyword_search
    # (lexical, recall) is added for enumeration questions when enabled.
    tools = [SEARCH_TOOL]
    if CONFIG.enable_keyword_search:
        tools.append(KEYWORD_TOOL)

    # Prompt caching: mark the system block (which caches the tools+system prefix)
    # and move a single breakpoint to the end of the conversation each turn, so
    # the growing prefix is written once and read back on the next turn (~0.1x)
    # instead of re-billed in full. No-op when the toggle is off.
    use_cache = CONFIG.enable_prompt_cache
    system_param = (
        [{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}]
        if use_cache else SYSTEM_PROMPT
    )
    last_cached: dict | None = None

    def cache_conversation(blocks: list[dict]) -> None:
        nonlocal last_cached
        if not use_cache or not blocks:
            return
        if last_cached is not None:
            last_cached.pop("cache_control", None)  # only the newest prefix is marked
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
        last_cached = blocks[-1]

    kw_hint = (
        "\n- For an ENUMERATION/coverage question ('which rules require X', 'list "
        "the regulations governing Y'), the answer is spread across many sections "
        "that share a term but not overall meaning, so semantic search alone will "
        "miss some. Use keyword_search on the key term(s) to find EVERY matching "
        "section, then search_corpus to pull detail on the relevant ones."
        if CONFIG.enable_keyword_search else ""
    )
    messages = [{
        "role": "user",
        "content": (
            f"QUESTION:\n{question}\n\nUse the search tools to gather evidence, "
            "then answer. Cite claims with [n] using the passage numbers shown in "
            "the tool results.\n"
            "- If the question is broad or spans multiple subjects, roles, parts, "
            "or requirements (e.g. 'summarize X for various roles', 'compare A "
            "and B'), do NOT answer from a single broad search — decompose it "
            "and run one targeted search per subject/role/facet, gathering "
            "evidence for each. One broad query returns shallow, scattered hits; "
            "several focused queries return the specific passages each part "
            "needs.\n"
            "- For a 'for various/each/all <category>' question (e.g. '...for "
            "various roles'), FIRST establish the full set the question ranges "
            "over — do NOT assume it from prior knowledge. Discover which distinct "
            "members the CORPUS actually defines: keyword_search the category term "
            "and its variants (for roles: 'crewmember', 'instructor', 'flight "
            "attendant', 'check pilot', 'maintenance') and read the section TITLES "
            "returned to enumerate them. THEN cover EVERY distinct member you "
            "find, explicitly including the less-prominent, non-pilot ones (flight "
            "attendants, check airmen, instructors, maintenance personnel). This "
            "corpus is pilot-heavy, so a few broad searches surface mostly pilot "
            "rules and SILENTLY MISS the other roles — the discovery pass is what "
            "prevents a pilot-only answer to a 'various roles' question.\n"
            "- The corpus uses precise regulatory terms, not everyday phrasing "
            "(e.g. 'continuing education' appears as flight review / recurrent "
            "training / recent flight experience). If a search returns little, "
            "reformulate with the corpus's own terminology and try again before "
            "concluding a topic is absent."
            f"{kw_hint}\n"
            "Answer only once every part the question asks about is either "
            "supported by a retrieved passage or confirmed absent from the corpus "
            "— then say which parts the corpus does not cover.\n"
            "- When you answer, output ONLY the final answer — no preamble, "
            "meta-commentary, or narration of your process (never open with "
            "phrases like 'Now I have…' or 'Let me synthesize…')."
        ),
    }]

    for i in range(CONFIG.max_search_iters):
        yield {"type": "think", "iter": i + 1}
        # Stream the turn: text deltas flow to the UI live. We don't yet know if
        # this turn answers or calls a tool, so we stream optimistically — if it
        # turns out to be a tool turn (rare preamble text), we tell the UI to
        # discard what streamed and continue the loop.
        text_streamed = False
        with client.messages.stream(
            model=CONFIG.claude_model,
            max_tokens=CONFIG.max_tokens,
            system=system_param,
            tools=tools,
            messages=messages,
        ) as stream:
            for piece in stream.text_stream:
                text_streamed = True
                yield {"type": "answer_delta", "text": piece}
            resp = stream.get_final_message()
        add_usage(TOKENS.record(resp.usage, "agentic_turn"))

        if resp.stop_reason != "tool_use":
            yield {"type": "answer", "iter": i + 1, "total": len(pool), "forced": False}
            reply = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
            yield {"type": "final", "reply": strip_preamble(reply),
                   "pool": pool, "usage": usage_total}
            return

        if text_streamed:  # streamed preamble was not the answer — clear it
            yield {"type": "answer_reset"}
        messages.append({"role": "assistant", "content": resp.content})

        def add_to_pool(candidates: list[dict],
                        snippet_chars: int | None = None) -> tuple[list[dict], str]:
            """Add new (deduped) chunks to the pool and render them with their
            stable global [n] numbers so citations stay consistent across tools.

            (B) Stops at CONFIG.max_pool_chunks — the loop re-sends the whole pool
            every turn, so the pool is capped and, once full, the model is told to
            answer from what it has. (A) snippet_chars truncates the rendered text
            (used for keyword_search discovery hits); the FULL chunk is still
            stored in the pool, so citations and the forced final answer are intact.
            """
            new_hits = []
            capped = False
            for h in candidates:
                if len(pool) >= CONFIG.max_pool_chunks:
                    capped = True
                    break
                if h["chunk_id"] in seen_chunks:
                    continue
                seen_chunks.add(h["chunk_id"])
                pool.append(h)
                new_hits.append(h)
            if not new_hits:
                if capped or len(pool) >= CONFIG.max_pool_chunks:
                    return new_hits, ("(evidence pool is full — answer from the "
                                      "passages already gathered)")
                return new_hits, "(no new passages found)"

            def render(h: dict) -> str:
                t = h["text"]
                if snippet_chars is not None and len(t) > snippet_chars:
                    t = t[:snippet_chars].rstrip() + " …"
                return t

            start = len(pool) - len(new_hits)
            listing = "\n\n".join(
                f"[{start + j + 1}] (출처: {source_label(h)})\n{render(h)}"
                for j, h in enumerate(new_hits)
            )
            return new_hits, listing

        tool_results = []
        for block in resp.content:
            if block.type != "tool_use":
                continue
            # Every tool_use MUST get a tool_result back or the next request 400s,
            # so route by name and fall through to an error result for anything
            # unrecognized rather than dropping it.
            if block.name == "search_corpus":
                sub_query = block.input.get("query", question)
                sub_hits, _ = retrieve(question, sub_query, recall=True)
                new_hits, listing = add_to_pool(sub_hits)
                yield {"type": "search", "iter": i + 1, "query": sub_query,
                       "found": len(new_hits), "total": len(pool)}
            elif block.name == "keyword_search":
                term = block.input.get("term", "")
                new_hits, listing = add_to_pool(
                    keyword_search(term), snippet_chars=CONFIG.keyword_snippet_chars)
                yield {"type": "search", "iter": i + 1, "query": f"keyword:{term}",
                       "found": len(new_hits), "total": len(pool)}
            else:
                listing = f"(unknown tool: {block.name})"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": listing,
            })
        cache_conversation(tool_results)  # cache the prefix up to this turn
        messages.append({"role": "user", "content": tool_results})

    # Hit the iteration cap with no answer yet — the model spent every iteration
    # searching (common for a broad, multi-part question). The whole pool is
    # already sitting in `messages` (as tool_results) behind the cache breakpoint,
    # so CONTINUE that cached conversation for the final answer instead of
    # re-sending the pool uncached: a fresh answer_stream(question, pool) re-bills
    # the entire pool as new input (~12k tokens on a broad question), whereas one
    # more no-tools turn reads it from cache (~0.1x). We drop tools and add an
    # explicit "stop searching, answer now" instruction so the tool-saturated
    # context doesn't emit yet another tool_use / a near-empty reply; if it still
    # comes back (near-)empty, fall back to the reliable one-shot path.
    yield {"type": "answer", "iter": CONFIG.max_search_iters, "total": len(pool),
           "forced": True}
    reply = ""
    if pool:
        messages.append({"role": "user", "content": (
            "Stop searching. Using ONLY the numbered passages already provided "
            "above, write the complete final answer now, citing claims with [n] "
            "by those passage numbers. Note any part the passages don't cover."
        )})
        # Keep tools in the request so the cached prefix (system + tools + the
        # whole conversation) still matches and the pool is read from cache, not
        # re-billed — dropping tools would change the prefix and force a full
        # cache miss. tool_choice "none" forbids another search while keeping that
        # prefix intact, so we get both the cache hit and a guaranteed text answer.
        fresp = client.messages.create(
            model=CONFIG.claude_model,
            max_tokens=CONFIG.max_tokens,
            system=system_param,
            tools=tools,
            tool_choice={"type": "none"},
            messages=messages,     # cached: pool read from cache, not re-billed
        )
        add_usage(TOKENS.record(fresp.usage, "answer_forced"))
        reply = strip_preamble("".join(
            b.text for b in fresp.content if getattr(b, "type", None) == "text"))
        if len(reply.strip()) >= 40:
            yield {"type": "answer_delta", "text": reply}
        else:  # tool-saturated context gave a near-empty reply — one-shot fallback
            reply = ""
            for piece in answer_stream(question, pool):
                if isinstance(piece, dict):  # terminal {"type": "final", ...}
                    reply = piece["text"]
                    add_usage(piece["usage"])
                else:
                    yield {"type": "answer_delta", "text": piece}
    yield {"type": "final", "reply": strip_preamble(reply), "pool": pool, "usage": usage_total}


def agentic_chat(question: str) -> tuple[str, list[dict], dict]:
    """Let Claude call search repeatedly until it can answer (bounded loop).

    Non-streaming wrapper over `agentic_chat_events` — drains the loop and
    returns (reply, pool, usage). Used by /api/chat and eval.py, which need the
    whole result in one shot; the streaming path consumes the events directly.
    """
    for evt in agentic_chat_events(question):
        if evt["type"] == "final":
            return evt["reply"], evt["pool"], evt["usage"]
    return "", [], {"input_tokens": 0, "output_tokens": 0}  # loop never yields nothing
