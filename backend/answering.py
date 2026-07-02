"""One-shot generation: search-query rewrite and the single-shot grounded answer
(sync `answer` + streaming `answer_stream`). The agentic tool-loop lives in
agentic.py; shared prompt text lives in prompts.py.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import CONFIG
from citations import build_context
from tokens import TOKENS, client
from prompts import CLARIFY_PREFIX, SYSTEM_PROMPT


# ════════════════════════════════════════════════════════════════
# Query rewrite
# ════════════════════════════════════════════════════════════════

def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """Ask Claude to turn the user's question into a clear search query.

    Resolves vague pronouns / context-dependent phrasing into explicit terms.
    Prior turns (when present) let it resolve follow-ups like "그럼 야간 요건은?"
    into a standalone query. Falls back to the original question on any error.
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
                "(including the prior conversation turns) into explicit "
                "regulatory terms (e.g. part/section names, airspace classes). "
                "Output ONLY the rewritten query, no quotes, no explanation.\n"
                "EXCEPTION — clarification: output "
                f"'{CLARIFY_PREFIX} <one short question, in the user's language, "
                "asking them to narrow it down, with 2-4 concrete example "
                "aspects>' ONLY when BOTH hold: (a) the question (after using the "
                "prior turns) is squarely WITHIN this aviation / 14 CFR domain, "
                "AND (b) it is nothing but a bare topic with NO specific aspect — "
                "e.g. just '비행기?', 'aircraft?', '항공 규정 알려줘', 'tell me about "
                "aviation rules'.\n"
                "Do NOT clarify — instead output a plain search query (or the "
                "question as-is) and let retrieval + the answer step report "
                "'not in the sources' — in EITHER of these cases:\n"
                "1. The question is OUTSIDE the aviation / 14 CFR domain (cooking, "
                "personal questions, general chit-chat, etc. — e.g. "
                "'사과파이 만드는 법', 'what is my name?', 'tell me a joke'). In this "
                "case output the user's question VERBATIM, exactly as written — do "
                "NOT translate it, add any aviation/regulatory terms, answer it, or "
                "remark that it is out of scope. A faithful off-domain query "
                "retrieves nothing and the system then reports no information. "
                "Never ask them to narrow an off-domain question down to an "
                "aviation topic.\n"
                "2. The question names ANY specific aspect — a requirement, "
                "section, number, name, person, event, date, comparison, or action "
                "(e.g. 'the name of the last selected pilot', 'night flight hours', "
                "'Class B rules') — even if the corpus likely can't answer it.\n"
                "Clarify is ONLY for a bare, in-domain aviation topic; never for an "
                "off-domain question and never for a specific but possibly-unanswerable one."
            ),
            messages=(history or []) + [{"role": "user", "content": question}],
        )
        TOKENS.record(resp.usage, "query_rewrite")
        rewritten = resp.content[0].text.strip()
        return rewritten or question
    except Exception as exc:  # noqa: BLE001 — never let rewrite break the request
        print(f"[query_rewrite] failed, using original question: {exc}")
        return question


# ════════════════════════════════════════════════════════════════
# Standard grounded answer
# ════════════════════════════════════════════════════════════════

def answer(question: str, hits: list[dict]) -> tuple[str, dict]:
    """Generate the grounded answer from the retrieved chunks. Returns (text, usage).

    Deliberately does NOT take conversation history: a follow-up's context is
    already resolved into the search query by `rewrite_query`, and retrieval
    fetches the right chunks, so the answer is generated from THIS turn's context
    only. Feeding prior turns here hurt more than it helped — past answers carry
    stale [n] markers pointing at context that isn't re-sent (miscitation) and
    bias the answer toward earlier topics, degrading Answer Quality/Citations on
    follow-up turns while growing input tokens every turn.
    """
    context = build_context(hits)
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    resp = client.messages.create(
        model=CONFIG.claude_model,
        max_tokens=CONFIG.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    usage = TOKENS.record(resp.usage, "answer")
    return resp.content[0].text, usage


def answer_stream(question: str, hits: list[dict]):
    """Streaming variant of `answer` for a typewriter UX.

    Yields the answer's text incrementally as Claude produces it, then a final
    {"type": "final", "text": <full answer>, "usage": <usage>} dict once the
    stream closes (so the caller can renumber citations and record tokens on the
    complete text). Same prompt/inputs as `answer` — only the transport differs;
    like `answer`, it generates from THIS turn's context only (no history).
    """
    context = build_context(hits)
    user_content = f"CONTEXT:\n{context}\n\nQUESTION:\n{question}"
    with client.messages.stream(
        model=CONFIG.claude_model,
        max_tokens=CONFIG.max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    ) as stream:
        for text in stream.text_stream:
            yield text
        final = stream.get_final_message()
    usage = TOKENS.record(final.usage, "answer")
    full_text = "".join(b.text for b in final.content if getattr(b, "type", None) == "text")
    yield {"type": "final", "text": full_text, "usage": usage}
