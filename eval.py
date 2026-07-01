"""Run the question set against a running backend and print a graded report.

Use it to compare a feature off vs. on: flip the toggle in .env, restart the
backend, run this, and diff the two reports (answer quality, cited sources,
input/output tokens, latency).

Per question it checks:
  - answerable questions: did a cited source match an expected one? (SRC ok/miss)
  - out-of-corpus questions: did the system return no-info / refuse? (NOINFO ok/miss)

Usage (activate the venv first: `source .venv/bin/activate`):
    python backend/app.py     # one terminal (backend on :5001)
    python eval.py            # the main question set
    python eval.py --negatives    # only the no-answer (negative) questions
    python eval.py --all          # main set + negatives
    python eval.py --followups    # multi-turn sequences (checks follow-up continuity)
    python eval.py --full         # also print each answer's full text

Env:
    BACKEND_URL   default http://127.0.0.1:5001
"""
import json
import os
import sys
import textwrap
import urllib.request
from collections import defaultdict

from eval_questions import FOLLOWUP_SEQUENCES, NO_ANSWER_QUESTIONS, QUESTIONS

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")
# Phrases that signal a graceful refusal / "not in the corpus" answer. Broad on
# purpose: a good refusal of a domain-adjacent question often names the related
# material it DID find (e.g. "the context covers fractional-ownership rules, not
# scheduled operations") — that's a correct refusal even though it cites the
# adjacent chunk, so we detect the disclaimer regardless of citations. A real
# hallucination asserts the answer instead and hits none of these.
NO_INFO_MARKERS = (
    "관련 정보를 찾지 못했습니다",
    "don't contain", "do not contain", "does not contain", "doesn't contain",
    "not contain", "not covered", "does not cover", "do not cover", "doesn't cover",
    "not addressed", "not in the provided", "not in the sources",
    "not in these sources", "no information",
)


def post_chat(message: str, history: list | None = None, timeout: int = 180) -> dict:
    payload = {"message": message}
    if history:
        payload["history"] = history
    req = urllib.request.Request(
        f"{BACKEND_URL}/api/chat",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def run_followups() -> None:
    """Send each multi-turn sequence, carrying history, so you can eyeball
    whether a follow-up ("How about for Class B?") stays on topic."""
    for i, turns in enumerate(FOLLOWUP_SEQUENCES, 1):
        print("=" * 80)
        print(f"Follow-up sequence {i}")
        history: list = []
        for t, q in enumerate(turns, 1):
            print(f"\n  Turn {t} — Q: {q}")
            try:
                data = post_chat(q, history)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! request failed: {exc}")
                break
            reply = data.get("reply", "")
            cited = sorted({c["source"].split("-")[-1].replace(".pdf", "")
                            for c in data.get("citations", [])})
            print(textwrap.indent(textwrap.fill(reply, 90), "    "))
            print(f"    cited={cited or '(none)'}")
            history += [{"role": "user", "text": q},
                        {"role": "assistant", "text": reply}]


def looks_like_no_info(reply: str, citations: list) -> bool:
    r = reply.lower()
    return not citations or any(m in r for m in NO_INFO_MARKERS)


def main() -> None:
    show_full = "--full" in sys.argv
    if "--followups" in sys.argv:
        run_followups()
        return
    if "--all" in sys.argv:
        dataset = QUESTIONS + NO_ANSWER_QUESTIONS
    elif "--negatives" in sys.argv:
        dataset = NO_ANSWER_QUESTIONS
    else:
        dataset = QUESTIONS

    total_in = total_out = total_ms = 0
    passes = 0
    by_cat = defaultdict(lambda: [0, 0])  # category -> [passed, total]

    for i, item in enumerate(dataset, 1):
        cat = item["category"]
        print("=" * 80)
        print(f"Q{i} [{cat}]  {item['q']}")
        try:
            data = post_chat(item["q"])
        except Exception as exc:  # noqa: BLE001
            print(f"  ! request failed: {exc}")
            by_cat[cat][1] += 1
            continue

        reply = data.get("reply", "")
        citations = data.get("citations", [])
        usage = data.get("usage") or {}
        timing = data.get("timing") or {}
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        ms = timing.get("total_ms", 0)
        total_in += in_tok
        total_out += out_tok
        total_ms += ms

        cited = sorted({c["source"] for c in citations})

        # Grade: source-match for answerable, no-info for out-of-corpus.
        if item["answerable"]:
            ok = any(s in item["expect_sources"] for s in cited)
            verdict = f"SRC {'ok ' if ok else 'MISS'}"
        else:
            ok = looks_like_no_info(reply, citations)
            verdict = f"NOINFO {'ok ' if ok else 'MISS'}"
        passes += ok
        by_cat[cat][0] += ok
        by_cat[cat][1] += 1

        if show_full:
            print("  Answer:")
            print(textwrap.indent(textwrap.fill(reply, 92), "    "))
        else:
            print(f"  → {textwrap.shorten(reply.replace(chr(10), ' '), 110)}")
        print(f"  [{verdict}]  cited={[s.split('-')[-1].replace('.pdf','') for s in cited] or '(none)'}"
              f"  expect={[s.split('-')[-1].replace('.pdf','') for s in item['expect_sources']] or '(no-info)'}")
        print(f"  tokens in={in_tok} out={out_tok}  latency={ms}ms")

    print("=" * 80)
    n = len(dataset)
    print(f"PASS {passes}/{n}  ({100 * passes // n}%)")
    print("by category:")
    for cat, (p, t) in by_cat.items():
        print(f"  {cat:28s} {p}/{t}")
    avg_ms = total_ms // max(1, n)
    print(f"tokens: input={total_in} output={total_out}  |  avg latency={avg_ms}ms")


if __name__ == "__main__":
    main()
