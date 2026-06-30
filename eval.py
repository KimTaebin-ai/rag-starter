"""Run the fixed question set against a running backend and print a report.

Use it to compare a feature off vs. on: flip the toggle in .env, restart the
backend, run this, and diff the two reports (answer quality, citation sources,
input/output tokens).

Usage:
    python app.py            # in one terminal (backend on :5000)
    python eval.py           # in another

Env:
    BACKEND_URL   default http://127.0.0.1:5001
"""
import json
import os
import textwrap
import urllib.request

from eval_questions import QUESTIONS

BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:5001")


def post_chat(message: str, timeout: int = 120) -> dict:
    req = urllib.request.Request(
        f"{BACKEND_URL}/api/chat",
        data=json.dumps({"message": message}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> None:
    total_in = total_out = 0
    for i, item in enumerate(QUESTIONS, 1):
        print("=" * 78)
        print(f"Q{i} [{item['type']}]")
        print(f"  {item['q']}")
        try:
            data = post_chat(item["q"])
        except Exception as exc:  # noqa: BLE001
            print(f"  ! request failed: {exc}")
            continue

        reply = data.get("reply", "")
        citations = data.get("citations", [])
        usage = data.get("usage") or {}
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        total_in += in_tok
        total_out += out_tok

        print("\n  Answer:")
        print(textwrap.indent(textwrap.fill(reply, 90), "    "))

        cited = sorted({c["source"] for c in citations})
        print(f"\n  Cited sources: {cited or '(none)'}")
        print(f"  Expected (rough): {item['expect_sources'] or '(none — should be no-info)'}")
        print(f"  Tokens: input={in_tok} output={out_tok}")

    print("=" * 78)
    print(f"TOTAL tokens over {len(QUESTIONS)} questions: input={total_in} output={total_out}")


if __name__ == "__main__":
    main()
