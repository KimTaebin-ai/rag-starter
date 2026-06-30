# Context Management Project — RAG over a document corpus

A starter that extends the Foundations chat with retrieval-augmented generation. You'll implement the indexing pipeline and wire retrieval + citations into the chat backend.

## What's in here

```
rag-starter/
├── documents/                  20 real Wikipedia articles (Apollo missions)
├── indexer.py                  walk docs → chunk → embed → store (title metadata)
├── config.py                   feature toggles (env-driven) — see below
├── backend/
│   ├── app.py                  RAG chat: rewrite → search → threshold → rerank → group
│   └── requirements.txt
├── frontend/                   React UI (Markdown rendering + token display)
├── eval_questions.py           fixed test question set
├── eval.py                     run the set against the backend, print a report
└── .env.example
```

## Feature toggles & measurement

Every improvement is gated behind an env toggle in [config.py](config.py) so you can
measure each one independently. Defaults are in `.env.example`. To A/B a feature:
flip its toggle in `.env`, restart `backend/app.py`, and re-run `python eval.py`,
then diff the answer quality, cited sources, and input/output token counts.

| Phase | Feature | Env toggle | Default |
| ----- | ------- | ---------- | ------- |
| 0-1 | Token usage tracking (log + UI) | `TRACK_TOKENS` | on |
| 0-2 | Retrieval debug logs | `DEBUG_SEARCH` | on |
| 1-1 | Markdown rendering (frontend) | — (always on) | on |
| 1-2 | Token display in UI | `TRACK_TOKENS` | on |
| 2-1 | Similarity threshold filter + no-info path | `ENABLE_THRESHOLD` / `SIMILARITY_THRESHOLD` | on / 0.35 |
| 2-2 | Query rewrite (extra LLM call) | `ENABLE_QUERY_REWRITE` | off |
| 2-3 | Title/source metadata on chunks | — (stored at index time) | on |
| 3-1 | Group/sort chunks by document | `ENABLE_CHUNK_GROUPING` | on |
| 3-2 | Inject source into context | `ENABLE_SOURCE_IN_CONTEXT` | on |
| 4-1 | LLM re-ranking | `ENABLE_RERANK` / `RERANK_FETCH_K` / `RERANK_TOP_N` | off |
| 4-2 | Agentic iterative search (tool loop) | `ENABLE_AGENTIC_SEARCH` / `MAX_SEARCH_ITERS` | off |

> The threshold is cosine **similarity** (higher = closer). With this multilingual
> model, in-corpus questions score ~0.6–0.76 and clearly-unrelated ones ~0.13–0.22,
> so 0.35 filters junk (→ "관련 정보를 찾지 못했습니다", no LLM call) without dropping real hits.
> Watch the `[search]` score logs to tune it.

## Evaluating

```bash
python backend/app.py     # terminal 1
python eval.py            # terminal 2 — runs the fixed question set
```

`eval.py` prints each question, the answer, cited sources vs. expected, and per-question
+ total token counts. Run it with a feature off, then on, and compare.

**The corpus** is the U.S. Federal Aviation Regulations (Title 14 CFR, 2025) as PDFs:

- Part 61 (pilot certification), Part 67 (medical), Part 71 (airspace designation),
  Part 73 (special-use airspace), Part 91 (general operating rules)
- Title 14 Vol 1 (Parts 1–59, ~970 pages) — included as realistic noise

PDFs are extracted with `pypdf` and chunked **by CFR section** (split on `§ N.NN`
boundaries), not by page. Each chunk is cleaned of PDF artifacts (line-break
hyphenation, running headers, docket citations) and prepended with its section
heading, so every chunk carries an anchor like `§ 61.109 Aeronautical experience`
plus a page number. Citations point to `Part N, § X.YY, p.Z`.

> **Why section-aware chunking matters here:** with naive page-by-page chunks, the
> §61.109 passage that literally answers "what flight hours are required?" embedded
> to only 0.39 similarity (rank ~2600) because each chunk blended the tail of one
> section with the head of the next, plus header/docket noise. Splitting on section
> boundaries and stripping boilerplate lifts that same passage to 0.80 (rank #1).
> Similarity ≠ relevance, and chunk hygiene dominated retrieval quality on this corpus.

## Setup

```bash
# from this directory
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r backend/requirements.txt

cp .env.example .env
# set ANTHROPIC_API_KEY
```

The first time you run anything that imports `sentence-transformers`, it will download the embedding model (multilingual, ~470 MB). One-time only.

## Your job

### 1. Implement chunking and build the index

Open `indexer.py`. There's a `TODO` for `chunk_text()`. Implement it (see the Context Management lecture slide for one working version). Then:

```bash
python indexer.py
# Indexing documents from documents/
#   01-overview.md: 3 chunks
#   02-streaks.md: 4 chunks
#   ...
# ✓ Indexed N chunks → index.pkl
```

### 2. Wire retrieval into the chat backend

Open `backend/app.py`. There are `TODO`s for:

- Updating `SYSTEM_PROMPT` with citation rules
- Calling `search(user_message, INDEX, k=5)` to get top chunks
- Formatting them as a numbered context block
- Building `user_content` with `CONTEXT:` + `QUESTION:`

The citation parser is already wired — it returns the source filenames the model cited, which the frontend already displays under each answer.

### 3. Run it

```bash
# Terminal 1 — backend (serves on http://127.0.0.1:5001)
python backend/app.py

# Terminal 2 — frontend
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Ask questions about the regulations. You should see:
- A Markdown-rendered answer that draws on the indexed PDFs
- A `Sources:` line citing `Part N, p.X`
- Per-answer **token usage** and a **🔍 검색 모니터링** panel showing the retrieved
  chunks, their similarity scores, and which ones the threshold dropped
- A running **session token total** in the top bar

**The 5 practice questions** (in [eval_questions.py](eval_questions.py)):

- Private-pilot aeronautical experience (single-engine) — Part 61
- First-class medical disqualifying conditions — Part 67
- VFR fuel reserves, day vs. night — Part 91
- Class B vs. Class C operating requirements — Parts 71 + 91
- What to do before operating in an active restricted area — Part 73 (+ §91.133)

### 4. Report

Pick **5 test questions** that probe the corpus from different angles. For each:
- The question
- The answer the system gave
- Whether the cited sources are correct (open the file, verify)
- A judgment: did the system answer well, weakly, or hallucinate?

Then write up **2 strengths and 2 weaknesses** of your implementation with the worked examples as evidence.

## What to present

- Your chunking choice (size, overlap, boundary rule) and why
- Your system prompt's citation rules
- One question that works cleanly, with the right citations
- One question that fails — wrong answer, missing citation, or hallucinated source
- What you would change (chunking? retrieval? prompt?) to fix the failure

## Alternative project

Want to RAG over your own corpus (your notes, a docs site you've cloned, a code repo's READMEs)? Replace the contents of `documents/` and re-run `python indexer.py`. The rest of the pipeline works unchanged.
