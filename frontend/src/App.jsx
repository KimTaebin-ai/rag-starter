import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

function srcLabel(c) {
  let label = c.title || c.source
  if (c.section) label += `, ${c.section}`
  if (c.page) label += `, p.${c.page}`
  return label
}

// Retrieval monitoring panel: shows what the vector search returned, the
// similarity scores, which chunks the LLM actually used, and which the
// threshold dropped — so you can judge retrieval accuracy at a glance.
function RetrievalPanel({ retrieval }) {
  if (!retrieval) return null
  const { chunks = [], dropped = [], threshold, search_query } = retrieval
  return (
    <details className="retrieval">
      <summary>🔍 검색 모니터링 — 사용 {chunks.length}개{dropped.length ? `, 제외 ${dropped.length}개` : ''}{threshold != null ? ` (임계값 ${threshold})` : ''}</summary>
      {search_query && <div className="rq">검색 쿼리: <code>{search_query}</code></div>}
      {chunks.map((h) => (
        <div key={`u${h.n}`} className="rhit">
          <div className="rhit-head">
            <span className="rn">[{h.n}]</span>
            <span className="rsrc">{srcLabel(h)}</span>
            <span className="rsim">sim {h.similarity}</span>
          </div>
          {h.preview && <div className="rprev">{h.preview}…</div>}
        </div>
      ))}
      {dropped.length > 0 && (
        <div className="rdropped">
          임계값 미만으로 제외:
          {dropped.map((h, i) => (
            <span key={`d${i}`} className="rdrop">{srcLabel(h)} (sim {h.similarity})</span>
          ))}
        </div>
      )}
    </details>
  )
}

export default function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  // Running token totals across the session (monitoring).
  const totalIn = messages.reduce((s, m) => s + (m.usage?.input_tokens || 0), 0)
  const totalOut = messages.reduce((s, m) => s + (m.usage?.output_tokens || 0), 0)

  async function send(e) {
    e.preventDefault()
    if (!input.trim() || loading) return

    const question = input
    setMessages((m) => [...m, { role: 'user', text: question }])
    setInput('')
    setLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question }),
      })
      const data = await res.json()

      setMessages((m) => [...m, {
        role: 'assistant',
        text: data.reply,
        citations: data.citations || [],
        usage: data.usage || null,
        retrieval: data.retrieval || null,
      }])
    } catch (err) {
      setMessages((m) => [...m, {
        role: 'assistant',
        text: `_요청 중 오류가 발생했습니다: ${err.message}_`,
        citations: [],
        usage: null,
        retrieval: null,
      }])
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="app">
      <header className="topbar">
        <h1>14 CFR RAG Chat</h1>
        <div className="totals">세션 누적 토큰: input {totalIn} / output {totalOut}</div>
      </header>
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={`msg msg-${m.role}`}>
            <div className="msg-role">{m.role}</div>
            {/* Phase 1-1: render assistant Markdown (tables via remark-gfm). */}
            {m.role === 'assistant' ? (
              <div className="msg-body markdown">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.text}</ReactMarkdown>
              </div>
            ) : (
              <div className="msg-body">{m.text}</div>
            )}

            {m.citations && m.citations.length > 0 && (
              <div className="sources">
                Sources: {m.citations.map((c) => (
                  <span key={c.n} className="source">
                    [{c.n}] {srcLabel(c)}
                  </span>
                ))}
              </div>
            )}

            {/* Phase 1-2 / 0-1: per-answer token usage. */}
            {m.usage && (
              <div className="usage">
                이번 답변: input {m.usage.input_tokens} / output {m.usage.output_tokens} 토큰
              </div>
            )}

            {/* Retrieval monitoring (accuracy). */}
            {m.role === 'assistant' && <RetrievalPanel retrieval={m.retrieval} />}
          </div>
        ))}
        {loading && <div className="msg msg-assistant"><div className="msg-body">…</div></div>}
      </div>
      <form onSubmit={send}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask about pilot certs, medical, airspace, operating rules…"
          autoFocus
        />
        <button type="submit" disabled={loading}>Send</button>
      </form>
    </div>
  )
}
