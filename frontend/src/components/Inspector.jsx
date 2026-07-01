import { useEffect } from "react";

import { pdfHref, srcLabel } from "../format";
import { AgentTrace } from "./AgentTrace";
import { RetrievalPanel } from "./RetrievalPanel";

// Right-hand monitoring surface for the currently selected answer: the agent
// loop, the sources it cited, per-answer token/latency, and the retrieval
// panel (what was returned, scored, used, and dropped). Keeping this out of the
// chat keeps the transcript readable while giving retrieval debugging its own
// dedicated column. `activeN` is set when an inline [n] in the answer is
// clicked, and reveals/scrolls to the matching source chunk below.
export function Inspector({ message, activeN }) {
  // Scroll the cited chunk into view within the panel when [n] is clicked.
  useEffect(() => {
    if (activeN == null || !message) return;
    document
      .getElementById(`insp-src-${activeN}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [activeN, message]);

  const citations = message?.citations || [];
  const hasMeta = message && (message.usage || message.timing || message.clientMs != null);

  return (
    <aside className="inspector">
      <div className="inspector-head">
        <span className="brand-mark">✦</span>
        <div>
          <h2>Inspector</h2>
          <p className="inspector-sub">Retrieval · sources · usage</p>
        </div>
      </div>

      {!message ? (
        <div className="inspector-empty">
          Ask a question to watch the agent loop and inspect the retrieved
          sources, similarity scores, and token usage for each answer here.
          Click any answer to bring its details back up.
        </div>
      ) : (
        <div className="inspector-body">
          <AgentTrace steps={message.agentSteps} streaming={message.streaming} />

          {citations.length > 0 && (
            <section className="insp-section">
              <h3 className="insp-title">
                {citations.length} source{citations.length > 1 ? "s" : ""}
              </h3>
              <div className="sources">
                {citations.map((c) => (
                  <a
                    key={c.n}
                    className={`source${c.xref ? " source-xref" : ""}`}
                    href={pdfHref(c)}
                    target="_blank"
                    rel="noreferrer"
                    title={`Open PDF${c.page ? ` (p.${c.page})` : ""}`}
                  >
                    <span className="source-n">[{c.n}]</span>
                    <span className="source-label">{srcLabel(c)}</span>
                    {c.xref && <span className="source-tag">↳ ref</span>}
                    <span className="source-open" aria-hidden="true">
                      ↗
                    </span>
                  </a>
                ))}
              </div>
            </section>
          )}

          {hasMeta && (
            <div className="meta">
              {message.usage && (
                <span className="meta-item">
                  {message.usage.input_tokens} in · {message.usage.output_tokens} out
                  {message.usage.cache_read > 0 && (
                    <span
                      className="meta-cache"
                      title="Input tokens served from prompt cache (~0.1× cost)"
                    >
                      {" "}
                      · {message.usage.cache_read.toLocaleString()} cached
                    </span>
                  )}
                </span>
              )}
              {message.usage && (message.timing?.total_ms != null || message.clientMs != null) && (
                <span className="meta-sep">·</span>
              )}
              {message.timing?.total_ms != null ? (
                <span className="meta-item total">
                  {message.timing.total_ms} ms
                  {message.timing.retrieval_ms != null && (
                    <span className="meta-item">
                      {" "}
                      (search {message.timing.retrieval_ms} · LLM {message.timing.llm_ms})
                    </span>
                  )}
                </span>
              ) : (
                message.clientMs != null && (
                  <span className="meta-item">{message.clientMs} ms</span>
                )
              )}
            </div>
          )}

          <RetrievalPanel retrieval={message.retrieval} activeN={activeN} idPrefix="insp" />
        </div>
      )}
    </aside>
  );
}
