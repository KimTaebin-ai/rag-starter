import { useEffect, useId, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { linkifyCitations, pdfHref, srcLabel } from "../format";
import { AgentTrace } from "./AgentTrace";
import { RetrievalPanel } from "./RetrievalPanel";

// One chat turn: an avatar + the body (assistant Markdown vs. a user bubble),
// its sources, per-answer token/latency meta, and the retrieval panel.
export function Message({ m }) {
  const uid = useId(); // scopes chunk element ids to this message
  const [panelOpen, setPanelOpen] = useState(false);
  const [activeN, setActiveN] = useState(null);
  const isUser = m.role === "user";

  const citeNs = new Set((m.citations || []).map((c) => c.n));

  // Clicking an inline [n] opens the retrieval panel and reveals that source.
  function goToCitation(n) {
    setActiveN(n);
    setPanelOpen(true);
  }

  useEffect(() => {
    if (activeN == null || !panelOpen) return;
    document
      .getElementById(`${uid}-src-${activeN}`)
      ?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [activeN, panelOpen, uid]);

  // Render [n](#cite-n) markers as inline citation buttons; leave real links alone.
  const markdownComponents = {
    a({ href, children, ...props }) {
      if (href && href.startsWith("#cite-")) {
        return (
          <button
            type="button"
            className="cite-ref"
            title="View source"
            onClick={() => goToCitation(Number(href.slice(6)))}
          >
            [{children}]
          </button>
        );
      }
      return (
        <a href={href} target="_blank" rel="noreferrer" {...props}>
          {children}
        </a>
      );
    },
  };

  return (
    <div className={`msg msg-${m.role}`}>
      <div className="avatar">{isUser ? "You" : "✦"}</div>

      <div className="msg-main">
        {!isUser && <AgentTrace steps={m.agentSteps} streaming={m.streaming} />}

        {isUser ? (
          <div className="bubble">{m.text}</div>
        ) : (
          <div className="msg-body markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
              {linkifyCitations(m.text, citeNs)}
            </ReactMarkdown>
            {m.streaming && <span className="stream-caret" aria-hidden="true" />}
          </div>
        )}

        {m.citations && m.citations.length > 0 && (
          <div className="sources">
            <span className="sources-label">
              {m.citations.length} source{m.citations.length > 1 ? "s" : ""}
            </span>
            {m.citations.map((c) => (
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
        )}

        {(m.usage || m.timing || m.clientMs != null) && (
          <div className="meta">
            {m.usage && (
              <span className="meta-item">
                {m.usage.input_tokens} in · {m.usage.output_tokens} out
                {m.usage.cache_read > 0 && (
                  <span className="meta-cache" title="Input tokens served from prompt cache (~0.1× cost)">
                    {" "}
                    · {m.usage.cache_read.toLocaleString()} cached
                  </span>
                )}
              </span>
            )}
            {m.usage && (m.timing?.total_ms != null || m.clientMs != null) && (
              <span className="meta-sep">·</span>
            )}
            {m.timing?.total_ms != null ? (
              <span className="meta-item total">
                {m.timing.total_ms} ms
                {m.timing.retrieval_ms != null && (
                  <span className="meta-item">
                    {" "}
                    (search {m.timing.retrieval_ms} · LLM {m.timing.llm_ms})
                  </span>
                )}
              </span>
            ) : (
              m.clientMs != null && <span className="meta-item">{m.clientMs} ms</span>
            )}
          </div>
        )}

        {!isUser && (
          <RetrievalPanel
            retrieval={m.retrieval}
            open={panelOpen}
            onToggle={setPanelOpen}
            activeN={activeN}
            idPrefix={uid}
          />
        )}
      </div>
    </div>
  );
}
