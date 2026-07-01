import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { linkifyCitations } from "../format";

// One chat turn: an avatar + the body (assistant Markdown vs. a user bubble).
// Monitoring detail — sources, tokens/latency, agent trace, and the retrieval
// panel — lives in the Inspector sidebar for the selected answer; clicking an
// assistant turn (or one of its inline [n] markers) selects it there.
export function Message({ m, index, active, onSelect, onCite }) {
  const isUser = m.role === "user";

  if (isUser) {
    return (
      <div className="msg msg-user">
        <div className="avatar">You</div>
        <div className="msg-main">
          <div className="bubble">{m.text}</div>
        </div>
      </div>
    );
  }

  const citeNs = new Set((m.citations || []).map((c) => c.n));

  // Render [n](#cite-n) markers as inline citation buttons: clicking one toggles
  // the inspector for that source (reveal/scroll, or close if already showing
  // it) — it never opens the PDF. Leave real links alone.
  const markdownComponents = {
    a({ href, children, ...props }) {
      if (href && href.startsWith("#cite-")) {
        return (
          <button
            type="button"
            className="cite-ref"
            title="Toggle source in inspector"
            onClick={(e) => {
              e.stopPropagation();
              onCite?.(index, Number(href.slice(6)));
            }}
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
    <div className="msg msg-assistant">
      <div className="avatar">✦</div>

      <div
        className={`msg-main${active ? " msg-active" : ""}`}
        onClick={() => onSelect?.(index)}
      >
        <div className="msg-body markdown">
          <ReactMarkdown remarkPlugins={[remarkGfm]} components={markdownComponents}>
            {linkifyCitations(m.text, citeNs)}
          </ReactMarkdown>
          {m.streaming && <span className="stream-caret" aria-hidden="true" />}
        </div>

        {m.citations && m.citations.length > 0 && (
          <div className="msg-cite-hint">
            {m.citations.length} source{m.citations.length > 1 ? "s" : ""} · details in
            inspector →
          </div>
        )}
      </div>
    </div>
  );
}
