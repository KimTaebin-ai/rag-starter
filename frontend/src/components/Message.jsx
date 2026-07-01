import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { linkifyCitations, pdfHref } from "../format";

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

  // Render [n](#cite-n) markers as inline citation links: they open the cited
  // source PDF (jumping to the page) and also reveal that source in the
  // inspector. Leave real links alone.
  const markdownComponents = {
    a({ href, children, ...props }) {
      if (href && href.startsWith("#cite-")) {
        const n = Number(href.slice(6));
        const cit = (m.citations || []).find((c) => c.n === n);
        const pdf = cit ? pdfHref(cit) : null;
        return (
          <a
            className="cite-ref"
            href={pdf || undefined}
            target={pdf ? "_blank" : undefined}
            rel="noreferrer"
            title={
              cit
                ? `Open source PDF${cit.page ? ` (p.${cit.page})` : ""}`
                : "View source"
            }
            onClick={(e) => {
              e.stopPropagation();
              onCite?.(index, n);
            }}
          >
            [{children}]
          </a>
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
