import { useEffect, useState } from "react";

// Loading state for an in-flight answer. Answers take ~5–9s (retrieval + rerank
// + xref + generation), so we show animated dots, a rough stage hint, and a live
// elapsed-time counter so the wait feels responsive instead of frozen.
export function LoadingBubble() {
  const [ms, setMs] = useState(0);

  useEffect(() => {
    const t0 = performance.now();
    const id = setInterval(() => setMs(performance.now() - t0), 100);
    return () => clearInterval(id);
  }, []);

  // Retrieval (incl. rerank) resolves first; the answer generation is the bulk.
  const stage = ms < 2000 ? "Searching documents" : "Generating answer";

  return (
    <div className="msg msg-assistant">
      <div className="avatar">✦</div>
      <div className="msg-main">
        <div className="loading">
          <span className="loading-dots" aria-hidden="true">
            <span />
            <span />
            <span />
          </span>
          <span className="loading-text">{stage}…</span>
          <span className="loading-elapsed">{(ms / 1000).toFixed(1)}s</span>
        </div>
      </div>
    </div>
  );
}
