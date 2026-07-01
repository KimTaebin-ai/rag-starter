import { srcLabel } from "../format";

// Retrieval monitoring panel: shows what the vector search returned, the
// similarity scores, which chunks the LLM actually used (incl. cross-references),
// and which the threshold dropped — so you can judge retrieval accuracy at a glance.
//
// Controlled open state + `activeN`/`idPrefix` let an inline [n] citation open
// this panel and scroll/highlight the exact source chunk it points to.
export function RetrievalPanel({ retrieval, open, onToggle, activeN, idPrefix }) {
  if (!retrieval) return null;
  const { chunks = [], dropped = [], threshold, search_query } = retrieval;
  return (
    <details
      className="retrieval"
      open={open}
      onToggle={(e) => onToggle?.(e.currentTarget.open)}
    >
      <summary>
        Retrieval — {chunks.length} used
        {dropped.length ? `, ${dropped.length} dropped` : ""}
        {threshold != null ? ` (threshold ${threshold})` : ""}
      </summary>
      {search_query && (
        <div className="rq">
          Search query: <code>{search_query}</code>
        </div>
      )}
      {chunks.map((h) => (
        <div
          key={`u${h.n}`}
          id={idPrefix ? `${idPrefix}-src-${h.n}` : undefined}
          className={`rhit${activeN === h.n ? " rhit-active" : ""}`}
        >
          <div className="rhit-head">
            <span className="rn">[{h.n}]</span>
            <span className="rsrc">{srcLabel(h)}</span>
            {h.xref ? (
              <span className="rxref">↳ cross-ref</span>
            ) : (
              <span className="rsim">sim {h.similarity}</span>
            )}
          </div>
          {h.preview && <div className="rprev">{h.preview}…</div>}
        </div>
      ))}
      {dropped.length > 0 && (
        <div className="rdropped">
          Dropped below threshold:
          {dropped.map((h, i) => (
            <span key={`d${i}`} className="rdrop">
              {srcLabel(h)} (sim {h.similarity})
            </span>
          ))}
        </div>
      )}
    </details>
  );
}
