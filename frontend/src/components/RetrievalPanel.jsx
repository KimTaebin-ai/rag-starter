import { pdfHref, srcLabel } from "../format";

// Retrieval monitoring section (rendered inside the Inspector): shows what the
// vector search returned, the similarity scores, which chunks the LLM actually
// used (incl. cross-references), and which the threshold dropped — so you can
// judge retrieval accuracy at a glance. `activeN`/`idPrefix` let an inline [n]
// citation scroll/highlight the exact source chunk it points to.
export function RetrievalPanel({ retrieval, activeN, idPrefix }) {
  if (!retrieval) return null;
  const { chunks = [], dropped = [], threshold, search_query } = retrieval;
  return (
    <section className="retrieval">
      <h3 className="insp-title">
        Retrieval — {chunks.length} used
        {dropped.length ? `, ${dropped.length} dropped` : ""}
        {threshold != null ? ` (threshold ${threshold})` : ""}
      </h3>
      {search_query && (
        <div className="rq">
          Search query: <code>{search_query}</code>
        </div>
      )}
      {chunks.map((h) => (
        <a
          key={`u${h.n}`}
          id={idPrefix ? `${idPrefix}-src-${h.n}` : undefined}
          className={`rhit${activeN === h.n ? " rhit-active" : ""}`}
          href={pdfHref(h) || undefined}
          target="_blank"
          rel="noreferrer"
          title={`Open PDF${h.page ? ` (p.${h.page})` : ""}`}
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
        </a>
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
    </section>
  );
}
