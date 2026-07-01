// Live monitor for the agentic search loop (backend `agent` SSE events). Each
// step Claude takes — deciding, searching the corpus, composing the answer — is
// rendered as a timeline row so you can watch the tool loop run and see exactly
// which queries it issued and how many passages each turn added to the pool.
//
// Steps arrive in order: {type:"think"} → {type:"search",...} (0+) per turn,
// then a final {type:"answer",...}. `streaming` keeps the panel open and shows a
// pulsing indicator on the last step while the loop is still running.
export function AgentTrace({ steps, streaming }) {
  if (!steps || steps.length === 0) return null;

  const searches = steps.filter((s) => s.type === "search");
  const answered = steps.some((s) => s.type === "answer");
  // A trailing "think" with nothing after it = the model is mid-decision.
  const last = steps[steps.length - 1];
  const deciding = streaming && last.type === "think";

  return (
    <details className="agent-trace" open={streaming}>
      <summary>
        <span className="at-spark">✦</span>
        Agent loop
        <span className="at-count">
          {searches.length} search{searches.length === 1 ? "" : "es"}
        </span>
        {streaming && <span className="at-live">running…</span>}
      </summary>

      <ol className="at-steps">
        {steps.map((s, i) => {
          const isLast = i === steps.length - 1;
          const pulse = streaming && isLast ? " at-pulse" : "";
          if (s.type === "search") {
            return (
              <li key={i} className={`at-step at-search${pulse}`}>
                <span className="at-icon">🔍</span>
                <span className="at-body">
                  <code className="at-query">{s.query}</code>
                  <span className="at-found">
                    {s.found > 0 ? `+${s.found} new` : "no new passages"}
                    <span className="at-total"> · pool {s.total}</span>
                  </span>
                </span>
              </li>
            );
          }
          if (s.type === "answer") {
            return (
              <li key={i} className={`at-step at-answer${pulse}`}>
                <span className="at-icon">✎</span>
                <span className="at-body">
                  Composing answer from {s.total} source{s.total === 1 ? "" : "s"}
                  {s.forced && <span className="at-forced"> (iteration cap reached)</span>}
                </span>
              </li>
            );
          }
          // "think" — only surfaced as a live row while it's the trailing step.
          if (s.type === "think" && deciding && isLast) {
            return (
              <li key={i} className="at-step at-think at-pulse">
                <span className="at-icon">…</span>
                <span className="at-body">Deciding next step</span>
              </li>
            );
          }
          return null;
        })}
      </ol>

      {!streaming && !answered && (
        <div className="at-note">Loop ended without a final answer step.</div>
      )}
    </details>
  );
}
