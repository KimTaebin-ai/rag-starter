// POST one chat turn to the backend. `history` is the prior turns (text only)
// so the server can resolve follow-up questions.
export async function postChat(message, history) {
  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  return res.json();
}

// Stream one chat turn (SSE). Fires onDelta(text) for each incremental chunk as
// the answer is generated, and onDone(payload) with the final reply + citations
// /usage/retrieval/timing. Parses the `data: {...}\n\n` event framing by hand
// (fetch + ReadableStream) so we don't need an EventSource POST shim.
export async function postChatStream(message, history, { onDelta, onDone, onAgent, onReset }) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });
  if (!res.ok || !res.body) throw new Error(`HTTP ${res.status}`);

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  const handle = (raw) => {
    const line = raw.trim();
    if (!line.startsWith("data:")) return;
    const evt = JSON.parse(line.slice(5).trim());
    if (evt.event === "delta") onDelta?.(evt.text);
    else if (evt.event === "agent") onAgent?.(evt);
    else if (evt.event === "reset") onReset?.();
    else if (evt.event === "done") onDone?.(evt);
  };

  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    // Events are separated by a blank line; process each complete one.
    let sep;
    while ((sep = buffer.indexOf("\n\n")) !== -1) {
      handle(buffer.slice(0, sep));
      buffer = buffer.slice(sep + 2);
    }
  }
  if (buffer.trim()) handle(buffer); // flush a trailing event without blank line
}
