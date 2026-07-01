import { useMemo, useState } from "react";

import { postChat } from "../api";

// Owns the conversation: message list, send flow, and the running session
// totals shown in the header.
export function useChat() {
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);

  const totals = useMemo(() => {
    const totalIn = messages.reduce((s, m) => s + (m.usage?.input_tokens || 0), 0);
    const totalOut = messages.reduce((s, m) => s + (m.usage?.output_tokens || 0), 0);
    const answered = messages.filter((m) => m.timing?.total_ms != null);
    const avgMs = answered.length
      ? Math.round(answered.reduce((s, m) => s + m.timing.total_ms, 0) / answered.length)
      : null;
    return { totalIn, totalOut, avgMs };
  }, [messages]);

  async function send(question) {
    if (!question.trim() || loading) return;

    // Snapshot prior turns as history before appending the new question.
    const history = messages.map((m) => ({ role: m.role, text: m.text }));
    setMessages((m) => [...m, { role: "user", text: question }]);
    setLoading(true);

    const t0 = performance.now();
    try {
      const data = await postChat(question, history);
      const clientMs = Math.round(performance.now() - t0); // round-trip incl. network
      setMessages((m) => [...m, {
        role: "assistant",
        text: data.reply,
        citations: data.citations || [],
        usage: data.usage || null,
        retrieval: data.retrieval || null,
        timing: data.timing || null,
        clientMs,
      }]);
    } catch (err) {
      setMessages((m) => [...m, {
        role: "assistant",
        text: `_Request failed: ${err.message}_`,
        citations: [],
        usage: null,
        retrieval: null,
        timing: null,
        clientMs: Math.round(performance.now() - t0),
      }]);
    } finally {
      setLoading(false);
    }
  }

  return { messages, loading, totals, send };
}
