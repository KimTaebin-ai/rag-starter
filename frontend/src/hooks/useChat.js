import { useMemo, useState } from "react";

import { postChatStream } from "../api";

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

    // Patch the LAST message in place — used to stream text into the assistant
    // turn and to swap in the final metadata once the stream closes.
    const patchLast = (patch) =>
      setMessages((m) => {
        const next = m.slice();
        const i = next.length - 1;
        next[i] = { ...next[i], ...(typeof patch === "function" ? patch(next[i]) : patch) };
        return next;
      });

    // Append the assistant bubble on the first delta so the loading indicator
    // shows during retrieval, then text starts filling this bubble.
    let started = false;
    const ensureBubble = () => {
      if (started) return;
      started = true;
      setMessages((m) => [...m, { role: "assistant", text: "", citations: [], streaming: true }]);
    };

    try {
      await postChatStream(question, history, {
        onDelta: (text) => {
          ensureBubble();
          patchLast((prev) => ({ text: prev.text + text }));
        },
        onDone: (data) => {
          ensureBubble();
          patchLast({
            text: data.reply,
            citations: data.citations || [],
            usage: data.usage || null,
            retrieval: data.retrieval || null,
            timing: data.timing || null,
            clientMs: Math.round(performance.now() - t0), // round-trip incl. network
            streaming: false,
          });
        },
      });
    } catch (err) {
      ensureBubble();
      patchLast({
        text: `_Request failed: ${err.message}_`,
        streaming: false,
        clientMs: Math.round(performance.now() - t0),
      });
    } finally {
      setLoading(false);
    }
  }

  return { messages, loading, totals, send };
}
