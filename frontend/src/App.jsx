import { useEffect, useState } from "react";

import { Composer } from "./components/Composer";
import { Inspector } from "./components/Inspector";
import { LoadingBubble } from "./components/LoadingBubble";
import { Message } from "./components/Message";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { messages, loading, totals, send } = useChat();

  // Which answer the inspector reflects, and (optionally) the source chunk an
  // inline [n] click asked it to reveal.
  const [activeIndex, setActiveIndex] = useState(null);
  const [activeN, setActiveN] = useState(null);

  // Show the loading indicator only while waiting (retrieval); once the answer
  // starts streaming into its own bubble, that bubble is the progress signal.
  const last = messages[messages.length - 1];
  const streaming = last?.role === "assistant" && last.streaming;

  // Index of the most recent assistant turn. The inspector auto-follows it so
  // the live agent trace / retrieval for the answer being streamed stays on
  // screen; a new answer resets any citation the user had drilled into.
  let lastAssistant = null;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      lastAssistant = i;
      break;
    }
  }
  useEffect(() => {
    if (lastAssistant != null) {
      setActiveIndex(lastAssistant);
      setActiveN(null);
    }
  }, [lastAssistant]);

  const selectMessage = (i) => {
    setActiveIndex(i);
    setActiveN(null);
  };
  const goToCitation = (i, n) => {
    setActiveIndex(i);
    setActiveN(n);
  };

  const activeMsg = activeIndex != null ? messages[activeIndex] : null;

  return (
    <div className="shell">
      <div className="app">
        <header className="topbar">
          <div className="brand">
            <span className="brand-mark">✦</span>
            <div>
              <h1>EKO</h1>
              <p className="brand-sub">FAA regulations assistant</p>
            </div>
          </div>
          <div className="stats">
            <span className="stat">↑ {totals.totalIn.toLocaleString()} in</span>
            <span className="stat">↓ {totals.totalOut.toLocaleString()} out</span>
            {totals.avgMs != null && <span className="stat">~{totals.avgMs} ms avg</span>}
          </div>
        </header>

        <main className="messages">
          {messages.length === 0 && !loading && (
            <div className="empty">
              <span className="brand-mark">✦</span>
              <h2>Ask about 14 CFR</h2>
              <p>
                Pilot certification, medical standards, airspace, and operating
                rules — grounded in the regulations with linked sources.
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <Message
              key={i}
              m={m}
              index={i}
              active={i === activeIndex}
              onSelect={selectMessage}
              onCite={goToCitation}
            />
          ))}
          {loading && !streaming && <LoadingBubble />}
        </main>

        <footer className="composer-bar">
          <Composer onSend={send} disabled={loading} />
        </footer>
      </div>

      <Inspector message={activeMsg} activeN={activeN} />
    </div>
  );
}
