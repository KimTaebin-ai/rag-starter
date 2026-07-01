import { Composer } from "./components/Composer";
import { LoadingBubble } from "./components/LoadingBubble";
import { Message } from "./components/Message";
import { useChat } from "./hooks/useChat";

export default function App() {
  const { messages, loading, totals, send } = useChat();

  return (
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
          <Message key={i} m={m} />
        ))}
        {loading && <LoadingBubble />}
      </main>

      <footer className="composer-bar">
        <Composer onSend={send} disabled={loading} />
      </footer>
    </div>
  );
}
