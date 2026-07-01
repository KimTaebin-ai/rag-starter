import { useState } from "react";

// The question input. Owns its own draft text and clears on submit; the parent
// just receives the submitted question via onSend.
export function Composer({ onSend, disabled }) {
  const [input, setInput] = useState("");

  function submit(e) {
    e.preventDefault();
    const question = input;
    if (!question.trim() || disabled) return;
    setInput("");
    onSend(question);
  }

  return (
    <form className="composer" onSubmit={submit}>
      <input
        value={input}
        onChange={(e) => setInput(e.target.value)}
        placeholder="Ask about pilot certs, medical, airspace, operating rules…"
        autoFocus
      />
      <button type="submit" disabled={disabled}>
        Send
      </button>
    </form>
  );
}
