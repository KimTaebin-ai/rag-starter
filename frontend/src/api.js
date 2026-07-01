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
