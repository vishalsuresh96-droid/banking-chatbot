const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const messages = document.querySelector("#messages");
const button = document.querySelector("#send-button");

function addMessage(text, kind, extra = "") {
  const message = document.createElement("article");
  message.className = `message ${kind}`;
  message.textContent = text;
  messages.append(message);
  if (extra) {
    const sources = document.createElement("div");
    sources.className = "sources";
    sources.textContent = extra;
    messages.append(sources);
  }
  messages.scrollTop = messages.scrollHeight;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;
  addMessage(question, "user");
  input.value = "";
  button.disabled = true;
  button.textContent = "Thinking...";
  try {
    const response = await fetch("/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "The server could not answer this question.");
    const sourceNames = data.sources.map((source) => `${source.section}: ${source.question}`).join(" • ");
    addMessage(data.answer, "bot", `Retrieved FAQ sources: ${sourceNames}`);
  } catch (error) {
    addMessage(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Send";
    input.focus();
  }
});
