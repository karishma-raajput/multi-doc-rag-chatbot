const fileInput = document.getElementById("fileInput");
const fileLabel = document.getElementById("fileLabel");
const uploadBtn = document.getElementById("uploadBtn");
const resetBtn = document.getElementById("resetBtn");
const uploadStatus = document.getElementById("uploadStatus");
const chatWindow = document.getElementById("chatWindow");
const chatCard = document.querySelector(".chat-card");
const queryInput = document.getElementById("queryInput");
const sendBtn = document.getElementById("sendBtn");

function markHasMessages() {
  chatCard.classList.add("has-messages");
}

fileInput.addEventListener("change", () => {
  const files = fileInput.files;
  if (!files.length) {
    fileLabel.textContent = "Choose files to upload";
  } else if (files.length === 1) {
    fileLabel.textContent = files[0].name;
  } else {
    fileLabel.textContent = `${files.length} files selected`;
  }
});

function addMessage(text, type) {
  markHasMessages();
  const div = document.createElement("div");
  div.className = `msg ${type}`;
  div.textContent = text;
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

function addSources(sources) {
  if (!sources || sources.length === 0) return;
  const div = document.createElement("div");
  div.className = "msg sources";
  div.textContent = "Sources: " + sources.join(", ");
  chatWindow.appendChild(div);
  chatWindow.scrollTop = chatWindow.scrollHeight;
}

uploadBtn.addEventListener("click", async () => {
  const files = fileInput.files;
  if (!files.length) {
    uploadStatus.textContent = "Please select at least one file.";
    return;
  }

  const formData = new FormData();
  for (const file of files) {
    formData.append("files", file);
  }

  uploadStatus.textContent = "Indexing documents...";

  try {
    const res = await fetch("/upload", { method: "POST", body: formData });
    const data = await res.json();
    uploadStatus.textContent = data.message || "Upload complete.";
  } catch (err) {
    uploadStatus.textContent = "Error uploading files.";
    console.error(err);
  }
});

resetBtn.addEventListener("click", async () => {
  await fetch("/reset", { method: "POST" });
  uploadStatus.textContent = "Knowledge base cleared.";
  chatWindow.innerHTML = "";
  chatCard.classList.remove("has-messages");
  fileInput.value = "";
  fileLabel.textContent = "Choose files to upload";
});

async function sendQuery() {
  const query = queryInput.value.trim();
  if (!query) return;

  addMessage(query, "user");
  queryInput.value = "";

  addMessage("Thinking...", "bot");
  const thinkingNode = chatWindow.lastChild;

  try {
    const res = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });
    const data = await res.json();

    chatWindow.removeChild(thinkingNode);
    addMessage(data.answer, "bot");
    addSources(data.sources);
  } catch (err) {
    chatWindow.removeChild(thinkingNode);
    addMessage("Error getting response.", "bot");
    console.error(err);
  }
}

sendBtn.addEventListener("click", sendQuery);
queryInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter") sendQuery();
});

// On page load, check if a knowledge base already exists on disk from a
// previous session (so the person doesn't have to re-upload every time).
(async function checkExistingKnowledgeBase() {
  try {
    const res = await fetch("/status");
    const data = await res.json();
    if (data.has_documents) {
      const fileCount = data.sources.length;
      const fileWord = fileCount === 1 ? "file" : "files";
      uploadStatus.textContent =
        `Loaded ${data.total_chunks} chunks from ${fileCount} previously uploaded ${fileWord}. Ask away, or upload more.`;
    }
  } catch (err) {
    console.error("Could not check existing knowledge base:", err);
  }
})();