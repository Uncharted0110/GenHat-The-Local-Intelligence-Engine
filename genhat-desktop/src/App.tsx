import { useState, useEffect } from "react";
import { invoke } from "@tauri-apps/api/core";

interface ModelFile {
  name: string;
  path: string;
}

function App() {
  const [models, setModels] = useState<ModelFile[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    invoke<ModelFile[]>("list_models")
      .then((list) => {
        setModels(list);
        if (list.length > 0) {
          // Default to first found
          // Note: Backend auto-starts the default hardcoded one or first found, 
          // but we should sync frontend state.
          // Ideally we ask backend "what is running?", but for now just picking the first 
          // matches the fallback logic fairly well.
          setSelectedModel(list[0].path);
        }
      })
      .catch(console.error);
  }, []);

  const handleModelChange = async (path: string) => {
    try {
      setSelectedModel(path);
      await invoke("switch_model", { modelPath: path });
      setResponse(""); 
      alert(`Switched to model: ${path}`);
    } catch (err) {
      console.error(err);
      alert("Failed to switch model");
    }
  };

  const sendPrompt = async () => {
  setResponse("");

  try {
    const res = await fetch("http://127.0.0.1:8081/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        max_tokens: 200,
        stream: true,
      }),
    });

    if (!res.body) throw new Error("No response body");

    const reader = res.body.getReader();
    const decoder = new TextDecoder();

    let buffer = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.startsWith("data:")) continue;

        const payload = line.replace("data:", "").trim();
        if (payload === "[DONE]") return;

        try {
          const json = JSON.parse(payload);
          if (json.content) {
            setResponse(prev => prev + json.content);
          }
        } catch {
        
        }
      }
    }
  } catch (err) {
    console.error(err);
    setResponse("Streaming error");
  }
};


  return (
    <div style={{ padding: 20 }}>
      <h1>GenHat LLM</h1>

      <div style={{ marginBottom: 20 }}>
        <label htmlFor="model-select">Model: </label>
        <select
          id="model-select"
          value={selectedModel}
          onChange={(e) => handleModelChange(e.target.value)}
          disabled={loading || models.length === 0}
        >
          {models.map((m) => (
            <option key={m.path} value={m.path}>
              {m.name}
            </option>
          ))}
        </select>
      </div>

      <textarea
        rows={4}
        cols={50}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder="Type your prompt here..."
      />

      <br />

      <button onClick={sendPrompt} disabled={loading}>
        {loading ? "Generating..." : "Send"}
      </button>

      <pre style={{ whiteSpace: "pre-wrap" }}>{response}</pre>
    </div>
  );
}

export default App;
