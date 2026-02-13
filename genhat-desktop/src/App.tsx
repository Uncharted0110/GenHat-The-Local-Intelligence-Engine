import { useState, useEffect } from "react";
import { invoke, convertFileSrc } from "@tauri-apps/api/core";

interface ModelFile {
  name: string;
  path: string;
}

function App() {
  const [models, setModels] = useState<ModelFile[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  
  const [audioModels, setAudioModels] = useState<ModelFile[]>([]);
  const [selectedAudioModel, setSelectedAudioModel] = useState("None");
  const [audioOutput, setAudioOutput] = useState("");

  const [prompt, setPrompt] = useState("");
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    invoke<ModelFile[]>("list_models")
      .then((list) => {
        setModels(list);
        if (list.length > 0) {
          setSelectedModel(list[0].path);
        }
      })
      .catch(console.error);

    invoke<ModelFile[]>("list_audio_models")
      .then((list) => {
        setAudioModels(list);
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
    setAudioOutput("");
    setLoading(true);

    try {
      // Audio Mode Check
      if (selectedAudioModel && selectedAudioModel !== "None") {
         try {
           const path = await invoke<string>("generate_speech", {
             modelPath: selectedAudioModel,
             input: prompt,
           });
           setAudioOutput(convertFileSrc(path));
         } catch (e) {
           console.error(e);
           setResponse(`Error generating audio: ${e}`);
         }
         setLoading(false);
         return;
      }

      // Normal LLM Mode
      const res = await fetch("http://127.0.0.1:8081/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: [
            { role: "user", content: prompt }
          ],
          // These parameters are optional if defaults are set in the server,
          // but we can override or ensure them here.
          max_tokens: 256,
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
          if (payload === "[DONE]") {
            setLoading(false);
            return;
          }

          try {
            const json = JSON.parse(payload);
            
            const delta = json.choices?.[0]?.delta;
            if (delta && delta.content) {
              setResponse(prev => prev + delta.content);
            }
          } catch {
            // ignore parse errors
          }
        }
      }
      setLoading(false);
    } catch (err) {
      console.error(err);
      setResponse("Streaming error");
      setLoading(false);
    }
  };


  return (
    <div style={{ padding: 20 }}>
      <h1>GenHat Local Intelligence</h1>

      <div style={{ display: 'flex', gap: '20px', marginBottom: 20 }}>
        <div>
          <label htmlFor="model-select" style={{ display: 'block', marginBottom: '5px' }}>LLM Model:</label>
          <select
            id="model-select"
            value={selectedModel}
            onChange={(e) => handleModelChange(e.target.value)}
            disabled={loading || models.length === 0}
            style={{ width: '200px' }}
          >
            {models.map((m) => (
              <option key={m.path} value={m.path}>
                {m.name}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label htmlFor="audio-select" style={{ display: 'block', marginBottom: '5px' }}>Audio Model:</label>
          <select
            id="audio-select"
            value={selectedAudioModel}
            onChange={(e) => setSelectedAudioModel(e.target.value)}
            disabled={loading}
            style={{ width: '200px' }}
          >
            <option value="None">None (Text Chat)</option>
            {audioModels.map((m) => (
              <option key={m.path} value={m.path}>
                {m.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      <textarea
        rows={4}
        style={{ width: '100%', marginBottom: '10px' }}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        placeholder={selectedAudioModel && selectedAudioModel !== "None" ? "Type text to generate speech..." : "Type your prompt for the LLM..."}
      />

      <br />

      <button onClick={sendPrompt} disabled={loading} style={{ padding: '8px 16px', cursor: 'pointer' }}>
        {loading ? "Processing..." : (selectedAudioModel && selectedAudioModel !== "None" ? "Generate Audio" : "Send to LLM")}
      </button>

      <div style={{ marginTop: 20 }}>
        {audioOutput && (
          <div style={{ marginBottom: 20, padding: 10, border: '1px solid #ccc', borderRadius: 4 }}>
            <p><strong>Generated Audio:</strong></p>
            <audio controls src={audioOutput} autoPlay style={{ width: '100%' }} />
          </div>
        )}
        <pre style={{ whiteSpace: "pre-wrap", background: '#f5f5f5', padding: 10, borderRadius: 4, minHeight: 50 }}>
          {response}
        </pre>
      </div>
    </div>
  );
}

export default App;
