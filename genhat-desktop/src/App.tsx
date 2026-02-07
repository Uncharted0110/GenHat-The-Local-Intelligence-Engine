import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";

function App() {
  const [message, setMessage] = useState<string>("");

  async function callBackend() {
    const msg = await invoke<string>("greet", { name: "GenHat" });
    setMessage(msg);
  }

  return (
    <>
      <h1>GenHat</h1>

      <button onClick={callBackend}>
        Call Rust Backend
      </button>

      {<p>{message}</p>}
    </>
  );
}

export default App;
