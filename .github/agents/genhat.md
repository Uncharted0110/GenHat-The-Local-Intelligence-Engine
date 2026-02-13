# GenHat — The Local Intelligence Engine

> Agent reference document. Read this first before making any changes to the repo.

---

IMPORTANT NOTE: This file has to be updated by the agents on every change to the repo. It serves as the single source of truth for how the application works, how it's structured, and how to develop on it. Always keep it up to date with the latest architectural decisions, file structure, and development guidelines. If anything is removed mention it here, if anything is added mention it here. A new agent must be able to get the complete picture of the current state of the project by just reading this MD file.

## 1. Project Overview

GenHat is a **cross-platform desktop application** that runs LLM inference entirely on the user's local machine — no cloud APIs required. It is built with:

- **Tauri v2** (Rust backend + webview frontend)
- **React + TypeScript** (via Vite)
- **llama.cpp** (`llama-server` binary) as the local inference engine
- **GGUF model format** for quantized models

The application spawns a `llama-server` process as a local HTTP server on port `8081` and the frontend communicates with it via the OpenAI-compatible `/v1/chat/completions` endpoint (with SSE streaming).

---

## 2. Repository Structure

```
GenHat-The-Local-Intelligence-Engine/
│
├── .github/agents/
│   └── genhat.md              ← THIS FILE (agent reference)
│
├── README.md                  ← Project README (minimal, needs expanding)
├── .gitignore                 ← Ignores models/ directory
│
├── models/                    ← GGUF model files (gitignored)
│   └── LFM-1.2B-INT8.gguf    ← Default model (Liquid Foundation Model 1.2B INT8)
│   └── tts-chatterbox-q4-k-m/ ← TTS model files (gitignored)
│       ├── s3gen-bf16.gguf
│       ├── t3_cfg-q4_k_m.gguf
│       └── ve_fp32-f16.gguf
│
├── The-Bare/                  ← Standalone Python inference scripts (prototyping)
│   ├── ASR-Inference/         ← (empty) Automatic Speech Recognition placeholder
│   ├── LLM-Inference/
│   │   └── Liquid-infer-INT8.py  ← CLI chat loop using llama-cpp-python
│   └── TTS-inference/         ← Text-to-Speech generation script & build tools
│       ├── aud_test.py        ← Inference script
│       ├── requirements.txt   ← Python deps
│       └── BUILD_INSTRUCTIONS.md ← How to build the TTS binary
│
└── genhat-desktop/            ← THE MAIN APPLICATION (Tauri + React)
    ├── package.json           ← npm deps (React 19, Tauri API, Vite 7)
    ├── vite.config.ts         ← Vite config (React plugin only)
    ├── tsconfig.json          ← TypeScript project references
    ├── index.html             ← Main HTML shell (1519 lines, full UI layout)
    ├── src/                   ← Frontend source (TypeScript/React)
    │   ├── main.tsx           ← React entry point
    │   ├── App.tsx            ← React App component (model selector + simple chat)
    │   ├── App.css            ← Default styles
    │   ├── index.css          ← Global styles
    │   ├── api.ts             ← Backend API layer (llama-server HTTP calls)
    │   ├── renderer.ts        ← Full chat UI logic (2940 lines, main UI orchestrator)
    │   ├── pdfViewer.ts       ← PDF.js-based viewer with text selection
    │   ├── mindmapVisualization.ts ← React Flow mindmap in popup window
    │   └── mindmap/
    │       └── index.html     ← Standalone mindmap visualization page
    │
    └── src-tauri/             ← Rust backend (Tauri)
        ├── Cargo.toml         ← Rust dependencies
        ├── build.rs           ← Tauri build script
        ├── tauri.conf.json    ← Tauri config (window, bundle, resources)
        ├── capabilities/
        │   └── default.json   ← Tauri permissions (core:default only)
        ├── icons/             ← App icons (PNG, ICO, ICNS)
        ├── src/
        │   ├── main.rs        ← MAIN RUST CODE (279 lines) — all backend logic
        │   └── lib.rs         ← Library entry (mobile support stub)
        └── bin/               ← Pre-built binaries (per-OS)
            ├── llama-lin/     ← Linux x86_64: llama-server + shared libs
            ├── llama-mac/     ← macOS ARM64: llama-server + dylibs
            ├── llama-win/     ← Windows x64: llama-server.exe + DLLs
            ├── tts-lin/       ← Linux x86_64: tts-inference (folder)
            ├── tts-mac/       ← macOS ARM64: tts-inference (folder)
            └── tts-win/       ← Windows x64: tts-inference.exe (folder)
```

---

## 3. Architecture & Data Flow

```
┌─────────────────────────────────────────────────┐
│  Tauri Webview (Frontend)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐  │
│  │ App.tsx   │  │renderer.ts│ │ api.ts       │  │
│  │(React)   │  │(vanilla)  │ │(fetch calls) │  │
│  └────┬─────┘  └─────┬────┘  └──────┬───────┘  │
│       │               │              │          │
│       │  invoke()     │              │ HTTP     │
│       ▼               │              ▼          │
│  ┌─────────────┐      │   ┌──────────────────┐  │
│  │ Tauri IPC   │      │   │ localhost:8081   │  │
│  │ (commands)  │      │   │ /v1/chat/        │  │
│  └──────┬──────┘      │   │  completions     │  │
│         │             │   └────────▲─────────┘  │
└─────────┼─────────────┼───────────┼─────────────┘
          │             │           │
          ▼             │           │
┌─────────────────┐     │   ┌───────┴──────────┐
│ Rust Backend    │     │   │ llama-server     │
│ (main.rs)       │─────┘   │ (child process)  │
│                 │         │ Spawned by Rust  │
│ • list_models   │         │ Serves GGUF model│
│ • switch_model  │         └──────────────────┘
│ • stop_llama    │
└─────────────────┘
```

**Communication channels:**
1. **Tauri IPC** (`invoke()`): Frontend ↔ Rust for model management
   - LLM: `list_models`, `switch_model`, `stop_llama`
   - TTS: `list_audio_models`, `generate_speech`
2. **HTTP** (`fetch()`): Frontend → `llama-server` for inference (`/v1/chat/completions`)

---

## 4. Rust Backend (src-tauri/src/main.rs)

### 4.1 State Management
- `AppState` holds a `Mutex<Option<Child>>` — the `llama-server` child process handle
- TTS processes are spawned ephemerally and not stored in state.

### 4.2 Key Functions

| Function | Purpose |
|---|---|
| `get_models_dir()` | Resolves models directory. Filters out TTS models (s3gen, t3*, ve*) for `list_models()` |
| `resolve_llama_exe()` | Finds `llama-server` binary. |
| `resolve_tts_exe()` | Finds `tts-inference` binary inside `bin/<os>/tts-inference/`. |
| `spawn_llama_process(model_path)` | Spawns `llama-server`. |

### 4.3 Tauri Commands (IPC)

| Command | Signature | Notes |
|---|---|---|
| `list_models` | `() -> Vec<ModelFile>` | Returns LLM GGUF models only (filters out TTS files) |
| `list_audio_models` | `() -> Vec<ModelFile>` | Returns available audio models (looks for `s3gen*.gguf`) |
| `switch_model` | `(state, model_path: String) -> Result` | Restarts `llama-server` |
| `stop_llama` | `(state)` | Kills `llama-server` |
| `generate_speech` | `(model_path, input) -> Result<String>` | Spawns `tts-inference` binary. Returns path to generated .wav file |

### 4.4 Startup Behavior
On app launch (`setup` hook):
1. Looks for `LFM-1.2B-INT8.gguf`.
2. Explicitly ignores TTS models (starting with `s3gen`, `t3_`, `ve_`) when auto-selecting a default model.
3. Auto-spawns `llama-server`.

### 4.5 llama-server Parameters
```
--ctx-size 4096  --port 8081  --host 127.0.0.1
-n 256  --temp 0.7  --top-p 0.9  --top-k 40  --repeat-penalty 1.1
```

---

## 5. Frontend Architecture

### 5.1 Two UI Systems (Know This!)

The frontend has **two parallel UI systems** that coexist:

1. **React App** (`App.tsx` + `main.tsx`): Simple model selector + chat prompt textarea. Mounted into `#root`.
2. **Vanilla TS UI** (`renderer.ts` + `index.html`): Full-featured chat interface with sidebars, tabs, PDF viewer, mindmaps, podcasts. Uses direct DOM manipulation. This is the **primary UI**.

The vanilla UI in `index.html` is the complete application layout (sidebars, chat area, popups, landing page). The React app provides a secondary/simpler interface.

### 5.2 Frontend Files

| File | Lines | Purpose |
|---|---|---|
| `App.tsx` | ~200 | React UI: **Now includes Audio Generation**. Model selection + Audio Model selection + Chat + Audio Player. |
| `index.html` | 1519 | Full HTML shell with inline CSS for the main app UI (sidebar, chat, modals, landing page) |
| `renderer.ts` | 2940 | Main UI orchestrator — chat messaging, tab management, file uploads, project save/load, PDF viewing, mindmap generation, podcast UI, message editing/branching |
| `api.ts` | 459 | API abstraction layer. `callLocalLlama()` calls `localhost:8081`. |
| `pdfViewer.ts` | 342 | PDF.js wrapper — loads PDFs, renders pages, toolbar with zoom/navigation, text selection |
| `mindmapVisualization.ts` | 630 | Generates standalone mindmap HTML with React Flow rendered in a popup window with particle background |
| `mindmap/index.html` | 810 | Standalone mindmap page template |

### 5.3 API Layer (api.ts) — Migration Status

| Function | Status | Notes |
|---|---|---|
| `callLocalLlama()` | ✅ Working | Core HTTP call to `localhost:8081/v1/chat/completions`, supports streaming |
| `analyzeChunksWithGemini()` | ✅ Working (adapted) | Now calls local llama instead of Gemini. No RAG retrieval yet |
| `cachePDFs()` | ⚠️ Mocked | Returns mock response. Needs Rust-side PDF parsing |
| `checkCacheStatus()` | ⚠️ Mocked | Always returns `{ ready: true }` |
| `queryPDFs()` | ⚠️ Mocked | Returns empty results |
| `removePDF()` | ⚠️ Mocked | Returns mock success |
| `exportProjectCache()` | ⚠️ Mocked | Returns empty cache |
| `importProjectCache()` | ⚠️ Mocked | Returns mock imported response |
| `generateMindmap()` | ⚠️ Mocked | Returns empty mindmap |
| `podcastFromPrompt()` | ⚠️ Mocked | Returns mock podcast |

### 5.4 External CDN Dependencies (loaded in index.html)
- **PDF.js v3.11.174** — PDF rendering
- **Lucide Icons** — Icon library (latest from unpkg)

---

## 6. Pre-built Binaries (bin/)

Each OS folder contains:
- `llama-server` — the HTTP inference server (the only one we spawn)
- Various other llama.cpp tools (llama-cli, llama-bench, llama-quantize, etc.)
- **Shared libraries** (must be colocated with the executable):
  - Linux: `libggml*.so`, `libllama.so`, `libmtmd.so`, CPU-variant `.so` files
  - Windows: `ggml*.dll`, `llama.dll`, etc.
  - macOS: corresponding dylibs

The `spawn_llama_process()` function sets `current_dir` to the binary folder so the OS linker finds the sibling shared libraries.

---

## 7. Models

- Stored in `<repo>/models/` (gitignored)
- Format: GGUF (llama.cpp's native quantized format)
- Default model: `LFM-1.2B-INT8.gguf` (Liquid Foundation Model, 1.2B params, INT8 quantization)
- Additional models: drop any `.gguf` file into `models/` and it appears in the UI dropdown
- Custom path: set `GENHAT_MODEL_PATH` env var to point to a different directory or file

---

## 8. Commands to Run

### Prerequisites
- **Node.js** (v18+) and **npm**
- **Rust** (stable, 1.77.2+) via `rustup`
- **Tauri v2 CLI**: `npm install -g @tauri-apps/cli` (or use npx)
- **System dependencies** (Linux): `sudo apt install libwebkit2gtk-4.1-dev libgtk-3-dev libayatana-appindicator3-dev librsvg2-dev`

### Development
```bash
cd genhat-desktop

# Install npm dependencies (first time)
npm install

# Run in development mode (starts Vite dev server + Tauri app)
npx tauri dev
```

### Production Build
```bash
cd genhat-desktop
npx tauri build
```
Output will be in `src-tauri/target/release/bundle/`.

### Run The-Bare Python Script (standalone testing)
_Deprecated: The app now uses the compiled binary in `bin/`._
```bash
cd The-Bare/TTS-inference
# See BUILD_INSTRUCTIONS.md for how to build tts-inference binary
```

---

## 9. Configuration Reference

### tauri.conf.json
- `productName`: "GenHat"
- `identifier`: "com.tauri.dev"
- Dev server: `http://localhost:5173` (Vite)
- Frontend dist: `../dist`
- Bundle resources: `bin/llama-lin/*` (change per target OS when cross-compiling)
- Window: 800×600, resizable, not fullscreen

### Cargo.toml
- Edition 2021, Rust 1.77.2+
- Dependencies: `tauri 2.10`, `serde`, `serde_json`, `log`, `tauri-plugin-log`

### package.json
- `react 19.2`, `@tauri-apps/api 2.10.1`
- Vite 7, TypeScript 5.9

---

## 10. Known Issues & Gotchas

1. **"Exec format error"**: The `llama-server` binary architecture must match the host OS. Linux needs ELF x86_64, macOS needs Mach-O arm64, Windows needs PE. The code selects `llama-lin/`, `llama-mac/`, or `llama-win/` via `cfg!()` at compile time.

2. **"Text file busy" build error**: If `llama-server` is running when you rebuild, the build fails because the binary is locked. Kill it first: `pkill -9 llama-server`

3. **Bundle resources**: `tauri.conf.json` resources must point to existing paths at build time. If building on Linux, only `bin/llama-lin/*` should be listed. Update before cross-platform builds.

4. **Two entry points**: Both `index.html` (vanilla renderer.ts) and `App.tsx` (React) render UI. The vanilla system is the full-featured one; `App.tsx` is a simpler interface. Future work should consolidate.

5. **Mocked API functions**: Most PDF/RAG/mindmap/podcast functions in `api.ts` return mock data. These were migrated from a cloud backend and need local implementations.

6. **Log file**: `llama-server` stdout/stderr is logged to `/tmp/genhat-llama-server.log` (Linux/macOS) or `%TEMP%\genhat-llama-server.log` (Windows). Check this for inference debugging.

7. **Port conflict**: `llama-server` binds to port `8081`. If another process uses this port, the server will fail silently. Check the log file.

9. **TTS Architecture**:
    - The TTS engine uses PyInstaller (`--onedir`) to bundle Python + Torch dependencies.
    - Located in `src-tauri/bin/tts-<os>/tts-inference/`.
    - Executable is spawned directly by Rust via `Command::new()`.
    - Requires sibling GGUF models (`s3gen`, `ve`, `t3_cfg`) to be present in `models/` or `models/tts-chatterbox-q4-k-m/`.

---

## 11. Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `GENHAT_MODEL_PATH` | Override models directory (or point to a specific model file) | `<repo>/models/` |
| `RUST_BACKTRACE` | Enable Rust stack traces (`1` or `full`) | Not set |

---

## 12. Development Guidelines for Future Agents

1. **Always kill `llama-server` before rebuilding**: `pkill -9 llama-server`
2. **Test changes in dev mode**: `cd genhat-desktop && npx tauri dev`
3. **Rust code is in one file**: All backend logic is in `src-tauri/src/main.rs` (279 lines). Keep it there until it grows enough to warrant splitting.
4. **Frontend has two UI systems**: Be aware of both `App.tsx` (React) and `renderer.ts` (vanilla DOM). Changes to chat behavior likely go in `renderer.ts`. Model management goes in `App.tsx`.
5. **API mocks**: When implementing a new local feature (PDF parsing, RAG, mindmaps), replace the corresponding mock in `api.ts` with a real implementation — either a Tauri IPC command or a local HTTP endpoint.
6. **Binary compatibility**: When updating `llama-server` binaries, update ALL three OS folders (`llama-lin`, `llama-mac`, `llama-win`) to the same version.
7. **Bundle config**: When building for a specific OS, ensure `tauri.conf.json` `resources` points to the correct OS folder(s). Currently set to all three with `bin/llama-lin/*`, `bin/llama-win/*`, `bin/llama-mac/*`.
8. **The-Bare scripts**: These are standalone prototypes, not used by the desktop app. They can be used for quick testing of model inference outside Tauri.