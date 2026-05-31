import { invoke } from "@tauri-apps/api/core";
import type {
  ChatMessage,
  DiscoveredModelUnit,
  ModelFile,
  RegisteredModel,
  IngestionStatus,
  RagResult,
  RagStreamSetup,
  DirectDocumentPromptSetup,
  MediaAsset,
  PodcastRequest,
  PodcastResult,
  ImportDownloadedModelRequest,
  WorkspaceOpenResult,
  WorkspaceRecord,
  RagModelPreferences,
  ChatContextCompactionRequest,
  ChatContextCompactionResult,
  WebSearchResult,
} from "./types";

export interface HFModel {
  _id: string;
  id: string;
  downloads?: number;
  likes?: number;
  tags?: string[];
  [key: string]: unknown;
}

export interface HFRepoFile {
  type: string;
  oid: string;
  size: number;
  path: string;
  file_name?: string;
  [key: string]: unknown;
}

/** Documented model requirements from README.md */
export interface DocumentedRequirements {
  minRAM?: number;        // GB
  recommendedRAM?: number; // GB
  minVRAM?: number;       // GB
  contextLength?: number;
  source: 'documented' | 'estimated';
  notes?: string;
}

/** Device hardware specifications */
export interface DeviceSpecs {
  total_ram_mb: number;
  available_ram_mb: number;
  total_ram_gb: number;
  available_ram_gb: number;
  cpu_cores: number;
  cpu_has_avx2?: boolean;
  cpu_model: string;
  os: string;
  available_disk_gb: number;
  total_disk_gb: number;
  /** The models directory path being used for disk space calculation */
  models_dir?: string;
}

/** Model compatibility rating */
export type CompatibilityRating =
  | "efficient"
  | "usable"
  | "veryslow"
  | "satisfies"
  | "notrecommended"
  | "wontrun"
  | "unknown";

/** Model tier classification */
export type ModelTier = "tiny" | "small" | "medium" | "large" | "verylarge";

/** Detailed breakdown of compatibility factors */
export interface CompatibilityDetails {
  ram_check: string;
  disk_check: string;
  cpu_check: string;
  performance_notes: string[];
}

/** Compatibility check result */
export interface ModelCompatibility {
  rating: CompatibilityRating;
  reason: string;
  estimated_memory_mb: number;
  available_memory_mb: number;
  can_run: boolean;
  disk_space_sufficient: boolean;
  required_disk_gb: number;
  available_disk_gb: number;
  ram_usage_percent: number;
  disk_usage_percent: number;
  cpu_suitable: boolean;
  details: CompatibilityDetails;
  calculation?: {
    model_params: string;
    quant_level: string;
    base_fp16_size_gb: number;
    quant_multiplier: number;
    estimated_file_size_gb: number;
    actual_file_size_gb: number;
    ram_multiplier: number;
    assumed_context: number;
    required_ram_gb: number;
    total_ram_gb: number;
    available_ram_gb: number;
    ram_decision: "OK" | "NOT_RECOMMENDED" | "DO_NOT_DOWNLOAD" | string;
    cpu_cores: number;
    cpu_has_avx2: boolean;
    cpu_score: number;
    model_factor: number;
    quant_boost: number;
    perf_score: number;
    perf_classification: string;
  };
  alternative?: {
    suggestion: string;
    reason: string;
  };
}

export interface GgufMetadata {
  [key: string]: unknown;
}

export interface PerformanceScore {
  [key: string]: unknown;
}

export const Api = {
  // ── Model Management ───────────────────────────────────────────────────────

  /** List LLM .gguf model files from the LiquidAI-LLM subfolder. */
  async listModels(): Promise<ModelFile[]> {
    return invoke<ModelFile[]>("list_models");
  },

  /** List VLM .gguf model files from the LiquidAI-VLM subfolder. */
  async listVisionModels(): Promise<ModelFile[]> {
    return invoke<ModelFile[]>("list_vision_models");
  },

  /** List all registered models with their current status and supported tasks. */
  async listRegisteredModels(): Promise<RegisteredModel[]> {
    return invoke<RegisteredModel[]>("list_registered_models");
  },

  /** List all models defined in models.toml, including not-yet-downloaded entries. */
  async listModelCatalog(): Promise<RegisteredModel[]> {
    return invoke<RegisteredModel[]>("list_model_catalog");
  },

  /** Scan model folders and return discovered repo-container model units. */
  async discoverLocalModelUnits(): Promise<DiscoveredModelUnit[]> {
    return invoke<DiscoveredModelUnit[]>("discover_local_model_units");
  },

  /** Force runtime sync from disk-scanned model units. */
  async syncDiscoveredModels(): Promise<RegisteredModel[]> {
    return invoke<RegisteredModel[]>("sync_discovered_models");
  },

  /** Update runtime params for a registered model. */
  async updateModelParams(
    modelId: string,
    params: Record<string, string>
  ): Promise<RegisteredModel> {
    return invoke<RegisteredModel>("update_model_params", { modelId, params });
  },

  async downloadModel(modelId: string): Promise<void> {
    return invoke<void>("download_model", { modelId });
  },

  async downloadModelCategory(category: "embedding" | "grader" | "classifier"): Promise<number> {
    return invoke<number>("download_model_category", { category });
  },

  async cancelDownload(modelId: string): Promise<void> {
    return invoke<void>("cancel_download", { modelId });
  },

  async uninstallModel(modelId: string): Promise<void> {
    return invoke<void>("uninstall_model", { modelId });
  },

  /** Switch to a different LLM model by registry ID or file path. */
  async switchModel(modelIdentifier: string): Promise<void> {
    await invoke("switch_model", { modelIdentifier });
  },

  /** Stop the currently active LLM server. */
  async stopLlama(): Promise<void> {
    await invoke("stop_llama");
  },

  /** Get the HTTP port of the running llama-server (triggers lazy start). */
  async getLlamaPort(): Promise<number> {
    return invoke<number>("get_llama_port");
  },

  /** Get estimated total memory usage of all loaded models (MB). */
  async getMemoryUsage(): Promise<number> {
    return invoke<number>("get_memory_usage");
  },

  /** Get a workspace identifier (cwd) for scoping local UI persistence. */
  async getWorkspaceScope(): Promise<string> {
    return invoke<string>("get_workspace_scope");
  },

  /** List all known app workspaces. */
  async listWorkspaces(): Promise<WorkspaceRecord[]> {
    return invoke<WorkspaceRecord[]>("list_workspaces");
  },

  /** Get currently active app workspace metadata. */
  async getActiveWorkspace(): Promise<WorkspaceRecord> {
    return invoke<WorkspaceRecord>("get_active_workspace");
  },

  /** Clear the active workspace (shows startup modal on next app load). */
  async clearActiveWorkspace(): Promise<void> {
    return invoke<void>("clear_active_workspace");
  },

  /** Create a new workspace and make it active. */
  async createWorkspace(name?: string): Promise<WorkspaceRecord> {
    return invoke<WorkspaceRecord>("create_workspace", {
      name: name ?? null,
    });
  },

  /** Open an existing workspace by id and make it active. */
  async openWorkspace(workspaceId: string): Promise<WorkspaceRecord> {
    return invoke<WorkspaceRecord>("open_workspace", {
      workspaceId,
    });
  },

  /** Delete a workspace by id; returns the active workspace after deletion. */
  async deleteWorkspace(workspaceId: string): Promise<WorkspaceRecord> {
    return invoke<WorkspaceRecord>("delete_workspace", {
      workspaceId,
    });
  },

  /** Rename a workspace by id; persists in the workspace registry. */
  async renameWorkspace(workspaceId: string, name: string): Promise<WorkspaceRecord> {
    return invoke<WorkspaceRecord>("rename_workspace", {
      workspaceId,
      name,
    });
  },

  /** Attach/update the saved .nela file path for a workspace. */
  async setWorkspaceFile(workspaceId: string, nelaPath: string): Promise<WorkspaceRecord> {
    return invoke<WorkspaceRecord>("set_workspace_file", {
      workspaceId,
      nelaPath,
    });
  },

  /** Read persisted frontend state JSON for the active workspace. */
  async getWorkspaceFrontendState(): Promise<string | null> {
    return invoke<string | null>("get_workspace_frontend_state");
  },

  /** Persist frontend state JSON for the active workspace. */
  async saveWorkspaceFrontendState(frontendStateJson: string): Promise<void> {
    await invoke("save_workspace_frontend_state", {
      frontendStateJson,
    });
  },

  /** Save active workspace to a chosen .nela file path. */
  async saveWorkspaceAsNela(
    nelaPath: string,
    frontendStateJson?: string
  ): Promise<WorkspaceRecord> {
    return invoke<WorkspaceRecord>("save_workspace_as_nela", {
      nelaPath,
      frontendStateJson: frontendStateJson ?? null,
    });
  },

  /** Save active workspace to its already-associated .nela path. */
  async saveWorkspaceNela(frontendStateJson?: string): Promise<WorkspaceRecord> {
    return invoke<WorkspaceRecord>("save_workspace_nela", {
      frontendStateJson: frontendStateJson ?? null,
    });
  },

  /** Open/import a .nela file and make its workspace active. */
  async openWorkspaceNela(
    nelaPath: string,
    name?: string
  ): Promise<WorkspaceOpenResult> {
    return invoke<WorkspaceOpenResult>("open_workspace_nela", {
      nelaPath,
      name: name ?? null,
    });
  },

  /** Get RAG model preferences for a workspace. */
  async getRagModelPreferences(workspaceId: string): Promise<RagModelPreferences> {
    return invoke<RagModelPreferences>("get_rag_model_preferences", {
      workspaceId,
    });
  },

  /** Save RAG model preferences for a workspace. */
  async saveRagModelPreferences(
    workspaceId: string,
    prefs: RagModelPreferences
  ): Promise<void> {
    await invoke("save_rag_model_preferences", {
      workspaceId,
      prefs,
    });
  },

  /** Manually start (pre-warm) a model by ID. */
  async startModel(modelId: string): Promise<string> {
    return invoke<string>("start_model", { modelId });
  },

  /** Stop a model (all instances) by ID. */
  async stopModel(modelId: string): Promise<void> {
    await invoke("stop_model", { modelId });
  },

  /** Get the runtime status of a specific model. */
  async getModelStatus(modelId: string): Promise<unknown> {
    return invoke("get_model_status", { modelId });
  },

  // ── Audio ──────────────────────────────────────────────────────────────────

  /** Generate speech from text using the TTS backend. Returns a playable data URL. */
  async generateSpeech(
    input: string,
    options?: { voice?: string; speed?: number }
  ): Promise<string> {
    // Backend returns a data:audio/wav;base64,… URL directly
    return invoke<string>("generate_speech", {
      input,
      voice: options?.voice ?? null,
      speed: options?.speed ?? null,
    });
  },

  /** Transcribe an audio file to text using Whisper. */
  async transcribeAudio(audioPath: string): Promise<unknown> {
    return invoke("transcribe_audio", { audioPath });
  },

  /**
   * Transcribe audio from base64-encoded WAV data.
   * Used for real-time voice input from the browser's microphone.
   */
  async transcribeAudioBase64(audioBase64: string): Promise<string> {
    return invoke<string>("transcribe_audio_base64", { audioBase64 });
  },

  /** Start recording from the native microphone (bypasses WebView limitations). */
  async startMicRecording(): Promise<void> {
    return invoke<void>("start_mic_recording");
  },

  /** Stop native mic recording and return base64-encoded 16 kHz mono WAV. */
  async stopMicRecording(): Promise<string> {
    return invoke<string>("stop_mic_recording");
  },

  /**
   * Generate a speech chunk for streaming TTS.
   * Returns a base64-encoded WAV audio chunk.
   */
  async generateSpeechChunk(
    text: string,
    options?: { voice?: string; speed?: number }
  ): Promise<string> {
    return invoke<string>("generate_speech_chunk", {
      text,
      voice: options?.voice ?? null,
      speed: options?.speed ?? null,
    });
  },

  // ── Vision ─────────────────────────────────────────────────────────────────

  /** Read an image file and return it as a base64-encoded data URL (for preview). */
  async readImageBase64(path: string): Promise<string> {
    return invoke<string>("read_image_base64", { path });
  },

  /** Send image + prompt to VLM and return full response (non-streaming). */
  async visionChat(imagePath: string, prompt: string): Promise<string> {
    return invoke<string>("vision_chat", { imagePath, prompt });
  },

  /**
   * Start streaming vision chat. Emits "vision-stream" Tauri events.
   * Frontend should `listen("vision-stream", handler)` before calling this.
   */
  async visionChatStream(
    imagePath: string | undefined,
    prompt: string,
    modelId?: string
  ): Promise<void> {
    await invoke("vision_chat_stream", {
      imagePath: imagePath ?? null,
      prompt,
      modelId: modelId || null,
    });
  },

  // ── RAG ────────────────────────────────────────────────────────────────────

  /** Ingest a single document into the RAG knowledge base. */
  async ingestDocument(path: string): Promise<IngestionStatus> {
    return invoke<IngestionStatus>("ingest_document", { path });
  },

  /** Ingest all supported files in a directory. */
  async ingestFolder(path: string): Promise<IngestionStatus[]> {
    return invoke<IngestionStatus[]>("ingest_folder", { path });
  },

  /** Query the RAG pipeline (non-streaming). */
  async queryRag(query: string, topK?: number): Promise<RagResult> {
    return invoke<RagResult>("query_rag", { query, topK });
  },

  /**
   * Search the web via DuckDuckGo (no API key required).
   * @param query - Search query (trimmed to 150 chars by backend).
   * @param maxResults - Number of results to return (1–10).
   * @param fetchContent - If true, fetches full-page markdown via Jina AI Reader.
   */
  async webSearch(
    query: string,
    maxResults: number,
    fetchContent: boolean
  ): Promise<WebSearchResult> {
    return invoke<WebSearchResult>("web_search", {
      query,
      maxResults,
      fetchContent,
    });
  },

  /**
   * Streaming RAG query — retrieves sources immediately, then returns
   * the llama-server port + augmented prompt for frontend SSE streaming.
   */
  async queryRagStream(
    query: string,
    topK?: number
  ): Promise<RagStreamSetup> {
    return invoke<RagStreamSetup>("query_rag_stream", { query, topK });
  },

  /**
   * Build a direct-to-model prompt from attached files, bypassing RAG retrieval.
   */
  async prepareDirectDocumentPrompt(
    query: string,
    filePaths: string[],
    options?: { maxCharsPerDocument?: number; maxTotalChars?: number }
  ): Promise<DirectDocumentPromptSetup> {
    return invoke<DirectDocumentPromptSetup>("prepare_direct_document_prompt", {
      query,
      filePaths,
      maxCharsPerDocument: options?.maxCharsPerDocument ?? null,
      maxTotalChars: options?.maxTotalChars ?? null,
    });
  },

  /** List all ingested documents with their ingestion status. */
  async listRagDocuments(): Promise<IngestionStatus[]> {
    return invoke<IngestionStatus[]>("list_rag_documents");
  },

  /** Delete a document from the knowledge base. */
  async deleteRagDocument(docId: number): Promise<void> {
    await invoke("delete_rag_document", { docId });
  },

  /** Read a file as base64 data URL for the frontend viewer. */
  async readFileBase64(path: string): Promise<string> {
    return invoke<string>("read_file_base64", { path });
  },

  /** Read a text-based file and return its content as a string. */
  async readFileText(path: string): Promise<string> {
    return invoke<string>("read_file_text", { path });
  },

  // ── Podcast ────────────────────────────────────────────────────────────────

  /** Generate a podcast from a RAG query with two-person dialogue + TTS. */
  async generatePodcast(request: PodcastRequest): Promise<PodcastResult> {
    return invoke<PodcastResult>("generate_podcast", { request });
  },

  /** Manually trigger a round of background enrichment. */
  async enrichRagDocuments(batchSize?: number): Promise<number> {
    return invoke<number>("enrich_rag_documents", { batchSize });
  },

  // ── RAPTOR ─────────────────────────────────────────────────────────────────

  /** Build a RAPTOR tree for a specific document (Phase 3). */
  async buildRaptorTree(docId: number): Promise<unknown> {
    return invoke("build_raptor_tree", { docId });
  },

  /** Check if a document has a RAPTOR tree. */
  async hasRaptorTree(docId: number): Promise<boolean> {
    return invoke<boolean>("has_raptor_tree", { docId });
  },

  /** Delete the RAPTOR tree for a document. */
  async deleteRaptorTree(docId: number): Promise<void> {
    await invoke("delete_raptor_tree", { docId });
  },

  /** Query using RAPTOR tree with confidence-aware traversal. */
  async queryRagWithRaptor(
    docId: number,
    query: string,
    topK?: number
  ): Promise<RagResult> {
    return invoke<RagResult>("query_rag_with_raptor", { docId, query, topK });
  },

  /** Streaming RAPTOR query — retrieve + return setup for SSE streaming. */
  async queryRagWithRaptorStream(
    docId: number,
    query: string,
    topK?: number
  ): Promise<RagStreamSetup> {
    return invoke<RagStreamSetup>("query_rag_with_raptor_stream", {
      docId,
      query,
      topK,
    });
  },

  // ── Media Retrieval ────────────────────────────────────────────────────────

  /**
   * Two-phase media retrieval: given the LLM's response text, find images/tables
   * whose captions are semantically similar to the response content.
   * Returns media assets that should be displayed alongside the chat answer.
   */
  async retrieveMediaForResponse(
    responseText: string,
    topK?: number,
    threshold?: number
  ): Promise<MediaAsset[]> {
    return invoke<MediaAsset[]>("retrieve_media_for_response", {
      responseText,
      topK: topK ?? null,
      threshold: threshold ?? null,
    });
  },

  /** Get all media assets for a specific ingested document. */
  async getMediaForDocument(docId: number): Promise<MediaAsset[]> {
    return invoke<MediaAsset[]>("get_media_for_document", { docId });
  },

  /**
   * Convert an absolute file path to a Tauri asset URL for display in an <img> tag.
   * Uses Tauri's convertFileSrc to create a localhost URL the webview can load.
   */
  mediaUrl(filePath: string): string {
    return convertFileSrc(filePath);
  },

  // ── Inference Routing ──────────────────────────────────────────────────────

  /**
   * Route any TaskRequest through the backend TaskRouter.
   * Supports: chat, summarize, mindmap, tts, podcast_script, transcribe,
   * embed, classify, enrich, grade, hyde, vision_chat, and custom tasks.
   */
  async routeRequest(
    taskType: string,
    input: string,
    modelOverride?: string,
    extra?: Record<string, string>
  ): Promise<unknown> {
    return invoke("route_request", {
      taskType,
      input,
      modelOverride: modelOverride || null,
      extra: extra || null,
    });
  },

  /** Analyze or compact chat context using backend token estimation and summarization fallback. */
  async compactChatContext(
    req: ChatContextCompactionRequest
  ): Promise<ChatContextCompactionResult> {
    return invoke<ChatContextCompactionResult>("compact_chat_context", {
      req,
    });
  },

  // ── Streaming Chat (HTTP → llama-server) ───────────────────────────────────

  /**
   * Stream a chat completion via llama-server's OpenAI-compatible SSE endpoint.
   * Fetches the dynamic port from the backend unless one is provided.
   */
  async streamChat(
    messages: ChatMessage[],
    onChunk: (chunk: string) => void,
    onThinking: (thinking: string) => void,
    onFinish: () => void,
    onError: (err: unknown) => void,
    port?: number,
    signal?: AbortSignal,
    disableThinking?: boolean,
    generationOptions?: {
      maxTokens?: number;
      temperature?: number;
      topP?: number;
      topK?: number;
      repeatPenalty?: number;
    }
  ) {
    try {
      const apiMessages = messages.map(({ role, content }) => ({ role, content }));

      const llamaPort =
        port || (await invoke<number>("get_llama_port"));

      const requestBody: Record<string, unknown> = {
        messages: apiMessages,
        stream: true,
        max_tokens: generationOptions?.maxTokens ?? 2048,
        temperature: generationOptions?.temperature ?? 0.7,
        top_p: generationOptions?.topP ?? 0.95,
        top_k: generationOptions?.topK ?? 40,
        repeat_penalty: generationOptions?.repeatPenalty ?? 1.1,
      };

      // Reasoning is OFF by default; callers can enable by passing disableThinking=false.
      // IMPORTANT: When disabling, set ALL THREE:
      //   - reasoning_budget = 0 (disables generation of thinking tokens)
      //   - reasoning_format = "none" (prevents parsing of <think> tags)
      //   - chat_template_kwargs = {"enable_thinking": false} (for Qwen3 models)
      const shouldDisableThinking = disableThinking ?? true;
      if (!shouldDisableThinking) {
        requestBody.reasoning_format = "deepseek";
        requestBody.reasoning_budget = -1; // Unrestricted
        requestBody.chat_template_kwargs = { enable_thinking: true };
      } else {
        requestBody.reasoning_format = "none"; // Prevent <think> tag parsing
        requestBody.reasoning_budget = 0; // Disable thinking generation
        requestBody.chat_template_kwargs = { enable_thinking: false };
      }

      const res = await fetch(
        `http://127.0.0.1:${llamaPort}/v1/chat/completions`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(requestBody),
          signal,
        }
      );

      if (!res.ok) {
        const errBody = await res.text().catch(() => res.statusText);
        throw new Error(
          `LLM server returned ${res.status}: ${errBody}`
        );
      }

      if (!res.body)
        throw new Error("No response body received from local LLM");

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
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
            onFinish();
            return;
          }

          try {
            const parsed = JSON.parse(payload);
            const delta = parsed.choices?.[0]?.delta;
            
            // Handle reasoning/thinking content
            const reasoningContent = delta?.reasoning_content;
            if (reasoningContent) {
              onThinking(reasoningContent);
            }
            
            // Handle regular content
            const content = delta?.content;
            if (content) {
              onChunk(content);
            }
          } catch (e) {
            console.warn("Failed to parse SSE JSON chunk", e);
          }
        }
      }

      onFinish();
    } catch (err) {
      // AbortError means the user cancelled — stop silently without error msg
      if (err instanceof DOMException && err.name === "AbortError") return;
      onError(err);
    }
  },

  // ── Hugging Face & Custom Downloads ────────────────────────────────────────

  /**
   * Invokes the Taurus backend to download an arbitrary file to a specified folder.
   */
  async downloadCustomFile(
    url: string,
    folder: string,
    filename: string,
    options?: { repoId?: string; relativePath?: string }
  ): Promise<void> {
    return invoke<void>("download_custom_file", {
      url,
      folder,
      filename,
      repoId: options?.repoId ?? null,
      relativePath: options?.relativePath ?? null,
    });
  },
  
  /**
   * Checks if a custom downloaded file already exists on disk.
   */
  async checkCustomFileExists(
    folder: string,
    filename: string,
    options?: { repoId?: string; relativePath?: string }
  ): Promise<boolean> {
    return invoke<boolean>("check_custom_file_exists", {
      folder,
      filename,
      repoId: options?.repoId ?? null,
      relativePath: options?.relativePath ?? null,
    });
  },

  /** Import a downloaded GGUF into runtime and persist custom registration. */
  async importDownloadedModel(request: ImportDownloadedModelRequest): Promise<RegisteredModel> {
    return invoke<RegisteredModel>("import_downloaded_model", { req: request });
  },

  async unregisterCustomModel(modelId: string): Promise<void> {
    await invoke("unregister_custom_model", { modelId });
  },

  /**
   * Searches Hugging Face for GGUF models matching a query.
   */
  async searchHuggingFace(query: string): Promise<HFModel[]> {
    const res = await fetch(`https://huggingface.co/api/models?search=${encodeURIComponent(query)}&filter=gguf&limit=20`);
    if (!res.ok) {
      throw new Error(`HF search failed: ${res.statusText}`);
    }
    return res.json();
  },

  /**
   * Gets specific .gguf files inside a single Hugging Face repository.
   */
  async getHuggingFaceRepoFiles(repoId: string): Promise<HFRepoFile[]> {
    const res = await fetch(`https://huggingface.co/api/models/${repoId}/tree/main`);
    if (!res.ok) {
      throw new Error(`HF tree fetch failed: ${res.statusText}`);
    }
    const files: HFRepoFile[] = await res.json();
    return files.filter(f => f.type === "file" && f.path.endsWith(".gguf"));
  },

  /**
   * Try to fetch documented model requirements from README.md
   */
  async fetchModelDocumentation(repoId: string): Promise<DocumentedRequirements> {
    try {
      const res = await fetch(`https://huggingface.co/${repoId}/raw/main/README.md`);
      if (!res.ok) {
        return { source: 'estimated' };
      }
      
      const readme = await res.text();
      const result: DocumentedRequirements = { source: 'estimated' };
      
      // Try to parse various RAM requirement patterns
      // Patterns like: "RAM: 8GB", "Requires 16GB RAM", "Minimum: 8 GB"
      const ramPatterns = [
        /(?:minimum|min|requires?|needs?|ram:?)\s*(?:~|≈)?\s*(\d+(?:\.\d+)?)\s*gb/gi,
        /(\d+(?:\.\d+)?)\s*gb\s+(?:of\s+)?(?:ram|memory)/gi,
        /(?:recommended|rec):?\s*(?:~|≈)?\s*(\d+(?:\.\d+)?)\s*gb/gi
      ];
      
      for (const pattern of ramPatterns) {
        const matches = [...readme.matchAll(pattern)];
        for (const match of matches) {
          const value = parseFloat(match[1]);
          if (value > 0 && value < 1024) { // Sanity check
            if (!result.minRAM || value < result.minRAM) {
              result.minRAM = value;
              result.source = 'documented';
            }
          }
        }
      }
      
      // Try to find recommended RAM
      const recPatterns = [
        /recommended:?\s*(?:~|≈)?\s*(\d+(?:\.\d+)?)\s*gb/gi,
        /suggested:?\s*(?:~|≈)?\s*(\d+(?:\.\d+)?)\s*gb/gi
      ];
      
      for (const pattern of recPatterns) {
        const match = readme.match(pattern);
        if (match) {
          const value = parseFloat(match[1]);
          if (value > 0 && value < 1024) {
            result.recommendedRAM = value;
            result.source = 'documented';
          }
        }
      }
      
      // Try to find context length
      const contextPatterns = [
        /(?:context|ctx)(?:\s+length)?:?\s*(\d+)k?/gi,
        /(\d+)k?\s+(?:context|tokens)/gi
      ];
      
      for (const pattern of contextPatterns) {
        const match = readme.match(pattern);
        if (match) {
          let value = parseInt(match[1]);
          // If it says "8k context", multiply by 1024
          if (readme.toLowerCase().includes(`${value}k`)) {
            value *= 1024;
          }
          if (value >= 512 && value <= 128000) { // Sanity check
            result.contextLength = value;
          }
        }
      }
      
      return result;
    } catch (e) {
      console.error('Failed to fetch model documentation:', e);
      return { source: 'estimated' };
    }
  },

  // ── System Info & Compatibility ─────────────────────────────────────────────

  /** Get device specifications (RAM, CPU, OS, AVX2 support) */
  async getSystemSpecs(): Promise<DeviceSpecs> {
    return invoke<DeviceSpecs>("get_system_specs");
  },

  /** Check if a model is compatible with the current device */
  async checkCompatibility(
    fileSizeMb: number, 
    memoryMb?: number, 
    quantization?: string,
    filename?: string,
    contextLength?: number,
  ): Promise<ModelCompatibility> {
    return invoke<ModelCompatibility>("check_compatibility", {
      fileSizeMb,
      memoryMb: memoryMb ?? null,
      quantization: quantization ?? null,
      filename: filename ?? null,
      contextLength: contextLength ?? null,
    });
  },

  /** Get the model tier classification based on file size */
  async getModelTier(fileSizeMb: number): Promise<ModelTier> {
    return invoke<ModelTier>("get_model_tier", { fileSizeMb });
  },

  /** Estimate memory requirements for a model based on its file size */
  async estimateModelMemory(fileSizeMb: number): Promise<number> {
    return invoke<number>("estimate_model_memory", { fileSizeMb });
  },
  
  /** Detect quantization level from filename */
  async detectQuantization(filename: string): Promise<string> {
    return invoke<string>("detect_quantization", { filename });
  },
  
  /** Detect model parameter size from filename */
  async detectModelParams(filename: string): Promise<string> {
    return invoke<string>("detect_model_params", { filename });
  },

  /** Parse GGUF file and extract metadata (params, quant, context) */
  async parseModelMetadata(modelPath: string): Promise<GgufMetadata> {
    return invoke<GgufMetadata>("parse_model_metadata", { modelPath });
  },

  /** Calculate performance score for a model based on GGUF metadata */
  async calculateModelPerformance(modelPath: string): Promise<PerformanceScore> {
    return invoke<PerformanceScore>("calculate_model_performance", { modelPath });
  },

  /** Enhanced compatibility check with performance scoring */
  async checkCompatibilityWithPerformance(
    modelPath: string | null,
    fileSizeMb: number,
    memoryMb?: number
  ): Promise<ModelCompatibility> {
    return invoke<ModelCompatibility>("check_compatibility_with_performance", {
      modelPath,
      fileSizeMb,
      memoryMb: memoryMb ?? null,
    });
  },

  /** Batch check compatibility for multiple models */
  async batchCheckCompatibility(
    models: Array<[string, number]> // [path, file_size_mb]
  ): Promise<Array<[string, ModelCompatibility]>> {
    return invoke("batch_check_compatibility", { models });
  },
};

// ── Playground / Pipeline commands ─────────────────────────────────────────────
// Imported directly by usePipelineStore to keep dependencies explicit.

import type { Pipeline } from "./app/playgroundTypes";

export async function listPipelines(): Promise<Pipeline[]> {
  return invoke<Pipeline[]>("playground_list_pipelines");
}

export async function loadPipeline(id: string): Promise<Pipeline> {
  return invoke<Pipeline>("playground_load_pipeline", { id });
}

export async function savePipeline(pipeline: Pipeline): Promise<void> {
  return invoke<void>("playground_save_pipeline", { pipeline });
}

export async function deletePipeline(id: string): Promise<void> {
  return invoke<void>("playground_delete_pipeline", { id });
}

export async function runPipeline(pipelineId: string): Promise<string> {
  return invoke<string>("playground_run_pipeline", { pipelineId });
}

export async function cancelPipelineRun(runId: string): Promise<void> {
  return invoke<void>("playground_cancel_run", { runId });
}

export async function storeCredential(key: string, value: string): Promise<void> {
  return invoke<void>("playground_store_credential", { key, value });
}

// ── Watched Paths / Auto-discovery ───────────────────────────────────────────

export async function addWatchedPath(path: string): Promise<void> {
  return invoke<void>("add_watched_path", { path });
}

export async function removeWatchedPath(path: string): Promise<void> {
  return invoke<void>("remove_watched_path", { path });
}

export async function listWatchedPaths(): Promise<import("./types").WatchedPath[]> {
  return invoke<import("./types").WatchedPath[]>("list_watched_paths");
}

export async function triggerScan(): Promise<import("./types").ScanResult> {
  return invoke<import("./types").ScanResult>("trigger_scan");
}

function convertFileSrc(filePath: string): string {
  return new URL(`file://${filePath}`).href;
}

