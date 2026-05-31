export interface ModelFile {
  name: string;
  path: string;
  is_downloaded?: boolean;
  gdrive_id?: string | null;
}

export interface DiscoveredModelUnit {
  key: string;
  category: string;
  repo_id: string;
  container_rel_path: string;
  llm_rel_path: string;
  llm_abs_path: string;
  llm_file_name: string;
  mmproj_rel_path?: string;
  supports_vision: boolean;
}

export interface WorkspaceRecord {
  id: string;
  name: string;
  nela_path: string | null;
  cache_dir: string;
  created_at: number;
  last_opened_at: number;
}

export interface WorkspaceOpenResult {
  workspace: WorkspaceRecord;
  frontend_state_json: string | null;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
  /** Optional image attached to a user message in vision mode. */
  visionImage?: {
    path: string;
    name: string;
  };
  /** Optional files attached directly to a user message (non-RAG document grounding). */
  directDocuments?: DirectDocumentAttachment[];
  generateTime?: number;
  firstTokenTime?: number;
  /** Optional audio output URL for assistant messages (audio mode, podcasts, etc). */
  audioUrl?: string;
  /** Whether this audio is saved in the sidebar (true), unsaved (false), or not applicable (undefined). */
  audioSaved?: boolean;
  /** Optional thinking/reasoning content for assistant messages (from reasoning models). */
  thinking?: string;
}

export interface ChatContextMessage {
  role: ChatMessage["role"];
  content: string;
}

export interface ChatContextUsage {
  contextWindowTokens: number;
  usedTokens: number;
  reservedOutputTokens: number;
  projectedTokens: number;
  remainingTokens: number;
  remainingAfterReserveTokens: number;
  usedPercent: number;
  projectedPercent: number;
  thresholdPercent: number;
}

export interface ChatContextCompactionRequest {
  messages: ChatContextMessage[];
  contextWindowTokens?: number | null;
  reservedOutputTokens?: number | null;
  thresholdPercent?: number | null;
  allowAutoCompaction?: boolean | null;
  forceCompaction?: boolean | null;
  preserveRecentMessages?: number | null;
  modelOverride?: string | null;
}

export interface ChatContextCompactionResult {
  messages: ChatContextMessage[];
  usage: ChatContextUsage;
  compacted: boolean;
  summaryApplied: boolean;
  droppedMessages: number;
  reason: string;
  keptIndices: number[];
  summaryInsertIndex: number | null;
}

export interface RegisteredModel {
  id: string;
  name: string;
  backend?: string;
  tasks: string[];
  status: string;
  instance_count: number;
  memory_mb: number;
  priority: number;
  is_downloaded: boolean;
  model_file?: string;
  gdrive_id?: string | null;
  model_source?: string;
  model_profile?: string | null;
  engine_adapter?: string | null;
  params?: Record<string, string>;
}

export type ImportModelProfile = "llm" | "vlm";

export interface ImportDownloadedModelRequest {
  folder: string;
  filename: string;
  profile: ImportModelProfile;
  display_name?: string;
  mmproj_file?: string;
  engine_adapter?: string;
}

export interface IngestionStatus {
  doc_id: number;
  title: string;
  file_path: string;
  total_chunks: number;
  embedded_chunks: number;
  enriched_chunks: number;
  phase: string;
}

export interface SourceChunk {
  chunk_id: number;
  doc_title: string;
  text: string;
  score: number;
  /** Page/slide provenance from the original document (e.g. "page:3", "slide:2"). */
  page_info?: string;
}

export interface RagResult {
  answer: string;
  sources: SourceChunk[];
}

export interface SearchHit {
  title: string;
  snippet: string;
  url: string;
}

export interface WebSearchResult {
  query: string;
  results: SearchHit[];
  formatted_context: string;
}

export interface RagStreamSetup {
  sources: SourceChunk[];
  prompt: string;
  llama_port: number;
  no_retrieval: boolean;
}

export interface DirectDocumentAttachment {
  path: string;
  name: string;
}

export interface DirectDocumentUsed {
  file_path: string;
  title: string;
  chars_used: number;
  truncated: boolean;
}

export interface DirectDocumentPromptSetup {
  prompt: string;
  documents: DirectDocumentUsed[];
  warnings: string[];
  truncated: boolean;
}

/** A media asset (image or table) extracted from an ingested document. */
export interface MediaAsset {
  id: number;
  doc_id: number;
  /** "image" or "table" */
  asset_type: string;
  /** Absolute path to the extracted PNG file on disk. */
  file_path: string;
  /** Context-aware caption derived from surrounding document text. */
  caption: string;
  /** Source metadata (e.g. "page:3:image:2"). */
  metadata: string;
  caption_hash: string | null;
}

export interface MindMapNode {
  id: string;
  label: string;
  children: MindMapNode[];
}

export interface MindMapGraph {
  id: string;
  title: string;
  query: string;
  generatedFrom: "documents" | "model";
  sourceCount: number;
  root: MindMapNode;
  createdAt: number;
}

// ── Watched Paths / Auto-discovery ───────────────────────────────────────────

export interface WatchedPath {
  id: number;
  workspace_id: string;
  path: string;
  added_at: string;
}

export interface ScanProgress {
  status: string;
  found: number;
  ingested: number;
  skipped: number;
  errors: number;
  done: boolean;
}

export interface ScanResult {
  ingested: number;
  skipped: number;
  errors: number;
  total_files: number;
}

export type ChatMode = "text" | "vision" | "audio" | "rag" | "podcast" | "mindmap" | "playground";

// ── Multi-Chat Session ────────────────────────────────────────────────────────

/** Represents a single, independent chat session (tab). */
export interface ChatSession {
  /** Unique session identifier (UUID). */
  id: string;
  /** Display title for the tab — derived from the first user message. */
  title: string;
  /** All messages in this session. */
  messages: ChatMessage[];
  /** Partial content currently being streamed for this session. */
  streamingContent: string;
  /** Whether this session is waiting for an LLM response. */
  loading: boolean;
  /** Audio data URLs for all TTS outputs in this session. */
  audioOutputs: string[];
  /** (Deprecated) Last TTS output for backward compatibility. */
  audioOutput?: string;
  /** Set to true when user manually cancels generation. */
  cancelled: boolean;
  /** Latest RAG result (sources + answer) for this session. */
  ragResult: RagResult | null;
  /** Latest web search result for this session. */
  webSearchResult: WebSearchResult | null;
  /** Media assets keyed by message index. */
  mediaAssets: Record<number, MediaAsset[]>;
  /** Unix timestamp when this session was created (ms). */
  createdAt: number;
}

/** Available KittenTTS voice names. */
export const KITTEN_TTS_VOICES = [
  "Bella",
  "Jasper",
  "Luna",
  "Bruno",
  "Rosie",
  "Hugo",
  "Kiki",
  "Leo",
] as const;

export type KittenTtsVoice = (typeof KITTEN_TTS_VOICES)[number];

export interface ChatCompletionRequest {
  model?: string;
  messages: ChatMessage[];
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
}

// ── Podcast Types ─────────────────────────────────────────────────────────────

export interface PodcastRequest {
  query: string;
  voice_a: string;
  voice_b: string;
  speaker_a_name: string;
  speaker_b_name: string;
  max_turns: number;
  top_k?: number;
}

export interface PodcastLine {
  speaker: string;
  voice: string;
  text: string;
  index: number;
}

export interface PodcastScript {
  title: string;
  lines: PodcastLine[];
  source_chunks: string[];
}

export interface PodcastSegment {
  line: PodcastLine;
  audio_data_url: string;
}

export interface PodcastResult {
  script: PodcastScript;
  segments: PodcastSegment[];
  combined_audio_data_url: string;
}

export interface PodcastProgress {
  stage: "rag" | "scripting" | "tts" | "merging" | "done";
  detail: string;
  progress: number;
}

/** User preferences for RAG pipeline model selection. */
export interface RagModelPreferences {
  /** Preferred embedding model ID for vector similarity search. */
  embed_model_id: string | null;
  /** Preferred LLM model ID for enrichment and chat tasks. */
  llm_model_id: string | null;
}

