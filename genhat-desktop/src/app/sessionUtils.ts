import type { ChatMessage, ChatSession } from "../types";

/** Create a fresh, empty ChatSession with a unique ID. */
export function createEmptySession(): ChatSession {
  return {
    id: crypto.randomUUID(),
    title: "New Chat",
    messages: [],
    streamingContent: "",
    loading: false,
    audioOutputs: [],
    cancelled: false,
    ragResult: null,
    webSearchResult: null,
    mediaAssets: {},
    createdAt: Date.now(),
  };
}

/** Derive a short title from the first user message in a session. */
export function deriveTitleFromMessage(text: string): string {
  const trimmed = text.trim().replace(/\n+/g, " ");
  return trimmed.length > 32 ? `${trimmed.slice(0, 32)}…` : trimmed || "New Chat";
}

/** Ensure persisted sessions are safely shaped after loading from localStorage. */
export function normalizeSession(raw: Partial<ChatSession>): ChatSession {
  const messages = Array.isArray(raw.messages)
    ? raw.messages.filter((m): m is ChatMessage =>
      !!m &&
      (m.role === "user" || m.role === "assistant" || m.role === "system") &&
      typeof m.content === "string"
    )
    : [];

  return {
    id: typeof raw.id === "string" && raw.id ? raw.id : crypto.randomUUID(),
    title: typeof raw.title === "string" && raw.title ? raw.title : "New Chat",
    messages,
    streamingContent: "",
    loading: false,
    audioOutputs: Array.isArray(raw.audioOutputs)
      ? raw.audioOutputs
      : (typeof raw.audioOutput === "string" && raw.audioOutput ? [raw.audioOutput] : []),
    cancelled: false,
    ragResult: raw.ragResult ?? null,
    webSearchResult: raw.webSearchResult ?? null,
    mediaAssets: raw.mediaAssets ?? {},
    createdAt: typeof raw.createdAt === "number" ? raw.createdAt : Date.now(),
  };
}
