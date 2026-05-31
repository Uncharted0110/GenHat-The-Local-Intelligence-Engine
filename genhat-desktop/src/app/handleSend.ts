import type {
  Dispatch,
  MutableRefObject,
  SetStateAction,
} from "react";
import { listen } from "@tauri-apps/api/event";
import { Api } from "../api";
import type {
  ChatMessage,
  ChatMode,
  ChatSession,
  ChatContextUsage,
  DirectDocumentAttachment,
  IngestionStatus,
  KittenTtsVoice,
  MindMapGraph,
  WebSearchResult,
} from "../types";
import { extractTaskText, parseMindMapGraph } from "./mindmapUtils";
import { deriveTitleFromMessage } from "./sessionUtils";
import {
  applyCompactionResultToSession,
  CONTEXT_COMPACTION_KEEP_RECENT,
  CONTEXT_COMPACTION_THRESHOLD,
  resolveReservedOutputTokens,
  toContextMessages,
} from "./contextCompaction";

export interface MindmapOverlayState {
  sessionId: string;
  mindmapId: string | null;
  isGenerating?: boolean;
  query?: string;
}

interface GenerationOptions {
  maxTokens?: number;
  temperature?: number;
  topP?: number;
  topK?: number;
  repeatPenalty?: number;
}

type UpdateSessionFn = (
  sessionId: string,
  patch: Partial<ChatSession> | ((prev: ChatSession) => Partial<ChatSession>)
) => void;

export interface SendHandlerContext {
  activeSessionId: string;
  sessions: ChatSession[];
  chatMode: ChatMode;
  ragEnabled: boolean;
  webEnabled: boolean;
  webDepth: "snippets" | "full";
  imagePath: string | null;
  directDocumentPaths: string[];
  ragDocs: IngestionStatus[];
  selectedModel: string;
  selectedVisionModel: string;
  selectedTtsEngine: string;
  ttsVoice: KittenTtsVoice;
  ttsSpeed: number;
  thinkingEnabled: boolean;
  abortControllersRef: MutableRefObject<Map<string, AbortController>>;
  visionUnlistenRef: MutableRefObject<(() => void) | null>;
  generalIntervalRef: MutableRefObject<ReturnType<typeof setInterval> | null>;
  ttsIntervalRef: MutableRefObject<ReturnType<typeof setInterval> | null>;
  updateSession: UpdateSessionFn;
  setActiveMindmapOverlay: Dispatch<SetStateAction<MindmapOverlayState | null>>;
  setGeneralGenerating: Dispatch<SetStateAction<boolean>>;
  setGeneralElapsedTime: Dispatch<SetStateAction<number>>;
  setGeneralGenerationTime: Dispatch<SetStateAction<number | null>>;
  setMindmapsBySession: Dispatch<SetStateAction<Record<string, MindMapGraph[]>>>;
  setStreamingThinking: Dispatch<SetStateAction<string>>;
  setTtsGenerating: Dispatch<SetStateAction<boolean>>;
  setTtsElapsedTime: Dispatch<SetStateAction<number>>;
  setTtsGenerationTime: Dispatch<SetStateAction<number | null>>;
  setContextUsageForSession: (sessionId: string, usage: ChatContextUsage) => void;
  clearImage: () => void;
  clearDirectDocuments: () => void;
  getContextWindowTokens: (modelIdentifier: string | null | undefined) => number;
  getChatGenerationOptions: (modelIdentifier: string | null | undefined) => GenerationOptions;
}

export async function executeHandleSend(
  text: string,
  ctx: SendHandlerContext
): Promise<void> {
  const sid = ctx.activeSessionId;
  const session = ctx.sessions.find((s) => s.id === sid);
  if (!session || session.loading) return;

  const currentVisionImagePath = ctx.chatMode === "vision" ? ctx.imagePath : null;
  const ragDocPaths = ctx.ragDocs.map((doc) => doc.file_path).filter((path) => !!path);
  const promptDocumentPaths =
    ctx.chatMode === "text" && !ctx.ragEnabled
      ? (ctx.directDocumentPaths.length > 0 ? ctx.directDocumentPaths : ragDocPaths)
      : ctx.directDocumentPaths;

  const visionAttachment =
    ctx.chatMode === "vision" && currentVisionImagePath
      ? {
          path: currentVisionImagePath,
          name: currentVisionImagePath.split(/[/\\]/).pop() ?? "image",
        }
      : undefined;

  const directDocAttachments: DirectDocumentAttachment[] | undefined =
    ctx.chatMode === "text" && ctx.directDocumentPaths.length > 0
      ? ctx.directDocumentPaths.map((path) => ({
          path,
          name: path.split(/[/\\]/).pop() ?? "document",
        }))
      : undefined;

  const newMsg: ChatMessage = {
    role: "user",
    content: text,
    ...(visionAttachment ? { visionImage: visionAttachment } : {}),
    ...(directDocAttachments && directDocAttachments.length > 0
      ? { directDocuments: directDocAttachments }
      : {}),
  };

  const isFirstMessage = session.messages.length === 0;
  const titlePatch = isFirstMessage ? { title: deriveTitleFromMessage(text) } : {};

  ctx.updateSession(sid, (prev) => ({
    messages: [...prev.messages, newMsg],
    loading: true,
    streamingContent: "",
    audioOutputs: prev.audioOutputs ?? [],
    cancelled: false,
    ...titlePatch,
  }));

  if (ctx.chatMode === "vision" && currentVisionImagePath) {
    ctx.clearImage();
  }
  if (
    ctx.chatMode === "text" &&
    ctx.directDocumentPaths.length > 0 &&
    directDocAttachments &&
    directDocAttachments.length > 0
  ) {
    ctx.clearDirectDocuments();
  }

  const ctrl = new AbortController();
  ctx.abortControllersRef.current.set(sid, ctrl);
  const generationOptions = ctx.getChatGenerationOptions(ctx.selectedModel);

  try {
    if (ctx.chatMode === "mindmap") {
      try {
        ctx.setActiveMindmapOverlay({
          sessionId: sid,
          mindmapId: null,
          isGenerating: true,
          query: text,
        });
        ctx.setGeneralGenerating(true);
        ctx.setGeneralElapsedTime(0);
        ctx.setGeneralGenerationTime(null);
        const startTime = Date.now();

        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        ctx.generalIntervalRef.current = setInterval(() => {
          const elapsed = Math.floor((Date.now() - startTime) / 100) / 10;
          ctx.setGeneralElapsedTime(elapsed);
        }, 100);

        let generatedFrom: "documents" | "model" = "model";
        let sourceCount = 0;
        let sourceContext = "";

        if (ctx.ragDocs.length > 0) {
          try {
            const setup = await Api.queryRagStream(text);
            ctx.updateSession(sid, { ragResult: { answer: "", sources: setup.sources } });
            if (!setup.no_retrieval && setup.sources.length > 0) {
              generatedFrom = "documents";
              sourceCount = setup.sources.length;
              sourceContext = setup.sources
                .map((source, index) => `Source ${index + 1} (${source.doc_title}):\n${source.text}`)
                .join("\n\n");
            }
          } catch (e) {
            console.warn("Mindmap RAG grounding failed; using model knowledge.", e);
          }
        }

        const prompt = generatedFrom === "documents"
          ? [
              `User query: ${text}`,
              "Build a concise mindmap grounded ONLY in the provided sources.",
              "Return ONLY valid JSON and no markdown/code fences.",
              "Schema:",
              '{"title":"string","root":{"label":"string","children":[{"label":"string","children":[...]}]}}',
              "Rules:",
              "- 3 to 6 first-level branches.",
              "- Keep labels short (2 to 8 words).",
              "- Depth max 3.",
              "- Do not invent unsupported facts.",
              "Sources:",
              sourceContext,
            ].join("\n")
          : [
              `User query: ${text}`,
              "Create a concise conceptual mindmap from your own knowledge.",
              "Return ONLY valid JSON and no markdown/code fences.",
              "Schema:",
              '{"title":"string","root":{"label":"string","children":[{"label":"string","children":[...]}]}}',
              "Rules:",
              "- 3 to 6 first-level branches.",
              "- Keep labels short (2 to 8 words).",
              "- Depth max 3.",
            ].join("\n");

        let graph: MindMapGraph | undefined;
        let lastError: unknown;

        for (let attempt = 1; attempt <= 3; attempt += 1) {
          try {
            const raw = await Api.routeRequest("mindmap", prompt, ctx.selectedModel || undefined);
            const modelText = extractTaskText(raw);
            graph = parseMindMapGraph(modelText, text, generatedFrom, sourceCount);
            break;
          } catch (e) {
            console.warn(`Mindmap generation attempt ${attempt} failed:`, e);
            lastError = e;
          }
        }

        if (!graph) {
          throw lastError;
        }

        ctx.setMindmapsBySession((prev) => ({
          ...prev,
          [sid]: [...(prev[sid] ?? []), graph],
        }));

        ctx.setActiveMindmapOverlay({
          sessionId: sid,
          mindmapId: graph.id,
          isGenerating: false,
          query: text,
        });

        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        const totalTime = Math.floor((Date.now() - startTime) / 100) / 10;
        ctx.setGeneralGenerating(false);
        ctx.setGeneralElapsedTime(totalTime);
        ctx.setGeneralGenerationTime(totalTime);

        ctx.updateSession(sid, (prev) => ({
          messages: [
            ...prev.messages,
            {
              role: "assistant" as const,
              content:
                generatedFrom === "documents"
                  ? `Mindmap generated from ${sourceCount} retrieved document source${sourceCount === 1 ? "" : "s"}.`
                  : "Mindmap generated from model knowledge.",
              generateTime: totalTime,
            },
          ],
          streamingContent: "",
          loading: false,
        }));
      } catch (e) {
        ctx.setActiveMindmapOverlay(null);
        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        ctx.setGeneralGenerating(false);
        console.error("Mindmap generation failed:", e);
        ctx.updateSession(sid, (prev) => ({
          messages: [
            ...prev.messages,
            {
              role: "assistant" as const,
              content:
                "Mindmap generation failed. The model produced malformed data. Try selecting a larger model or rewording your input.",
            },
          ],
          loading: false,
        }));
      }
      return;
    }

    if (ctx.chatMode === "text" && !ctx.ragEnabled && promptDocumentPaths.length > 0) {
      try {
        ctx.setGeneralGenerating(true);
        ctx.setGeneralElapsedTime(0);
        ctx.setGeneralGenerationTime(null);
        const directStartTime = Date.now();

        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        ctx.generalIntervalRef.current = setInterval(() => {
          const elapsed = Math.floor((Date.now() - directStartTime) / 100) / 10;
          ctx.setGeneralElapsedTime(elapsed);
        }, 100);

        const contextWindowTokens = ctx.getContextWindowTokens(ctx.selectedModel);
        const maxTotalChars = Math.max(
          6_000,
          Math.min(160_000, Math.floor(contextWindowTokens * 2.5))
        );
        const maxCharsPerDocument = Math.max(
          1_500,
          Math.floor(maxTotalChars / Math.max(promptDocumentPaths.length, 1))
        );

        const directSetup = await Api.prepareDirectDocumentPrompt(
          text,
          promptDocumentPaths,
          {
            maxCharsPerDocument,
            maxTotalChars,
          }
        );

        if (directSetup.warnings.length > 0) {
          console.warn("Direct document prompt warnings:", directSetup.warnings);
        }

        const directMessages: ChatMessage[] = [
          {
            role: "system",
            content:
              "You are a helpful assistant. Treat the attached document excerpts in the next message as the primary source of truth. If the answer is not present in those excerpts, explicitly say you don't know.",
          },
          { role: "user", content: directSetup.prompt },
        ];

        try {
          const directCtx = await Api.compactChatContext({
            messages: toContextMessages(directMessages),
            contextWindowTokens,
            reservedOutputTokens: resolveReservedOutputTokens(generationOptions.maxTokens),
            thresholdPercent: CONTEXT_COMPACTION_THRESHOLD,
            allowAutoCompaction: false,
            forceCompaction: false,
            preserveRecentMessages: CONTEXT_COMPACTION_KEEP_RECENT,
            modelOverride: ctx.selectedModel || null,
          });
          ctx.setContextUsageForSession(sid, directCtx.usage);
        } catch (err) {
          console.warn("Failed to analyze direct-document context usage:", err);
        }

        let fullAnswer = "";
        let fullThinking = "";
        let firstTokenTimeMs: number | null = null;

        await Api.streamChat(
          directMessages,
          (chunk) => {
            if (firstTokenTimeMs === null) {
              firstTokenTimeMs = Date.now();
            }
            fullAnswer += chunk;
            ctx.updateSession(sid, (prev) => ({
              streamingContent: prev.streamingContent + chunk,
            }));
          },
          (thinkingChunk) => {
            fullThinking += thinkingChunk;
            ctx.setStreamingThinking((prev) => prev + thinkingChunk);
          },
          () => {
            if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
            const totalTime = Math.floor((Date.now() - directStartTime) / 100) / 10;
            const timeToFirstToken =
              firstTokenTimeMs
                ? Math.floor((firstTokenTimeMs - directStartTime) / 100) / 10
                : null;

            ctx.setGeneralGenerating(false);
            ctx.setGeneralElapsedTime(totalTime);
            ctx.setGeneralGenerationTime(totalTime);
            ctx.setStreamingThinking("");

            if (fullAnswer) {
              ctx.updateSession(sid, (prev) => ({
                messages: [
                  ...prev.messages,
                  {
                    role: "assistant" as const,
                    content: fullAnswer,
                    thinking: fullThinking || undefined,
                    generateTime: totalTime,
                    firstTokenTime:
                      timeToFirstToken !== null ? timeToFirstToken : undefined,
                  },
                ],
                streamingContent: "",
                loading: false,
              }));
            } else {
              ctx.updateSession(sid, { loading: false });
            }
          },
          (err) => {
            console.error("Direct-document stream error:", err);
            ctx.updateSession(sid, (prev) => ({
              messages: [
                ...prev.messages,
                {
                  role: "assistant" as const,
                  content: `Direct document query error: ${err}`,
                },
              ],
              loading: false,
            }));
          },
          undefined,
          ctrl.signal,
          !ctx.thinkingEnabled,
          generationOptions
        );
        return;
      } catch (e) {
        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        ctx.setGeneralGenerating(false);
        console.error("Direct-document attempt failed, falling back to normal chat:", e);
      }
    }

    if (ctx.chatMode === "text" && ctx.ragEnabled && ctx.ragDocs.length > 0) {
      try {
        ctx.setGeneralGenerating(true);
        ctx.setGeneralElapsedTime(0);
        ctx.setGeneralGenerationTime(null);
        const ragStartTime = Date.now();

        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        ctx.generalIntervalRef.current = setInterval(() => {
          const elapsed = Math.floor((Date.now() - ragStartTime) / 100) / 10;
          ctx.setGeneralElapsedTime(elapsed);
        }, 100);

        const setup = await Api.queryRagStream(text);
        ctx.updateSession(sid, { ragResult: { answer: "", sources: setup.sources } });

        if (!setup.prompt || setup.sources.length === 0) {
          // Fall through to plain chat
        } else {
          const ragMessages: ChatMessage[] = [
            {
              role: "system",
              content:
                "You are a helpful assistant. Answer the question using the provided reference text. Write a clear, natural response without repeating source labels, tags, or brackets. If the user asks for a specific format (table, list, bullet points, etc.), use that format. If the reference text does not cover the question, say you don't know.",
            },
            { role: "user", content: setup.prompt },
          ];

          try {
            const ragCtx = await Api.compactChatContext({
              messages: toContextMessages(ragMessages),
              contextWindowTokens: ctx.getContextWindowTokens(ctx.selectedModel),
              reservedOutputTokens: resolveReservedOutputTokens(generationOptions.maxTokens),
              thresholdPercent: CONTEXT_COMPACTION_THRESHOLD,
              allowAutoCompaction: false,
              forceCompaction: false,
              preserveRecentMessages: CONTEXT_COMPACTION_KEEP_RECENT,
              modelOverride: ctx.selectedModel || null,
            });
            ctx.setContextUsageForSession(sid, ragCtx.usage);
          } catch (err) {
            console.warn("Failed to analyze RAG context window usage:", err);
          }

          let fullAnswer = "";
          let fullThinking = "";
          let firstTokenTimeMs: number | null = null;

          await Api.streamChat(
            ragMessages,
            (chunk) => {
              if (firstTokenTimeMs === null) {
                firstTokenTimeMs = Date.now();
              }
              fullAnswer += chunk;
              ctx.updateSession(sid, (prev) => ({
                streamingContent: prev.streamingContent + chunk,
              }));
            },
            (thinkingChunk) => {
              fullThinking += thinkingChunk;
              ctx.setStreamingThinking((prev) => prev + thinkingChunk);
            },
            () => {
              if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
              const totalTime = Math.floor((Date.now() - ragStartTime) / 100) / 10;
              const timeToFirstToken =
                firstTokenTimeMs
                  ? Math.floor((firstTokenTimeMs - ragStartTime) / 100) / 10
                  : null;

              ctx.setGeneralGenerating(false);
              ctx.setGeneralElapsedTime(totalTime);
              ctx.setGeneralGenerationTime(totalTime);
              ctx.setStreamingThinking("");

              ctx.updateSession(sid, (prev) => {
                const updated: ChatMessage[] = [
                  ...prev.messages,
                  {
                    role: "assistant",
                    content: fullAnswer,
                    thinking: fullThinking || undefined,
                    generateTime: totalTime,
                    firstTokenTime:
                      timeToFirstToken !== null ? timeToFirstToken : undefined,
                  },
                ];

                const assistantIdx = updated.length - 1;
                Api.retrieveMediaForResponse(fullAnswer)
                  .then((assets) => {
                    console.log(`Media retrieval: found ${assets.length} assets`);
                    if (assets.length > 0) {
                      ctx.updateSession(sid, (prev2) => ({
                        mediaAssets: {
                          ...prev2.mediaAssets,
                          [assistantIdx]: assets,
                        },
                      }));
                    }
                  })
                  .catch((e) => console.error("Media retrieval failed:", e));

                return {
                  messages: updated,
                  ragResult: prev.ragResult
                    ? { ...prev.ragResult, answer: fullAnswer }
                    : null,
                  streamingContent: "",
                  loading: false,
                };
              });
            },
            (err) => {
              console.error("RAG stream error:", err);
              ctx.updateSession(sid, (prev) => ({
                messages: [
                  ...prev.messages,
                  { role: "assistant" as const, content: `RAG query error: ${err}` },
                ],
                loading: false,
              }));
            },
            setup.llama_port,
            ctrl.signal,
            !ctx.thinkingEnabled,
            generationOptions
          );
          return;
        }
      } catch (e) {
        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        ctx.setGeneralGenerating(false);
        console.error("RAG attempt failed, falling back to normal chat:", e);
      }
    }

    if (ctx.chatMode === "audio" && ctx.selectedTtsEngine) {
      try {
        ctx.setTtsGenerating(true);
        ctx.setTtsElapsedTime(0);
        ctx.setTtsGenerationTime(null);
        const startTime = Date.now();

        if (ctx.ttsIntervalRef.current) clearInterval(ctx.ttsIntervalRef.current);
        ctx.ttsIntervalRef.current = setInterval(() => {
          const elapsed = Math.floor((Date.now() - startTime) / 100) / 10;
          ctx.setTtsElapsedTime(elapsed);
        }, 100);

        const audioUrl = await Api.generateSpeech(text, {
          voice: ctx.ttsVoice,
          speed: ctx.ttsSpeed,
        });

        if (ctx.ttsIntervalRef.current) clearInterval(ctx.ttsIntervalRef.current);
        const totalTime = Math.floor((Date.now() - startTime) / 100) / 10;
        ctx.setTtsGenerating(false);
        ctx.setTtsElapsedTime(totalTime);
        ctx.setTtsGenerationTime(totalTime);

        ctx.updateSession(sid, (prev) => ({
          audioOutputs: [(prev.audioOutputs ?? []), audioUrl].flat(),
          audioOutput: audioUrl,
          messages: [
            ...prev.messages,
            {
              role: "assistant" as const,
              content: `🔊 Audio generated (${ctx.ttsVoice}, ${ctx.ttsSpeed}x speed).`,
              generateTime: totalTime,
              audioUrl,
            },
          ],
        }));
      } catch (e) {
        console.error(e);
        if (ctx.ttsIntervalRef.current) clearInterval(ctx.ttsIntervalRef.current);
        ctx.setTtsGenerating(false);
        ctx.updateSession(sid, (prev) => ({
          messages: [
            ...prev.messages,
            { role: "assistant" as const, content: `Error generating audio: ${e}` },
          ],
        }));
      }
      ctx.updateSession(sid, { loading: false });
      return;
    }

    if (ctx.chatMode === "vision") {
      try {
        ctx.setGeneralGenerating(true);
        ctx.setGeneralElapsedTime(0);
        ctx.setGeneralGenerationTime(null);
        const startTime = Date.now();

        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        ctx.generalIntervalRef.current = setInterval(() => {
          const elapsed = Math.floor((Date.now() - startTime) / 100) / 10;
          ctx.setGeneralElapsedTime(elapsed);
        }, 100);

        ctx.visionUnlistenRef.current?.();
        ctx.visionUnlistenRef.current = null;

        let visionResponse = "";
        let firstTokenTimeMs: number | null = null;

        const unlisten = await listen<{ chunk: string; done: boolean }>(
          "vision-stream",
          (event) => {
            if (event.payload.done) {
              if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
              const totalTime = Math.floor((Date.now() - startTime) / 100) / 10;
              const timeToFirstToken =
                firstTokenTimeMs
                  ? Math.floor((firstTokenTimeMs - startTime) / 100) / 10
                  : null;

              ctx.setGeneralGenerating(false);
              ctx.setGeneralElapsedTime(totalTime);
              ctx.setGeneralGenerationTime(totalTime);

              if (visionResponse) {
                ctx.updateSession(sid, (prev) => ({
                  messages: [
                    ...prev.messages,
                    {
                      role: "assistant" as const,
                      content: visionResponse,
                      generateTime: totalTime,
                      firstTokenTime:
                        timeToFirstToken !== null ? timeToFirstToken : undefined,
                    },
                  ],
                  streamingContent: "",
                  loading: false,
                }));
              } else {
                ctx.updateSession(sid, { loading: false });
              }
              ctx.visionUnlistenRef.current?.();
              ctx.visionUnlistenRef.current = null;
            } else if (event.payload.chunk) {
              if (firstTokenTimeMs === null) {
                firstTokenTimeMs = Date.now();
              }
              visionResponse += event.payload.chunk;
              ctx.updateSession(sid, (prev) => ({
                streamingContent: prev.streamingContent + event.payload.chunk,
              }));
            }
          }
        );

        ctx.visionUnlistenRef.current = unlisten;

        const visionPrompt =
          text ||
          (currentVisionImagePath ? "What's in this image?" : "Hello! Let's chat.");

        await Api.visionChatStream(
          currentVisionImagePath || undefined,
          visionPrompt,
          ctx.selectedVisionModel || undefined
        );
      } catch (e) {
        console.error(e);
        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        ctx.setGeneralGenerating(false);
        ctx.updateSession(sid, (prev) => ({
          messages: [
            ...prev.messages,
            { role: "assistant" as const, content: `Vision error: ${e}` },
          ],
          loading: false,
        }));
        ctx.visionUnlistenRef.current?.();
        ctx.visionUnlistenRef.current = null;
      }
      return;
    }

    // ── Web search context injection ───────────────────────────────────────
    let webSearchResult: WebSearchResult | null = null;
    if (ctx.chatMode === "text" && ctx.webEnabled) {
      try {
        const fetchContent = ctx.webDepth === "full";
        const maxResults = fetchContent ? 2 : 5;
        const result = await Api.webSearch(text.slice(0, 150), maxResults, fetchContent);
        if (result.results.length > 0) {
          webSearchResult = result;
          ctx.updateSession(sid, { webSearchResult: result });
        }
      } catch (e) {
        console.warn("[web_search] Failed, continuing without web context:", e);
      }
    }

    ctx.setGeneralGenerating(true);
    ctx.setGeneralElapsedTime(0);
    ctx.setGeneralGenerationTime(null);
    const chatStartTime = Date.now();

    if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
    ctx.generalIntervalRef.current = setInterval(() => {
      const elapsed = Math.floor((Date.now() - chatStartTime) / 100) / 10;
      ctx.setGeneralElapsedTime(elapsed);
    }, 100);

    let fullResponse = "";
    let fullThinking = "";
    let textFirstTokenTimeMs: number | null = null;

    const sessionMessages = session.messages;
    const fullSessionMessages: ChatMessage[] = [...sessionMessages, newMsg];
    let apiMessages = toContextMessages(fullSessionMessages);

    // Prepend web search context as a system message so the model can cite it
    if (webSearchResult && webSearchResult.formatted_context) {
      apiMessages = [
        { role: "system", content: webSearchResult.formatted_context },
        ...apiMessages,
      ];
    }

    try {
      const compaction = await Api.compactChatContext({
        messages: apiMessages,
        contextWindowTokens: ctx.getContextWindowTokens(ctx.selectedModel),
        reservedOutputTokens: resolveReservedOutputTokens(generationOptions.maxTokens),
        thresholdPercent: CONTEXT_COMPACTION_THRESHOLD,
        allowAutoCompaction: true,
        forceCompaction: false,
        preserveRecentMessages: CONTEXT_COMPACTION_KEEP_RECENT,
        modelOverride: ctx.selectedModel || null,
      });

      ctx.setContextUsageForSession(sid, compaction.usage);
      apiMessages = compaction.messages;

      if (compaction.compacted) {
        const rebuilt = applyCompactionResultToSession(
          fullSessionMessages,
          session.mediaAssets ?? {},
          compaction
        );
        ctx.updateSession(sid, {
          messages: rebuilt.messages,
          mediaAssets: rebuilt.mediaAssets,
        });
      }
    } catch (err) {
      console.warn("Context compaction failed; continuing with original context:", err);
    }

    Api.streamChat(
      apiMessages,
      (chunk) => {
        if (textFirstTokenTimeMs === null) {
          textFirstTokenTimeMs = Date.now();
        }
        ctx.updateSession(sid, (prev) => ({
          streamingContent: prev.streamingContent + chunk,
        }));
        fullResponse += chunk;
      },
      (thinkingChunk) => {
        fullThinking += thinkingChunk;
        ctx.setStreamingThinking((prev) => prev + thinkingChunk);
      },
      () => {
        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        const totalTime = Math.floor((Date.now() - chatStartTime) / 100) / 10;
        const timeToFirstToken =
          textFirstTokenTimeMs
            ? Math.floor((textFirstTokenTimeMs - chatStartTime) / 100) / 10
            : null;

        ctx.setGeneralGenerating(false);
        ctx.setGeneralElapsedTime(totalTime);
        ctx.setGeneralGenerationTime(totalTime);
        ctx.setStreamingThinking("");

        if (fullResponse) {
          ctx.updateSession(sid, (prev) => ({
            messages: [
              ...prev.messages,
              {
                role: "assistant" as const,
                content: fullResponse,
                thinking: fullThinking || undefined,
                generateTime: totalTime,
                firstTokenTime:
                  timeToFirstToken !== null ? timeToFirstToken : undefined,
              },
            ],
            streamingContent: "",
            loading: false,
          }));
        } else {
          ctx.updateSession(sid, { loading: false });
        }
      },
      (err) => {
        if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
        ctx.setGeneralGenerating(false);
        ctx.setStreamingThinking("");
        console.error("Stream error", err);
        ctx.updateSession(sid, (prev) => ({
          messages: [
            ...prev.messages,
            { role: "assistant" as const, content: `Error: ${err}` },
          ],
          loading: false,
        }));
      },
      undefined,
      ctrl.signal,
      !ctx.thinkingEnabled,
      generationOptions
    );
  } catch (err) {
    if (ctx.generalIntervalRef.current) clearInterval(ctx.generalIntervalRef.current);
    ctx.setGeneralGenerating(false);
    console.error(err);
    ctx.updateSession(sid, (prev) => ({
      messages: [
        ...prev.messages,
        { role: "assistant" as const, content: "An unexpected error occurred." },
      ],
      loading: false,
    }));
  }
}
