import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import { listen } from "@tauri-apps/api/event";
import { open } from "@tauri-apps/plugin-dialog";
import { Api } from "./api";
import type {
  ChatContextUsage,
  ChatMode,
  ChatSession,
  ModelFile,
  RegisteredModel,
  IngestionStatus,
  KittenTtsVoice,
  MindMapGraph,
  WorkspaceRecord,
  ImportModelProfile,
} from "./types";
import { KITTEN_TTS_VOICES } from "./types";
import type { DownloadStateMap, StartupModelToastState } from "./app/types";
import {
  MODE_CONFIG,
  SESSION_STORAGE_PREFIX,
  STARTUP_OPTIONAL_DOWNLOAD_KEY,
  STARTUP_MODEL_SELECTOR,
  VIEWABLE_EXTS,
} from "./app/constants";
import {
  findRegisteredModelByIdentifier,
  modelRefBasename,
  normalizeModelRef,
} from "./app/modelUtils";
import {
  normalizeMindmapsStore,
} from "./app/mindmapUtils";
import {
  createEmptySession,
  normalizeSession,
} from "./app/sessionUtils";
import {
  executeHandleSend,
  type MindmapOverlayState,
} from "./app/handleSend";
import {
  createNewWorkspaceAction,
  deleteWorkspaceByIdAction,
  loadRagDocsAction,
  openWorkspaceFromFileAction,
  refreshWorkspaceListOnlyAction,
  refreshWorkspaceRegistryAction,
  renameWorkspaceByIdAction,
  saveWorkspaceAsFileAction,
  saveWorkspaceFileAction,
  switchWorkspaceByIdAction,
} from "./app/workspaceActions";
import ChatHistorySidebar from "./components/ChatHistorySidebar";
import SidebarNav from "./components/SidebarNav";
import { type RuntimeParamsTarget } from "./components/ActiveModelParamsDock";
import type { AppModalKind } from "./components/AppModal";
import AudioSidebar from "./components/AudioSidebar";
import MindmapsSidebar from "./components/MindmapsSidebar";
import PlaygroundSidebar from "./components/PlaygroundSidebar";
import StartupModelToast from "./components/StartupModelToast";
import AppMainContent from "./components/AppMainContent";
import AppDialogsLayer from "./components/AppDialogsLayer";
import AppRightSidebar from "./components/AppRightSidebar";
import { useTour } from "./hooks/useTour";
import {
  applyCompactionResultToSession,
  CONTEXT_COMPACTION_KEEP_RECENT,
  CONTEXT_COMPACTION_THRESHOLD,
  resolveReservedOutputTokens,
  toContextMessages,
} from "./app/contextCompaction";
import "./App.css";

function App() {
  // ── Theme ──────────────────────────────────────────────────────────────────
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    return (localStorage.getItem("nela-theme") as "dark" | "light") ?? "dark";
  });

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("nela-theme", theme);
  }, [theme]);

  const toggleTheme = () => setTheme((t) => (t === "dark" ? "light" : "dark"));

  // ── Model state ────────────────────────────────────────────────────────────
  const [models, setModels] = useState<ModelFile[]>([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [registeredModels, setRegisteredModels] = useState<RegisteredModel[]>([]);
  const [modelCatalog, setModelCatalog] = useState<RegisteredModel[]>([]);
  const [sessionModelParamOverrides, setSessionModelParamOverrides] = useState<Record<string, Record<string, string>>>({});
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [hfModalOpen, setHfModalOpen] = useState(false);
  const [hfModalPreset, setHfModalPreset] = useState<{
    folder: string;
    profile: "none" | ImportModelProfile;
  }>({
    folder: "LLM",
    profile: "llm",
  });
  const [downloadOptionalOnStart] = useState(() => {
    return localStorage.getItem(STARTUP_OPTIONAL_DOWNLOAD_KEY) === "true";
  });
  const [appModal, setAppModal] = useState({
    open: false,
    kind: "info" as AppModalKind,
    title: "",
    message: "",
    confirmLabel: "OK",
    cancelLabel: "Cancel",
    showCancel: false,
  });
  const modalResolveRef = useRef<((value: boolean) => void) | null>(null);

  // ── Tours ─────────────────────────────────────────────────────────────────
  const { startTour, setBindings } = useTour();
  const [toursOpen, setToursOpen] = useState(false);
  const [suppressStartupModal, setSuppressStartupModal] = useState(false);

  useEffect(() => {
    setBindings({
      openSettings: () => setSettingsOpen(true),
      openTours: () => setToursOpen(true),
      openDocPanel: () => setDocPanelOpen(true),
      switchMode: (mode: ChatMode) => {
        setChatMode(mode);
        if (mode !== "vision") {
          setImagePath(null);
          setImagePreview(null);
        }
        if (mode !== "text") {
          setDirectDocumentPaths([]);
        }
        if (mode !== "text" && mode !== "mindmap") {
          setDocPanelOpen(false);
        }
      },
    });
  }, [setBindings]);

  // ── TTS engine state (registered models with TTS task) ─────────────────
  const [ttsEngines, setTtsEngines] = useState<RegisteredModel[]>([]);
  const [selectedTtsEngine, setSelectedTtsEngine] = useState("");
  const [ttsVoice, setTtsVoice] = useState<KittenTtsVoice>("Leo");
  const [ttsSpeed, setTtsSpeed] = useState(1.0);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_ttsGenerating, setTtsGenerating] = useState(false);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_ttsElapsedTime, setTtsElapsedTime] = useState(0);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_ttsGenerationTime, setTtsGenerationTime] = useState<number | null>(null);
  const ttsIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [visionModels, setVisionModels] = useState<RegisteredModel[]>([]);
  const [selectedVisionModel, setSelectedVisionModel] = useState("");

  // ── Response time tracking for all modes ────────────────────────────────────
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_generalElapsedTime, setGeneralElapsedTime] = useState(0);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_generalGenerationTime, setGeneralGenerationTime] = useState<number | null>(null);
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  const [_generalGenerating, setGeneralGenerating] = useState(false);
  const generalIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // ── Multi-session chat state ───────────────────────────────────────────────
  const [chatMode, setChatMode] = useState<ChatMode>("text");
  const [workspaceScope, setWorkspaceScope] = useState<string | null>(null);
  const [workspaces, setWorkspaces] = useState<WorkspaceRecord[]>([]);
  const [activeWorkspace, setActiveWorkspace] = useState<WorkspaceRecord | null>(null);
  const [startupContinueWorkspace, setStartupContinueWorkspace] = useState<WorkspaceRecord | null>(null);
  const [workspaceBusy, setWorkspaceBusy] = useState(false);
  const [modelLoadingStatus, setModelLoadingStatus] = useState<{
    loading: boolean;
    modelId: string;
    message: string;
  }>({ loading: false, modelId: "", message: "" });
  const [sessionStoreReady, setSessionStoreReady] = useState(false);
  const [sessions, setSessions] = useState<ChatSession[]>(() => [createEmptySession()]);
  const [openSessionIds, setOpenSessionIds] = useState<string[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string>("");
  /** AbortControllers keyed by session ID — persists across renders. */
  const abortControllersRef = useRef<Map<string, AbortController>>(new Map());
  
  // ── Thinking/Reasoning state ───────────────────────────────────────────────
  const [thinkingEnabled, setThinkingEnabled] = useState(false);
  const [streamingThinking, setStreamingThinking] = useState<string>("");
  const [contextUsageBySession, setContextUsageBySession] = useState<Record<string, ChatContextUsage>>({});
  const [contextCompacting, setContextCompacting] = useState(false);

  // ── Vision state ───────────────────────────────────────────────────────────
  const [imagePath, setImagePath] = useState<string | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [directDocumentPaths, setDirectDocumentPaths] = useState<string[]>([]);
  const visionUnlistenRef = useRef<(() => void) | null>(null);

  // ── RAG state ──────────────────────────────────────────────────────────────
  const [ragEnabled, setRagEnabled] = useState(false);
  const [ragDocs, setRagDocs] = useState<IngestionStatus[]>([]);
  const [ragIngesting, setRagIngesting] = useState(false);
  const [enrichmentStatus, setEnrichmentStatus] = useState<string | null>(null);

  // ── Web search state ───────────────────────────────────────────────────────
  const [webEnabled, setWebEnabled] = useState(false);
  const [webDepth, setWebDepth] = useState<"snippets" | "full">("snippets");
  const [mindmapsBySession, setMindmapsBySession] = useState<Record<string, MindMapGraph[]>>({});
  const [activeMindmapOverlay, setActiveMindmapOverlay] = useState<MindmapOverlayState | null>(null);

  // ── Right sidebar (Knowledge Base) ─────────────────────────────────────────
  const [docPanelOpen, setDocPanelOpen] = useState(false);
  const [paramsDockOpen, setParamsDockOpen] = useState(false);
  const [modeSwitchNotice, setModeSwitchNotice] = useState<string | null>(null);
  const modeSwitchTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const startupToastTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const startupPresenceNoticeShownRef = useRef(false);
  const legacySessionStorageDisabledRef = useRef(false);
  const sessionQuotaPromptedRef = useRef(false);
  const [startupModelToast, setStartupModelToast] = useState<StartupModelToastState>({
    open: false,
    phase: "info",
    message: "",
    missingIds: [],
    missingNames: [],
    missingSizesMb: [],
    selectedIds: [],
    doneIds: [],
    failedIds: [],
    completed: 0,
    total: 0,
    failed: 0,
  });
  const [startupToastMinimized, setStartupToastMinimized] = useState(false);

  // ── Session accessor helpers ───────────────────────────────────────────────

  /** Get the currently active session object (read-only snapshot). */
  const activeSession = sessions.find((s) => s.id === activeSessionId) ?? null;

  const promptClearSessionStorage = useCallback(() => {
    if (sessionQuotaPromptedRef.current) return;
    if (modalResolveRef.current) return;
    sessionQuotaPromptedRef.current = true;

    modalResolveRef.current = (confirmed) => {
      sessionQuotaPromptedRef.current = false;
      if (!confirmed) return;
      try {
        for (let i = localStorage.length - 1; i >= 0; i -= 1) {
          const key = localStorage.key(i);
          if (!key) continue;
          if (key.startsWith(SESSION_STORAGE_PREFIX)) {
            localStorage.removeItem(key);
          }
        }
        legacySessionStorageDisabledRef.current = false;
        setAppModal({
          open: true,
          kind: "info",
          title: "Session storage cleared",
          message: "Local cached session storage was cleared. You can continue normally.",
          confirmLabel: "OK",
          cancelLabel: "Cancel",
          showCancel: false,
        });
      } catch (err) {
        console.warn("Failed to clear session storage cache:", err);
      }
    };

    setAppModal({
      open: true,
      kind: "confirm",
      title: "Session storage is full",
      message: "Local session cache is full. Do you want to clear cached session storage now?",
      confirmLabel: "Clear storage",
      cancelLabel: "Not now",
      showCancel: true,
    });
  }, []);

  const parseModelParamNumber = useCallback((raw: string | undefined): number | undefined => {
    if (!raw) return undefined;
    const parsed = Number.parseFloat(raw);
    return Number.isFinite(parsed) ? parsed : undefined;
  }, []);

  const getModelParams = useCallback(
    (modelIdentifier: string | null | undefined): Record<string, string> => {
      if (!modelIdentifier) return {};
      const registered = findRegisteredModelByIdentifier(registeredModels, modelIdentifier);
      const persisted = registered?.params ?? {};
      const override =
        sessionModelParamOverrides[registered?.id ?? modelIdentifier] ??
        sessionModelParamOverrides[modelIdentifier] ??
        {};
      return {
        ...persisted,
        ...override,
      };
    },
    [registeredModels, sessionModelParamOverrides]
  );

  const getChatGenerationOptions = useCallback(
    (modelIdentifier: string | null | undefined) => {
      const params = getModelParams(modelIdentifier);
      return {
        maxTokens: parseModelParamNumber(params.max_tokens),
        temperature: parseModelParamNumber(params.temp),
        topP: parseModelParamNumber(params.top_p),
        topK: parseModelParamNumber(params.top_k),
        repeatPenalty: parseModelParamNumber(params.repeat_penalty),
      };
    },
    [getModelParams, parseModelParamNumber]
  );

  const getContextWindowTokens = useCallback(
    (modelIdentifier: string | null | undefined): number => {
      const params = getModelParams(modelIdentifier);
      const ctxSize = parseModelParamNumber(params.ctx_size);
      if (ctxSize === undefined) return 4096;
      return Math.max(1024, Math.min(262_144, Math.round(ctxSize)));
    },
    [getModelParams, parseModelParamNumber]
  );

  const activeRuntimeParamTarget = useMemo<RuntimeParamsTarget | null>(() => {
    const createTarget = (
      identifier: string | null | undefined,
      fallbackBackend: string,
      fallbackName?: string
    ): RuntimeParamsTarget | null => {
      if (!identifier) return null;

      const resolved = findRegisteredModelByIdentifier(registeredModels, identifier);
      const discoveredName =
        fallbackName ||
        models.find((model) => model.path === identifier)?.name ||
        identifier;

      return {
        key: `${chatMode}:${resolved?.id ?? identifier}`,
        identifier,
        displayName: discoveredName.replace(/\s*\(Unregistered\)$/i, ""),
        backend: resolved?.backend ?? fallbackBackend,
        modelFile: resolved?.model_file,
        memoryMb: resolved?.memory_mb,
        params: getModelParams(identifier),
        isRegistered: !!resolved,
      };
    };

    if (chatMode === "text" || chatMode === "mindmap") {
      return createTarget(selectedModel, "LlamaServer");
    }

    if (chatMode === "audio") {
      const selectedEngine = ttsEngines.find((engine) => engine.id === selectedTtsEngine);
      return createTarget(selectedTtsEngine, selectedEngine?.backend ?? "KittenTts", selectedEngine?.name);
    }

    if (chatMode === "vision") {
      const selectedVision = visionModels.find((model) => model.id === selectedVisionModel);
      return createTarget(selectedVisionModel, selectedVision?.backend ?? "LlamaServer", selectedVision?.name);
    }

    return null;
  }, [
    chatMode,
    getModelParams,
    models,
    registeredModels,
    selectedModel,
    selectedTtsEngine,
    selectedVisionModel,
    ttsEngines,
    visionModels,
  ]);

  const lastRuntimeTargetKeyRef = useRef<string | null>(null);
  useEffect(() => {
    if (!activeRuntimeParamTarget) {
      lastRuntimeTargetKeyRef.current = null;
      setParamsDockOpen(false);
      return;
    }

    // Track active target changes, but keep the params dock hidden by default
    // until the user explicitly opens it.
    if (lastRuntimeTargetKeyRef.current !== activeRuntimeParamTarget.key) {
      lastRuntimeTargetKeyRef.current = activeRuntimeParamTarget.key;
    }
  }, [activeRuntimeParamTarget]);

  const handleApplyRuntimeParams = async (nextParams: Record<string, string>) => {
    if (!activeRuntimeParamTarget) return;

    const targetIdentifier = activeRuntimeParamTarget.identifier;
    let resolved = findRegisteredModelByIdentifier(registeredModels, targetIdentifier);

    // Auto-bind path-only models (for manually placed/HF-downloaded GGUF files)
    // by forcing a model switch first, then refreshing the registry.
    if (!resolved && (chatMode === "text" || chatMode === "mindmap" || chatMode === "vision")) {
      try {
        await Api.switchModel(targetIdentifier);
        const refreshed = await refreshModels();
        resolved = findRegisteredModelByIdentifier(refreshed, targetIdentifier);
      } catch (err) {
        console.warn("Failed to auto-bind model before applying params", err);
      }
    }

    if (resolved) {
      await Api.updateModelParams(resolved.id, nextParams);
      setSessionModelParamOverrides((prev) => {
        const next = { ...prev };
        delete next[targetIdentifier];
        delete next[resolved.id];
        return next;
      });

      // Ensure the selected chat model is activated so startup-level params
      // (for example ctx_size / flash_attn / mlock) are effective immediately.
      if (chatMode === "text" || chatMode === "mindmap") {
        await Api.switchModel(resolved.id);
      }

      await refreshModels();

      if (
        (chatMode === "text" || chatMode === "mindmap") &&
        selectedModel === targetIdentifier
      ) {
        setSelectedModel(resolved.id);
      }
      return;
    }

    throw new Error(
      "Could not apply runtime parameters because the selected model is not bound to the runtime registry. Re-select the model and try again."
    );
  };

  /** Immutably update a specific session by ID. */
  const updateSession = useCallback(
    (sessionId: string, patch: Partial<ChatSession> | ((prev: ChatSession) => Partial<ChatSession>)) => {
      setSessions((prev) =>
        prev.map((s) => {
          if (s.id !== sessionId) return s;
          const changes = typeof patch === "function" ? patch(s) : patch;
          return { ...s, ...changes };
        })
      );
    },
    []
  );

  /** Create a new session, add it to the list, and activate it. */
  const addNewSession = useCallback(() => {
    // No active workspace means we keep the app in the "empty window" state.
    if (!activeWorkspace) return;
    const newSession = createEmptySession();
    setSessions((prev) => [...prev, newSession]);
    setOpenSessionIds((prev) => [...prev, newSession.id]);
    setActiveSessionId(newSession.id);
  }, [activeWorkspace]);

  /** Open a chat from history into the top viewer tabs and activate it. */
  const openSessionInViewer = useCallback((sessionId: string) => {
    setOpenSessionIds((prev) => (prev.includes(sessionId) ? prev : [...prev, sessionId]));
    setActiveSessionId(sessionId);
  }, []);

  const openMindmapOverlay = useCallback((sessionId: string, mindmapId: string) => {
    openSessionInViewer(sessionId);
    setActiveMindmapOverlay({ sessionId, mindmapId, isGenerating: false });
  }, [openSessionInViewer]);

  /** Close only the viewer tab (does not delete chat history). */
  const closeViewerTab = useCallback(
    (sessionId: string) => {
      setOpenSessionIds((prev) => {
        if (!prev.includes(sessionId)) return prev;
        const next = prev.filter((id) => id !== sessionId);

        if (activeSessionId === sessionId) {
          if (next.length === 0) {
            setActiveSessionId("");
          } else {
            const closedIdx = prev.findIndex((id) => id === sessionId);
            const nextIdx = Math.min(closedIdx, next.length - 1);
            setActiveSessionId(next[nextIdx]);
          }
        }

        return next;
      });
    },
    [activeSessionId]
  );

  /** Close a session by ID. Allow zero sessions. */
  const closeSession = useCallback(
    (sessionId: string) => {
      // Abort any in-flight request for this session
      abortControllersRef.current.get(sessionId)?.abort();
      abortControllersRef.current.delete(sessionId);
      setContextUsageBySession((prev) => {
        if (!(sessionId in prev)) return prev;
        const next = { ...prev };
        delete next[sessionId];
        return next;
      });
      setMindmapsBySession((prev) => {
        const next = { ...prev };
        delete next[sessionId];
        return next;
      });
      setActiveMindmapOverlay((prev) => (prev?.sessionId === sessionId ? null : prev));

      setSessions((prev) => {
        const remaining = prev.filter((s) => s.id !== sessionId);
        setOpenSessionIds((openPrev) => openPrev.filter((id) => id !== sessionId));
        if (remaining.length === 0) {
          setOpenSessionIds([]);
          setActiveSessionId("");
          return [];
        }
        // If the closed tab was active, activate the nearest neighbor
        if (sessionId === activeSessionId) {
          const closedIdx = prev.findIndex((s) => s.id === sessionId);
          const nextIdx = Math.min(closedIdx, remaining.length - 1);
          setActiveSessionId(remaining[nextIdx].id);
        }
        return remaining;
      });
    },
    [activeSessionId]
  );

  /** Reorder only open viewer tabs (VS Code style). */
  const reorderViewerTabs = useCallback((reordered: ChatSession[]) => {
    setOpenSessionIds(reordered.map((s) => s.id));
  }, []);

  // ── PDF Viewer state ───────────────────────────────────────────────────────
  const [pdfViewerData, setPdfViewerData] = useState<{
    data: string;
    title: string;
  } | null>(null);
  const [pdfLoading, setPdfLoading] = useState(false);

  // ── Document Viewer state (non-PDF formats) ────────────────────────────
  const [docViewerFile, setDocViewerFile] = useState<{
    filePath: string;
    title: string;
  } | null>(null);

  // ── Keyboard shortcut: Ctrl+T to open a new chat ──────────────────────────
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "t") {
        e.preventDefault();
        addNewSession();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [addNewSession]);

  // ── Lifecycle ──────────────────────────────────────────────────────────────

  const buildWorkspaceFrontendState = useCallback(
    (safeActive: string) =>
      JSON.stringify({
        sessions,
        activeSessionId: safeActive,
        openSessionIds,
        mindmapsBySession,
        selectedModel,
        selectedTtsEngine,
        selectedVisionModel,
      }),
    [sessions, openSessionIds, mindmapsBySession, selectedModel, selectedTtsEngine, selectedVisionModel]
  );

  const refreshWorkspaceRegistry = useCallback(async () => {
    await refreshWorkspaceRegistryAction({
      setWorkspaces,
      setActiveWorkspace,
    });
  }, []);

  // ── RAG helpers ────────────────────────────────────────────────────────────
  // NOTE: This must be declared before any hook dependency arrays that reference
  // `loadRagDocs` (e.g. workspace switching), otherwise it hits TDZ at runtime.
  const loadRagDocs = useCallback(async () => {
    await loadRagDocsAction({ setRagDocs });
  }, []);

  useEffect(() => {
    refreshModels();
    loadRagDocs(); // Always load RAG docs on startup
    return () => {
      visionUnlistenRef.current?.();
      visionUnlistenRef.current = null;
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Listen for backend events (model loading, workspace ready)
  useEffect(() => {
    let unlistenModelLoading: (() => void) | null = null;
    let unlistenWorkspaceReady: (() => void) | null = null;

    const setupListeners = async () => {
      // Model loading progress events
      unlistenModelLoading = await listen<{
        model_id: string;
        status: "starting" | "ready" | "error" | "timeout";
        message: string;
      }>("model-loading", (event) => {
        const { status, model_id, message } = event.payload;
        if (status === "starting") {
          setModelLoadingStatus({ loading: true, modelId: model_id, message });
        } else {
          // Clear loading state on ready, error, or timeout
          setModelLoadingStatus({ loading: false, modelId: "", message: "" });
        }
      });

      // Workspace ready events (emitted after RAG pipeline is reloaded)
      unlistenWorkspaceReady = await listen<{
        workspace_id: string;
        status: string;
      }>("workspace-ready", (event) => {
        console.log("Workspace ready:", event.payload.workspace_id);
        // The workspace is now fully initialized - state restoration can proceed
      });
    };

    void setupListeners();

    return () => {
      unlistenModelLoading?.();
      unlistenWorkspaceReady?.();
    };
  }, []);

  // Start from a neutral scope and let startup actions choose the real workspace scope.
  useEffect(() => {
    setSessionStoreReady(false);
    setWorkspaceScope("workspace:none");
  }, []);

  useEffect(() => {
    const initializeApp = async () => {
      try {
        const [all, active] = await Promise.all([
          Api.listWorkspaces(),
          Api.getActiveWorkspace().catch(() => null),
        ]);
        setWorkspaces(all);
        setStartupContinueWorkspace(
          active && all.some((workspace) => workspace.id === active.id) ? active : null
        );
        setActiveWorkspace(null);
      } catch (err) {
        console.warn("Failed to initialize workspace state:", err);
      }
    };
    void initializeApp();
  }, [loadRagDocs]);

  // Restore persisted chat sessions for the active workspace.
  useEffect(() => {
    if (!workspaceScope) return;

    let cancelled = false;

    const applyRawState = (raw: string | null) => {
      if (!raw) {
        const fresh = createEmptySession();
        setSessions([fresh]);
        setOpenSessionIds([fresh.id]);
        setActiveSessionId(fresh.id);
        setMindmapsBySession({});
        setActiveMindmapOverlay(null);
        return;
      }

      const parsed = JSON.parse(raw) as {
        sessions?: Partial<ChatSession>[];
        activeSessionId?: string;
        openSessionIds?: string[];
        mindmapsBySession?: Record<string, unknown>;
        selectedModel?: string;
        selectedTtsEngine?: string;
        selectedVisionModel?: string;
      };
      const loaded = Array.isArray(parsed.sessions)
        ? parsed.sessions.map(normalizeSession)
        : [];
      const restoredMindmaps = normalizeMindmapsStore(parsed.mindmapsBySession);

      if (loaded.length === 0) {
        const fresh = createEmptySession();
        setSessions([fresh]);
        setOpenSessionIds([fresh.id]);
        setActiveSessionId(fresh.id);
        setMindmapsBySession({});
      } else {
        setSessions(loaded);
        const nextActive =
          parsed.activeSessionId && loaded.some((s) => s.id === parsed.activeSessionId)
            ? parsed.activeSessionId
            : loaded[0].id;
        const restoredOpen = Array.isArray(parsed.openSessionIds)
          ? parsed.openSessionIds.filter((id) => loaded.some((s) => s.id === id))
          : [];
        setOpenSessionIds(restoredOpen.length > 0 ? restoredOpen : [nextActive]);
        setActiveSessionId(nextActive);
        setMindmapsBySession(restoredMindmaps);
      }

      // Restore per-workspace model selections
      if (parsed.selectedModel) setSelectedModel(parsed.selectedModel);
      if (parsed.selectedTtsEngine) setSelectedTtsEngine(parsed.selectedTtsEngine);
      if (parsed.selectedVisionModel) setSelectedVisionModel(parsed.selectedVisionModel);

      setActiveMindmapOverlay(null);
    };

    (async () => {
      try {
        // Primary store: workspace backend state blob.
        const backendState = await Api.getWorkspaceFrontendState();
        if (cancelled) return;
        if (backendState) {
          applyRawState(backendState);
          return;
        }

        // Compatibility fallback: legacy localStorage key.
        const storageKey = `${SESSION_STORAGE_PREFIX}${workspaceScope}`;
        const raw = localStorage.getItem(storageKey);
        if (cancelled) return;
        applyRawState(raw);
      } catch (err) {
        console.error("Failed to restore workspace sessions:", err);
        if (cancelled) return;
        const fresh = createEmptySession();
        setSessions([fresh]);
        setOpenSessionIds([fresh.id]);
        setActiveSessionId(fresh.id);
        setMindmapsBySession({});
        setActiveMindmapOverlay(null);
      } finally {
        if (!cancelled) setSessionStoreReady(true);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [workspaceScope]);

  // Persist sessions whenever they change, scoped to the current workspace.
  useEffect(() => {
    if (!workspaceScope || !sessionStoreReady || sessions.length === 0) return;
    if (workspaceScope === "workspace:none") return;

    const safeActive = sessions.some((s) => s.id === activeSessionId)
      ? activeSessionId
      : sessions[0].id;

    const storageKey = `${SESSION_STORAGE_PREFIX}${workspaceScope}`;
    const fullLegacyState = JSON.stringify({
      sessions,
      activeSessionId: safeActive,
      openSessionIds,
      mindmapsBySession,
      selectedModel,
      selectedTtsEngine,
      selectedVisionModel,
    });

    if (!legacySessionStorageDisabledRef.current) {
      try {
        localStorage.setItem(storageKey, fullLegacyState);
      } catch (err) {
        const isQuotaError =
          err instanceof DOMException &&
          (err.name === "QuotaExceededError" || err.name === "NS_ERROR_DOM_QUOTA_REACHED");

        if (isQuotaError) {
          promptClearSessionStorage();
          try {
            for (let i = localStorage.length - 1; i >= 0; i -= 1) {
              const key = localStorage.key(i);
              if (!key) continue;
              if (key.startsWith(SESSION_STORAGE_PREFIX) && key !== storageKey) {
                localStorage.removeItem(key);
              }
            }
            localStorage.setItem(storageKey, fullLegacyState);
          } catch {
            try {
              localStorage.setItem(
                storageKey,
                JSON.stringify({
                  sessions: [],
                  activeSessionId: safeActive,
                  openSessionIds: [safeActive],
                  selectedModel,
                  selectedTtsEngine,
                  selectedVisionModel,
                })
              );
            } catch (retryErr) {
              legacySessionStorageDisabledRef.current = true;
              console.warn("Disabling legacy localStorage session mirror due to quota:", retryErr);
            }
          }
        } else {
          console.warn("Failed to persist legacy localStorage session state:", err);
        }
      }
    }

    // Mirror into active workspace cache for .nela save/open flows.
    void Api.saveWorkspaceFrontendState(
      buildWorkspaceFrontendState(safeActive)
    ).catch((err) => {
      console.warn("Failed to persist workspace frontend state to backend:", err);
    });
  }, [workspaceScope, sessionStoreReady, sessions, activeSessionId, openSessionIds, mindmapsBySession, selectedModel, selectedTtsEngine, selectedVisionModel, buildWorkspaceFrontendState, promptClearSessionStorage]);

  const refreshWorkspaceListOnly = useCallback(async () => {
    await refreshWorkspaceListOnlyAction({ setWorkspaces });
  }, []);

  const switchWorkspaceById = useCallback(async (workspaceId: string) => {
    await switchWorkspaceByIdAction(workspaceId, {
      workspaceBusy,
      setWorkspaceBusy,
      setSessionStoreReady,
      setRagDocs,
      setSessions,
      setOpenSessionIds,
      setActiveSessionId,
      setMindmapsBySession,
      setActiveMindmapOverlay,
      setActiveWorkspace,
      setWorkspaceScope,
      setStartupContinueWorkspace,
      refreshWorkspaceRegistry,
      refreshWorkspaceListOnly,
      loadRagDocs,
    });
  }, [workspaceBusy, refreshWorkspaceRegistry, refreshWorkspaceListOnly, loadRagDocs]);

  const createNewWorkspace = useCallback(async () => {
    await createNewWorkspaceAction({
      workspaceBusy,
      setWorkspaceBusy,
      setSessionStoreReady,
      setRagDocs,
      setSessions,
      setOpenSessionIds,
      setActiveSessionId,
      setMindmapsBySession,
      setActiveMindmapOverlay,
      setActiveWorkspace,
      setWorkspaceScope,
      setStartupContinueWorkspace,
      refreshWorkspaceRegistry,
      refreshWorkspaceListOnly,
      loadRagDocs,
    });
  }, [workspaceBusy, refreshWorkspaceRegistry, refreshWorkspaceListOnly, loadRagDocs]);

  const saveWorkspaceAsFile = useCallback(async () => {
    await saveWorkspaceAsFileAction({
      workspaceBusy,
      activeSession,
      activeWorkspace,
      sessions,
      activeSessionId,
      buildWorkspaceFrontendState,
      setWorkspaceBusy,
      setActiveWorkspace,
      refreshWorkspaceRegistry,
    });
  }, [
    workspaceBusy,
    activeSession,
    activeWorkspace,
    sessions,
    activeSessionId,
    buildWorkspaceFrontendState,
    refreshWorkspaceRegistry,
  ]);

  const saveWorkspaceFile = useCallback(async () => {
    await saveWorkspaceFileAction({
      workspaceBusy,
      activeSession,
      activeWorkspace,
      sessions,
      activeSessionId,
      buildWorkspaceFrontendState,
      setWorkspaceBusy,
      setActiveWorkspace,
      refreshWorkspaceRegistry,
      saveWorkspaceAsFile,
    });
  }, [
    workspaceBusy,
    activeSession,
    activeWorkspace,
    sessions,
    activeSessionId,
    buildWorkspaceFrontendState,
    refreshWorkspaceRegistry,
    saveWorkspaceAsFile,
  ]);

  const openWorkspaceFromFile = useCallback(async () => {
    await openWorkspaceFromFileAction({
      workspaceBusy,
      setWorkspaceBusy,
      setSessionStoreReady,
      setRagDocs,
      setSessions,
      setOpenSessionIds,
      setActiveSessionId,
      setMindmapsBySession,
      setActiveMindmapOverlay,
      setActiveWorkspace,
      setWorkspaceScope,
      setStartupContinueWorkspace,
      refreshWorkspaceRegistry,
      refreshWorkspaceListOnly,
      loadRagDocs,
    });
  }, [workspaceBusy, refreshWorkspaceRegistry, refreshWorkspaceListOnly, loadRagDocs]);

  const renameWorkspaceById = useCallback(
    async (workspaceId: string, newName: string) => {
      await renameWorkspaceByIdAction(workspaceId, newName, {
        workspaceBusy,
        activeWorkspace,
        setWorkspaceBusy,
        refreshWorkspaceRegistry,
        refreshWorkspaceListOnly,
      });
    },
    [workspaceBusy, activeWorkspace, refreshWorkspaceRegistry, refreshWorkspaceListOnly]
  );

  const deleteWorkspaceById = useCallback(
    async (workspaceId: string) => {
      await deleteWorkspaceByIdAction(workspaceId, {
        workspaceBusy,
        setWorkspaceBusy,
        setSessionStoreReady,
        setRagDocs,
        setSessions,
        setOpenSessionIds,
        setActiveSessionId,
        setMindmapsBySession,
        setActiveMindmapOverlay,
        setActiveWorkspace,
        setWorkspaceScope,
        setStartupContinueWorkspace,
        refreshWorkspaceRegistry,
        refreshWorkspaceListOnly,
        loadRagDocs,
      });
    },
    [workspaceBusy, refreshWorkspaceRegistry, refreshWorkspaceListOnly, loadRagDocs]
  );

  // Keep active session aligned with currently open viewer tabs.
  useEffect(() => {
    if (openSessionIds.length === 0) {
      if (activeSessionId) setActiveSessionId("");
      return;
    }

    const isActiveOpen = openSessionIds.includes(activeSessionId);
    if (!isActiveOpen) {
      setActiveSessionId(openSessionIds[0]);
    }
  }, [openSessionIds, activeSessionId]);

  // Keep open viewer tabs valid if chat history changes.
  useEffect(() => {
    if (sessions.length === 0) return;
    setOpenSessionIds((prev) => {
      const valid = prev.filter((id) => sessions.some((s) => s.id === id));
      return valid.length > 0 ? valid : [sessions[0].id];
    });
  }, [sessions]);

  useEffect(() => {
    const valid = new Set(sessions.map((session) => session.id));
    setContextUsageBySession((prev) => {
      let changed = false;
      const next: Record<string, ChatContextUsage> = {};
      Object.entries(prev).forEach(([sessionId, usage]) => {
        if (valid.has(sessionId)) {
          next[sessionId] = usage;
        } else {
          changed = true;
        }
      });
      return changed ? next : prev;
    });
  }, [sessions]);

  useEffect(() => {
    const validSessionIds = new Set(sessions.map((session) => session.id));
    setMindmapsBySession((prev) => {
      const next: Record<string, MindMapGraph[]> = {};
      let changed = false;
      Object.entries(prev).forEach(([sessionId, maps]) => {
        if (validSessionIds.has(sessionId) && maps.length > 0) {
          next[sessionId] = maps;
        } else {
          changed = true;
        }
      });
      return changed ? next : prev;
    });

    setActiveMindmapOverlay((prev) => {
      if (!prev) return prev;
      if (!validSessionIds.has(prev.sessionId)) return null;
      if (!prev.mindmapId) return prev;
      const list = mindmapsBySession[prev.sessionId] ?? [];
      return list.some((map) => map.id === prev.mindmapId) ? prev : null;
    });
  }, [sessions, mindmapsBySession]);

  // Clear one-time mode notice after a short duration.
  useEffect(() => {
    if (!modeSwitchNotice) return;
    if (modeSwitchTimeoutRef.current) clearTimeout(modeSwitchTimeoutRef.current);
    modeSwitchTimeoutRef.current = setTimeout(() => {
      setModeSwitchNotice(null);
      modeSwitchTimeoutRef.current = null;
    }, 1800);
    return () => {
      if (modeSwitchTimeoutRef.current) {
        clearTimeout(modeSwitchTimeoutRef.current);
      }
    };
  }, [modeSwitchNotice]);

  // Never carry mode-switch text into another chat tab.
  useEffect(() => {
    setModeSwitchNotice(null);
  }, [activeSessionId]);

  useEffect(() => {
    if (!activeSession) return;
    if (chatMode !== "text" && chatMode !== "mindmap") return;

    let cancelled = false;

    const analyzeContext = async () => {
      try {
        const generation = getChatGenerationOptions(selectedModel);
        const result = await Api.compactChatContext({
          messages: toContextMessages(activeSession.messages),
          contextWindowTokens: getContextWindowTokens(selectedModel),
          reservedOutputTokens: resolveReservedOutputTokens(generation.maxTokens),
          thresholdPercent: CONTEXT_COMPACTION_THRESHOLD,
          allowAutoCompaction: false,
          forceCompaction: false,
          preserveRecentMessages: CONTEXT_COMPACTION_KEEP_RECENT,
          modelOverride: selectedModel || null,
        });

        if (cancelled) return;
        setContextUsageBySession((prev) => ({
          ...prev,
          [activeSession.id]: result.usage,
        }));
      } catch (err) {
        if (!cancelled) {
          console.warn("Failed to analyze context usage:", err);
        }
      }
    };

    void analyzeContext();

    return () => {
      cancelled = true;
    };
  }, [
    activeSession,
    activeSession?.id,
    activeSession?.messages,
    chatMode,
    selectedModel,
    getChatGenerationOptions,
    getContextWindowTokens,
  ]);

  // On first entry to the chat screen, notify user which optional models are present.
  useEffect(() => {
    if (startupPresenceNoticeShownRef.current) return;
    if (!activeWorkspace || !sessionStoreReady) return;
    if (modelCatalog.length === 0) return;

    const optionalForStartup = modelCatalog.filter((model) =>
      model.tasks.some((task) => STARTUP_MODEL_SELECTOR.tasks.has(task)) ||
      STARTUP_MODEL_SELECTOR.ids.has(model.id)
    );
    if (optionalForStartup.length === 0) return;

    const present = optionalForStartup.filter((model) => model.is_downloaded);
    const missing = optionalForStartup.filter((model) => !model.is_downloaded);
    if (missing.length > 0) {
      setStartupModelToast({
        open: true,
        phase: "prompt",
        message: `${present.length}/${optionalForStartup.length} models are present.`,
        missingIds: missing.map((model) => model.id),
        missingNames: missing.map((model) => model.name),
        missingSizesMb: missing.map((model) => model.memory_mb),
        selectedIds: missing.map((model) => model.id),
        doneIds: [],
        failedIds: [],
        completed: 0,
        total: missing.length,
        failed: 0,
      });
    } else {
      setStartupModelToast({
        open: true,
        phase: "info",
        message: `All ${optionalForStartup.length} models are already present.`,
        missingIds: [],
        missingNames: [],
        missingSizesMb: [],
        selectedIds: [],
        doneIds: [],
        failedIds: [],
        completed: 0,
        total: 0,
        failed: 0,
      });
    }
    startupPresenceNoticeShownRef.current = true;
  }, [activeWorkspace, sessionStoreReady, modelCatalog]);

  useEffect(() => {
    if (!startupModelToast.open) return;
    if (startupModelToast.phase === "prompt" || startupModelToast.phase === "downloading") return;
    if (startupToastTimeoutRef.current) clearTimeout(startupToastTimeoutRef.current);
    startupToastTimeoutRef.current = setTimeout(() => {
      setStartupModelToast((prev) => ({ ...prev, open: false }));
      startupToastTimeoutRef.current = null;
    }, 5000);
    return () => {
      if (startupToastTimeoutRef.current) {
        clearTimeout(startupToastTimeoutRef.current);
      }
    };
  }, [startupModelToast]);

  useEffect(() => {
    if (startupModelToast.phase !== "downloading") {
      setStartupToastMinimized(false);
    }
  }, [startupModelToast.phase]);

  // Reload RAG docs periodically when in text/mindmap modes
  useEffect(() => {
    if (chatMode === "text" || chatMode === "mindmap") loadRagDocs();
  }, [chatMode, loadRagDocs]);

  // Listen for background enrichment progress events
  useEffect(() => {
    let unlisten: (() => void) | null = null;
    listen<{ enriched_this_round: number; status: string }>(
      "rag:enrichment_progress",
      (event) => {
        if (event.payload.status === "in_progress") {
          setEnrichmentStatus(
            `Enriched ${event.payload.enriched_this_round} chunks`
          );
          loadRagDocs();
          setTimeout(() => setEnrichmentStatus(null), 5000);
        }
      }
    ).then((fn) => {
      unlisten = fn;
    });
    return () => {
      unlisten?.();
    };
  }, [loadRagDocs]);

  // Clean up TTS and general timers on unmount
  useEffect(() => {
    const ttsInterval = ttsIntervalRef.current;
    const generalInterval = generalIntervalRef.current;
    return () => {
      if (ttsInterval) clearInterval(ttsInterval);
      if (generalInterval) clearInterval(generalInterval);
    };
  }, []);


  // ── Download state ─────────────────────────────────────────────────────────
  const [downloads, setDownloads] = useState<DownloadStateMap>({});
  const startupCancelRequestedRef = useRef(false);
  const [startupCancellingIds, setStartupCancellingIds] = useState<string[]>([]);
  const [startupCancelledIds, setStartupCancelledIds] = useState<string[]>([]);
  const refreshModelsOnDownloadRef = useRef<() => Promise<RegisteredModel[]>>(async () => []);

  useEffect(() => {
    let unlisten: (() => void) | null = null;
    listen<{ model_id: string; progress: number; status: string; speed_bps?: number }>(
      "model-download-progress",
      (e) => {
        setDownloads((prev) => ({
          ...prev,
          [e.payload.model_id]: {
            progress: e.payload.progress,
            status: e.payload.status,
            speedBps: typeof e.payload.speed_bps === "number" ? e.payload.speed_bps : undefined,
          },
        }));
        if (e.payload.progress >= 100 && e.payload.status === "Complete") {
          setTimeout(() => {
            void refreshModelsOnDownloadRef.current();
          }, 1000);
          setTimeout(() => {
            setDownloads((prev) => {
              const newD = { ...prev };
              delete newD[e.payload.model_id];
              return newD;
            });
          }, 3000);
        }
      }
    ).then((fn) => {
      unlisten = fn;
    });
    return () => {
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(
        STARTUP_OPTIONAL_DOWNLOAD_KEY,
        downloadOptionalOnStart ? "true" : "false"
      );
    } catch (err) {
      console.warn("Failed to persist startup optional-download preference:", err);
    }
  }, [downloadOptionalOnStart]);

  // ── Model helpers ──────────────────────────────────────────────────────────


  const handleDownloadModel = async (modelId: string) => {
    try {
      await Api.downloadModel(modelId);
    } catch (e) {
      console.error("Failed to download model", e);
      showError(`Failed to download model: ${String(e)}`);
    }
  };

  const handleCancelDownload = async (modelId: string) => {
    try {
      await Api.cancelDownload(modelId);
      setDownloads((prev) => {
        const next = { ...prev };
        delete next[modelId];
        return next;
      });
    } catch (e) {
      console.error("Failed to cancel download", e);
      showError(`Failed to cancel download: ${String(e)}`);
    }
  };

  const handleUninstall = async (modelId: string) => {
    try {
      await Api.uninstallModel(modelId);
      setTimeout(refreshModels, 1000);
    } catch (e) {
      console.error("Failed to uninstall model", e);
      showError(`Failed to uninstall model: ${String(e)}`);
    }
  };

  const handleStartupToastDecline = () => {
    setStartupModelToast((prev) => ({
      ...prev,
      open: true,
      phase: "declined",
      message: "You can download these models from Settings or clicking on the drop-down any time.",
    }));
  };

  const handleStartupToastAccept = async () => {
    const ids = startupModelToast.selectedIds;
    if (ids.length === 0) {
      setStartupModelToast((prev) => ({
        ...prev,
        open: true,
        phase: "declined",
        message: "No models selected. You can download models from Settings any time.",
      }));
      return;
    }

    setStartupModelToast((prev) => ({
      ...prev,
      open: true,
      phase: "downloading",
      message: "Starting parallel downloads...",
      total: ids.length,
      doneIds: [],
      failedIds: [],
      completed: 0,
      failed: 0,
    }));
    setStartupToastMinimized(false);
    startupCancelRequestedRef.current = false;
    setStartupCancellingIds([]);
    setStartupCancelledIds([]);

    let completed = 0;
    let failed = 0;
    await Promise.all(
      ids.map(async (modelId) => {
        try {
          await Api.downloadModel(modelId);
          completed += 1;
          setStartupModelToast((prev) => ({
            ...prev,
            doneIds: prev.doneIds.includes(modelId) ? prev.doneIds : [...prev.doneIds, modelId],
          }));
        } catch (e) {
          const message = e instanceof Error ? e.message : String(e);
          const cancelled = /cancel/i.test(message);

          if (cancelled) {
            startupCancelRequestedRef.current = true;
            setStartupCancelledIds((prev) => (prev.includes(modelId) ? prev : [...prev, modelId]));
            setStartupModelToast((prev) => ({
              ...prev,
              failedIds: prev.failedIds.includes(modelId) ? prev.failedIds : [...prev.failedIds, modelId],
            }));
          } else {
            failed += 1;
            console.error("Failed startup model download", e);
            setStartupModelToast((prev) => ({
              ...prev,
              failedIds: prev.failedIds.includes(modelId) ? prev.failedIds : [...prev.failedIds, modelId],
            }));
          }
        } finally {
          setStartupCancellingIds((prev) => prev.filter((id) => id !== modelId));
          setStartupModelToast((prev) => ({
            ...prev,
            completed,
            failed,
          }));
        }
      })
    );

    if (startupCancelRequestedRef.current) {
      setStartupModelToast((prev) => ({
        ...prev,
        open: true,
        phase: "done",
        message: `Download cancelled. ${completed}/${ids.length} completed before cancellation.`,
      }));
    } else if (failed > 0) {
      setStartupModelToast((prev) => ({
        ...prev,
        open: true,
        phase: "done",
        message: `Downloads finished: ${completed}/${ids.length} completed, ${failed} failed. You can retry from Settings.`,
      }));
    } else {
      setStartupModelToast((prev) => ({
        ...prev,
        open: true,
        phase: "done",
        message: `All ${ids.length} model download(s) completed.`,
      }));
    }
  };

  const startupOverallSpeedBps = useMemo(() => {
    if (startupModelToast.phase !== "downloading") return 0;
    return startupModelToast.selectedIds.reduce((sum, modelId) => {
      const speed = downloads[modelId]?.speedBps;
      if (typeof speed !== "number" || !Number.isFinite(speed) || speed <= 0) {
        return sum;
      }
      return sum + speed;
    }, 0);
  }, [startupModelToast.phase, startupModelToast.selectedIds, downloads]);

  const handleStartupToastCancelSingleDownload = async (modelId: string) => {
    if (startupModelToast.phase !== "downloading") return;
    if (startupModelToast.doneIds.includes(modelId) || startupModelToast.failedIds.includes(modelId)) return;

    startupCancelRequestedRef.current = true;
    setStartupCancellingIds((prev) => (prev.includes(modelId) ? prev : [...prev, modelId]));

    try {
      await Api.cancelDownload(modelId);
      setDownloads((prev) => {
        const next = { ...prev };
        delete next[modelId];
        return next;
      });
    } catch (e) {
      setStartupCancellingIds((prev) => prev.filter((id) => id !== modelId));
      console.error("Failed to cancel startup model download", e);
      showError(`Failed to cancel download: ${String(e)}`);
    }
  };

  const handleStartupToastCancelDownloads = async () => {
    if (startupModelToast.phase !== "downloading") return;

    const activeIds = startupModelToast.selectedIds.filter(
      (modelId) =>
        !startupModelToast.doneIds.includes(modelId) &&
        !startupModelToast.failedIds.includes(modelId)
    );

    if (activeIds.length === 0) return;

    startupCancelRequestedRef.current = true;
    setStartupCancellingIds((prev) => Array.from(new Set([...prev, ...activeIds])));
    setStartupModelToast((prev) => ({
      ...prev,
      message: "Cancelling downloads...",
    }));

    await Promise.allSettled(activeIds.map((modelId) => handleStartupToastCancelSingleDownload(modelId)));
  };

  const toggleStartupModelSelection = (modelId: string) => {
    setStartupModelToast((prev) => {
      if (prev.phase !== "prompt") return prev;
      const selected = prev.selectedIds.includes(modelId)
        ? prev.selectedIds.filter((id) => id !== modelId)
        : [...prev.selectedIds, modelId];
      return { ...prev, selectedIds: selected };
    });
  };

  const startupSelectedTotalMb = useMemo(() => {
    let total = 0;
    startupModelToast.selectedIds.forEach((modelId) => {
      const idx = startupModelToast.missingIds.indexOf(modelId);
      if (idx < 0) return;
      const mb = startupModelToast.missingSizesMb[idx];
      if (typeof mb === "number" && Number.isFinite(mb) && mb > 0) total += mb;
    });
    return total;
  }, [startupModelToast.selectedIds, startupModelToast.missingIds, startupModelToast.missingSizesMb]);

  const getOptionalModels = (list: RegisteredModel[]) =>
    list.filter(
      (model) =>
        model.tasks.some((t) => STARTUP_MODEL_SELECTOR.tasks.has(t)) ||
        STARTUP_MODEL_SELECTOR.ids.has(model.id)
    );

  const downloadMissingOptionalModels = async () => {
    const list = registeredModels.length > 0
      ? registeredModels
      : await Api.listRegisteredModels();
    if (registeredModels.length === 0) setRegisteredModels(list);

    const missing = getOptionalModels(list)
      .filter((model) => model.gdrive_id && !model.is_downloaded)
      .filter((model) => !downloads[model.id]);

    for (const model of missing) {
      await handleDownloadModel(model.id);
    }
  };

  const showModal = useCallback((
    kind: AppModalKind,
    title: string,
    message: string,
    options?: { confirmLabel?: string; cancelLabel?: string; showCancel?: boolean }
  ) => {
    setAppModal({
      open: true,
      kind,
      title,
      message,
      confirmLabel: options?.confirmLabel ?? "OK",
      cancelLabel: options?.cancelLabel ?? "Cancel",
      showCancel: options?.showCancel ?? false,
    });
  }, []);

  const showError = useCallback((message: string, title = "Error") => {
    showModal("error", title, message);
  }, [showModal]);

  const confirmAction = (
    title: string,
    message: string,
    confirmLabel = "OK",
    cancelLabel = "Cancel"
  ) =>
    new Promise<boolean>((resolve) => {
      modalResolveRef.current = resolve;
      showModal("confirm", title, message, {
        confirmLabel,
        cancelLabel,
        showCancel: true,
      });
    });

  const handleModalConfirm = () => {
    const resolver = modalResolveRef.current;
    modalResolveRef.current = null;
    if (resolver) resolver(true);
    setAppModal((prev) => ({ ...prev, open: false }));
  };

  const handleModalCancel = () => {
    const resolver = modalResolveRef.current;
    modalResolveRef.current = null;
    if (resolver) resolver(false);
    setAppModal((prev) => ({ ...prev, open: false }));
  };

  const refreshModels = useCallback(async (): Promise<RegisteredModel[]> => {
    try {
      const [list, discoveredUnits, catalog] = await Promise.all([
        Api.listRegisteredModels(),
        Api.discoverLocalModelUnits().catch(() => []),
        Api.listModelCatalog().catch(() => []),
      ]);
      setRegisteredModels(list);
      setModelCatalog(catalog);

      // Vision models
      const vision = list.filter((m) => m.tasks.includes("vision_chat"));
      setVisionModels(vision);
      if (vision.length > 0) {
        setSelectedVisionModel((prev) => prev || vision[0].id);
      }

      // Text models from registry
      const chatModels = list
        .filter((m) => m.tasks.includes("chat"))
        .sort((a, b) => b.priority - a.priority)
        .map((m) => ({
          name:
            m.model_source === "custom"
              ? `${m.name} (Custom${m.model_profile ? ` ${m.model_profile.toUpperCase()}` : ""})`
              : m.name,
          path: m.id,
          is_downloaded: m.is_downloaded,
          gdrive_id: m.gdrive_id,
        }));

      // Also include any discovered local model files that are not currently
      // runtime-registered, so manual file drops still appear for selection.
      const registeredTokens = new Set<string>();
      const registeredRepoIds = new Set<string>();
      for (const model of list) {
        registeredTokens.add(normalizeModelRef(model.id));
        if (model.model_file) {
          registeredTokens.add(normalizeModelRef(model.model_file));
          registeredTokens.add(modelRefBasename(model.model_file));
        }
        const repoId = model.params?.hf_repo_id;
        if (repoId) {
          registeredRepoIds.add(repoId.toLowerCase());
        }
      }

      const unregistered = discoveredUnits
        .filter((unit) => {
          const normalizedRel = normalizeModelRef(unit.llm_rel_path);
          const normalizedAbs = normalizeModelRef(unit.llm_abs_path);
          const basename = modelRefBasename(unit.llm_rel_path);
          const byPath =
            registeredTokens.has(normalizedRel) ||
            registeredTokens.has(normalizedAbs) ||
            registeredTokens.has(basename);
          const byRepo = registeredRepoIds.has(unit.repo_id.toLowerCase());
          return !byPath && !byRepo;
        })
        .map((unit) => ({
          name: `${unit.repo_id} (${unit.llm_file_name}) (Unregistered)`,
          path: unit.llm_abs_path,
          is_downloaded: true,
          gdrive_id: null as string | null,
        }))
        .filter(
          (entry, index, all) =>
            all.findIndex((candidate) => candidate.path === entry.path) === index
        );

      const allChatModels = [...chatModels, ...unregistered];
      setModels(allChatModels);
      if (allChatModels.length > 0) {
        setSelectedModel((prev) => prev || allChatModels[0].path);
      }

      // TTS engines from the registry
      const tts = list.filter((m) => m.tasks.includes("tts"));
      setTtsEngines(tts);
      if (tts.length > 0) {
        setSelectedTtsEngine((prev) => prev || tts[0].id);
      }

      return list;
    } catch (error) {
      console.error(error);
      return [];
    }
  }, []);
  refreshModelsOnDownloadRef.current = refreshModels;

  useEffect(() => {
    if (!selectedTtsEngine) return;
    const engine = registeredModels.find((m) => m.id === selectedTtsEngine);
    if (!engine) return;

    const configuredVoice = engine.params?.voice;
    if (configuredVoice && KITTEN_TTS_VOICES.includes(configuredVoice as KittenTtsVoice)) {
      setTtsVoice(configuredVoice as KittenTtsVoice);
    }

    const configuredSpeed = parseModelParamNumber(engine.params?.speed);
    if (configuredSpeed !== undefined) {
      const clamped = Math.max(0.5, Math.min(2.0, configuredSpeed));
      setTtsSpeed(clamped);
    }
  }, [selectedTtsEngine, registeredModels, parseModelParamNumber]);

  const handleModelChange = async (path: string) => {
    try {
      setSelectedModel(path);
      await Api.switchModel(path);

      const refreshed = await refreshModels();
      const resolved = findRegisteredModelByIdentifier(refreshed, path);
      if (resolved) {
        setSelectedModel(resolved.id);
      }
    } catch (err) {
      console.error(err);
      showError("Failed to switch model");
    }
  };

  const handleAddModel = () => {
    setHfModalPreset({ folder: "LLM", profile: "llm" });
    setHfModalOpen(true);
  };

  const handleAddVisionModel = () => {
    setHfModalPreset({ folder: "LiquidAI-VLM", profile: "vlm" });
    setHfModalOpen(true);
  };

  // ── Vision helpers ─────────────────────────────────────────────────────────

  const selectImage = async () => {
    try {
      const selected = await open({
        multiple: false,
        filters: [
          {
            name: "Images",
            extensions: ["jpg", "jpeg", "png", "webp", "gif", "bmp"],
          },
        ],
      });
      if (selected && typeof selected === "string") {
        setImagePath(selected);
        const dataUrl = await Api.readImageBase64(selected);
        setImagePreview(dataUrl);
      }
    } catch (err) {
      console.error("Failed to select image:", err);
    }
  };

  const clearImage = useCallback(() => {
    setImagePath(null);
    setImagePreview(null);
  }, []);

  // ── RAG helpers ────────────────────────────────────────────────────────────

  const DOCUMENT_PICKER_EXTENSIONS = [
    "pdf", "docx", "pptx", "xlsx", "xls", "ods",
    "txt", "md", "csv", "tsv", "json", "xml", "html", "htm",
    "rs", "py", "js", "ts", "jsx", "tsx", "java", "c", "cpp",
    "h", "go", "rb", "sh", "toml", "yaml", "yml", "css",
    "scss", "sql", "log", "ini", "cfg",
    "mp3", "wav", "m4a", "ogg", "flac",
  ];

  const attachDirectDocuments = async () => {
    try {
      const selected = await open({
        multiple: true,
        filters: [
          {
            name: "Documents",
            extensions: DOCUMENT_PICKER_EXTENSIONS,
          },
        ],
      });
      if (!selected) return;

      const files = Array.isArray(selected) ? selected : [selected];
      if (files.length === 0) return;

      setDirectDocumentPaths((prev) => {
        const merged = new Set(prev);
        for (const filePath of files) {
          merged.add(filePath);
        }
        return Array.from(merged);
      });
    } catch (err) {
      console.error("Failed to select direct documents:", err);
      showError(`Failed to select documents: ${err}`);
    }
  };

  const removeDirectDocument = (path: string) => {
    setDirectDocumentPaths((prev) => prev.filter((docPath) => docPath !== path));
  };

  const clearDirectDocuments = useCallback(() => {
    setDirectDocumentPaths([]);
  }, []);

  const ingestFile = async () => {
    try {
      const selected = await open({
        multiple: true,
        filters: [
          {
            name: "Documents",
            extensions: DOCUMENT_PICKER_EXTENSIONS,
          },
        ],
      });
      if (!selected) return;
      const files = Array.isArray(selected) ? selected : [selected];
      if (files.length === 0) return;

      // Add placeholder entries to the side panel immediately so users can
      // click to view the file while ingestion is still running.
      const placeholders: IngestionStatus[] = files.map((f, i) => ({
        doc_id: -(i + 1), // negative IDs to avoid clashing with real docs
        title: f.split(/[\\/]/).pop() || f,
        file_path: f,
        total_chunks: 0,
        embedded_chunks: 0,
        enriched_chunks: 0,
        phase: "ingesting",
      }));
      setRagDocs((prev) => [...placeholders, ...prev]);

      setRagIngesting(true);

      // Ingest all files in parallel so the UI doesn't hang waiting on each one.
      // As each file finishes, refresh the doc list to replace its placeholder.
      const results = await Promise.allSettled(
        files.map((f) =>
          Api.ingestDocument(f).then(async (res) => {
            await loadRagDocs(); // refresh side panel as each file completes
            return res;
          })
        )
      );

      await loadRagDocs();
      setRagIngesting(false);

      const failures = results.filter((r) => r.status === "rejected");
      if (failures.length > 0) {
        const msgs = failures.map((r) => (r as PromiseRejectedResult).reason).join("\n");
        alert(`${failures.length} file(s) failed to ingest:\n${msgs}`);
      }
    } catch (e) {
      console.error(e);
      setRagIngesting(false);
      await loadRagDocs();
      showError(`Ingest failed: ${e}`);
    }
  };

  const ingestDir = async () => {
    try {
      const selected = await open({ directory: true });
      if (selected && typeof selected === "string") {
        setRagIngesting(true);
        await Api.ingestFolder(selected);
        await loadRagDocs();
        setRagIngesting(false);
      }
    } catch (e) {
      console.error(e);
      setRagIngesting(false);
      showError(`Folder ingest failed: ${e}`);
    }
  };

  const deleteRagDoc = async (docId: number) => {
    try {
      // Close the viewer if the deleted document is currently open
      const doc = ragDocs.find((d) => d.doc_id === docId);
      if (doc) {
        const delPath = doc.file_path;
        if (pdfViewerData && doc.title === pdfViewerData.title) {
          setPdfViewerData(null);
        }
        if (docViewerFile && docViewerFile.filePath === delPath) {
          setDocViewerFile(null);
        }
      }

      await Api.deleteRagDocument(docId);
      await loadRagDocs();
    } catch (e) {
      console.error(e);
      showError(`Delete failed: ${e}`);
    }
  };

  // ── Viewer handlers ────────────────────────────────────────────────────────

  /** Open the right viewer for any supported document format. */
  const openDocViewer = async (doc: IngestionStatus) => {
    const ext = doc.file_path.split(".").pop()?.toLowerCase() || "";

    if (ext === "pdf") {
      // PDF uses the dedicated PdfViewer
      try {
        setPdfLoading(true);
        const data = await Api.readFileBase64(doc.file_path);
        setPdfViewerData({ data, title: doc.title });
        setDocViewerFile(null); // Clear any other open viewer
      } catch (e) {
        console.error("Failed to load PDF:", e);
        showError(`Failed to open PDF: ${e}`);
      } finally {
        setPdfLoading(false);
      }
    } else if (VIEWABLE_EXTS.has(ext)) {
      // Everything else uses the universal DocumentViewer
      setDocViewerFile({ filePath: doc.file_path, title: doc.title });
      setPdfViewerData(null); // Clear any PDF viewer
    }
  };

  const closePdfViewer = () => {
    setPdfViewerData(null);
  };

  const closeDocViewer = () => {
    setDocViewerFile(null);
  };

  // ── Cancel handler ────────────────────────────────────────────────────────

  const handleCancel = () => {
    const sid = activeSessionId;
    if (!sid) return;
    // Abort the SSE fetch (text / RAG modes)
    abortControllersRef.current.get(sid)?.abort();
    abortControllersRef.current.delete(sid);
    // Stop vision event listener
    visionUnlistenRef.current?.();
    visionUnlistenRef.current = null;
    // Commit whatever partial response was received
    updateSession(sid, (prev) => ({
      messages: prev.streamingContent
        ? [...prev.messages, { role: "assistant" as const, content: prev.streamingContent }]
        : prev.messages,
      streamingContent: "",
      loading: false,
      cancelled: true,
    }));
  };

  // ── Main send handler ─────────────────────────────────────────────────────

  const handleSend = useCallback(
    async (text: string) => {
      await executeHandleSend(text, {
        activeSessionId,
        sessions,
        chatMode,
        ragEnabled,
        webEnabled,
        webDepth,
        imagePath,
        directDocumentPaths,
        ragDocs,
        selectedModel,
        selectedVisionModel,
        selectedTtsEngine,
        ttsVoice,
        ttsSpeed,
        thinkingEnabled,
        abortControllersRef,
        visionUnlistenRef,
        generalIntervalRef,
        ttsIntervalRef,
        updateSession,
        setActiveMindmapOverlay,
        setGeneralGenerating,
        setGeneralElapsedTime,
        setGeneralGenerationTime,
        setMindmapsBySession,
        setStreamingThinking,
        setTtsGenerating,
        setTtsElapsedTime,
        setTtsGenerationTime,
        setContextUsageForSession: (sessionId, usage) => {
          setContextUsageBySession((prev) => ({
            ...prev,
            [sessionId]: usage,
          }));
        },
        clearImage,
        clearDirectDocuments,
        getContextWindowTokens,
        getChatGenerationOptions,
      });
    },
    [
      activeSessionId,
      sessions,
      chatMode,
      ragEnabled,
      webEnabled,
      webDepth,
      imagePath,
      directDocumentPaths,
      ragDocs,
      selectedModel,
      selectedVisionModel,
      selectedTtsEngine,
      ttsVoice,
      ttsSpeed,
      thinkingEnabled,
      updateSession,
      clearImage,
      clearDirectDocuments,
      getContextWindowTokens,
      getChatGenerationOptions,
    ]
  );

  // ── Derived ────────────────────────────────────────────────────────────────

  const handleModeSwitch = (mode: ChatMode) => {
    if (mode === chatMode) return;

    // Preserve existing behavior: stop any active generation before switching mode.
    if (activeSession?.loading) {
      handleCancel();
    }

    const switchedTo = MODE_CONFIG.find((m) => m.mode === mode)?.label ?? mode;
    setModeSwitchNotice(`Switched to ${switchedTo} mode`);

    if (mode !== "vision") {
      setImagePath(null);
      setImagePreview(null);
    }
    if (mode !== "text") {
      setDirectDocumentPaths([]);
    }
    if (mode !== "text" && mode !== "mindmap") {
      setDocPanelOpen(false);
    }

    setChatMode(mode);
  };

  const handleRagToggle = useCallback((enabled: boolean) => {
    setRagEnabled(enabled);
    if (enabled) {
      setDirectDocumentPaths([]);
    }
  }, []);

  const handleWebToggle = useCallback((enabled: boolean) => {
    setWebEnabled(enabled);
  }, []);

  const handleWebDepthChange = useCallback((depth: "snippets" | "full") => {
    setWebDepth(depth);
  }, []);

  const getPlaceholder = (): string => {
    switch (chatMode) {
      case "vision":
        return "Ask about the image (e.g., 'What's in this image?')";
      case "audio":
        return "Type text to generate speech...";
      case "podcast":
        return "What topic should the podcast cover?";
      case "mindmap":
        return ragDocs.length > 0
          ? "Ask for a mindmap (auto-grounds on relevant documents)..."
          : "Describe a topic to generate a mindmap...";
      default:
        if (ragEnabled) {
          return ragDocs.length > 0
            ? "RAG ON: ask about your ingested documents..."
            : "RAG ON: ingest documents or chat freely...";
        }

        return directDocumentPaths.length > 0 || ragDocs.length > 0
          ? "RAG OFF: documents will be sent directly to the model..."
          : "RAG OFF: attach documents or chat freely...";
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  const currentModeConfig = MODE_CONFIG.find((m) => m.mode === chatMode)!;
  const activeContextUsage = activeSession ? contextUsageBySession[activeSession.id] ?? null : null;
  const canManualCompactContext =
    !!activeSession &&
    !activeSession.loading &&
    (chatMode === "text" || chatMode === "mindmap");

  const openViewerSessions = openSessionIds
    .map((id) => sessions.find((s) => s.id === id))
    .filter((s): s is ChatSession => !!s);

  // Sidebar section state with toggle logic
  const [sidebarSection, setSidebarSection] = useState<"chats" | "audio" | "mindmaps" | "playground" | null>("chats");

  // Toggle handler for sidebar
  const handleSidebarNav = (section: "chats" | "audio" | "mindmaps" | "playground") => {
    setSidebarSection((prev) => (prev === section ? null : section));
    if (section === "playground") {
      setChatMode("playground");
    } else if (chatMode === "playground") {
      setChatMode("text");
    }
  };

  // Exit playground and return to chats view
  const handleExitPlayground = () => {
    setChatMode("text");
    setSidebarSection("chats");
  };

  const activeSessionMindmaps = activeSession ? (mindmapsBySession[activeSession.id] ?? []) : [];
  const handleStartupAction = async (action: () => Promise<void>) => {
    await action();
    if (downloadOptionalOnStart) {
      await downloadMissingOptionalModels();
    }
  };

  const canContinueStartupWorkspace = !!(
    startupContinueWorkspace &&
    workspaces.some((workspace) => workspace.id === startupContinueWorkspace.id)
  );

  const continueExistingWorkspace = () => {
    if (!startupContinueWorkspace || !canContinueStartupWorkspace) return;
    void handleStartupAction(() => switchWorkspaceById(startupContinueWorkspace.id));
  };

  const createWorkspaceFromStartup = () => {
    void handleStartupAction(createNewWorkspace);
  };

  const importWorkspaceFromStartup = () => {
    void handleStartupAction(openWorkspaceFromFile);
  };

  const startTourFromStartup = () => {
    setSuppressStartupModal(true);
    startTour("getting-started", {
      source: "startup",
      onExit: () => setSuppressStartupModal(false),
      onComplete: () => setSuppressStartupModal(false),
    });
  };

  const mindmaps = activeSessionMindmaps
    .map((map) => ({
      id: map.id,
      sessionId: activeSession?.id ?? "",
      name: map.title,
      query: map.query,
      generatedFrom: map.generatedFrom,
      createdAt: map.createdAt,
    }))
    .sort((a, b) => b.createdAt - a.createdAt);

  const activeMindmapGraph = activeMindmapOverlay
    ? (mindmapsBySession[activeMindmapOverlay.sessionId] ?? []).find(
        (map) => map.id === activeMindmapOverlay.mindmapId
      ) ?? null
    : null;

  // Handler to save audio to sidebar (set audioSaved=true)
  const handleSaveAudioToSidebar = (msgIdx: number) => {
    if (!activeSession) return;
    updateSession(activeSession.id, (prev) => ({
      messages: prev.messages.map((m, i) => i === msgIdx ? { ...m, audioSaved: true } : m)
    }));
  };

  const handleManualContextCompaction = useCallback(async () => {
    if (!activeSession) return;
    if (activeSession.loading) return;
    if (chatMode !== "text" && chatMode !== "mindmap") return;

    setContextCompacting(true);
    try {
      const generation = getChatGenerationOptions(selectedModel);
      const result = await Api.compactChatContext({
        messages: toContextMessages(activeSession.messages),
        contextWindowTokens: getContextWindowTokens(selectedModel),
        reservedOutputTokens: resolveReservedOutputTokens(generation.maxTokens),
        thresholdPercent: CONTEXT_COMPACTION_THRESHOLD,
        allowAutoCompaction: false,
        forceCompaction: true,
        preserveRecentMessages: CONTEXT_COMPACTION_KEEP_RECENT,
        modelOverride: selectedModel || null,
      });

      setContextUsageBySession((prev) => ({
        ...prev,
        [activeSession.id]: result.usage,
      }));

      if (result.compacted) {
        const rebuilt = applyCompactionResultToSession(
          activeSession.messages,
          activeSession.mediaAssets ?? {},
          result
        );

        updateSession(activeSession.id, {
          messages: rebuilt.messages,
          mediaAssets: rebuilt.mediaAssets,
        });

        setAppModal({
          open: true,
          kind: "info",
          title: "Context compacted",
          message: `Conversation context was compacted. ${result.droppedMessages} message(s) were compressed or removed to free context space.`,
          confirmLabel: "OK",
          cancelLabel: "Cancel",
          showCancel: false,
        });
      } else {
        setAppModal({
          open: true,
          kind: "info",
          title: "Context already efficient",
          message: "No additional compaction was needed for the current session context.",
          confirmLabel: "OK",
          cancelLabel: "Cancel",
          showCancel: false,
        });
      }
    } catch (err) {
      console.error("Manual context compaction failed:", err);
      showError(`Failed to compact context: ${String(err)}`);
    } finally {
      setContextCompacting(false);
    }
  }, [
    activeSession,
    chatMode,
    getChatGenerationOptions,
    getContextWindowTokens,
    selectedModel,
    showError,
    updateSession,
  ]);

  // Show startup modal if no active workspace yet (unless we're running the tour from it)
  const showStartupModal = !activeWorkspace && !suppressStartupModal;
  const showParamsDock = !!activeRuntimeParamTarget && paramsDockOpen;
  const showRightSidebar = showParamsDock || docPanelOpen;

  return (
    <div className="relative w-full h-full overflow-hidden">
      <AppDialogsLayer
        showStartupModal={showStartupModal}
        onContinueWorkspace={continueExistingWorkspace}
        canContinueWorkspace={canContinueStartupWorkspace}
        continueWorkspaceName={startupContinueWorkspace?.name ?? null}
        onNewProject={createWorkspaceFromStartup}
        onImportProject={importWorkspaceFromStartup}
        onStartTour={startTourFromStartup}
        workspaceBusy={workspaceBusy}
        appModal={appModal}
        onModalConfirm={handleModalConfirm}
        onModalCancel={handleModalCancel}
        settingsOpen={settingsOpen}
        onCloseSettings={() => setSettingsOpen(false)}
        registeredModels={registeredModels}
        modelCatalog={modelCatalog}
        onModelsUpdated={refreshModels}
        downloads={downloads}
        onDownloadModel={handleDownloadModel}
        onCancelDownload={handleCancelDownload}
        onUninstallModel={handleUninstall}
        onDownloadMissingOptional={downloadMissingOptionalModels}
        onConfirmAction={confirmAction}
        activeWorkspaceId={activeWorkspace?.id}
        hfModalOpen={hfModalOpen}
        onCloseHfModal={() => setHfModalOpen(false)}
        hfModalPreset={hfModalPreset}
        onModelImported={refreshModels}
        toursOpen={toursOpen}
        onCloseTours={() => setToursOpen(false)}
      />

      {/* Main app content stays visible in background behind startup modal */}
      <div className="flex h-full w-full relative z-10">
          <SidebarNav
        selected={sidebarSection}
        onSelect={handleSidebarNav}
        onImportProject={() => void openWorkspaceFromFile()}
        onExportProject={() => void saveWorkspaceFile()}
        onOpenSettings={() => setSettingsOpen(true)}
        onOpenTours={() => setToursOpen(true)}
        onOpenHuggingFaceSearch={handleAddModel}
        workspaceBusy={workspaceBusy}
        canExport={!!activeWorkspace}
        theme={theme}
        onToggleTheme={toggleTheme}
      />
      {/* Vertical blue line when sidebar is minimized */}
      {sidebarSection === null && (
        <div className="w-1 min-w-1 h-full bg-neon rounded-full mx-1 shadow-[0_0_16px_#00d4ff88] transition-all duration-200 opacity-100" />
      )}
      {/* Side section (chats/audio/mindmaps) */}
      {sidebarSection !== null && (
        <>
          {/* Side section content */}
          {sidebarSection === "chats" && (
            <ChatHistorySidebar
              sessions={sessions}
              activeSessionId={activeSessionId}
              onSelectSession={openSessionInViewer}
              onNewSession={addNewSession}
              onDeleteSession={closeSession}
            />
          )}
          {sidebarSection === "audio" && (
            <AudioSidebar
              sessions={sessions}
              onDeleteAudio={(sessionId, msgIdx) => {
                updateSession(sessionId, (prev) => ({
                  messages: prev.messages.map((m, i) =>
                    i === msgIdx ? { ...m, audioSaved: false } : m
                  ),
                }));
              }}
            />
          )}
          {sidebarSection === "mindmaps" && (
            <MindmapsSidebar
              mindmaps={mindmaps}
              activeMindmapOverlay={activeMindmapOverlay}
              onOpenMindmap={openMindmapOverlay}
            />
          )}
          {sidebarSection === "playground" && (
            <PlaygroundSidebar
              onOpen={() => {
                setChatMode("playground");
                setSidebarSection(null);
              }}
            />
          )}
          {/* Vertical blue line between side section and chat window (less blue) */}
          <div className="w-1 min-w-1 h-full bg-neon/40 rounded-full mx-1 shadow-[0_0_8px_#00d4ff33] transition-all duration-200 opacity-60" />
        </>
      )}

      <AppMainContent
        chatMode={chatMode}
        openViewerSessions={openViewerSessions}
        activeSessionId={activeSessionId}
        onSelectSession={setActiveSessionId}
        onNewSession={addNewSession}
        onCloseSession={closeViewerTab}
        onReorderSessions={reorderViewerTabs}
        currentModeConfig={currentModeConfig}
        workspaces={workspaces}
        activeWorkspace={activeWorkspace}
        onSelectWorkspace={(id) => {
          void switchWorkspaceById(id);
        }}
        onCreateWorkspace={() => {
          void createNewWorkspace();
        }}
        onDeleteWorkspace={(id) => {
          void deleteWorkspaceById(id);
        }}
        onRenameWorkspace={renameWorkspaceById}
        workspaceBusy={workspaceBusy}
        modelLoadingStatus={modelLoadingStatus}
        models={models}
        selectedModel={selectedModel}
        onModelChange={handleModelChange}
        onAddModel={handleAddModel}
        onDownloadModel={handleDownloadModel}
        onCancelDownload={handleCancelDownload}
        onUninstallModel={handleUninstall}
        onConfirmAction={confirmAction}
        downloads={downloads}
        ttsEngines={ttsEngines}
        selectedTtsEngine={selectedTtsEngine}
        onSelectTtsEngine={setSelectedTtsEngine}
        visionModels={visionModels}
        selectedVisionModel={selectedVisionModel}
        onSelectVisionModel={setSelectedVisionModel}
        onAddVisionModel={handleAddVisionModel}
        activeRuntimeParamTarget={activeRuntimeParamTarget}
        paramsDockOpen={paramsDockOpen}
        onToggleParamsDock={() => setParamsDockOpen((open) => !open)}
        contextUsage={activeContextUsage}
        onCompactContext={() => {
          void handleManualContextCompaction();
        }}
        canCompactContext={canManualCompactContext}
        isCompactingContext={contextCompacting}
        ragDocs={ragDocs}
        ragEnabled={ragEnabled}
        modeOptions={MODE_CONFIG.map(({ mode, label }) => ({ mode, label }))}
        onSelectMode={handleModeSwitch}
        onToggleRagEnabled={handleRagToggle}
        webEnabled={webEnabled}
        onToggleWebEnabled={handleWebToggle}
        webDepth={webDepth}
        onWebDepthChange={handleWebDepthChange}
        activeSession={activeSession}
        onSend={(text) => {
          void handleSend(text);
        }}
        onCancel={handleCancel}
        placeholder={getPlaceholder()}
        ragIngesting={ragIngesting}
        enrichmentStatus={enrichmentStatus}
        onIngestFile={() => {
          void ingestFile();
        }}
        onIngestDir={() => {
          void ingestDir();
        }}
        onAttachDirectDocuments={() => {
          void attachDirectDocuments();
        }}
        directDocumentPaths={directDocumentPaths}
        onRemoveDirectDocument={removeDirectDocument}
        onClearDirectDocuments={clearDirectDocuments}
        onSelectVisionImage={() => {
          void selectImage();
        }}
        visionImagePath={imagePath}
        visionImagePreview={imagePreview}
        onClearVisionImage={clearImage}
        docPanelOpen={docPanelOpen}
        onToggleDocPanel={() => setDocPanelOpen((v) => !v)}
        modeSwitchNotice={modeSwitchNotice}
        onSaveAudioToSidebar={handleSaveAudioToSidebar}
        streamingThinking={streamingThinking}
        thinkingEnabled={thinkingEnabled}
        onToggleThinking={() => setThinkingEnabled(!thinkingEnabled)}
        activeMindmapOverlay={activeMindmapOverlay}
        activeMindmapGraph={activeMindmapGraph}
        onCloseMindmapOverlay={() => setActiveMindmapOverlay(null)}
        pdfLoading={pdfLoading}
        pdfViewerData={pdfViewerData}
        onClosePdfViewer={closePdfViewer}
        docViewerFile={docViewerFile}
        onCloseDocViewer={closeDocViewer}
        onExitPlayground={handleExitPlayground}
      />

      <StartupModelToast
        toast={startupModelToast}
        minimized={startupToastMinimized}
        downloads={downloads}
        startupCancelledIds={startupCancelledIds}
        startupCancellingIds={startupCancellingIds}
        startupOverallSpeedBps={startupOverallSpeedBps}
        startupSelectedTotalMb={startupSelectedTotalMb}
        onToggleMinimized={setStartupToastMinimized}
        onDecline={handleStartupToastDecline}
        onAccept={() => {
          void handleStartupToastAccept();
        }}
        onToggleModelSelection={toggleStartupModelSelection}
        onCancelAllDownloads={() => {
          void handleStartupToastCancelDownloads();
        }}
        onCancelSingleDownload={(modelId) => {
          void handleStartupToastCancelSingleDownload(modelId);
        }}
      />

      <AppRightSidebar
        showRightSidebar={showRightSidebar}
        showParamsDock={showParamsDock}
        docPanelOpen={docPanelOpen}
        activeRuntimeParamTarget={activeRuntimeParamTarget}
        onApplyRuntimeParams={handleApplyRuntimeParams}
        onCloseParamsDock={() => setParamsDockOpen(false)}
        ragIngesting={ragIngesting}
        enrichmentStatus={enrichmentStatus}
        ragDocs={ragDocs}
        activeSession={activeSession}
        onCloseDocPanel={() => setDocPanelOpen(false)}
        onIngestFile={() => {
          void ingestFile();
        }}
        onIngestDir={() => {
          void ingestDir();
        }}
        onOpenDocViewer={openDocViewer}
        onDeleteRagDoc={(docId) => {
          void deleteRagDoc(docId);
        }}
      />
    </div>

    </div>
  );
}


export default App;
