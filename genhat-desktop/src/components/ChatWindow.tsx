import React, { useState, useEffect, useRef, memo } from "react";
import { MessageSquare, Eye, Volume2, Mic, FileText, Share2, Workflow, X } from "lucide-react";
import MarkdownRenderer from "./MarkdownRenderer";
import AudioPlayer from "./AudioPlayer";
import VoiceInputButton from "./VoiceInputButton";
import SpeakButton from "./SpeakButton";
import { Api } from "../api";
import type { ChatMessage, MediaAsset, IngestionStatus, ChatMode, ChatSession } from "../types";

const MODE_ICON_MAP: Record<ChatMode, React.ElementType> = {
  text: MessageSquare,
  vision: Eye,
  audio: Volume2,
  rag: FileText,
  podcast: Mic,
  mindmap: Share2,
  playground: Workflow,
};

/** Copy button for a full assistant response */
const CopyMsgButton: React.FC<{ text: string }> = ({ text }) => {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = text;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand("copy");
      document.body.removeChild(ta);
    }
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <button className="p-1.5 glass border border-glass-border text-txt-muted cursor-pointer rounded-lg transition-all duration-200 hover:text-neon hover:border-neon/30 hover:shadow-[0_0_8px_rgba(0,212,255,0.1)]" onClick={handleCopy} title="Copy response">
      {copied ? (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
          <path d="M20 6L9 17L4 12" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ) : (
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none">
          <rect x="9" y="2" width="6" height="4" rx="1" stroke="currentColor" strokeWidth="2" />
          <path d="M9 3H6a2 2 0 00-2 2v14a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2h-3" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
        </svg>
      )}
    </button>
  );
};

/** Collapsible thinking/reasoning box for assistant messages */
const ThinkingBox: React.FC<{ thinking: string }> = ({ thinking }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  
  if (!thinking) return null;

  return (
    <div className="mb-3">
      <button
        className="flex items-center gap-2 text-xs text-txt-muted hover:text-txt transition-colors cursor-pointer w-full"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="opacity-50">{isExpanded ? "▼" : "▶"}</span>
        <span className="font-medium">Thinking</span>
        <span className="text-[0.65rem] opacity-40">({thinking.length} chars)</span>
      </button>
      {isExpanded && (
        <div className="mt-2 p-3 rounded-lg bg-black/20 border border-white/5 text-xs text-txt-muted leading-relaxed opacity-70 max-h-60 overflow-y-auto">
          <pre className="whitespace-pre-wrap font-mono">{thinking}</pre>
        </div>
      )}
    </div>
  );
};

interface ChatWindowProps {
  messages: ChatMessage[];
  streamingContent: string;
  isLoading: boolean;
  onSend: (text: string) => void;
  onCancel?: () => void;
  cancelled?: boolean;
  audioSrc?: string;
  audioOutputs?: string[];
  placeholder?: string;
  mediaAssets?: Record<number, MediaAsset[]>;
  chatMode?: string;
  ttsGenerating?: boolean;
  ttsElapsedTime?: number;
  ttsGenerationTime?: number | null;
  generalGenerating?: boolean;
  generalElapsedTime?: number;
  generalGenerationTime?: number | null;
  ragDocs?: IngestionStatus[];
  ragIngesting?: boolean;
  enrichmentStatus?: string | null;
  ragEnabled?: boolean;
  onToggleRagEnabled?: (enabled: boolean) => void;
  webEnabled?: boolean;
  onToggleWebEnabled?: (enabled: boolean) => void;
  webDepth?: "snippets" | "full";
  onWebDepthChange?: (depth: "snippets" | "full") => void;
  webSearchResult?: import("../types").WebSearchResult | null;
  onIngestFile?: () => void;
  onIngestDir?: () => void;
  onAttachDirectDocuments?: () => void;
  directDocumentPaths?: string[];
  onRemoveDirectDocument?: (path: string) => void;
  onClearDirectDocuments?: () => void;
  onSelectVisionImage?: () => void;
  visionImagePath?: string | null;
  visionImagePreview?: string | null;
  onClearVisionImage?: () => void;
  onToggleDocPanel?: () => void;
  showRagControls?: boolean;
  docPanelOpen?: boolean;
  modeOptions?: { mode: ChatMode; label: string }[];
  currentMode?: ChatMode;
  onSelectMode?: (mode: ChatMode) => void;
  modeSwitchNotice?: string | null;
  streamingThinking?: string;
  thinkingEnabled?: boolean;
  onToggleThinking?: () => void;
  saveAudioToSidebar?: (msgIdx: number) => void;
  session?: ChatSession;
}

/** Inline gallery for extracted images/tables attached to an assistant message. */
const MediaGallery: React.FC<{ assets: MediaAsset[] }> = ({ assets }) => {
  const [expanded, setExpanded] = useState<number | null>(null);
  const [dataUrls, setDataUrls] = useState<Record<number, string>>({});

  useEffect(() => {
    if (!assets || assets.length === 0) return;
    let cancelled = false;

    const loadAll = async () => {
      const entries: [number, string][] = [];
      for (const asset of assets) {
        try {
          const dataUrl = await Api.readImageBase64(asset.file_path);
          if (!cancelled) entries.push([asset.id, dataUrl]);
        } catch (e) {
          console.warn(`Failed to load media ${asset.id}:`, e);
        }
      }
      if (!cancelled) {
        setDataUrls(Object.fromEntries(entries));
      }
    };

    loadAll();
    return () => { cancelled = true; };
  }, [assets]);

  if (!assets || assets.length === 0) return null;

  return (
    <div className="mt-2.5 pt-2.5 border-t border-glass-border">
      <div className="text-[0.75rem] text-txt-muted mb-2 font-medium">
        📎 {assets.length} related {assets.length === 1 ? "figure" : "figures"}
      </div>
      <div className="flex flex-wrap gap-2">
        {assets.map((asset) => (
          <div
            key={asset.id}
            className={`media-thumb relative rounded-lg overflow-hidden cursor-pointer transition-all duration-200 border border-glass-border hover:border-neon hover:shadow-md ${expanded === asset.id ? "max-w-full flex-[1_1_100%]" : "max-w-[200px]"}`}
            onClick={() => setExpanded(expanded === asset.id ? null : asset.id)}
          >
            {dataUrls[asset.id] ? (
              <img
                src={dataUrls[asset.id]}
                alt={asset.caption || `${asset.asset_type} from document`}
                loading="lazy"
                className={`block w-full h-auto ${expanded === asset.id ? "" : "max-h-[160px] object-cover"}`}
              />
            ) : (
              <div className="flex items-center justify-center w-[160px] h-[120px] text-txt-muted text-[0.75rem] bg-void-800">Loading…</div>
            )}
            <span className="absolute top-1 right-1 text-[0.7rem] bg-black/60 rounded px-1 py-0.5 leading-none">
              {asset.asset_type === "table" ? "📊" : "🖼️"}
            </span>
            {expanded === asset.id && asset.caption && (
              <div className="p-1.5 px-2 text-[0.72rem] text-txt-muted bg-void-800 leading-snug max-h-[100px] overflow-y-auto">{asset.caption}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

/** Thumbnail renderer for user-uploaded vision images stored on a chat message. */
const VisionMessageImage: React.FC<{
  imagePath: string;
  imageName: string;
  onOpen: (src: string, title: string) => void;
}> = ({ imagePath, imageName, onOpen }) => {
  const [previewSrc, setPreviewSrc] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    Api.readImageBase64(imagePath)
      .then((src) => {
        if (!cancelled) setPreviewSrc(src);
      })
      .catch((err) => {
        console.warn(`Failed to load vision image preview for ${imagePath}:`, err);
      });

    return () => {
      cancelled = true;
    };
  }, [imagePath]);

  return (
    <button
      className="group block w-[180px] rounded-xl overflow-hidden border border-glass-border bg-void-800/70 hover:border-neon/40 transition-colors"
      onClick={() => previewSrc && onOpen(previewSrc, imageName)}
      title={previewSrc ? `Preview ${imageName}` : imageName}
      disabled={!previewSrc}
    >
      {previewSrc ? (
        <img src={previewSrc} alt={imageName} className="w-full h-[130px] object-cover" />
      ) : (
        <div className="w-full h-[130px] flex items-center justify-center text-[0.72rem] text-txt-muted">Loading image...</div>
      )}
      <div className="px-2.5 py-1.5 text-[0.72rem] text-txt-secondary truncate text-left group-hover:text-txt">
        {imageName}
      </div>
    </button>
  );
};

/** Collapsible panel showing web search sources used for the last response. */
const WebSourcesPanel: React.FC<{
  webSearchResult: import("../types").WebSearchResult;
}> = ({ webSearchResult }) => {
  const [open, setOpen] = useState(false);

  return (
    <div className="px-6 py-2 border-t border-glass-border bg-void-900/60">
      <button
        className="inline-flex items-center gap-1.5 text-[0.72rem] text-txt-secondary hover:text-neon transition-colors"
        onClick={() => setOpen((v) => !v)}
      >
        <span>{open ? "▾" : "▸"}</span>
        <span>Web sources ({webSearchResult.results.length})</span>
      </button>
      {open && (
        <ul className="mt-1.5 space-y-1.5 max-h-48 overflow-y-auto">
          {webSearchResult.results.map((hit, i) => (
            <li key={i} className="text-[0.7rem] leading-snug">
              <a
                href={hit.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-neon hover:underline font-medium block truncate"
                title={hit.url}
              >
                {hit.title || hit.url}
              </a>
              {hit.snippet && (
                <p className="text-txt-muted mt-0.5 line-clamp-2">{hit.snippet}</p>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

const ChatWindow: React.FC<ChatWindowProps> = memo(({
  messages,
  streamingContent,
  isLoading,
  onSend,
  onCancel,
  cancelled = false,
  placeholder = "Message NELA...",
  mediaAssets = {},
  chatMode = "text",
  ttsGenerating = false,
  ttsElapsedTime = 0,
  generalGenerating = false,
  generalElapsedTime = 0,
  generalGenerationTime = null,
  ragDocs = [],
  ragIngesting = false,
  enrichmentStatus = null,
  ragEnabled = false,
  onToggleRagEnabled,
  webEnabled = false,
  onToggleWebEnabled,
  webDepth = "snippets",
  onWebDepthChange,
  webSearchResult = null,
  onIngestFile,
  onIngestDir,
  onAttachDirectDocuments,
  directDocumentPaths = [],
  onRemoveDirectDocument,
  onClearDirectDocuments,
  onSelectVisionImage,
  visionImagePath = null,
  visionImagePreview = null,
  onClearVisionImage,
  onToggleDocPanel,
  showRagControls = false,
  docPanelOpen = false,
  modeOptions = [],
  currentMode = "text",
  onSelectMode,
  modeSwitchNotice = null,
  saveAudioToSidebar = () => {},
  streamingThinking = "",
  thinkingEnabled = false,
  onToggleThinking,
}) => {
  const [inputObj, setInputObj] = useState("");
  const [showAttachMenu, setShowAttachMenu] = useState(false);
  const [showModeMenu, setShowModeMenu] = useState(false);
  const [previewModal, setPreviewModal] = useState<{ src: string; title: string } | null>(null);
  const attachMenuRef = useRef<HTMLDivElement>(null);
  const modeMenuRef = useRef<HTMLDivElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  /** Tracks the number of messages that have already been rendered and animated.
   *  Only messages at index >= this value get the entrance animation.
   *  We use state (not a ref) so ESLint doesn't flag .current reads during render. */
  const [prevMsgCount, setPrevMsgCount] = useState(0);
  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: track previous render's msg count
    setPrevMsgCount(messages.length);
  }, [messages.length]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, streamingContent]);

  // Close attach menu on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (attachMenuRef.current && !attachMenuRef.current.contains(e.target as Node)) {
        setShowAttachMenu(false);
      }
      if (modeMenuRef.current && !modeMenuRef.current.contains(e.target as Node)) {
        setShowModeMenu(false);
      }
    };
    if (showAttachMenu) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    if (showModeMenu) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [showAttachMenu, showModeMenu]);

  const handleSend = () => {
    if (!inputObj.trim() || isLoading) return;
    onSend(inputObj);
    setInputObj("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!isLoading) handleSend();
    }
  };

  const hasMessages = messages.length > 0 || isLoading;
  const showAttachButton = showRagControls || chatMode === "vision";
  const visionFileName = visionImagePath ? visionImagePath.split(/[/\\]/).pop() ?? "image" : "image";
  const currentModeLabel = modeOptions.find((option) => option.mode === currentMode)?.label ?? "Mode";
  const CurrentModeIcon = MODE_ICON_MAP[currentMode] ?? MessageSquare;

  const renderAttachMenu = () => {
    if (chatMode === "vision") {
      return (
        <button
          className="w-full flex items-center gap-2.5 py-2 px-3 rounded-lg text-sm text-txt-secondary bg-transparent border-none cursor-pointer transition-all duration-150 hover:bg-glass-hover hover:text-txt"
          onClick={() => {
            onSelectVisionImage?.();
            setShowAttachMenu(false);
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <path d="M21 15l-5-5L5 21" />
          </svg>
          <div className="flex flex-col items-start">
            <span className="font-medium">Upload Image</span>
            <span className="text-[0.68rem] text-txt-muted">JPG, PNG, WEBP, GIF, BMP</span>
          </div>
        </button>
      );
    }

    if (chatMode === "text" && !ragEnabled) {
      return (
        <button
          className="w-full flex items-center gap-2.5 py-2 px-3 rounded-lg text-sm text-txt-secondary bg-transparent border-none cursor-pointer transition-all duration-150 hover:bg-glass-hover hover:text-txt"
          onClick={() => {
            onAttachDirectDocuments?.();
            setShowAttachMenu(false);
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
          </svg>
          <div className="flex flex-col items-start">
            <span className="font-medium">Attach Files</span>
            <span className="text-[0.68rem] text-txt-muted">Send directly to model</span>
          </div>
        </button>
      );
    }

    return (
      <>
        <button
          className="w-full flex items-center gap-2.5 py-2 px-3 rounded-lg text-sm text-txt-secondary bg-transparent border-none cursor-pointer transition-all duration-150 hover:bg-glass-hover hover:text-txt"
          onClick={() => {
            onIngestFile?.();
            setShowAttachMenu(false);
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
            <polyline points="14 2 14 8 20 8" />
          </svg>
          <div className="flex flex-col items-start">
            <span className="font-medium">Add Files</span>
            <span className="text-[0.68rem] text-txt-muted">PDF, DOCX, TXT, code...</span>
          </div>
        </button>
        <button
          className="w-full flex items-center gap-2.5 py-2 px-3 rounded-lg text-sm text-txt-secondary bg-transparent border-none cursor-pointer transition-all duration-150 hover:bg-glass-hover hover:text-txt"
          onClick={() => {
            onIngestDir?.();
            setShowAttachMenu(false);
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
            <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
          </svg>
          <div className="flex flex-col items-start">
            <span className="font-medium">Add Folder</span>
            <span className="text-[0.68rem] text-txt-muted">Ingest entire directory</span>
          </div>
        </button>
      </>
    );
  };

  const renderRagToggle = () => {
    if (chatMode !== "text" || !onToggleRagEnabled) return null;

    return (
      <div className="flex items-center justify-center mb-2">
        <button
          className={`inline-flex items-center gap-2 py-1 px-2 rounded-full border text-[0.72rem] transition-all duration-200 ${
            ragEnabled
              ? "bg-neon-subtle text-neon border-neon/40"
              : "bg-glass-bg text-txt-secondary border-glass-border hover:border-neon/30 hover:text-neon"
          }`}
          onClick={() => onToggleRagEnabled(!ragEnabled)}
          title="Toggle RAG retrieval"
        >
          <span className="font-semibold">RAG {ragEnabled ? "On" : "Off"}</span>
          <span
            className={`relative inline-flex h-4 w-8 rounded-full transition-colors ${
              ragEnabled ? "bg-neon" : "bg-void-700"
            }`}
          >
            <span
              className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${
                ragEnabled ? "translate-x-4" : "translate-x-0.5"
              }`}
            />
          </span>
        </button>
      </div>
    );
  };

  const renderWebToggle = () => {
    if (chatMode !== "text" || !onToggleWebEnabled) return null;

    return (
      <div className="flex items-center justify-center gap-2 mb-2">
        <button
          className={`inline-flex items-center gap-2 py-1 px-2 rounded-full border text-[0.72rem] transition-all duration-200 ${
            webEnabled
              ? "bg-neon-subtle text-neon border-neon/40"
              : "bg-glass-bg text-txt-secondary border-glass-border hover:border-neon/30 hover:text-neon"
          }`}
          onClick={() => onToggleWebEnabled(!webEnabled)}
          title="Toggle web search"
        >
          <span className="font-semibold">Web {webEnabled ? "On" : "Off"}</span>
          <span
            className={`relative inline-flex h-4 w-8 rounded-full transition-colors ${
              webEnabled ? "bg-neon" : "bg-void-700"
            }`}
          >
            <span
              className={`absolute top-0.5 h-3 w-3 rounded-full bg-white transition-transform ${
                webEnabled ? "translate-x-4" : "translate-x-0.5"
              }`}
            />
          </span>
        </button>
        {webEnabled && (
          <div className="inline-flex rounded-full border border-glass-border overflow-hidden text-[0.68rem]">
            <button
              className={`px-2 py-0.5 transition-colors ${
                webDepth === "snippets"
                  ? "bg-neon-subtle text-neon"
                  : "bg-glass-bg text-txt-muted hover:text-txt"
              }`}
              onClick={() => onWebDepthChange?.("snippets")}
            >
              Snippets
            </button>
            <button
              className={`px-2 py-0.5 transition-colors ${
                webDepth === "full"
                  ? "bg-neon-subtle text-neon"
                  : "bg-glass-bg text-txt-muted hover:text-txt"
              }`}
              onClick={() => onWebDepthChange?.("full")}
            >
              Full
            </button>
          </div>
        )}
      </div>
    );
  };

  const renderVisionAttachment = () => {
    if (chatMode !== "vision" || !visionImagePreview) return null;

    return (
      <div className="w-full mb-2">
        <div className="inline-flex items-center gap-2 py-1.5 pl-1.5 pr-2 rounded-xl bg-void-700 border border-glass-border max-w-full">
          <button
            className="shrink-0 w-11 h-11 rounded-lg overflow-hidden border border-glass-border hover:border-neon transition-colors duration-150"
            onClick={() => {
              setPreviewModal({ src: visionImagePreview, title: visionFileName });
            }}
            title="Preview image"
          >
            <img src={visionImagePreview} alt="Vision attachment" className="w-full h-full object-cover" />
          </button>
          <button
            className="text-[0.74rem] text-txt-secondary max-w-[180px] truncate text-left hover:text-txt"
            onClick={() => {
              setPreviewModal({ src: visionImagePreview, title: visionFileName });
            }}
            title={visionFileName}
          >
            {visionFileName}
          </button>
          <button
            className="ml-1 w-6 h-6 inline-flex items-center justify-center rounded-md text-txt-muted hover:text-danger hover:bg-danger/10"
            onClick={onClearVisionImage}
            title="Remove image"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
      </div>
    );
  };

  const renderDirectDocumentAttachments = () => {
    if (chatMode !== "text" || directDocumentPaths.length === 0) return null;

    return (
      <div className="w-full mb-2">
        <div className="flex items-start gap-2 mb-1.5">
          <span className="text-[0.7rem] text-txt-muted">
            Direct docs ({directDocumentPaths.length})
          </span>
          <button
            className="text-[0.68rem] text-txt-muted hover:text-danger"
            onClick={onClearDirectDocuments}
            title="Clear all attached documents"
          >
            Clear all
          </button>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {directDocumentPaths.map((path) => {
            const name = path.split(/[/\\]/).pop() ?? "document";
            return (
              <span
                key={path}
                className="inline-flex items-center gap-1 py-1 px-2 rounded-lg bg-void-700 border border-glass-border text-[0.68rem] text-txt-secondary max-w-55"
                title={path}
              >
                <span className="truncate">{name}</span>
                <button
                  className="w-4 h-4 inline-flex items-center justify-center rounded text-txt-muted hover:text-danger"
                  onClick={() => onRemoveDirectDocument?.(path)}
                  title="Remove document"
                >
                  <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="18" y1="6" x2="6" y2="18" />
                    <line x1="6" y1="6" x2="18" y2="18" />
                  </svg>
                </button>
              </span>
            );
          })}
        </div>
      </div>
    );
  };

  // ─── Centered Welcome State (Claude/Copilot style) ───
  if (!hasMessages) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center relative px-6">
        {/* Animated orb */}
        <div className="welcome-orb" />

        {/* Brand & Greeting */}
        <div className="relative z-10 flex flex-col items-center mb-8">
          <img
            src="/logo-dark.png"
            alt="NELA"
            className="w-14 h-14 rounded-2xl object-contain shadow-[0_4px_30px_rgba(0,212,255,0.3)] mb-4"
            draggable={false}
          />
          <h2 className="text-2xl font-bold text-txt m-0 mb-1">What can I help with?</h2>
          <p className="text-sm text-txt-muted m-0">Ask anything — chat, analyze images, or explore your documents</p>
        </div>

        {/* Centered Input */}
        <div className="relative z-10 w-full max-w-2xl">
          {renderRagToggle()}
          {renderWebToggle()}

          {/* RAG doc indicators */}
          {showRagControls && (
            <div className="flex items-center gap-2 mb-2 justify-center">
              <button
                className={`glass-btn inline-flex items-center gap-1.5 py-1 px-3 text-[0.78rem] font-medium rounded-full cursor-pointer transition-all duration-200 border backdrop-blur-md ${docPanelOpen ? "bg-neon-subtle text-neon border-neon/30 shadow-[0_0_12px_rgba(0,212,255,0.12)]" : "bg-glass-bg text-txt-secondary border-glass-border hover:border-neon hover:text-neon hover:shadow-[0_0_12px_rgba(0,212,255,0.08)]"}`}
                onClick={onToggleDocPanel}
                title="Toggle knowledge base panel"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                  <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
                </svg>
                {ragDocs.length > 0 ? `${ragDocs.length} file${ragDocs.length !== 1 ? "s" : ""} loaded` : "Knowledge Base"}
              </button>
              {ragIngesting && (
                <span className="inline-flex items-center gap-1 text-[0.72rem] text-warning">
                  <svg className="spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 1 1-6.219-8.56" /></svg>
                  Ingesting...
                </span>
              )}
              {enrichmentStatus && (
                <span className="inline-flex items-center gap-1 text-[0.72rem] text-success">
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
                  {enrichmentStatus}
                </span>
              )}
            </div>
          )}

          <div className="input-wrapper glass-strong flex flex-col gap-2 rounded-2xl px-3 py-2 transition-all duration-200 shadow-[0_4px_24px_rgba(0,0,0,0.3)] focus-within:border-neon focus-within:shadow-[0_0_24px_rgba(0,212,255,0.15),0_4px_24px_rgba(0,0,0,0.3)]">
            {renderVisionAttachment()}
            {renderDirectDocumentAttachments()}
            <div className="flex items-end gap-2">
            {showAttachButton && (
              <div className="relative" ref={attachMenuRef}>
                <button
                  className="glass-btn flex items-center justify-center w-9 h-9 bg-glass-bg border border-glass-border text-txt-muted cursor-pointer rounded-lg transition-all duration-200 backdrop-blur-sm hover:text-neon hover:border-neon/30 hover:shadow-[0_0_8px_rgba(0,212,255,0.1)] disabled:opacity-40"
                  onClick={() => setShowAttachMenu(!showAttachMenu)}
                  title={
                    chatMode === "vision"
                      ? "Upload image"
                      : chatMode === "text" && !ragEnabled
                        ? "Attach documents directly to the model"
                        : "Add documents to knowledge base"
                  }
                  disabled={isLoading}
                  data-tour="attach-button"
                >
                  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                    style={{ transform: showAttachMenu ? "rotate(45deg)" : "none", transition: "transform 0.2s" }}>
                    <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
                  </svg>
                </button>

                {showAttachMenu && (
                  <div className="animate-attach-menu absolute bottom-full left-0 mb-2 w-[220px] rounded-xl bg-void-700/80 backdrop-blur-xl border border-glass-border shadow-[0_8px_32px_rgba(0,0,0,0.5)] p-1 z-50">
                    {renderAttachMenu()}
                  </div>
                )}
              </div>
            )}

            <textarea
              value={inputObj}
              onChange={(e) => setInputObj(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder={placeholder}
              rows={1}
              className="flex-1 bg-transparent border-none outline-none text-txt text-[0.92rem] py-2 px-1 min-h-[40px] max-h-[200px] resize-none leading-relaxed font-inherit placeholder:text-txt-muted"
              data-tour="chat-input"
            />
            {/* Voice input button - allows speaking instead of typing */}
            <VoiceInputButton
              onTranscript={(text) => {
                setInputObj((prev) => (prev ? prev + " " + text : text));
              }}
              disabled={isLoading}
            />
            {onToggleThinking && (
              <button
                className={`glass-btn flex items-center gap-1.5 h-10 px-2.5 rounded-xl border transition-all duration-200 ${thinkingEnabled ? "bg-neon/10 border-neon/30 text-neon" : "bg-glass-bg border-glass-border text-txt-muted hover:text-txt hover:border-glass-border-hover"}`}
                onClick={onToggleThinking}
                title={thinkingEnabled ? "Disable thinking (faster)" : "Enable thinking"}
                disabled={isLoading}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 2a10 10 0 0 0-9.95 9 10 10 0 0 0 19.4 3.1 10 10 0 0 0-9.45-12.1zM12 6a6 6 0 0 1 5.82 7.5" />
                  <circle cx="12" cy="12" r="1" fill="currentColor" />
                </svg>
                <span className="text-[0.74rem] font-medium leading-none">{thinkingEnabled ? "On" : "Off"}</span>
              </button>
            )}
            <div className="relative" ref={modeMenuRef}>
              <button
                className="glass-btn flex items-center gap-1.5 h-10 px-2.5 rounded-xl bg-glass-bg border border-glass-border text-txt-muted cursor-pointer transition-all duration-200 hover:text-neon hover:border-neon/30 hover:shadow-[0_0_10px_rgba(0,212,255,0.12)]"
                onClick={() => setShowModeMenu((v) => !v)}
                title="Switch mode"
                disabled={isLoading}
                data-tour="mode-switch"
              >
                <CurrentModeIcon size={16} strokeWidth={1.9} />
                <span className="text-[0.74rem] font-medium leading-none">{currentModeLabel}</span>
              </button>

              {showModeMenu && (
                <div className="animate-attach-menu absolute bottom-full right-0 mb-2 w-[180px] rounded-xl bg-void-700/90 backdrop-blur-xl border border-glass-border shadow-[0_8px_32px_rgba(0,0,0,0.5)] p-1 z-50">
                  {modeOptions.map((option) => {
                    const active = option.mode === currentMode;
                    return (
                      <button
                        key={option.mode}
                        className={`w-full flex items-center gap-2 py-2 px-3 rounded-lg text-sm transition-all duration-150 ${active ? "bg-neon-subtle text-neon" : "text-txt-secondary hover:bg-glass-hover hover:text-txt"}`}
                        onClick={() => {
                          onSelectMode?.(option.mode);
                          setShowModeMenu(false);
                        }}
                      >
                        {(() => {
                          const OptionIcon = MODE_ICON_MAP[option.mode] ?? MessageSquare;
                          return <OptionIcon size={15} strokeWidth={1.9} />;
                        })()}
                        {option.label}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
            <button className="send-btn flex items-center justify-center w-10 h-10 rounded-xl bg-neon text-void-900 border border-neon/50 cursor-pointer transition-all duration-200 shadow-[0_0_16px_rgba(0,212,255,0.2)] hover:bg-neon-hover hover:shadow-[0_0_24px_rgba(0,212,255,0.35)] disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none shrink-0" onClick={handleSend} disabled={!inputObj.trim()}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
            </div>
          </div>
        </div>

        {previewModal && (
          <div className="fixed inset-0 z-[70] bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
            <div className="relative w-full max-w-6xl h-[90vh] bg-void-800 border border-glass-border rounded-2xl overflow-hidden flex flex-col">
              <div className="h-12 shrink-0 border-b border-glass-border flex items-center justify-between px-3">
                <span className="text-[0.78rem] text-txt-secondary truncate max-w-[75%]" title={previewModal.title}>{previewModal.title}</span>
                <button className="glass-btn w-8 h-8 rounded-lg text-txt-secondary hover:text-danger" onClick={() => setPreviewModal(null)} title="Close">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></svg>
                </button>
              </div>
              <div className="flex-1 overflow-hidden flex items-center justify-center p-4 sm:p-6">
                <img
                  src={previewModal.src}
                  alt="Vision preview"
                  className="max-w-full max-h-full object-contain"
                />
              </div>
            </div>
          </div>
        )}
      </div>
    );
  }

  // ─── Normal Chat State ───
  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="messages-area flex-1 overflow-y-auto px-6 py-4">
        {messages.map((msg, idx) => {
          const isNew = idx >= prevMsgCount;
          return (
            <React.Fragment key={idx}>
              <div className={`${isNew ? "animate-msg-fade" : ""} flex gap-3 mb-6 max-w-3xl mx-auto ${msg.role === "user" ? "justify-end" : ""}`}>
                {msg.role === "user" ? (
                  <>
                    <div className="flex flex-col items-end flex-1 min-w-0">
                      <div className="py-3 px-4 rounded-2xl rounded-tr-sm text-[0.9rem] leading-relaxed text-txt max-w-[85%] bg-[rgba(0,212,255,0.04)] backdrop-blur-lg border border-[rgba(0,212,255,0.1)] shadow-[0_4px_20px_rgba(0,0,0,0.25),inset_0_1px_0_rgba(255,255,255,0.04)]">
                        {msg.visionImage && (
                          <div className="mb-2.5">
                            <VisionMessageImage
                              imagePath={msg.visionImage.path}
                              imageName={msg.visionImage.name}
                              onOpen={(src, title) => {
                                setPreviewModal({ src, title });
                              }}
                            />
                          </div>
                        )}
                        {msg.directDocuments && msg.directDocuments.length > 0 && (
                          <div className="mb-2 flex flex-wrap gap-1.5">
                            {msg.directDocuments.map((doc, docIdx) => (
                              <span
                                key={`${doc.path}-${docIdx}`}
                                className="inline-flex items-center gap-1 py-0.5 px-2 rounded-md bg-void-800/70 border border-glass-border text-[0.68rem] text-txt-secondary"
                                title={doc.path}
                              >
                                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                  <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                                  <polyline points="14 2 14 8 20 8" />
                                </svg>
                                <span className="max-w-45 truncate">{doc.name}</span>
                              </span>
                            ))}
                          </div>
                        )}
                        {msg.content}
                      </div>
                    </div>
                    <div className="w-8 h-8 rounded-xl bg-neon-subtle text-neon flex items-center justify-center shrink-0 border border-neon/15 shadow-[0_2px_8px_rgba(0,212,255,0.1)]">
                      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <path d="M12 12c2.761 0 5-2.239 5-5s-2.239-5-5-5-5 2.239-5 5 2.239 5 5 5z" fill="currentColor" />
                        <path d="M4 20c0-3.3137 2.6863-6 6-6h4c3.3137 0 6 2.6863 6 6v1H4v-1z" fill="currentColor" />
                      </svg>
                    </div>
                  </>
                ) : (
                  <>
                    <img
                      src="/logo-dark.png"
                      alt="NELA"
                      className="w-8 h-8 rounded-xl object-contain shrink-0 shadow-[0_2px_12px_rgba(0,212,255,0.25)]"
                      draggable={false}
                    />
                    <div className="flex-1 min-w-0">
                      <div className="text-[0.9rem] leading-relaxed text-txt glass rounded-2xl rounded-tl-sm py-3 px-4 shadow-[0_4px_20px_rgba(0,0,0,0.2),inset_0_1px_0_rgba(255,255,255,0.03)]">
                        {msg.thinking && <ThinkingBox thinking={msg.thinking} />}
                        <MarkdownRenderer content={msg.content} />
                        {mediaAssets[idx] && <MediaGallery assets={mediaAssets[idx]} />}
                        <div className="flex items-center gap-1 mt-2 pt-1.5">
                          <CopyMsgButton text={msg.content} />
                          {/* Read response aloud button */}
                          <SpeakButton text={msg.content} compact />
                          {msg.generateTime !== undefined && (
                            <span className="text-[0.7rem] text-txt-muted ml-1" title={msg.firstTokenTime !== undefined ? `Generated in ${msg.generateTime}s\nFirst token in ${msg.firstTokenTime}s` : `Generated in ${msg.generateTime}s`}>
                              Generated in {msg.generateTime}s {msg.firstTokenTime !== undefined && `• First token in ${msg.firstTokenTime}s`}
                            </span>
                          )}
                        </div>
                      </div>
                      {/* Render AudioPlayer after assistant message if audioUrl is present */}
                      {msg.audioUrl && (
                        <div className="mt-2 flex items-center gap-2">
                          <AudioPlayer src={msg.audioUrl} key={msg.audioUrl} />
                          <button
                            className="px-2 py-1 text-xs rounded bg-neon/10 text-neon border border-neon/30 hover:bg-neon/20 transition ml-1"
                            style={{ marginLeft: 0 }}
                            onClick={() => saveAudioToSidebar && saveAudioToSidebar(idx)}
                            title="Save audio to sidebar"
                          >
                            Save
                          </button>
                        </div>
                      )}
                    </div>
                  </>
                )}
              </div>
            </React.Fragment>
          );
        })}

        {isLoading && (
          <div className="animate-msg-fade flex gap-3 mb-5 max-w-3xl mx-auto">
            <img
              src="/logo-dark.png"
              alt="NELA"
              className="w-8 h-8 rounded-xl object-contain shrink-0 shadow-[0_2px_10px_rgba(0,212,255,0.2)]"
              draggable={false}
            />
            <div className="flex-1 min-w-0 text-[0.9rem] leading-relaxed text-txt glass rounded-2xl rounded-tl-sm py-3 px-4">
              {streamingThinking && (
                <div className="mb-3 p-3 rounded-lg bg-black/20 border border-white/5 text-xs text-txt-muted leading-relaxed opacity-70">
                  <div className="flex items-center gap-2 mb-2">
                    <span className="font-medium">Thinking...</span>
                  </div>
                  <pre className="whitespace-pre-wrap font-mono">{streamingThinking}</pre>
                </div>
              )}
              {streamingContent ? (
                <MarkdownRenderer content={streamingContent} />
              ) : !streamingThinking ? (
                <div className="typing-dots flex gap-1.5 py-2">
                  <span></span><span></span><span></span>
                </div>
              ) : null}
            </div>
          </div>
        )}

        {/* Response Time Timer - Audio Mode */}
        {chatMode === "audio" && ttsGenerating && (
          <div className="flex items-center gap-2 py-1.5 px-3 rounded-full bg-neon-subtle border border-neon/20 max-w-3xl mx-auto text-sm text-txt-secondary">
            <div className="tts-timer-pulse" />
            <span>Generating speech... <span className="text-neon font-semibold tabular-nums">{ttsElapsedTime.toFixed(1)}s</span></span>
          </div>
        )}

        {/* Response Time Timer - Chat/Vision/RAG Modes */}
        {chatMode !== "audio" && generalGenerating && (
          <div className="flex items-center gap-2 py-1.5 px-3 rounded-full bg-neon-subtle border border-neon/20 max-w-3xl mx-auto text-sm text-txt-secondary">
            <div className="tts-timer-pulse" />
            <span>
              {chatMode === "vision" && "Analyzing image... "}
              {chatMode === "rag" && "Processing query... "}
              {chatMode === "text" && "Generating response... "}
              {chatMode === "mindmap" && "Building mindmap... "}
              <span className="text-neon font-semibold tabular-nums">{generalElapsedTime.toFixed(1)}s</span>
            </span>
          </div>
        )}

        {/* Audio Player (legacy block removed; now only rendered inline after assistant messages) */}

        {/* Response time completion */}
        {chatMode !== "audio" && generalGenerationTime !== null && !generalGenerating && (
          <div className="flex items-center gap-1.5 py-1 px-3 rounded-full max-w-3xl mx-auto text-[0.72rem] text-success">
            <span>✓</span>
            <span>
              {chatMode === "vision" && `Analyzed in ${generalGenerationTime.toFixed(1)}s`}
              {chatMode === "rag" && `Processed in ${generalGenerationTime.toFixed(1)}s`}
              {chatMode === "text" && `Generated in ${generalGenerationTime.toFixed(1)}s`}
              {chatMode === "mindmap" && `Mindmap built in ${generalGenerationTime.toFixed(1)}s`}
            </span>
          </div>
        )}

        {/* Cancelled notice */}
        {cancelled && (
          <div className="text-center py-1.5 text-[0.78rem] text-txt-muted">⏹ Response stopped</div>
        )}

        {/* Mode switch notice */}
        {modeSwitchNotice && (
          <div className="text-center py-1.5 text-[0.78rem] text-txt-muted">{modeSwitchNotice}</div>
        )}

        <div ref={endRef} />
      </div>

      {/* ── Web Sources Panel ── */}
      {webSearchResult && webSearchResult.results.length > 0 && (
        <WebSourcesPanel webSearchResult={webSearchResult} />
      )}

      {/* ── Input Area ── */}
      <div className="px-6 py-3 shrink-0 border-t border-glass-border bg-void-900">
        {renderRagToggle()}
        {renderWebToggle()}

        {/* RAG doc indicators */}
        {showRagControls && (
          <div className="flex items-center gap-2 mb-2 max-w-3xl mx-auto">
            <button
              className={`glass-btn inline-flex items-center gap-1.5 py-1 px-3 text-[0.78rem] font-medium rounded-full cursor-pointer transition-all duration-200 border backdrop-blur-md ${docPanelOpen ? "bg-neon-subtle text-neon border-neon/30 shadow-[0_0_12px_rgba(0,212,255,0.12)]" : "bg-glass-bg text-txt-secondary border-glass-border hover:border-neon hover:text-neon hover:shadow-[0_0_12px_rgba(0,212,255,0.08)]"}`}
              onClick={onToggleDocPanel}
              title="Toggle knowledge base panel"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
              </svg>
              {ragDocs.length > 0 ? `${ragDocs.length} file${ragDocs.length !== 1 ? "s" : ""} loaded` : "Knowledge Base"}
            </button>
            {ragIngesting && (
              <span className="inline-flex items-center gap-1 text-[0.72rem] text-warning">
                <svg className="spin" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 12a9 9 0 1 1-6.219-8.56" /></svg>
                Ingesting...
              </span>
            )}
            {enrichmentStatus && (
              <span className="inline-flex items-center gap-1 text-[0.72rem] text-success">
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 6L9 17l-5-5" /></svg>
                {enrichmentStatus}
              </span>
            )}
          </div>
        )}

          <div className="input-wrapper glass-strong flex flex-col gap-2 rounded-2xl px-3 py-2 max-w-3xl mx-auto transition-all duration-200 shadow-[0_4px_24px_rgba(0,0,0,0.3)] focus-within:border-neon focus-within:shadow-[0_0_24px_rgba(0,212,255,0.15),0_4px_24px_rgba(0,0,0,0.3)]">
            {renderVisionAttachment()}
            {renderDirectDocumentAttachments()}
            <div className="flex items-end gap-2">
          {showAttachButton && (
            <div className="relative" ref={attachMenuRef}>
              <button
                className="glass-btn flex items-center justify-center w-9 h-9 bg-glass-bg border border-glass-border text-txt-muted cursor-pointer rounded-lg transition-all duration-200 backdrop-blur-sm hover:text-neon hover:border-neon/30 hover:shadow-[0_0_8px_rgba(0,212,255,0.1)] disabled:opacity-40"
                onClick={() => setShowAttachMenu(!showAttachMenu)}
                title={
                  chatMode === "vision"
                    ? "Upload image"
                    : chatMode === "text" && !ragEnabled
                      ? "Attach documents directly to the model"
                      : "Add documents to knowledge base"
                }
                disabled={isLoading}
                data-tour="attach-button"
              >
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
                  style={{ transform: showAttachMenu ? "rotate(45deg)" : "none", transition: "transform 0.2s" }}>
                  <line x1="12" y1="5" x2="12" y2="19" /><line x1="5" y1="12" x2="19" y2="12" />
                </svg>
              </button>

              {showAttachMenu && (
                <div className="animate-attach-menu absolute bottom-full left-0 mb-2 w-[220px] rounded-xl bg-void-700/80 backdrop-blur-xl border border-glass-border shadow-[0_8px_32px_rgba(0,0,0,0.5)] p-1 z-50">
                  {renderAttachMenu()}
                </div>
              )}
            </div>
          )}

          <textarea
            value={inputObj}
            onChange={(e) => setInputObj(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            rows={1}
            className="flex-1 bg-transparent border-none outline-none text-txt text-[0.92rem] py-2 px-1 min-h-[40px] max-h-[200px] resize-none leading-relaxed font-inherit placeholder:text-txt-muted"
            data-tour="chat-input"
          />
          {/* Voice input button - allows speaking instead of typing */}
          <VoiceInputButton
            onTranscript={(text) => {
              setInputObj((prev) => (prev ? prev + " " + text : text));
            }}
            disabled={isLoading}
          />
          <div className="relative" ref={modeMenuRef}>
            <button
              className="glass-btn flex items-center gap-1.5 h-10 px-2.5 rounded-xl bg-glass-bg border border-glass-border text-txt-muted cursor-pointer transition-all duration-200 hover:text-neon hover:border-neon/30 hover:shadow-[0_0_10px_rgba(0,212,255,0.12)]"
              onClick={() => setShowModeMenu((v) => !v)}
              title="Switch mode"
              disabled={isLoading}
              data-tour="mode-switch"
            >
              <CurrentModeIcon size={16} strokeWidth={1.9} />
              <span className="text-[0.74rem] font-medium leading-none">{currentModeLabel}</span>
            </button>

            {showModeMenu && (
              <div className="animate-attach-menu absolute bottom-full right-0 mb-2 w-[180px] rounded-xl bg-void-700/90 backdrop-blur-xl border border-glass-border shadow-[0_8px_32px_rgba(0,0,0,0.5)] p-1 z-50">
                {modeOptions.map((option) => {
                  const active = option.mode === currentMode;
                  return (
                    <button
                      key={option.mode}
                      className={`w-full flex items-center gap-2 py-2 px-3 rounded-lg text-sm transition-all duration-150 ${active ? "bg-neon-subtle text-neon" : "text-txt-secondary hover:bg-glass-hover hover:text-txt"}`}
                      onClick={() => {
                        onSelectMode?.(option.mode);
                        setShowModeMenu(false);
                      }}
                    >
                      {(() => {
                        const OptionIcon = MODE_ICON_MAP[option.mode] ?? MessageSquare;
                        return <OptionIcon size={15} strokeWidth={1.9} />;
                      })()}
                      {option.label}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
          {isLoading ? (
            <button className="flex items-center justify-center w-10 h-10 rounded-xl bg-danger/80 backdrop-blur-sm text-white border border-danger/30 cursor-pointer transition-all duration-200 shadow-[0_0_12px_rgba(248,113,113,0.2)] hover:bg-danger hover:shadow-[0_0_20px_rgba(248,113,113,0.3)] shrink-0" onClick={onCancel} title="Stop generation">
              <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2" /></svg>
            </button>
          ) : (
            <button className="send-btn flex items-center justify-center w-10 h-10 rounded-xl bg-neon text-void-900 border border-neon/50 cursor-pointer transition-all duration-200 shadow-[0_0_16px_rgba(0,212,255,0.2)] hover:bg-neon-hover hover:shadow-[0_0_24px_rgba(0,212,255,0.35)] disabled:opacity-30 disabled:cursor-not-allowed disabled:shadow-none shrink-0" onClick={handleSend} disabled={!inputObj.trim()}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M5 12H19M19 12L12 5M19 12L12 19" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </button>
          )}
          </div>
        </div>
      </div>

      {previewModal && (
        <div className="fixed inset-0 z-[70] bg-black/75 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="relative w-full max-w-6xl h-[90vh] bg-void-800 border border-glass-border rounded-2xl overflow-hidden flex flex-col">
            <div className="h-12 shrink-0 border-b border-glass-border flex items-center justify-between px-3">
              <span className="text-[0.78rem] text-txt-secondary truncate max-w-[75%]" title={previewModal.title}>{previewModal.title}</span>
              <button className="flex items-center justify-center w-8 h-8 rounded-lg text-txt-secondary hover:text-danger transition-colors duration-200" onClick={() => setPreviewModal(null)} title="Close">
                <X size={16} strokeWidth={2} />
              </button>
            </div>
            <div className="flex-1 overflow-hidden flex items-center justify-center p-4 sm:p-6">
              <img
                src={previewModal.src}
                alt="Vision preview"
                className="max-w-full max-h-full object-contain"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
});

ChatWindow.displayName = "ChatWindow";

export default ChatWindow;
