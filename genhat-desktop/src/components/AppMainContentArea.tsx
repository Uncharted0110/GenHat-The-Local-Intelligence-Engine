import type {
  ChatMode,
  ChatSession,
  IngestionStatus,
  MindMapGraph,
} from "../types";
import PodcastTab from "./PodcastTab";
import ChatWindow from "./ChatWindow";
import MindMapOverlay from "./MindMapOverlay";
import PdfViewer from "./PdfViewer";
import DocumentViewer from "./DocumentViewer";
import PlaygroundMode from "./PlaygroundMode";

interface ModeOption {
  mode: ChatMode;
  label: string;
}

interface AppMainContentAreaProps {
  chatMode: ChatMode;
  ragDocs: IngestionStatus[];
  ragEnabled: boolean;
  modeOptions: ModeOption[];
  onSelectMode: (mode: ChatMode) => void;
  onToggleRagEnabled: (enabled: boolean) => void;
  webEnabled?: boolean;
  onToggleWebEnabled?: (enabled: boolean) => void;
  webDepth?: "snippets" | "full";
  onWebDepthChange?: (depth: "snippets" | "full") => void;
  activeSession: ChatSession | null;
  activeWorkspace: { id: string } | null;
  onSend: (text: string) => void;
  onCancel: () => void;
  placeholder: string;
  ragIngesting: boolean;
  enrichmentStatus: string | null;
  onIngestFile: () => void;
  onIngestDir: () => void;
  onAttachDirectDocuments: () => void;
  directDocumentPaths: string[];
  onRemoveDirectDocument: (path: string) => void;
  onClearDirectDocuments: () => void;
  onSelectVisionImage: () => void;
  visionImagePath: string | null;
  visionImagePreview: string | null;
  onClearVisionImage: () => void;
  docPanelOpen: boolean;
  onToggleDocPanel: () => void;
  modeSwitchNotice: string | null;
  onSaveAudioToSidebar: (msgIdx: number) => void;
  streamingThinking: string;
  thinkingEnabled: boolean;
  onToggleThinking: () => void;
  activeMindmapOverlay: {
    sessionId: string;
    mindmapId: string | null;
    isGenerating?: boolean;
    query?: string;
  } | null;
  activeMindmapGraph: MindMapGraph | null;
  onCloseMindmapOverlay: () => void;
  pdfLoading: boolean;
  pdfViewerData: {
    data: string;
    title: string;
  } | null;
  onClosePdfViewer: () => void;
  docViewerFile: {
    filePath: string;
    title: string;
  } | null;
  onCloseDocViewer: () => void;
  onExitPlayground?: () => void;
}

export default function AppMainContentArea({
  chatMode,
  ragDocs,
  ragEnabled,
  modeOptions,
  onSelectMode,
  onToggleRagEnabled,
  webEnabled,
  onToggleWebEnabled,
  webDepth,
  onWebDepthChange,
  activeSession,
  activeWorkspace,
  onSend,
  onCancel,
  placeholder,
  ragIngesting,
  enrichmentStatus,
  onIngestFile,
  onIngestDir,
  onAttachDirectDocuments,
  directDocumentPaths,
  onRemoveDirectDocument,
  onClearDirectDocuments,
  onSelectVisionImage,
  visionImagePath,
  visionImagePreview,
  onClearVisionImage,
  docPanelOpen,
  onToggleDocPanel,
  modeSwitchNotice,
  onSaveAudioToSidebar,
  streamingThinking,
  thinkingEnabled,
  onToggleThinking,
  activeMindmapOverlay,
  activeMindmapGraph,
  onCloseMindmapOverlay,
  pdfLoading,
  pdfViewerData,
  onClosePdfViewer,
  docViewerFile,
  onCloseDocViewer,
  onExitPlayground,
}: AppMainContentAreaProps) {
  return (
    <>
      {chatMode === "playground" ? (
        <PlaygroundMode onNavigateBack={onExitPlayground} />
      ) : chatMode === "podcast" ? (
        <PodcastTab
          hasDocuments={ragDocs.length > 0}
          modeOptions={modeOptions}
          currentMode={chatMode}
          onSelectMode={onSelectMode}
        />
      ) : !activeSession ? (
        <div className="flex-1 flex items-center justify-center text-txt-muted text-sm">
          {activeWorkspace
            ? "Open a chat from the left sidebar or create a new chat."
            : "No workspace selected. Create a workspace from the left sidebar."}
        </div>
      ) : (
        <ChatWindow
          key={activeSession.id}
          messages={activeSession.messages}
          streamingContent={activeSession.streamingContent}
          isLoading={activeSession.loading}
          onSend={onSend}
          onCancel={onCancel}
          cancelled={activeSession.cancelled}
          audioSrc={activeSession.audioOutput}
          audioOutputs={activeSession.audioOutputs}
          placeholder={placeholder}
          mediaAssets={activeSession.mediaAssets}
          ragDocs={ragDocs}
          ragIngesting={ragIngesting}
          enrichmentStatus={enrichmentStatus}
          onIngestFile={onIngestFile}
          onIngestDir={onIngestDir}
          onAttachDirectDocuments={onAttachDirectDocuments}
          directDocumentPaths={directDocumentPaths}
          onRemoveDirectDocument={onRemoveDirectDocument}
          onClearDirectDocuments={onClearDirectDocuments}
          onSelectVisionImage={onSelectVisionImage}
          visionImagePath={visionImagePath}
          visionImagePreview={visionImagePreview}
          onClearVisionImage={onClearVisionImage}
          onToggleDocPanel={onToggleDocPanel}
          chatMode={chatMode}
          ragEnabled={ragEnabled}
          onToggleRagEnabled={onToggleRagEnabled}
          webEnabled={webEnabled}
          onToggleWebEnabled={onToggleWebEnabled}
          webDepth={webDepth}
          onWebDepthChange={onWebDepthChange}
          webSearchResult={activeSession?.webSearchResult ?? null}
          showRagControls={chatMode === "text" || chatMode === "mindmap"}
          docPanelOpen={docPanelOpen}
          modeOptions={modeOptions}
          currentMode={chatMode}
          onSelectMode={onSelectMode}
          modeSwitchNotice={modeSwitchNotice}
          saveAudioToSidebar={onSaveAudioToSidebar}
          session={activeSession}
          streamingThinking={streamingThinking}
          thinkingEnabled={thinkingEnabled}
          onToggleThinking={onToggleThinking}
        />
      )}

      {activeMindmapOverlay && (activeMindmapGraph || activeMindmapOverlay.isGenerating) && (
        <MindMapOverlay
          graph={activeMindmapGraph}
          isGenerating={!!activeMindmapOverlay.isGenerating}
          query={activeMindmapOverlay.query}
          onClose={onCloseMindmapOverlay}
        />
      )}

      {pdfLoading && (
        <div className="absolute inset-0 z-[55] bg-void-900/80 backdrop-blur-sm flex flex-col items-center justify-center gap-3 text-txt-muted text-sm">
          <div className="pdf-spinner" />
          <span>Loading PDF...</span>
        </div>
      )}

      {pdfViewerData && (
        <PdfViewer
          pdfData={pdfViewerData.data}
          title={pdfViewerData.title}
          onClose={onClosePdfViewer}
        />
      )}

      {docViewerFile && (
        <DocumentViewer
          key={docViewerFile.filePath}
          filePath={docViewerFile.filePath}
          title={docViewerFile.title}
          onClose={onCloseDocViewer}
        />
      )}
    </>
  );
}
