import type {
  ChatContextUsage,
  ChatMode,
  ChatSession,
  IngestionStatus,
  MindMapGraph,
  ModelFile,
  RegisteredModel,
  WorkspaceRecord,
} from "../types";
import type { DownloadStateMap } from "../app/types";
import type { RuntimeParamsTarget } from "./ActiveModelParamsDock";
import ChatTabBar from "./ChatTabBar";
import AppMainTopBar from "./AppMainTopBar";
import AppMainModeControls from "./AppMainModeControls";
import AppMainContentArea from "./AppMainContentArea";

interface ModeOption {
  mode: ChatMode;
  label: string;
}

interface AppMainContentProps {
  chatMode: ChatMode;
  openViewerSessions: ChatSession[];
  activeSessionId: string;
  onSelectSession: (sessionId: string) => void;
  onNewSession: () => void;
  onCloseSession: (sessionId: string) => void;
  onReorderSessions: (reordered: ChatSession[]) => void;
  currentModeConfig: {
    icon: React.ElementType;
    label: string;
    desc: string;
  };
  workspaces: WorkspaceRecord[];
  activeWorkspace: WorkspaceRecord | null;
  onSelectWorkspace: (workspaceId: string) => void;
  onCreateWorkspace: () => void;
  onDeleteWorkspace: (workspaceId: string) => void;
  onRenameWorkspace: (workspaceId: string, name: string) => void;
  workspaceBusy: boolean;
  modelLoadingStatus: {
    loading: boolean;
    modelId: string;
    message: string;
  };
  models: ModelFile[];
  selectedModel: string;
  onModelChange: (path: string) => void;
  onAddModel: () => void;
  onDownloadModel: (modelId: string) => void;
  onCancelDownload: (modelId: string) => void;
  onUninstallModel: (modelId: string) => void;
  onConfirmAction: (
    title: string,
    message: string,
    confirmLabel?: string,
    cancelLabel?: string
  ) => Promise<boolean>;
  downloads: DownloadStateMap;
  ttsEngines: RegisteredModel[];
  selectedTtsEngine: string;
  onSelectTtsEngine: (engineId: string) => void;
  visionModels: RegisteredModel[];
  selectedVisionModel: string;
  onSelectVisionModel: (modelId: string) => void;
  onAddVisionModel: () => void;
  activeRuntimeParamTarget: RuntimeParamsTarget | null;
  paramsDockOpen: boolean;
  onToggleParamsDock: () => void;
  contextUsage: ChatContextUsage | null;
  onCompactContext: () => void;
  canCompactContext: boolean;
  isCompactingContext: boolean;
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

export default function AppMainContent({
  chatMode,
  openViewerSessions,
  activeSessionId,
  onSelectSession,
  onNewSession,
  onCloseSession,
  onReorderSessions,
  currentModeConfig,
  workspaces,
  activeWorkspace,
  onSelectWorkspace,
  onCreateWorkspace,
  onDeleteWorkspace,
  onRenameWorkspace,
  workspaceBusy,
  modelLoadingStatus,
  models,
  selectedModel,
  onModelChange,
  onAddModel,
  onDownloadModel,
  onCancelDownload,
  onUninstallModel,
  onConfirmAction,
  downloads,
  ttsEngines,
  selectedTtsEngine,
  onSelectTtsEngine,
  visionModels,
  selectedVisionModel,
  onSelectVisionModel,
  onAddVisionModel,
  activeRuntimeParamTarget,
  paramsDockOpen,
  onToggleParamsDock,
  contextUsage,
  onCompactContext,
  canCompactContext,
  isCompactingContext,
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
}: AppMainContentProps) {
  return (
    <main className="flex-1 flex flex-col bg-void-900 min-w-0 relative">
      {chatMode !== "podcast" && (
        <ChatTabBar
          sessions={openViewerSessions}
          activeSessionId={activeSessionId}
          onSelectSession={onSelectSession}
          onNewSession={onNewSession}
          onCloseSession={onCloseSession}
          onReorderSessions={onReorderSessions}
        />
      )}

      <AppMainTopBar
        currentModeConfig={currentModeConfig}
        workspaces={workspaces}
        activeWorkspace={activeWorkspace}
        onSelectWorkspace={onSelectWorkspace}
        onCreateWorkspace={onCreateWorkspace}
        onDeleteWorkspace={onDeleteWorkspace}
        onRenameWorkspace={onRenameWorkspace}
        workspaceBusy={workspaceBusy}
        modelLoadingStatus={modelLoadingStatus}
        contextUsage={contextUsage}
        modeControls={(
          <AppMainModeControls
            chatMode={chatMode}
            models={models}
            selectedModel={selectedModel}
            onModelChange={onModelChange}
            onAddModel={onAddModel}
            onDownloadModel={onDownloadModel}
            onCancelDownload={onCancelDownload}
            onUninstallModel={onUninstallModel}
            onConfirmAction={onConfirmAction}
            downloads={downloads}
            ttsEngines={ttsEngines}
            selectedTtsEngine={selectedTtsEngine}
            onSelectTtsEngine={onSelectTtsEngine}
            visionModels={visionModels}
            selectedVisionModel={selectedVisionModel}
            onSelectVisionModel={onSelectVisionModel}
            onAddVisionModel={onAddVisionModel}
            activeRuntimeParamTarget={activeRuntimeParamTarget}
            paramsDockOpen={paramsDockOpen}
            onToggleParamsDock={onToggleParamsDock}
            contextUsage={contextUsage}
            onCompactContext={onCompactContext}
            canCompactContext={canCompactContext}
            isCompactingContext={isCompactingContext}
          />
        )}
      />

      <AppMainContentArea
        chatMode={chatMode}
        ragDocs={ragDocs}
        ragEnabled={ragEnabled}
        modeOptions={modeOptions}
        onSelectMode={onSelectMode}
        onToggleRagEnabled={onToggleRagEnabled}
        webEnabled={webEnabled}
        onToggleWebEnabled={onToggleWebEnabled}
        webDepth={webDepth}
        onWebDepthChange={onWebDepthChange}
        activeSession={activeSession}
        activeWorkspace={activeWorkspace}
        onSend={onSend}
        onCancel={onCancel}
        placeholder={placeholder}
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
        docPanelOpen={docPanelOpen}
        onToggleDocPanel={onToggleDocPanel}
        modeSwitchNotice={modeSwitchNotice}
        onSaveAudioToSidebar={onSaveAudioToSidebar}
        streamingThinking={streamingThinking}
        thinkingEnabled={thinkingEnabled}
        onToggleThinking={onToggleThinking}
        activeMindmapOverlay={activeMindmapOverlay}
        activeMindmapGraph={activeMindmapGraph}
        onCloseMindmapOverlay={onCloseMindmapOverlay}
        pdfLoading={pdfLoading}
        pdfViewerData={pdfViewerData}
        onClosePdfViewer={onClosePdfViewer}
        docViewerFile={docViewerFile}
        onCloseDocViewer={onCloseDocViewer}
        onExitPlayground={onExitPlayground}
      />
    </main>
  );
}
