//! Tauri IPC commands for the RAG subsystem.

use crate::rag::pipeline::{IngestionStatus, RagPipeline, RagResult};
use crate::commands::models::ProcessManagerState;
use crate::registry::types::TaskType;
use std::collections::HashSet;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::RwLock;
use tauri::State;

async fn resolve_runnable_chat_port(
    pm: &crate::process::ProcessManager,
) -> Result<(String, u16), String> {
    let active_id = pm.active_llm_id().await;
    let mut candidates = Vec::new();
    if !active_id.is_empty() {
        candidates.push(active_id);
    }

    for id in pm.find_models_for_task(&TaskType::Chat).await {
        if !candidates.contains(&id) {
            candidates.push(id);
        }
    }

    if candidates.is_empty() {
        return Err("No chat-capable model is registered".to_string());
    }

    let mut errors = Vec::new();
    for id in candidates {
        if let Some(port) = pm.get_llama_port(&id).await {
            return Ok((id, port));
        }

        match tokio::time::timeout(
            std::time::Duration::from_secs(25),
            pm.ensure_running(&id, false),
        )
        .await
        {
            Ok(Ok(_)) => {
                if let Some(port) = pm.get_llama_port(&id).await {
                    return Ok((id, port));
                }
                errors.push(format!("{id}: started but no port assigned"));
            }
            Ok(Err(e)) => errors.push(format!("{id}: {e}")),
            Err(_) => errors.push(format!("{id}: timed out while starting")),
        }
    }

    Err(format!(
        "No runnable chat model found for streaming: {}",
        errors.join(" | ")
    ))
}

/// Tauri-managed state wrapper for the RAG pipeline.
pub struct RagPipelineState(pub RwLock<Arc<RagPipeline>>);

impl RagPipelineState {
    pub fn active_pipeline(&self) -> Result<Arc<RagPipeline>, String> {
        self.0
            .read()
            .map(|guard| guard.clone())
            .map_err(|_| "RAG pipeline state lock poisoned".to_string())
    }

    /// Replace the current pipeline with a new one.
    /// IMPORTANT: This stops the old pipeline's enrichment worker before replacing.
    pub fn replace_pipeline(&self, pipeline: Arc<RagPipeline>) -> Result<(), String> {
        let mut guard = self
            .0
            .write()
            .map_err(|_| "RAG pipeline state lock poisoned".to_string())?;
        
        // Stop the old pipeline's enrichment worker before replacing
        guard.stop_enrichment();
        log::info!("Stopped old enrichment worker, replacing RAG pipeline");
        
        *guard = pipeline;
        Ok(())
    }

    pub fn active_data_dir(&self) -> Result<std::path::PathBuf, String> {
        let guard = self
            .0
            .read()
            .map_err(|_| "RAG pipeline state lock poisoned".to_string())?;

        guard
            .db
            .path
            .parent()
            .map(|p| p.to_path_buf())
            .ok_or_else(|| "Unable to determine active RAG data directory".to_string())
    }
}

/// Response from the streaming RAG retrieve command.
/// Frontend uses sources immediately, then streams the answer from llama-server SSE.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RagStreamSetup {
    /// Source chunks retrieved for the query.
    pub sources: Vec<crate::rag::pipeline::SourceChunk>,
    /// The prompt to send to llama-server (augmented with context, or raw query).
    pub prompt: String,
    /// The llama-server port to stream from.
    pub llama_port: u16,
    /// Whether the classifier decided no retrieval was needed.
    pub no_retrieval: bool,
}

/// Metadata for one directly attached document used in a prompt.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DirectDocumentUsed {
    /// Original absolute file path.
    pub file_path: String,
    /// Display title (usually filename / parsed title).
    pub title: String,
    /// Number of characters included from this file.
    pub chars_used: usize,
    /// Whether this file content had to be truncated.
    pub truncated: bool,
}

/// Setup payload for direct document prompting (non-RAG).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DirectDocumentPromptSetup {
    /// Prompt to send directly to the chat model.
    pub prompt: String,
    /// Files successfully included in the prompt.
    pub documents: Vec<DirectDocumentUsed>,
    /// Non-fatal issues (unsupported/failed files, empty extracts, etc.).
    pub warnings: Vec<String>,
    /// Whether any content was truncated due to budget limits.
    pub truncated: bool,
}

fn clip_chars(text: &str, max_chars: usize) -> (String, bool) {
    if max_chars == 0 {
        return (String::new(), !text.is_empty());
    }

    let mut iter = text.char_indices();
    if let Some((idx, _)) = iter.nth(max_chars) {
        return (format!("{}\n[...]", &text[..idx]), true);
    }

    (text.to_string(), false)
}

fn build_direct_document_prompt(
    query: &str,
    file_paths: &[String],
    max_chars_per_document: usize,
    max_total_chars: usize,
) -> Result<DirectDocumentPromptSetup, String> {
    let mut unique_paths = Vec::new();
    let mut seen = HashSet::new();
    for path in file_paths {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            continue;
        }
        if seen.insert(trimmed.to_string()) {
            unique_paths.push(trimmed.to_string());
        }
    }

    if unique_paths.is_empty() {
        return Err("No document paths were provided".to_string());
    }

    let mut remaining_chars = max_total_chars;
    let mut context_blocks: Vec<String> = Vec::new();
    let mut documents = Vec::new();
    let mut warnings = Vec::new();
    let mut any_truncated = false;

    for path_str in unique_paths {
        if remaining_chars < 400 {
            any_truncated = true;
            break;
        }

        let parsed = match crate::rag::parsers::parse_document(Path::new(&path_str)) {
            Ok(parsed) => parsed,
            Err(err) => {
                warnings.push(format!("Skipped {path_str}: {err}"));
                continue;
            }
        };

        let title = if parsed.title.trim().is_empty() {
            Path::new(&path_str)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("Document")
                .to_string()
        } else {
            parsed.title.trim().to_string()
        };

        let combined = parsed
            .sections
            .iter()
            .filter_map(|section| {
                let text = section.text.trim();
                if text.is_empty() {
                    return None;
                }

                if section.metadata.trim().is_empty() {
                    Some(text.to_string())
                } else {
                    Some(format!("[{}]\n{}", section.metadata.trim(), text))
                }
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        if combined.trim().is_empty() {
            warnings.push(format!("Skipped {path_str}: no extractable text content"));
            continue;
        }

        let budget = remaining_chars.min(max_chars_per_document);
        let (snippet, truncated_doc) = clip_chars(&combined, budget);
        let chars_used = snippet.chars().count();

        if chars_used == 0 {
            warnings.push(format!("Skipped {path_str}: no usable content within budget"));
            continue;
        }

        remaining_chars = remaining_chars.saturating_sub(chars_used);
        any_truncated = any_truncated || truncated_doc;

        documents.push(DirectDocumentUsed {
            file_path: path_str.clone(),
            title: title.clone(),
            chars_used,
            truncated: truncated_doc,
        });

        context_blocks.push(format!(
            "### Document: {title}\nPath: {path_str}\n\n{snippet}"
        ));
    }

    if context_blocks.is_empty() {
        return Err(if warnings.is_empty() {
            "No document content could be extracted for direct prompting".to_string()
        } else {
            format!(
                "No document content could be extracted for direct prompting. {}",
                warnings.join(" | ")
            )
        });
    }

    let prompt = format!(
        "Use ONLY the attached document excerpts to answer the user's question.\n\
         If the excerpts do not contain the answer, say you don't know.\n\
         Do NOT claim to have read files beyond the provided excerpts.\n\
         Keep your answer concise and grounded in the excerpts.\n\n\
         Attached document excerpts:\n\
         {}\n\n\
         User question: {}\n\n\
         Answer:",
        context_blocks.join("\n\n---\n\n"),
        query
    );

    Ok(DirectDocumentPromptSetup {
        prompt,
        documents,
        warnings,
        truncated: any_truncated,
    })
}

/// Ingest a single document into the RAG knowledge base.
#[tauri::command]
pub async fn ingest_document(
    path: String,
    state: State<'_, RagPipelineState>,
) -> Result<IngestionStatus, String> {
    let pipeline = state.active_pipeline()?;
    pipeline.ingest_document(&PathBuf::from(path)).await
}

/// Ingest all supported files in a directory.
#[tauri::command]
pub async fn ingest_folder(
    path: String,
    state: State<'_, RagPipelineState>,
) -> Result<Vec<IngestionStatus>, String> {
    let pipeline = state.active_pipeline()?;
    pipeline.ingest_folder(&PathBuf::from(path)).await
}

/// Query the RAG knowledge base.
#[tauri::command]
pub async fn query_rag(
    query: String,
    top_k: Option<usize>,
    state: State<'_, RagPipelineState>,
) -> Result<RagResult, String> {
    let pipeline = state.active_pipeline()?;
    let k = top_k.unwrap_or(5);
    pipeline.query(&query, k).await
}

/// List all ingested documents.
#[tauri::command]
pub async fn list_rag_documents(
    state: State<'_, RagPipelineState>,
) -> Result<Vec<IngestionStatus>, String> {
    let pipeline = state.active_pipeline()?;
    pipeline.list_documents()
}

/// Delete a document from the knowledge base.
#[tauri::command]
pub async fn delete_rag_document(
    doc_id: i64,
    state: State<'_, RagPipelineState>,
) -> Result<(), String> {
    let pipeline = state.active_pipeline()?;
    pipeline.delete_document(doc_id).await
}

/// Manually trigger a round of background enrichment.
#[tauri::command]
pub async fn enrich_rag_documents(
    batch_size: Option<usize>,
    state: State<'_, RagPipelineState>,
) -> Result<usize, String> {
    let pipeline = state.active_pipeline()?;
    let size = batch_size.unwrap_or(10);
    pipeline.enrich_pending(size).await
}

// ═══════════════════════════════════════════════════════════════════════════════
// RAPTOR Commands
// ═══════════════════════════════════════════════════════════════════════════════

/// Build a RAPTOR tree for a specific document (Phase 3).
#[tauri::command]
pub async fn build_raptor_tree(
    doc_id: i64,
    state: State<'_, RagPipelineState>,
) -> Result<crate::rag::raptor::RaptorTreeStatus, String> {
    let pipeline = state.active_pipeline()?;
    pipeline.build_raptor_tree(doc_id).await
}

/// Check if a document has a RAPTOR tree.
#[tauri::command]
pub async fn has_raptor_tree(
    doc_id: i64,
    state: State<'_, RagPipelineState>,
) -> Result<bool, String> {
    let pipeline = state.active_pipeline()?;
    pipeline.has_raptor_tree(doc_id)
}

/// Delete the RAPTOR tree for a document.
#[tauri::command]
pub async fn delete_raptor_tree(
    doc_id: i64,
    state: State<'_, RagPipelineState>,
) -> Result<(), String> {
    let pipeline = state.active_pipeline()?;
    pipeline.delete_raptor_tree(doc_id).await
}

/// Query using RAPTOR tree with confidence-aware traversal.
#[tauri::command]
pub async fn query_rag_with_raptor(
    doc_id: i64,
    query: String,
    top_k: Option<usize>,
    state: State<'_, RagPipelineState>,
) -> Result<RagResult, String> {
    let pipeline = state.active_pipeline()?;
    let k = top_k.unwrap_or(5);
    pipeline.query_with_raptor(doc_id, &query, k).await
}

// ═══════════════════════════════════════════════════════════════════════════════
// Streaming RAG Commands
// ═══════════════════════════════════════════════════════════════════════════════

/// Retrieve RAG sources and return everything needed for frontend SSE streaming.
/// The frontend receives sources immediately and streams the LLM answer directly
/// from the llama-server SSE endpoint.
#[tauri::command]
pub async fn query_rag_stream(
    query: String,
    top_k: Option<usize>,
    rag_state: State<'_, RagPipelineState>,
    pm_state: State<'_, ProcessManagerState>,
) -> Result<RagStreamSetup, String> {
    let pipeline = rag_state.active_pipeline()?;
    let k = top_k.unwrap_or(5);

    // Phase 1: Retrieval (classify → HyDE → search → RRF → grade)
    let retrieval = pipeline.retrieve_for_query(&query, k).await?;

    // Phase 2: Resolve a runnable chat model + llama-server port
    let (_model_id, llama_port) = resolve_runnable_chat_port(&pm_state.0).await?;

    // Determine the prompt to stream — use raw query for no_retrieval, else augmented prompt
    let prompt = if retrieval.no_retrieval {
        query
    } else if retrieval.augmented_prompt.is_empty() {
        // No sources found — return empty result so frontend can show a message
        return Ok(RagStreamSetup {
            sources: vec![],
            prompt: String::new(),
            llama_port,
            no_retrieval: false,
        });
    } else {
        retrieval.augmented_prompt
    };

    Ok(RagStreamSetup {
        sources: retrieval.sources,
        prompt,
        llama_port,
        no_retrieval: retrieval.no_retrieval,
    })
}

/// Build a direct-to-model prompt from attached document files, bypassing RAG.
///
/// This is useful when users want strict "uploaded file" grounding without
/// retrieval/classification hops.
#[tauri::command]
pub async fn prepare_direct_document_prompt(
    query: String,
    file_paths: Vec<String>,
    max_chars_per_document: Option<usize>,
    max_total_chars: Option<usize>,
) -> Result<DirectDocumentPromptSetup, String> {
    let trimmed_query = query.trim().to_string();
    if trimmed_query.is_empty() {
        return Err("Query cannot be empty".to_string());
    }

    let per_doc_limit = max_chars_per_document.unwrap_or(6_000).clamp(1_200, 32_000);
    let total_limit = max_total_chars.unwrap_or(20_000).clamp(4_000, 160_000);

    let join = tokio::task::spawn_blocking(move || {
        build_direct_document_prompt(&trimmed_query, &file_paths, per_doc_limit, total_limit)
    })
    .await
    .map_err(|e| format!("Direct document prompt task failed: {e}"))?;

    join
}

// ═══════════════════════════════════════════════════════════════════════════════
// Media Retrieval Commands
// ═══════════════════════════════════════════════════════════════════════════════

/// Two-phase media retrieval: given the LLM's response text, find images/tables
/// whose captions are semantically similar to the response content.
/// Returns media assets that should be displayed alongside the chat answer.
///
/// Automatically attempts to re-embed any unembedded media before searching
/// (recovers from embedding model not being loaded at ingestion time).
#[tauri::command]
pub async fn retrieve_media_for_response(
    response_text: String,
    top_k: Option<usize>,
    threshold: Option<f32>,
    state: State<'_, RagPipelineState>,
) -> Result<Vec<crate::rag::db::MediaAssetRecord>, String> {
    let pipeline = state.active_pipeline()?;
    let k = top_k.unwrap_or(3);
    let sim_threshold = threshold.unwrap_or(0.50);

    // Recover any media assets whose caption embedding failed during ingestion
    let re_embedded = pipeline.re_embed_unembedded_media().await;
    if re_embedded > 0 {
        log::info!("Recovered {re_embedded} media embeddings before retrieval");
    }

    Ok(pipeline
        .retrieve_media_for_response(&response_text, k, sim_threshold)
        .await)
}

/// Get all media assets for a specific document.
#[tauri::command]
pub async fn get_media_for_document(
    doc_id: i64,
    state: State<'_, RagPipelineState>,
) -> Result<Vec<crate::rag::db::MediaAssetRecord>, String> {
    let pipeline = state.active_pipeline()?;
    pipeline.db.get_media_for_doc(doc_id)
}

/// Streaming RAPTOR query — retrieve using RAPTOR tree and return setup for SSE streaming.
#[tauri::command]
pub async fn query_rag_with_raptor_stream(
    doc_id: i64,
    query: String,
    top_k: Option<usize>,
    rag_state: State<'_, RagPipelineState>,
    pm_state: State<'_, ProcessManagerState>,
) -> Result<RagStreamSetup, String> {
    let pipeline = rag_state.active_pipeline()?;
    let k = top_k.unwrap_or(5);

    // Phase 1: RAPTOR retrieval
    let retrieval = pipeline
        .retrieve_for_raptor_query(doc_id, &query, k)
        .await?;

    // Phase 2: Resolve a runnable chat model + llama-server port
    let (_model_id, llama_port) = resolve_runnable_chat_port(&pm_state.0).await?;

    let prompt = if retrieval.no_retrieval {
        query
    } else if retrieval.augmented_prompt.is_empty() {
        return Ok(RagStreamSetup {
            sources: vec![],
            prompt: String::new(),
            llama_port,
            no_retrieval: false,
        });
    } else {
        retrieval.augmented_prompt
    };

    Ok(RagStreamSetup {
        sources: retrieval.sources,
        prompt,
        llama_port,
        no_retrieval: retrieval.no_retrieval,
    })
}

/// Read a file (PDF or other) as base64 data URL for the frontend viewer.
#[tauri::command]
pub fn read_file_base64(path: String) -> Result<String, String> {
    use base64::engine::general_purpose::STANDARD;
    use base64::Engine;

    let p = std::path::Path::new(&path);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }

    let mime = match p
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .as_deref()
    {
        Some("pdf") => "application/pdf",
        Some("txt") => "text/plain",
        Some("md") => "text/markdown",
        Some("docx") => "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        Some("pptx") => "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        Some("xlsx") => "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        Some("xls") => "application/vnd.ms-excel",
        Some("ods") => "application/vnd.oasis.opendocument.spreadsheet",
        Some("odt") => "application/vnd.oasis.opendocument.text",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        Some("svg") => "image/svg+xml",
        Some("mp3") => "audio/mpeg",
        Some("wav") => "audio/wav",
        Some("ogg") => "audio/ogg",
        Some("m4a") => "audio/mp4",
        Some("flac") => "audio/flac",
        Some("json") => "application/json",
        Some("xml") => "application/xml",
        Some("html" | "htm") => "text/html",
        Some("css") => "text/css",
        Some("csv") => "text/csv",
        _ => "application/octet-stream",
    };

    let data = std::fs::read(p).map_err(|e| format!("Failed to read file: {e}"))?;
    let b64 = STANDARD.encode(&data);
    Ok(format!("data:{};base64,{}", mime, b64))
}

/// Read a file and return the raw text content (for text-based files).
/// For binary files (docx, pptx, xlsx, images, audio), use read_file_base64 instead.
#[tauri::command]
pub fn read_file_text(path: String) -> Result<String, String> {
    let p = std::path::Path::new(&path);
    if !p.exists() {
        return Err(format!("File not found: {}", p.display()));
    }
    std::fs::read_to_string(p).map_err(|e| format!("Failed to read file as text: {e}"))
}

// ═══════════════════════════════════════════════════════════════════════════════
// Watched-paths commands
// ═══════════════════════════════════════════════════════════════════════════════

/// Register a directory as a watched path for the active workspace.
/// The directory will be scanned for supported files on the next `trigger_scan` call.
#[tauri::command]
pub async fn add_watched_path(
    path: String,
    rag: State<'_, RagPipelineState>,
    workspace: State<'_, crate::commands::workspace::WorkspaceState>,
) -> Result<(), String> {
    let pipeline = rag.active_pipeline()?;
    let ws_id = workspace
        .0
        .active_workspace_id()
        .ok_or("No active workspace")?;
    let wm = crate::rag::watchman::WatchedPathsManager::new(pipeline.db.clone());
    wm.add_path(&ws_id, &path)
}

/// Unregister a watched path from the active workspace.
/// Already-ingested documents from this path are NOT removed from the KB.
#[tauri::command]
pub async fn remove_watched_path(
    path: String,
    rag: State<'_, RagPipelineState>,
    workspace: State<'_, crate::commands::workspace::WorkspaceState>,
) -> Result<(), String> {
    let pipeline = rag.active_pipeline()?;
    let ws_id = workspace
        .0
        .active_workspace_id()
        .ok_or("No active workspace")?;
    let wm = crate::rag::watchman::WatchedPathsManager::new(pipeline.db.clone());
    wm.remove_path(&ws_id, &path)
}

/// List all watched paths registered for the active workspace.
#[tauri::command]
pub async fn list_watched_paths(
    rag: State<'_, RagPipelineState>,
    workspace: State<'_, crate::commands::workspace::WorkspaceState>,
) -> Result<Vec<crate::rag::db::WatchedPathRecord>, String> {
    let pipeline = rag.active_pipeline()?;
    let ws_id = workspace
        .0
        .active_workspace_id()
        .ok_or("No active workspace")?;
    let wm = crate::rag::watchman::WatchedPathsManager::new(pipeline.db.clone());
    wm.list_paths(&ws_id)
}

/// Trigger an incremental scan of all watched paths for the active workspace.
/// Emits `rag:scan_progress` events during the scan.
/// Returns a summary of the scan result.
#[tauri::command]
pub async fn trigger_scan(
    rag: State<'_, RagPipelineState>,
    workspace: State<'_, crate::commands::workspace::WorkspaceState>,
    app_handle: tauri::AppHandle,
) -> Result<crate::rag::watchman::ScanResult, String> {
    let pipeline = rag.active_pipeline()?;
    let ws_id = workspace
        .0
        .active_workspace_id()
        .ok_or("No active workspace")?;
    let wm = crate::rag::watchman::WatchedPathsManager::new(pipeline.db.clone());
    Ok(wm.scan_diff(pipeline, &ws_id, app_handle).await)
}
