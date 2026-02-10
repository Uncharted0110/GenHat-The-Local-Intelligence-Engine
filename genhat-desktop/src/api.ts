// API service for backend communication
// Base URL can be configured via environment or hardcoded

const BACKEND_URL = 'http://127.0.0.1:8081' // Changed to Local Llama Server Port

/**
 * Helper to call Local Llama Server
 */
async function callLocalLlama(messages: { role: string, content: string }[], stream = false, onToken?: (token: string) => void) {
  const response = await fetch(`${BACKEND_URL}/v1/chat/completions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      messages,
      max_tokens: 1024,
      stream
    })
  });
  
  if (!response.ok) {
     throw new Error(`Llama server error: ${response.statusText}`);
  }
  
  if (stream && onToken && response.body) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      
      const lines = buffer.split("\n");
      buffer = lines.pop() || ""; 

      for (const line of lines) {
        if (line.trim() === "") continue;
        if (line.trim() === "data: [DONE]") return;
        if (line.startsWith("data: ")) {
          try {
            const data = JSON.parse(line.slice(6));
             const content = data.choices?.[0]?.delta?.content;
             if (content) onToken(content);
          } catch (e) {
            console.error("Error parsing stream line", e);
          }
        }
      }
    }
    return { choices: [{ message: { content: "" } }] }; // Return empty dummy for callers expecting result
  } else {
    return response.json();
  }
}

export interface CachePDFsResponse {
  cache_key: string
  message: string
  pdf_count: number
  project_name: string
  reused: boolean
  empty?: boolean
}

export interface CacheStatusResponse {
  ready: boolean
  chunk_count?: number
  pdf_files?: string[]
  domain?: string
  project_name?: string
  file_progress?: Record<string, {
    index: number
    progress: number
    status: 'pending' | 'processing' | 'completed' | 'error'
    total_files: number
    error?: string
  }>
  processing?: boolean
}

export interface QueryPDFsResponse {
  metadata: {
    input_documents: string[]
    persona: string
    job_to_be_done: string
    domain: string
  }
  extracted_sections: Array<{
    document: string
    section_title: string
    refined_text: string
    page_number: number
    importance_rank: number
    bm25_score: number
    embedding_score: number
  }>
  subsection_analysis: Array<{
    document: string
    refined_text: string
    page_number: number
  }>
}

export interface AnalyzeChunksResponse {
  metadata: {
    input_documents: string[]
    persona: string
    job_to_be_done: string
    domain: string
    total_chunks_found: number
    chunks_analyzed: number
    gemini_model: string
    project_name: string
  }
  retrieval_results: Array<{
    document: string
    section_title: string
    content: string
    page_number: number
    hybrid_score: number
    bm25_score: number
    embedding_score: number
  }>
  gemini_analysis: Array<{
    chunk_index: number
    combined: boolean
    included_chunk_count: number
    included_sections: Array<{
      index: number
      document: string
      section_title: string
      page_number: number
      hybrid_score: number
      bm25_score: number
      embedding_score: number
    }>
    gemini_analysis: string
    analysis_timestamp: number
  }>
  summary: {
    top_insights: string[]
  }
  insight_id: string
}

export interface PodcastFromPromptResponse {
  insight_id: string
  title: string
  script: string
  analysis: string
  audio_url: string | null
  retrieved_chunk_count: number
  project_name: string
  persona: string
  prompt: string
  domain: string
}

/**
 * Generate podcast from prompt (retrieval + analysis + script + TTS)
 * Mocked for now
 */
export async function podcastFromPrompt(
  projectName: string,
  prompt: string,
  k: number = 5,
  persona: string = 'Podcast Host'
): Promise<PodcastFromPromptResponse> {
  return {
    insight_id: 'mock-podcast',
    title: 'Podcast feature coming soon to Local AI',
    script: 'This feature is currently being migrated to run locally.',
    analysis: 'N/A',
    audio_url: null,
    retrieved_chunk_count: 0,
    project_name: projectName,
    persona,
    prompt,
    domain: 'general'
  }
}

/**
 * Upload PDFs to backend and get a cache key
 * MOCKED for Local Migration
 */
export async function cachePDFs(files: File[], projectName: string = ''): Promise<CachePDFsResponse> {
  // Mock response since we don't have Rust PDF parsing yet
  return new Promise(resolve => {
    setTimeout(() => {
      resolve({
        cache_key: 'mock-cache-key',
        message: 'PDFs uploaded (Mocked)',
        pdf_count: files.length,
        project_name: projectName,
        reused: false
      });
    }, 1000);
  });
}

/**
 * Check if the PDF cache is ready for querying
 * MOCKED for Local Migration
 */
export async function checkCacheStatus(cacheKey: string): Promise<CacheStatusResponse> {
   return { ready: true, chunk_count: 0, processing: false };
}

/**
 * Query PDFs with a persona and task
 * MOCKED for Local Migration
 */
export async function queryPDFs(
  cacheKey: string,
  persona: string,
  task: string,
  k: number = 5
): Promise<QueryPDFsResponse> {
    return {
      metadata: { input_documents: [], persona, job_to_be_done: task, domain: 'general' },
      extracted_sections: [],
      subsection_analysis: []
    }
}

/**
 * Analyze chunks using Local Llama Model
 */
export async function analyzeChunksWithGemini(
  cacheKey: string,
  persona: string,
  task: string,
  k: number = 5,
  maxChunksToAnalyze: number = 5,
  analysisPrompt?: string,
  geminiModel: string = 'local-llama',
  onToken?: (token: string) => void
): Promise<AnalyzeChunksResponse> {
  console.log("Analyzing with Local Llama...", { persona, task });

  // 1. Construct Prompt
  const prompt = `You are a ${persona}. Your task is: ${task}.
  
  (Note: RAG Retrieval is not yet implemented in this local version. Please answer based on your general knowledge).
  `;

  // 2. Call Local Llama
  try {
     let fullReply = "";
     const data = await callLocalLlama(
       [ { role: 'user', content: prompt } ], 
       !!onToken, 
       (token) => {
          fullReply += token;
          if (onToken) onToken(token);
       }
     );
     
     const reply = fullReply || data.choices?.[0]?.message?.content || "No response from model.";
     
     // 3. Mock the strict return structure expected by the UI
     return {
        metadata: {
            input_documents: ["local-conversation"],
            persona,
            job_to_be_done: task,
            domain: "general",
            total_chunks_found: 0,
            chunks_analyzed: 0,
            gemini_model: "local-llama",
            project_name: "Local Project"
        },
        retrieval_results: [],
        gemini_analysis: [{
            chunk_index: 0,
            combined: true,
            included_chunk_count: 0,
            included_sections: [],
            gemini_analysis: reply, // <--- Injection point
            analysis_timestamp: Date.now()
        }],
        summary: { top_insights: ["Analysis generated by Local Llama"] },
        insight_id: "local-" + Date.now()
     };

  } catch (err: any) {
    console.error(err);
    throw new Error(`Local Llama failure: ${err.message}`);
  }
}

/**
 * Poll cache status until ready, with progress callback
 */
export async function waitForCacheReadyWithProgress(
  cacheKey: string,
  onProgress?: (status: CacheStatusResponse) => void,
  maxAttempts: number = 60,
  intervalMs: number = 1000
): Promise<CacheStatusResponse> {
  for (let i = 0; i < maxAttempts; i++) {
    const status = await checkCacheStatus(cacheKey)
    
    // Call progress callback if provided
    if (onProgress) {
      onProgress(status)
    }
    
    if (status.ready) {
      return status
    }
    
    // Wait before next attempt
    await new Promise(resolve => setTimeout(resolve, intervalMs))
  }
  
  throw new Error('Cache preparation timed out')
}

export interface RemovePDFResponse {
  cache_key: string
  message: string
  removed: boolean
  remaining_pdfs: number
  remaining_chunks?: number
}

/**
 * Remove a PDF from the project and recompute embeddings
 * Mocked
 */
export async function removePDF(
  projectName: string,
  filename: string
): Promise<RemovePDFResponse> {
  return { 
    cache_key: 'mock-key', 
    message: 'MOCKED: PDF removed', 
    removed: true, 
    remaining_pdfs: 0 
  };
}

/**
 * Export project cache (embeddings, chunks, meta, prompt cache)
 */
export interface ExportProjectCacheResponse {
  project_name: string
  meta: Record<string, any>
  chunks: Array<Record<string, any>>
  embeddings: {
    chunk_ids: string[]
    embeddings: number[][]
    model_name: string
  } | null
  prompt_cache: Array<{
    hash: string
    prompt: string
    response: string
    context: Record<string, any>
    metadata: Record<string, any>
    created_at: string
  }>
  export_timestamp: string
}

export async function exportProjectCache(projectName: string): Promise<ExportProjectCacheResponse> {
  // Mocked for local migration
  return {
    project_name: projectName,
    meta: {},
    chunks: [],
    embeddings: null,
    prompt_cache: [],
    export_timestamp: new Date().toISOString()
  };
}

/**
 * Import project cache (embeddings, chunks, meta, prompt cache)
 */
export interface ImportProjectCacheRequest {
  project_name: string
  meta: Record<string, any>
  chunks: Array<Record<string, any>>
  embeddings?: {
    chunk_ids: string[]
    embeddings: number[][]
    model_name: string
  } | null
  prompt_cache?: Array<{
    hash: string
    prompt: string
    response: string
    context: Record<string, any>
    metadata: Record<string, any>
    created_at: string
  }>
}

export interface ImportProjectCacheResponse {
  cache_key: string
  project_name: string
  message: string
  chunk_count: number
  embeddings_restored: boolean
  prompt_cache_restored: number
}

export async function importProjectCache(data: ImportProjectCacheRequest): Promise<ImportProjectCacheResponse> {
  // Mocked for local migration
  return {
    cache_key: 'mock-imported-key',
    project_name: data.project_name,
    message: 'MOCKED: Project imported',
    chunk_count: 0,
    embeddings_restored: false,
    prompt_cache_restored: 0
  };
}

/**
 * Mindmap tree data structure
 */
export interface MindmapTreeData {
  id: string
  label: string
  collapsed: boolean
  children: MindmapTreeData[]
}

/**
 * Generate mindmap JSON structure from document analysis
 */
export interface GenerateMindmapResponse {
  success: boolean
  mindmap: MindmapTreeData
  prompt: string
  chunks_used: number
  project_name: string
}

export async function generateMindmap(
  cacheKey: string,
  prompt: string,
  k: number = 10
): Promise<GenerateMindmapResponse> {
    return {
        success: true,
        mindmap: {id: "root", label: "Mock Root", collapsed: false, children: []},
        prompt: prompt,
        chunks_used: 0,
        project_name: "Mock Project"
    };
}

