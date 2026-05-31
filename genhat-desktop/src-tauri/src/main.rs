#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

//! GenHat Desktop — main entry point.
//!
//! Slim bootstrap: loads config, initializes the process control module,
//! registers Tauri commands, and handles app lifecycle.

use app_lib::commands::audio::MicRecorderState;
use app_lib::commands::inference::TaskRouterState;
use app_lib::commands::models::ProcessManagerState;
use app_lib::commands::playground::PlaygroundState;
use app_lib::commands::rag::RagPipelineState;
use app_lib::commands::workspace::WorkspaceState;
use app_lib::commands::download::DownloadState;
use app_lib::process::ProcessManager;
use app_lib::rag::pipeline::RagPipeline;
use app_lib::registry::ModelRegistry;
use app_lib::router::TaskRouter;
use app_lib::workspace::WorkspaceManager;
#[cfg(all(target_os = "linux", not(debug_assertions)))]
use std::path::Path;
use std::sync::Arc;
use std::sync::RwLock;
use tauri::Manager;

#[cfg(all(target_os = "linux", not(debug_assertions)))]
fn copy_missing_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    if !src.is_dir() {
        return Ok(());
    }

    std::fs::create_dir_all(dst)?;

    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        let src_path = entry.path();
        let dst_path = dst.join(entry.file_name());

        if file_type.is_dir() {
            copy_missing_tree(&src_path, &dst_path)?;
        } else if file_type.is_file() && !dst_path.exists() {
            if let Some(parent) = dst_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::copy(src_path, dst_path)?;
        }
    }

    Ok(())
}

fn main() {
    tauri::Builder::default()
        .plugin(
            tauri_plugin_log::Builder::default()
                .level(log::LevelFilter::Info)
                .build(),
        )
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .setup(|app| {
            // 1. Load model registry from embedded models.toml
            let registry = Arc::new(
                ModelRegistry::load().expect("Failed to load model registry"),
            );

            // 2. Resolve models directory
            let models_dir = app_lib::commands::models::get_models_dir();

            #[cfg(all(target_os = "linux", not(debug_assertions)))]
            let models_dir = {
                let mut selected = models_dir;

                // Linux bundles install resources in /usr/lib/<ProductName>/..., which is
                // typically read-only for non-root users. Route models to app data unless
                // the user explicitly overrides GENHAT_MODEL_PATH.
                if std::env::var_os("GENHAT_MODEL_PATH").is_none() {
                    let bundled_models_dir = selected.clone();
                    let user_models_dir = app
                        .path()
                        .app_data_dir()
                        .unwrap_or_else(|_| std::path::PathBuf::from(".genhat_data"))
                        .join("models");

                    if let Err(e) = std::fs::create_dir_all(&user_models_dir) {
                        log::warn!(
                            "Failed to create writable models directory {}: {}",
                            user_models_dir.display(),
                            e
                        );
                    } else {
                        if bundled_models_dir.is_dir() && bundled_models_dir != user_models_dir {
                            if let Err(e) = copy_missing_tree(&bundled_models_dir, &user_models_dir)
                            {
                                log::warn!(
                                    "Failed to seed bundled models from {} to {}: {}",
                                    bundled_models_dir.display(),
                                    user_models_dir.display(),
                                    e
                                );
                            }
                        }

                        std::env::set_var("GENHAT_MODEL_PATH", &user_models_dir);
                        selected = user_models_dir;
                    }
                }

                selected
            };

            log::info!("Models directory: {}", models_dir.display());

            // Ensure the models directory exists at runtime. We create the directory
            // inside the app resources (or next to the executable) so users can
            // drop downloaded models there later.
            if let Err(e) = std::fs::create_dir_all(&models_dir) {
                log::warn!(
                    "Failed to create models directory {}: {}",
                    models_dir.display(),
                    e
                );
            }

            // 2b. Kill stale llama-server processes from previous app runs
            app_lib::backends::llama_server::kill_stale_llama_servers();

            // 3. Initialize the process manager
            let process_manager = Arc::new(ProcessManager::new(&registry, models_dir));

            // 4. Initialize the task router
            let router = Arc::new(TaskRouter::new(
                registry.clone(),
                process_manager.clone(),
            ));

            // 5. Start the lifecycle manager (background health checks + reaping)
            let pm_clone = process_manager.clone();
            app_lib::process::lifecycle::start_lifecycle_thread(pm_clone, 30);

            // 6. Auto-start models marked with auto_start = true
            let auto_models: Vec<String> = registry
                .auto_start_models()
                .iter()
                .map(|m| m.id.clone())
                .collect();

            if !auto_models.is_empty() {
                let pm_clone = process_manager.clone();
                tauri::async_runtime::spawn(async move {
                    for model_id in auto_models {
                        log::info!("Auto-starting model: {model_id}");
                        match pm_clone.ensure_running(&model_id, false).await {
                            Ok(id) => log::info!("Auto-started {model_id} (instance: {id})"),
                            Err(e) => log::warn!("Failed to auto-start {model_id}: {e}"),
                        }
                    }
                });
            }

            // 7. Initialize workspace manager
            let app_data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from(".genhat_data"));

            let workspace_manager = Arc::new(
                WorkspaceManager::new(&app_data_dir)
                    .expect("Failed to initialize workspace manager"),
            );

            // 8. Initialize RAG pipeline for active workspace cache
            let rag_dir = workspace_manager
                .active_rag_dir()
                .expect("Failed to resolve active workspace RAG directory");
            let rag_pipeline = Arc::new(
                RagPipeline::open(&rag_dir, router.clone())
                    .expect("Failed to initialize RAG pipeline"),
            );

            // Start background enrichment worker (with app handle for event emission)
            RagPipeline::start_enrichment_worker(rag_pipeline.clone(), app.handle().clone());

            // Start background scan for watched paths on launch
            {
                let db_for_scan = rag_pipeline.db.clone();
                let pipeline_for_scan = rag_pipeline.clone();
                let ws_id_for_scan = workspace_manager
                    .active_workspace_id()
                    .unwrap_or_else(|| "default".to_string());
                let handle_for_scan = app.handle().clone();
                tauri::async_runtime::spawn(async move {
                    let wm = app_lib::rag::watchman::WatchedPathsManager::new(db_for_scan);
                    wm.scan_diff(pipeline_for_scan, &ws_id_for_scan, handle_for_scan).await;
                });
            }

            // 9. Register state for Tauri commands
            app.manage(ProcessManagerState(process_manager));
            app.manage(TaskRouterState(router.clone()));
            app.manage(RagPipelineState(RwLock::new(rag_pipeline)));
            app.manage(WorkspaceState(workspace_manager));
            app.manage(DownloadState::default());
            app.manage(MicRecorderState::default());

            // 10. Initialize playground state
            match PlaygroundState::new(&app_data_dir) {
                Ok(pg_state) => {
                    let pg_store = pg_state.store.clone();
                    app.manage(pg_state);

                    // Start scheduler for auto-resume pipelines
                    let router_for_sched = router.clone();
                    let data_dir_for_sched = app_data_dir.clone();
                    let app_handle_for_sched = app.app_handle().clone();
                    tauri::async_runtime::spawn(async move {
                        app_lib::playground::scheduler::start_scheduler(
                            pg_store,
                            router_for_sched,
                            data_dir_for_sched,
                            app_handle_for_sched,
                        )
                        .await;
                    });
                }
                Err(e) => {
                    log::error!("Failed to initialize playground: {e}");
                }
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Legacy-compatible commands
            app_lib::commands::models::list_models,
            app_lib::commands::models::list_vision_models,
            app_lib::commands::models::discover_local_model_units,
            app_lib::commands::models::sync_discovered_models,
            app_lib::commands::models::switch_model,
            app_lib::commands::models::stop_llama,
            app_lib::commands::download::download_model,
            app_lib::commands::download::download_model_category,
            app_lib::commands::download::cancel_download,
            app_lib::commands::download::uninstall_model,
            app_lib::commands::download::download_custom_file,
            app_lib::commands::download::check_custom_file_exists,
            app_lib::commands::audio::generate_speech,
            app_lib::commands::audio::transcribe_audio_base64,
            app_lib::commands::audio::generate_speech_chunk,
            app_lib::commands::audio::start_mic_recording,
            app_lib::commands::audio::stop_mic_recording,
            // New unified commands
            app_lib::commands::models::list_registered_models,
            app_lib::commands::models::list_model_catalog,
            app_lib::commands::models::update_model_params,
            app_lib::commands::models::get_model_status,
            app_lib::commands::models::start_model,
            app_lib::commands::models::stop_model,
            app_lib::commands::models::get_llama_port,
            app_lib::commands::models::get_memory_usage,
            app_lib::commands::models::get_workspace_scope,
            app_lib::commands::models::read_image_base64,
            app_lib::commands::models::import_downloaded_model,
            app_lib::commands::models::unregister_custom_model,
            // Workspace commands
            app_lib::commands::workspace::list_workspaces,
            app_lib::commands::workspace::get_active_workspace,
            app_lib::commands::workspace::create_workspace,
            app_lib::commands::workspace::open_workspace,
            app_lib::commands::workspace::set_workspace_file,
            app_lib::commands::workspace::get_workspace_frontend_state,
            app_lib::commands::workspace::save_workspace_frontend_state,
            app_lib::commands::workspace::save_workspace_as_nela,
            app_lib::commands::workspace::save_workspace_nela,
            app_lib::commands::workspace::delete_workspace,
            app_lib::commands::workspace::rename_workspace,
            app_lib::commands::workspace::open_workspace_nela,
            app_lib::commands::workspace::get_rag_model_preferences,
            app_lib::commands::workspace::save_rag_model_preferences,
            app_lib::commands::inference::route_request,
            app_lib::commands::inference::compact_chat_context,
            app_lib::commands::inference::vision_chat,
            app_lib::commands::inference::vision_chat_stream,
            app_lib::commands::audio::transcribe_audio,
            app_lib::commands::audio::read_audio_base64,
            // RAG commands
            app_lib::commands::rag::ingest_document,
            app_lib::commands::rag::ingest_folder,
            app_lib::commands::rag::query_rag,
            app_lib::commands::rag::list_rag_documents,
            app_lib::commands::rag::delete_rag_document,
            app_lib::commands::rag::enrich_rag_documents,
            // RAPTOR commands
            app_lib::commands::rag::build_raptor_tree,
            app_lib::commands::rag::has_raptor_tree,
            app_lib::commands::rag::delete_raptor_tree,
            app_lib::commands::rag::query_rag_with_raptor,
            // Streaming RAG commands
            app_lib::commands::rag::query_rag_stream,
            app_lib::commands::rag::query_rag_with_raptor_stream,
            app_lib::commands::rag::prepare_direct_document_prompt,
            // Media retrieval commands
            app_lib::commands::rag::retrieve_media_for_response,
            app_lib::commands::rag::get_media_for_document,
            // File viewer commands
            app_lib::commands::rag::read_file_base64,
            app_lib::commands::rag::read_file_text,
            // Watched-paths / auto-discovery commands
            app_lib::commands::rag::add_watched_path,
            app_lib::commands::rag::remove_watched_path,
            app_lib::commands::rag::list_watched_paths,
            app_lib::commands::rag::trigger_scan,
            // Podcast commands
            app_lib::commands::podcast::generate_podcast,
            // System commands
            app_lib::commands::system::get_system_specs,
            app_lib::commands::system::check_compatibility,
            app_lib::commands::system::get_model_tier,
            app_lib::commands::system::estimate_model_memory,
            app_lib::commands::system::detect_quantization,
            app_lib::commands::system::detect_model_params,
            // Playground commands
            app_lib::commands::playground::playground_list_pipelines,
            app_lib::commands::playground::playground_load_pipeline,
            app_lib::commands::playground::playground_save_pipeline,
            app_lib::commands::playground::playground_delete_pipeline,
            app_lib::commands::playground::playground_run_pipeline,
            app_lib::commands::playground::playground_cancel_run,
            app_lib::commands::playground::playground_store_credential,
            // Web search commands
            app_lib::commands::web_search::web_search,
        ])
        .build(tauri::generate_context!())
        .expect("error building tauri app")
        .run(|app_handle, event| {
            if let tauri::RunEvent::Exit = event {
                log::info!("App exiting — stopping all models...");
                let pm = app_handle.state::<ProcessManagerState>();
                let pm = pm.0.clone();
                // Block on stopping all processes before exit
                tauri::async_runtime::block_on(async {
                    pm.stop_all().await;
                });
            }
        });
}
