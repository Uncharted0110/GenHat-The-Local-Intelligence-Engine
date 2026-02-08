#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use tauri::{Manager, State};

#[derive(serde::Serialize)]
struct ModelFile {
    name: String,
    path: String,
}

struct AppState {
    llama: Mutex<Option<Child>>,
}

// ---- Helpers ----

fn get_models_dir() -> PathBuf {
    if let Ok(val) = std::env::var("GENHAT_MODEL_PATH") {
        let p = PathBuf::from(val);
        // If it's a file, return its parent
        if p.is_file() {
            if let Some(parent) = p.parent() {
                return parent.to_path_buf();
            }
        } else if p.is_dir() {
            // If it's a dir, return it
            return p;
        }
    }
    // Hardcoded path as requested
    PathBuf::from(r"D:\GenHat---The-Local-Intelligence-Engine\models")
}


fn resolve_llama_exe() -> PathBuf {
    let exe_path = std::env::current_exe().unwrap();
    let mut checked = Vec::new();
    exe_path
        .ancestors()
        // Walk upward to find src-tauri/bin/llama in dev builds or bin/llama in release.
        .find_map(|dir| {
            // Check for dev path
            let dev = dir.join("src-tauri/bin/llama/llama-server.exe");
            checked.push(dev.clone());
            if dev.exists() { return Some(dev); }
            
            // Check for release path (typically in bundled resources)
            let rel = dir.join("bin/llama/llama-server.exe");
            checked.push(rel.clone());
            if rel.exists() { return Some(rel); }

            // Check for resources directory structure
            let res = dir.join("resources/bin/llama/llama-server.exe");
            checked.push(res.clone());
            if res.exists() { return Some(res); }

            None
        })
        .unwrap_or_else(|| {
            let checked_list = checked
                .into_iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join("\n");
            panic!(
                "llama-server.exe not found. Checked the following paths:\n{checked_list}"
            );
        })
}

fn spawn_llama_process(model_path: PathBuf) -> Child {
    let exe = resolve_llama_exe();
    
    // Logging setup
    let log_path = std::env::temp_dir().join("genhat-llama-server.log");
    let mut log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .expect("Failed to open llama log file");
    
    let _ = writeln!(log_file, "--- llama-server start ---");
    let _ = writeln!(log_file, "exe: {}", exe.display());
    let _ = writeln!(log_file, "model: {}", model_path.display());

    // IMPORTANT: Set current_dir to the binary's folder so it finds sibling DLLs (llama.dll, etc.)
    let work_dir = exe.parent().expect("Exe has no parent");

    let mut child = Command::new(&exe)
        .args([
            "-m",
            model_path.to_str().unwrap(),
            "--ctx-size",
            "4096",
            "--port",
            "8081",
            "--host",
            "127.0.0.1",
        ])
        .current_dir(work_dir)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Failed to start llama-server");

    let _ = writeln!(log_file, "spawned pid: {}", child.id());

    // Redirect stdout to log file
    if let Some(stdout) = child.stdout.take() {
        let log_path_clone = log_path.clone();
        std::thread::spawn(move || {
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&log_path_clone) {
                 let reader = BufReader::new(stdout);
                 for line in reader.lines().flatten() {
                     let _ = writeln!(file, "[stdout] {line}");
                 }
            }
        });
    }

    // Redirect stderr to log file
    if let Some(stderr) = child.stderr.take() {
        let log_path_clone = log_path.clone();
        std::thread::spawn(move || {
            if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&log_path_clone) {
                 let reader = BufReader::new(stderr);
                 for line in reader.lines().flatten() {
                     let _ = writeln!(file, "[stderr] {line}");
                 }
            }
        });
    }

    child
}

// ---- Commands ----

#[tauri::command]
fn list_models() -> Vec<ModelFile> {
    let dir = get_models_dir();
    let mut models = Vec::new();

    if let Ok(entries) = std::fs::read_dir(&dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            // Check for .gguf extension
            if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    models.push(ModelFile {
                        name: name.to_string(),
                        path: path.to_string_lossy().to_string(),
                    });
                }
            }
        }
    }
    models
}

#[tauri::command]
fn switch_model(state: State<AppState>, model_path: String) -> Result<String, String> {
    let path = PathBuf::from(&model_path);
    if !path.exists() {
        return Err(format!("Model file not found: {}", model_path));
    }

    {
        let mut guard = state.llama.lock().unwrap();
        // Kill existing
        if let Some(mut child) = guard.take() {
            let _ = child.kill();
        }
        // Spawn new
        let child = spawn_llama_process(path);
        // Store
        *guard = Some(child);
    }
    
    Ok("server started".into())
}

#[tauri::command]
fn stop_llama(state: State<AppState>) {
    if let Some(mut child) = state.llama.lock().unwrap().take() {
        let _ = child.kill();
    }
}


fn main() {
    tauri::Builder::default()
        .manage(AppState {
            llama: Mutex::new(None),
        })
        .setup(|app| {
            // Auto-start default model if found
            let dir = get_models_dir();
            let default_path = dir.join("LFM-1.2B-INT8.gguf");
            
            let model_to_load = if default_path.exists() {
                Some(default_path)
            } else {
                // Find first available
                std::fs::read_dir(&dir).ok().and_then(|mut entries| {
                    entries.find_map(|e| {
                         e.ok().map(|ent| ent.path())
                          .filter(|p| p.extension().map(|s| s == "gguf").unwrap_or(false))
                    })
                })
            };

            if let Some(p) = model_to_load {
                let child = spawn_llama_process(p);
                app.state::<AppState>().llama.lock().unwrap().replace(child);
            } else {
                println!("No models found in {}, server not started automatically.", dir.display());
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![stop_llama, list_models, switch_model])
        .run(tauri::generate_context!())
        .expect("error running tauri");
}

