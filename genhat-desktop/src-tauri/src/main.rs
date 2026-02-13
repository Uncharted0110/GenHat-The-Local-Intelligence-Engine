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
    // Resolve the models dir relative to the cargo manifest dir at compile time,
    // so the path is absolute and works regardless of the working directory.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")); // .../src-tauri
    let models = manifest_dir.join("../../models");
    // Canonicalize to get a clean absolute path; fall back to the joined path
    models.canonicalize().unwrap_or(models)
}


fn resolve_llama_exe() -> PathBuf {
    // Determine OS-specific folder name
    let os_folder = if cfg!(windows) {
        "llama-win"
    } else if cfg!(target_os = "macos") {
        "llama-mac"
    } else {
        "llama-lin"
    };

    // Build a list of candidate executable names depending on platform.
    let exe_names: Vec<&str> = if cfg!(windows) {
        vec!["llama-server.exe"]
    } else if cfg!(target_os = "macos") {
        // macOS builds may include architecture or platform suffixes.
        vec![
            "llama-server",
            "llama-server-macos",
            "llama-server-macos-arm64",
            "llama-server-macos-x86_64",
            "llama-server-arm64",
            "llama-server-x86_64",
        ]
    } else {
        vec!["llama-server"]
    };

    let exe_path = std::env::current_exe().unwrap();
    let mut checked = Vec::new();

    exe_path
        .ancestors()
        // Walk upward to find src-tauri/bin/{os_folder} in dev builds or bin/{os_folder} in release.
        .find_map(|dir| {
            for &exe_name in &exe_names {
                // Check for dev path
                let dev = dir.join("src-tauri/bin").join(os_folder).join(exe_name);
                checked.push(dev.clone());
                if dev.exists() {
                    return Some(dev);
                }

                // Check for release path (typically in bundled resources)
                let rel = dir.join("bin").join(os_folder).join(exe_name);
                checked.push(rel.clone());
                if rel.exists() {
                    return Some(rel);
                }

                // Check for resources directory structure
                let res = dir.join("resources/bin").join(os_folder).join(exe_name);
                checked.push(res.clone());
                if res.exists() {
                    return Some(res);
                }
            }
            None
        })
        .unwrap_or_else(|| {
            let checked_list = checked
                .into_iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join("\n");
            panic!(
                "llama-server not found. Checked the following paths:\n{checked_list}",
            );
        })
}

fn resolve_tts_exe() -> PathBuf {
    let os_folder = if cfg!(windows) {
        "tts-win"
    } else if cfg!(target_os = "macos") {
        "tts-mac"
    } else {
        "tts-lin"
    };

    let exe_name = if cfg!(windows) { "tts-inference.exe" } else { "tts-inference" };
    let exe_path = std::env::current_exe().unwrap();

    // Since we switched to --onedir, the executable is inside a folder of the same name
    // e.g. bin/tts-lin/tts-inference/tts-inference
    // logic: bin -> os_folder -> folder_name -> exe_name
    // folder_name is usually "tts-inference" provided by --name in pyinstaller

    let relative_path = PathBuf::from("tts-inference").join(exe_name);

    exe_path.ancestors().find_map(|dir| {
         // Dev path (src-tauri/bin/tts-lin/tts-inference/tts-inference)
         let dev = dir.join("src-tauri/bin").join(os_folder).join(&relative_path);
         if dev.exists() { return Some(dev); }
         
         // Release path
         let rel = dir.join("bin").join(os_folder).join(&relative_path);
         if rel.exists() { return Some(rel); }
         
         // Resources path
         let res = dir.join("resources/bin").join(os_folder).join(&relative_path);
         if res.exists() { return Some(res); }

         // Fallback for older --onefile structure (just in case)
         let old_onefile = dir.join("src-tauri/bin").join(os_folder).join(exe_name);
         if old_onefile.exists() && old_onefile.parent().unwrap().file_name().unwrap() != "tts-inference" {
            return Some(old_onefile);
         }

         None
    }).expect("TTS executable not found")
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
            "-n", // max_tokens
            "256",
            "--temp",
            "0.7",
            "--top-p",
            "0.9",
            "--top-k",
            "40",
            "--repeat-penalty",
            "1.1",
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
                    // Exclude TTS models
                    if name.starts_with("t3_") || name.starts_with("s3gen") || name.starts_with("ve_") {
                        continue;
                    }
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
fn list_audio_models() -> Vec<ModelFile> {
    let dir = get_models_dir();
    // Check main dir and specific subdir
    let search_dirs = vec![dir.clone(), dir.join("tts-chatterbox-q4-k-m")];
    let mut models = Vec::new();

    for d in search_dirs {
        if let Ok(entries) = std::fs::read_dir(d) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|s| s.to_str()) == Some("gguf") {
                     if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                         // We treat 's3gen' files as the selectable "Model" for TTS
                         if name.starts_with("s3gen") {
                             models.push(ModelFile {
                                 name: name.to_string(),
                                 path: path.to_string_lossy().to_string(),
                             });
                         }
                     }
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

#[tauri::command]
async fn generate_speech(
    model_path: String,
    input: String,
) -> Result<String, String> {
    // Resolve Exe
    let exe = resolve_tts_exe();

    // Resolve model files
    let s3_path = PathBuf::from(&model_path);
    if !s3_path.exists() {
        return Err(format!("Model path not found: {:?}", s3_path));
    }
    let parent = s3_path.parent().unwrap_or(Path::new(""));
    
    // We expect siblings: ve_fp32-f16.gguf and t3_cfg-q4_k_m.gguf
    let vae_path = parent.join("ve_fp32-f16.gguf");
    let clip_path = parent.join("t3_cfg-q4_k_m.gguf");
    
    if !vae_path.exists() {
        return Err(format!("Sibling VAE model (ve_fp32-f16.gguf) not found in {:?}", parent));
    }
    if !clip_path.exists() {
         return Err(format!("Sibling CLIP model (t3_cfg-q4_k_m.gguf) not found in {:?}", parent));
    }

    // Prepare Output Path
    let mut temp = std::env::temp_dir();
    let timestamp = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_millis();
    let filename = format!("genhat_tts_{}.wav", timestamp);
    temp.push(&filename);
    let output_str = temp.to_string_lossy().to_string();

    // Run Exe
    // work dir should be the exe's dir so it finds its internal libs (it's a one-dir bundle)
    let cwd = exe.parent().unwrap_or(&std::path::Path::new("."));

    let output = Command::new(&exe)
        .current_dir(cwd)
        .arg("--text")
        .arg(&input)
        .arg("--output")
        .arg(&output_str)
        .arg("--model_gguf")
        .arg(&s3_path)
        .arg("--vae_gguf")
        .arg(&vae_path)
        .arg("--clip_gguf")
        .arg(&clip_path)
        .output()
        .map_err(|e| format!("Failed to spawn tts executable '{}': {}", exe.display(), e))?;

    if !output.status.success() {
         let stderr = String::from_utf8_lossy(&output.stderr);
         let stdout = String::from_utf8_lossy(&output.stdout);
         return Err(format!("TTS process failed: {}\nStdout: {}", stderr, stdout));
    }

    Ok(output_str) // Return the absolute path to the wav file
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
                // Find first available that isn't a TTS model
                std::fs::read_dir(&dir).ok().and_then(|mut entries| {
                    entries.find_map(|e| {
                         e.ok().map(|ent| ent.path())
                          .filter(|p| {
                            let is_gguf = p.extension().map(|s| s == "gguf").unwrap_or(false);
                            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
                            is_gguf && !name.starts_with("t3_") && !name.starts_with("s3gen") && !name.starts_with("ve_")
                          })
                    })
                })
            };

            if let Some(p) = model_to_load {
                let child = spawn_llama_process(p);
                app.state::<AppState>().llama.lock().unwrap().replace(child);
            } else {
                println!("No valid LLM models found in {}, server not started automatically.", dir.display());
            }

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![stop_llama, list_models, list_audio_models, switch_model, generate_speech])
        .build(tauri::generate_context!())
        .expect("error building tauri app")
        .run(|app_handle, event| match event {
            tauri::RunEvent::Exit => {
                let state = app_handle.state::<AppState>();
                if let Some(mut child) = state.llama.lock().unwrap().take() {
                    let _ = child.kill();
                };
            }
            _ => {}
        });
}

