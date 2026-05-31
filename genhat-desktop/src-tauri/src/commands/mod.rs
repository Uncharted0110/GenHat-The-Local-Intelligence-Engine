//! Tauri command handlers — the frontend-facing API.
//!
//! All `#[tauri::command]` functions live here, organized by domain.

pub mod models;
pub mod inference;
pub mod audio;
pub mod rag;
pub mod podcast;
pub mod workspace;
pub mod download;
pub mod system;
pub mod playground;
pub mod web_search;
