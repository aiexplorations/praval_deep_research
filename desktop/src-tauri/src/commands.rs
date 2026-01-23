// Tauri Commands - Functions callable from the frontend

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tauri::command;

use crate::sidecar;

/// Application version info
#[derive(Serialize)]
pub struct AppInfo {
    version: String,
    tauri_version: String,
    platform: String,
    arch: String,
}

/// Backend status info
#[derive(Serialize)]
pub struct BackendStatus {
    running: bool,
    port: u16,
    url: String,
    pid: Option<u32>,
}

/// System info
#[derive(Serialize)]
pub struct SystemInfo {
    os: String,
    arch: String,
    cpu_count: usize,
    memory_gb: f64,
    data_dir: String,
}

/// Get application version and info
#[command]
pub fn get_app_version() -> AppInfo {
    AppInfo {
        version: env!("CARGO_PKG_VERSION").to_string(),
        tauri_version: tauri::VERSION.to_string(),
        platform: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
    }
}

/// Get the data directory path
#[command]
pub fn get_data_dir() -> Result<String, String> {
    let data_dir = get_data_directory()?;
    Ok(data_dir.to_string_lossy().to_string())
}

/// Open the data folder in system file manager
#[command]
pub async fn open_data_folder() -> Result<(), String> {
    let data_dir = get_data_directory()?;

    #[cfg(target_os = "macos")]
    {
        std::process::Command::new("open")
            .arg(&data_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "windows")]
    {
        std::process::Command::new("explorer")
            .arg(&data_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    #[cfg(target_os = "linux")]
    {
        std::process::Command::new("xdg-open")
            .arg(&data_dir)
            .spawn()
            .map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Get backend status
#[command]
pub fn get_backend_status() -> BackendStatus {
    let running = sidecar::is_backend_running();
    BackendStatus {
        running,
        port: 8000,
        url: "http://localhost:8000".to_string(),
        pid: sidecar::get_backend_pid(),
    }
}

/// Start the Python backend
#[command]
pub async fn start_backend() -> Result<(), String> {
    sidecar::start_python_backend().await
}

/// Stop the Python backend
#[command]
pub async fn stop_backend() -> Result<(), String> {
    sidecar::stop_python_backend()
}

/// Import a PDF file
#[command]
pub async fn import_pdf(path: String) -> Result<String, String> {
    let file_path = PathBuf::from(&path);

    if !file_path.exists() {
        return Err(format!("File not found: {}", path));
    }

    if !path.to_lowercase().ends_with(".pdf") {
        return Err("File must be a PDF".to_string());
    }

    // Read file and send to backend
    let pdf_data = std::fs::read(&file_path)
        .map_err(|e| format!("Failed to read file: {}", e))?;

    let file_name = file_path
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();

    // Upload to backend API
    let client = reqwest::Client::new();
    let form = reqwest::multipart::Form::new()
        .part(
            "file",
            reqwest::multipart::Part::bytes(pdf_data)
                .file_name(file_name.clone())
                .mime_str("application/pdf")
                .unwrap(),
        );

    let response = client
        .post("http://localhost:8000/papers/upload")
        .multipart(form)
        .send()
        .await
        .map_err(|e| format!("Failed to upload: {}", e))?;

    if response.status().is_success() {
        Ok(format!("Uploaded: {}", file_name))
    } else {
        let error = response.text().await.unwrap_or_default();
        Err(format!("Upload failed: {}", error))
    }
}

/// Export papers to a directory
#[command]
pub async fn export_papers(export_path: String) -> Result<u32, String> {
    let client = reqwest::Client::new();

    let response = client
        .post("http://localhost:8000/papers/export")
        .json(&serde_json::json!({ "path": export_path }))
        .send()
        .await
        .map_err(|e| format!("Export request failed: {}", e))?;

    if response.status().is_success() {
        let result: serde_json::Value = response.json().await.unwrap_or_default();
        let count = result.get("count").and_then(|v| v.as_u64()).unwrap_or(0) as u32;
        Ok(count)
    } else {
        let error = response.text().await.unwrap_or_default();
        Err(format!("Export failed: {}", error))
    }
}

/// Get system information
#[command]
pub fn get_system_info() -> Result<SystemInfo, String> {
    let data_dir = get_data_directory()?;

    Ok(SystemInfo {
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_count: num_cpus(),
        memory_gb: get_memory_gb(),
        data_dir: data_dir.to_string_lossy().to_string(),
    })
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Get the data directory for the application
fn get_data_directory() -> Result<PathBuf, String> {
    let data_dir = if cfg!(target_os = "macos") {
        dirs::home_dir()
            .ok_or("Could not find home directory")?
            .join("Library")
            .join("Application Support")
            .join("Praval")
    } else if cfg!(target_os = "windows") {
        dirs::data_dir()
            .ok_or("Could not find data directory")?
            .join("Praval")
    } else {
        dirs::home_dir()
            .ok_or("Could not find home directory")?
            .join(".praval")
    };

    Ok(data_dir)
}

/// Ensure all data directories exist
pub fn ensure_data_dirs() -> Result<(), String> {
    let data_dir = get_data_directory()?;

    let dirs = vec![
        data_dir.clone(),
        data_dir.join("storage"),
        data_dir.join("storage").join("research-papers"),
        data_dir.join("storage").join("research-papers").join("papers"),
        data_dir.join("storage").join("research-papers").join("metadata"),
        data_dir.join("vectors"),
        data_dir.join("cache"),
        data_dir.join("vajra_indexes"),
        data_dir.join("logs"),
    ];

    for dir in dirs {
        std::fs::create_dir_all(&dir)
            .map_err(|e| format!("Failed to create {:?}: {}", dir, e))?;
    }

    log::info!("Data directories created at {:?}", data_dir);
    Ok(())
}

/// Get number of CPUs
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(1)
}

/// Get system memory in GB (approximate)
fn get_memory_gb() -> f64 {
    // This is a simplified version - real implementation would use sysinfo crate
    #[cfg(target_os = "macos")]
    {
        // Default to 8GB on macOS if we can't determine
        8.0
    }
    #[cfg(not(target_os = "macos"))]
    {
        8.0
    }
}
