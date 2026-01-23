// Python Backend Sidecar Management
// Handles starting, stopping, and monitoring the Python FastAPI backend
// Supports both bundled binary (production) and Python module (development)

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::Duration;
use tokio::time::sleep;

use crate::config::AppConfig;

// Global backend process handle
static BACKEND_PROCESS: OnceLock<Mutex<Option<Child>>> = OnceLock::new();

/// Get the path to the bundled backend binary
fn get_bundled_binary_path() -> Option<PathBuf> {
    // Get the directory where the app executable is located
    let exe_path = std::env::current_exe().ok()?;
    let exe_dir = exe_path.parent()?;

    // Platform-specific binary name
    let binary_name = if cfg!(target_os = "windows") {
        "praval-backend.exe"
    } else {
        "praval-backend"
    };

    // Check multiple possible locations for the binary
    let possible_paths = vec![
        // Same directory as executable (production - Tauri bundle)
        exe_dir.join(binary_name),
        // Resources directory on macOS
        exe_dir.join("../Resources").join(binary_name),
        // Sidecars directory (Tauri sidecar convention)
        exe_dir.join("sidecars").join(binary_name),
        // Development: project dist directory
        exe_dir.join("../../../dist").join(binary_name),
        // Development: look in project root dist
        PathBuf::from("dist").join(binary_name),
    ];

    for path in possible_paths {
        if path.exists() {
            log::info!("Found bundled backend binary at: {:?}", path);
            return Some(path.canonicalize().unwrap_or(path));
        }
    }

    log::debug!("No bundled binary found, will fall back to Python");
    None
}

/// Check if we should use the bundled binary or Python
fn should_use_bundled_binary() -> bool {
    // Environment variable override for development
    if std::env::var("PRAVAL_USE_PYTHON").is_ok() {
        log::info!("PRAVAL_USE_PYTHON set, using Python module");
        return false;
    }

    // Check if bundled binary exists
    get_bundled_binary_path().is_some()
}

/// Start the Python backend server
pub async fn start_python_backend() -> Result<(), String> {
    log::info!("Starting Python backend...");

    // Load configuration
    let app_config = AppConfig::load().unwrap_or_default();
    log::info!("Using LLM provider: {}", app_config.llm_provider);

    // Get data directory for environment
    let data_dir = get_data_directory()?;

    // Build command based on whether we have a bundled binary
    let mut cmd = if should_use_bundled_binary() {
        let binary_path = get_bundled_binary_path()
            .ok_or("Bundled binary not found but should_use_bundled_binary returned true")?;

        log::info!("Using bundled binary: {:?}", binary_path);
        let mut c = Command::new(&binary_path);
        c.args([
            "--host", "127.0.0.1",
            "--port", "8000",
        ]);
        c
    } else {
        // Fall back to Python module (development mode)
        log::info!("Using Python module (development mode)");

        let python_cmd = if cfg!(target_os = "windows") {
            "python"
        } else {
            "python3"
        };

        let mut c = Command::new(python_cmd);
        c.args([
            "-m",
            "uvicorn",
            "agentic_research.api.main:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8000",
            "--log-level",
            "info",
        ]);
        c
    };

    // Set environment variables from config
    for (key, value) in app_config.to_env_vars() {
        cmd.env(&key, &value);
        // Don't log API keys
        if !key.contains("KEY") {
            log::debug!("Setting env: {}={}", key, value);
        }
    }

    // Additional environment settings
    cmd.env("PRAVAL_DATA_DIR", data_dir.to_string_lossy().to_string());
    cmd.env("LOG_LEVEL", "INFO");

    // Redirect output to log files
    let log_dir = data_dir.join("logs");
    std::fs::create_dir_all(&log_dir).map_err(|e| e.to_string())?;

    let stdout_log = std::fs::File::create(log_dir.join("backend.log"))
        .map_err(|e| format!("Failed to create stdout log: {}", e))?;
    let stderr_log = std::fs::File::create(log_dir.join("backend.error.log"))
        .map_err(|e| format!("Failed to create stderr log: {}", e))?;

    cmd.stdout(Stdio::from(stdout_log));
    cmd.stderr(Stdio::from(stderr_log));

    // Start the process
    let child = cmd.spawn().map_err(|e| {
        if should_use_bundled_binary() {
            format!(
                "Failed to start bundled backend binary. Error: {}",
                e
            )
        } else {
            format!(
                "Failed to start Python backend. Is Python installed? Error: {}",
                e
            )
        }
    })?;

    let pid = child.id();
    log::info!("Backend process started with PID: {}", pid);

    // Store the process handle
    let mutex = BACKEND_PROCESS.get_or_init(|| Mutex::new(None));
    *mutex.lock().unwrap() = Some(child);

    // Wait for backend to be ready
    wait_for_backend_ready().await?;

    log::info!("Backend is ready");
    Ok(())
}

/// Wait for backend to respond to health check
async fn wait_for_backend_ready() -> Result<(), String> {
    let client = reqwest::Client::new();
    let health_url = "http://localhost:8000/health";

    for i in 0..30 {
        // 30 second timeout
        sleep(Duration::from_secs(1)).await;

        match client.get(health_url).send().await {
            Ok(response) if response.status().is_success() => {
                log::info!("Backend health check passed after {} seconds", i + 1);
                return Ok(());
            }
            Ok(response) => {
                log::debug!("Health check returned status: {}", response.status());
            }
            Err(e) => {
                log::debug!("Health check failed (attempt {}): {}", i + 1, e);
            }
        }

        // Check if process is still running
        if !is_backend_running() {
            return Err("Backend process exited unexpectedly. Check logs for details.".to_string());
        }
    }

    Err("Backend failed to start within 30 seconds".to_string())
}

/// Stop the Python backend server
pub fn stop_python_backend() -> Result<(), String> {
    log::info!("Stopping Python backend...");

    if let Some(mutex) = BACKEND_PROCESS.get() {
        if let Ok(mut guard) = mutex.lock() {
            if let Some(mut child) = guard.take() {
                // Try graceful shutdown first via API
                let _ = reqwest::blocking::Client::new()
                    .post("http://localhost:8000/shutdown")
                    .send();

                // Wait a bit for graceful shutdown
                std::thread::sleep(Duration::from_secs(2));

                // Force kill if still running
                match child.try_wait() {
                    Ok(Some(_)) => {
                        log::info!("Backend stopped gracefully");
                    }
                    _ => {
                        log::warn!("Backend did not stop gracefully, forcing kill");
                        let _ = child.kill();
                    }
                }
            }
        }
    }

    Ok(())
}

/// Check if backend is running
pub fn is_backend_running() -> bool {
    if let Some(mutex) = BACKEND_PROCESS.get() {
        if let Ok(mut guard) = mutex.lock() {
            if let Some(ref mut child) = *guard {
                match child.try_wait() {
                    Ok(Some(_)) => {
                        // Process has exited
                        return false;
                    }
                    Ok(None) => {
                        // Process is still running
                        return true;
                    }
                    Err(_) => {
                        return false;
                    }
                }
            }
        }
    }
    false
}

/// Get backend process ID
pub fn get_backend_pid() -> Option<u32> {
    if let Some(mutex) = BACKEND_PROCESS.get() {
        if let Ok(guard) = mutex.lock() {
            if let Some(ref child) = *guard {
                return Some(child.id());
            }
        }
    }
    None
}

/// Get the data directory
fn get_data_directory() -> Result<std::path::PathBuf, String> {
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
