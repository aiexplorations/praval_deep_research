// Praval Deep Research - Desktop Application
// Main entry point for Tauri app

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod commands;
mod config;
mod sidecar;

use tauri::{
    menu::{Menu, MenuItem},
    tray::{TrayIcon, TrayIconBuilder},
    Manager, RunEvent, WindowEvent,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

// Track if backend is running
static BACKEND_RUNNING: AtomicBool = AtomicBool::new(false);

fn main() {
    // Initialize logging
    env_logger::init();
    log::info!("Starting Praval Deep Research Desktop");

    tauri::Builder::default()
        // Plugins
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_fs::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_updater::Builder::new().build())

        // Commands
        .invoke_handler(tauri::generate_handler![
            commands::get_app_version,
            commands::get_data_dir,
            commands::open_data_folder,
            commands::get_backend_status,
            commands::start_backend,
            commands::stop_backend,
            commands::import_pdf,
            commands::export_papers,
            commands::get_system_info,
            // Config commands
            config::get_config,
            config::save_config,
            config::set_openai_api_key,
            config::set_anthropic_api_key,
            config::set_gemini_api_key,
            config::has_required_api_key,
            config::get_config_path,
            config::set_llm_provider,
        ])

        // Setup
        .setup(|app| {
            log::info!("Setting up application...");

            // Create data directories
            if let Err(e) = commands::ensure_data_dirs() {
                log::error!("Failed to create data directories: {}", e);
            }

            // Load and check configuration
            match config::AppConfig::load() {
                Ok(app_config) => {
                    log::info!("Configuration loaded, provider: {}", app_config.llm_provider);

                    // Check if API key is needed but missing
                    if !app_config.has_required_api_key() {
                        log::warn!("No API key configured for provider: {}", app_config.llm_provider);

                        // Notify frontend to show settings
                        if let Some(window) = app.get_webview_window("main") {
                            let _ = window.emit("show-settings", "api_key_required");
                        }
                    }
                }
                Err(e) => {
                    log::warn!("Failed to load config, using defaults: {}", e);
                }
            }

            // Build tray menu
            let quit = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let show = MenuItem::with_id(app, "show", "Show Window", true, None::<&str>)?;
            let status = MenuItem::with_id(app, "status", "Backend: Starting...", false, None::<&str>)?;

            let menu = Menu::with_items(app, &[&show, &status, &quit])?;

            // Create tray icon
            let _tray = TrayIconBuilder::new()
                .icon(app.default_window_icon().unwrap().clone())
                .menu(&menu)
                .tooltip("Praval Deep Research")
                .on_menu_event(|app, event| {
                    match event.id.as_ref() {
                        "quit" => {
                            log::info!("Quit requested from tray");
                            // Stop backend before quitting
                            let _ = sidecar::stop_python_backend();
                            app.exit(0);
                        }
                        "show" => {
                            if let Some(window) = app.get_webview_window("main") {
                                let _ = window.show();
                                let _ = window.set_focus();
                            }
                        }
                        _ => {}
                    }
                })
                .build(app)?;

            // Start Python backend
            log::info!("Starting Python backend...");
            let app_handle = app.handle().clone();

            tauri::async_runtime::spawn(async move {
                match sidecar::start_python_backend().await {
                    Ok(_) => {
                        BACKEND_RUNNING.store(true, Ordering::SeqCst);
                        log::info!("Python backend started successfully");

                        // Notify frontend
                        if let Some(window) = app_handle.get_webview_window("main") {
                            let _ = window.emit("backend-ready", true);
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to start Python backend: {}", e);

                        // Notify frontend of error
                        if let Some(window) = app_handle.get_webview_window("main") {
                            let _ = window.emit("backend-error", e.to_string());
                        }
                    }
                }
            });

            Ok(())
        })

        // Window event handling
        .on_window_event(|window, event| {
            match event {
                WindowEvent::CloseRequested { api, .. } => {
                    // Hide window instead of closing (keep running in tray)
                    #[cfg(not(target_os = "macos"))]
                    {
                        window.hide().unwrap();
                        api.prevent_close();
                    }
                }
                _ => {}
            }
        })

        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|app_handle, event| {
            match event {
                RunEvent::ExitRequested { api, .. } => {
                    // Prevent exit, we want to keep running in tray
                    api.prevent_exit();
                }
                RunEvent::Exit => {
                    // Cleanup when actually exiting
                    log::info!("Application exiting, stopping backend...");
                    let _ = sidecar::stop_python_backend();
                }
                _ => {}
            }
        });
}
