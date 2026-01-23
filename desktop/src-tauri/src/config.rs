// Configuration management for Praval Deep Research Desktop
// Handles API keys, provider selection, and app settings

use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use tauri::command;

/// Application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    /// OpenAI API key
    #[serde(default)]
    pub openai_api_key: Option<String>,

    /// Anthropic API key
    #[serde(default)]
    pub anthropic_api_key: Option<String>,

    /// Google Gemini API key
    #[serde(default)]
    pub gemini_api_key: Option<String>,

    /// Embedding model (OpenAI)
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,

    /// LLM model name
    #[serde(default = "default_llm_model")]
    pub llm_model: String,

    /// LLM provider (openai, anthropic, ollama)
    #[serde(default = "default_llm_provider")]
    pub llm_provider: String,

    /// Ollama base URL (for local models)
    #[serde(default = "default_ollama_url")]
    pub ollama_base_url: String,

    /// Ollama model name
    #[serde(default = "default_ollama_model")]
    pub ollama_model: String,

    /// LangExtract provider for PDF extraction (gemini, openai, ollama)
    #[serde(default = "default_langextract_provider")]
    pub langextract_provider: String,

    /// LangExtract model
    #[serde(default = "default_langextract_model")]
    pub langextract_model: String,

    /// UI theme (light, dark, system)
    #[serde(default = "default_theme")]
    pub theme: String,

    /// Auto-start backend on launch
    #[serde(default = "default_auto_start")]
    pub auto_start: bool,

    /// Check for updates on startup
    #[serde(default = "default_check_updates")]
    pub check_updates: bool,

    /// Custom data directory (optional)
    #[serde(default)]
    pub custom_data_dir: Option<String>,
}

// Default value functions
fn default_embedding_model() -> String {
    "text-embedding-3-small".to_string()
}

fn default_llm_model() -> String {
    "gpt-4o-mini".to_string()
}

fn default_llm_provider() -> String {
    "openai".to_string()
}

fn default_ollama_url() -> String {
    "http://localhost:11434".to_string()
}

fn default_ollama_model() -> String {
    "llama3.2".to_string()
}

fn default_langextract_provider() -> String {
    "gemini".to_string()
}

fn default_langextract_model() -> String {
    "gemini-2.5-flash".to_string()
}

fn default_theme() -> String {
    "system".to_string()
}

fn default_auto_start() -> bool {
    true
}

fn default_check_updates() -> bool {
    true
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            openai_api_key: None,
            anthropic_api_key: None,
            gemini_api_key: None,
            embedding_model: default_embedding_model(),
            llm_model: default_llm_model(),
            llm_provider: default_llm_provider(),
            ollama_base_url: default_ollama_url(),
            ollama_model: default_ollama_model(),
            langextract_provider: default_langextract_provider(),
            langextract_model: default_langextract_model(),
            theme: default_theme(),
            auto_start: default_auto_start(),
            check_updates: default_check_updates(),
            custom_data_dir: None,
        }
    }
}

impl AppConfig {
    /// Get the config file path
    pub fn config_path() -> Result<PathBuf, String> {
        let config_dir = if cfg!(target_os = "macos") {
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
                .join(".config")
                .join("praval")
        };

        // Ensure directory exists
        fs::create_dir_all(&config_dir)
            .map_err(|e| format!("Failed to create config directory: {}", e))?;

        Ok(config_dir.join("config.json"))
    }

    /// Load configuration from file
    pub fn load() -> Result<Self, String> {
        let config_path = Self::config_path()?;

        if config_path.exists() {
            let content = fs::read_to_string(&config_path)
                .map_err(|e| format!("Failed to read config: {}", e))?;

            serde_json::from_str(&content)
                .map_err(|e| format!("Failed to parse config: {}", e))
        } else {
            // Return default config if file doesn't exist
            Ok(Self::default())
        }
    }

    /// Save configuration to file
    pub fn save(&self) -> Result<(), String> {
        let config_path = Self::config_path()?;

        let content = serde_json::to_string_pretty(self)
            .map_err(|e| format!("Failed to serialize config: {}", e))?;

        fs::write(&config_path, content)
            .map_err(|e| format!("Failed to write config: {}", e))?;

        log::info!("Configuration saved to {:?}", config_path);
        Ok(())
    }

    /// Check if any API key is configured for the selected provider
    pub fn has_required_api_key(&self) -> bool {
        match self.llm_provider.as_str() {
            "openai" => self.openai_api_key.is_some(),
            "anthropic" => self.anthropic_api_key.is_some(),
            "ollama" => true, // Ollama doesn't require API key
            _ => false,
        }
    }

    /// Convert config to environment variables for the Python backend
    pub fn to_env_vars(&self) -> Vec<(String, String)> {
        let mut env_vars = vec![
            ("PRAVAL_EMBEDDED_MODE".to_string(), "true".to_string()),
            ("PRAVAL_DEFAULT_PROVIDER".to_string(), self.llm_provider.clone()),
            ("PRAVAL_DEFAULT_MODEL".to_string(), self.llm_model.clone()),
            ("OPENAI_EMBEDDING_MODEL".to_string(), self.embedding_model.clone()),
            ("OLLAMA_BASE_URL".to_string(), self.ollama_base_url.clone()),
            ("OLLAMA_MODEL".to_string(), self.ollama_model.clone()),
            ("LANGEXTRACT_PROVIDER".to_string(), self.langextract_provider.clone()),
            ("LANGEXTRACT_MODEL".to_string(), self.langextract_model.clone()),
        ];

        // Add API keys if present
        if let Some(ref key) = self.openai_api_key {
            env_vars.push(("OPENAI_API_KEY".to_string(), key.clone()));
        }

        if let Some(ref key) = self.anthropic_api_key {
            env_vars.push(("ANTHROPIC_API_KEY".to_string(), key.clone()));
        }

        if let Some(ref key) = self.gemini_api_key {
            env_vars.push(("GEMINI_API_KEY".to_string(), key.clone()));
        }

        // Data directory
        if let Some(ref data_dir) = self.custom_data_dir {
            env_vars.push(("PRAVAL_DATA_DIR".to_string(), data_dir.clone()));
        } else if let Ok(path) = Self::config_path() {
            if let Some(parent) = path.parent() {
                env_vars.push(("PRAVAL_DATA_DIR".to_string(), parent.to_string_lossy().to_string()));
            }
        }

        env_vars
    }
}

// ============================================================================
// Tauri Commands
// ============================================================================

/// Get current configuration
#[command]
pub fn get_config() -> Result<AppConfig, String> {
    AppConfig::load()
}

/// Save configuration
#[command]
pub fn save_config(config: AppConfig) -> Result<(), String> {
    config.save()
}

/// Set just the OpenAI API key
#[command]
pub fn set_openai_api_key(key: String) -> Result<(), String> {
    let mut config = AppConfig::load()?;
    config.openai_api_key = if key.is_empty() { None } else { Some(key) };
    config.save()
}

/// Set just the Anthropic API key
#[command]
pub fn set_anthropic_api_key(key: String) -> Result<(), String> {
    let mut config = AppConfig::load()?;
    config.anthropic_api_key = if key.is_empty() { None } else { Some(key) };
    config.save()
}

/// Set just the Gemini API key
#[command]
pub fn set_gemini_api_key(key: String) -> Result<(), String> {
    let mut config = AppConfig::load()?;
    config.gemini_api_key = if key.is_empty() { None } else { Some(key) };
    config.save()
}

/// Check if required API key exists for current provider
#[command]
pub fn has_required_api_key() -> Result<bool, String> {
    let config = AppConfig::load()?;
    Ok(config.has_required_api_key())
}

/// Get config file path
#[command]
pub fn get_config_path() -> Result<String, String> {
    let path = AppConfig::config_path()?;
    Ok(path.to_string_lossy().to_string())
}

/// Set LLM provider
#[command]
pub fn set_llm_provider(provider: String) -> Result<(), String> {
    let valid_providers = ["openai", "anthropic", "ollama"];
    if !valid_providers.contains(&provider.as_str()) {
        return Err(format!("Invalid provider: {}. Must be one of: {:?}", provider, valid_providers));
    }

    let mut config = AppConfig::load()?;
    config.llm_provider = provider;
    config.save()
}
