# Praval Deep Research - Desktop App

This directory contains the Tauri-based desktop application for Praval Deep Research.

## Overview

The desktop app bundles the React frontend with embedded Python backend services,
eliminating the need for Docker while providing a native desktop experience.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Praval Desktop App                      │
├─────────────────────────────────────────────────────┤
│  Tauri Shell (Rust)                                 │
│  ├── Window management                              │
│  ├── File system access                             │
│  ├── System tray                                    │
│  └── Native dialogs                                 │
├─────────────────────────────────────────────────────┤
│  System Webview                                      │
│  └── React Frontend (Vite build)                    │
├─────────────────────────────────────────────────────┤
│  Python Sidecar                                      │
│  └── FastAPI Server (embedded mode)                 │
│      ├── EmbeddedStorageClient (replaces MinIO)     │
│      ├── EmbeddedVectorDB (replaces Qdrant)         │
│      ├── EmbeddedCacheStore (replaces Redis)        │
│      ├── LocalMessageQueue (replaces RabbitMQ)      │
│      └── SQLite (replaces PostgreSQL)              │
├─────────────────────────────────────────────────────┤
│  Local Data (~/.praval/ or ./data/)                 │
│  ├── storage/research-papers/ (PDFs)               │
│  ├── vectors/ (embeddings)                          │
│  ├── cache/ (session data)                          │
│  ├── vajra_indexes/ (BM25 search)                  │
│  └── praval.db (SQLite)                            │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Node.js 18+
- Rust (install via rustup)
- Python 3.10+
- OpenAI API key

### Setup

```bash
# 1. Install Tauri CLI
cd desktop
npm install

# 2. Install Rust dependencies
cd src-tauri
cargo build

# 3. Set up Python backend
cd ../../
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[desktop]"

# 4. Set environment variables
export OPENAI_API_KEY="your-key-here"

# 5. Run in development mode
cd desktop
npm run tauri dev
```

### Build for Distribution

```bash
# Build production app
npm run tauri build

# Output locations:
# macOS:   target/release/bundle/dmg/Praval Deep Research.dmg
# Windows: target/release/bundle/msi/Praval Deep Research.msi
# Linux:   target/release/bundle/appimage/praval-deep-research.AppImage
```

## Project Structure

```
desktop/
├── src-tauri/
│   ├── Cargo.toml           # Rust dependencies
│   ├── tauri.conf.json      # Tauri configuration
│   ├── src/
│   │   ├── main.rs          # Tauri entry point
│   │   ├── commands.rs      # Rust commands (called from JS)
│   │   ├── config.rs        # Configuration management
│   │   └── sidecar.rs       # Python sidecar management
│   └── icons/               # App icons
├── package.json             # Node dependencies
├── vite.config.ts           # Vite config for Tauri
└── README.md                # This file
```

## Features

### Native Features (via Tauri)
- **Drag & Drop**: Drop PDFs directly onto the app to upload
- **File Dialogs**: Native open/save dialogs
- **System Tray**: Background operation with status indicator
- **Notifications**: Research completion alerts
- **Auto-Launch**: Start on system boot (optional)
- **Auto-Update**: Built-in update mechanism

### Embedded Services
- **No Docker Required**: All services run in-process
- **Offline Capable**: Works without internet (except for embeddings)
- **Portable Data**: Data stored in user-accessible folder
- **Fast Startup**: No container orchestration overhead

## Configuration

The app stores configuration in platform-specific locations:

| Platform | Config Location |
|----------|-----------------|
| macOS    | `~/Library/Application Support/Praval/config.json` |
| Windows  | `%APPDATA%/Praval/config.json` |
| Linux    | `~/.config/praval/config.json` |

### Config Options

```json
{
  "openai_api_key": "sk-...",
  "anthropic_api_key": "sk-ant-...",
  "gemini_api_key": "AI...",
  "llm_provider": "openai",
  "llm_model": "gpt-4o-mini",
  "embedding_model": "text-embedding-3-small",
  "ollama_base_url": "http://localhost:11434",
  "ollama_model": "llama3.2",
  "langextract_provider": "gemini",
  "langextract_model": "gemini-2.5-flash",
  "theme": "system",
  "auto_start": false,
  "check_updates": true
}
```

### Supported LLM Providers

| Provider | Models | Requires API Key |
|----------|--------|------------------|
| OpenAI | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo | Yes |
| Anthropic | claude-sonnet-4, claude-3-5-sonnet, claude-3-haiku | Yes |
| Ollama | llama3.2, mistral, mixtral, codellama, phi3, gemma2 | No (local) |

### Using Ollama (Free Local Models)

For fully offline operation:

```bash
# Install Ollama from https://ollama.ai
brew install ollama  # macOS
# or download from website for Windows/Linux

# Start the Ollama service
ollama serve

# Pull a model
ollama pull llama3.2

# In Praval Settings, select "Ollama" as provider
# The app will auto-detect your installed models
```

## Development

### Running Tests

```bash
# Rust tests
cd src-tauri && cargo test

# Frontend tests
npm test

# Python backend tests
pytest ../tests/
```

### Debugging

```bash
# Enable Tauri devtools
npm run tauri dev -- --devtools

# View Python sidecar logs
tail -f ~/.praval/logs/backend.log
```

## Troubleshooting

### "API key not set"
Go to **Settings** in the app to configure your API key, or use environment variable:
```bash
export OPENAI_API_KEY="sk-..."
```

Alternatively, select **Ollama** as your provider to use free local models without any API key.

### "Python sidecar failed to start"
Ensure Python 3.10+ is installed and the virtual environment is activated.

### "Port already in use"
The backend runs on port 8000 by default. Kill any existing process:
```bash
lsof -i :8000 | grep LISTEN | awk '{print $2}' | xargs kill
```

## License

Same as main Praval Deep Research project.
