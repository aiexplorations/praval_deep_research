# Praval Deep Research - Desktop App Implementation

## Status: Implementation Complete ✅

The embedded services and Tauri scaffold have been created. This document describes the architecture and how to build/run the desktop app.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Praval Desktop App                      │
├─────────────────────────────────────────────────────┤
│  Tauri Shell (Rust)                                 │
│  ├── Window management                              │
│  ├── System tray                                    │
│  ├── Native file dialogs                            │
│  └── Python sidecar management                      │
├─────────────────────────────────────────────────────┤
│  System Webview                                      │
│  └── React Frontend (from frontend-new/)            │
├─────────────────────────────────────────────────────┤
│  Python Sidecar (FastAPI)                           │
│  └── Embedded Services:                             │
│      ├── EmbeddedStorageClient (→ filesystem)       │
│      ├── EmbeddedVectorDB (→ LanceDB/numpy)         │
│      ├── EmbeddedCacheStore (→ diskcache)           │
│      ├── LocalMessageQueue (→ asyncio.Queue)        │
│      └── SQLite (→ SQLAlchemy)                     │
├─────────────────────────────────────────────────────┤
│  Local Data Storage                                  │
│  └── ~/Library/Application Support/Praval/ (macOS) │
│      ~/.praval/ (Linux) | %APPDATA%/Praval (Win)   │
└─────────────────────────────────────────────────────┘
```

---

## Files Created

### Embedded Services (`src/agentic_research/storage/embedded/`)

| File | Replaces | Description |
|------|----------|-------------|
| `storage_client.py` | MinIO | Filesystem-based PDF/metadata storage |
| `vector_db.py` | Qdrant | NumPy/LanceDB vector database |
| `cache_store.py` | Redis | diskcache/dict-based caching |
| `message_queue.py` | RabbitMQ | In-process async message queue |
| `config.py` | Docker env | Unified configuration + service init |
| `__init__.py` | - | Package exports |

### Tauri Desktop App (`desktop/`)

| File | Description |
|------|-------------|
| `package.json` | Node dependencies for Tauri |
| `README.md` | Setup and usage instructions |
| `src-tauri/Cargo.toml` | Rust dependencies |
| `src-tauri/tauri.conf.json` | Tauri configuration |
| `src-tauri/src/main.rs` | App entry point |
| `src-tauri/src/commands.rs` | Tauri commands (called from JS) |
| `src-tauri/src/sidecar.rs` | Python backend management |

### API Entry Point

| File | Description |
|------|-------------|
| `src/agentic_research/api/embedded_main.py` | FastAPI app for embedded mode |

---

## Service Replacement Summary

| Docker Service | Embedded Alternative | Implementation |
|----------------|---------------------|----------------|
| **MinIO** | Local filesystem | `EmbeddedStorageClient` - stores PDFs in `data/storage/` |
| **Qdrant** | NumPy + LanceDB | `EmbeddedVectorDB` - cosine similarity search |
| **Redis** | diskcache | `EmbeddedCacheStore` - SQLite-backed cache |
| **RabbitMQ** | asyncio.Queue | `LocalMessageQueue` - in-process pub/sub |
| **PostgreSQL** | SQLite | SQLAlchemy with `sqlite+aiosqlite://` |
| **Neo4j** | Disabled | Knowledge graph optional in desktop mode |

---

## Quick Start

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Node.js 18+
# (use nvm, brew, or download from nodejs.org)

# Python 3.10+
python3 --version
```

### Setup

```bash
# 1. Clone and enter project
cd praval_deep_research

# 2. Install Python dependencies with desktop extras
pip install -e ".[desktop]"

# 3. Install Tauri CLI and dependencies
cd desktop
npm install

# 4. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 5. Run in development mode
npm run tauri:dev
```

### Build for Distribution

```bash
cd desktop
npm run tauri:build

# Outputs:
# - macOS: target/release/bundle/dmg/Praval Deep Research.dmg
# - Windows: target/release/bundle/msi/Praval Deep Research.msi
# - Linux: target/release/bundle/appimage/praval-deep-research.AppImage
```

---

## Running in Embedded Mode (Without Tauri)

You can also run just the Python backend in embedded mode:

```bash
# Set environment variables
export PRAVAL_EMBEDDED_MODE=true
export PRAVAL_DATA_DIR=./data
export OPENAI_API_KEY=sk-...

# Run the embedded server
python -m uvicorn agentic_research.api.embedded_main:app --host 127.0.0.1 --port 8000

# Then open frontend in browser
cd frontend-new
npm run dev
# Visit http://localhost:5173
```

---

## Data Storage Locations

### Desktop App (Tauri)

| Platform | Location |
|----------|----------|
| macOS | `~/Library/Application Support/Praval/` |
| Windows | `%APPDATA%/Praval/` |
| Linux | `~/.praval/` |

### Development Mode

| Mode | Location |
|------|----------|
| Embedded | `./data/` (project root) |
| Docker | `./data/` (bind mounts from docker-compose.yml) |

### Data Structure

```
data/
├── storage/
│   └── research-papers/
│       ├── papers/           # PDF files
│       ├── metadata/         # JSON metadata
│       └── extracted_text/   # Processed text
├── vectors/
│   ├── research_vectors/     # Paper chunk embeddings
│   └── paper_summaries/      # Summary embeddings
├── cache/                    # diskcache SQLite
├── vajra_indexes/           # BM25 search indexes
└── praval.db                # SQLite database
```

---

## API Endpoints (Embedded Mode)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with service status |
| `/config` | GET | Current configuration |
| `/stats` | GET | Storage and service statistics |
| `/papers` | GET | List all papers |
| `/papers/upload` | POST | Upload a PDF |
| `/papers/export` | POST | Export papers to directory |
| `/search` | POST | Search papers |
| `/chat` | POST | Q&A with papers |
| `/shutdown` | POST | Graceful shutdown |

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PRAVAL_EMBEDDED_MODE` | `false` | Enable embedded mode |
| `PRAVAL_DATA_DIR` | `./data` | Data storage directory |
| `OPENAI_API_KEY` | - | Required for embeddings |
| `PRAVAL_DEFAULT_MODEL` | `gpt-4o-mini` | LLM model for Q&A |
| `LOG_LEVEL` | `INFO` | Logging level |

### Config File (Desktop)

Location: `<data_dir>/config.json`

```json
{
  "openai_api_key": "sk-...",
  "embedding_model": "text-embedding-3-small",
  "llm_model": "gpt-4o-mini",
  "theme": "system",
  "auto_start": false
}
```

---

## Features Comparison

| Feature | Docker Mode | Desktop Mode |
|---------|-------------|--------------|
| Paper upload | ✅ | ✅ |
| Semantic search | ✅ | ✅ |
| BM25 keyword search | ✅ | ✅ |
| Chat/Q&A | ✅ | ✅ |
| Knowledge graph | ✅ | ❌ (optional) |
| Distributed agents | ✅ | ❌ (single process) |
| Multi-user | ✅ | ❌ (single user) |
| System tray | ❌ | ✅ |
| Native dialogs | ❌ | ✅ |
| Drag & drop upload | Browser only | ✅ Native |
| Auto-updates | Manual | ✅ Built-in |

---

## Next Steps

1. **Test the embedded services**: Run `embedded_main.py` and verify all endpoints work
2. **Build Tauri app**: Run `npm run tauri:dev` from `desktop/` directory
3. **Add app icons**: Generate icons using `npm run tauri:icon`
4. **Configure signing**: Set up code signing for distribution
5. **Set up auto-updates**: Configure update server URL in `tauri.conf.json`

---

## Migration from Docker

To migrate existing papers from Docker to desktop:

```bash
# 1. Run the migration script (created earlier)
./scripts/migrate_to_local_storage.sh

# 2. Or manually copy from Docker volume:
docker cp research_minio:/data/research-papers ./data/storage/

# 3. The embedded app will read from ./data/storage/
```

---

## Troubleshooting

### "Python not found"
Ensure Python 3.10+ is installed and in PATH.

### "Failed to start backend"
Check `~/.praval/logs/backend.log` for errors.

### "Embeddings failing"
Verify `OPENAI_API_KEY` is set correctly.

### "Port 8000 in use"
Kill existing process: `lsof -i :8000 | xargs kill`
