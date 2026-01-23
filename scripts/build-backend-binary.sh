#!/bin/bash
#
# Build Python Backend as Standalone Binary
#
# This script uses PyInstaller to create a single executable
# that bundles Python + all dependencies. No Python installation
# required on the target machine.
#
# Usage: ./scripts/build-backend-binary.sh
#
# Output: dist/praval-backend (single executable ~150-200MB)
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${BLUE}==>${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

# Header
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}     Praval Deep Research - Backend Binary Builder          ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_status "Project root: $PROJECT_ROOT"

# Detect OS
OS="$(uname -s)"
ARCH="$(uname -m)"
print_status "Building for: $OS $ARCH"

# ============================================================================
# Setup Virtual Environment (if needed)
# ============================================================================
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
print_success "Virtual environment activated"

# ============================================================================
# Install Build Dependencies
# ============================================================================
print_status "Installing build dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install pyinstaller -q

# Install project dependencies (needed for PyInstaller to analyze)
print_status "Installing project dependencies (for analysis)..."
pip install -r requirements.txt -q 2>/dev/null || {
    print_warning "Some dependencies failed, continuing anyway..."
}

print_success "Build dependencies ready"

# ============================================================================
# Create PyInstaller Spec File
# ============================================================================
print_status "Creating PyInstaller spec file..."

cat > praval-backend.spec << 'EOF'
# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Praval Deep Research Backend

import sys
from pathlib import Path

block_cipher = None

# Project paths
project_root = Path(SPECPATH)
src_path = project_root / 'src'

# Collect all source files
a = Analysis(
    ['src/agentic_research/api/embedded_main.py'],
    pathex=[str(src_path)],
    binaries=[],
    datas=[
        # Include any data files needed
        ('src/agentic_research', 'agentic_research'),
    ],
    hiddenimports=[
        # FastAPI and web
        'uvicorn',
        'uvicorn.logging',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'uvicorn.protocols',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'fastapi',
        'starlette',
        'pydantic',
        'pydantic_settings',

        # Async
        'asyncio',
        'aiohttp',
        'aio_pika',
        'asyncpg',

        # ML/AI
        'openai',
        'anthropic',
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'numpy',
        'pandas',

        # Storage
        'qdrant_client',
        'minio',
        'redis',
        'sqlalchemy',
        'sqlalchemy.dialects.sqlite',
        'sqlalchemy.dialects.postgresql',

        # Utils
        'structlog',
        'httpx',
        'requests',
        'feedparser',
        'PyPDF2',
        'pdfplumber',

        # Praval framework
        'praval',
        'praval.agent',
        'praval.memory',
        'praval.transport',

        # Vajra BM25
        'vajra_bm25',

        # LangExtract
        'langextract',
        'google.generativeai',
        'ollama',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary large packages
        'matplotlib',
        'scipy',
        'sklearn',
        'torch',
        'tensorflow',
        'keras',
        'cv2',
        'PIL',
        'tkinter',
        'test',
        'tests',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='praval-backend',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
EOF

print_success "Spec file created"

# ============================================================================
# Build the Binary
# ============================================================================
print_status "Building standalone binary..."
echo "This may take 5-10 minutes on first build..."
echo ""

pyinstaller --clean --noconfirm praval-backend.spec 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"Building"* ]] || [[ "$line" == *"Copying"* ]]; then
        echo -e "${BLUE}$line${NC}"
    elif [[ "$line" == *"completed successfully"* ]]; then
        echo -e "${GREEN}$line${NC}"
    elif [[ "$line" == *"WARNING"* ]]; then
        # Suppress most warnings, they're usually harmless
        :
    elif [[ "$line" == *"ERROR"* ]] || [[ "$line" == *"error"* ]]; then
        echo -e "${RED}$line${NC}"
    fi
done

# ============================================================================
# Verify Build
# ============================================================================
BINARY_PATH="dist/praval-backend"

if [ -f "$BINARY_PATH" ]; then
    BINARY_SIZE=$(du -h "$BINARY_PATH" | cut -f1)
    print_success "Build complete!"
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}                    Binary Build Complete!                   ${GREEN}║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Binary location: $PROJECT_ROOT/dist/praval-backend"
    echo "Binary size: $BINARY_SIZE"
    echo ""
    echo "To test:"
    echo "  ./dist/praval-backend"
    echo ""
    echo "The binary will start the FastAPI server on http://localhost:8000"
    echo ""

    # Create a version info file
    echo "{\"version\": \"1.0.0\", \"os\": \"$OS\", \"arch\": \"$ARCH\", \"built\": \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\"}" > dist/praval-backend.json

else
    print_error "Build failed. Binary not found."
    echo "Check the output above for errors."
    exit 1
fi

# ============================================================================
# Cleanup
# ============================================================================
print_status "Cleaning up build artifacts..."
rm -rf build/
rm -f praval-backend.spec
print_success "Cleanup complete"

echo ""
print_status "Next steps:"
echo "  1. Copy dist/praval-backend to desktop/src-tauri/binaries/"
echo "  2. Run: cd desktop && npm run tauri build"
echo "  3. The Tauri app will bundle the binary automatically"
