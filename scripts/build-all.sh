#!/bin/bash
#
# Build Praval Deep Research - Complete Build Script
#
# This script builds both the Python backend binary and the Tauri desktop app.
# The Python backend is bundled into the Tauri app, so end users don't need Python.
#
# Usage: ./scripts/build-all.sh [options]
#   --skip-backend   Skip building Python backend binary
#   --skip-frontend  Skip building frontend
#   --dev           Build in development mode
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
echo -e "${BLUE}║${NC}        Praval Deep Research - Full Build Script            ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
SKIP_BACKEND=false
SKIP_FRONTEND=false
DEV_MODE=false

for arg in "$@"; do
    case $arg in
        --skip-backend)
            SKIP_BACKEND=true
            ;;
        --skip-frontend)
            SKIP_FRONTEND=true
            ;;
        --dev)
            DEV_MODE=true
            ;;
    esac
done

# Get project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_status "Project root: $PROJECT_ROOT"

# Detect OS and Architecture
OS="$(uname -s)"
ARCH="$(uname -m)"
print_status "Building for: $OS $ARCH"

# Set binary name based on platform
case "$OS" in
    Darwin)
        BINARY_NAME="praval-backend-x86_64-apple-darwin"
        if [ "$ARCH" = "arm64" ]; then
            BINARY_NAME="praval-backend-aarch64-apple-darwin"
        fi
        ;;
    Linux)
        BINARY_NAME="praval-backend-x86_64-unknown-linux-gnu"
        if [ "$ARCH" = "aarch64" ]; then
            BINARY_NAME="praval-backend-aarch64-unknown-linux-gnu"
        fi
        ;;
    MINGW*|CYGWIN*|MSYS*)
        BINARY_NAME="praval-backend-x86_64-pc-windows-msvc.exe"
        ;;
    *)
        print_error "Unsupported OS: $OS"
        exit 1
        ;;
esac

# ============================================================================
# Build Python Backend Binary
# ============================================================================
if [ "$SKIP_BACKEND" = false ]; then
    print_status "Building Python backend binary..."

    # Setup virtual environment if needed
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi

    source venv/bin/activate

    # Install build dependencies
    print_status "Installing build dependencies..."
    pip install --upgrade pip setuptools wheel -q
    pip install pyinstaller -q

    # Install project dependencies
    print_status "Installing project dependencies..."
    pip install -r requirements.txt -q 2>/dev/null || {
        print_warning "Some dependencies failed, continuing..."
    }

    # Clean previous builds
    rm -rf build/ dist/praval-backend dist/praval-backend.exe

    # Build with PyInstaller
    print_status "Running PyInstaller..."

    pyinstaller \
        --name praval-backend \
        --onefile \
        --console \
        --clean \
        --noconfirm \
        --add-data "src/agentic_research:agentic_research" \
        --hidden-import uvicorn \
        --hidden-import uvicorn.logging \
        --hidden-import uvicorn.loops \
        --hidden-import uvicorn.loops.auto \
        --hidden-import uvicorn.protocols \
        --hidden-import uvicorn.protocols.http \
        --hidden-import uvicorn.protocols.http.auto \
        --hidden-import uvicorn.protocols.websockets \
        --hidden-import uvicorn.protocols.websockets.auto \
        --hidden-import uvicorn.lifespan \
        --hidden-import uvicorn.lifespan.on \
        --hidden-import fastapi \
        --hidden-import starlette \
        --hidden-import pydantic \
        --hidden-import pydantic_settings \
        --hidden-import asyncio \
        --hidden-import aiohttp \
        --hidden-import aio_pika \
        --hidden-import asyncpg \
        --hidden-import openai \
        --hidden-import anthropic \
        --hidden-import tiktoken \
        --hidden-import tiktoken_ext \
        --hidden-import tiktoken_ext.openai_public \
        --hidden-import numpy \
        --hidden-import pandas \
        --hidden-import qdrant_client \
        --hidden-import minio \
        --hidden-import redis \
        --hidden-import sqlalchemy \
        --hidden-import sqlalchemy.dialects.sqlite \
        --hidden-import structlog \
        --hidden-import httpx \
        --hidden-import requests \
        --hidden-import feedparser \
        --hidden-import PyPDF2 \
        --hidden-import pdfplumber \
        --hidden-import praval \
        --hidden-import praval.agent \
        --hidden-import praval.memory \
        --hidden-import praval.transport \
        --hidden-import vajra_bm25 \
        --hidden-import langextract \
        --hidden-import google.generativeai \
        --hidden-import ollama \
        --exclude-module matplotlib \
        --exclude-module scipy \
        --exclude-module sklearn \
        --exclude-module torch \
        --exclude-module tensorflow \
        --exclude-module keras \
        --exclude-module cv2 \
        --exclude-module PIL \
        --exclude-module tkinter \
        --exclude-module test \
        --exclude-module tests \
        src/agentic_research/api/embedded_main.py

    if [ -f "dist/praval-backend" ] || [ -f "dist/praval-backend.exe" ]; then
        BINARY_SIZE=$(du -h dist/praval-backend* 2>/dev/null | cut -f1)
        print_success "Backend binary built: $BINARY_SIZE"

        # Copy to Tauri binaries directory with platform-specific name
        mkdir -p desktop/src-tauri/binaries
        if [ -f "dist/praval-backend" ]; then
            cp dist/praval-backend "desktop/src-tauri/binaries/$BINARY_NAME"
        else
            cp dist/praval-backend.exe "desktop/src-tauri/binaries/$BINARY_NAME"
        fi
        print_success "Binary copied to desktop/src-tauri/binaries/$BINARY_NAME"
    else
        print_error "Backend build failed!"
        exit 1
    fi

    # Cleanup PyInstaller artifacts
    rm -rf build/ praval-backend.spec

    deactivate 2>/dev/null || true
else
    print_warning "Skipping backend build"
fi

# ============================================================================
# Build Frontend
# ============================================================================
if [ "$SKIP_FRONTEND" = false ]; then
    print_status "Building frontend..."
    cd frontend-new

    # Install dependencies if needed
    if [ ! -d "node_modules" ]; then
        npm install --silent
    fi

    npm run build
    cd ..
    print_success "Frontend built"
else
    print_warning "Skipping frontend build"
fi

# ============================================================================
# Build Tauri Desktop App
# ============================================================================
print_status "Building Tauri desktop app..."
cd desktop

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    npm install --silent
fi

if [ "$DEV_MODE" = true ]; then
    npm run tauri dev
else
    npm run tauri build

    # Find and report build output
    echo ""
    case "$OS" in
        Darwin)
            DMG_PATH=$(find src-tauri/target/release/bundle/dmg -name "*.dmg" 2>/dev/null | head -1)
            if [ -n "$DMG_PATH" ] && [ -f "$DMG_PATH" ]; then
                print_success "Build complete!"
                echo ""
                echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
                echo -e "${GREEN}║${NC}                    Build Complete!                          ${GREEN}║${NC}"
                echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
                echo ""
                echo "macOS Installer: $DMG_PATH"
                echo ""
            fi
            ;;
        Linux)
            APPIMAGE_PATH=$(find src-tauri/target/release/bundle/appimage -name "*.AppImage" 2>/dev/null | head -1)
            DEB_PATH=$(find src-tauri/target/release/bundle/deb -name "*.deb" 2>/dev/null | head -1)
            if [ -n "$APPIMAGE_PATH" ] || [ -n "$DEB_PATH" ]; then
                print_success "Build complete!"
                echo ""
                echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
                echo -e "${GREEN}║${NC}                    Build Complete!                          ${GREEN}║${NC}"
                echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
                echo ""
                [ -n "$APPIMAGE_PATH" ] && echo "AppImage: $APPIMAGE_PATH"
                [ -n "$DEB_PATH" ] && echo "Debian Package: $DEB_PATH"
                echo ""
            fi
            ;;
    esac
fi

cd ..
print_success "All builds complete!"
