#!/bin/bash
#
# Praval Deep Research - Linux Install Script
#
# This script installs all dependencies and builds the desktop app.
# Supports: Ubuntu/Debian, Fedora/RHEL, Arch Linux
#
# Run with: ./scripts/install-linux.sh
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with color
print_status() { echo -e "${BLUE}==>${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

# Header
echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║${NC}        Praval Deep Research - Linux Installer              ${BLUE}║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

print_status "Project root: $PROJECT_ROOT"

# ============================================================================
# Detect Linux Distribution
# ============================================================================
print_status "Detecting Linux distribution..."

if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
    DISTRO_LIKE=$ID_LIKE
else
    print_error "Cannot detect Linux distribution"
    exit 1
fi

print_success "Detected: $DISTRO ($PRETTY_NAME)"

# Determine package manager
PKG_MANAGER=""
PKG_INSTALL=""
PKG_UPDATE=""

case "$DISTRO" in
    ubuntu|debian|linuxmint|pop)
        PKG_MANAGER="apt"
        PKG_INSTALL="sudo apt install -y"
        PKG_UPDATE="sudo apt update"
        ;;
    fedora|rhel|centos|rocky|almalinux)
        PKG_MANAGER="dnf"
        PKG_INSTALL="sudo dnf install -y"
        PKG_UPDATE="sudo dnf check-update || true"
        ;;
    arch|manjaro|endeavouros)
        PKG_MANAGER="pacman"
        PKG_INSTALL="sudo pacman -S --noconfirm"
        PKG_UPDATE="sudo pacman -Sy"
        ;;
    opensuse*|suse)
        PKG_MANAGER="zypper"
        PKG_INSTALL="sudo zypper install -y"
        PKG_UPDATE="sudo zypper refresh"
        ;;
    *)
        # Try to detect from ID_LIKE
        if [[ "$DISTRO_LIKE" == *"debian"* ]] || [[ "$DISTRO_LIKE" == *"ubuntu"* ]]; then
            PKG_MANAGER="apt"
            PKG_INSTALL="sudo apt install -y"
            PKG_UPDATE="sudo apt update"
        elif [[ "$DISTRO_LIKE" == *"fedora"* ]] || [[ "$DISTRO_LIKE" == *"rhel"* ]]; then
            PKG_MANAGER="dnf"
            PKG_INSTALL="sudo dnf install -y"
            PKG_UPDATE="sudo dnf check-update || true"
        elif [[ "$DISTRO_LIKE" == *"arch"* ]]; then
            PKG_MANAGER="pacman"
            PKG_INSTALL="sudo pacman -S --noconfirm"
            PKG_UPDATE="sudo pacman -Sy"
        else
            print_error "Unsupported distribution: $DISTRO"
            print_warning "Please install dependencies manually and run: cd desktop && npm run tauri build"
            exit 1
        fi
        ;;
esac

print_success "Using package manager: $PKG_MANAGER"

# ============================================================================
# Update Package Lists
# ============================================================================
print_status "Updating package lists..."
$PKG_UPDATE
print_success "Package lists updated"

# ============================================================================
# Install System Dependencies for Tauri
# ============================================================================
print_status "Installing Tauri system dependencies..."

case "$PKG_MANAGER" in
    apt)
        $PKG_INSTALL \
            libwebkit2gtk-4.1-dev \
            libappindicator3-dev \
            librsvg2-dev \
            patchelf \
            libssl-dev \
            libgtk-3-dev \
            libayatana-appindicator3-dev \
            build-essential \
            curl \
            wget \
            file
        ;;
    dnf)
        $PKG_INSTALL \
            webkit2gtk4.1-devel \
            openssl-devel \
            curl \
            wget \
            file \
            libappindicator-gtk3-devel \
            librsvg2-devel \
            gtk3-devel \
            patchelf \
            @development-tools
        ;;
    pacman)
        $PKG_INSTALL \
            webkit2gtk-4.1 \
            base-devel \
            curl \
            wget \
            file \
            openssl \
            appmenu-gtk-module \
            libappindicator-gtk3 \
            librsvg \
            patchelf
        ;;
    zypper)
        $PKG_INSTALL \
            webkit2gtk3-soup2-devel \
            libopenssl-devel \
            curl \
            wget \
            file \
            libappindicator3-1 \
            librsvg-devel \
            patchelf \
            -t pattern devel_basis
        ;;
esac

print_success "System dependencies installed"

# ============================================================================
# Check/Install Node.js
# ============================================================================
print_status "Checking for Node.js..."
if ! command -v node &> /dev/null; then
    print_warning "Node.js not found. Installing..."

    # Use NodeSource for consistent Node.js version
    case "$PKG_MANAGER" in
        apt)
            curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
            sudo apt install -y nodejs
            ;;
        dnf)
            curl -fsSL https://rpm.nodesource.com/setup_20.x | sudo bash -
            sudo dnf install -y nodejs
            ;;
        pacman)
            $PKG_INSTALL nodejs npm
            ;;
        zypper)
            $PKG_INSTALL nodejs20 npm20
            ;;
    esac
    print_success "Node.js installed"
else
    NODE_VERSION=$(node --version)
    print_success "Node.js found: $NODE_VERSION"

    # Check version is 18+
    NODE_MAJOR=$(echo "$NODE_VERSION" | cut -d'.' -f1 | tr -d 'v')
    if [ "$NODE_MAJOR" -lt 18 ]; then
        print_warning "Node.js 18+ required. Please upgrade Node.js manually."
        exit 1
    fi
fi

# ============================================================================
# Check/Install Rust
# ============================================================================
print_status "Checking for Rust..."
if ! command -v rustc &> /dev/null; then
    print_warning "Rust not found. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
    print_success "Rust installed"
else
    RUST_VERSION=$(rustc --version)
    print_success "Rust found: $RUST_VERSION"
fi

# Ensure cargo is in PATH
export PATH="$HOME/.cargo/bin:$PATH"

# ============================================================================
# Check/Install Python
# ============================================================================
print_status "Checking for Python 3.11+..."
PYTHON_CMD=""
for cmd in python3.13 python3.12 python3.11 python3; do
    if command -v $cmd &> /dev/null; then
        PY_VERSION=$($cmd --version 2>&1 | cut -d' ' -f2)
        PY_MAJOR=$(echo "$PY_VERSION" | cut -d'.' -f1)
        PY_MINOR=$(echo "$PY_VERSION" | cut -d'.' -f2)
        if [ "$PY_MAJOR" -ge 3 ] && [ "$PY_MINOR" -ge 11 ]; then
            PYTHON_CMD=$cmd
            break
        fi
    fi
done

if [ -z "$PYTHON_CMD" ]; then
    print_warning "Python 3.11+ not found. Installing..."
    case "$PKG_MANAGER" in
        apt)
            # Add deadsnakes PPA for newer Python on older Ubuntu
            sudo add-apt-repository -y ppa:deadsnakes/ppa 2>/dev/null || true
            sudo apt update
            $PKG_INSTALL python3.12 python3.12-venv python3.12-dev
            PYTHON_CMD="python3.12"
            ;;
        dnf)
            $PKG_INSTALL python3.12 python3.12-devel
            PYTHON_CMD="python3.12"
            ;;
        pacman)
            $PKG_INSTALL python python-pip
            PYTHON_CMD="python"
            ;;
        zypper)
            $PKG_INSTALL python312 python312-devel
            PYTHON_CMD="python3.12"
            ;;
    esac
    print_success "Python installed"
else
    print_success "Python found: $($PYTHON_CMD --version)"
fi

# ============================================================================
# Optional: Install Ollama for local LLM support
# ============================================================================
print_status "Checking for Ollama (optional, for local LLM support)..."
if ! command -v ollama &> /dev/null; then
    echo ""
    read -p "Would you like to install Ollama for free local LLM support? (y/N) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Installing Ollama..."
        curl -fsSL https://ollama.ai/install.sh | sh
        print_success "Ollama installed"
        echo ""
        print_status "To use Ollama, run these commands after installation:"
        echo "    ollama serve        # Start the Ollama service (or use systemd)"
        echo "    ollama pull llama3.2  # Download a model"
        echo ""
        echo "To enable Ollama as a service:"
        echo "    sudo systemctl enable ollama"
        echo "    sudo systemctl start ollama"
        echo ""
    else
        print_warning "Skipping Ollama. You'll need an OpenAI or Anthropic API key."
    fi
else
    print_success "Ollama found: $(ollama --version 2>/dev/null || echo 'installed')"
fi

# ============================================================================
# Build Python Backend Binary
# ============================================================================
print_status "Building Python backend binary..."
echo "This creates a standalone executable (no Python needed at runtime)"
echo ""

# Create virtual environment for building
if [ ! -d "venv" ]; then
    $PYTHON_CMD -m venv venv
    print_success "Virtual environment created"
fi

source venv/bin/activate

# Install build dependencies
print_status "Installing build dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install pyinstaller -q

# Install project dependencies (needed for PyInstaller analysis)
print_status "Installing project dependencies for analysis..."
pip install -r requirements.txt -q 2>/dev/null || {
    print_warning "Some dependencies had issues, continuing..."
}

# Build binary with PyInstaller
print_status "Running PyInstaller (this may take a few minutes)..."

pyinstaller \
    --name praval-backend \
    --onefile \
    --console \
    --clean \
    --noconfirm \
    --add-data "src/agentic_research:agentic_research" \
    --hidden-import uvicorn \
    --hidden-import uvicorn.logging \
    --hidden-import uvicorn.loops.auto \
    --hidden-import uvicorn.protocols.http.auto \
    --hidden-import uvicorn.protocols.websockets.auto \
    --hidden-import uvicorn.lifespan.on \
    --hidden-import fastapi \
    --hidden-import starlette \
    --hidden-import pydantic \
    --hidden-import pydantic_settings \
    --hidden-import aiohttp \
    --hidden-import openai \
    --hidden-import anthropic \
    --hidden-import tiktoken \
    --hidden-import tiktoken_ext.openai_public \
    --hidden-import numpy \
    --hidden-import pandas \
    --hidden-import qdrant_client \
    --hidden-import structlog \
    --hidden-import httpx \
    --hidden-import praval \
    --hidden-import vajra_bm25 \
    --hidden-import langextract \
    --hidden-import ollama \
    --exclude-module matplotlib \
    --exclude-module scipy \
    --exclude-module torch \
    --exclude-module tensorflow \
    --exclude-module PIL \
    --exclude-module tkinter \
    src/agentic_research/api/embedded_main.py 2>&1 | grep -E "(Building|Copying|completed|ERROR)" || true

# Check if binary was built
if [ -f "dist/praval-backend" ]; then
    BINARY_SIZE=$(du -h dist/praval-backend | cut -f1)
    print_success "Backend binary built: $BINARY_SIZE"

    # Copy to Tauri binaries directory
    ARCH=$(uname -m)
    if [ "$ARCH" = "aarch64" ]; then
        BINARY_NAME="praval-backend-aarch64-unknown-linux-gnu"
    else
        BINARY_NAME="praval-backend-x86_64-unknown-linux-gnu"
    fi

    mkdir -p desktop/src-tauri/binaries
    cp dist/praval-backend "desktop/src-tauri/binaries/$BINARY_NAME"
    print_success "Binary copied to desktop/src-tauri/binaries/"
else
    print_error "Backend binary build failed!"
    print_warning "Check the build output above for errors."
    print_warning "Continuing with Tauri build (may use Python fallback)..."
fi

# Cleanup PyInstaller artifacts
rm -rf build/ praval-backend.spec 2>/dev/null || true

deactivate 2>/dev/null || true
print_success "Python backend build complete"

# ============================================================================
# Install Frontend Dependencies
# ============================================================================
print_status "Installing frontend dependencies..."
cd frontend-new
npm install --silent
cd ..
print_success "Frontend dependencies installed"

# ============================================================================
# Install Tauri CLI and Desktop Dependencies
# ============================================================================
print_status "Installing Tauri CLI..."
cd desktop
npm install --silent
cd ..
print_success "Tauri CLI installed"

# ============================================================================
# Build the Desktop App
# ============================================================================
echo ""
print_status "Building Praval Deep Research desktop app..."
echo "This may take a few minutes on first build..."
echo ""

cd desktop
npm run tauri build 2>&1 | while IFS= read -r line; do
    if [[ "$line" == *"Finished"* ]] || [[ "$line" == *"Bundling"* ]]; then
        echo -e "${GREEN}$line${NC}"
    elif [[ "$line" == *"error"* ]] || [[ "$line" == *"Error"* ]]; then
        echo -e "${RED}$line${NC}"
    else
        echo "$line"
    fi
done
cd ..

# ============================================================================
# Find and Report Build Output
# ============================================================================
echo ""

# Look for AppImage, deb, or rpm
APPIMAGE_PATH=$(find desktop/src-tauri/target/release/bundle/appimage -name "*.AppImage" 2>/dev/null | head -1)
DEB_PATH=$(find desktop/src-tauri/target/release/bundle/deb -name "*.deb" 2>/dev/null | head -1)
RPM_PATH=$(find desktop/src-tauri/target/release/bundle/rpm -name "*.rpm" 2>/dev/null | head -1)

if [ -n "$APPIMAGE_PATH" ] || [ -n "$DEB_PATH" ] || [ -n "$RPM_PATH" ]; then
    print_success "Build complete!"
    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║${NC}                    Installation Complete!                   ${GREEN}║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Your app packages are ready:"

    if [ -n "$APPIMAGE_PATH" ]; then
        echo -e "  AppImage: ${BLUE}$APPIMAGE_PATH${NC}"
    fi
    if [ -n "$DEB_PATH" ]; then
        echo -e "  Debian:   ${BLUE}$DEB_PATH${NC}"
    fi
    if [ -n "$RPM_PATH" ]; then
        echo -e "  RPM:      ${BLUE}$RPM_PATH${NC}"
    fi

    echo ""
    echo "To install:"
    if [ -n "$APPIMAGE_PATH" ]; then
        echo "  AppImage (universal): chmod +x $APPIMAGE_PATH && $APPIMAGE_PATH"
    fi
    if [ -n "$DEB_PATH" ]; then
        echo "  Debian/Ubuntu: sudo dpkg -i $DEB_PATH"
    fi
    if [ -n "$RPM_PATH" ]; then
        echo "  Fedora/RHEL: sudo rpm -i $RPM_PATH"
    fi

    echo ""
    echo "First launch:"
    echo "  • Go to Settings to configure your LLM provider"
    echo "  • Enter API keys for OpenAI/Anthropic, OR"
    echo "  • Select Ollama for free local models"
    echo ""

    # Offer to run AppImage
    if [ -n "$APPIMAGE_PATH" ]; then
        read -p "Would you like to run the app now? (Y/n) " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            chmod +x "$APPIMAGE_PATH"
            "$APPIMAGE_PATH" &
        fi
    fi
else
    print_error "Build may have failed. Check the output above for errors."
    echo ""
    echo "You can try building manually:"
    echo "  cd desktop && npm run tauri build"
    exit 1
fi
