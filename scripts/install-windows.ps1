#
# Praval Deep Research - Windows Install Script
#
# This script installs all dependencies and builds the desktop app.
# Run with: .\scripts\install-windows.ps1
#
# Note: Run PowerShell as Administrator for best results
#

$ErrorActionPreference = "Stop"

# Colors
function Write-Status { param($msg) Write-Host "==> " -ForegroundColor Blue -NoNewline; Write-Host $msg }
function Write-Success { param($msg) Write-Host "✓ " -ForegroundColor Green -NoNewline; Write-Host $msg }
function Write-Warning { param($msg) Write-Host "⚠ " -ForegroundColor Yellow -NoNewline; Write-Host $msg }
function Write-Error { param($msg) Write-Host "✗ " -ForegroundColor Red -NoNewline; Write-Host $msg }

# Header
Write-Host ""
Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Blue
Write-Host "║        Praval Deep Research - Windows Installer            ║" -ForegroundColor Blue
Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Blue
Write-Host ""

# Get project root
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
Set-Location $ProjectRoot

Write-Status "Project root: $ProjectRoot"

# ============================================================================
# Check for winget (Windows Package Manager)
# ============================================================================
Write-Status "Checking for winget..."
$hasWinget = Get-Command winget -ErrorAction SilentlyContinue
if (-not $hasWinget) {
    Write-Error "winget not found. Please install App Installer from Microsoft Store."
    Write-Host "https://www.microsoft.com/store/productId/9NBLGGH4NNS1"
    exit 1
}
Write-Success "winget found"

# ============================================================================
# Check/Install Visual Studio Build Tools
# ============================================================================
Write-Status "Checking for Visual Studio Build Tools..."
$vsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
$hasVS = $false

if (Test-Path $vsWhere) {
    $vsInstall = & $vsWhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
    if ($vsInstall) {
        $hasVS = $true
    }
}

if (-not $hasVS) {
    Write-Warning "Visual Studio Build Tools not found. Installing..."
    Write-Host "This will install the C++ build tools required for Rust compilation."
    winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
    Write-Success "Visual Studio Build Tools installed"
    Write-Warning "You may need to restart your terminal after installation."
} else {
    Write-Success "Visual Studio Build Tools found"
}

# ============================================================================
# Check/Install Node.js
# ============================================================================
Write-Status "Checking for Node.js..."
$nodeCmd = Get-Command node -ErrorAction SilentlyContinue

if (-not $nodeCmd) {
    Write-Warning "Node.js not found. Installing..."
    winget install OpenJS.NodeJS.LTS
    # Refresh PATH
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    Write-Success "Node.js installed"
} else {
    $nodeVersion = node --version
    Write-Success "Node.js found: $nodeVersion"

    # Check version
    $major = [int]($nodeVersion -replace 'v(\d+)\..*', '$1')
    if ($major -lt 18) {
        Write-Warning "Node.js 18+ required. Upgrading..."
        winget upgrade OpenJS.NodeJS.LTS
    }
}

# ============================================================================
# Check/Install Rust
# ============================================================================
Write-Status "Checking for Rust..."
$rustCmd = Get-Command rustc -ErrorAction SilentlyContinue

if (-not $rustCmd) {
    Write-Warning "Rust not found. Installing..."
    Write-Host "Downloading rustup installer..."

    $rustupInit = "$env:TEMP\rustup-init.exe"
    Invoke-WebRequest -Uri "https://win.rustup.rs/x86_64" -OutFile $rustupInit

    & $rustupInit -y

    # Add to PATH
    $env:Path += ";$env:USERPROFILE\.cargo\bin"
    [System.Environment]::SetEnvironmentVariable("Path", $env:Path + ";$env:USERPROFILE\.cargo\bin", "User")

    Write-Success "Rust installed"
} else {
    $rustVersion = rustc --version
    Write-Success "Rust found: $rustVersion"
}

# ============================================================================
# Check/Install Python
# ============================================================================
Write-Status "Checking for Python 3.11+..."
$pythonCmd = $null

foreach ($cmd in @("python3.12", "python3.11", "python")) {
    $testCmd = Get-Command $cmd -ErrorAction SilentlyContinue
    if ($testCmd) {
        $pyVersion = & $cmd --version 2>&1
        if ($pyVersion -match "Python (\d+)\.(\d+)") {
            $pyMajor = [int]$Matches[1]
            $pyMinor = [int]$Matches[2]
            if ($pyMajor -ge 3 -and $pyMinor -ge 11) {
                $pythonCmd = $cmd
                break
            }
        }
    }
}

if (-not $pythonCmd) {
    Write-Warning "Python 3.11+ not found. Installing..."
    winget install Python.Python.3.12
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")
    $pythonCmd = "python"
    Write-Success "Python installed"
} else {
    $pyVersion = & $pythonCmd --version
    Write-Success "Python found: $pyVersion"
}

# ============================================================================
# Optional: Install Ollama
# ============================================================================
Write-Status "Checking for Ollama (optional, for local LLM support)..."
$ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue

if (-not $ollamaCmd) {
    Write-Host ""
    $response = Read-Host "Would you like to install Ollama for free local LLM support? (y/N)"
    if ($response -match "^[Yy]") {
        Write-Status "Installing Ollama..."
        winget install Ollama.Ollama
        Write-Success "Ollama installed"
        Write-Host ""
        Write-Status "To use Ollama, run these commands after installation:"
        Write-Host "    ollama serve        # Start the Ollama service"
        Write-Host "    ollama pull llama3.2  # Download a model"
        Write-Host ""
    } else {
        Write-Warning "Skipping Ollama. You'll need an OpenAI or Anthropic API key."
    }
} else {
    Write-Success "Ollama found"
}

# ============================================================================
# Build Python Backend Binary
# ============================================================================
Write-Status "Building Python backend binary..."
Write-Host "This creates a standalone executable (no Python needed at runtime)"
Write-Host ""

# Create virtual environment for building
if (-not (Test-Path "venv")) {
    & $pythonCmd -m venv venv
    Write-Success "Virtual environment created"
}

# Activate virtual environment
& ".\venv\Scripts\Activate.ps1"

# Install build dependencies
Write-Status "Installing build dependencies..."
pip install --upgrade pip setuptools wheel -q
pip install pyinstaller -q

# Install project dependencies (needed for PyInstaller analysis)
Write-Status "Installing project dependencies for analysis..."
pip install -r requirements.txt -q 2>$null
Write-Success "Dependencies installed"

# Build binary with PyInstaller
Write-Status "Running PyInstaller (this may take a few minutes)..."

pyinstaller `
    --name praval-backend `
    --onefile `
    --console `
    --clean `
    --noconfirm `
    --add-data "src/agentic_research;agentic_research" `
    --hidden-import uvicorn `
    --hidden-import uvicorn.logging `
    --hidden-import uvicorn.loops.auto `
    --hidden-import uvicorn.protocols.http.auto `
    --hidden-import uvicorn.protocols.websockets.auto `
    --hidden-import uvicorn.lifespan.on `
    --hidden-import fastapi `
    --hidden-import starlette `
    --hidden-import pydantic `
    --hidden-import pydantic_settings `
    --hidden-import aiohttp `
    --hidden-import openai `
    --hidden-import anthropic `
    --hidden-import tiktoken `
    --hidden-import tiktoken_ext.openai_public `
    --hidden-import numpy `
    --hidden-import pandas `
    --hidden-import qdrant_client `
    --hidden-import structlog `
    --hidden-import httpx `
    --hidden-import praval `
    --hidden-import vajra_bm25 `
    --hidden-import langextract `
    --hidden-import ollama `
    --exclude-module matplotlib `
    --exclude-module scipy `
    --exclude-module torch `
    --exclude-module tensorflow `
    --exclude-module PIL `
    --exclude-module tkinter `
    src/agentic_research/api/embedded_main.py

# Check if binary was built
if (Test-Path "dist\praval-backend.exe") {
    $binarySize = (Get-Item "dist\praval-backend.exe").Length / 1MB
    Write-Success "Backend binary built: $([math]::Round($binarySize, 1)) MB"

    # Copy to Tauri binaries directory
    $binaryName = "praval-backend-x86_64-pc-windows-msvc.exe"

    New-Item -ItemType Directory -Force -Path "desktop\src-tauri\binaries" | Out-Null
    Copy-Item "dist\praval-backend.exe" "desktop\src-tauri\binaries\$binaryName"
    Write-Success "Binary copied to desktop\src-tauri\binaries\"
} else {
    Write-Error "Backend binary build failed!"
    Write-Warning "Check the build output above for errors."
    Write-Warning "Continuing with Tauri build (may use Python fallback)..."
}

# Cleanup PyInstaller artifacts
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
Remove-Item -Force praval-backend.spec -ErrorAction SilentlyContinue

Write-Success "Python backend build complete"

# ============================================================================
# Install Frontend Dependencies
# ============================================================================
Write-Status "Installing frontend dependencies..."
Set-Location frontend-new
npm install --silent 2>$null
Set-Location ..
Write-Success "Frontend dependencies installed"

# ============================================================================
# Install Tauri CLI and Desktop Dependencies
# ============================================================================
Write-Status "Installing Tauri CLI..."
Set-Location desktop
npm install --silent 2>$null
Set-Location ..
Write-Success "Tauri CLI installed"

# ============================================================================
# Build the Desktop App
# ============================================================================
Write-Host ""
Write-Status "Building Praval Deep Research desktop app..."
Write-Host "This may take a few minutes on first build..."
Write-Host ""

Set-Location desktop
npm run tauri build
Set-Location ..

# ============================================================================
# Find and Report Build Output
# ============================================================================
Write-Host ""
$msiPath = Get-ChildItem -Path "desktop\src-tauri\target\release\bundle\msi\*.msi" -ErrorAction SilentlyContinue | Select-Object -First 1

if ($msiPath) {
    Write-Success "Build complete!"
    Write-Host ""
    Write-Host "╔════════════════════════════════════════════════════════════╗" -ForegroundColor Green
    Write-Host "║                    Installation Complete!                   ║" -ForegroundColor Green
    Write-Host "╚════════════════════════════════════════════════════════════╝" -ForegroundColor Green
    Write-Host ""
    Write-Host "Your app is ready at:"
    Write-Host "  $($msiPath.FullName)" -ForegroundColor Blue
    Write-Host ""
    Write-Host "To install:"
    Write-Host "  1. Double-click the .msi file"
    Write-Host "  2. Follow the installation wizard"
    Write-Host "  3. Launch from Start Menu"
    Write-Host ""
    Write-Host "First launch:"
    Write-Host "  - Go to Settings to configure your LLM provider"
    Write-Host "  - Enter API keys for OpenAI/Anthropic, OR"
    Write-Host "  - Select Ollama for free local models"
    Write-Host ""

    # Offer to run installer
    $response = Read-Host "Would you like to run the installer now? (Y/n)"
    if ($response -notmatch "^[Nn]") {
        Start-Process $msiPath.FullName
    }
} else {
    Write-Error "Build may have failed. Check the output above for errors."
    Write-Host ""
    Write-Host "You can try building manually:"
    Write-Host "  cd desktop; npm run tauri build"
    exit 1
}
