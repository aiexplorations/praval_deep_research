# Backend Binaries

This directory contains the bundled Python backend binaries for different platforms.

## Binary Naming Convention

Tauri expects external binaries to follow a specific naming convention:
```
{name}-{target-triple}[.exe]
```

For the Praval backend, the binaries should be named:

| Platform | Architecture | Binary Name |
|----------|-------------|-------------|
| macOS | Intel | `praval-backend-x86_64-apple-darwin` |
| macOS | Apple Silicon | `praval-backend-aarch64-apple-darwin` |
| Linux | x64 | `praval-backend-x86_64-unknown-linux-gnu` |
| Linux | ARM64 | `praval-backend-aarch64-unknown-linux-gnu` |
| Windows | x64 | `praval-backend-x86_64-pc-windows-msvc.exe` |

## Building Binaries

The install scripts (`install-macos.sh`, `install-linux.sh`, `install-windows.ps1`) automatically build and copy the appropriate binary to this directory.

You can also build manually:

```bash
# From project root
./scripts/build-all.sh

# Or just the backend
./scripts/build-backend-binary.sh
```

## How It Works

1. **Build Time**: PyInstaller bundles Python + all dependencies into a single executable (~150-200MB)
2. **Distribution**: Tauri includes the binary in the app bundle
3. **Runtime**: The Tauri app launches the binary as a sidecar process
4. **No Python Required**: End users don't need Python installed

## Development Mode

If no binary is found, the Tauri app falls back to using the Python module directly:
```bash
# Force Python mode for development
PRAVAL_USE_PYTHON=1 npm run tauri dev
```

## Excluded Dependencies

To reduce binary size, these packages are excluded:
- matplotlib
- scipy
- sklearn
- torch
- tensorflow
- keras
- cv2/PIL
- tkinter

If you need these, edit the PyInstaller command in the build scripts.
