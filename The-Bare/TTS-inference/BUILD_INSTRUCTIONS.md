# TTS Inference Build Instructions

This document provides step-by-step instructions for setting up the environment and building the standalone executable for the TTS inference engine (`aud_test.py`) on Linux, macOS, and Windows.

The resulting executable removes the need for users to have Python installed and allows the application to run the TTS engine via a binary.

## Prerequisites

- **Python 3.10+** installed.
- **Git** (to clone the repo).

## 1. Setup Virtual Environment

Open your terminal or command prompt and navigate to the `The-Bare/TTS-inference` directory.

### Linux / macOS

```bash
cd The-Bare/TTS-inference

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade base tools
pip install --upgrade pip setuptools wheel
```

### Windows (PowerShell)

```powershell
cd The-Bare\TTS-inference

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\Activate.ps1

# Upgrade base tools
pip install --upgrade pip setuptools wheel
```

## 2. Install Dependencies

Install the required packages from `requirements.txt` and install `pyinstaller`.

```bash
pip install -r requirements.txt
pip install pyinstaller
```

## 3. Build the Executable

We use PyInstaller to build a "One-Directory" (`--onedir`) bundle. We name the executable `tts-inference` (instead of the default `aud_test`).

### Command

```bash
pyinstaller --clean --onedir --name tts-inference aud_test.py
```

*Note: If you encounter missing module errors during runtime (e.g. `chichat` not found), you may need to add `--hidden-import=chichat` to the command, though PyInstaller usually detects it automatically.*

## 4. Output Location

The build process will create:
- A `build/` directory (temporary intermediate files).
- A `dist/` directory containing the final output.
- A `tts-inference.spec` file.

The runnable executable will be located at:

- **Linux**: `dist/tts-inference/tts-inference`
- **macOS**: `dist/tts-inference/tts-inference`
- **Windows**: `dist/tts-inference/tts-inference.exe`

## 5. Integration with GenHat

To use this built binary in the GenHat desktop app, you must move the entire `tts-inference` directory from `dist/` to the appropriate platform folder in `src-tauri/bin/`.

### Directory Structure

The destination structure must look like this:

```
genhat-desktop/src-tauri/bin/
├── tts-lin/
│   └── tts-inference/       <-- The folder from dist/
│       ├── tts-inference    <-- The executable
│       └── _internal/       <-- Dependencies
├── tts-mac/
│   └── tts-inference/
│       ├── tts-inference
│       └── ...
└── tts-win/
    └── tts-inference/
        ├── tts-inference.exe
        └── ...
```

### Automation Commands

**Linux:**
```bash
# From The-Bare/TTS-inference
rm -rf ../../genhat-desktop/src-tauri/bin/tts-lin/tts-inference
mkdir -p ../../genhat-desktop/src-tauri/bin/tts-lin
cp -r dist/tts-inference ../../genhat-desktop/src-tauri/bin/tts-lin/
```

**macOS:**
```bash
# From The-Bare/TTS-inference
rm -rf ../../genhat-desktop/src-tauri/bin/tts-mac/tts-inference
mkdir -p ../../genhat-desktop/src-tauri/bin/tts-mac
cp -r dist/tts-inference ../../genhat-desktop/src-tauri/bin/tts-mac/
```

**Windows (PowerShell):**
```powershell
# From The-Bare\TTS-inference
if (Test-Path ..\..\genhat-desktop\src-tauri\bin\tts-win\tts-inference) {
    Remove-Item -Recurse -Force ..\..\genhat-desktop\src-tauri\bin\tts-win\tts-inference
}
New-Item -ItemType Directory -Force -Path ..\..\genhat-desktop\src-tauri\bin\tts-win
Copy-Item -Recurse -Path dist\tts-inference -Destination ..\..\genhat-desktop\src-tauri\bin\tts-win\
```
