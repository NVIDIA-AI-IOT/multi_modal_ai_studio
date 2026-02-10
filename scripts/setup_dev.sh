#!/bin/bash
#
# Development Environment Setup Script
# Sets up virtual environment and installs all dependencies
#

set -e  # Exit on error

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "========================================="
echo "Multi-modal AI Studio - Dev Setup"
echo "========================================="
echo

# Check if venv exists
if [ -d ".venv" ]; then
    echo "✓ Virtual environment already exists"
else
    echo "→ Creating virtual environment..."
    python3 -m venv .venv
    echo "✓ Virtual environment created"
fi

echo

# Activate venv
echo "→ Activating virtual environment..."
source .venv/bin/activate

echo "→ Upgrading pip..."
pip install --upgrade pip setuptools wheel

echo

# Install package in development mode first (no portaudio required for ALSA-only use)
echo "→ Installing multi-modal-ai-studio in development mode..."
pip install -e .

# Optionally install audio support (PyAudio for pyaudio:N devices; ALSA devices like EMEET use arecord and don't need this)
echo
read -p "Install USB audio support (pyaudio)? Only needed for pyaudio:N devices; ALSA mics (e.g. EMEET) work without it [y/N]: " install_audio
if [[ "$install_audio" =~ ^[Yy]$ ]]; then
    # PyAudio build requires portaudio dev headers
    if ! pkg-config --exists portaudio-2.0; then
        echo "→ Installing portaudio (required for pyaudio build)..."
        sudo apt-get update
        sudo apt-get install -y portaudio19-dev
    fi
    echo "→ Installing audio dependencies..."
    pip install -e ".[audio]"
    echo "✓ Audio support installed"
else
    echo "⊘ Skipping audio support (WebUI with browser audio will still work)"
fi

echo
echo "========================================="
echo "✓ Setup complete!"
echo "========================================="
echo
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo
echo "To deactivate:"
echo "  deactivate"
echo
echo "To test backends:"
echo "  python3 scripts/test_backends.py"
echo
