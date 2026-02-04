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

# Check for system dependencies (pyaudio needs portaudio)
echo "→ Checking system dependencies..."
if ! pkg-config --exists portaudio-2.0; then
    echo "⚠  portaudio-dev not found"
    echo "   Installing system dependencies (requires sudo)..."
    echo
    sudo apt-get update
    sudo apt-get install -y portaudio19-dev python3-pyaudio
    echo "✓ System dependencies installed"
else
    echo "✓ System dependencies OK"
fi

echo

# Install package in development mode
echo "→ Installing multi-modal-ai-studio in development mode..."
pip install -e .

# Optionally install audio support (for USB devices / headless mode)
echo
read -p "Install USB audio support (pyaudio)? Needed for headless mode [y/N]: " install_audio
if [[ "$install_audio" =~ ^[Yy]$ ]]; then
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
