#!/bin/bash
# Launch Multi-modal AI Studio WebUI (default: sessions/). Use --session-dir mock_sessions to load sample data.

cd "$(dirname "$0")/.."

echo "========================================="
echo "Multi-modal AI Studio - WebUI Dev Server"
echo "========================================="
echo

# Check if virtual environment is active
if [ -z "$VIRTUAL_ENV" ]; then
    echo "⚠️  Virtual environment not active"
    echo "Run: source .venv/bin/activate"
    echo
fi

# Start the server (default session dir: sessions/)
echo "🚀 Starting WebUI server..."
echo "📂 Session directory: $(pwd)/sessions (default; use --session-dir mock_sessions to load sample data)"
echo
python src/multi_modal_ai_studio/webui/server.py \
    --host 0.0.0.0 \
    --port 8080
