#!/bin/bash
# Launch Multi-modal AI Studio WebUI with mock data

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

# Check if mock sessions exist
if [ ! -d "mock_sessions" ] || [ -z "$(ls -A mock_sessions/*.json 2>/dev/null)" ]; then
    echo "📝 Generating mock session data..."
    python scripts/generate_mock_sessions.py
    echo
fi

# Start the server
echo "🚀 Starting WebUI server..."
echo "📂 Session directory: $(pwd)/mock_sessions"
echo
python src/multi_modal_ai_studio/webui/server.py \
    --host 0.0.0.0 \
    --port 8080 \
    --session-dir mock_sessions
