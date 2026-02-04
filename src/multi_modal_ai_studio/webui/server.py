#!/usr/bin/env python3
"""
Simple web server for Multi-modal AI Studio WebUI.

Serves static HTML/CSS/JS and provides API endpoints for session data.
"""

import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any

from aiohttp import web

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebUIServer:
    """Web server for Multi-modal AI Studio UI."""
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8080, session_dir: Path = None):
        """Initialize web server.
        
        Args:
            host: Host address to bind to
            port: Port to listen on
            session_dir: Directory containing session JSON files
        """
        self.host = host
        self.port = port
        self.session_dir = session_dir or Path("mock_sessions")
        self.app = web.Application()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup HTTP routes."""
        self.app.router.add_get('/api/sessions', self.handle_list_sessions)
        self.app.router.add_get('/api/sessions/{session_id}', self.handle_get_session)
        
        # Serve static files
        static_dir = Path(__file__).parent / 'static'
        self.app.router.add_get('/', self.handle_index)
        self.app.router.add_static('/static', static_dir, name='static')
        self.app.router.add_get('/{filename}', self.handle_static_file)
    
    async def handle_index(self, request):
        """Serve index.html."""
        static_dir = Path(__file__).parent / 'static'
        index_path = static_dir / 'index.html'
        
        if not index_path.exists():
            return web.Response(text="index.html not found", status=404)
        
        return web.FileResponse(index_path)
    
    async def handle_static_file(self, request):
        """Serve static files (CSS, JS)."""
        filename = request.match_info['filename']
        static_dir = Path(__file__).parent / 'static'
        file_path = static_dir / filename
        
        if not file_path.exists():
            return web.Response(text=f"{filename} not found", status=404)
        
        return web.FileResponse(file_path)
    
    async def handle_list_sessions(self, request):
        """API endpoint to list all sessions."""
        try:
            sessions = self.load_all_sessions()
            return web.json_response(sessions)
        except Exception as e:
            logger.error(f"Error loading sessions: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def handle_get_session(self, request):
        """API endpoint to get a specific session."""
        session_id = request.match_info['session_id']
        
        try:
            sessions = self.load_all_sessions()
            session = next((s for s in sessions if s['session_id'] == session_id), None)
            
            if session is None:
                return web.json_response(
                    {"error": "Session not found"},
                    status=404
                )
            
            return web.json_response(session)
        except Exception as e:
            logger.error(f"Error loading session {session_id}: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    def load_all_sessions(self) -> List[Dict[str, Any]]:
        """Load all session JSON files from session directory.
        
        Returns:
            List of session dictionaries
        """
        sessions = []
        
        if not self.session_dir.exists():
            logger.warning(f"Session directory not found: {self.session_dir}")
            return sessions
        
        for session_file in self.session_dir.glob("*.json"):
            try:
                with open(session_file, 'r') as f:
                    session_data = json.load(f)
                    sessions.append(session_data)
            except Exception as e:
                logger.error(f"Error loading {session_file}: {e}")
        
        # Sort by created_at (newest first)
        sessions.sort(key=lambda s: s.get('created_at', ''), reverse=True)
        
        logger.info(f"Loaded {len(sessions)} sessions from {self.session_dir}")
        return sessions
    
    async def start(self):
        """Start the web server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        logger.info(f"🚀 Multi-modal AI Studio WebUI")
        logger.info(f"📡 Server running at http://{self.host}:{self.port}")
        logger.info(f"📂 Session directory: {self.session_dir.absolute()}")
        logger.info(f"Press Ctrl+C to stop")
        
        # Keep running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            await runner.cleanup()


def main():
    """Main entry point for development server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-modal AI Studio WebUI Server")
    parser.add_argument('--host', default='0.0.0.0', help='Host address')
    parser.add_argument('--port', type=int, default=8080, help='Port number')
    parser.add_argument('--session-dir', type=Path, default=Path('mock_sessions'),
                        help='Directory containing session JSON files')
    
    args = parser.parse_args()
    
    server = WebUIServer(
        host=args.host,
        port=args.port,
        session_dir=args.session_dir
    )
    
    asyncio.run(server.start())


if __name__ == '__main__':
    main()
