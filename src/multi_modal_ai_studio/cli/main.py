"""
Main CLI entry point for Multi-modal AI Studio.

This module handles command-line argument parsing and dispatches to
either WebUI mode (default) or headless mode.
"""

import argparse
import asyncio
import sys
import logging
from pathlib import Path


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Multi-modal AI Studio - Voice/Text/Video AI Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # WebUI mode (view sessions / timeline)
  multi-modal-ai-studio --port 8092

  # WebUI with Riva (use --asr-server and --tts-server)
  multi-modal-ai-studio --port 8092 --asr-server localhost:50051 --tts-server localhost:50051 --llm-api-base http://localhost:11434/v1 --llm-model llama3.2:3b

  # WebUI with preset
  multi-modal-ai-studio --preset low-latency

  # Headless mode with config file
  multi-modal-ai-studio --mode headless --config my-preset.yaml

  # Text-only mode
  multi-modal-ai-studio --audio-input none --audio-output none

For more information, visit: https://github.com/yourusername/multi-modal-ai-studio
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["webui", "headless"],
        default="webui",
        help="Operation mode (default: webui)"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Load configuration from YAML/JSON file"
    )

    parser.add_argument(
        "--preset",
        help="Load system preset (default, low-latency, high-accuracy, text-only, openai-realtime)"
    )

    # Server options (WebUI mode)
    parser.add_argument(
        "--port",
        type=int,
        default=8092,
        help="Web server port (default: 8092, avoids conflict with Live RIVA WebUI on 8091)",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Web server host (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--ssl-cert",
        default=None,
        help="Path to SSL certificate (default: <config_dir>/cert.pem, auto-generated if missing)",
    )
    parser.add_argument(
        "--ssl-key",
        default=None,
        help="Path to SSL private key (default: <config_dir>/key.pem, auto-generated if missing)",
    )
    parser.add_argument(
        "--no-ssl",
        action="store_true",
        help="Disable HTTPS (not recommended - getUserMedia/WebRTC require HTTPS)",
    )

    # ASR options
    asr_group = parser.add_argument_group("ASR (Speech Recognition)")
    asr_group.add_argument("--asr-scheme", choices=["riva", "openai-rest", "openai-realtime", "azure", "none"], help="ASR backend")
    asr_group.add_argument("--asr-server", help="ASR server address (for Riva)")
    asr_group.add_argument("--asr-api-key", help="ASR API key (for OpenAI, Azure)")
    asr_group.add_argument("--asr-model", help="ASR model name")
    asr_group.add_argument("--asr-language", help="Language code (e.g., en-US)")
    asr_group.add_argument("--asr-vad-start", type=float, help="VAD start threshold (0.0-1.0)")
    asr_group.add_argument("--asr-vad-stop", type=float, help="VAD stop threshold (0.0-1.0)")

    # LLM options
    llm_group = parser.add_argument_group("LLM (Language Model)")
    llm_group.add_argument("--llm-scheme", choices=["openai", "anthropic", "none"], help="LLM backend")
    llm_group.add_argument("--llm-api-base", help="LLM API base URL")
    llm_group.add_argument("--llm-api-key", help="LLM API key")
    llm_group.add_argument("--llm-model", help="LLM model name")
    llm_group.add_argument(
        "--llm-cheap-model",
        help="LLM model name for non-conversational helper tasks (e.g. session title generation)",
    )
    llm_group.add_argument("--llm-temperature", type=float, help="Sampling temperature (0.0-2.0)")
    llm_group.add_argument("--llm-max-tokens", type=int, help="Maximum tokens to generate")
    llm_group.add_argument("--llm-minimal-output", action="store_true",
                           help="Minimal output only (e.g. single number); no reasoning (for Nemotron-style models)")

    # TTS options
    tts_group = parser.add_argument_group("TTS (Text-to-Speech)")
    tts_group.add_argument("--tts-scheme", choices=["riva", "openai-rest", "openai-realtime", "elevenlabs", "none"], help="TTS backend")
    tts_group.add_argument("--tts-server", help="TTS server address (for Riva)")
    tts_group.add_argument("--tts-api-key", help="TTS API key")
    tts_group.add_argument("--tts-voice", help="TTS voice identifier")

    # Device options
    device_group = parser.add_argument_group("Device Configuration")
    device_group.add_argument("--audio-input", help="Audio input (browser, usb:device, alsa:device, none)")
    device_group.add_argument("--audio-output", help="Audio output (browser, usb:device, alsa:device, none)")
    device_group.add_argument("--video-input", help="Video input (browser, usb:device, none)")

    # App options
    app_group = parser.add_argument_group("Application Settings")
    app_group.add_argument("--barge-in", action="store_true", help="Enable barge-in (interrupt AI)")
    app_group.add_argument("--no-barge-in", action="store_true", help="Disable barge-in")
    app_group.add_argument("--timeline-buffer", type=int, help="Timeline buffer in seconds (15-300)")
    app_group.add_argument("--session-output-dir", help="Directory for session storage")
    app_group.add_argument("--session-dir", type=Path, default=Path("sessions"),
                           help="Directory to load/save session JSON files (default: sessions). Use e.g. --session-dir mock_sessions for sample data.")

    # Logging
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger = logging.getLogger(__name__)
    logger.info("Multi-modal AI Studio starting...")

    if args.mode == "webui":
        import os
        import ssl

        from multi_modal_ai_studio.webui.server import (
            WebUIServer,
            get_app_config_dir,
            generate_self_signed_cert,
        )

        session_dir = Path(args.session_dir).resolve()
        if not session_dir.exists():
            session_dir.mkdir(parents=True, exist_ok=True)

        # SSL: same logic as Live RIVA WebUI (default cert/key in config dir, auto-generate if missing)
        ssl_context = None
        if not args.no_ssl:
            config_dir = get_app_config_dir()
            ssl_cert = args.ssl_cert or str(config_dir / "cert.pem")
            ssl_key = args.ssl_key or str(config_dir / "key.pem")
            if not os.path.exists(ssl_cert) or not os.path.exists(ssl_key):
                if not generate_self_signed_cert(ssl_cert, ssl_key):
                    logger.error("❌ Cannot start without SSL certificates (getUserMedia/WebRTC need HTTPS)")
                    logger.error("   Install openssl or run with --no-ssl (not recommended)")
                    sys.exit(1)
            if os.path.exists(ssl_cert) and os.path.exists(ssl_key):
                ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                ssl_context.load_cert_chain(ssl_cert, ssl_key)
            else:
                logger.error("❌ SSL certificates missing after generation")
                sys.exit(1)
        else:
            logger.warning("⚠️  SSL disabled with --no-ssl (camera/mic may not work without HTTPS)")

        # Load preset or config YAML if specified
        initial_config = None
        if args.preset:
            import yaml
            preset_name = args.preset.removesuffix(".yaml")
            candidates = [
                Path(args.preset),  # exact/absolute path
                Path(__file__).resolve().parent.parent.parent.parent / "presets" / f"{preset_name}.yaml",
                Path("presets") / f"{preset_name}.yaml",
            ]
            preset_path = next((p for p in candidates if p.exists()), None)
            if preset_path:
                with open(preset_path) as f:
                    initial_config = yaml.safe_load(f)
                logger.info(f"Loaded preset '{args.preset}' from {preset_path}")
            else:
                logger.warning(f"Preset '{args.preset}' not found (searched: {[str(c) for c in candidates]})")
        elif args.config:
            import yaml
            if args.config.exists():
                with open(args.config) as f:
                    initial_config = yaml.safe_load(f)
                logger.info(f"Loaded config from {args.config}")
            else:
                logger.warning(f"Config file not found: {args.config}")

        logger.info(f"Starting WebUI server on {args.host}:{args.port}")
        logger.info(f"Session directory: {session_dir}")
        server = WebUIServer(
            host=args.host,
            port=args.port,
            session_dir=session_dir,
            ssl_context=ssl_context,
            initial_config=initial_config,
        )
        try:
            asyncio.run(server.start())
        except KeyboardInterrupt:
            pass
        return 0
    elif args.mode == "headless":
        logger.info("Starting headless mode")
        logger.error("Headless mode not yet implemented")
        sys.exit(1)

    return 0


if __name__ == "__main__":
    sys.exit(main())
