"""
OpenAI-compatible LLM Backend

Supports OpenAI, Ollama, vLLM, SGLang, and other OpenAI-compatible APIs.
Provides streaming text generation for conversational AI.
Adapted from live-riva-webui with timeline event support.
"""

import json
import logging
import re
from typing import AsyncIterator, Optional, List, Dict

import aiohttp

from multi_modal_ai_studio.backends.base import (
    LLMBackend,
    LLMToken,
    ConnectionError,
    ConfigError,
)
from multi_modal_ai_studio.config.schema import LLMConfig

logger = logging.getLogger(__name__)


def strip_markdown(text: str, preserve_spaces: bool = False) -> str:
    """Remove markdown formatting from text.
    
    Args:
        text: Text to process
        preserve_spaces: If True, preserve leading/trailing spaces (for streaming chunks)
    
    Returns:
        Text with markdown removed
    """
    # Remove code blocks
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'`([^`]+)`', r'\1', text)

    # Remove headers
    text = re.sub(r'#{1,6}\s+', '', text)

    # Remove bold/italic
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    text = re.sub(r'\*([^*]+)\*', r'\1', text)
    text = re.sub(r'__([^_]+)__', r'\1', text)
    text = re.sub(r'_([^_]+)_', r'\1', text)

    # Remove links [text](url) -> text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # Remove list markers
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Clean up extra whitespace (but preserve spaces in streaming mode)
    if not preserve_spaces:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()

    return text


class OpenAILLMBackend(LLMBackend):
    """OpenAI-compatible LLM backend with streaming support.
    
    Supports:
    - OpenAI API
    - Ollama
    - vLLM
    - SGLang
    - Any OpenAI-compatible endpoint
    
    Features:
    - Streaming token generation
    - Conversation history management
    - Automatic model detection (for Ollama)
    - Markdown stripping for voice output
    """
    
    def __init__(self, config: LLMConfig):
        """Initialize OpenAI-compatible LLM backend.
        
        Args:
            config: LLMConfig instance
        
        Raises:
            ConfigError: If configuration is invalid
        """
        super().__init__(config)
        
        # Validate configuration
        if config.scheme not in ["openai", "anthropic"]:
            raise ConfigError(f"Unsupported LLM scheme: {config.scheme}")
        
        if not config.api_base:
            raise ConfigError("LLM API base URL is required")
        
        self.api_base = config.api_base.rstrip("/")
        self.api_key = config.api_key or "EMPTY"
        
        self.logger.info(f"Initialized OpenAI-compatible LLM: {config.model} @ {self.api_base}")
    
    async def list_available_models(self) -> List[str]:
        """List available models from the LLM API.
        
        Attempts to detect models from Ollama's native API or OpenAI endpoint.
        
        Returns:
            List of model names, or empty list if detection fails
        """
        try:
            # Try Ollama's native API first (/api/tags)
            ollama_base = self.api_base.replace("/v1", "")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{ollama_base}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        model_names = [m["name"] for m in models]
                        self.logger.info(f"Detected {len(model_names)} Ollama models")
                        return model_names
        
        except Exception as e:
            self.logger.debug(f"Ollama model detection failed: {e}")
        
        # Try OpenAI /v1/models endpoint
        try:
            async with aiohttp.ClientSession() as session:
                headers = {"Authorization": f"Bearer {self.api_key}"}
                async with session.get(
                    f"{self.api_base}/models",
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        model_names = [m["id"] for m in data.get("data", [])]
                        self.logger.info(f"Detected {len(model_names)} OpenAI models")
                        return model_names
        
        except Exception as e:
            self.logger.debug(f"OpenAI model detection failed: {e}")
        
        self.logger.warning("Failed to detect models from API")
        return []
    
    async def generate_stream(
        self,
        prompt: str,
        history: Optional[List[Dict[str, str]]] = None,
        system_prompt: Optional[str] = None
    ) -> AsyncIterator[LLMToken]:
        """Generate response tokens in streaming fashion.
        
        Args:
            prompt: User prompt/message
            history: Conversation history in format:
                     [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            system_prompt: Optional system prompt (overrides config if provided)
        
        Yields:
            LLMToken: Generated tokens
        
        Raises:
            ConnectionError: If unable to connect to LLM service
        """
        # Build messages array
        messages = []
        
        # Add system prompt
        sys_prompt = system_prompt or self.config.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        # Add history
        if history:
            messages.extend(history)
        
        # Add current prompt
        messages.append({"role": "user", "content": prompt})
        
        # Prepare API request
        url = f"{self.api_base}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p,
            "frequency_penalty": self.config.frequency_penalty,
            "presence_penalty": self.config.presence_penalty,
            "stream": True,
        }
        
        self.logger.debug(f"LLM request: {prompt[:50]}...")
        
        full_response = ""
        token_count = 0
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=payload) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        self.logger.error(f"LLM API error: {resp.status} - {error_text}")
                        raise ConnectionError(f"LLM API error: {resp.status}")
                    
                    # Stream response chunks
                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        
                        # Skip empty lines and non-data lines
                        if not line or not line.startswith("data: "):
                            continue
                        
                        # Remove "data: " prefix
                        line = line[6:]
                        
                        # Check for end of stream
                        if line == "[DONE]":
                            # Yield final token
                            yield LLMToken(
                                token="",
                                is_final=True,
                                metadata={
                                    "token_count": token_count,
                                    "full_response": full_response,
                                }
                            )
                            break
                        
                        try:
                            # Parse JSON chunk
                            chunk = json.loads(line)
                            
                            # Extract content from chunk
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                
                                if content:
                                    # Strip markdown for voice output
                                    content = strip_markdown(content, preserve_spaces=True)
                                    
                                    full_response += content
                                    token_count += 1
                                    
                                    # Yield token
                                    yield LLMToken(
                                        token=content,
                                        is_final=False,
                                        metadata={
                                            "token_count": token_count,
                                            "partial_response": full_response,
                                        }
                                    )
                        
                        except json.JSONDecodeError as e:
                            self.logger.warning(f"Failed to parse JSON: {e}")
                            continue
        
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP error during LLM streaming: {e}")
            raise ConnectionError(f"Failed to connect to LLM API: {e}")
        
        except Exception as e:
            self.logger.error(f"Unexpected error during LLM streaming: {e}", exc_info=True)
            raise
        
        finally:
            if full_response:
                self.logger.info(f"LLM response: {len(full_response)} chars, {token_count} tokens")
