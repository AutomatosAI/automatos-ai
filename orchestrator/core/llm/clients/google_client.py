"""
Google Gemini Provider Implementation
=====================================

Google Gemini models provider.
"""

import logging
from typing import Dict, Any, List, Optional

from config import config
from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    import google.generativeai as genai
except ImportError:
    genai = None

logger = logging.getLogger(__name__)


class GoogleProvider(BaseLLMProvider):
    """Google Gemini provider implementation"""
    
    def _initialize_client(self):
        if genai is None:
            raise ImportError("Google Generative AI package not installed. Run: pip install google-generativeai")
        
        # Try multiple sources for API key
        api_key = self.config.api_key or config.GOOGLE_API_KEY
        
        # BOOTSTRAP STRATEGY: Don't require key at initialization
        if not api_key:
            logger.warning(
                "Google API key not configured. "
                "LLM features will fail until key is added. "
                "Configure 'development_google' credential or set GOOGLE_API_KEY env var."
            )
            self.client = None
        else:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.config.model)
            logger.info(f"Initialized Google Gemini client with model: {self.config.model}")
    
    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using Google Gemini API"""
        if self.client is None:
            raise ValueError(
                "Google API key not configured. Cannot generate response. "
                "Please configure 'development_google' credential or set GOOGLE_API_KEY env var."
            )
        
        import asyncio
        loop = asyncio.get_running_loop()
        
        try:
            # Convert messages to Gemini format
            prompt = self._convert_messages_to_prompt(messages)
            
            def _call():
                generation_config = {
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_tokens,
                }
                response = self.client.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                return response
            
            response = await loop.run_in_executor(None, _call)
            
            content = response.text if response.text else ""
            
            # Extract usage info if available
            usage = None
            if hasattr(response, 'usage_metadata'):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=self.config.model,
                provider="google",
                finish_reason=getattr(response, 'candidates', [{}])[0].get('finish_reason', 'stop') if hasattr(response, 'candidates') else None
            )
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            raise
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using Google Gemini API (synchronous)"""
        if self.client is None:
            raise ValueError(
                "Google API key not configured. Cannot generate response. "
                "Please configure 'development_google' credential or set GOOGLE_API_KEY env var."
            )
        
        try:
            # Convert messages to Gemini format
            prompt = self._convert_messages_to_prompt(messages)
            
            generation_config = {
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_tokens,
            }
            
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            content = response.text if response.text else ""
            
            # Extract usage info if available
            usage = None
            if hasattr(response, 'usage_metadata'):
                usage = {
                    "prompt_tokens": response.usage_metadata.prompt_token_count,
                    "completion_tokens": response.usage_metadata.candidates_token_count,
                    "total_tokens": response.usage_metadata.total_token_count
                }
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=self.config.model,
                provider="google"
            )
        except Exception as e:
            logger.error(f"Google Gemini API error: {e}")
            raise
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Gemini prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                # Gemini uses system instruction differently
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)

