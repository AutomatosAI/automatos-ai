"""
HuggingFace Provider Implementation
====================================

HuggingFace Inference API provider.
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional

from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    import requests
except ImportError:
    requests = None

logger = logging.getLogger(__name__)


class HuggingFaceProvider(BaseLLMProvider):
    """HuggingFace Inference API provider implementation"""
    
    def _initialize_client(self):
        if requests is None:
            raise ImportError("Requests package not installed. Run: pip install requests")
        
        # Try multiple sources for API token
        api_token = self.config.api_key or os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_API_TOKEN")
        
        # BOOTSTRAP STRATEGY: Don't require token at initialization
        if not api_token:
            logger.warning(
                "HuggingFace API token not configured. "
                "LLM features will fail until token is added. "
                "Configure 'development_huggingface' credential or set HUGGINGFACE_API_TOKEN env var."
            )
            self.api_token = None
        else:
            self.api_token = api_token
            # HuggingFace API base URL
            self.api_base_url = "https://api-inference.huggingface.co/models"
            logger.info(f"Initialized HuggingFace client with model: {self.config.model}")
    
    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using HuggingFace Inference API"""
        if self.api_token is None:
            raise ValueError(
                "HuggingFace API token not configured. Cannot generate response. "
                "Please configure 'development_huggingface' credential or set HUGGINGFACE_API_TOKEN env var."
            )
        
        import asyncio
        import httpx
        
        try:
            # Convert messages to prompt
            prompt = self._convert_messages_to_prompt(messages)
            
            # HuggingFace Inference API endpoint
            url = f"{self.api_base_url}/{self.config.model}"
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.config.temperature,
                    "max_new_tokens": self.config.max_tokens,
                    "return_full_text": False
                }
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
            
            # HuggingFace returns a list of results
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                content = result.get("generated_text", "")
            else:
                content = str(result)
            
            # Extract usage if available
            usage = None
            if isinstance(result, list) and len(result) > 0:
                usage_info = result[0].get("details", {})
                if usage_info:
                    usage = {
                        "prompt_tokens": usage_info.get("prompt_tokens", 0),
                        "completion_tokens": usage_info.get("completion_tokens", 0),
                        "total_tokens": usage_info.get("total_tokens", 0)
                    }
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=self.config.model,
                provider="huggingface",
                finish_reason="stop"
            )
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using HuggingFace Inference API (synchronous)"""
        if self.api_token is None:
            raise ValueError(
                "HuggingFace API token not configured. Cannot generate response. "
                "Please configure 'development_huggingface' credential or set HUGGINGFACE_API_TOKEN env var."
            )
        
        try:
            # Convert messages to prompt
            prompt = self._convert_messages_to_prompt(messages)
            
            # HuggingFace Inference API endpoint
            url = f"{self.api_base_url}/{self.config.model}"
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "temperature": self.config.temperature,
                    "max_new_tokens": self.config.max_tokens,
                    "return_full_text": False
                }
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            # HuggingFace returns a list of results
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                content = result.get("generated_text", "")
            else:
                content = str(result)
            
            # Extract usage if available
            usage = None
            if isinstance(result, list) and len(result) > 0:
                usage_info = result[0].get("details", {})
                if usage_info:
                    usage = {
                        "prompt_tokens": usage_info.get("prompt_tokens", 0),
                        "completion_tokens": usage_info.get("completion_tokens", 0),
                        "total_tokens": usage_info.get("total_tokens", 0)
                    }
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=self.config.model,
                provider="huggingface"
            )
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to HuggingFace prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n\n".join(prompt_parts)

