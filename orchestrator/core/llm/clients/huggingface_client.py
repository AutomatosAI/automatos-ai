"""
HuggingFace Provider Implementation
====================================

HuggingFace Inference API provider using the new router.huggingface.co endpoint.
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
    """HuggingFace Inference API provider implementation (OpenAI-compatible endpoint)"""
    
    # New HuggingFace router endpoint (replaces deprecated api-inference.huggingface.co)
    API_BASE_URL = "https://router.huggingface.co/v1"
    
    def _initialize_client(self):
        if requests is None:
            raise ImportError("Requests package not installed. Run: pip install requests")
        
        # HuggingFace must be configured via credential store only
        api_token = self.config.api_key
        
        # BOOTSTRAP STRATEGY: Don't require token at initialization
        if not api_token:
            logger.error(
                "HuggingFace API token not configured. "
                "Add an active credential named 'development_huggingface' (or environment-specific variation)."
            )
            self.api_token = None
        else:
            self.api_token = api_token
            logger.info(f"Initialized HuggingFace client with model: {self.config.model}")
    
    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using HuggingFace router (OpenAI-compatible)"""
        if self.api_token is None:
            raise ValueError(
                "HuggingFace API token not configured. Cannot generate response. "
                "Please add an active credential named 'development_huggingface' (or environment-specific variation) in the credential store."
            )
        
        import httpx
        
        try:
            url = f"{self.API_BASE_URL}/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.model,
                "messages": messages,
                    "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                result = response.json()
            
            # OpenAI-compatible response format
            content = ""
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                content = choice.get("message", {}).get("content", "")
            
            # Extract usage
            usage = None
            if "usage" in result:
                    usage = {
                    "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                    "completion_tokens": result["usage"].get("completion_tokens", 0),
                    "total_tokens": result["usage"].get("total_tokens", 0)
                    }
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=result.get("model", self.config.model),
                provider="huggingface",
                finish_reason=result.get("choices", [{}])[0].get("finish_reason", "stop")
            )
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text if e.response else str(e)
            logger.error(f"HuggingFace API HTTP error: {e.response.status_code} - {error_detail}")
            raise ValueError(f"HuggingFace API error: {error_detail}")
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using HuggingFace router (synchronous)"""
        if self.api_token is None:
            raise ValueError(
                "HuggingFace API token not configured. Cannot generate response. "
                "Please add an active credential named 'development_huggingface' (or environment-specific variation) in the credential store."
            )
        
        try:
            url = f"{self.API_BASE_URL}/chat/completions"
            
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": self.config.model,
                "messages": messages,
                    "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }
            
            response = requests.post(url, headers=headers, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            
            # OpenAI-compatible response format
            content = ""
            if "choices" in result and len(result["choices"]) > 0:
                choice = result["choices"][0]
                content = choice.get("message", {}).get("content", "")
            
            # Extract usage
            usage = None
            if "usage" in result:
                    usage = {
                    "prompt_tokens": result["usage"].get("prompt_tokens", 0),
                    "completion_tokens": result["usage"].get("completion_tokens", 0),
                    "total_tokens": result["usage"].get("total_tokens", 0)
                    }
            
            return LLMResponse(
                content=content,
                usage=usage,
                model=result.get("model", self.config.model),
                provider="huggingface",
                finish_reason=result.get("choices", [{}])[0].get("finish_reason", "stop")
            )
        except requests.HTTPError as e:
            error_detail = e.response.text if e.response else str(e)
            logger.error(f"HuggingFace API HTTP error: {e.response.status_code} - {error_detail}")
            raise ValueError(f"HuggingFace API error: {error_detail}")
        except Exception as e:
            logger.error(f"HuggingFace API error: {e}")
            raise
