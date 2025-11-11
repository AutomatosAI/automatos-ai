"""
AWS Bedrock Provider Implementation
====================================

AWS Bedrock unified provider for multiple LLM models:
- Anthropic Claude 3 (Haiku, Sonnet, Opus)
- Meta Llama 3.1 (8B, 70B, 405B)
- Mistral AI (7B, 8x7B, Large)
- Amazon Titan
- Cohere Command
- AI21 Jurassic

Bedrock acts as a cost-effective gateway to multiple providers.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional

from .base import BaseLLMProvider, LLMConfig, LLMResponse

try:
    import boto3
    from botocore.config import Config
except ImportError:
    boto3 = None

logger = logging.getLogger(__name__)


class BedrockProvider(BaseLLMProvider):
    """AWS Bedrock provider implementation - gateway to multiple LLM models"""
    
    # Model ID mappings for easier use
    MODEL_IDS = {
        # Claude 3 models (Anthropic via Bedrock)
        "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
        "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
        "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
        "claude-3.5-sonnet": "anthropic.claude-3-5-sonnet-20240620-v1:0",
        
        # Llama models (Meta via Bedrock)
        "llama-3.1-405b": "meta.llama3-1-405b-instruct-v1:0",
        "llama-3.1-70b": "meta.llama3-1-70b-instruct-v1:0",
        "llama-3.1-8b": "meta.llama3-1-8b-instruct-v1:0",
        
        # Mistral models
        "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
        "mixtral-8x7b": "mistral.mixtral-8x7b-instruct-v0:1",
        "mistral-large": "mistral.mistral-large-2402-v1:0",
        
        # Amazon Titan
        "titan-text-lite": "amazon.titan-text-lite-v1",
        "titan-text-express": "amazon.titan-text-express-v1",
    }
    
    def _initialize_client(self):
        """Initialize AWS Bedrock client with support for new API Keys or IAM credentials"""
        if boto3 is None:
            raise ImportError("boto3 package not installed. Run: pip install boto3")
        
        # Get region
        aws_region = getattr(self.config, 'region', None) or os.getenv("AWS_REGION", "us-east-1")
        
        # Configure boto3 client with optimized settings
        boto_config = Config(
            region_name=aws_region,
            signature_version='v4',
            retries={'max_attempts': 3, 'mode': 'adaptive'}
        )
        
        # Method 1: NEW Bedrock API Keys (single key authentication)
        bedrock_api_key = (
            getattr(self.config, 'bedrock_api_key', None) or 
            os.getenv("BEDROCK_API_KEY") or
            os.getenv("AWS_BEDROCK_API_KEY")
        )
        
        if bedrock_api_key and bedrock_api_key.startswith('bedrock-api-'):
            # New API Key method - set as environment variable for boto3
            os.environ['AWS_BEDROCK_API_KEY'] = bedrock_api_key
            logger.info("Using new Bedrock API Key authentication")
            
            try:
                self.client = boto3.client(
                    service_name='bedrock-runtime',
                    config=boto_config
                )
            except Exception as e:
                logger.warning(f"Bedrock API Key method failed: {e}. Falling back to default credentials.")
                self.client = None
        
        # Method 2: OLD IAM credentials (Access Key ID + Secret)
        else:
            aws_access_key = self.config.api_key  # api_key field repurposed for access_key_id
            aws_secret_key = getattr(self.config, 'secret_key', None) or os.getenv("AWS_SECRET_ACCESS_KEY")
            
            if aws_access_key and aws_secret_key:
                logger.info("Using IAM Access Key authentication")
                self.client = boto3.client(
                    service_name='bedrock-runtime',
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key,
                    config=boto_config
                )
            else:
                # Try default AWS credential chain (environment, ~/.aws/credentials, IAM role, etc.)
                logger.info("Trying default AWS credential chain")
                try:
                    self.client = boto3.client(
                        service_name='bedrock-runtime',
                        config=boto_config
                    )
                except Exception as e:
                    logger.warning(
                        "AWS Bedrock credentials not configured. "
                        "LLM features will fail until credentials are added. "
                        "Add Bedrock API Key or configure IAM credentials."
                    )
                    self.client = None
        
        # Resolve model ID (handle both short names and full IDs)
        if self.client:
            model_id = self.config.model
            if model_id in self.MODEL_IDS:
                self.model_id = self.MODEL_IDS[model_id]
                logger.info(f"Resolved model '{model_id}' to '{self.model_id}'")
            else:
                self.model_id = model_id  # Assume it's already a full ID
            
            logger.info(f"Initialized AWS Bedrock client with model: {self.model_id}")
    
    def _is_claude_model(self) -> bool:
        """Check if current model is Claude (Anthropic)"""
        return "anthropic.claude" in self.model_id
    
    def _is_llama_model(self) -> bool:
        """Check if current model is Llama (Meta)"""
        return "meta.llama" in self.model_id
    
    def _is_mistral_model(self) -> bool:
        """Check if current model is Mistral"""
        return "mistral." in self.model_id
    
    def _convert_to_claude_format(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> Dict:
        """Convert to Claude/Anthropic format (same as Anthropic API)"""
        system_message = ""
        user_messages = []
        
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                system_message = content if isinstance(content, str) else str(content)
            elif role in ["user", "assistant"]:
                if isinstance(content, str):
                    user_messages.append({"role": role, "content": content})
                elif isinstance(content, list):
                    user_messages.append({"role": role, "content": content})
                else:
                    user_messages.append({"role": role, "content": str(content)})
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "messages": user_messages
        }
        
        if system_message:
            body["system"] = system_message
        
        # Add tools if provided (Claude format)
        if tools:
            body["tools"] = self._convert_tools_to_anthropic_format(tools)
        
        return body
    
    def _convert_to_llama_format(self, messages: List[Dict[str, str]]) -> Dict:
        """Convert to Llama format"""
        # Llama uses a prompt-based format
        prompt = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "user":
                prompt += f"<|start_header_id|>user<|end_header_id|>\n\n{content}<|eot_id|>"
            elif role == "assistant":
                prompt += f"<|start_header_id|>assistant<|end_header_id|>\n\n{content}<|eot_id|>"
        
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        
        return {
            "prompt": prompt,
            "max_gen_len": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": 0.9
        }
    
    def _convert_to_mistral_format(self, messages: List[Dict[str, str]]) -> Dict:
        """Convert to Mistral format"""
        # Mistral uses a simpler prompt format
        prompt = ""
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                prompt += f"<s>[INST] {content} [/INST]</s>\n"
            elif role == "user":
                prompt += f"<s>[INST] {content} [/INST]"
            elif role == "assistant":
                prompt += f" {content}</s>\n"
        
        return {
            "prompt": prompt,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "top_p": 0.9,
            "top_k": 50
        }
    
    def _convert_tools_to_anthropic_format(self, tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI-style tools to Anthropic/Claude format"""
        anthropic_tools = []
        for tool in tools:
            if isinstance(tool, dict):
                if tool.get("type") == "function" and "function" in tool:
                    func_def = tool["function"]
                    anthropic_tools.append({
                        "name": func_def.get("name"),
                        "description": func_def.get("description", ""),
                        "input_schema": func_def.get("parameters", {})
                    })
                elif "name" in tool:
                    anthropic_tools.append({
                        "name": tool.get("name"),
                        "description": tool.get("description", ""),
                        "input_schema": tool.get("parameters") or tool.get("input_schema", {})
                    })
        return anthropic_tools
    
    async def generate_response(self, messages: List[Dict[str, str]], tools: List[Dict] = None) -> LLMResponse:
        """Generate response using AWS Bedrock without blocking the event loop"""
        if self.client is None:
            raise ValueError(
                "AWS Bedrock credentials not configured. Cannot generate response. "
                "Please configure 'development_aws' credential or set AWS env vars."
            )
        
        import asyncio
        loop = asyncio.get_running_loop()
        
        try:
            # Convert to appropriate format based on model
            if self._is_claude_model():
                body = self._convert_to_claude_format(messages, tools)
            elif self._is_llama_model():
                body = self._convert_to_llama_format(messages)
            elif self._is_mistral_model():
                body = self._convert_to_mistral_format(messages)
            else:
                # Default to Claude format for unknown models
                body = self._convert_to_claude_format(messages, tools)
            
            def _call():
                response = self.client.invoke_model(
                    modelId=self.model_id,
                    contentType='application/json',
                    accept='application/json',
                    body=json.dumps(body)
                )
                return json.loads(response['body'].read())
            
            response_body = await loop.run_in_executor(None, _call)
            
            # Parse response based on model type
            if self._is_claude_model():
                return self._parse_claude_response(response_body)
            elif self._is_llama_model():
                return self._parse_llama_response(response_body)
            elif self._is_mistral_model():
                return self._parse_mistral_response(response_body)
            else:
                # Default parser
                return self._parse_claude_response(response_body)
                
        except Exception as e:
            logger.error(f"AWS Bedrock API error: {e}")
            raise
    
    def _parse_claude_response(self, response_body: Dict) -> LLMResponse:
        """Parse Claude/Anthropic model response"""
        content = ""
        tool_calls = None
        
        for block in response_body.get("content", []):
            if block.get("type") == "text":
                content += block.get("text", "")
            elif block.get("type") == "tool_use":
                if tool_calls is None:
                    tool_calls = []
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {}))
                    }
                })
        
        usage = response_body.get("usage", {})
        return LLMResponse(
            content=content or "",
            usage={
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            },
            model=self.model_id,
            provider="aws_bedrock",
            tool_calls=tool_calls,
            finish_reason=response_body.get("stop_reason")
        )
    
    def _parse_llama_response(self, response_body: Dict) -> LLMResponse:
        """Parse Llama model response"""
        generation = response_body.get("generation", "")
        prompt_token_count = response_body.get("prompt_token_count", 0)
        generation_token_count = response_body.get("generation_token_count", 0)
        
        return LLMResponse(
            content=generation,
            usage={
                "prompt_tokens": prompt_token_count,
                "completion_tokens": generation_token_count,
                "total_tokens": prompt_token_count + generation_token_count
            },
            model=self.model_id,
            provider="aws_bedrock",
            tool_calls=None,  # Llama doesn't have native function calling
            finish_reason=response_body.get("stop_reason", "end_turn")
        )
    
    def _parse_mistral_response(self, response_body: Dict) -> LLMResponse:
        """Parse Mistral model response"""
        outputs = response_body.get("outputs", [])
        content = outputs[0].get("text", "") if outputs else ""
        
        return LLMResponse(
            content=content,
            usage={
                "prompt_tokens": 0,  # Mistral doesn't always return token counts
                "completion_tokens": 0,
                "total_tokens": 0
            },
            model=self.model_id,
            provider="aws_bedrock",
            tool_calls=None,  # Mistral doesn't have native function calling
            finish_reason="stop"
        )
    
    def generate_response_sync(self, messages: List[Dict[str, str]]) -> LLMResponse:
        """Generate response using AWS Bedrock (synchronous)"""
        if self.client is None:
            raise ValueError(
                "AWS Bedrock credentials not configured. Cannot generate response. "
                "Please configure 'development_aws' credential or set AWS env vars."
            )
        
        try:
            # Convert to appropriate format
            if self._is_claude_model():
                body = self._convert_to_claude_format(messages)
            elif self._is_llama_model():
                body = self._convert_to_llama_format(messages)
            elif self._is_mistral_model():
                body = self._convert_to_mistral_format(messages)
            else:
                body = self._convert_to_claude_format(messages)
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                contentType='application/json',
                accept='application/json',
                body=json.dumps(body)
            )
            
            response_body = json.loads(response['body'].read())
            
            # Parse based on model type
            if self._is_claude_model():
                return self._parse_claude_response(response_body)
            elif self._is_llama_model():
                return self._parse_llama_response(response_body)
            elif self._is_mistral_model():
                return self._parse_mistral_response(response_body)
            else:
                return self._parse_claude_response(response_body)
                
        except Exception as e:
            logger.error(f"AWS Bedrock API error: {e}")
            raise

