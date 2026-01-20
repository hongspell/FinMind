"""
Multi-Model LLM Gateway
========================
Unified interface for multiple LLM providers with intelligent routing,
automatic fallback, rate limiting, and cost tracking.

Supported Providers:
- OpenAI (GPT-4o, GPT-4o-mini, o1, o1-mini)
- Anthropic (Claude Sonnet, Claude Opus, Claude Haiku)
- Google (Gemini 1.5 Pro, Gemini 1.5 Flash)
- DeepSeek (DeepSeek-V3, DeepSeek-R1)
- Ollama (Local models)
"""

import asyncio
import time
import json
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, Any, Optional, List, AsyncGenerator, Callable, Union
import logging
import os
from collections import defaultdict

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """Model capabilities for routing decisions"""
    FAST = "fast"                    # Quick responses
    REASONING = "reasoning"          # Complex reasoning
    CODING = "coding"                # Code generation
    ANALYSIS = "analysis"            # Data analysis
    CREATIVE = "creative"            # Creative writing
    STRUCTURED = "structured"        # Structured output (JSON)
    VISION = "vision"                # Image understanding
    LONG_CONTEXT = "long_context"    # Large context window
    COST_EFFECTIVE = "cost_effective"  # Budget friendly


@dataclass
class ModelInfo:
    """Information about a specific model"""
    provider: str
    model_id: str
    display_name: str
    context_window: int
    max_output_tokens: int
    cost_per_1k_input: float     # USD per 1K input tokens
    cost_per_1k_output: float    # USD per 1K output tokens
    capabilities: List[ModelCapability]
    supports_streaming: bool = True
    supports_tools: bool = True
    supports_json_mode: bool = True
    rate_limit_rpm: int = 60     # Requests per minute
    rate_limit_tpm: int = 100000  # Tokens per minute


# Model Registry - Comprehensive list of available models
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    # OpenAI Models
    "gpt-4o": ModelInfo(
        provider="openai",
        model_id="gpt-4o",
        display_name="GPT-4o",
        context_window=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.0025,
        cost_per_1k_output=0.01,
        capabilities=[
            ModelCapability.REASONING, ModelCapability.ANALYSIS,
            ModelCapability.CODING, ModelCapability.VISION,
            ModelCapability.STRUCTURED
        ]
    ),
    "gpt-4o-mini": ModelInfo(
        provider="openai",
        model_id="gpt-4o-mini",
        display_name="GPT-4o Mini",
        context_window=128000,
        max_output_tokens=16384,
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        capabilities=[
            ModelCapability.FAST, ModelCapability.COST_EFFECTIVE,
            ModelCapability.CODING, ModelCapability.STRUCTURED
        ]
    ),
    "o1": ModelInfo(
        provider="openai",
        model_id="o1",
        display_name="OpenAI o1",
        context_window=200000,
        max_output_tokens=100000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.06,
        capabilities=[
            ModelCapability.REASONING, ModelCapability.ANALYSIS,
            ModelCapability.CODING
        ],
        supports_streaming=False,
        supports_tools=False
    ),
    "o1-mini": ModelInfo(
        provider="openai",
        model_id="o1-mini",
        display_name="OpenAI o1-mini",
        context_window=128000,
        max_output_tokens=65536,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.012,
        capabilities=[
            ModelCapability.REASONING, ModelCapability.CODING,
            ModelCapability.COST_EFFECTIVE
        ],
        supports_streaming=False,
        supports_tools=False
    ),
    
    # Anthropic Models
    "claude-sonnet": ModelInfo(
        provider="anthropic",
        model_id="claude-sonnet-4-20250514",
        display_name="Claude Sonnet 4",
        context_window=200000,
        max_output_tokens=64000,
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        capabilities=[
            ModelCapability.REASONING, ModelCapability.ANALYSIS,
            ModelCapability.CODING, ModelCapability.CREATIVE,
            ModelCapability.STRUCTURED, ModelCapability.LONG_CONTEXT
        ]
    ),
    "claude-opus": ModelInfo(
        provider="anthropic",
        model_id="claude-opus-4-20250514",
        display_name="Claude Opus 4",
        context_window=200000,
        max_output_tokens=32000,
        cost_per_1k_input=0.015,
        cost_per_1k_output=0.075,
        capabilities=[
            ModelCapability.REASONING, ModelCapability.ANALYSIS,
            ModelCapability.CODING, ModelCapability.CREATIVE,
            ModelCapability.LONG_CONTEXT
        ]
    ),
    "claude-haiku": ModelInfo(
        provider="anthropic",
        model_id="claude-haiku-4-20250514",
        display_name="Claude Haiku 4",
        context_window=200000,
        max_output_tokens=8192,
        cost_per_1k_input=0.0008,
        cost_per_1k_output=0.004,
        capabilities=[
            ModelCapability.FAST, ModelCapability.COST_EFFECTIVE,
            ModelCapability.STRUCTURED
        ]
    ),
    
    # Google Models
    "gemini-pro": ModelInfo(
        provider="google",
        model_id="gemini-1.5-pro",
        display_name="Gemini 1.5 Pro",
        context_window=2000000,
        max_output_tokens=8192,
        cost_per_1k_input=0.00125,
        cost_per_1k_output=0.005,
        capabilities=[
            ModelCapability.REASONING, ModelCapability.ANALYSIS,
            ModelCapability.LONG_CONTEXT, ModelCapability.VISION,
            ModelCapability.COST_EFFECTIVE
        ]
    ),
    "gemini-flash": ModelInfo(
        provider="google",
        model_id="gemini-1.5-flash",
        display_name="Gemini 1.5 Flash",
        context_window=1000000,
        max_output_tokens=8192,
        cost_per_1k_input=0.000075,
        cost_per_1k_output=0.0003,
        capabilities=[
            ModelCapability.FAST, ModelCapability.COST_EFFECTIVE,
            ModelCapability.LONG_CONTEXT
        ]
    ),
    "gemini-2-flash": ModelInfo(
        provider="google",
        model_id="gemini-2.0-flash",
        display_name="Gemini 2.0 Flash",
        context_window=1000000,
        max_output_tokens=8192,
        cost_per_1k_input=0.0001,
        cost_per_1k_output=0.0004,
        capabilities=[
            ModelCapability.FAST, ModelCapability.REASONING,
            ModelCapability.COST_EFFECTIVE, ModelCapability.LONG_CONTEXT
        ]
    ),
    
    # DeepSeek Models
    "deepseek-v3": ModelInfo(
        provider="deepseek",
        model_id="deepseek-chat",
        display_name="DeepSeek V3",
        context_window=64000,
        max_output_tokens=8192,
        cost_per_1k_input=0.00014,
        cost_per_1k_output=0.00028,
        capabilities=[
            ModelCapability.REASONING, ModelCapability.CODING,
            ModelCapability.COST_EFFECTIVE, ModelCapability.ANALYSIS
        ]
    ),
    "deepseek-r1": ModelInfo(
        provider="deepseek",
        model_id="deepseek-reasoner",
        display_name="DeepSeek R1",
        context_window=64000,
        max_output_tokens=8192,
        cost_per_1k_input=0.00055,
        cost_per_1k_output=0.00219,
        capabilities=[
            ModelCapability.REASONING, ModelCapability.ANALYSIS,
            ModelCapability.CODING
        ],
        supports_streaming=True
    ),
    
    # Ollama (Local) - Costs are $0 but we track for consistency
    "ollama-llama3": ModelInfo(
        provider="ollama",
        model_id="llama3.3:70b",
        display_name="Llama 3.3 70B (Local)",
        context_window=128000,
        max_output_tokens=8192,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        capabilities=[
            ModelCapability.REASONING, ModelCapability.CODING,
            ModelCapability.COST_EFFECTIVE
        ],
        rate_limit_rpm=1000,
        rate_limit_tpm=1000000
    ),
    "ollama-qwen": ModelInfo(
        provider="ollama",
        model_id="qwen2.5:72b",
        display_name="Qwen 2.5 72B (Local)",
        context_window=32000,
        max_output_tokens=8192,
        cost_per_1k_input=0.0,
        cost_per_1k_output=0.0,
        capabilities=[
            ModelCapability.REASONING, ModelCapability.CODING,
            ModelCapability.COST_EFFECTIVE
        ],
        rate_limit_rpm=1000,
        rate_limit_tpm=1000000
    ),
}


@dataclass
class LLMResponse:
    """Standardized response from any LLM"""
    content: str
    model: str
    provider: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost: float
    latency_ms: float
    finish_reason: str
    raw_response: Optional[Dict] = None
    tool_calls: Optional[List[Dict]] = None
    thinking: Optional[str] = None  # For reasoning models


@dataclass
class RateLimitState:
    """Track rate limit state for a model"""
    requests_this_minute: int = 0
    tokens_this_minute: int = 0
    minute_start: float = field(default_factory=time.time)
    
    def reset_if_needed(self):
        now = time.time()
        if now - self.minute_start >= 60:
            self.requests_this_minute = 0
            self.tokens_this_minute = 0
            self.minute_start = now


class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._client = None
    
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate a completion"""
        pass
    
    @abstractmethod
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a completion"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI API provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"))
        self._init_client()
    
    def _init_client(self):
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(api_key=self.api_key)
        except ImportError:
            logger.warning("OpenAI package not installed")
            self._client = None
    
    def is_available(self) -> bool:
        return self._client is not None and self.api_key is not None
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        
        # Build request
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            request["max_tokens"] = max_tokens
        if tools:
            request["tools"] = tools
        if tool_choice:
            request["tool_choice"] = tool_choice
        if response_format:
            request["response_format"] = response_format
        
        response = await self._client.chat.completions.create(**request)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate cost
        model_info = MODEL_REGISTRY.get(model) or MODEL_REGISTRY.get("gpt-4o")
        input_cost = (response.usage.prompt_tokens / 1000) * model_info.cost_per_1k_input
        output_cost = (response.usage.completion_tokens / 1000) * model_info.cost_per_1k_output
        
        # Extract tool calls if present
        tool_calls = None
        if response.choices[0].message.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                }
                for tc in response.choices[0].message.tool_calls
            ]
        
        return LLMResponse(
            content=response.choices[0].message.content or "",
            model=model,
            provider="openai",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cost=input_cost + output_cost,
            latency_ms=latency_ms,
            finish_reason=response.choices[0].finish_reason,
            raw_response=response.model_dump(),
            tool_calls=tool_calls
        )
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        
        if max_tokens:
            request["max_tokens"] = max_tokens
        
        stream = await self._client.chat.completions.create(**request)
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) API provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"))
        self._init_client()
    
    def _init_client(self):
        try:
            from anthropic import AsyncAnthropic
            self._client = AsyncAnthropic(api_key=self.api_key)
        except ImportError:
            logger.warning("Anthropic package not installed")
            self._client = None
    
    def is_available(self) -> bool:
        return self._client is not None and self.api_key is not None
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        
        # Extract system message if present
        system = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)
        
        # Build request
        request = {
            "model": model,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        if system:
            request["system"] = system
        if tools:
            # Convert OpenAI tool format to Anthropic
            request["tools"] = self._convert_tools(tools)
        
        response = await self._client.messages.create(**request)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate cost
        model_info = MODEL_REGISTRY.get("claude-sonnet")  # Default
        for key, info in MODEL_REGISTRY.items():
            if info.model_id == model:
                model_info = info
                break
        
        input_cost = (response.usage.input_tokens / 1000) * model_info.cost_per_1k_input
        output_cost = (response.usage.output_tokens / 1000) * model_info.cost_per_1k_output
        
        # Extract content
        content = ""
        tool_calls = []
        for block in response.content:
            if block.type == "text":
                content += block.text
            elif block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "type": "function",
                    "function": {
                        "name": block.name,
                        "arguments": json.dumps(block.input)
                    }
                })
        
        return LLMResponse(
            content=content,
            model=model,
            provider="anthropic",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            cost=input_cost + output_cost,
            latency_ms=latency_ms,
            finish_reason=response.stop_reason,
            raw_response={"id": response.id, "model": response.model},
            tool_calls=tool_calls if tool_calls else None
        )
    
    def _convert_tools(self, openai_tools: List[Dict]) -> List[Dict]:
        """Convert OpenAI tool format to Anthropic format"""
        anthropic_tools = []
        for tool in openai_tools:
            if tool["type"] == "function":
                anthropic_tools.append({
                    "name": tool["function"]["name"],
                    "description": tool["function"].get("description", ""),
                    "input_schema": tool["function"].get("parameters", {})
                })
        return anthropic_tools
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        # Extract system message
        system = None
        user_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                user_messages.append(msg)
        
        request = {
            "model": model,
            "messages": user_messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 4096,
        }
        
        if system:
            request["system"] = system
        
        async with self._client.messages.stream(**request) as stream:
            async for text in stream.text_stream:
                yield text


class GoogleProvider(LLMProvider):
    """Google (Gemini) API provider"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("GOOGLE_API_KEY"))
        self._init_client()
    
    def _init_client(self):
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
            self._genai = genai
        except ImportError:
            logger.warning("Google generativeai package not installed")
            self._genai = None
    
    def is_available(self) -> bool:
        return self._genai is not None and self.api_key is not None
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        
        # Convert messages to Gemini format
        gemini_messages = []
        system_instruction = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})
        
        # Create model
        generation_config = {
            "temperature": temperature,
        }
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        model_obj = self._genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        
        # Generate
        response = await asyncio.to_thread(
            model_obj.generate_content,
            gemini_messages
        )
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Get token counts
        input_tokens = response.usage_metadata.prompt_token_count
        output_tokens = response.usage_metadata.candidates_token_count
        
        # Calculate cost
        model_info = MODEL_REGISTRY.get("gemini-pro")
        input_cost = (input_tokens / 1000) * model_info.cost_per_1k_input
        output_cost = (output_tokens / 1000) * model_info.cost_per_1k_output
        
        return LLMResponse(
            content=response.text,
            model=model,
            provider="google",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost=input_cost + output_cost,
            latency_ms=latency_ms,
            finish_reason=response.candidates[0].finish_reason.name if response.candidates else "unknown"
        )
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        # Similar setup as complete
        gemini_messages = []
        system_instruction = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})
        
        generation_config = {"temperature": temperature}
        if max_tokens:
            generation_config["max_output_tokens"] = max_tokens
        
        model_obj = self._genai.GenerativeModel(
            model_name=model,
            generation_config=generation_config,
            system_instruction=system_instruction
        )
        
        response = model_obj.generate_content(gemini_messages, stream=True)
        
        for chunk in response:
            if chunk.text:
                yield chunk.text


class DeepSeekProvider(LLMProvider):
    """DeepSeek API provider (OpenAI-compatible)"""
    
    def __init__(self, api_key: Optional[str] = None):
        super().__init__(api_key or os.getenv("DEEPSEEK_API_KEY"))
        self._init_client()
    
    def _init_client(self):
        try:
            from openai import AsyncOpenAI
            self._client = AsyncOpenAI(
                api_key=self.api_key,
                base_url="https://api.deepseek.com/v1"
            )
        except ImportError:
            logger.warning("OpenAI package not installed (needed for DeepSeek)")
            self._client = None
    
    def is_available(self) -> bool:
        return self._client is not None and self.api_key is not None
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        
        if max_tokens:
            request["max_tokens"] = max_tokens
        if tools:
            request["tools"] = tools
        
        response = await self._client.chat.completions.create(**request)
        
        latency_ms = (time.time() - start_time) * 1000
        
        # Calculate cost
        model_key = "deepseek-r1" if "reasoner" in model else "deepseek-v3"
        model_info = MODEL_REGISTRY.get(model_key)
        input_cost = (response.usage.prompt_tokens / 1000) * model_info.cost_per_1k_input
        output_cost = (response.usage.completion_tokens / 1000) * model_info.cost_per_1k_output
        
        # Extract thinking for R1 model
        thinking = None
        content = response.choices[0].message.content or ""
        if "reasoner" in model and hasattr(response.choices[0].message, 'reasoning_content'):
            thinking = response.choices[0].message.reasoning_content
        
        return LLMResponse(
            content=content,
            model=model,
            provider="deepseek",
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
            cost=input_cost + output_cost,
            latency_ms=latency_ms,
            finish_reason=response.choices[0].finish_reason,
            thinking=thinking
        )
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        
        if max_tokens:
            request["max_tokens"] = max_tokens
        
        stream = await self._client.chat.completions.create(**request)
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class OllamaProvider(LLMProvider):
    """Ollama (local) provider"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        super().__init__()
        self.base_url = base_url
        self._init_client()
    
    def _init_client(self):
        try:
            import httpx
            self._client = httpx.AsyncClient(base_url=self.base_url, timeout=120.0)
        except ImportError:
            logger.warning("httpx package not installed")
            self._client = None
    
    def is_available(self) -> bool:
        return self._client is not None
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        start_time = time.time()
        
        request = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            request["options"]["num_predict"] = max_tokens
        
        response = await self._client.post("/api/chat", json=request)
        data = response.json()
        
        latency_ms = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=data.get("message", {}).get("content", ""),
            model=model,
            provider="ollama",
            input_tokens=data.get("prompt_eval_count", 0),
            output_tokens=data.get("eval_count", 0),
            total_tokens=data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
            cost=0.0,  # Local models are free
            latency_ms=latency_ms,
            finish_reason=data.get("done_reason", "stop")
        )
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        request = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
            }
        }
        
        if max_tokens:
            request["options"]["num_predict"] = max_tokens
        
        async with self._client.stream("POST", "/api/chat", json=request) as response:
            async for line in response.aiter_lines():
                if line:
                    data = json.loads(line)
                    if "message" in data and "content" in data["message"]:
                        yield data["message"]["content"]


class LLMGateway:
    """
    Unified LLM Gateway with intelligent routing, fallback, and cost tracking.
    
    Features:
    - Multi-provider support
    - Automatic fallback on failures
    - Rate limiting
    - Cost tracking
    - Model aliasing
    - Task-based routing
    - Caching (optional)
    
    Usage:
        gateway = LLMGateway()
        
        # Simple call
        response = await gateway.complete(
            messages=[{"role": "user", "content": "Hello"}],
            model="claude-sonnet"
        )
        
        # Task-based routing
        response = await gateway.complete_for_task(
            messages=[...],
            task="analysis",  # Will pick best model for analysis
            budget="low"      # Cost constraint
        )
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        enable_fallback: bool = True,
        enable_caching: bool = False,
        max_retries: int = 3
    ):
        self.config = config or {}
        self.enable_fallback = enable_fallback
        self.enable_caching = enable_caching
        self.max_retries = max_retries
        
        # Initialize providers
        self._providers: Dict[str, LLMProvider] = {}
        self._init_providers()
        
        # Model aliases for convenience
        self._aliases: Dict[str, str] = {
            "gpt4": "gpt-4o",
            "gpt4-mini": "gpt-4o-mini",
            "claude": "claude-sonnet",
            "gemini": "gemini-pro",
            "deepseek": "deepseek-v3",
            "local": "ollama-llama3",
        }
        
        # Fallback chains
        self._fallback_chains: Dict[str, List[str]] = {
            "openai": ["claude-sonnet", "gemini-pro", "deepseek-v3"],
            "anthropic": ["gpt-4o", "gemini-pro", "deepseek-v3"],
            "google": ["claude-sonnet", "gpt-4o", "deepseek-v3"],
            "deepseek": ["gpt-4o-mini", "gemini-flash", "ollama-llama3"],
            "ollama": ["deepseek-v3", "gpt-4o-mini", "gemini-flash"],
        }
        
        # Task-based model recommendations
        self._task_models: Dict[str, Dict[str, str]] = {
            "analysis": {
                "high": "claude-sonnet",
                "medium": "gpt-4o",
                "low": "deepseek-v3"
            },
            "reasoning": {
                "high": "o1",
                "medium": "deepseek-r1",
                "low": "o1-mini"
            },
            "coding": {
                "high": "claude-sonnet",
                "medium": "deepseek-v3",
                "low": "gpt-4o-mini"
            },
            "quick": {
                "high": "gpt-4o-mini",
                "medium": "gemini-flash",
                "low": "ollama-llama3"
            },
            "long_context": {
                "high": "gemini-pro",
                "medium": "claude-sonnet",
                "low": "gemini-flash"
            }
        }
        
        # Rate limiting state
        self._rate_limits: Dict[str, RateLimitState] = defaultdict(RateLimitState)
        
        # Cost tracking
        self._total_cost: float = 0.0
        self._cost_by_model: Dict[str, float] = defaultdict(float)
        self._cost_by_task: Dict[str, float] = defaultdict(float)
        
        # Request history for analytics
        self._request_history: List[Dict] = []
        
        # Cache
        self._cache: Dict[str, LLMResponse] = {}
        
        logger.info(f"LLMGateway initialized with providers: {list(self._providers.keys())}")
    
    def _init_providers(self):
        """Initialize all available providers"""
        providers = [
            ("openai", OpenAIProvider),
            ("anthropic", AnthropicProvider),
            ("google", GoogleProvider),
            ("deepseek", DeepSeekProvider),
            ("ollama", OllamaProvider),
        ]
        
        for name, provider_class in providers:
            try:
                provider = provider_class()
                if provider.is_available():
                    self._providers[name] = provider
                    logger.info(f"Provider {name} initialized")
                else:
                    logger.debug(f"Provider {name} not available (missing API key or package)")
            except Exception as e:
                logger.warning(f"Failed to initialize provider {name}: {e}")
    
    def _resolve_model(self, model: str) -> tuple[str, ModelInfo]:
        """Resolve model alias to actual model ID and get info"""
        # Check aliases
        resolved = self._aliases.get(model, model)
        
        # Get model info
        info = MODEL_REGISTRY.get(resolved)
        if not info:
            raise ValueError(f"Unknown model: {model}")
        
        return resolved, info
    
    def _get_provider(self, model_info: ModelInfo) -> Optional[LLMProvider]:
        """Get provider for a model"""
        return self._providers.get(model_info.provider)
    
    def _check_rate_limit(self, model: str, estimated_tokens: int = 1000) -> bool:
        """Check if request would exceed rate limit"""
        model_info = MODEL_REGISTRY.get(model)
        if not model_info:
            return True
        
        state = self._rate_limits[model]
        state.reset_if_needed()
        
        if state.requests_this_minute >= model_info.rate_limit_rpm:
            return False
        if state.tokens_this_minute + estimated_tokens > model_info.rate_limit_tpm:
            return False
        
        return True
    
    def _update_rate_limit(self, model: str, tokens: int):
        """Update rate limit counters"""
        state = self._rate_limits[model]
        state.reset_if_needed()
        state.requests_this_minute += 1
        state.tokens_this_minute += tokens
    
    def _get_cache_key(self, messages: List[Dict], model: str, **kwargs) -> str:
        """Generate cache key for request"""
        key_data = json.dumps({
            "messages": messages,
            "model": model,
            **{k: v for k, v in kwargs.items() if k not in ["stream"]}
        }, sort_keys=True)
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]
    
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-sonnet",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: Optional[str] = None,
        response_format: Optional[Dict] = None,
        task: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate a completion using the specified model.
        
        Args:
            messages: Chat messages in OpenAI format
            model: Model name or alias
            temperature: Sampling temperature
            max_tokens: Maximum output tokens
            tools: Tool definitions for function calling
            tool_choice: How to select tools
            response_format: Response format (e.g., JSON mode)
            task: Task type for analytics
            use_cache: Whether to use cached responses
        """
        # Resolve model
        resolved_model, model_info = self._resolve_model(model)
        
        # Check cache
        if self.enable_caching and use_cache:
            cache_key = self._get_cache_key(messages, resolved_model, temperature=temperature)
            if cache_key in self._cache:
                logger.debug(f"Cache hit for {cache_key}")
                return self._cache[cache_key]
        
        # Get provider
        provider = self._get_provider(model_info)
        if not provider:
            if self.enable_fallback:
                return await self._fallback_complete(
                    messages, model_info, temperature, max_tokens, tools, task
                )
            raise ValueError(f"No provider available for model {model}")
        
        # Check rate limit
        if not self._check_rate_limit(resolved_model):
            if self.enable_fallback:
                return await self._fallback_complete(
                    messages, model_info, temperature, max_tokens, tools, task
                )
            raise RuntimeError(f"Rate limit exceeded for {model}")
        
        # Make request with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await provider.complete(
                    messages=messages,
                    model=model_info.model_id,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    **kwargs
                )
                
                # Update rate limits
                self._update_rate_limit(resolved_model, response.total_tokens)
                
                # Update cost tracking
                self._total_cost += response.cost
                self._cost_by_model[resolved_model] += response.cost
                if task:
                    self._cost_by_task[task] += response.cost
                
                # Record history
                self._request_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "model": resolved_model,
                    "task": task,
                    "tokens": response.total_tokens,
                    "cost": response.cost,
                    "latency_ms": response.latency_ms
                })
                
                # Cache response
                if self.enable_caching and use_cache:
                    cache_key = self._get_cache_key(messages, resolved_model, temperature=temperature)
                    self._cache[cache_key] = response
                
                return response
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed for {model}: {e}")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        # All retries failed
        if self.enable_fallback:
            return await self._fallback_complete(
                messages, model_info, temperature, max_tokens, tools, task
            )
        raise last_error
    
    async def _fallback_complete(
        self,
        messages: List[Dict],
        original_model: ModelInfo,
        temperature: float,
        max_tokens: Optional[int],
        tools: Optional[List[Dict]],
        task: Optional[str]
    ) -> LLMResponse:
        """Attempt completion with fallback models"""
        fallback_chain = self._fallback_chains.get(original_model.provider, [])
        
        for fallback_model in fallback_chain:
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                return await self.complete(
                    messages=messages,
                    model=fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    tools=tools,
                    task=task
                )
            except Exception as e:
                logger.warning(f"Fallback {fallback_model} failed: {e}")
                continue
        
        raise RuntimeError("All fallback models failed")
    
    async def complete_for_task(
        self,
        messages: List[Dict[str, str]],
        task: str,
        budget: str = "medium",
        **kwargs
    ) -> LLMResponse:
        """
        Complete using the best model for a specific task.
        
        Args:
            messages: Chat messages
            task: Task type (analysis, reasoning, coding, quick, long_context)
            budget: Budget level (high, medium, low)
        """
        task_models = self._task_models.get(task, self._task_models["analysis"])
        model = task_models.get(budget, task_models["medium"])
        
        return await self.complete(messages=messages, model=model, task=task, **kwargs)
    
    async def stream(
        self,
        messages: List[Dict[str, str]],
        model: str = "claude-sonnet",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream a completion"""
        resolved_model, model_info = self._resolve_model(model)
        
        provider = self._get_provider(model_info)
        if not provider:
            raise ValueError(f"No provider available for model {model}")
        
        if not model_info.supports_streaming:
            # Fall back to non-streaming
            response = await self.complete(messages, model, temperature, max_tokens, **kwargs)
            yield response.content
            return
        
        async for chunk in provider.stream(
            messages=messages,
            model=model_info.model_id,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        ):
            yield chunk
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        available = []
        for model_name, info in MODEL_REGISTRY.items():
            if info.provider in self._providers:
                available.append(model_name)
        return available
    
    def get_model_info(self, model: str) -> Optional[ModelInfo]:
        """Get information about a model"""
        resolved, info = self._resolve_model(model)
        return info
    
    def get_cost_summary(self) -> Dict[str, Any]:
        """Get cost tracking summary"""
        return {
            "total_cost": self._total_cost,
            "by_model": dict(self._cost_by_model),
            "by_task": dict(self._cost_by_task),
            "request_count": len(self._request_history)
        }
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get detailed analytics"""
        if not self._request_history:
            return {"message": "No requests recorded"}
        
        total_latency = sum(r["latency_ms"] for r in self._request_history)
        total_tokens = sum(r["tokens"] for r in self._request_history)
        
        return {
            "total_requests": len(self._request_history),
            "total_tokens": total_tokens,
            "total_cost": self._total_cost,
            "avg_latency_ms": total_latency / len(self._request_history),
            "avg_tokens_per_request": total_tokens / len(self._request_history),
            "cost_by_model": dict(self._cost_by_model),
            "cost_by_task": dict(self._cost_by_task),
            "providers_available": list(self._providers.keys())
        }
    
    def clear_cache(self):
        """Clear response cache"""
        self._cache.clear()
    
    def add_alias(self, alias: str, model: str):
        """Add a model alias"""
        self._aliases[alias] = model
    
    def set_fallback_chain(self, provider: str, fallbacks: List[str]):
        """Set custom fallback chain for a provider"""
        self._fallback_chains[provider] = fallbacks


# Convenience function
def create_gateway(
    config: Optional[Dict[str, Any]] = None,
    enable_fallback: bool = True,
    enable_caching: bool = False
) -> LLMGateway:
    """Create a configured LLM Gateway"""
    return LLMGateway(
        config=config,
        enable_fallback=enable_fallback,
        enable_caching=enable_caching
    )


# Example usage
if __name__ == "__main__":
    async def main():
        gateway = LLMGateway()
        
        print("Available models:", gateway.get_available_models())
        
        # Example completion (would need API keys)
        # response = await gateway.complete(
        #     messages=[{"role": "user", "content": "What is 2+2?"}],
        #     model="gpt-4o-mini"
        # )
        # print(response.content)
        
        print("\nCost summary:", gateway.get_cost_summary())
    
    asyncio.run(main())
