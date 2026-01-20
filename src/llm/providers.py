"""
LLM Provider Implementations
支持多种LLM后端的统一接口实现
"""

import os
import json
import asyncio
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, AsyncIterator, Union
from enum import Enum
import logging
import httpx
from functools import lru_cache

logger = logging.getLogger(__name__)


class ModelCapability(Enum):
    """模型能力标识"""
    TEXT_GENERATION = "text_generation"
    CODE_GENERATION = "code_generation"
    FUNCTION_CALLING = "function_calling"
    VISION = "vision"
    LONG_CONTEXT = "long_context"
    STRUCTURED_OUTPUT = "structured_output"
    STREAMING = "streaming"


@dataclass
class ModelInfo:
    """模型信息"""
    provider: str
    model_id: str
    display_name: str
    context_window: int
    max_output_tokens: int
    capabilities: List[ModelCapability]
    cost_per_1k_input: float  # USD
    cost_per_1k_output: float  # USD
    latency_ms: int  # 平均响应时间
    quality_score: float  # 0-1，质量评分
    
    @property
    def supports_functions(self) -> bool:
        return ModelCapability.FUNCTION_CALLING in self.capabilities
    
    @property
    def supports_vision(self) -> bool:
        return ModelCapability.VISION in self.capabilities


@dataclass
class LLMRequest:
    """LLM请求"""
    messages: List[Dict[str, Any]]
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[str] = None
    response_format: Optional[Dict] = None
    stream: bool = False
    stop: Optional[List[str]] = None
    
    # 元数据
    task_type: Optional[str] = None  # 用于智能路由
    priority: str = "normal"  # high, normal, low
    timeout: int = 60


@dataclass
class LLMResponse:
    """LLM响应"""
    content: str
    model: str
    provider: str
    usage: Dict[str, int]
    finish_reason: str
    tool_calls: Optional[List[Dict]] = None
    
    # 元数据
    latency_ms: int = 0
    cost_usd: float = 0.0
    cached: bool = False


@dataclass
class ProviderConfig:
    """Provider配置"""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    rate_limit_rpm: int = 60
    rate_limit_tpm: int = 100000


class BaseLLMProvider(ABC):
    """LLM Provider基类"""
    
    PROVIDER_NAME: str = "base"
    
    def __init__(self, config: ProviderConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None
        self._request_count = 0
        self._token_count = 0
        self._last_reset = time.time()
    
    async def get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout,
                headers=self._get_headers()
            )
        return self._client
    
    @abstractmethod
    def _get_headers(self) -> Dict[str, str]:
        """获取请求头"""
        pass
    
    @abstractmethod
    def _get_endpoint(self) -> str:
        """获取API端点"""
        pass
    
    @abstractmethod
    def _format_request(self, request: LLMRequest) -> Dict[str, Any]:
        """格式化请求体"""
        pass
    
    @abstractmethod
    def _parse_response(self, response: Dict[str, Any], request: LLMRequest) -> LLMResponse:
        """解析响应"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[ModelInfo]:
        """获取可用模型列表"""
        pass
    
    async def _check_rate_limit(self, tokens: int = 0):
        """检查速率限制"""
        now = time.time()
        if now - self._last_reset > 60:
            self._request_count = 0
            self._token_count = 0
            self._last_reset = now
        
        if self._request_count >= self.config.rate_limit_rpm:
            wait_time = 60 - (now - self._last_reset)
            if wait_time > 0:
                logger.warning(f"Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                self._request_count = 0
                self._token_count = 0
                self._last_reset = time.time()
        
        self._request_count += 1
        self._token_count += tokens
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """执行补全请求"""
        await self._check_rate_limit()
        
        client = await self.get_client()
        endpoint = self._get_endpoint()
        body = self._format_request(request)
        
        start_time = time.time()
        last_error = None
        
        for attempt in range(self.config.max_retries):
            try:
                response = await client.post(endpoint, json=body)
                response.raise_for_status()
                
                result = response.json()
                llm_response = self._parse_response(result, request)
                llm_response.latency_ms = int((time.time() - start_time) * 1000)
                
                return llm_response
                
            except httpx.HTTPStatusError as e:
                last_error = e
                if e.response.status_code == 429:  # Rate limit
                    wait = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Rate limited, retrying in {wait}s")
                    await asyncio.sleep(wait)
                elif e.response.status_code >= 500:  # Server error
                    wait = self.config.retry_delay * (2 ** attempt)
                    logger.warning(f"Server error, retrying in {wait}s")
                    await asyncio.sleep(wait)
                else:
                    raise
            except Exception as e:
                last_error = e
                wait = self.config.retry_delay * (2 ** attempt)
                logger.warning(f"Request failed: {e}, retrying in {wait}s")
                await asyncio.sleep(wait)
        
        raise last_error or Exception("Max retries exceeded")
    
    async def stream(self, request: LLMRequest) -> AsyncIterator[str]:
        """流式响应"""
        request.stream = True
        await self._check_rate_limit()
        
        client = await self.get_client()
        endpoint = self._get_endpoint()
        body = self._format_request(request)
        
        async with client.stream("POST", endpoint, json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = self._extract_stream_content(chunk)
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue
    
    def _extract_stream_content(self, chunk: Dict) -> Optional[str]:
        """从流式chunk提取内容"""
        return None
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            request = LLMRequest(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5
            )
            await self.complete(request)
            return True
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False
    
    async def close(self):
        """关闭客户端"""
        if self._client:
            await self._client.aclose()
            self._client = None


class OpenAIProvider(BaseLLMProvider):
    """OpenAI Provider"""
    
    PROVIDER_NAME = "openai"
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    def _get_endpoint(self) -> str:
        base = self.config.base_url or "https://api.openai.com/v1"
        return f"{base}/chat/completions"
    
    def _format_request(self, request: LLMRequest) -> Dict[str, Any]:
        body = {
            "model": request.model or "gpt-4o",
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }
        
        if request.tools:
            body["tools"] = request.tools
            if request.tool_choice:
                body["tool_choice"] = request.tool_choice
        
        if request.response_format:
            body["response_format"] = request.response_format
        
        if request.stop:
            body["stop"] = request.stop
        
        return body
    
    def _parse_response(self, response: Dict, request: LLMRequest) -> LLMResponse:
        choice = response["choices"][0]
        message = choice["message"]
        usage = response.get("usage", {})
        
        # 计算成本
        model = request.model or "gpt-4o"
        cost = self._calculate_cost(model, usage)
        
        return LLMResponse(
            content=message.get("content", ""),
            model=response["model"],
            provider=self.PROVIDER_NAME,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=message.get("tool_calls"),
            cost_usd=cost
        )
    
    def _extract_stream_content(self, chunk: Dict) -> Optional[str]:
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {})
            return delta.get("content")
        return None
    
    def _calculate_cost(self, model: str, usage: Dict) -> float:
        """计算API调用成本"""
        # 2025年价格（示例）
        pricing = {
            "gpt-4o": (0.005, 0.015),
            "gpt-4o-mini": (0.00015, 0.0006),
            "gpt-4-turbo": (0.01, 0.03),
            "o1-preview": (0.015, 0.06),
            "o1-mini": (0.003, 0.012)
        }
        
        input_price, output_price = pricing.get(model, (0.01, 0.03))
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        return (input_tokens * input_price + output_tokens * output_price) / 1000
    
    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                provider="openai",
                model_id="gpt-4o",
                display_name="GPT-4o",
                context_window=128000,
                max_output_tokens=16384,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION,
                    ModelCapability.STRUCTURED_OUTPUT,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.005,
                cost_per_1k_output=0.015,
                latency_ms=800,
                quality_score=0.92
            ),
            ModelInfo(
                provider="openai",
                model_id="gpt-4o-mini",
                display_name="GPT-4o Mini",
                context_window=128000,
                max_output_tokens=16384,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.00015,
                cost_per_1k_output=0.0006,
                latency_ms=400,
                quality_score=0.85
            ),
            ModelInfo(
                provider="openai",
                model_id="o1-preview",
                display_name="o1 Preview",
                context_window=128000,
                max_output_tokens=32768,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.LONG_CONTEXT
                ],
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.06,
                latency_ms=5000,
                quality_score=0.98
            )
        ]


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude Provider"""
    
    PROVIDER_NAME = "anthropic"
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.config.api_key,
            "anthropic-version": "2024-01-01",
            "Content-Type": "application/json"
        }
    
    def _get_endpoint(self) -> str:
        base = self.config.base_url or "https://api.anthropic.com/v1"
        return f"{base}/messages"
    
    def _format_request(self, request: LLMRequest) -> Dict[str, Any]:
        # 分离system message
        system = None
        messages = []
        for msg in request.messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                messages.append(msg)
        
        body = {
            "model": request.model or "claude-sonnet-4-20250514",
            "messages": messages,
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }
        
        if system:
            body["system"] = system
        
        # Anthropic用不同的temperature范围
        body["temperature"] = min(request.temperature, 1.0)
        
        if request.tools:
            body["tools"] = self._convert_tools(request.tools)
        
        if request.stop:
            body["stop_sequences"] = request.stop
        
        return body
    
    def _convert_tools(self, tools: List[Dict]) -> List[Dict]:
        """转换OpenAI格式的tools到Anthropic格式"""
        anthropic_tools = []
        for tool in tools:
            if tool.get("type") == "function":
                func = tool["function"]
                anthropic_tools.append({
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}})
                })
        return anthropic_tools
    
    def _parse_response(self, response: Dict, request: LLMRequest) -> LLMResponse:
        content_blocks = response.get("content", [])
        text_content = ""
        tool_calls = []
        
        for block in content_blocks:
            if block["type"] == "text":
                text_content += block["text"]
            elif block["type"] == "tool_use":
                tool_calls.append({
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block["input"])
                    }
                })
        
        usage = response.get("usage", {})
        model = request.model or "claude-sonnet-4-20250514"
        cost = self._calculate_cost(model, usage)
        
        return LLMResponse(
            content=text_content,
            model=response.get("model", model),
            provider=self.PROVIDER_NAME,
            usage={
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            },
            finish_reason=response.get("stop_reason", "end_turn"),
            tool_calls=tool_calls if tool_calls else None,
            cost_usd=cost
        )
    
    def _extract_stream_content(self, chunk: Dict) -> Optional[str]:
        if chunk.get("type") == "content_block_delta":
            delta = chunk.get("delta", {})
            if delta.get("type") == "text_delta":
                return delta.get("text")
        return None
    
    def _calculate_cost(self, model: str, usage: Dict) -> float:
        pricing = {
            "claude-opus-4-20250514": (0.015, 0.075),
            "claude-sonnet-4-20250514": (0.003, 0.015),
            "claude-haiku-3-5-20241022": (0.0008, 0.004)
        }
        
        input_price, output_price = pricing.get(model, (0.003, 0.015))
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)
        
        return (input_tokens * input_price + output_tokens * output_price) / 1000
    
    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                provider="anthropic",
                model_id="claude-opus-4-20250514",
                display_name="Claude Opus 4",
                context_window=200000,
                max_output_tokens=32768,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION,
                    ModelCapability.LONG_CONTEXT,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.015,
                cost_per_1k_output=0.075,
                latency_ms=1200,
                quality_score=0.96
            ),
            ModelInfo(
                provider="anthropic",
                model_id="claude-sonnet-4-20250514",
                display_name="Claude Sonnet 4",
                context_window=200000,
                max_output_tokens=16384,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION,
                    ModelCapability.LONG_CONTEXT,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                latency_ms=600,
                quality_score=0.93
            ),
            ModelInfo(
                provider="anthropic",
                model_id="claude-haiku-3-5-20241022",
                display_name="Claude 3.5 Haiku",
                context_window=200000,
                max_output_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.0008,
                cost_per_1k_output=0.004,
                latency_ms=300,
                quality_score=0.88
            )
        ]


class GoogleProvider(BaseLLMProvider):
    """Google Gemini Provider"""
    
    PROVIDER_NAME = "google"
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json"
        }
    
    def _get_endpoint(self) -> str:
        base = self.config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        return f"{base}/models/{{model}}:generateContent?key={self.config.api_key}"
    
    def _format_request(self, request: LLMRequest) -> Dict[str, Any]:
        # 转换消息格式
        contents = []
        system_instruction = None
        
        for msg in request.messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            else:
                role = "user" if msg["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": msg["content"]}]
                })
        
        body = {
            "contents": contents,
            "generationConfig": {
                "temperature": request.temperature,
                "maxOutputTokens": request.max_tokens
            }
        }
        
        if system_instruction:
            body["systemInstruction"] = {"parts": [{"text": system_instruction}]}
        
        if request.stop:
            body["generationConfig"]["stopSequences"] = request.stop
        
        return body
    
    def _get_endpoint_with_model(self, model: str) -> str:
        base = self.config.base_url or "https://generativelanguage.googleapis.com/v1beta"
        return f"{base}/models/{model}:generateContent?key={self.config.api_key}"
    
    async def complete(self, request: LLMRequest) -> LLMResponse:
        """重写以支持动态model URL"""
        await self._check_rate_limit()
        
        client = await self.get_client()
        model = request.model or "gemini-2.0-flash"
        endpoint = self._get_endpoint_with_model(model)
        body = self._format_request(request)
        
        start_time = time.time()
        response = await client.post(endpoint, json=body)
        response.raise_for_status()
        
        result = response.json()
        llm_response = self._parse_response(result, request)
        llm_response.latency_ms = int((time.time() - start_time) * 1000)
        
        return llm_response
    
    def _parse_response(self, response: Dict, request: LLMRequest) -> LLMResponse:
        candidates = response.get("candidates", [{}])
        content = candidates[0].get("content", {})
        parts = content.get("parts", [{}])
        text = parts[0].get("text", "") if parts else ""
        
        usage = response.get("usageMetadata", {})
        
        return LLMResponse(
            content=text,
            model=request.model or "gemini-2.0-flash",
            provider=self.PROVIDER_NAME,
            usage={
                "input_tokens": usage.get("promptTokenCount", 0),
                "output_tokens": usage.get("candidatesTokenCount", 0),
                "total_tokens": usage.get("totalTokenCount", 0)
            },
            finish_reason=candidates[0].get("finishReason", "STOP"),
            cost_usd=0.0  # Gemini 目前很多情况下免费
        )
    
    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                provider="google",
                model_id="gemini-2.0-flash",
                display_name="Gemini 2.0 Flash",
                context_window=1000000,
                max_output_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION,
                    ModelCapability.LONG_CONTEXT,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                latency_ms=400,
                quality_score=0.90
            ),
            ModelInfo(
                provider="google",
                model_id="gemini-1.5-pro",
                display_name="Gemini 1.5 Pro",
                context_window=2000000,
                max_output_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.VISION,
                    ModelCapability.LONG_CONTEXT,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.00125,
                cost_per_1k_output=0.005,
                latency_ms=1000,
                quality_score=0.91
            )
        ]


class DeepSeekProvider(BaseLLMProvider):
    """DeepSeek Provider"""
    
    PROVIDER_NAME = "deepseek"
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
    
    def _get_endpoint(self) -> str:
        base = self.config.base_url or "https://api.deepseek.com/v1"
        return f"{base}/chat/completions"
    
    def _format_request(self, request: LLMRequest) -> Dict[str, Any]:
        # DeepSeek API兼容OpenAI格式
        body = {
            "model": request.model or "deepseek-chat",
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": request.stream
        }
        
        if request.tools:
            body["tools"] = request.tools
        
        if request.stop:
            body["stop"] = request.stop
        
        return body
    
    def _parse_response(self, response: Dict, request: LLMRequest) -> LLMResponse:
        choice = response["choices"][0]
        message = choice["message"]
        usage = response.get("usage", {})
        
        return LLMResponse(
            content=message.get("content", ""),
            model=response.get("model", request.model or "deepseek-chat"),
            provider=self.PROVIDER_NAME,
            usage={
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            finish_reason=choice.get("finish_reason", "stop"),
            tool_calls=message.get("tool_calls"),
            cost_usd=self._calculate_cost(request.model or "deepseek-chat", usage)
        )
    
    def _calculate_cost(self, model: str, usage: Dict) -> float:
        # DeepSeek价格非常便宜
        pricing = {
            "deepseek-chat": (0.00014, 0.00028),
            "deepseek-coder": (0.00014, 0.00028),
            "deepseek-reasoner": (0.00055, 0.00219)
        }
        
        input_price, output_price = pricing.get(model, (0.00014, 0.00028))
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        
        return (input_tokens * input_price + output_tokens * output_price) / 1000
    
    def _extract_stream_content(self, chunk: Dict) -> Optional[str]:
        if "choices" in chunk and chunk["choices"]:
            delta = chunk["choices"][0].get("delta", {})
            return delta.get("content")
        return None
    
    def get_available_models(self) -> List[ModelInfo]:
        return [
            ModelInfo(
                provider="deepseek",
                model_id="deepseek-chat",
                display_name="DeepSeek Chat",
                context_window=64000,
                max_output_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.FUNCTION_CALLING,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.00014,
                cost_per_1k_output=0.00028,
                latency_ms=600,
                quality_score=0.89
            ),
            ModelInfo(
                provider="deepseek",
                model_id="deepseek-reasoner",
                display_name="DeepSeek Reasoner (R1)",
                context_window=64000,
                max_output_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.00055,
                cost_per_1k_output=0.00219,
                latency_ms=3000,
                quality_score=0.94
            )
        ]


class OllamaProvider(BaseLLMProvider):
    """Ollama本地Provider"""
    
    PROVIDER_NAME = "ollama"
    
    def _get_headers(self) -> Dict[str, str]:
        return {"Content-Type": "application/json"}
    
    def _get_endpoint(self) -> str:
        base = self.config.base_url or "http://localhost:11434"
        return f"{base}/api/chat"
    
    def _format_request(self, request: LLMRequest) -> Dict[str, Any]:
        return {
            "model": request.model or "llama3.2",
            "messages": request.messages,
            "stream": request.stream,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_tokens
            }
        }
    
    def _parse_response(self, response: Dict, request: LLMRequest) -> LLMResponse:
        message = response.get("message", {})
        
        return LLMResponse(
            content=message.get("content", ""),
            model=response.get("model", request.model or "llama3.2"),
            provider=self.PROVIDER_NAME,
            usage={
                "input_tokens": response.get("prompt_eval_count", 0),
                "output_tokens": response.get("eval_count", 0),
                "total_tokens": response.get("prompt_eval_count", 0) + response.get("eval_count", 0)
            },
            finish_reason="stop",
            cost_usd=0.0  # 本地运行无成本
        )
    
    def _extract_stream_content(self, chunk: Dict) -> Optional[str]:
        message = chunk.get("message", {})
        return message.get("content")
    
    async def list_local_models(self) -> List[str]:
        """列出本地可用模型"""
        try:
            client = await self.get_client()
            base = self.config.base_url or "http://localhost:11434"
            response = await client.get(f"{base}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.warning(f"Failed to list Ollama models: {e}")
            return []
    
    def get_available_models(self) -> List[ModelInfo]:
        # 常见的本地模型
        return [
            ModelInfo(
                provider="ollama",
                model_id="llama3.2",
                display_name="Llama 3.2 (8B)",
                context_window=128000,
                max_output_tokens=4096,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                latency_ms=500,
                quality_score=0.82
            ),
            ModelInfo(
                provider="ollama",
                model_id="qwen2.5:32b",
                display_name="Qwen 2.5 (32B)",
                context_window=131072,
                max_output_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                latency_ms=1500,
                quality_score=0.88
            ),
            ModelInfo(
                provider="ollama",
                model_id="deepseek-r1:32b",
                display_name="DeepSeek R1 (32B)",
                context_window=64000,
                max_output_tokens=8192,
                capabilities=[
                    ModelCapability.TEXT_GENERATION,
                    ModelCapability.CODE_GENERATION,
                    ModelCapability.STREAMING
                ],
                cost_per_1k_input=0.0,
                cost_per_1k_output=0.0,
                latency_ms=2000,
                quality_score=0.90
            )
        ]


# ============ Provider Registry ============

class LLMProviderRegistry:
    """LLM Provider注册表"""
    
    _providers: Dict[str, type] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "deepseek": DeepSeekProvider,
        "ollama": OllamaProvider
    }
    
    _instances: Dict[str, BaseLLMProvider] = {}
    
    @classmethod
    def register(cls, name: str, provider_class: type):
        """注册新Provider"""
        cls._providers[name] = provider_class
    
    @classmethod
    def get(cls, name: str, config: Optional[ProviderConfig] = None) -> BaseLLMProvider:
        """获取Provider实例"""
        if name not in cls._instances:
            if name not in cls._providers:
                raise ValueError(f"Unknown provider: {name}")
            
            if config is None:
                config = cls._get_default_config(name)
            
            cls._instances[name] = cls._providers[name](config)
        
        return cls._instances[name]
    
    @classmethod
    def _get_default_config(cls, name: str) -> ProviderConfig:
        """从环境变量获取默认配置"""
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GOOGLE_API_KEY",
            "deepseek": "DEEPSEEK_API_KEY"
        }
        
        api_key = os.getenv(env_map.get(name, ""))
        return ProviderConfig(api_key=api_key)
    
    @classmethod
    def list_providers(cls) -> List[str]:
        """列出所有可用Provider"""
        return list(cls._providers.keys())
    
    @classmethod
    def list_all_models(cls) -> List[ModelInfo]:
        """列出所有Provider的所有模型"""
        models = []
        for name in cls._providers:
            try:
                provider = cls.get(name)
                models.extend(provider.get_available_models())
            except Exception:
                pass
        return models


# ============ Intelligent Router ============

class IntelligentRouter:
    """智能模型路由器"""
    
    # 任务类型到最佳模型的映射
    TASK_MODEL_MAP = {
        "deep_analysis": ["claude-opus-4-20250514", "o1-preview", "deepseek-reasoner"],
        "quick_response": ["gpt-4o-mini", "claude-haiku-3-5-20241022", "gemini-2.0-flash"],
        "code_generation": ["claude-sonnet-4-20250514", "gpt-4o", "deepseek-coder"],
        "long_document": ["gemini-1.5-pro", "claude-sonnet-4-20250514"],
        "vision": ["gpt-4o", "claude-sonnet-4-20250514", "gemini-2.0-flash"],
        "structured_output": ["gpt-4o", "claude-sonnet-4-20250514"],
        "cost_efficient": ["deepseek-chat", "gpt-4o-mini", "gemini-2.0-flash"]
    }
    
    def __init__(self):
        self.registry = LLMProviderRegistry
        self._model_stats: Dict[str, Dict] = {}
    
    def select_model(
        self,
        task_type: str = "general",
        required_capabilities: Optional[List[ModelCapability]] = None,
        max_cost_per_1k: Optional[float] = None,
        max_latency_ms: Optional[int] = None,
        min_quality: Optional[float] = None,
        preferred_providers: Optional[List[str]] = None,
        fallback_enabled: bool = True
    ) -> List[ModelInfo]:
        """选择最佳模型"""
        
        all_models = self.registry.list_all_models()
        candidates = []
        
        for model in all_models:
            # 检查能力
            if required_capabilities:
                if not all(cap in model.capabilities for cap in required_capabilities):
                    continue
            
            # 检查成本
            if max_cost_per_1k is not None:
                avg_cost = (model.cost_per_1k_input + model.cost_per_1k_output) / 2
                if avg_cost > max_cost_per_1k:
                    continue
            
            # 检查延迟
            if max_latency_ms is not None and model.latency_ms > max_latency_ms:
                continue
            
            # 检查质量
            if min_quality is not None and model.quality_score < min_quality:
                continue
            
            # 检查Provider偏好
            if preferred_providers and model.provider not in preferred_providers:
                continue
            
            candidates.append(model)
        
        # 按任务类型排序
        if task_type in self.TASK_MODEL_MAP:
            preferred_models = self.TASK_MODEL_MAP[task_type]
            candidates.sort(
                key=lambda m: (
                    preferred_models.index(m.model_id) if m.model_id in preferred_models else 999,
                    -m.quality_score
                )
            )
        else:
            # 默认按质量排序
            candidates.sort(key=lambda m: -m.quality_score)
        
        if not candidates and fallback_enabled:
            # 返回默认fallback
            return [m for m in all_models if m.model_id == "gpt-4o-mini"]
        
        return candidates
    
    def get_provider_for_model(self, model_id: str) -> Optional[BaseLLMProvider]:
        """根据模型ID获取对应的Provider"""
        all_models = self.registry.list_all_models()
        for model in all_models:
            if model.model_id == model_id:
                return self.registry.get(model.provider)
        return None
    
    def record_result(self, model_id: str, success: bool, latency_ms: int, error: Optional[str] = None):
        """记录模型调用结果用于智能路由"""
        if model_id not in self._model_stats:
            self._model_stats[model_id] = {
                "total_calls": 0,
                "success_calls": 0,
                "total_latency": 0,
                "errors": []
            }
        
        stats = self._model_stats[model_id]
        stats["total_calls"] += 1
        if success:
            stats["success_calls"] += 1
        stats["total_latency"] += latency_ms
        if error:
            stats["errors"].append(error)
            if len(stats["errors"]) > 100:
                stats["errors"] = stats["errors"][-100:]
    
    def get_model_health(self, model_id: str) -> Dict[str, Any]:
        """获取模型健康状态"""
        stats = self._model_stats.get(model_id, {})
        if not stats.get("total_calls"):
            return {"status": "unknown", "success_rate": None, "avg_latency": None}
        
        success_rate = stats["success_calls"] / stats["total_calls"]
        avg_latency = stats["total_latency"] / stats["total_calls"]
        
        status = "healthy"
        if success_rate < 0.9:
            status = "degraded"
        if success_rate < 0.5:
            status = "unhealthy"
        
        return {
            "status": status,
            "success_rate": success_rate,
            "avg_latency": avg_latency,
            "total_calls": stats["total_calls"]
        }
