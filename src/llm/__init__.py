"""
FinanceAI Pro - LLM模块

统一的LLM网关，支持多个模型提供商：
- OpenAI (GPT-4o, GPT-4o-mini)
- Anthropic (Claude Sonnet, Claude Opus)
- Google (Gemini Pro, Gemini Flash)
- DeepSeek
- Ollama (本地模型)

特性：
- 智能路由：根据任务类型选择最优模型
- 自动降级：主模型失败时自动切换备用
- 成本追踪：记录API调用成本
- 流式支持：支持流式响应
"""

from .gateway import (
    LLMGateway,
    LLMResponse,
    ModelInfo,
    ModelCapability,
    LLMProvider,
    RateLimitState
)

from .providers import (
    BaseLLMProvider,
    OpenAIProvider,
    AnthropicProvider,
    GoogleProvider,
    DeepSeekProvider,
    OllamaProvider
)

__all__ = [
    # Gateway
    "LLMGateway",
    "LLMResponse",
    "ModelInfo",
    "ModelCapability",
    "LLMProvider",
    "RateLimitState",

    # Providers
    "BaseLLMProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "GoogleProvider",
    "DeepSeekProvider",
    "OllamaProvider"
]


# 模型别名映射
MODEL_ALIASES = {
    # OpenAI
    "gpt4": "gpt-4o",
    "gpt4o": "gpt-4o",
    "gpt4-mini": "gpt-4o-mini",
    
    # Anthropic
    "claude": "claude-sonnet-4-20250514",
    "claude-sonnet": "claude-sonnet-4-20250514",
    "claude-opus": "claude-opus-4-20250514",
    "claude-haiku": "claude-haiku-4-20250514",
    
    # Google
    "gemini": "gemini-1.5-pro",
    "gemini-pro": "gemini-1.5-pro",
    "gemini-flash": "gemini-1.5-flash",
    
    # DeepSeek
    "deepseek": "deepseek-chat",
    
    # Ollama
    "llama": "ollama/llama3",
    "llama3": "ollama/llama3",
    "mistral": "ollama/mistral"
}


def resolve_model_alias(alias: str) -> str:
    """解析模型别名"""
    return MODEL_ALIASES.get(alias.lower(), alias)
