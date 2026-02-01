"""
FinMind - Agent Base Classes
核心 Agent 框架实现
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import logging
import yaml
import json
import asyncio
from pathlib import Path

logger = logging.getLogger(__name__)


# ============== 基础数据结构 ==============

class ConfidenceLevel(Enum):
    """置信度等级"""
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


class AnalysisType(Enum):
    """分析类型"""
    VALUATION = "valuation"
    TECHNICAL = "technical"
    FUNDAMENTAL = "fundamental"
    SENTIMENT = "sentiment"
    MACRO = "macro"
    RISK = "risk"
    STRATEGY = "strategy"


@dataclass
class DataSource:
    """数据来源追踪"""
    name: str
    type: str  # api, file, calculation, llm_inference
    timestamp: datetime
    quality_score: float
    url: Optional[str] = None


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_number: int
    description: str
    inputs: List[str]
    output: str
    confidence: float
    supporting_data: Optional[Dict] = None


@dataclass
class Uncertainty:
    """不确定性因素"""
    factor: str
    description: str
    impact: str  # low, medium, high
    mitigation: Optional[str] = None


@dataclass
class Assumption:
    """分析假设"""
    category: str
    description: str
    impact: str  # low, medium, high
    sensitivity: float = 0.0


@dataclass
class ConfidenceScore:
    """置信度评分结果"""
    overall: float
    factors: Dict[str, float]
    weights: Dict[str, float]
    explanation: str


@dataclass
class AgentOutput:
    """Agent 标准输出格式"""
    agent_name: str
    analysis_type: AnalysisType
    timestamp: datetime
    
    # 核心结果
    result: Dict[str, Any]
    summary: str
    
    # 可追溯性
    confidence: float
    confidence_breakdown: Dict[str, float]
    reasoning_chain: List[ReasoningStep]
    data_sources: List[DataSource]
    
    # 假设与不确定性
    key_assumptions: List[str]
    uncertainties: List[Uncertainty]
    
    # 警告与元数据
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'agent_name': self.agent_name,
            'analysis_type': self.analysis_type.value,
            'timestamp': self.timestamp.isoformat(),
            'result': self.result,
            'summary': self.summary,
            'confidence': self.confidence,
            'confidence_breakdown': self.confidence_breakdown,
            'reasoning_chain': [
                {
                    'step': r.step_number,
                    'description': r.description,
                    'confidence': r.confidence
                } for r in self.reasoning_chain
            ],
            'data_sources': [
                {'name': d.name, 'type': d.type, 'quality': d.quality_score}
                for d in self.data_sources
            ],
            'key_assumptions': self.key_assumptions,
            'uncertainties': [
                {'factor': u.factor, 'impact': u.impact}
                for u in self.uncertainties
            ],
            'warnings': self.warnings
        }


@dataclass
class AnalysisContext:
    """分析上下文"""
    target: str  # 分析标的 (股票代码、公司名等)
    analysis_date: datetime
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    previous_results: Dict[str, AgentOutput] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)
    session_id: Optional[str] = None
    trace_id: Optional[str] = None


# ============== LLM 抽象层 ==============

@dataclass
class LLMResponse:
    """LLM 响应"""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    raw_response: Optional[Dict] = None


class LLMProvider(ABC):
    """LLM 提供者基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> LLMResponse:
        pass


class LLMGateway:
    """LLM 统一网关"""
    
    # 模型别名映射
    MODEL_ALIASES = {
        # 快速/便宜模型
        'fast': 'gpt-4o-mini',
        'cheap': 'gemini-flash',
        
        # 深度分析模型
        'deep': 'claude-sonnet',
        'best': 'claude-opus',
        
        # 代码专用
        'code': 'deepseek-coder',
        
        # 具体模型
        'gpt-4o': ('openai', 'gpt-4o'),
        'gpt-4o-mini': ('openai', 'gpt-4o-mini'),
        'claude-opus': ('anthropic', 'claude-3-opus-20240229'),
        'claude-sonnet': ('anthropic', 'claude-3-5-sonnet-20241022'),
        'claude-haiku': ('anthropic', 'claude-3-5-haiku-20241022'),
        'gemini-pro': ('google', 'gemini-1.5-pro'),
        'gemini-flash': ('google', 'gemini-1.5-flash'),
        'deepseek-chat': ('deepseek', 'deepseek-chat'),
        'deepseek-coder': ('deepseek', 'deepseek-coder'),
    }
    
    # 任务类型到推荐模型的映射
    TASK_MODEL_MAPPING = {
        'quick_summary': 'gpt-4o-mini',
        'deep_analysis': 'claude-sonnet',
        'valuation': 'claude-sonnet',
        'code_generation': 'deepseek-coder',
        'batch_processing': 'gemini-flash',
        'creative': 'claude-opus',
        'factual': 'gpt-4o',
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers: Dict[str, LLMProvider] = {}
        self.fallback_chain = config.get('fallback_chain', [
            'claude-sonnet', 'gpt-4o', 'gemini-pro'
        ])
        self._init_providers()
    
    def _init_providers(self):
        """初始化各个 LLM 提供者"""
        # 这里会根据配置初始化各个提供者
        # 实际实现中需要导入各个 provider 的实现类
        pass
    
    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        task_type: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        统一的补全接口
        
        Args:
            model: 模型名称或别名
            system: 系统提示词
            user: 用户输入
            temperature: 温度
            max_tokens: 最大 token 数
            task_type: 任务类型（用于自动选择模型）
        """
        # 解析模型别名
        if task_type and model == 'auto':
            model = self.TASK_MODEL_MAPPING.get(task_type, 'claude-sonnet')
        
        # 获取提供者和实际模型名
        resolved = self.MODEL_ALIASES.get(model)
        if isinstance(resolved, str):
            # 是别名，再次解析
            resolved = self.MODEL_ALIASES.get(resolved)
        
        if isinstance(resolved, tuple):
            provider_name, model_id = resolved
        else:
            # 未知模型，尝试解析
            parts = model.split('/')
            provider_name = parts[0] if len(parts) > 1 else 'openai'
            model_id = parts[-1]
        
        # 尝试调用
        provider = self.providers.get(provider_name)
        if provider:
            try:
                return await provider.complete(
                    model=model_id,
                    system=system,
                    user=user,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs
                )
            except Exception as e:
                print(f"Provider {provider_name} failed: {e}")
        
        # Fallback
        return await self._fallback_complete(system, user, temperature, max_tokens, **kwargs)
    
    async def _fallback_complete(self, system, user, temperature, max_tokens, **kwargs):
        """按 fallback 链尝试"""
        for model in self.fallback_chain:
            resolved = self.MODEL_ALIASES.get(model)
            if isinstance(resolved, tuple):
                provider_name, model_id = resolved
                provider = self.providers.get(provider_name)
                if provider:
                    try:
                        return await provider.complete(
                            model=model_id,
                            system=system,
                            user=user,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            **kwargs
                        )
                    except:
                        continue
        raise RuntimeError("All LLM providers failed")
    
    def get_model_for_task(self, task_type: str) -> str:
        """根据任务类型获取推荐模型"""
        return self.TASK_MODEL_MAPPING.get(task_type, 'claude-sonnet')


# ============== 置信度评估 ==============

class ConfidenceScorer:
    """置信度评分器"""
    
    DEFAULT_WEIGHTS = {
        'data_quality': 0.30,
        'data_completeness': 0.15,
        'reasoning_strength': 0.25,
        'external_validation': 0.15,
        'methodology_fit': 0.15,
    }
    
    def __init__(self, custom_weights: Dict[str, float] = None):
        self.weights = custom_weights or self.DEFAULT_WEIGHTS
    
    def calculate(
        self,
        data_quality: float,
        data_completeness: float,
        reasoning_strength: float,
        external_validation: float = 0.5,
        methodology_fit: float = 0.7,
        penalties: List[Dict] = None
    ) -> Dict[str, Any]:
        """
        计算综合置信度
        
        所有输入参数范围：0.0 - 1.0
        """
        # 基础加权分数
        base_score = (
            data_quality * self.weights['data_quality'] +
            data_completeness * self.weights['data_completeness'] +
            reasoning_strength * self.weights['reasoning_strength'] +
            external_validation * self.weights['external_validation'] +
            methodology_fit * self.weights['methodology_fit']
        )
        
        # 应用惩罚
        applied_penalties = []
        total_penalty = 0
        
        # 内置惩罚规则
        if data_quality < 0.4:
            p = (0.4 - data_quality) * 0.3
            total_penalty += p
            applied_penalties.append({
                'reason': 'low_data_quality',
                'penalty': p,
                'message': '数据质量较低，建议补充更多数据源'
            })
        
        if data_completeness < 0.5:
            p = (0.5 - data_completeness) * 0.2
            total_penalty += p
            applied_penalties.append({
                'reason': 'incomplete_data',
                'penalty': p,
                'message': '关键数据缺失，结论可能不完整'
            })
        
        if external_validation < 0.3:
            p = 0.1
            total_penalty += p
            applied_penalties.append({
                'reason': 'no_external_validation',
                'penalty': p,
                'message': '缺乏外部数据交叉验证'
            })
        
        # 自定义惩罚
        if penalties:
            for pen in penalties:
                total_penalty += pen.get('value', 0)
                applied_penalties.append(pen)
        
        # 最终分数
        final_score = max(0.1, min(0.95, base_score - total_penalty))
        
        return {
            'overall': final_score,
            'level': self._score_to_level(final_score),
            'breakdown': {
                'data_quality': data_quality,
                'data_completeness': data_completeness,
                'reasoning_strength': reasoning_strength,
                'external_validation': external_validation,
                'methodology_fit': methodology_fit,
            },
            'penalties': applied_penalties,
            'explanation': self._generate_explanation(final_score, applied_penalties)
        }
    
    def _score_to_level(self, score: float) -> ConfidenceLevel:
        if score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif score >= 0.65:
            return ConfidenceLevel.HIGH
        elif score >= 0.5:
            return ConfidenceLevel.MEDIUM
        elif score >= 0.35:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
    
    def _generate_explanation(self, score: float, penalties: List) -> str:
        if score >= 0.8:
            base = "高置信度分析：数据充分，推理清晰，结论可靠。"
        elif score >= 0.6:
            base = "中等置信度：整体分析合理，但存在一些不确定因素需注意。"
        elif score >= 0.4:
            base = "较低置信度：分析仅供参考，建议结合其他来源验证。"
        else:
            base = "低置信度：数据或推理存在明显不足，请谨慎对待结论。"
        
        if penalties:
            issues = [p.get('message', p.get('reason', '')) for p in penalties[:3]]
            base += f"\n⚠️ 注意: {'; '.join(issues)}"
        
        return base


# ============== Agent 基类 ==============

class BaseAgent(ABC):
    """所有分析 Agent 的基类"""
    
    def __init__(
        self,
        config_path: str,
        llm_gateway: LLMGateway,
        confidence_scorer: ConfidenceScorer = None
    ):
        self.config = self._load_config(config_path)
        self.llm = llm_gateway
        self.scorer = confidence_scorer or ConfidenceScorer()
        
        # 从配置中提取
        self.name = self.config.get('name', self.__class__.__name__)
        self.persona = self.config.get('persona', '')
        self.guardrails = self.config.get('guardrails', [])
        self.tools = self.config.get('tools', [])
        self.llm_config = self.config.get('llm_config', {})
    
    def _load_config(self, path: str) -> Dict:
        """加载 Agent 配置"""
        config_path = Path(path)
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")
    
    @property
    @abstractmethod
    def analysis_type(self) -> AnalysisType:
        """返回分析类型"""
        pass
    
    def _build_system_prompt(self, context: AnalysisContext) -> str:
        """构建系统提示词"""
        guardrail_text = '\n'.join(f"- {g}" for g in self.guardrails)
        
        return f"""{self.persona}

## 当前分析任务
- 分析标的: {context.target}
- 分析日期: {context.analysis_date.strftime('%Y-%m-%d')}
- 会话 ID: {context.session_id or 'N/A'}

## 用户偏好
{json.dumps(context.user_preferences, ensure_ascii=False, indent=2) if context.user_preferences else '无特殊偏好'}

## 输出要求
1. 所有结论必须有明确的数据支撑
2. 必须清晰说明关键假设
3. 必须给出置信度评估（不是单一数字，而是区间）
4. 必须列出不确定性因素和风险点
5. 永远不要给出"100% 确定"的结论

## 行为准则
{guardrail_text}

## 输出格式
请使用结构化的 JSON 格式输出，包含以下字段：
- summary: 一句话总结
- conclusion: 主要结论
- supporting_evidence: 支持证据列表
- assumptions: 关键假设列表
- uncertainties: 不确定性因素
- confidence_factors: 影响置信度的因素
"""
    
    @abstractmethod
    async def analyze(
        self,
        context: AnalysisContext,
        inputs: Dict[str, Any]
    ) -> AgentOutput:
        """
        执行分析 - 子类必须实现
        
        Args:
            context: 分析上下文
            inputs: 输入数据
            
        Returns:
            AgentOutput: 标准化的分析结果
        """
        pass
    
    async def _call_llm(
        self,
        context: AnalysisContext,
        user_prompt: str,
        task_type: str = 'deep_analysis',
        **kwargs
    ) -> str:
        """调用 LLM"""
        system_prompt = self._build_system_prompt(context)

        model = kwargs.pop('model', None) or self.llm_config.get('preferred_model', 'auto')
        temperature = kwargs.pop('temperature', None) or self.llm_config.get('temperature', 0.3)

        trace_id = getattr(context, 'trace_id', None) or ''
        if trace_id:
            logger.info(f"[trace:{trace_id}] LLM call: model={model}, task={task_type}")

        response = await self.llm.complete(
            model=model,
            system=system_prompt,
            user=user_prompt,
            temperature=temperature,
            task_type=task_type,
            **kwargs
        )

        return response.content
    
    def _create_output(
        self,
        result: Dict[str, Any],
        summary: str,
        reasoning_chain: List[ReasoningStep],
        data_sources: List[DataSource],
        assumptions: List[str],
        uncertainties: List[Uncertainty],
        confidence_inputs: Dict[str, float],
        warnings: List[str] = None
    ) -> AgentOutput:
        """创建标准化输出"""
        
        # 计算置信度
        confidence_result = self.scorer.calculate(**confidence_inputs)
        
        return AgentOutput(
            agent_name=self.name,
            analysis_type=self.analysis_type,
            timestamp=datetime.now(),
            result=result,
            summary=summary,
            confidence=confidence_result['overall'],
            confidence_breakdown=confidence_result['breakdown'],
            reasoning_chain=reasoning_chain,
            data_sources=data_sources,
            key_assumptions=assumptions,
            uncertainties=uncertainties,
            warnings=warnings or []
        )
    
    def _add_reasoning_step(
        self,
        steps: List[ReasoningStep],
        description: str,
        inputs: List[str],
        output: str,
        confidence: float,
        supporting_data: Dict = None
    ):
        """添加推理步骤"""
        steps.append(ReasoningStep(
            step_number=len(steps) + 1,
            description=description,
            inputs=inputs,
            output=output,
            confidence=confidence,
            supporting_data=supporting_data
        ))


# ============== 工具注册 ==============

class ToolRegistry:
    """Agent 工具注册中心"""
    
    _tools: Dict[str, Callable] = {}
    
    @classmethod
    def register(cls, name: str, func: Callable, description: str = ""):
        """注册工具"""
        cls._tools[name] = {
            'func': func,
            'description': description
        }
    
    @classmethod
    def get(cls, name: str) -> Optional[Callable]:
        """获取工具"""
        tool = cls._tools.get(name)
        return tool['func'] if tool else None
    
    @classmethod
    def list_all(cls) -> List[Dict]:
        """列出所有工具"""
        return [
            {'name': name, 'description': info['description']}
            for name, info in cls._tools.items()
        ]


# ============== 导出 ==============

__all__ = [
    'ConfidenceLevel',
    'ConfidenceScore',
    'AnalysisType',
    'DataSource',
    'ReasoningStep',
    'Uncertainty',
    'Assumption',
    'AgentOutput',
    'AnalysisContext',
    'LLMResponse',
    'LLMProvider',
    'LLMGateway',
    'ConfidenceScorer',
    'BaseAgent',
    'ToolRegistry',
]
