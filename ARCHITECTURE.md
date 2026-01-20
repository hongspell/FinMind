# FinanceAI Pro - 模块化金融分析平台架构设计

> 🎯 核心理念：**方法论可配置、数据源可插拔、分析链可组合、模型可切换**

---

## 一、设计哲学

### 1.1 为什么 FinRobot 模式不可持续

```
❌ 传统模式：代码 = 逻辑 + 数据 + 分析方法
   → 改一个指标要改代码
   → 换一个模型要重构
   → 加一个数据源要大改

✅ 新模式：代码 = 框架 + 配置 + 插件
   → 方法论是 YAML/JSON 配置
   → 数据源是可插拔适配器
   → 分析链是可组合的 DAG
```

### 1.2 核心设计原则

| 原则 | 实现方式 |
|------|----------|
| **方法论外置** | 分析逻辑用配置文件定义，不硬编码 |
| **数据源抽象** | 统一 DataProvider 接口，支持热插拔 |
| **Agent 可组合** | 每个 Agent 是独立单元，可任意组合成分析链 |
| **模型无关** | 通过 LLM Gateway 抽象，一键切换任何模型 |
| **决策可追溯** | 每个结论都有推理链路和数据依据 |
| **风险优先** | 内置多层风险评估，永不给"确定性"结论 |

---

## 二、整体架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           FinanceAI Pro Platform                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐ │
│  │   Web UI    │   │   CLI Tool  │   │  REST API   │   │  MCP Server │ │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘ │
│         │                 │                 │                 │         │
│  ═══════╪═════════════════╪═════════════════╪═════════════════╪═══════  │
│                      【Orchestration Layer】                            │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Task Planner  │  Agent Router  │  Chain Executor  │  State Mgr  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                        【Agent Layer】                                  │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│  │   Macro    │ │ Valuation  │ │ Technical  │ │ Sentiment  │          │
│  │   Agent    │ │   Agent    │ │   Agent    │ │   Agent    │          │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘          │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│  │    Risk    │ │  Strategy  │ │  Earnings  │ │  Sector    │          │
│  │   Agent    │ │   Agent    │ │   Agent    │ │   Agent    │          │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                     【Knowledge Layer】                                 │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Vector Store  │  Knowledge Graph  │  Time Series DB  │  Cache   │  │
│  │  (Qdrant/PG)   │   (Neo4j/PG)      │  (TimescaleDB)   │ (Redis)  │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                       【Data Layer】                                    │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐          │
│  │   Market   │ │  Financial │ │    News    │ │   Macro    │          │
│  │  Provider  │ │  Provider  │ │  Provider  │ │  Provider  │          │
│  │(yfinance)  │ │(SEC/报表)  │ │(RSS/API)   │ │(FRED/WB)   │          │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘          │
│                                                                         │
│  ═══════════════════════════════════════════════════════════════════   │
│                       【LLM Gateway】                                   │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  OpenAI  │  Claude  │  Gemini  │  DeepSeek  │  Local(Ollama)     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 三、目录结构

```
financeai-pro/
│
├── config/                          # 🎯 核心：所有方法论配置
│   ├── methodologies/               # 分析方法论定义
│   │   ├── valuation/
│   │   │   ├── dcf.yaml            # DCF 估值方法配置
│   │   │   ├── comparables.yaml    # 可比公司估值
│   │   │   ├── sum_of_parts.yaml   # 分部估值
│   │   │   └── custom/             # 用户自定义方法
│   │   ├── technical/
│   │   │   ├── trend_following.yaml
│   │   │   ├── mean_reversion.yaml
│   │   │   └── momentum.yaml
│   │   ├── fundamental/
│   │   │   ├── quality_factors.yaml
│   │   │   ├── growth_analysis.yaml
│   │   │   └── profitability.yaml
│   │   └── risk/
│   │       ├── var_calculation.yaml
│   │       ├── scenario_analysis.yaml
│   │       └── stress_test.yaml
│   │
│   ├── agents/                      # Agent 行为配置
│   │   ├── macro_agent.yaml
│   │   ├── valuation_agent.yaml
│   │   ├── risk_agent.yaml
│   │   └── strategy_agent.yaml
│   │
│   ├── chains/                      # 分析链配置
│   │   ├── full_analysis.yaml      # 完整分析流程
│   │   ├── quick_scan.yaml         # 快速扫描
│   │   ├── earnings_review.yaml    # 财报分析
│   │   └── sector_rotation.yaml    # 板块轮动
│   │
│   ├── prompts/                     # Prompt 模板
│   │   ├── base/
│   │   │   ├── analyst_persona.txt
│   │   │   └── risk_disclaimer.txt
│   │   ├── valuation/
│   │   ├── technical/
│   │   └── synthesis/
│   │
│   ├── data_sources.yaml           # 数据源配置
│   ├── models.yaml                 # LLM 模型配置
│   └── risk_rules.yaml             # 风险规则配置
│
├── src/
│   ├── core/                        # 核心框架
│   │   ├── __init__.py
│   │   ├── config_loader.py        # 配置加载器
│   │   ├── registry.py             # 组件注册中心
│   │   ├── pipeline.py             # 分析管道
│   │   └── context.py              # 分析上下文
│   │
│   ├── llm/                         # LLM 抽象层
│   │   ├── __init__.py
│   │   ├── gateway.py              # 统一网关
│   │   ├── providers/
│   │   │   ├── openai_provider.py
│   │   │   ├── claude_provider.py
│   │   │   ├── gemini_provider.py
│   │   │   ├── deepseek_provider.py
│   │   │   └── ollama_provider.py
│   │   ├── router.py               # 智能路由（按任务选模型）
│   │   └── fallback.py             # 降级策略
│   │
│   ├── data/                        # 数据层
│   │   ├── __init__.py
│   │   ├── base.py                 # DataProvider 基类
│   │   ├── providers/
│   │   │   ├── market/
│   │   │   │   ├── yfinance_provider.py
│   │   │   │   ├── polygon_provider.py
│   │   │   │   └── crypto_provider.py
│   │   │   ├── fundamental/
│   │   │   │   ├── sec_provider.py      # SEC 财报
│   │   │   │   ├── simfin_provider.py
│   │   │   │   └── manual_provider.py   # 手动导入
│   │   │   ├── news/
│   │   │   │   ├── newsapi_provider.py
│   │   │   │   ├── rss_provider.py
│   │   │   │   └── twitter_provider.py
│   │   │   ├── macro/
│   │   │   │   ├── fred_provider.py     # 美联储数据
│   │   │   │   ├── worldbank_provider.py
│   │   │   │   └── china_provider.py    # 中国宏观数据
│   │   │   └── alternative/
│   │   │       ├── satellite_provider.py
│   │   │       └── sentiment_provider.py
│   │   ├── normalizers/             # 数据标准化
│   │   │   ├── financial_normalizer.py
│   │   │   └── time_series_normalizer.py
│   │   └── quality/                 # 数据质量检查
│   │       ├── validator.py
│   │       └── completeness.py
│   │
│   ├── knowledge/                   # 知识层
│   │   ├── __init__.py
│   │   ├── vectorstore/
│   │   │   ├── base.py
│   │   │   ├── qdrant_store.py
│   │   │   └── pgvector_store.py
│   │   ├── graph/
│   │   │   ├── base.py
│   │   │   ├── neo4j_graph.py
│   │   │   └── pg_graph.py
│   │   ├── indexing/
│   │   │   ├── document_indexer.py
│   │   │   ├── earnings_indexer.py
│   │   │   └── news_indexer.py
│   │   └── retrieval/
│   │       ├── hybrid_retriever.py  # 混合检索
│   │       ├── temporal_retriever.py # 时序检索
│   │       └── graph_retriever.py   # 图检索
│   │
│   ├── agents/                      # Agent 实现
│   │   ├── __init__.py
│   │   ├── base.py                 # Agent 基类
│   │   ├── macro_agent.py          # 宏观分析
│   │   ├── valuation_agent.py      # 估值分析
│   │   ├── technical_agent.py      # 技术分析
│   │   ├── sentiment_agent.py      # 情绪分析
│   │   ├── risk_agent.py           # 风险评估
│   │   ├── earnings_agent.py       # 财报分析
│   │   ├── sector_agent.py         # 板块分析
│   │   ├── strategy_agent.py       # 策略综合
│   │   └── tools/                   # Agent 工具
│   │       ├── calculator.py       # 财务计算器
│   │       ├── screener.py         # 股票筛选器
│   │       ├── comparator.py       # 对比工具
│   │       └── chart_analyzer.py   # 图表分析
│   │
│   ├── chains/                      # 分析链
│   │   ├── __init__.py
│   │   ├── base.py                 # Chain 基类
│   │   ├── builder.py              # Chain 构建器（从配置）
│   │   ├── executor.py             # Chain 执行器
│   │   └── templates/
│   │       ├── stock_analysis.py
│   │       ├── sector_analysis.py
│   │       ├── portfolio_review.py
│   │       └── event_analysis.py
│   │
│   ├── orchestrator/                # 编排层
│   │   ├── __init__.py
│   │   ├── planner.py              # 任务规划
│   │   ├── router.py               # Agent 路由
│   │   ├── aggregator.py           # 结果聚合
│   │   ├── conflict_resolver.py    # 冲突解决
│   │   └── state_manager.py        # 状态管理
│   │
│   ├── output/                      # 输出层
│   │   ├── __init__.py
│   │   ├── report_generator.py     # 报告生成
│   │   ├── templates/
│   │   │   ├── full_report.jinja2
│   │   │   ├── summary.jinja2
│   │   │   └── risk_alert.jinja2
│   │   ├── visualizations/
│   │   │   ├── price_charts.py
│   │   │   ├── valuation_charts.py
│   │   │   └── risk_dashboard.py
│   │   └── exporters/
│   │       ├── pdf_exporter.py
│   │       ├── html_exporter.py
│   │       └── notion_exporter.py
│   │
│   ├── risk/                        # 风险管理（贯穿全局）
│   │   ├── __init__.py
│   │   ├── confidence_scorer.py    # 置信度评分
│   │   ├── uncertainty_tracker.py  # 不确定性追踪
│   │   ├── bias_detector.py        # 偏见检测
│   │   ├── data_quality_guard.py   # 数据质量守卫
│   │   └── rules_engine.py         # 风险规则引擎
│   │
│   └── api/                         # API 层
│       ├── __init__.py
│       ├── rest/
│       │   ├── app.py
│       │   └── routes/
│       ├── mcp/                     # MCP 协议支持
│       │   └── server.py
│       └── cli/
│           └── main.py
│
├── plugins/                         # 插件系统
│   ├── data_providers/             # 自定义数据源
│   ├── agents/                     # 自定义 Agent
│   ├── methodologies/              # 自定义方法论
│   └── exporters/                  # 自定义导出器
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── backtests/                  # 策略回测
│
├── docker/
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── docker-compose.dev.yml
│
├── docs/
│   ├── getting_started.md
│   ├── methodology_guide.md        # 如何自定义方法论
│   ├── agent_development.md        # 如何开发 Agent
│   └── api_reference.md
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 四、核心模块详解

### 4.1 方法论配置示例

**DCF 估值方法配置** (`config/methodologies/valuation/dcf.yaml`)

```yaml
name: "DCF Valuation"
version: "1.0"
description: "Discounted Cash Flow valuation methodology"

inputs:
  required:
    - free_cash_flow_history      # 历史 FCF
    - revenue_growth_estimates    # 收入增长预估
    - operating_margin_estimates  # 营业利润率预估
  optional:
    - analyst_estimates          # 分析师预测
    - management_guidance        # 管理层指引

parameters:
  projection_years: 10
  terminal_growth_rate:
    default: 0.025
    min: 0.01
    max: 0.04
    description: "永续增长率，通常为 GDP 长期增长率"
  
  discount_rate:
    method: "wacc"  # wacc | capm | custom
    equity_risk_premium: 0.055
    risk_free_rate_source: "10Y_TREASURY"
    
  scenario_weights:
    bull: 0.25
    base: 0.50
    bear: 0.25

calculation_steps:
  - name: "project_fcf"
    function: "calculate_future_fcf"
    inputs: ["free_cash_flow_history", "revenue_growth_estimates"]
    
  - name: "calculate_terminal_value"
    function: "gordon_growth_model"
    inputs: ["final_year_fcf", "terminal_growth_rate", "discount_rate"]
    
  - name: "discount_cash_flows"
    function: "npv_calculation"
    inputs: ["projected_fcf", "terminal_value", "discount_rate"]
    
  - name: "calculate_per_share"
    function: "divide_by_shares"
    inputs: ["enterprise_value", "net_debt", "shares_outstanding"]

sensitivity_analysis:
  variables:
    - name: "discount_rate"
      range: [-0.02, 0.02]
      step: 0.005
    - name: "terminal_growth_rate"
      range: [-0.01, 0.01]
      step: 0.0025

output_format:
  primary: "fair_value_per_share"
  supporting:
    - "enterprise_value"
    - "sensitivity_matrix"
    - "scenario_values"
    - "key_assumptions"

confidence_factors:
  - name: "data_quality"
    weight: 0.3
    checks:
      - "fcf_history_length >= 5"
      - "no_major_data_gaps"
  - name: "business_stability"
    weight: 0.4
    checks:
      - "revenue_volatility < 0.3"
      - "margin_trend_stable"
  - name: "estimate_reliability"
    weight: 0.3
    checks:
      - "analyst_coverage >= 3"
      - "estimate_dispersion < 0.2"

warnings:
  - condition: "fcf_negative_years > 2"
    message: "公司 FCF 多年为负，DCF 可能不适用"
    severity: "high"
  - condition: "terminal_value_pct > 0.7"
    message: "终值占比过高，估值对永续假设敏感"
    severity: "medium"
```

### 4.2 Agent 配置示例

**估值 Agent 配置** (`config/agents/valuation_agent.yaml`)

```yaml
name: "ValuationAgent"
version: "1.0"
description: "负责公司估值分析的专业 Agent"

persona: |
  你是一位拥有 15 年经验的估值分析师，曾在顶级投行工作。
  你擅长 DCF、可比公司、历史估值等多种方法。
  你总是明确说明假设前提，并给出置信区间而非单一数字。
  你对过于乐观或悲观的假设保持警惕。

capabilities:
  - "dcf_valuation"
  - "comparable_analysis"
  - "historical_valuation"
  - "sum_of_parts"
  - "scenario_analysis"

llm_config:
  preferred_model: "claude-sonnet"      # 深度分析用好模型
  fallback_model: "gpt-4o-mini"
  temperature: 0.3                       # 估值需要确定性
  max_tokens: 4000

tools:
  - name: "financial_calculator"
    description: "执行财务计算（NPV、IRR、WACC 等）"
  - name: "data_fetcher"
    description: "获取财务数据"
  - name: "comparable_finder"
    description: "寻找可比公司"
  - name: "chart_generator"
    description: "生成估值图表"

methodologies:
  - "valuation/dcf"
  - "valuation/comparables"
  - "valuation/historical_multiples"

input_requirements:
  required:
    - "ticker_or_company"
  optional:
    - "custom_assumptions"
    - "peer_group"
    - "target_date"

output_schema:
  valuation_summary:
    fair_value_range: [float, float]
    primary_method: string
    confidence_score: float
  
  method_results:
    - method: string
      value: float
      weight: float
      key_assumptions: list
  
  risk_factors:
    - factor: string
      impact: string
      probability: string
  
  recommendation:
    stance: enum[undervalued, fairly_valued, overvalued]
    conviction: enum[low, medium, high]
    key_catalysts: list
    key_risks: list

guardrails:
  - "永远不给出精确的目标价，只给范围"
  - "必须说明关键假设"
  - "必须提供多种方法交叉验证"
  - "对周期性行业，必须考虑周期位置"
  - "对高增长公司，必须讨论终值敏感性"

collaboration:
  depends_on:
    - "MacroAgent"      # 需要宏观环境判断
    - "RiskAgent"       # 需要风险评估
  provides_to:
    - "StrategyAgent"   # 为策略提供估值输入
```

### 4.3 分析链配置示例

**完整股票分析链** (`config/chains/full_analysis.yaml`)

```yaml
name: "FullStockAnalysis"
description: "对单只股票进行全面深度分析"
version: "1.0"

# 分析链 DAG 定义
stages:
  # Stage 1: 并行数据收集
  - name: "data_collection"
    parallel: true
    tasks:
      - agent: "DataCollector"
        action: "fetch_market_data"
        output_key: "market_data"
        
      - agent: "DataCollector"
        action: "fetch_financials"
        output_key: "financial_data"
        
      - agent: "DataCollector"
        action: "fetch_news"
        params:
          lookback_days: 30
        output_key: "news_data"

  # Stage 2: 并行初步分析
  - name: "initial_analysis"
    parallel: true
    depends_on: ["data_collection"]
    tasks:
      - agent: "MacroAgent"
        action: "analyze_environment"
        inputs: ["market_data"]
        output_key: "macro_view"
        
      - agent: "TechnicalAgent"
        action: "analyze_price_action"
        inputs: ["market_data"]
        output_key: "technical_view"
        
      - agent: "SentimentAgent"
        action: "analyze_sentiment"
        inputs: ["news_data"]
        output_key: "sentiment_view"

  # Stage 3: 深度分析（需要前置结果）
  - name: "deep_analysis"
    parallel: true
    depends_on: ["initial_analysis"]
    tasks:
      - agent: "ValuationAgent"
        action: "comprehensive_valuation"
        inputs: ["financial_data", "macro_view"]
        output_key: "valuation_view"
        
      - agent: "EarningsAgent"
        action: "analyze_fundamentals"
        inputs: ["financial_data"]
        output_key: "fundamental_view"

  # Stage 4: 风险评估（需要所有分析结果）
  - name: "risk_assessment"
    depends_on: ["deep_analysis"]
    tasks:
      - agent: "RiskAgent"
        action: "comprehensive_risk_assessment"
        inputs: 
          - "valuation_view"
          - "technical_view"
          - "sentiment_view"
          - "macro_view"
        output_key: "risk_view"

  # Stage 5: 策略综合
  - name: "strategy_synthesis"
    depends_on: ["risk_assessment"]
    tasks:
      - agent: "StrategyAgent"
        action: "synthesize_recommendation"
        inputs:
          - "valuation_view"
          - "technical_view"
          - "sentiment_view"
          - "fundamental_view"
          - "risk_view"
          - "macro_view"
        output_key: "final_recommendation"

# 冲突解决策略
conflict_resolution:
  method: "weighted_vote"
  weights:
    ValuationAgent: 0.30
    TechnicalAgent: 0.15
    SentimentAgent: 0.10
    EarningsAgent: 0.25
    RiskAgent: 0.20

# 输出配置
output:
  format: "structured_report"
  sections:
    - "executive_summary"
    - "valuation_analysis"
    - "fundamental_analysis"
    - "technical_analysis"
    - "sentiment_analysis"
    - "risk_assessment"
    - "recommendation"
    - "appendix"
  
  required_disclaimers:
    - "investment_risk"
    - "data_limitations"
    - "ai_limitations"

# 质量门控
quality_gates:
  - name: "data_completeness"
    threshold: 0.8
    action: "warn"
    
  - name: "agent_agreement"
    threshold: 0.6
    action: "highlight_disagreement"
    
  - name: "confidence_score"
    threshold: 0.5
    action: "add_low_confidence_warning"

# 超时配置
timeouts:
  stage_timeout: 60  # 单阶段超时（秒）
  total_timeout: 300 # 总超时
  
# 重试策略
retry:
  max_retries: 2
  backoff: "exponential"
```

---

## 五、核心代码实现

### 5.1 Agent 基类

```python
# src/agents/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import yaml

from src.llm.gateway import LLMGateway
from src.core.context import AnalysisContext
from src.risk.confidence_scorer import ConfidenceScorer


class Confidence(Enum):
    VERY_LOW = 0.2
    LOW = 0.4
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


@dataclass
class AgentOutput:
    """Agent 输出的标准格式"""
    agent_name: str
    action: str
    result: Dict[str, Any]
    confidence: float
    reasoning_chain: List[str]           # 推理链路
    data_sources: List[str]              # 数据来源
    assumptions: List[str]               # 关键假设
    uncertainties: List[str]             # 不确定性
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """所有 Agent 的基类"""
    
    def __init__(
        self,
        config_path: str,
        llm_gateway: LLMGateway,
        confidence_scorer: ConfidenceScorer
    ):
        self.config = self._load_config(config_path)
        self.llm = llm_gateway
        self.scorer = confidence_scorer
        self.name = self.config['name']
        self.persona = self.config.get('persona', '')
        self.guardrails = self.config.get('guardrails', [])
        
    def _load_config(self, path: str) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _build_system_prompt(self, context: AnalysisContext) -> str:
        """构建系统提示词"""
        base_prompt = f"""
{self.persona}

## 当前分析上下文
- 分析标的: {context.target}
- 分析日期: {context.analysis_date}
- 用户偏好: {context.user_preferences}

## 输出要求
1. 所有结论必须有数据支撑
2. 必须说明关键假设
3. 必须标明置信度
4. 必须列出不确定性因素

## 行为准则
{chr(10).join(f'- {g}' for g in self.guardrails)}
"""
        return base_prompt
    
    @abstractmethod
    async def analyze(
        self, 
        context: AnalysisContext,
        inputs: Dict[str, Any]
    ) -> AgentOutput:
        """执行分析，子类必须实现"""
        pass
    
    async def _call_llm(
        self,
        context: AnalysisContext,
        user_prompt: str,
        **kwargs
    ) -> str:
        """调用 LLM"""
        system_prompt = self._build_system_prompt(context)
        
        response = await self.llm.complete(
            model=self.config['llm_config'].get('preferred_model'),
            system=system_prompt,
            user=user_prompt,
            temperature=self.config['llm_config'].get('temperature', 0.5),
            **kwargs
        )
        return response
    
    def _calculate_confidence(
        self,
        data_quality: float,
        reasoning_strength: float,
        external_validation: float
    ) -> float:
        """计算置信度"""
        return self.scorer.calculate(
            data_quality=data_quality,
            reasoning_strength=reasoning_strength,
            external_validation=external_validation,
            agent_config=self.config.get('confidence_factors', [])
        )
    
    def _apply_guardrails(self, output: Dict) -> Dict:
        """应用护栏规则"""
        # 检查是否违反了任何护栏规则
        for guardrail in self.guardrails:
            # 实现护栏检查逻辑
            pass
        return output
```

### 5.2 LLM Gateway

```python
# src/llm/gateway.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import asyncio

from src.llm.providers.openai_provider import OpenAIProvider
from src.llm.providers.claude_provider import ClaudeProvider
from src.llm.providers.gemini_provider import GeminiProvider
from src.llm.providers.deepseek_provider import DeepSeekProvider
from src.llm.providers.ollama_provider import OllamaProvider


@dataclass
class LLMResponse:
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    metadata: Dict[str, Any]


class LLMGateway:
    """统一的 LLM 访问网关"""
    
    MODEL_MAPPING = {
        # OpenAI
        'gpt-4o': ('openai', 'gpt-4o'),
        'gpt-4o-mini': ('openai', 'gpt-4o-mini'),
        'gpt-4-turbo': ('openai', 'gpt-4-turbo'),
        
        # Claude
        'claude-opus': ('claude', 'claude-3-opus-20240229'),
        'claude-sonnet': ('claude', 'claude-3-5-sonnet-20241022'),
        'claude-haiku': ('claude', 'claude-3-5-haiku-20241022'),
        
        # Gemini
        'gemini-pro': ('gemini', 'gemini-1.5-pro'),
        'gemini-flash': ('gemini', 'gemini-1.5-flash'),
        
        # DeepSeek
        'deepseek-chat': ('deepseek', 'deepseek-chat'),
        'deepseek-coder': ('deepseek', 'deepseek-coder'),
        
        # Local
        'local-llama': ('ollama', 'llama3.1'),
        'local-qwen': ('ollama', 'qwen2.5'),
    }
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.providers = self._init_providers()
        self.fallback_chain = config.get('fallback_chain', [
            'claude-sonnet', 'gpt-4o', 'gemini-pro'
        ])
        
    def _init_providers(self) -> Dict[str, Any]:
        providers = {}
        
        if self.config.get('openai_api_key'):
            providers['openai'] = OpenAIProvider(self.config['openai_api_key'])
        if self.config.get('anthropic_api_key'):
            providers['claude'] = ClaudeProvider(self.config['anthropic_api_key'])
        if self.config.get('google_api_key'):
            providers['gemini'] = GeminiProvider(self.config['google_api_key'])
        if self.config.get('deepseek_api_key'):
            providers['deepseek'] = DeepSeekProvider(self.config['deepseek_api_key'])
        if self.config.get('ollama_enabled'):
            providers['ollama'] = OllamaProvider(self.config.get('ollama_url'))
            
        return providers
    
    async def complete(
        self,
        model: str,
        system: str,
        user: str,
        temperature: float = 0.7,
        max_tokens: int = 4000,
        **kwargs
    ) -> LLMResponse:
        """统一的补全接口"""
        
        provider_name, model_id = self.MODEL_MAPPING.get(
            model, 
            (model.split('/')[0], model)
        )
        
        provider = self.providers.get(provider_name)
        if not provider:
            # 尝试 fallback
            return await self._fallback_complete(system, user, temperature, max_tokens)
        
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
            # 自动 fallback
            return await self._fallback_complete(system, user, temperature, max_tokens)
    
    async def _fallback_complete(self, system, user, temperature, max_tokens):
        """按 fallback 链尝试"""
        for model in self.fallback_chain:
            provider_name, model_id = self.MODEL_MAPPING.get(model, (None, None))
            if provider_name and provider_name in self.providers:
                try:
                    return await self.providers[provider_name].complete(
                        model=model_id,
                        system=system,
                        user=user,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except:
                    continue
        raise Exception("All LLM providers failed")
    
    def select_model_for_task(self, task_type: str) -> str:
        """根据任务类型智能选择模型"""
        task_model_mapping = {
            'quick_summary': 'gpt-4o-mini',
            'deep_analysis': 'claude-sonnet',
            'code_generation': 'deepseek-coder',
            'batch_processing': 'gemini-flash',
            'creative_writing': 'claude-opus',
        }
        return task_model_mapping.get(task_type, 'claude-sonnet')
```

### 5.3 数据提供者基类

```python
# src/data/base.py
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum


class DataQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class DataPoint:
    """单个数据点"""
    value: Any
    timestamp: datetime
    source: str
    quality: DataQuality
    metadata: Dict[str, Any] = None


@dataclass
class DataResult:
    """数据查询结果"""
    data: Any
    source: str
    fetched_at: datetime
    quality_score: float
    completeness: float
    warnings: List[str]
    metadata: Dict[str, Any]


class DataProvider(ABC):
    """数据提供者基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        pass
    
    @property
    @abstractmethod
    def supported_data_types(self) -> List[str]:
        """支持的数据类型"""
        pass
    
    @abstractmethod
    async def fetch(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        **kwargs
    ) -> DataResult:
        """获取数据"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass
    
    def _assess_quality(self, data: Any) -> DataQuality:
        """评估数据质量"""
        # 子类可以覆盖实现具体逻辑
        return DataQuality.UNKNOWN
    
    def _calculate_completeness(self, data: Any, expected_fields: List[str]) -> float:
        """计算数据完整度"""
        if not data:
            return 0.0
        if isinstance(data, dict):
            present = sum(1 for f in expected_fields if f in data and data[f] is not None)
            return present / len(expected_fields) if expected_fields else 1.0
        return 1.0


class DataProviderRegistry:
    """数据提供者注册中心"""
    
    _providers: Dict[str, DataProvider] = {}
    
    @classmethod
    def register(cls, provider: DataProvider):
        cls._providers[provider.name] = provider
    
    @classmethod
    def get(cls, name: str) -> Optional[DataProvider]:
        return cls._providers.get(name)
    
    @classmethod
    def get_for_data_type(cls, data_type: str) -> List[DataProvider]:
        """获取支持指定数据类型的所有提供者"""
        return [
            p for p in cls._providers.values()
            if data_type in p.supported_data_types
        ]
```

### 5.4 风险管理模块

```python
# src/risk/confidence_scorer.py
from typing import Any, Dict, List
from dataclasses import dataclass


@dataclass
class ConfidenceBreakdown:
    """置信度分解"""
    overall: float
    components: Dict[str, float]
    penalties: List[Dict[str, Any]]
    explanation: str


class ConfidenceScorer:
    """置信度评分器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.base_weights = {
            'data_quality': 0.3,
            'reasoning_strength': 0.4,
            'external_validation': 0.2,
            'historical_accuracy': 0.1
        }
    
    def calculate(
        self,
        data_quality: float,
        reasoning_strength: float,
        external_validation: float,
        agent_config: List[Dict] = None,
        **kwargs
    ) -> ConfidenceBreakdown:
        """
        计算综合置信度
        
        Args:
            data_quality: 数据质量评分 (0-1)
            reasoning_strength: 推理强度评分 (0-1)
            external_validation: 外部验证评分 (0-1)
            agent_config: Agent 特定的置信度配置
        """
        
        # 基础分数
        base_score = (
            data_quality * self.base_weights['data_quality'] +
            reasoning_strength * self.base_weights['reasoning_strength'] +
            external_validation * self.base_weights['external_validation']
        )
        
        # 应用惩罚
        penalties = []
        
        # 数据质量过低惩罚
        if data_quality < 0.5:
            penalty = (0.5 - data_quality) * 0.3
            penalties.append({
                'reason': 'low_data_quality',
                'penalty': penalty,
                'message': '数据质量较低，结论可靠性受限'
            })
            base_score -= penalty
        
        # 缺乏外部验证惩罚
        if external_validation < 0.3:
            penalty = 0.1
            penalties.append({
                'reason': 'no_external_validation',
                'penalty': penalty,
                'message': '缺乏外部数据交叉验证'
            })
            base_score -= penalty
        
        # 应用 Agent 特定规则
        if agent_config:
            for factor in agent_config:
                # 实现 Agent 特定的置信度调整
                pass
        
        # 确保在有效范围内
        final_score = max(0.1, min(0.95, base_score))
        
        return ConfidenceBreakdown(
            overall=final_score,
            components={
                'data_quality': data_quality,
                'reasoning_strength': reasoning_strength,
                'external_validation': external_validation
            },
            penalties=penalties,
            explanation=self._generate_explanation(final_score, penalties)
        )
    
    def _generate_explanation(self, score: float, penalties: List) -> str:
        if score >= 0.8:
            base = "高置信度：数据充分，推理清晰，有外部验证支持"
        elif score >= 0.6:
            base = "中等置信度：整体可靠，但存在一些不确定因素"
        elif score >= 0.4:
            base = "较低置信度：结论仅供参考，需要更多数据验证"
        else:
            base = "低置信度：数据或推理存在明显不足，建议谨慎对待"
        
        if penalties:
            penalty_msgs = [p['message'] for p in penalties]
            base += f"\n注意事项: {'; '.join(penalty_msgs)}"
        
        return base


class UncertaintyTracker:
    """不确定性追踪器"""
    
    def __init__(self):
        self.uncertainties: List[Dict] = []
    
    def add(
        self,
        source: str,
        description: str,
        impact: str,
        mitigations: List[str] = None
    ):
        """添加不确定性"""
        self.uncertainties.append({
            'source': source,
            'description': description,
            'impact': impact,
            'mitigations': mitigations or []
        })
    
    def get_summary(self) -> Dict:
        """获取不确定性摘要"""
        return {
            'total_count': len(self.uncertainties),
            'by_source': self._group_by('source'),
            'high_impact': [u for u in self.uncertainties if u['impact'] == 'high'],
            'items': self.uncertainties
        }
    
    def _group_by(self, key: str) -> Dict:
        result = {}
        for u in self.uncertainties:
            k = u.get(key, 'unknown')
            result[k] = result.get(k, 0) + 1
        return result
```

---

## 六、使用示例

### 6.1 命令行使用

```bash
# 完整分析
finai analyze AAPL --chain full_analysis --output report.pdf

# 快速扫描
finai scan TSLA NVDA MSFT --chain quick_scan

# 财报分析
finai earnings AAPL --quarter 2024Q3

# 板块轮动分析
finai sector --sectors technology healthcare energy

# 自定义分析链
finai analyze AAPL --chain my_custom_chain.yaml
```

### 6.2 Python API 使用

```python
from financeai import FinanceAI, AnalysisConfig

# 初始化
ai = FinanceAI(
    config_path="config/",
    llm_config={
        'anthropic_api_key': 'your-key',
        'openai_api_key': 'your-key'
    }
)

# 执行分析
result = await ai.analyze(
    target="AAPL",
    chain="full_analysis",
    custom_params={
        'valuation': {
            'discount_rate': 0.10,
            'terminal_growth': 0.025
        }
    }
)

# 获取结果
print(result.executive_summary)
print(result.valuation.fair_value_range)
print(result.risk_assessment.key_risks)
print(result.recommendation)

# 导出报告
result.export_pdf("AAPL_analysis.pdf")
result.export_notion(notion_page_id="xxx")
```

### 6.3 自定义方法论

```python
# 创建自定义估值方法
from financeai.methodologies import ValuationMethodology

class MyCustomValuation(ValuationMethodology):
    name = "my_saas_valuation"
    
    def calculate(self, data, params):
        # 自定义 SaaS 公司估值逻辑
        arr = data.get('annual_recurring_revenue')
        growth = data.get('revenue_growth')
        nrr = data.get('net_retention_rate')
        
        # Rule of 40
        rule_of_40 = growth + data.get('free_cash_flow_margin', 0)
        
        # ARR Multiple 根据增长率动态调整
        if growth > 0.5:
            multiple = 15 + (growth - 0.5) * 20
        elif growth > 0.3:
            multiple = 10 + (growth - 0.3) * 25
        else:
            multiple = 5 + growth * 16.67
        
        # NRR 调整
        if nrr > 1.2:
            multiple *= 1.2
        elif nrr < 1.0:
            multiple *= 0.8
        
        return {
            'enterprise_value': arr * multiple,
            'multiple_used': multiple,
            'rule_of_40_score': rule_of_40,
            'assumptions': [
                f'ARR Multiple: {multiple:.1f}x',
                f'Based on {growth*100:.1f}% growth',
                f'NRR adjustment applied: {nrr}'
            ]
        }

# 注册自定义方法
ai.register_methodology(MyCustomValuation())
```

---

## 七、部署架构

```yaml
# docker-compose.yml
version: '3.8'

services:
  financeai-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql://postgres:password@db:5432/financeai
      - REDIS_URL=redis://redis:6379
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - db
      - redis
      - qdrant
    volumes:
      - ./config:/app/config
      - ./plugins:/app/plugins

  db:
    image: timescale/timescaledb:latest-pg15
    environment:
      POSTGRES_PASSWORD: password
      POSTGRES_DB: financeai
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    volumes:
      - redis_data:/data

  qdrant:
    image: qdrant/qdrant:latest
    volumes:
      - qdrant_data:/qdrant/storage

  # 可选：本地 LLM
  ollama:
    image: ollama/ollama:latest
    volumes:
      - ollama_data:/root/.ollama
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  postgres_data:
  redis_data:
  qdrant_data:
  ollama_data:
```

---

## 八、与 FinRobot 的对比

| 维度 | FinRobot | FinanceAI Pro |
|------|----------|---------------|
| **方法论** | 硬编码在代码里 | YAML 配置，可热更新 |
| **数据源** | 固定几个 | 插件化，可自由扩展 |
| **模型支持** | 主要 OpenAI | 全平台支持 + 智能路由 |
| **分析链** | 固定流程 | DAG 可配置，可组合 |
| **风险管理** | 基本没有 | 贯穿全局的置信度系统 |
| **可追溯性** | 只有结论 | 完整推理链路 |
| **扩展性** | 改代码 | 配置 + 插件 |
| **本地部署** | 困难 | Docker 一键部署 |

---

## 九、路线图

### Phase 1: 核心框架 (4 周)
- [ ] 配置加载系统
- [ ] LLM Gateway
- [ ] 基础 Agent 框架
- [ ] 数据提供者框架

### Phase 2: 核心 Agents (6 周)
- [ ] ValuationAgent
- [ ] TechnicalAgent
- [ ] SentimentAgent
- [ ] RiskAgent
- [ ] StrategyAgent

### Phase 3: 分析链 & 编排 (4 周)
- [ ] Chain Builder
- [ ] Orchestrator
- [ ] Conflict Resolution
- [ ] 报告生成

### Phase 4: 数据源集成 (4 周)
- [ ] yfinance / polygon
- [ ] SEC 财报
- [ ] 新闻 API
- [ ] 中国市场数据

### Phase 5: 生产化 (4 周)
- [ ] REST API
- [ ] Web UI
- [ ] Docker 部署
- [ ] 监控 & 告警

---

*这个架构设计的核心是：把"金融分析智慧"从代码中解放出来，变成可配置、可组合、可演进的模块。*
