# FinMind

<div align="center">

**模块化AI金融分析平台**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

*让金融分析的方法论可配置、数据源可插拔、分析链可组合*

[English](README.md) | [中文](README_zh.md)

</div>

---

## 🎯 项目愿景

FinMind 旨在解决传统金融AI工具的核心痛点：

| 问题 | 传统工具 | FinMind |
|------|----------|---------------|
| 方法论 | 硬编码在Python中 | YAML配置，热更新 |
| 数据源 | 固定3-4个API | 插件系统，无限扩展 |
| LLM支持 | 仅OpenAI | 所有主流模型+智能路由 |
| 分析流程 | 固定顺序 | DAG配置，可组合 |
| 风险管理 | 最小化 | 全链路置信度系统 |
| 可追溯性 | 仅结论 | 完整推理链 |

## ✨ 核心特性

### 🔧 配置驱动架构

```yaml
# config/methodologies/dcf.yaml
methodology_name: "dcf_valuation"
projection_period:
  default_years: 5
terminal_value:
  method: "gordon_growth"
  terminal_growth_rate:
    default: 0.025
    max: 0.04  # 永不超过GDP增长
```

### 🤖 多Agent协作

```
┌─────────────────────────────────────────────────────────┐
│                    Strategy Agent                        │
│                   (综合决策层)                            │
└─────────────────────────────────────────────────────────┘
         ▲              ▲              ▲              ▲
         │              │              │              │
    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
    │Valuation│    │Technical│    │Sentiment│    │  Risk   │
    │  Agent  │    │  Agent  │    │  Agent  │    │ Agent   │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### 📊 全链路置信度

```python
# 每个输出都带有置信度评分
confidence = ConfidenceScore(
    overall=0.72,
    factors={
        "data_quality": 0.85,
        "completeness": 0.70,
        "reasoning": 0.75,
        "validation": 0.65
    }
)
# 永远不会输出100%确定的结论
```

### 🔌 插件化数据源

```python
# 注册自定义数据提供者
class MyDataProvider(DataProvider):
    async def fetch(self, target, params):
        # 您的数据获取逻辑
        return data

registry.register(MyDataProvider())
```

## 🚀 快速开始

### 安装

```bash
# 克隆项目
git clone https://github.com/hongspell/FinMind.git
cd FinMind

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 添加您的API密钥
```

### Docker 部署

```bash
# 使用 Docker Compose 启动完整栈
docker-compose up -d

# 查看日志
docker-compose logs -f financeai-api
```

### 本地开发（推荐）

```bash
# 1. 启动数据库和缓存服务
make docker-up
# 启动 TimescaleDB (PostgreSQL) 端口 5432 和 Redis 端口 6379

# 2. 启动 API 服务器（终端 1）
make api
# API 运行在 http://localhost:8000

# 3. 启动前端开发服务器（终端 2）
make web
# 前端运行在 http://localhost:5173
```

**数据库管理:**

```bash
# 连接数据库
make db-shell

# 重置数据库（注意：会删除所有数据）
make db-reset

# 查看数据库日志
docker-compose logs -f timescaledb
```

### 基本用法

#### 命令行

```bash
# 完整分析（英文输出，默认）
python -m src.main analyze AAPL

# 完整分析（中文输出）
python -m src.main --lang zh analyze AAPL

# 保存报告到指定文件（Markdown）
python -m src.main analyze TSLA --output ./reports/tesla_report.md

# 保存为JSON格式
python -m src.main analyze AAPL --output ./data/aapl.json

# 快速扫描多只股票
python -m src.main scan AAPL MSFT GOOGL TSLA

# 仅估值分析
python -m src.main valuation AAPL --scenarios bull,base,bear

# 启动API服务器
python -m src.main serve --port 8000
```

#### 输出格式

| 格式 | 命令 | 说明 |
|------|------|------|
| 终端 + Markdown | `analyze AAPL` | 终端显示摘要，自动保存完整报告到 `reports/` |
| 仅Markdown | `analyze AAPL -o report.md` | 保存详细Markdown报告 |
| JSON | `analyze AAPL -o data.json` | 保存原始数据供程序使用 |

#### 语言支持

| 语言 | 参数 | 示例 |
|------|------|------|
| 英文（默认） | `--lang en` 或省略 | `python -m src.main analyze AAPL` |
| 中文 | `--lang zh` | `python -m src.main --lang zh analyze AAPL` |

#### 示例输出

```
================================================================
  AAPL - 分析摘要
================================================================

  当前价格: $255.52    市值: $3.78T    市盈率: 34.25

  技术分析:
    信号: 中性
    趋势: 强烈看跌
    置信度: 29.7% (可信度极低，不建议作为决策依据)

  分析日期: 2026-01-20 02:57:09
================================================================

  完整报告已保存至: reports/AAPL_2026-01-20.md
================================================================
```

#### Python API

```python
from src.core.data_and_chain import FinanceAI

# 初始化
ai = FinanceAI(config_path="config/")

# 执行分析
result = await ai.analyze(
    target="AAPL",
    chain="full_analysis",
    custom_params={"scenarios": ["bull", "base", "bear"]}
)

# 访问结果
print(f"公允价值: ${result.valuation['fair_value_mid']:.2f}")
print(f"建议: {result.recommendation['action']}")
print(f"置信度: {result.confidence.overall:.1%}")
```

#### REST API

```bash
# 创建分析任务（异步）
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"target": "AAPL", "chain": "full_analysis"}'

# 查询任务状态
curl "http://localhost:8000/api/v1/analyze/{task_id}"

# 流式获取任务进度 (SSE)
curl "http://localhost:8000/api/v1/analyze/{task_id}/stream"

# 获取快速报价
curl "http://localhost:8000/api/quote/AAPL"

# DCF 敏感性分析
curl -X POST "http://localhost:8000/api/v1/valuation/sensitivity" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "discount_rate": 0.10, "growth_rate": 0.08}'

# 量化回测
curl -X POST "http://localhost:8000/api/v1/backtest" \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "backtest_date": "2025-01-01", "forward_days": 90}'
```

## 📁 项目结构

```
FinMind/
├── api/
│   └── main.py              # FastAPI 应用入口
├── config/
│   ├── agents/              # Agent 行为配置
│   │   ├── valuation_agent.yaml
│   │   ├── technical_agent.yaml
│   │   ├── earnings_agent.yaml
│   │   ├── risk_agent.yaml
│   │   ├── sentiment_agent.yaml
│   │   └── strategy_agent.yaml
│   ├── chains/              # 分析链 DAG 定义
│   │   ├── full_analysis.yaml
│   │   ├── quick_scan.yaml
│   │   └── earnings_deep_dive.yaml
│   ├── methodologies/       # 方法论配置
│   │   ├── dcf.yaml
│   │   └── comparables.yaml
│   └── prompts/             # 提示词模板
│       └── valuation_prompts.yaml
├── src/
│   ├── core/                # 核心框架
│   │   ├── base.py          # 基础类定义
│   │   ├── config_loader.py # 配置加载器
│   │   ├── data_and_chain.py# 数据提供者 + 链执行器
│   │   ├── cache.py         # Redis 缓存层
│   │   ├── backtest.py      # 量化回测引擎
│   │   ├── monte_carlo.py   # 蒙特卡洛模拟
│   │   ├── portfolio_analysis.py  # 投资组合健康度 & 风险评分
│   │   ├── portfolio_tracker.py   # 投资组合跟踪
│   │   ├── quote_service.py # 实时行情服务
│   │   ├── market_hours.py  # 市场交易时段检测
│   │   ├── report_generator.py    # Markdown 报告生成
│   │   └── database.py      # 数据库模型 (TimescaleDB)
│   ├── llm/                 # LLM 网关
│   │   ├── gateway.py       # 统一接口 + 成本追踪
│   │   └── providers.py     # 各模型实现
│   ├── agents/              # Agent 实现
│   │   ├── valuation_agent.py     # DCF、可比公司、历史估值
│   │   ├── technical_agent.py     # 趋势、指标、形态
│   │   ├── earnings_agent.py      # 收入质量、利润率
│   │   ├── sentiment_risk_agent.py# 情绪 + 风险评估
│   │   ├── strategy_agent.py      # 综合决策
│   │   ├── macro_agent.py         # 宏观环境
│   │   └── sector_agent.py        # 行业 & 竞争
│   ├── brokers/             # 券商适配器（只读）
│   │   ├── base.py          # 抽象基类 + 数据模型
│   │   ├── trade_store.py   # 本地交易记录持久化组件
│   │   ├── ibkr.py          # IBKR TWS API 适配器
│   │   ├── ibkr_cpapi.py    # IBKR Client Portal REST 适配器
│   │   ├── ibkr_flex.py     # IBKR Flex Queries（历史交易导入）
│   │   ├── futu.py          # 富途 OpenD 适配器
│   │   ├── tiger.py         # 老虎证券 Open API 适配器
│   │   └── portfolio.py     # 适配器注册 + 工厂
│   ├── api/                 # API 路由模块
│   │   ├── broker_routes.py # 券商连接 & 投资组合端点
│   │   ├── analysis_routes.py # 分析端点
│   │   ├── models.py        # Pydantic 请求/响应模型
│   │   └── task_store.py    # 异步任务管理
│   └── main.py              # CLI 入口
├── web/                     # React 前端 (Vite + Ant Design)
│   └── src/
│       ├── pages/           # 首页、分析、投资组合、设置
│       ├── components/      # 图表、分析面板、布局
│       ├── services/        # API & 券商 API 客户端
│       ├── stores/          # Zustand 状态管理
│       ├── hooks/           # 自定义 React Hooks
│       ├── types/           # TypeScript 类型定义
│       └── styles/          # 主题 & 全局样式
├── tests/                   # 测试套件
├── scripts/                 # 工具脚本 (init-db.sql, start-dev.sh)
├── docker-compose.yml
├── Dockerfile
├── Makefile                 # 开发快捷命令 (make api, make web 等)
└── requirements.txt
```

## 🧩 Agent介绍

| Agent | 职责 | 主要输出 |
|-------|------|----------|
| **ValuationAgent** | DCF、可比公司、历史估值 | 公允价值区间、估值评级 |
| **TechnicalAgent** | 趋势、指标、形态识别 | 技术信号、入场/止损位 |
| **EarningsAgent** | 收入质量、利润率、现金流 | 财务健康评分 |
| **SentimentAgent** | 新闻、社交媒体、分析师 | 情绪评分、舆情趋势 |
| **RiskAgent** | 多维度风险评估、压力测试 | 风险矩阵、情景分析 |
| **MacroAgent** | 经济周期、货币政策、通胀 | 宏观环境评估 |
| **SectorAgent** | 波特五力、竞争格局、护城河 | 竞争地位评级 |
| **StrategyAgent** | 综合所有Agent输出 | 投资建议、行动计划 |

## ⚙️ 配置说明

### 环境变量

```bash
# .env 文件
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=xxx

# 数据源
POLYGON_API_KEY=xxx
ALPHA_VANTAGE_KEY=xxx

# 数据库
DATABASE_URL=postgresql://user:pass@localhost:5432/financeai
REDIS_URL=redis://localhost:6379
```

所有配置选项及定价信息请参阅 [.env.example](.env.example)。

### LLM路由配置

```yaml
# config/llm_config.yaml
routing:
  deep_analysis:
    preferred: "claude-opus"
    fallback: "gpt-4o"
  quick_tasks:
    preferred: "claude-haiku"
    fallback: "gpt-4o-mini"
  cost_sensitive:
    preferred: "deepseek-chat"
    fallback: "ollama/llama3"
```

## 📈 分析链示例

### 完整分析链

```
阶段1: 数据收集（并行）
├── fetch_market_data
├── fetch_financials
├── fetch_news
└── fetch_analyst_data

阶段2: 初步分析（并行）
├── MacroAgent
├── TechnicalAgent
├── SentimentAgent
└── SectorAgent

阶段3: 深度分析（并行）
├── ValuationAgent (DCF + 可比公司)
├── EarningsAgent
└── CompetitiveAgent

阶段4: 风险评估（顺序）
└── RiskAgent (综合风险评估)

阶段5: 策略综合（顺序）
└── StrategyAgent (最终建议)
```

## 📊 分析结果解读

### 信号强度 (SignalStrength)

| 信号 | 含义 | 建议操作 |
|------|------|----------|
| `STRONG_BUY` | 强烈买入 | 多个指标一致看涨，可考虑建仓 |
| `BUY` | 买入 | 技术指标偏向看涨，可考虑小仓位 |
| `NEUTRAL` | 中性 | 方向不明确，建议观望 |
| `SELL` | 卖出 | 技术指标偏向看跌，可考虑减仓 |
| `STRONG_SELL` | 强烈卖出 | 多个指标一致看跌，建议离场 |

### 趋势方向 (TrendDirection)

| 趋势 | 含义 | 说明 |
|------|------|------|
| `STRONG_BULLISH` | 强烈上涨趋势 | 价格持续上升，均线多头排列 |
| `BULLISH` | 上涨趋势 | 整体向上，但力度一般 |
| `SIDEWAYS` | 横盘/震荡 | 无明显方向，价格在区间内波动 |
| `BEARISH` | 下跌趋势 | 整体向下，但力度一般 |
| `STRONG_BEARISH` | 强烈下跌趋势 | 价格持续下降，均线空头排列 |

### 置信度评分

| 置信度区间 | 可信度 | 投资建议 |
|-----------|--------|----------|
| **70%+** | 高 | 可作为重要参考依据 |
| **50-70%** | 中等 | 需结合其他因素综合判断 |
| **40-50%** | 低 | 谨慎参考，信号不够明确 |
| **<40%** | 极低 | 不建议作为决策依据，市场方向混乱 |

> **注意**: 低置信度通常意味着技术指标之间存在矛盾，或者市场正处于转折点。即使信号显示"买入"，如果置信度低于40%，也应谨慎对待。

### 常见组合解读

| 信号 | 趋势 | 置信度 | 解读 |
|------|------|--------|------|
| BUY | STRONG_BULLISH | 70%+ | ✅ 强烈买入机会 |
| BUY | STRONG_BULLISH | <40% | ⚠️ 信号矛盾，可能是反弹非反转 |
| NEUTRAL | STRONG_BEARISH | 50%+ | 处于下跌趋势，等待企稳 |
| SELL | BEARISH | 70%+ | ⚠️ 考虑止损或减仓 |

## 🔒 风险管理设计

1. **置信度系统**: 每个输出都有0.1-0.95的置信度评分
2. **不确定性追踪**: 所有假设和不确定性都被明确标记
3. **护栏规则**: 防止过度自信的陈述
4. **质量门控**: 当数据质量不足时阻止输出
5. **免责声明**: 所有报告自动包含风险警告

## 🧪 测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_report_generator.py -v

# 生成覆盖率报告
pytest tests/ -v --cov=src --cov-report=html
```

## 🔗 券商集成

FinMind 支持多家券商的数据对接，用于个性化投资组合分析。所有适配器均为**只读**（不支持交易），并提供本地交易历史持久化。

| 券商 | API | 状态 | 功能 |
|------|-----|------|------|
| **IBKR** (盈透证券) | TWS API (`ib_insync`) | ✅ 就绪 | 组合、持仓、余额、交易历史 |
| **IBKR** (盈透证券) | Client Portal REST API | ✅ 就绪 | 组合、持仓、余额、交易历史 |
| **IBKR** (盈透证券) | Flex Queries | ✅ 就绪 | 完整历史交易导入 |
| **富途证券** | OpenD API (`futu-api`) | ✅ 就绪 | 组合、持仓、余额、交易历史 |
| **老虎证券** | Tiger Open API (`tigeropen`) | ✅ 就绪 | 组合、持仓、余额、交易历史 |

### 架构设计

- **`BrokerAdapter`** 抽象基类定义统一接口
- **`TradeStore`** 组件处理本地 JSON 持久化，支持可配置的去重字段
- 每个适配器通过组合注入 `TradeStore` 实例 —— 无重复存储逻辑
- 市场/交易所识别通过 `BrokerAdapter._resolve_market()` 统一处理
- 所有适配器都包含 **Mock** 变体，可在无真实连接的情况下进行演示/测试

### Web UI 设置

1. 进入 **设置** 页面
2. 找到 **券商连接** 区域
3. 点击对应券商的 **连接** 按钮
4. 填写连接信息（主机、端口、凭证）
5. 在 `/portfolio` 查看投资组合

也可以启用 **演示模式**，使用模拟数据进行测试，无需连接真实券商。

### API 设置

```bash
# IBKR TWS: 运行 IB Gateway / TWS 并启用 API
# IBKR Client Portal: 运行 CP Gateway，在 https://localhost:5000 登录
# 富途: 运行 OpenD 并登录
# 老虎: 在开发者平台注册应用

# 通过 API 连接
curl -X POST "http://localhost:8000/api/v1/broker/connect" \
  -H "Content-Type: application/json" \
  -d '{"broker_type": "ibkr", "ibkr_port": 4001}'

# 通过 Client Portal API 连接
curl -X POST "http://localhost:8000/api/v1/broker/connect" \
  -H "Content-Type: application/json" \
  -d '{"broker_type": "ibkr_cp"}'

# 获取统一投资组合
curl "http://localhost:8000/api/v1/broker/unified"

# 获取交易历史
curl "http://localhost:8000/api/v1/broker/trades/ibkr"

# 通过 Flex Queries 导入历史交易
curl -X POST "http://localhost:8000/api/v1/broker/ibkr/flex-import" \
  -H "Content-Type: application/json" \
  -d '{"token": "your-flex-token", "query_id": "your-query-id", "account_id": "your-account"}'
```

### 持仓感知分析

```python
from src.core.portfolio_analysis import PortfolioAnalyzer

analyzer = PortfolioAnalyzer()
result = analyzer.analyze(portfolio_summary)

print(f"健康评分: {result.health_score}/100")
print(f"风险评分: {result.risk_score}/100")
for rec in result.recommendations:
    print(f"{rec.symbol}: {rec.action} - {rec.reason}")
```

## 📊 高级功能

### 蒙特卡洛模拟

```python
from src.core.monte_carlo import MonteCarloSimulator

simulator = MonteCarloSimulator()

# 单只股票模拟
result = simulator.simulate_price(
    symbol="AAPL",
    current_price=175.0,
    annual_return=0.10,
    annual_volatility=0.25,
)
print(f"95% VaR: ${result.var_values[0.95]:.2f}")

# 投资组合 VaR
portfolio_result = simulator.simulate_portfolio(holdings)
print(f"夏普比率: {portfolio_result.sharpe_ratio:.2f}")
```

### Redis 缓存

```python
from src.core.cache import CacheService

cache = CacheService()
await cache.initialize()

# 缓存函数
@cache.cached(ttl=300, key_prefix="stock:")
async def get_stock_data(symbol: str):
    return await fetch_from_api(symbol)
```

## 🖥️ Web UI 功能

Web 界面提供完整的投资组合管理和分析体验：

### 页面

| 页面 | 路径 | 说明 |
|------|------|------|
| 首页 | `/` | 快速股票搜索、热门股票 |
| 分析 | `/analysis/:symbol` | 多时间框架技术分析 |
| 投资组合 | `/portfolio` | 统一组合视图、健康评分、风险指标 |
| 自选股 | `/watchlist` | 跟踪关注的股票 |
| 设置 | `/settings` | 券商连接、API 密钥、偏好设置 |

### 风险分析功能

- **蒙特卡洛模拟**: 可配置时间范围的价格路径可视化
- **VaR/CVaR**: 95% 和 99% 置信水平的在险价值
- **投资组合评分**: 健康度 (0-100)、风险 (0-100)、分散度 (0-100)
- **持仓建议**: AI 驱动的买入/持有/卖出建议

## 🛣️ 路线图

- [x] 核心框架
- [x] LLM 网关（多模型 + 成本追踪）
- [x] 基础 Agent (Valuation, Technical, Earnings, Macro, Sector, Strategy)
- [x] 分析链执行器（DAG 驱动）
- [x] REST API (FastAPI + 异步任务管理)
- [x] CLI 工具 (analyze, scan, valuation, serve)
- [x] 双语支持 (中文/英文)
- [x] Web UI (React + Vite + Ant Design)
- [x] 券商集成 (IBKR TWS, IBKR Client Portal, IBKR Flex, 富途, 老虎)
- [x] Redis 缓存层
- [x] 蒙特卡洛模拟 & VaR/CVaR
- [x] 投资组合分析（健康度、风险、分散度评分）
- [x] 投资组合管理 UI（持仓、余额、交易历史）
- [x] 风险分析图表
- [x] 量化回测引擎（技术指标 + DCF）
- [x] DCF 敏感性分析（5x5 矩阵）
- [ ] 实时数据流
- [ ] MCP Server 集成

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 🤝 贡献

欢迎贡献！请阅读我们的 [贡献指南](CONTRIBUTING.md) 了解详情。

1. Fork本项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## ⚠️ 免责声明

本工具仅供研究和教育目的。不构成投资建议。投资有风险，决策需谨慎。作者不对任何投资损失负责。

---

<div align="center">

**为金融分析社区用心打造 ❤️**

</div>
