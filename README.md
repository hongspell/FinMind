# FinMind

<div align="center">

**Modular AI-Powered Financial Analysis Platform**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docker](https://img.shields.io/badge/docker-ready-brightgreen.svg)](https://www.docker.com/)

*Configurable methodologies, pluggable data sources, composable analysis chains*

[English](README.md) | [中文](README_zh.md)

</div>

---

## 🎯 Project Vision

FinMind addresses the core pain points of traditional financial AI tools:

| Problem | Traditional Tools | FinMind |
|---------|-------------------|---------------|
| Methodology | Hard-coded in Python | YAML configuration, hot-reload |
| Data Sources | Fixed 3-4 APIs | Plugin system, unlimited extensibility |
| LLM Support | OpenAI only | All major models + smart routing |
| Analysis Flow | Fixed sequence | DAG configuration, composable |
| Risk Management | Minimal | Full-chain confidence system |
| Traceability | Conclusions only | Complete reasoning chain |

## ✨ Core Features

### 🔧 Configuration-Driven Architecture

```yaml
# config/methodologies/dcf.yaml
methodology_name: "dcf_valuation"
projection_period:
  default_years: 5
terminal_value:
  method: "gordon_growth"
  terminal_growth_rate:
    default: 0.025
    max: 0.04  # Never exceeds GDP growth
```

### 🤖 Multi-Agent Collaboration

```
┌─────────────────────────────────────────────────────────┐
│                    Strategy Agent                        │
│                 (Decision Synthesis Layer)               │
└─────────────────────────────────────────────────────────┘
         ▲              ▲              ▲              ▲
         │              │              │              │
    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐    ┌────┴────┐
    │Valuation│    │Technical│    │Sentiment│    │  Risk   │
    │  Agent  │    │  Agent  │    │  Agent  │    │ Agent   │
    └─────────┘    └─────────┘    └─────────┘    └─────────┘
```

### 📊 Full-Chain Confidence System

```python
# Every output includes confidence scoring
confidence = ConfidenceScore(
    overall=0.72,
    factors={
        "data_quality": 0.85,
        "completeness": 0.70,
        "reasoning": 0.75,
        "validation": 0.65
    }
)
# Never outputs 100% certain conclusions
```

### 🔌 Pluggable Data Sources

```python
# Register custom data providers
class MyDataProvider(DataProvider):
    async def fetch(self, target, params):
        # Your data fetching logic
        return data

registry.register(MyDataProvider())
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/hongspell/FinMind.git
cd FinMind

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env
# Edit .env to add your API keys
```

### Docker Deployment

```bash
# Start all services (database, cache, API)
docker-compose up -d

# View logs
docker-compose logs -f financeai-api
```

### Development Setup (Recommended)

For local development, you can run services individually:

```bash
# 1. Start database and cache services
make docker-up
# This starts TimescaleDB (PostgreSQL) on port 5432 and Redis on port 6379

# 2. Start API server (in terminal 1)
make api
# API runs at http://localhost:8000

# 3. Start frontend dev server (in terminal 2)
make web
# Frontend runs at http://localhost:5173
```

**Database Management:**

```bash
# Connect to database shell
make db-shell

# Reset database (caution: deletes all data)
make db-reset

# View database logs
docker-compose logs -f timescaledb
```

### Basic Usage

#### Command Line

```bash
# Full analysis (English output, default)
python -m src.main analyze AAPL

# Full analysis (Chinese output)
python -m src.main --lang zh analyze AAPL

# Save report to specific file (Markdown)
python -m src.main analyze TSLA --output ./reports/tesla_report.md

# Save as JSON
python -m src.main analyze AAPL --output ./data/aapl.json

# Quick scan multiple stocks
python -m src.main scan AAPL MSFT GOOGL TSLA

# Valuation analysis only
python -m src.main valuation AAPL --scenarios bull,base,bear

# Start API server
python -m src.main serve --port 8000
```

#### Output Formats

| Format | Command | Description |
|--------|---------|-------------|
| Terminal + Markdown | `analyze AAPL` | Shows summary in terminal, auto-saves full report to `reports/` |
| Markdown only | `analyze AAPL -o report.md` | Saves detailed Markdown report |
| JSON | `analyze AAPL -o data.json` | Saves raw data for programmatic use |

#### Language Support

| Language | Flag | Example |
|----------|------|---------|
| English (default) | `--lang en` or omit | `python -m src.main analyze AAPL` |
| Chinese | `--lang zh` | `python -m src.main --lang zh analyze AAPL` |

#### Sample Output

```
================================================================
  AAPL - Analysis Summary
================================================================

  Current Price: $255.52    Market Cap: $3.78T    P/E Ratio: 34.25

  Technical Analysis:
    Signal: NEUTRAL
    Trend: STRONG BEARISH
    Confidence: 29.7% (Very low reliability, not recommended for decisions)

  Analysis Date: 2026-01-20 02:57:09
================================================================

  Full report saved to: reports/AAPL_2026-01-20.md
================================================================
```

#### Python API

```python
from src.core.data_and_chain import FinanceAI

# Initialize
ai = FinanceAI(config_path="config/")

# Execute analysis
result = await ai.analyze(
    target="AAPL",
    chain="full_analysis",
    custom_params={"scenarios": ["bull", "base", "bear"]}
)

# Access results
print(f"Fair Value: ${result.valuation['fair_value_mid']:.2f}")
print(f"Recommendation: {result.recommendation['action']}")
print(f"Confidence: {result.confidence.overall:.1%}")
```

#### REST API

```bash
# Create analysis task
curl -X POST "http://localhost:8000/api/v1/analyze" \
  -H "Content-Type: application/json" \
  -d '{"target": "AAPL", "chain": "full_analysis"}'

# Check task status
curl "http://localhost:8000/api/v1/analyze/{task_id}"

# Get quick quote
curl "http://localhost:8000/api/v1/quote/AAPL"
```

## 📁 Project Structure

```
FinMind/
├── config/
│   ├── agents/              # Agent behavior configuration
│   │   ├── valuation_agent.yaml
│   │   └── technical_agent.yaml
│   ├── chains/              # Analysis chain DAG definitions
│   │   ├── full_analysis.yaml
│   │   └── quick_scan.yaml
│   ├── methodologies/       # Methodology configuration
│   │   └── dcf.yaml
│   └── prompts/             # Prompt templates
├── src/
│   ├── core/                # Core framework
│   │   ├── base.py          # Base class definitions
│   │   ├── config_loader.py # Configuration loader
│   │   └── data_and_chain.py# Data providers + chain executor
│   ├── llm/                 # LLM gateway
│   │   ├── gateway.py       # Unified interface
│   │   └── providers.py     # Provider implementations
│   ├── agents/              # Agent implementations
│   │   ├── valuation_agent.py
│   │   ├── technical_agent.py
│   │   ├── earnings_agent.py
│   │   ├── sentiment_risk_agent.py
│   │   ├── strategy_agent.py
│   │   ├── macro_agent.py
│   │   └── sector_agent.py
│   ├── api/                 # REST API
│   │   └── main.py
│   └── main.py              # CLI entry point
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## 🧩 Agent Overview

| Agent | Responsibility | Main Output |
|-------|----------------|-------------|
| **ValuationAgent** | DCF, comparable companies, historical valuation | Fair value range, valuation rating |
| **TechnicalAgent** | Trends, indicators, pattern recognition | Technical signals, entry/stop-loss levels |
| **EarningsAgent** | Revenue quality, margins, cash flow | Financial health score |
| **SentimentAgent** | News, social media, analyst opinions | Sentiment score, trend |
| **RiskAgent** | Multi-dimensional risk assessment, stress testing | Risk matrix, scenario analysis |
| **MacroAgent** | Economic cycles, monetary policy, inflation | Macro environment assessment |
| **SectorAgent** | Porter's five forces, competitive landscape, moat | Competitive position rating |
| **StrategyAgent** | Synthesizes all agent outputs | Investment recommendation, action plan |

## ⚙️ Configuration

### Environment Variables

```bash
# .env file
OPENAI_API_KEY=sk-xxx
ANTHROPIC_API_KEY=sk-ant-xxx
GOOGLE_API_KEY=xxx

# Data sources
POLYGON_API_KEY=xxx
ALPHA_VANTAGE_KEY=xxx

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/financeai
REDIS_URL=redis://localhost:6379
```

See [.env.example](.env.example) for all available configuration options with pricing information.

### LLM Routing Configuration

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

## 📈 Analysis Chain Example

### Full Analysis Chain

```
Stage 1: Data Collection (Parallel)
├── fetch_market_data
├── fetch_financials
├── fetch_news
└── fetch_analyst_data

Stage 2: Initial Analysis (Parallel)
├── MacroAgent
├── TechnicalAgent
├── SentimentAgent
└── SectorAgent

Stage 3: Deep Analysis (Parallel)
├── ValuationAgent (DCF + Comps)
├── EarningsAgent
└── CompetitiveAgent

Stage 4: Risk Assessment (Sequential)
└── RiskAgent (Comprehensive risk evaluation)

Stage 5: Strategy Synthesis (Sequential)
└── StrategyAgent (Final recommendation)
```

## 📊 Understanding Analysis Results

### Signal Strength (SignalStrength)

| Signal | Meaning | Suggested Action |
|--------|---------|------------------|
| `STRONG_BUY` | Strong Buy | Multiple indicators aligned bullish, consider entering position |
| `BUY` | Buy | Technical indicators lean bullish, consider small position |
| `NEUTRAL` | Neutral | Direction unclear, recommend watching |
| `SELL` | Sell | Technical indicators lean bearish, consider reducing position |
| `STRONG_SELL` | Strong Sell | Multiple indicators aligned bearish, recommend exiting |

### Trend Direction (TrendDirection)

| Trend | Meaning | Description |
|-------|---------|-------------|
| `STRONG_BULLISH` | Strong Uptrend | Sustained price increase, moving averages in bullish alignment |
| `BULLISH` | Uptrend | Generally upward, but moderate strength |
| `SIDEWAYS` | Sideways/Range-bound | No clear direction, price oscillating in range |
| `BEARISH` | Downtrend | Generally downward, but moderate strength |
| `STRONG_BEARISH` | Strong Downtrend | Sustained price decrease, moving averages in bearish alignment |

### Confidence Score

| Confidence Range | Reliability | Investment Advice |
|-----------------|-------------|-------------------|
| **70%+** | High | Can be used as important reference |
| **50-70%** | Medium | Should combine with other factors |
| **40-50%** | Low | Use with caution, signals unclear |
| **<40%** | Very Low | Not recommended for decisions, market direction confused |

> **Note**: Low confidence typically means technical indicators are contradicting each other, or the market is at a turning point. Even if the signal shows "BUY", if confidence is below 40%, exercise caution.

### Common Combination Interpretations

| Signal | Trend | Confidence | Interpretation |
|--------|-------|------------|----------------|
| BUY | STRONG_BULLISH | 70%+ | ✅ Strong buying opportunity |
| BUY | STRONG_BULLISH | <40% | ⚠️ Conflicting signals, may be bounce not reversal |
| NEUTRAL | STRONG_BEARISH | 50%+ | In downtrend, wait for stabilization |
| SELL | BEARISH | 70%+ | ⚠️ Consider stop-loss or reducing position |

## 🔒 Risk Management Design

1. **Confidence System**: Every output has a 0.1-0.95 confidence score
2. **Uncertainty Tracking**: All assumptions and uncertainties are explicitly marked
3. **Guardrail Rules**: Prevents overconfident statements
4. **Quality Gates**: Blocks output when data quality is insufficient
5. **Disclaimers**: All reports automatically include risk warnings

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_report_generator.py -v

# Generate coverage report
pytest tests/ -v --cov=src --cov-report=html
```

## 🔗 Broker Integration

FinMind supports integration with multiple brokerages for personalized portfolio analysis:

| Broker | API | Status | Features |
|--------|-----|--------|----------|
| **IBKR** (盈透证券) | TWS API | ✅ Ready | Portfolio, Positions, Balance |
| **Futu** (富途证券) | OpenD API | ✅ Ready | Portfolio, Positions, Balance |
| **Tiger** (老虎证券) | Tiger Open API | ✅ Ready | Portfolio, Positions, Balance |

### Web UI Setup

1. Navigate to **Settings** page
2. Find **Broker Connections** section
3. Click **Connect** on your broker
4. Enter connection details (host, port, credentials)
5. View your portfolio at `/portfolio`

You can also enable **Demo Mode** to test with sample data without connecting a real broker.

### API Setup

```bash
# IBKR: Run IB Gateway and enable API
# Futu: Run OpenD and login
# Tiger: Register app at developer portal

# Connect via API
curl -X POST "http://localhost:8000/api/v1/broker/connect" \
  -H "Content-Type: application/json" \
  -d '{"broker_type": "ibkr", "ibkr_port": 4001}'

# Get unified portfolio
curl "http://localhost:8000/api/v1/broker/unified"
```

### Position-Aware Analysis

```python
from src.core.portfolio_analysis import PortfolioAnalyzer

analyzer = PortfolioAnalyzer()
result = analyzer.analyze(portfolio_summary)

print(f"Health Score: {result.health_score}/100")
print(f"Risk Score: {result.risk_score}/100")
for rec in result.recommendations:
    print(f"{rec.symbol}: {rec.action} - {rec.reason}")
```

## 📊 Advanced Features

### Monte Carlo Simulation

```python
from src.core.monte_carlo import MonteCarloSimulator

simulator = MonteCarloSimulator()

# Single stock simulation
result = simulator.simulate_price(
    symbol="AAPL",
    current_price=175.0,
    annual_return=0.10,
    annual_volatility=0.25,
)
print(f"95% VaR: ${result.var_values[0.95]:.2f}")

# Portfolio VaR
portfolio_result = simulator.simulate_portfolio(holdings)
print(f"Sharpe Ratio: {portfolio_result.sharpe_ratio:.2f}")
```

### Redis Caching

```python
from src.core.cache import CacheService

cache = CacheService()
await cache.initialize()

# Cached function
@cache.cached(ttl=300, key_prefix="stock:")
async def get_stock_data(symbol: str):
    return await fetch_from_api(symbol)
```

## 🖥️ Web UI Features

The web interface provides a complete portfolio management and analysis experience:

### Pages

| Page | Path | Description |
|------|------|-------------|
| Dashboard | `/` | Quick stock search, trending stocks |
| Analysis | `/analysis/:symbol` | Technical analysis with multi-timeframe signals |
| Portfolio | `/portfolio` | Unified portfolio view, health scores, risk metrics |
| Watchlist | `/watchlist` | Track your favorite stocks |
| Settings | `/settings` | Broker connections, API keys, preferences |

### Risk Analysis Features

- **Monte Carlo Simulation**: Price path visualization with configurable time horizons
- **VaR/CVaR**: Value at Risk at 95% and 99% confidence levels
- **Portfolio Scores**: Health (0-100), Risk (0-100), Diversification (0-100)
- **Position Recommendations**: AI-driven buy/hold/sell suggestions

## 🛣️ Roadmap

- [x] Core framework
- [x] LLM gateway
- [x] Basic Agents (Valuation, Technical, Earnings)
- [x] Analysis chain executor
- [x] REST API
- [x] CLI tool
- [x] Bilingual support (English/Chinese)
- [x] Web UI (React + Ant Design)
- [x] Broker Integration (IBKR, Futu, Tiger)
- [x] Redis Caching Layer
- [x] Monte Carlo Simulation
- [x] Portfolio Context Analysis
- [x] Portfolio Management UI
- [x] Risk Analysis Charts
- [ ] Real-time data streaming
- [ ] Backtesting framework
- [ ] MCP Server integration

## 📄 License

MIT License - See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## ⚠️ Disclaimer

This tool is for research and educational purposes only. It does not constitute investment advice. Investing involves risk; decisions should be made carefully. The authors are not responsible for any investment losses.

---

<div align="center">

**Built with ❤️ for the financial analysis community**

</div>
