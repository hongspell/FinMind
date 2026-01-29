"""
FinMind API - FastAPI Backend
提供股票分析、报价等 REST API 接口
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import asyncio
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import FinanceAI, get_market_status_summary
from src.api.broker_routes import router as broker_router
from src.api.analysis_routes import router as analysis_router

app = FastAPI(
    title="FinMind API",
    description="AI-Powered Investment Analysis API",
    version="0.1.0",
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(broker_router)
app.include_router(analysis_router)

# 全局 FinanceAI 实例
finance_ai: Optional[FinanceAI] = None


# ============ Pydantic Models ============

class AnalyzeRequest(BaseModel):
    symbol: str


class QuotesRequest(BaseModel):
    symbols: List[str]


class TimeframeAnalysis(BaseModel):
    timeframe: str
    timeframe_label: str
    signal: str
    trend: str
    trend_strength: float
    confidence: float
    key_indicators: List[str]
    description: str


class TechnicalAnalysis(BaseModel):
    overall_signal: str
    trend: str
    signal_confidence: float
    timeframe_analyses: List[TimeframeAnalysis]
    support_levels: List[float]
    resistance_levels: List[float]


class MarketData(BaseModel):
    current_price: float
    regular_price: Optional[float] = None
    pre_market_price: Optional[float] = None
    post_market_price: Optional[float] = None
    price_source: Optional[str] = None
    market_session: Optional[str] = None
    market_status: Optional[str] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    ps_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    beta: Optional[float] = None
    fifty_two_week_high: Optional[float] = None
    fifty_two_week_low: Optional[float] = None
    volume: Optional[int] = None
    previous_close: Optional[float] = None
    change: Optional[float] = None
    change_percent: Optional[float] = None


class PriceHistory(BaseModel):
    dates: List[str]
    open: List[float]
    high: List[float]
    low: List[float]
    close: List[float]
    volume: List[int]


class AnalysisResult(BaseModel):
    symbol: str
    timestamp: str
    market_data: MarketData
    price_history: Optional[PriceHistory] = None
    technical_analysis: TechnicalAnalysis


class ApiResponse(BaseModel):
    success: bool
    data: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str


class Quote(BaseModel):
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: str


# ============ Helper Functions ============

def get_finance_ai() -> FinanceAI:
    """获取或创建 FinanceAI 实例"""
    global finance_ai
    if finance_ai is None:
        # 配置目录相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(project_root, 'config')
        finance_ai = FinanceAI(config_dir=config_dir)
    return finance_ai


def format_signal(signal) -> str:
    """格式化信号枚举为字符串"""
    if hasattr(signal, 'value'):
        return signal.value
    return str(signal).replace('SignalStrength.', '').lower()


def format_trend(trend) -> str:
    """格式化趋势枚举为字符串"""
    if hasattr(trend, 'value'):
        return trend.value
    return str(trend).replace('TrendDirection.', '').lower()


# ============ API Endpoints ============

@app.get("/")
async def root():
    """API 根路径"""
    return {
        "name": "FinMind API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs",
    }


@app.post("/api/analyze", response_model=ApiResponse)
async def analyze_stock(request: AnalyzeRequest):
    """
    执行完整股票分析

    - 技术分析（多时间框架）
    - 市场数据
    - 价格历史
    """
    try:
        symbol = request.symbol.upper()
        ai = get_finance_ai()

        # 执行分析
        result = await ai.analyze(symbol)

        # 提取数据
        data = result.to_dict()
        results = data.get('results', {})

        # 获取市场数据
        market_data_raw = results.get('market_data', {})
        if hasattr(market_data_raw, 'data'):
            market_data_raw = market_data_raw.data

        # 获取技术分析
        tech = results.get('technical_view')

        # 计算价格变动
        current_price = market_data_raw.get('current_price', 0)
        previous_close = market_data_raw.get('previous_close') or market_data_raw.get('regular_price')
        price_change = None
        change_percent = None
        if previous_close and current_price:
            price_change = current_price - previous_close
            change_percent = (price_change / previous_close) * 100

        # 构建响应
        market_data = MarketData(
            current_price=current_price,
            regular_price=market_data_raw.get('regular_price'),
            pre_market_price=market_data_raw.get('pre_market_price'),
            post_market_price=market_data_raw.get('post_market_price'),
            price_source=market_data_raw.get('price_source'),
            market_session=market_data_raw.get('market_session'),
            market_status=market_data_raw.get('market_status'),
            market_cap=market_data_raw.get('market_cap'),
            pe_ratio=market_data_raw.get('trailing_pe'),
            forward_pe=market_data_raw.get('forward_pe'),
            ps_ratio=market_data_raw.get('price_to_sales'),
            pb_ratio=market_data_raw.get('price_to_book'),
            beta=market_data_raw.get('beta'),
            fifty_two_week_high=market_data_raw.get('fifty_two_week_high'),
            fifty_two_week_low=market_data_raw.get('fifty_two_week_low'),
            volume=market_data_raw.get('volume'),
            previous_close=previous_close,
            change=price_change,
            change_percent=change_percent,
        )

        # 构建技术分析
        timeframe_analyses = []
        if tech and hasattr(tech, 'timeframe_analyses'):
            for tf in tech.timeframe_analyses:
                timeframe_analyses.append(TimeframeAnalysis(
                    timeframe=tf.timeframe.value if hasattr(tf.timeframe, 'value') else str(tf.timeframe),
                    timeframe_label=tf.timeframe_label,
                    signal=format_signal(tf.signal),
                    trend=format_trend(tf.trend),
                    trend_strength=tf.trend_strength,
                    confidence=tf.confidence,
                    key_indicators=tf.key_indicators or [],
                    description=tf.description or '',
                ))

        # 获取支撑/阻力位
        support_levels = []
        resistance_levels = []
        if tech:
            if hasattr(tech, 'support_levels') and tech.support_levels:
                support_levels = [s.level if hasattr(s, 'level') else s for s in tech.support_levels[:3]]
            if hasattr(tech, 'resistance_levels') and tech.resistance_levels:
                resistance_levels = [r.level if hasattr(r, 'level') else r for r in tech.resistance_levels[:3]]

        technical_analysis = TechnicalAnalysis(
            overall_signal=format_signal(tech.overall_signal) if tech else 'neutral',
            trend=format_trend(tech.trend) if tech else 'neutral',
            signal_confidence=tech.signal_confidence if tech else 0.5,
            timeframe_analyses=timeframe_analyses,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
        )

        # 获取价格历史
        price_history = None
        if 'price_history' in market_data_raw:
            ph = market_data_raw['price_history']
            if ph and 'close' in ph:
                price_history = PriceHistory(
                    dates=ph.get('dates', []),
                    open=ph.get('open', []),
                    high=ph.get('high', []),
                    low=ph.get('low', []),
                    close=ph.get('close', []),
                    volume=[int(v) for v in ph.get('volume', [])],
                )

        analysis_result = AnalysisResult(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            market_data=market_data,
            price_history=price_history,
            technical_analysis=technical_analysis,
        )

        return ApiResponse(
            success=True,
            data=analysis_result.model_dump(),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat(),
        )


@app.get("/api/quote/{symbol}", response_model=ApiResponse)
async def get_quote(symbol: str):
    """获取单只股票快速报价"""
    try:
        import yfinance as yf

        symbol = symbol.upper()
        ticker = yf.Ticker(symbol)
        info = ticker.info

        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        prev_close = info.get('previousClose', current_price)
        change = current_price - prev_close
        change_percent = (change / prev_close * 100) if prev_close else 0

        quote = Quote(
            symbol=symbol,
            price=current_price,
            change=change,
            change_percent=change_percent,
            volume=info.get('volume', 0),
            timestamp=datetime.now().isoformat(),
        )

        return ApiResponse(
            success=True,
            data=quote.model_dump(),
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat(),
        )


@app.post("/api/quotes", response_model=ApiResponse)
async def get_quotes(request: QuotesRequest):
    """获取多只股票报价 - 使用并行获取优化性能"""
    try:
        import yfinance as yf
        from concurrent.futures import ThreadPoolExecutor, as_completed

        def fetch_quote(symbol: str) -> Optional[Quote]:
            """获取单只股票报价"""
            try:
                ticker = yf.Ticker(symbol.upper())
                info = ticker.info

                current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
                if not current_price:
                    return None

                prev_close = info.get('previousClose', current_price)
                change = current_price - prev_close
                change_percent = (change / prev_close * 100) if prev_close else 0

                return Quote(
                    symbol=symbol.upper(),
                    price=current_price,
                    change=change,
                    change_percent=change_percent,
                    volume=info.get('volume', 0),
                    timestamp=datetime.now().isoformat(),
                )
            except Exception:
                return None

        quotes = []
        # 使用线程池并行获取（最多5个并发）
        with ThreadPoolExecutor(max_workers=min(5, len(request.symbols))) as executor:
            futures = {executor.submit(fetch_quote, symbol): symbol for symbol in request.symbols}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    quotes.append(result)

        # 按原始顺序排序
        symbol_order = {s.upper(): i for i, s in enumerate(request.symbols)}
        quotes.sort(key=lambda q: symbol_order.get(q.symbol, 999))

        return ApiResponse(
            success=True,
            data=[q.model_dump() for q in quotes],
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat(),
        )


@app.get("/api/market/status", response_model=ApiResponse)
async def get_market_status():
    """获取当前市场状态"""
    try:
        status = get_market_status_summary()

        return ApiResponse(
            success=True,
            data={"status": status},
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat(),
        )


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============ 配置管理 API ============

class ConfigItem(BaseModel):
    key: str
    value: str
    category: str
    description: str
    is_secret: bool = True


class ConfigUpdateRequest(BaseModel):
    configs: List[Dict[str, str]]


def get_env_path():
    """获取 .env 文件路径"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, '.env')


def mask_secret(value: str) -> str:
    """隐藏敏感信息，只显示最后4位"""
    if not value or len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


def parse_env_file(env_path: str) -> Dict[str, str]:
    """解析 .env 文件"""
    config = {}
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
    return config


# 配置项定义
CONFIG_DEFINITIONS = [
    # LLM API Keys
    {"key": "OPENAI_API_KEY", "category": "llm", "description": "OpenAI API 密钥", "is_secret": True},
    {"key": "ANTHROPIC_API_KEY", "category": "llm", "description": "Anthropic (Claude) API 密钥", "is_secret": True},
    {"key": "GOOGLE_API_KEY", "category": "llm", "description": "Google (Gemini) API 密钥", "is_secret": True},
    {"key": "DEEPSEEK_API_KEY", "category": "llm", "description": "DeepSeek API 密钥", "is_secret": True},
    # Data APIs
    {"key": "POLYGON_API_KEY", "category": "data", "description": "Polygon.io API 密钥 (实时市场数据)", "is_secret": True},
    {"key": "ALPHA_VANTAGE_KEY", "category": "data", "description": "Alpha Vantage API 密钥", "is_secret": True},
    {"key": "NEWS_API_KEY", "category": "data", "description": "News API 密钥", "is_secret": True},
    {"key": "SEC_USER_AGENT", "category": "data", "description": "SEC EDGAR 用户代理 (邮箱)", "is_secret": False},
    # Database
    {"key": "DATABASE_URL", "category": "database", "description": "PostgreSQL 数据库连接", "is_secret": True},
    {"key": "REDIS_URL", "category": "database", "description": "Redis 缓存连接", "is_secret": True},
    {"key": "QDRANT_URL", "category": "database", "description": "Qdrant 向量数据库", "is_secret": False},
    # App Settings
    {"key": "ENVIRONMENT", "category": "app", "description": "运行环境 (development/production)", "is_secret": False},
    {"key": "LOG_LEVEL", "category": "app", "description": "日志级别 (DEBUG/INFO/WARNING/ERROR)", "is_secret": False},
    {"key": "API_PORT", "category": "app", "description": "API 服务端口", "is_secret": False},
    # Local LLM
    {"key": "OLLAMA_BASE_URL", "category": "local_llm", "description": "Ollama 本地 LLM 地址", "is_secret": False},
    # China Market
    {"key": "TUSHARE_TOKEN", "category": "china", "description": "Tushare A股数据 Token", "is_secret": True},
]


@app.get("/api/config", response_model=ApiResponse)
async def get_config():
    """获取所有配置项（敏感信息会被隐藏）"""
    try:
        env_path = get_env_path()
        current_config = parse_env_file(env_path)

        configs = []
        for definition in CONFIG_DEFINITIONS:
            key = definition["key"]
            value = current_config.get(key, "")

            configs.append({
                "key": key,
                "value": mask_secret(value) if definition["is_secret"] and value else value,
                "hasValue": bool(value and not value.startswith("your-")),
                "category": definition["category"],
                "description": definition["description"],
                "is_secret": definition["is_secret"],
            })

        return ApiResponse(
            success=True,
            data={"configs": configs},
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat(),
        )


@app.post("/api/config", response_model=ApiResponse)
async def update_config(request: ConfigUpdateRequest):
    """更新配置项"""
    try:
        env_path = get_env_path()
        current_config = parse_env_file(env_path)

        # 更新配置
        for item in request.configs:
            key = item.get("key")
            value = item.get("value")
            if key and value and not value.startswith("*"):
                current_config[key] = value

        # 读取原始文件保留注释和格式
        lines = []
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

        # 更新或添加配置
        updated_keys = set()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith('#') and '=' in stripped:
                key = stripped.split('=', 1)[0].strip()
                if key in current_config:
                    new_lines.append(f"{key}={current_config[key]}\n")
                    updated_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        # 添加新配置项
        for key, value in current_config.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}\n")

        # 写回文件
        with open(env_path, 'w', encoding='utf-8') as f:
            f.writelines(new_lines)

        return ApiResponse(
            success=True,
            data={"message": "配置已更新，部分设置可能需要重启服务生效"},
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        return ApiResponse(
            success=False,
            error=str(e),
            timestamp=datetime.now().isoformat(),
        )


# ============ 启动配置 ============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
