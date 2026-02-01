"""
FinMind API - FastAPI Backend
提供股票分析、报价等 REST API 接口

Security features:
- SecurityHeadersMiddleware (X-Content-Type-Options, X-Frame-Options, etc.)
- Environment-driven CORS origins
- Global exception handler with error sanitization
- Input validation (symbol regex)
- Config key whitelist / blocklist
- IP-based rate limiting
- Production Swagger disable
"""

import asyncio
import logging
import os
import re
import sys
import time
import uuid
from collections import defaultdict
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core import FinanceAI, get_market_status_summary
from src.core.cache import CacheService, get_cache_service, shutdown_cache_service
from src.core.quote_service import (
    _fetch_single_quote,
    fetch_quotes_batch,
    fetch_price_history,
    search_symbol,
    fetch_volatility,
)
from src.api.broker_routes import router as broker_router
from src.api.analysis_routes import router as analysis_router
from src.brokers.ibkr_flex import import_flex_trades, load_persisted_trades
from src.api.task_store import (
    TaskStore,
    AnalysisStatus,
    AnalysisRequest,
    AnalysisResponse,
    BatchScanRequest,
)
from src.api.models import (
    AnalyzeRequest,
    QuotesRequest,
    TimeframeAnalysis,
    TechnicalAnalysis,
    MarketData,
    PriceHistory,
    AnalysisResult,
    ApiResponse,
    Quote,
    ConfigUpdateRequest,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Symbol validation
# ============================================================================

_SYMBOL_RE = re.compile(r"^[A-Za-z0-9.\-]{1,20}$")


def validate_symbol(symbol: str) -> str:
    """Validate and normalize a ticker symbol."""
    symbol = symbol.strip().upper()
    if not _SYMBOL_RE.match(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format")
    return symbol


# ============================================================================
# Security Headers Middleware
# ============================================================================


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


# ============================================================================
# Rate Limiting
# ============================================================================


class SimpleRateLimiter:
    """Token-bucket style rate limiter keyed by arbitrary string."""

    def __init__(self, max_requests: int, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window = window_seconds
        self._hits: Dict[str, list] = defaultdict(list)

    def is_allowed(self, key: str) -> tuple:
        """Returns (allowed: bool, remaining: int)."""
        now = time.time()
        cutoff = now - self.window
        hits = self._hits[key]
        # Prune old entries
        self._hits[key] = [t for t in hits if t > cutoff]
        hits = self._hits[key]
        if len(hits) >= self.max_requests:
            return False, 0
        hits.append(now)
        return True, self.max_requests - len(hits)


class RateLimitMiddleware(BaseHTTPMiddleware):
    """IP-based global rate limiting middleware."""

    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.limiter = SimpleRateLimiter(requests_per_minute, 60)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        allowed, remaining = self.limiter.is_allowed(client_ip)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"},
                headers={
                    "X-RateLimit-Limit": str(self.limiter.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": "60",
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.limiter.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        return response


# Analysis-specific rate limiter (10 req/min)
_analyze_limiter = SimpleRateLimiter(max_requests=10, window_seconds=60)
# Config POST rate limiter (5 req/min)
_config_limiter = SimpleRateLimiter(max_requests=5, window_seconds=60)


# ============================================================================
# Global state
# ============================================================================

finance_ai: Optional[FinanceAI] = None
task_store = TaskStore()
_start_time = datetime.utcnow()


# ============================================================================
# Lifespan
# ============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
    await task_store.connect_redis(redis_url)
    # Initialize cache service (lazy, will connect on first use)
    await get_cache_service()
    logger.info("FinMind API started")
    yield
    # Shutdown
    await task_store.close_redis()
    await shutdown_cache_service()
    logger.info("FinMind API stopped")


# ============================================================================
# App creation
# ============================================================================

_environment = os.environ.get("ENVIRONMENT", "development")
_is_production = _environment == "production"

_rate_limit_per_minute = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "60"))

app = FastAPI(
    title="FinMind API",
    description="AI-Powered Investment Analysis API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
)

# Middleware order matters: security headers first, then rate limit, then CORS
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(RateLimitMiddleware, requests_per_minute=_rate_limit_per_minute)

_cors_origins_env = os.environ.get("CORS_ORIGINS", "")
_allowed_origins = (
    [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins_env
    else ["http://localhost:3000", "http://127.0.0.1:3000"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register sub-routers
app.include_router(broker_router)
app.include_router(analysis_router)


# ============================================================================
# Global exception handler
# ============================================================================


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(
        f"Unhandled exception on {request.method} {request.url.path}: {exc}",
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ============================================================================
# Helpers
# ============================================================================


def get_finance_ai() -> FinanceAI:
    global finance_ai
    if finance_ai is None:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_dir = os.path.join(project_root, "config")
        finance_ai = FinanceAI(config_dir=config_dir)
    return finance_ai


def format_signal(signal) -> str:
    if hasattr(signal, "value"):
        return signal.value
    return str(signal).replace("SignalStrength.", "").lower()


def format_trend(trend) -> str:
    if hasattr(trend, "value"):
        return trend.value
    return str(trend).replace("TrendDirection.", "").lower()


# ============================================================================
# Root / Health
# ============================================================================


@app.get("/")
async def root():
    return {
        "name": "FinMind API",
        "version": "0.1.0",
        "status": "running",
        "docs": "/docs" if not _is_production else None,
    }


@app.get("/api/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ============================================================================
# Sync Analysis (existing)
# ============================================================================


@app.post("/api/analyze", response_model=ApiResponse)
async def analyze_stock(request: AnalyzeRequest, req: Request = None):
    # Rate limit for analysis
    client_ip = req.client.host if req and req.client else "unknown"
    allowed, _ = _analyze_limiter.is_allowed(client_ip)
    if not allowed:
        return ApiResponse(
            success=False,
            error="Analysis rate limit exceeded (10/min). Please try again later.",
            timestamp=datetime.now().isoformat(),
        )

    symbol = validate_symbol(request.symbol)

    # Check cache
    cache = await get_cache_service()
    cache_key = f"analysis:{symbol}"
    cached = await cache.get(cache_key)
    if cached:
        return ApiResponse(
            success=True,
            data=cached,
            timestamp=datetime.now().isoformat(),
        )

    try:
        ai = get_finance_ai()
        trace_id = str(uuid.uuid4())
        result = await ai.analyze(symbol, trace_id=trace_id)

        data = result.to_dict()
        results = data.get("results", {})

        market_data_raw = results.get("market_data", {})
        if hasattr(market_data_raw, "data"):
            market_data_raw = market_data_raw.data

        tech = results.get("technical_view")

        current_price = market_data_raw.get("current_price", 0)
        previous_close = market_data_raw.get("previous_close") or market_data_raw.get(
            "regular_price"
        )
        price_change = None
        change_percent = None
        if previous_close and current_price:
            price_change = current_price - previous_close
            change_percent = (price_change / previous_close) * 100

        market_data = MarketData(
            current_price=current_price,
            regular_price=market_data_raw.get("regular_price"),
            pre_market_price=market_data_raw.get("pre_market_price"),
            post_market_price=market_data_raw.get("post_market_price"),
            price_source=market_data_raw.get("price_source"),
            market_session=market_data_raw.get("market_session"),
            market_status=market_data_raw.get("market_status"),
            market_cap=market_data_raw.get("market_cap"),
            pe_ratio=market_data_raw.get("trailing_pe"),
            forward_pe=market_data_raw.get("forward_pe"),
            ps_ratio=market_data_raw.get("price_to_sales"),
            pb_ratio=market_data_raw.get("price_to_book"),
            beta=market_data_raw.get("beta"),
            fifty_two_week_high=market_data_raw.get("fifty_two_week_high"),
            fifty_two_week_low=market_data_raw.get("fifty_two_week_low"),
            volume=market_data_raw.get("volume"),
            previous_close=previous_close,
            change=price_change,
            change_percent=change_percent,
        )

        timeframe_analyses = []
        if tech and hasattr(tech, "timeframe_analyses"):
            for tf in tech.timeframe_analyses:
                timeframe_analyses.append(
                    TimeframeAnalysis(
                        timeframe=(
                            tf.timeframe.value
                            if hasattr(tf.timeframe, "value")
                            else str(tf.timeframe)
                        ),
                        timeframe_label=tf.timeframe_label,
                        signal=format_signal(tf.signal),
                        trend=format_trend(tf.trend),
                        trend_strength=tf.trend_strength,
                        confidence=tf.confidence,
                        key_indicators=tf.key_indicators or [],
                        description=tf.description or "",
                    )
                )

        support_levels = []
        resistance_levels = []
        if tech:
            if hasattr(tech, "support_levels") and tech.support_levels:
                support_levels = [
                    s.level if hasattr(s, "level") else s
                    for s in tech.support_levels[:3]
                ]
            if hasattr(tech, "resistance_levels") and tech.resistance_levels:
                resistance_levels = [
                    r.level if hasattr(r, "level") else r
                    for r in tech.resistance_levels[:3]
                ]

        technical_analysis = TechnicalAnalysis(
            overall_signal=format_signal(tech.overall_signal) if tech else "neutral",
            trend=format_trend(tech.trend) if tech else "neutral",
            signal_confidence=tech.signal_confidence if tech else 0.5,
            timeframe_analyses=timeframe_analyses,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
        )

        price_history = None
        if "price_history" in market_data_raw:
            ph = market_data_raw["price_history"]
            if ph and "close" in ph:
                price_history = PriceHistory(
                    dates=ph.get("dates", []),
                    open=ph.get("open", []),
                    high=ph.get("high", []),
                    low=ph.get("low", []),
                    close=ph.get("close", []),
                    volume=[int(v) for v in ph.get("volume", [])],
                )

        analysis_result = AnalysisResult(
            symbol=symbol,
            timestamp=datetime.now().isoformat(),
            market_data=market_data,
            price_history=price_history,
            technical_analysis=technical_analysis,
        )

        result_data = analysis_result.model_dump()
        result_data["trace_id"] = trace_id

        # Cache for 5 minutes
        await cache.set(cache_key, result_data, ttl=300)

        return ApiResponse(
            success=True,
            data=result_data,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Analysis failed for {symbol}: {e}", exc_info=True)
        return ApiResponse(
            success=False,
            error="Analysis failed. Please check the symbol and try again.",
            timestamp=datetime.now().isoformat(),
        )


# ============================================================================
# Quote endpoints (with caching)
# ============================================================================


@app.get("/api/quote/{symbol}", response_model=ApiResponse)
async def get_quote(symbol: str):
    symbol = validate_symbol(symbol)

    cache = await get_cache_service()
    cache_key = f"quote:{symbol}"
    cached = await cache.get(cache_key)
    if cached:
        return ApiResponse(
            success=True,
            data=cached,
            timestamp=datetime.now().isoformat(),
        )

    try:
        quote_data = _fetch_single_quote(symbol)
        if not quote_data:
            return ApiResponse(
                success=False,
                error=f"Unable to fetch quote for {symbol}",
                timestamp=datetime.now().isoformat(),
            )

        await cache.set(cache_key, quote_data, ttl=60)

        return ApiResponse(
            success=True,
            data=quote_data,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Quote fetch failed for {symbol}: {e}", exc_info=True)
        return ApiResponse(
            success=False,
            error="Failed to fetch quote data",
            timestamp=datetime.now().isoformat(),
        )


@app.post("/api/quotes", response_model=ApiResponse)
async def get_quotes(request: QuotesRequest):
    if not request.symbols:
        return ApiResponse(
            success=False,
            error="No symbols provided",
            timestamp=datetime.now().isoformat(),
        )

    # Validate all symbols
    validated = []
    for s in request.symbols:
        validated.append(validate_symbol(s))

    try:
        quotes = fetch_quotes_batch(validated)

        return ApiResponse(
            success=True,
            data=quotes,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Batch quote fetch failed: {e}", exc_info=True)
        return ApiResponse(
            success=False,
            error="Failed to fetch quotes",
            timestamp=datetime.now().isoformat(),
        )


# ============================================================================
# Market Status (with caching)
# ============================================================================


@app.get("/api/market/status", response_model=ApiResponse)
async def get_market_status():
    cache = await get_cache_service()
    cache_key = "market:status"
    cached = await cache.get(cache_key)
    if cached:
        return ApiResponse(
            success=True,
            data=cached,
            timestamp=datetime.now().isoformat(),
        )

    try:
        status = get_market_status_summary()
        data = {"status": status}
        await cache.set(cache_key, data, ttl=30)

        return ApiResponse(
            success=True,
            data=data,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Market status check failed: {e}", exc_info=True)
        return ApiResponse(
            success=False,
            error="Failed to get market status",
            timestamp=datetime.now().isoformat(),
        )


# ============================================================================
# Search endpoint (Phase 7)
# ============================================================================


@app.get("/api/search", response_model=ApiResponse)
async def search_stocks(q: str = Query(..., min_length=1, max_length=20)):
    q = q.strip().upper()
    if not _SYMBOL_RE.match(q):
        return ApiResponse(
            success=True,
            data=[],
            timestamp=datetime.now().isoformat(),
        )

    try:
        results = search_symbol(q)
        return ApiResponse(
            success=True,
            data=results,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Search failed for {q}: {e}", exc_info=True)
        return ApiResponse(
            success=False,
            error="Search failed",
            timestamp=datetime.now().isoformat(),
        )


# ============================================================================
# Price History endpoint (Phase 7)
# ============================================================================


@app.get("/api/history/{symbol}", response_model=ApiResponse)
async def get_history(
    symbol: str,
    period: str = Query(default="1y", regex="^(1d|5d|1mo|3mo|6mo|1y|2y|5y|10y|ytd|max)$"),
):
    symbol = validate_symbol(symbol)

    cache = await get_cache_service()
    cache_key = f"history:{symbol}:{period}"
    cached = await cache.get(cache_key)
    if cached:
        return ApiResponse(success=True, data=cached, timestamp=datetime.now().isoformat())

    try:
        data = fetch_price_history(symbol, period)
        if not data:
            return ApiResponse(
                success=False,
                error=f"No history data for {symbol}",
                timestamp=datetime.now().isoformat(),
            )

        await cache.set(cache_key, data, ttl=300)

        return ApiResponse(
            success=True,
            data=data,
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"History fetch failed for {symbol}: {e}", exc_info=True)
        return ApiResponse(
            success=False,
            error="Failed to fetch price history",
            timestamp=datetime.now().isoformat(),
        )


# ============================================================================
# Config Management (Phase 2: hardened)
# ============================================================================

CONFIG_DEFINITIONS = [
    {"key": "OPENAI_API_KEY", "category": "llm", "description": "OpenAI API 密钥", "is_secret": True},
    {"key": "ANTHROPIC_API_KEY", "category": "llm", "description": "Anthropic (Claude) API 密钥", "is_secret": True},
    {"key": "GOOGLE_API_KEY", "category": "llm", "description": "Google (Gemini) API 密钥", "is_secret": True},
    {"key": "DEEPSEEK_API_KEY", "category": "llm", "description": "DeepSeek API 密钥", "is_secret": True},
    {"key": "POLYGON_API_KEY", "category": "data", "description": "Polygon.io API 密钥 (实时市场数据)", "is_secret": True},
    {"key": "ALPHA_VANTAGE_KEY", "category": "data", "description": "Alpha Vantage API 密钥", "is_secret": True},
    {"key": "NEWS_API_KEY", "category": "data", "description": "News API 密钥", "is_secret": True},
    {"key": "SEC_USER_AGENT", "category": "data", "description": "SEC EDGAR 用户代理 (邮箱)", "is_secret": False},
    {"key": "DATABASE_URL", "category": "database", "description": "PostgreSQL 数据库连接", "is_secret": True},
    {"key": "REDIS_URL", "category": "database", "description": "Redis 缓存连接", "is_secret": True},
    {"key": "QDRANT_URL", "category": "database", "description": "Qdrant 向量数据库", "is_secret": False},
    {"key": "ENVIRONMENT", "category": "app", "description": "运行环境 (development/production)", "is_secret": False},
    {"key": "LOG_LEVEL", "category": "app", "description": "日志级别 (DEBUG/INFO/WARNING/ERROR)", "is_secret": False},
    {"key": "API_PORT", "category": "app", "description": "API 服务端口", "is_secret": False},
    {"key": "OLLAMA_BASE_URL", "category": "local_llm", "description": "Ollama 本地 LLM 地址", "is_secret": False},
    {"key": "TUSHARE_TOKEN", "category": "china", "description": "Tushare A股数据 Token", "is_secret": True},
]

# Phase 2: Writable key whitelist (only keys defined in CONFIG_DEFINITIONS)
_WRITABLE_CONFIG_KEYS = frozenset(d["key"] for d in CONFIG_DEFINITIONS)
# Infrastructure keys that should never be written via API
_BLOCKED_CONFIG_KEYS = frozenset({"DATABASE_URL", "REDIS_URL"})


def get_env_path():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, ".env")


def mask_secret(value: str) -> str:
    if not value or len(value) <= 4:
        return "****"
    return "*" * (len(value) - 4) + value[-4:]


def parse_env_file(env_path: str) -> Dict[str, str]:
    config = {}
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
    return config


@app.get("/api/config", response_model=ApiResponse)
async def get_config():
    try:
        env_path = get_env_path()
        current_config = parse_env_file(env_path)

        configs = []
        for definition in CONFIG_DEFINITIONS:
            key = definition["key"]
            value = current_config.get(key, "")

            configs.append(
                {
                    "key": key,
                    "value": mask_secret(value) if definition["is_secret"] and value else value,
                    "hasValue": bool(value and not value.startswith("your-")),
                    "category": definition["category"],
                    "description": definition["description"],
                    "is_secret": definition["is_secret"],
                }
            )

        return ApiResponse(
            success=True,
            data={"configs": configs},
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Config read failed: {e}", exc_info=True)
        return ApiResponse(
            success=False,
            error="Failed to read configuration",
            timestamp=datetime.now().isoformat(),
        )


@app.post("/api/config", response_model=ApiResponse)
async def update_config(request: ConfigUpdateRequest, req: Request = None):
    # Rate limit
    client_ip = req.client.host if req and req.client else "unknown"
    allowed, _ = _config_limiter.is_allowed(client_ip)
    if not allowed:
        return ApiResponse(
            success=False,
            error="Config update rate limit exceeded (5/min)",
            timestamp=datetime.now().isoformat(),
        )

    try:
        env_path = get_env_path()
        current_config = parse_env_file(env_path)

        for item in request.configs:
            key = item.get("key", "")
            value = item.get("value", "")

            # Whitelist check
            if key not in _WRITABLE_CONFIG_KEYS:
                continue
            # Blocklist check
            if key in _BLOCKED_CONFIG_KEYS:
                continue
            # Value validation
            if not value or value.startswith("*"):
                continue
            if len(value) > 500:
                continue
            if "\n" in value or "\r" in value:
                continue

            current_config[key] = value

        # Read original file preserving comments
        lines = []
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

        updated_keys = set()
        new_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in current_config:
                    new_lines.append(f"{key}={current_config[key]}\n")
                    updated_keys.add(key)
                else:
                    new_lines.append(line)
            else:
                new_lines.append(line)

        for key, value in current_config.items():
            if key not in updated_keys:
                new_lines.append(f"{key}={value}\n")

        with open(env_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        return ApiResponse(
            success=True,
            data={"message": "配置已更新，部分设置可能需要重启服务生效"},
            timestamp=datetime.now().isoformat(),
        )
    except Exception as e:
        logger.error(f"Config update failed: {e}", exc_info=True)
        return ApiResponse(
            success=False,
            error="Failed to update configuration",
            timestamp=datetime.now().isoformat(),
        )


# ============================================================================
# DCF Sensitivity Analysis (Stage 1)
# ============================================================================


class SensitivityRequest(BaseModel):
    symbol: str
    discount_rate: float = Field(default=0.10, ge=0.01, le=0.30)
    growth_rate: float = Field(default=0.08, ge=-0.10, le=0.50)
    terminal_growth: float = Field(default=0.025, ge=0.005, le=0.06)
    projection_years: int = Field(default=10, ge=5, le=15)


# Cache for financial base data (symbol -> (data, timestamp))
_sensitivity_data_cache: Dict[str, tuple] = {}
_SENSITIVITY_CACHE_TTL = 300  # 5 minutes


def _fetch_sensitivity_base_data(symbol: str) -> Dict[str, Any]:
    """Fetch financial base data for DCF sensitivity (blocking, cached 5 min)."""
    import yfinance as yf

    now = time.time()
    cached = _sensitivity_data_cache.get(symbol)
    if cached and (now - cached[1]) < _SENSITIVITY_CACHE_TTL:
        return cached[0]

    ticker = yf.Ticker(symbol)
    info = ticker.info

    data = {
        "fcf": info.get("freeCashflow") or 0,
        "net_debt": (info.get("totalDebt") or 0) - (info.get("totalCash") or 0),
        "shares": info.get("sharesOutstanding") or 1,
        "beta": info.get("beta") or 1.0,
        "current_price": info.get("currentPrice") or info.get("regularMarketPrice") or 0,
        "company_name": info.get("shortName") or symbol,
    }
    _sensitivity_data_cache[symbol] = (data, now)
    return data


@app.post("/api/v1/valuation/sensitivity", tags=["Valuation"])
async def sensitivity_analysis(request: SensitivityRequest):
    """
    DCF sensitivity analysis endpoint — pure math, no LLM calls.
    Returns fair value + 5x5 sensitivity matrix.
    Target response time: < 2s.
    """
    symbol = validate_symbol(request.symbol)

    try:
        loop = asyncio.get_running_loop()
        base_data = await loop.run_in_executor(None, _fetch_sensitivity_base_data, symbol)

        fcf = base_data["fcf"]
        if not fcf or fcf <= 0:
            raise HTTPException(
                status_code=400,
                detail=f"No positive free cash flow available for {symbol}",
            )

        from src.agents.valuation_agent import calculate_dcf

        result = calculate_dcf(
            fcf=fcf,
            growth_rate=request.growth_rate,
            discount_rate=request.discount_rate,
            terminal_growth=request.terminal_growth,
            projection_years=request.projection_years,
            net_debt=base_data["net_debt"],
            shares=base_data["shares"],
        )

        current_price = base_data["current_price"]
        fair_value = result["fair_value"]
        upside = ((fair_value - current_price) / current_price * 100) if current_price else 0

        return {
            "symbol": symbol,
            "company_name": base_data["company_name"],
            "current_price": round(current_price, 2),
            "fair_value": round(fair_value, 2),
            "enterprise_value": round(result["enterprise_value"], 2),
            "upside_downside": round(upside, 2),
            "parameters": {
                "discount_rate": request.discount_rate,
                "growth_rate": request.growth_rate,
                "terminal_growth": request.terminal_growth,
                "projection_years": request.projection_years,
            },
            "sensitivity_matrix": result["sensitivity_matrix"],
            "beta": base_data["beta"],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sensitivity analysis failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Sensitivity analysis failed")


# ============================================================================
# LLM Cost Tracking (Stage 4)
# ============================================================================


@app.get("/api/v1/llm/costs", tags=["LLM"])
async def get_llm_costs():
    """Return LLM cost summary from the gateway."""
    try:
        ai = get_finance_ai()
        if hasattr(ai, "llm_gateway") and hasattr(ai.llm_gateway, "get_cost_summary"):
            return ai.llm_gateway.get_cost_summary()
        return {"total_cost": 0, "by_model": {}, "by_task": {}, "request_count": 0}
    except Exception as e:
        logger.error(f"Failed to get LLM costs: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get LLM costs")


@app.get("/api/v1/llm/analytics", tags=["LLM"])
async def get_llm_analytics():
    """Return LLM analytics from the gateway."""
    try:
        ai = get_finance_ai()
        if hasattr(ai, "llm_gateway") and hasattr(ai.llm_gateway, "get_analytics"):
            return ai.llm_gateway.get_analytics()
        return {
            "total_requests": 0,
            "total_tokens": 0,
            "avg_latency": 0,
            "cost_by_model": {},
            "cost_by_task": {},
        }
    except Exception as e:
        logger.error(f"Failed to get LLM analytics: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get LLM analytics")


# ============================================================================
# Quantitative Backtest (Stage 5)
# ============================================================================


class BacktestRequest(BaseModel):
    symbol: str
    backtest_date: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$")
    forward_days: int = Field(default=90, ge=5, le=365)


@app.post("/api/v1/backtest", tags=["Backtest"])
async def run_backtest_endpoint(request: BacktestRequest):
    """
    Quantitative backtest — pure math (technical indicators + DCF), no LLM.
    Compares prediction at backtest_date vs actual outcome over forward_days.
    """
    symbol = validate_symbol(request.symbol)

    try:
        from src.core.backtest import run_backtest

        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            None, run_backtest, symbol, request.backtest_date, request.forward_days
        )

        if "error" in result and not result.get("prediction"):
            raise HTTPException(status_code=400, detail=result["error"])

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Backtest failed for {symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Backtest failed")


# ============================================================================
# Async Analysis endpoints (Phase 3: v1 API)
# ============================================================================


@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def create_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
    req: Request = None,
):
    # Rate limit
    client_ip = req.client.host if req and req.client else "unknown"
    allowed, _ = _analyze_limiter.is_allowed(client_ip)
    if not allowed:
        raise HTTPException(status_code=429, detail="Analysis rate limit exceeded (10/min)")

    task = await task_store.create_task(request)
    background_tasks.add_task(run_analysis, task.task_id, request)
    return task


async def run_analysis(task_id: str, request: AnalysisRequest):
    """Background analysis task connected to real FinanceAI."""
    trace_id = str(uuid.uuid4())
    try:
        await task_store.update_task(
            task_id,
            status=AnalysisStatus.RUNNING,
            started_at=datetime.utcnow(),
            current_stage="initializing",
        )

        ai = get_finance_ai()

        # Map chain parameter
        chain = request.chain if request.chain in ("full_analysis", "quick_scan", "valuation_only") else "full_analysis"

        await task_store.update_task(task_id, current_stage="analyzing", progress=0.2)
        logger.info(f"[trace:{trace_id}] Async analysis started for {request.target}")

        result = await ai.analyze(
            target=request.target,
            chain=chain,
            custom_params=request.parameters or None,
            trace_id=trace_id,
        )

        await task_store.update_task(
            task_id,
            status=AnalysisStatus.COMPLETED,
            completed_at=datetime.utcnow(),
            progress=1.0,
            current_stage="completed",
            result=result.to_dict(),
        )

    except Exception as e:
        logger.error(f"Async analysis failed for {request.target}: {e}", exc_info=True)
        await task_store.update_task(
            task_id,
            status=AnalysisStatus.FAILED,
            error="Analysis failed. Please check the symbol and configuration.",
        )


@app.get("/api/v1/analyze/{task_id}", response_model=AnalysisResponse, tags=["Analysis"])
async def get_analysis(task_id: str):
    task = await task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.delete("/api/v1/analyze/{task_id}", tags=["Analysis"])
async def cancel_analysis(task_id: str):
    task = await task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if task.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed task")
    await task_store.update_task(task_id, status=AnalysisStatus.CANCELLED)
    return {"message": "Task cancelled", "task_id": task_id}


@app.get("/api/v1/tasks", response_model=List[AnalysisResponse], tags=["Analysis"])
async def list_tasks(
    status: Optional[AnalysisStatus] = Query(None),
    limit: int = Query(20, ge=1, le=100),
):
    return await task_store.list_tasks(status=status, limit=limit)


@app.post("/api/v1/scan", tags=["Scanning"])
async def batch_scan(
    request: BatchScanRequest,
    background_tasks: BackgroundTasks,
):
    scan_id = str(uuid.uuid4())
    tasks = []
    for target in request.targets:
        analysis_request = AnalysisRequest(
            target=target,
            chain=request.chain,
            parameters=request.filters,
        )
        task = await task_store.create_task(analysis_request)
        tasks.append(task)
        background_tasks.add_task(run_analysis, task.task_id, analysis_request)
    return {
        "scan_id": scan_id,
        "total_targets": len(request.targets),
        "tasks": [t.task_id for t in tasks],
    }


# ============================================================================
# IBKR Flex Import
# ============================================================================


class FlexImportRequest(BaseModel):
    token: str = Field(..., min_length=1, max_length=200)
    query_id: str = Field(..., min_length=1, max_length=50)
    account_id: str = Field(..., min_length=1, max_length=50)


@app.post("/api/v1/broker/ibkr/flex-import", tags=["Broker"])
async def flex_import(request: FlexImportRequest):
    """
    导入 IBKR Flex 交易记录

    通过 Flex Web Service 获取历史交易记录并存入本地持久化文件。
    数据存入 ~/.finmind/ibkr_trades_{account}.json，与 TWS 适配器共享。
    使用 execId / tradeId 去重。
    """
    try:
        imported = await import_flex_trades(
            token=request.token,
            query_id=request.query_id,
            account_id=request.account_id,
        )

        persisted = load_persisted_trades(request.account_id)

        return {
            "imported": imported,
            "total_persisted": len(persisted),
            "message": f"Successfully imported {imported} new trades",
        }

    except Exception as e:
        logger.error(f"Flex import failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Flex import failed: {str(e)}")


@app.get("/api/v1/analyze/{task_id}/stream", tags=["Analysis"])
async def stream_analysis(task_id: str):
    max_poll_seconds = 3600

    async def event_generator():
        elapsed = 0
        while elapsed < max_poll_seconds:
            task = await task_store.get_task(task_id)
            if not task:
                yield "event: error\ndata: Task not found\n\n"
                break
            yield f"data: {task.model_dump_json()}\n\n"
            if task.status in [
                AnalysisStatus.COMPLETED,
                AnalysisStatus.FAILED,
                AnalysisStatus.CANCELLED,
            ]:
                break
            await asyncio.sleep(1)
            elapsed += 1
        else:
            yield f"event: timeout\ndata: Stream timeout after {max_poll_seconds}s\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ============================================================================
# Startup
# ============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
