"""
FinMind - Shared Pydantic Models

Common API models shared across routers.
"""

from typing import List, Optional, Any
from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    symbol: str


class QuotesRequest(BaseModel):
    symbols: List[str] = Field(..., max_length=20)


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


class ConfigItem(BaseModel):
    key: str
    value: str
    category: str
    description: str
    is_secret: bool = True


class ConfigUpdateRequest(BaseModel):
    configs: List[dict]
