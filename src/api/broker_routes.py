"""
FinMind - 券商 API 路由

提供券商集成的 REST API 接口。
"""

import logging
import re
from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from ..brokers import (
    BrokerConfig,
    BrokerError,
    AuthenticationError,
)
from ..brokers.portfolio import UnifiedPortfolio, create_broker_adapter

logger = logging.getLogger(__name__)

_SYMBOL_PATTERN = re.compile(r'^[A-Za-z0-9.\-]{1,20}$')
_VALID_BROKER_TYPES = frozenset({"ibkr", "ibkr_cp", "futu", "tiger"})


def validate_symbol(symbol: str) -> str:
    """验证股票代码格式"""
    if not _SYMBOL_PATTERN.match(symbol):
        raise HTTPException(status_code=400, detail="Invalid symbol format")
    return symbol


def validate_broker_type(broker_type: str) -> str:
    """验证券商类型"""
    bt = broker_type.lower()
    if bt not in _VALID_BROKER_TYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported broker type: {broker_type}")
    return bt


def _position_to_response(p) -> 'PositionResponse':
    """将 Position 对象转换为 PositionResponse（消除 3 处重复构造逻辑）"""
    return PositionResponse(
        symbol=p.symbol,
        market=p.market.value,
        quantity=p.quantity,
        avg_cost=p.avg_cost,
        current_price=p.current_price,
        market_value=p.market_value,
        unrealized_pnl=p.unrealized_pnl,
        unrealized_pnl_percent=p.unrealized_pnl_percent,
        realized_pnl=p.realized_pnl,
        side=p.side.value,
        currency=p.currency.value,
        company_name=p.company_name,
        sector=p.sector,
    )


def get_connected_adapter(portfolio: UnifiedPortfolio, broker_type: str):
    """获取已连接的券商适配器，失败时抛出 HTTPException"""
    broker_type = validate_broker_type(broker_type)
    adapter = portfolio.get_adapter(broker_type)

    if not adapter:
        raise HTTPException(status_code=404, detail=f"Broker not found: {broker_type}")

    if not adapter.is_connected:
        raise HTTPException(status_code=400, detail=f"Broker not connected: {broker_type}")

    return adapter


router = APIRouter(prefix="/api/v1/broker", tags=["Broker"])


# =============================================================================
# Pydantic Models
# =============================================================================

class BrokerConfigRequest(BaseModel):
    """券商配置请求"""
    broker_type: str = Field(..., description="券商类型: ibkr, ibkr_cp, futu, tiger")
    # IBKR TWS
    ibkr_host: str = Field(default="127.0.0.1")
    ibkr_port: int = Field(default=4001)
    ibkr_client_id: int = Field(default=1)
    # IBKR Client Portal
    ibkr_cp_base_url: str = Field(default="https://localhost:5000/v1/api")
    # Futu
    futu_host: str = Field(default="127.0.0.1")
    futu_port: int = Field(default=11111)
    futu_trade_password: Optional[str] = None
    # Tiger
    tiger_id: Optional[str] = None
    tiger_account: Optional[str] = None
    tiger_private_key: Optional[str] = None  # 私钥内容（非路径）

    def to_config(self) -> BrokerConfig:
        """转换为 BrokerConfig

        tiger_private_key 是字符串内容，TigerAdapter._read_private_key()
        已支持直接接受字符串内容（检测 -----BEGIN 头标识）。
        """
        return BrokerConfig(
            broker_type=self.broker_type,
            ibkr_host=self.ibkr_host,
            ibkr_port=self.ibkr_port,
            ibkr_client_id=self.ibkr_client_id,
            ibkr_cp_base_url=self.ibkr_cp_base_url,
            futu_host=self.futu_host,
            futu_port=self.futu_port,
            futu_trade_password=self.futu_trade_password,
            tiger_id=self.tiger_id,
            tiger_account=self.tiger_account,
            tiger_private_key_path=self.tiger_private_key,
        )


class PositionResponse(BaseModel):
    """持仓响应"""
    symbol: str
    market: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    realized_pnl: float = 0.0
    side: str
    currency: str
    company_name: Optional[str] = None
    sector: Optional[str] = None


class TradeResponse(BaseModel):
    """交易记录响应"""
    symbol: str
    market: str
    action: str  # buy/sell
    quantity: float
    price: float
    total_value: float
    commission: float = 0.0
    currency: str
    trade_time: Optional[str] = None
    order_id: Optional[str] = None
    execution_id: Optional[str] = None
    realized_pnl: Optional[float] = None


class AccountBalanceResponse(BaseModel):
    """账户余额响应"""
    total_assets: float
    cash: float
    market_value: float
    buying_power: float
    currency: str
    day_pnl: float
    total_pnl: float
    margin_used: float = 0.0
    margin_available: float = 0.0


class BrokerSummaryResponse(BaseModel):
    """券商摘要响应"""
    broker: str
    account_id: str
    balance: AccountBalanceResponse
    positions: List[PositionResponse]
    position_count: int
    top_holdings: List[Dict[str, Any]]
    market_allocation: Dict[str, float]
    currency_exposure: Dict[str, float]
    last_updated: str


class UnifiedSummaryResponse(BaseModel):
    """统一投资组合响应"""
    total_assets: float
    total_cash: float
    total_market_value: float
    total_unrealized_pnl: float
    total_realized_pnl: float
    broker_allocation: Dict[str, float]
    market_allocation: Dict[str, float]
    currency_exposure: Dict[str, float]
    top_holdings: List[Dict[str, Any]]
    broker_count: int
    position_count: int
    last_updated: str


class BrokerStatusResponse(BaseModel):
    """券商状态响应"""
    broker_type: str
    connected: bool
    account_id: Optional[str] = None
    error: Optional[str] = None


# =============================================================================
# 全局状态（生产环境应使用依赖注入）
# =============================================================================

# 统一投资组合管理器
# use_mock=False 表示连接真实券商，use_mock=True 表示使用模拟数据
_portfolio = UnifiedPortfolio(use_mock=False)


def get_portfolio() -> UnifiedPortfolio:
    """获取投资组合管理器"""
    return _portfolio


# =============================================================================
# API 端点
# =============================================================================

@router.get("/supported", summary="获取支持的券商列表")
async def get_supported_brokers():
    """获取支持的券商列表及配置说明"""
    return {
        "brokers": [
            {
                "type": "ibkr",
                "name": "盈透证券 (Interactive Brokers) - TWS API",
                "description": "使用 TWS API 连接到 IB Gateway 或 TWS",
                "required_fields": ["ibkr_host", "ibkr_port", "ibkr_client_id"],
                "setup_guide": "需要运行 IB Gateway 并启用 API",
            },
            {
                "type": "ibkr_cp",
                "name": "盈透证券 (Interactive Brokers) - Client Portal",
                "description": "使用 Client Portal REST API 连接到 IBKR Gateway",
                "required_fields": ["ibkr_cp_base_url"],
                "setup_guide": (
                    "需要运行 Client Portal Gateway 并在浏览器完成登录。"
                    "下载地址: https://www.interactivebrokers.com/en/trading/ib-api.php"
                ),
            },
            {
                "type": "futu",
                "name": "富途证券",
                "description": "使用 OpenAPI 连接到 OpenD Gateway",
                "required_fields": ["futu_host", "futu_port"],
                "optional_fields": ["futu_trade_password"],
                "setup_guide": "需要运行 FutuOpenD 并登录",
            },
            {
                "type": "tiger",
                "name": "老虎证券",
                "description": "使用 Tiger Open API",
                "required_fields": ["tiger_id", "tiger_account", "tiger_private_key"],
                "setup_guide": "需要在老虎开发者平台注册应用并获取密钥",
            },
        ]
    }


@router.post("/connect", response_model=BrokerStatusResponse, summary="连接券商")
async def connect_broker(
    config: BrokerConfigRequest,
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """
    连接到指定券商

    注意：当前使用模拟模式，不需要实际的券商连接。
    生产环境需要实际配置券商网关。
    """
    try:
        broker_config = config.to_config()
        portfolio.add_broker(broker_config)

        results = await portfolio.connect_all()
        broker_type = config.broker_type.lower()

        if results.get(broker_type, False):
            adapter = portfolio.get_adapter(broker_type)
            return BrokerStatusResponse(
                broker_type=broker_type,
                connected=True,
                account_id=adapter.account_id if adapter else None,
            )
        else:
            return BrokerStatusResponse(
                broker_type=broker_type,
                connected=False,
                error="Connection failed",
            )

    except AuthenticationError as e:
        logger.error(f"Broker authentication error: {e}")
        raise HTTPException(status_code=401, detail="Authentication failed")
    except BrokerError as e:
        logger.error(f"Broker connection error: {e}")
        raise HTTPException(status_code=500, detail="Broker operation failed")


@router.delete("/disconnect/{broker_type}", summary="断开券商连接")
async def disconnect_broker(
    broker_type: str,
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """断开指定券商的连接"""
    broker_type = validate_broker_type(broker_type)

    if broker_type not in portfolio.broker_names:
        raise HTTPException(status_code=404, detail=f"Broker not found: {broker_type}")

    adapter = portfolio.get_adapter(broker_type)
    if adapter:
        await adapter.disconnect()

    portfolio.remove_broker(broker_type)

    return {"message": f"Disconnected from {broker_type}"}


@router.get("/status", summary="获取所有券商连接状态")
async def get_broker_status(
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """获取所有已配置券商的连接状态"""
    health = await portfolio.health_check_all()

    statuses = []
    for name in portfolio.get_all_adapter_names():
        adapter = portfolio.get_adapter(name)
        statuses.append(BrokerStatusResponse(
            broker_type=name,
            connected=health.get(name, False),
            account_id=adapter.account_id if adapter else None,
        ))

    return {"brokers": statuses}


@router.get("/balance/{broker_type}", response_model=AccountBalanceResponse, summary="获取账户余额")
async def get_account_balance(
    broker_type: str,
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """获取指定券商的账户余额"""
    adapter = get_connected_adapter(portfolio, broker_type)

    try:
        balance = await adapter.get_account_balance()
        return AccountBalanceResponse(
            total_assets=balance.total_assets,
            cash=balance.cash,
            market_value=balance.market_value,
            buying_power=balance.buying_power,
            currency=balance.currency.value,
            day_pnl=balance.day_pnl,
            total_pnl=balance.total_pnl,
            margin_used=balance.margin_used,
            margin_available=balance.margin_available,
        )
    except BrokerError as e:
        logger.error(f"Failed to get account balance: {e}")
        raise HTTPException(status_code=500, detail="Failed to get account balance")


@router.get("/positions/{broker_type}", response_model=List[PositionResponse], summary="获取持仓列表")
async def get_positions(
    broker_type: str,
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """获取指定券商的持仓列表"""
    adapter = get_connected_adapter(portfolio, broker_type)

    try:
        positions = await adapter.get_positions()
        return [_position_to_response(p) for p in positions]
    except BrokerError as e:
        logger.error(f"Failed to get positions: {e}")
        raise HTTPException(status_code=500, detail="Failed to get positions")


@router.get("/trades/{broker_type}", response_model=List[TradeResponse], summary="获取交易历史")
async def get_trades(
    broker_type: str,
    days: int = Query(default=365, ge=1, le=3650),
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """
    获取指定券商的交易历史

    - days: 获取最近多少天的交易记录，默认365天，上限3650天（10年）
    """
    adapter = get_connected_adapter(portfolio, broker_type)

    try:
        trades = await adapter.get_trades(days)
        return [
            TradeResponse(
                symbol=t.symbol,
                market=t.market.value,
                action=t.action.value,
                quantity=t.quantity,
                price=t.price,
                total_value=t.quantity * t.price,
                commission=t.commission,
                currency=t.currency.value,
                trade_time=t.trade_time.isoformat() if t.trade_time else None,
                order_id=t.order_id,
                execution_id=t.execution_id,
                realized_pnl=t.realized_pnl,
            )
            for t in trades
        ]
    except BrokerError as e:
        logger.error(f"Failed to get trades: {e}")
        raise HTTPException(status_code=500, detail="Failed to get trades")


@router.get("/summary/{broker_type}", response_model=BrokerSummaryResponse, summary="获取券商摘要")
async def get_broker_summary(
    broker_type: str,
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """获取指定券商的投资组合摘要"""
    broker_type = validate_broker_type(broker_type)
    summary = await portfolio.get_broker_summary(broker_type)

    if not summary:
        raise HTTPException(status_code=404, detail=f"Broker not found or not connected: {broker_type}")

    return BrokerSummaryResponse(
        broker=summary.broker,
        account_id=summary.account_id,
        balance=AccountBalanceResponse(
            total_assets=summary.balance.total_assets,
            cash=summary.balance.cash,
            market_value=summary.balance.market_value,
            buying_power=summary.balance.buying_power,
            currency=summary.balance.currency.value,
            day_pnl=summary.balance.day_pnl,
            total_pnl=summary.balance.total_pnl,
            margin_used=summary.balance.margin_used,
            margin_available=summary.balance.margin_available,
        ),
        positions=[_position_to_response(p) for p in summary.positions],
        position_count=summary.position_count,
        top_holdings=summary.top_holdings,
        market_allocation=summary.market_allocation,
        currency_exposure=summary.currency_exposure,
        last_updated=summary.last_updated.isoformat() if summary.last_updated else "",
    )


@router.get("/unified", response_model=UnifiedSummaryResponse, summary="获取统一投资组合")
async def get_unified_portfolio(
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """
    获取所有券商的统一投资组合摘要

    聚合所有已连接券商的数据，计算总资产、总持仓等。
    """
    if not portfolio.connected_brokers:
        raise HTTPException(status_code=400, detail="No brokers connected")

    try:
        summary = await portfolio.get_unified_summary()
        return UnifiedSummaryResponse(
            total_assets=summary.total_assets,
            total_cash=summary.total_cash,
            total_market_value=summary.total_market_value,
            total_unrealized_pnl=summary.total_unrealized_pnl,
            total_realized_pnl=summary.total_realized_pnl,
            broker_allocation=summary.broker_allocation,
            market_allocation=summary.market_allocation,
            currency_exposure=summary.currency_exposure,
            top_holdings=summary.top_holdings,
            broker_count=summary.broker_count,
            position_count=summary.position_count,
            last_updated=summary.last_updated.isoformat(),
        )
    except Exception as e:
        logger.error(f"Failed to get unified portfolio: {e}")
        raise HTTPException(status_code=500, detail="Failed to get unified portfolio")


@router.get("/position/{symbol}", summary="跨券商查询持仓")
async def get_position_across_brokers(
    symbol: str,
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """
    查询某只股票在所有券商的持仓情况

    用于检查同一股票在不同账户的分布。
    """
    symbol = validate_symbol(symbol)
    positions = await portfolio.get_position_across_brokers(symbol.upper())

    result = {}
    for broker, pos in positions.items():
        if pos:
            result[broker] = _position_to_response(pos).model_dump()
        else:
            result[broker] = None

    return {
        "symbol": symbol.upper(),
        "positions": result,
    }


# =============================================================================
# 演示端点（模拟数据）
# =============================================================================

@router.post("/demo/setup", summary="设置演示环境")
async def setup_demo(
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """
    设置演示环境

    使用模拟数据连接三个券商（IBKR、富途、老虎），
    便于前端开发和演示。
    """
    # 添加三个模拟券商
    configs = [
        BrokerConfig(broker_type="ibkr"),
        BrokerConfig(broker_type="futu"),
        BrokerConfig(broker_type="tiger"),
    ]

    for config in configs:
        if config.broker_type not in portfolio.broker_names:
            portfolio.add_broker(config)

    # 连接所有
    results = await portfolio.connect_all()

    return {
        "message": "Demo environment setup complete",
        "brokers": results,
    }


@router.post("/demo/teardown", summary="清理演示环境")
async def teardown_demo(
    portfolio: UnifiedPortfolio = Depends(get_portfolio),
):
    """清理演示环境，断开所有连接"""
    await portfolio.disconnect_all()

    # 移除所有券商
    for broker in list(portfolio.broker_names):
        portfolio.remove_broker(broker)

    return {"message": "Demo environment cleared"}
