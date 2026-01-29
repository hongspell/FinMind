"""
FinMind - 高级分析 API 路由

提供蒙特卡洛模拟和投资组合分析 API 接口
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field
import numpy as np
import logging

from .broker_routes import get_portfolio, UnifiedPortfolio
from ..core.portfolio_analysis import PortfolioAnalyzer
from ..core.portfolio_tracker import PortfolioTracker
from ..core.database import get_database

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["Advanced Analysis"])


# ============================================================================
# Pydantic Models
# ============================================================================

class MonteCarloRequest(BaseModel):
    """蒙特卡洛模拟请求"""
    symbol: Optional[str] = None
    current_price: Optional[float] = None
    annual_return: Optional[float] = 0.10
    annual_volatility: Optional[float] = 0.25
    days: int = Field(default=30, ge=1, le=365)
    simulations: int = Field(default=1000, ge=100, le=10000)
    confidence_levels: List[float] = Field(default=[0.95, 0.99])


class PriceSimulationResult(BaseModel):
    """价格模拟结果"""
    symbol: str
    current_price: float
    simulations: int
    days: int
    paths: List[List[float]]  # 采样路径
    final_prices: Dict[str, float]
    var_values: Dict[str, float]
    cvar_values: Dict[str, float]
    probability_of_profit: float
    expected_return: float


class PortfolioRiskMetrics(BaseModel):
    """投资组合风险指标"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    beta: float


class PositionRecommendation(BaseModel):
    """持仓建议"""
    symbol: str
    action: str
    reason: str
    priority: str
    current_weight: float
    suggested_weight: Optional[float] = None


class PortfolioAnalysisResult(BaseModel):
    """投资组合分析结果"""
    health_score: float
    risk_score: float
    diversification_score: float
    risk_metrics: PortfolioRiskMetrics
    recommendations: List[PositionRecommendation]
    concentration_risk: Dict[str, float]
    analysis_timestamp: str


# ============================================================================
# 蒙特卡洛模拟端点
# ============================================================================

@router.post("/monte-carlo/price", response_model=dict)
async def simulate_price(request: MonteCarloRequest):
    """
    蒙特卡洛价格模拟

    使用几何布朗运动模型模拟股票价格路径
    """
    try:
        current_price = request.current_price
        symbol = request.symbol or "STOCK"

        if not current_price or current_price <= 0:
            raise HTTPException(
                status_code=400,
                detail="必须提供有效的 current_price 参数（当前价格）"
            )

        # 参数
        dt = 1 / 252  # 日步长
        days = request.days
        simulations = request.simulations
        mu = request.annual_return or 0.10
        sigma = request.annual_volatility or 0.25

        # 生成随机数
        np.random.seed(None)
        random_shocks = np.random.standard_normal((simulations, days))

        # GBM 模拟
        daily_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks
        price_paths = current_price * np.exp(np.cumsum(daily_returns, axis=1))

        # 添加初始价格
        price_paths = np.column_stack([np.full(simulations, current_price), price_paths])

        # 计算最终价格统计
        final_prices = price_paths[:, -1]

        # VaR 和 CVaR
        var_values = {}
        cvar_values = {}
        for level in request.confidence_levels:
            var_percentile = (1 - level) * 100
            var_price = np.percentile(final_prices, var_percentile)
            var_loss = current_price - var_price
            var_values[level] = max(0, var_loss)

            # CVaR (Expected Shortfall)
            tail_prices = final_prices[final_prices <= var_price]
            if len(tail_prices) > 0:
                cvar_values[level] = current_price - np.mean(tail_prices)
            else:
                cvar_values[level] = var_values[level]

        # 计算分位数
        percentiles = {
            5: float(np.percentile(final_prices, 5)),
            25: float(np.percentile(final_prices, 25)),
            50: float(np.percentile(final_prices, 50)),
            75: float(np.percentile(final_prices, 75)),
            95: float(np.percentile(final_prices, 95)),
        }

        # 采样路径用于可视化（取前 20 条）
        sample_paths = price_paths[:20].tolist()

        # 盈利概率
        prob_profit = np.mean(final_prices > current_price)

        # 预期收益率
        expected_return = (np.mean(final_prices) - current_price) / current_price

        return {
            "success": True,
            "data": {
                "symbol": symbol,
                "current_price": current_price,
                "simulations": simulations,
                "days": days,
                "paths": sample_paths,
                "final_prices": {
                    "mean": float(np.mean(final_prices)),
                    "median": float(np.median(final_prices)),
                    "std": float(np.std(final_prices)),
                    "min": float(np.min(final_prices)),
                    "max": float(np.max(final_prices)),
                    "percentiles": percentiles,
                },
                "var_values": {str(k): float(v) for k, v in var_values.items()},
                "cvar_values": {str(k): float(v) for k, v in cvar_values.items()},
                "probability_of_profit": float(prob_profit),
                "expected_return": float(expected_return),
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Monte Carlo simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monte-carlo/volatility/{symbol}")
async def get_volatility(
    symbol: str,
    days: int = Query(default=252, ge=30, le=1000)
):
    """
    获取股票历史波动率

    注意：此接口需要外部数据源支持，当前返回错误提示
    """
    # TODO: 集成真实的历史数据源（如 yfinance, polygon.io 等）
    raise HTTPException(
        status_code=501,
        detail={
            "error": "历史波动率数据暂不可用",
            "message": f"获取 {symbol.upper()} 的历史波动率需要集成外部数据源",
            "suggestion": "请在蒙特卡洛模拟请求中手动指定 annual_volatility 参数"
        }
    )


# ============================================================================
# 投资组合分析端点
# ============================================================================

@router.get("/portfolio/analyze", response_model=dict)
async def analyze_portfolio(
    portfolio: UnifiedPortfolio = Depends(get_portfolio)
):
    """
    投资组合分析

    分析当前投资组合的健康度、风险和分散度
    基于已连接券商的真实持仓数据
    """
    try:
        # 检查是否有连接的券商
        if not portfolio.connected_brokers:
            return {
                "success": False,
                "error": "no_broker_connected",
                "message": "尚未连接任何券商账户，无法进行投资组合分析",
                "data": None
            }

        # 获取真实的投资组合摘要
        try:
            portfolio_summary = await portfolio.get_portfolio_summary()
        except Exception as e:
            logger.error(f"Failed to get portfolio summary: {e}")
            return {
                "success": False,
                "error": "data_fetch_failed",
                "message": f"获取投资组合数据失败: {str(e)}",
                "data": None
            }

        # 检查是否有持仓
        if not portfolio_summary.positions:
            return {
                "success": False,
                "error": "no_positions",
                "message": "当前投资组合没有持仓，无法进行分析",
                "data": None
            }

        # 使用真实数据进行分析
        analyzer = PortfolioAnalyzer()
        analysis_result = analyzer.analyze(portfolio_summary)

        # 转换为 API 响应格式
        # 计算权重
        total_value = portfolio_summary.balance.total_assets
        positions = portfolio_summary.positions

        # HHI 指数
        weights = [p.market_value / total_value for p in positions] if total_value > 0 else []
        hhi = sum(w**2 for w in weights) if weights else 0

        # 集中度
        sorted_weights = sorted(weights, reverse=True) if weights else []
        top_holding_weight = sorted_weights[0] if sorted_weights else 0
        top_3_weight = sum(sorted_weights[:3]) if len(sorted_weights) >= 3 else sum(sorted_weights)
        top_5_weight = sum(sorted_weights[:5]) if len(sorted_weights) >= 5 else sum(sorted_weights)

        # 风险指标 - 使用真实数据计算
        portfolio_volatility = analysis_result.risk_metrics.expected_volatility
        var_95 = analysis_result.risk_metrics.var_95
        var_99 = analysis_result.risk_metrics.var_99

        # 尝试从历史数据计算最大回撤
        max_drawdown = 0.0
        historical_volatility = portfolio_volatility
        historical_sharpe = 0.0
        try:
            tracker = PortfolioTracker(portfolio)
            max_drawdown = await tracker.get_max_drawdown(days=365)
            hist_vol = await tracker.get_volatility(days=252)
            if hist_vol > 0:
                historical_volatility = hist_vol
            historical_sharpe = await tracker.get_sharpe_ratio(days=252)
        except Exception as e:
            logger.debug(f"Could not calculate historical metrics: {e}")

        risk_metrics = {
            "var_95": float(var_95),
            "var_99": float(var_99),
            "cvar_95": float(var_95 * 1.2),  # 简化估算
            "cvar_99": float(var_99 * 1.2),
            "volatility": float(historical_volatility),
            "sharpe_ratio": float(historical_sharpe) if historical_sharpe != 0 else (float(analysis_result.risk_metrics.win_rate / 50) if analysis_result.risk_metrics.win_rate > 0 else 0.5),
            "max_drawdown": float(max_drawdown),
            "beta": float(analysis_result.risk_metrics.beta),
        }

        # 生成建议 - 基于真实持仓
        recommendations = []
        for rec in analysis_result.recommendations:
            # 找到对应持仓的权重
            pos_weight = 0
            for p in positions:
                if p.symbol.upper() == rec.symbol.upper():
                    pos_weight = p.market_value / total_value if total_value > 0 else 0
                    break

            recommendations.append({
                "symbol": rec.symbol,
                "action": rec.action,
                "reason": rec.reason,
                "priority": rec.urgency,
                "current_weight": pos_weight,
                "suggested_weight": rec.target_weight,
            })

        return {
            "success": True,
            "data": {
                "health_score": float(analysis_result.health_score),
                "risk_score": float(analysis_result.risk_score),
                "diversification_score": float(analysis_result.diversification_score),
                "risk_metrics": risk_metrics,
                "recommendations": recommendations,
                "concentration_risk": {
                    "top_holding_weight": float(top_holding_weight),
                    "top_3_weight": float(top_3_weight),
                    "top_5_weight": float(top_5_weight),
                    "hhi_index": float(hhi),
                },
                "position_count": len(positions),
                "total_value": float(total_value),
                "total_unrealized_pnl": float(analysis_result.risk_metrics.total_unrealized_pnl),
                "total_unrealized_pnl_percent": float(analysis_result.risk_metrics.total_unrealized_pnl_percent),
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }
        }

    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}", exc_info=True)
        return {
            "success": False,
            "error": "analysis_failed",
            "message": f"投资组合分析失败: {str(e)}",
            "data": None
        }


@router.get("/portfolio/risk")
async def get_portfolio_risk(
    portfolio: UnifiedPortfolio = Depends(get_portfolio)
):
    """获取投资组合风险指标"""
    # 检查是否有连接的券商
    if not portfolio.connected_brokers:
        return {
            "success": False,
            "error": "no_broker_connected",
            "message": "尚未连接任何券商账户",
            "data": None
        }

    try:
        portfolio_summary = await portfolio.get_portfolio_summary()

        if not portfolio_summary.positions:
            return {
                "success": False,
                "error": "no_positions",
                "message": "当前没有持仓",
                "data": None
            }

        analyzer = PortfolioAnalyzer()
        analysis_result = analyzer.analyze(portfolio_summary)

        total_value = portfolio_summary.balance.total_assets
        var_95 = analysis_result.risk_metrics.var_95
        var_99 = analysis_result.risk_metrics.var_99

        return {
            "success": True,
            "data": {
                "var_95": float(var_95),
                "var_99": float(var_99),
                "cvar_95": float(var_95 * 1.2),
                "cvar_99": float(var_99 * 1.2),
                "volatility": float(analysis_result.risk_metrics.expected_volatility),
                "sharpe_ratio": float(analysis_result.risk_metrics.win_rate / 50) if analysis_result.risk_metrics.win_rate > 0 else 0.5,
                "max_drawdown": 0.0,  # 需要历史数据
                "beta": float(analysis_result.risk_metrics.beta),
            }
        }
    except Exception as e:
        logger.error(f"Portfolio risk calculation error: {e}")
        return {
            "success": False,
            "error": "calculation_failed",
            "message": f"风险指标计算失败: {str(e)}",
            "data": None
        }


@router.post("/portfolio/snapshot")
async def record_portfolio_snapshot(
    portfolio: UnifiedPortfolio = Depends(get_portfolio)
):
    """
    手动记录投资组合快照

    记录当前投资组合状态到数据库，用于计算历史指标（如最大回撤）。
    建议在交易日收盘后调用，或设置定时任务自动记录。
    """
    if not portfolio.connected_brokers:
        return {
            "success": False,
            "error": "no_broker_connected",
            "message": "尚未连接任何券商账户",
        }

    try:
        tracker = PortfolioTracker(portfolio)
        success = await tracker.record_snapshot()

        if success:
            return {
                "success": True,
                "message": "投资组合快照已记录",
            }
        else:
            return {
                "success": False,
                "error": "database_unavailable",
                "message": "数据库不可用，请确保已启动 TimescaleDB 服务 (make docker-up)",
            }
    except Exception as e:
        logger.error(f"Failed to record snapshot: {e}")
        return {
            "success": False,
            "error": "snapshot_failed",
            "message": str(e),
        }


@router.get("/portfolio/history")
async def get_portfolio_history(
    days: int = Query(default=30, ge=1, le=365),
    portfolio: UnifiedPortfolio = Depends(get_portfolio)
):
    """
    获取投资组合历史数据

    返回指定天数内的投资组合快照数据，用于绘制净值曲线等。
    """
    try:
        db = await get_database()
        if not db.is_available:
            return {
                "success": False,
                "error": "database_unavailable",
                "message": "数据库不可用，请确保已启动 TimescaleDB 服务",
                "data": None
            }

        snapshots = await db.get_portfolio_history(days=days)

        return {
            "success": True,
            "data": {
                "snapshots": [
                    {
                        "time": s.time.isoformat(),
                        "total_assets": s.total_assets,
                        "total_cash": s.total_cash,
                        "total_market_value": s.total_market_value,
                        "total_unrealized_pnl": s.total_unrealized_pnl,
                        "position_count": s.position_count,
                    }
                    for s in snapshots
                ],
                "count": len(snapshots),
                "days": days,
            }
        }
    except Exception as e:
        logger.error(f"Failed to get portfolio history: {e}")
        return {
            "success": False,
            "error": "fetch_failed",
            "message": str(e),
            "data": None
        }


@router.get("/portfolio/recommendations")
async def get_portfolio_recommendations(
    portfolio: UnifiedPortfolio = Depends(get_portfolio)
):
    """获取持仓建议"""
    # 检查是否有连接的券商
    if not portfolio.connected_brokers:
        return {
            "success": False,
            "error": "no_broker_connected",
            "message": "尚未连接任何券商账户",
            "data": None
        }

    try:
        portfolio_summary = await portfolio.get_portfolio_summary()

        if not portfolio_summary.positions:
            return {
                "success": True,
                "data": {
                    "recommendations": [],
                    "message": "当前没有持仓，无需建议"
                }
            }

        analyzer = PortfolioAnalyzer()
        analysis_result = analyzer.analyze(portfolio_summary)

        total_value = portfolio_summary.balance.total_assets
        positions = portfolio_summary.positions

        recommendations = []
        for rec in analysis_result.recommendations:
            # 找到对应持仓的权重
            pos_weight = 0
            for p in positions:
                if p.symbol.upper() == rec.symbol.upper():
                    pos_weight = p.market_value / total_value if total_value > 0 else 0
                    break

            recommendations.append({
                "symbol": rec.symbol,
                "action": rec.action,
                "reason": rec.reason,
                "priority": rec.urgency,
                "current_weight": pos_weight,
                "suggested_weight": rec.target_weight,
            })

        # 如果没有特别的建议，返回空列表
        return {
            "success": True,
            "data": {
                "recommendations": recommendations,
                "portfolio_recommendations": analysis_result.portfolio_recommendations,
            }
        }

    except Exception as e:
        logger.error(f"Portfolio recommendations error: {e}")
        return {
            "success": False,
            "error": "calculation_failed",
            "message": f"获取建议失败: {str(e)}",
            "data": None
        }
