"""
FinMind - 高级分析 API 路由

提供蒙特卡洛模拟和投资组合分析 API 接口
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
import numpy as np
import logging

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
        current_price = request.current_price or 100.0
        symbol = request.symbol or "STOCK"

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

    except Exception as e:
        logger.error(f"Monte Carlo simulation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monte-carlo/volatility/{symbol}")
async def get_volatility(
    symbol: str,
    days: int = Query(default=252, ge=30, le=1000)
):
    """获取股票历史波动率"""
    # 模拟数据（实际应从数据源获取）
    np.random.seed(hash(symbol) % 2**32)
    volatility = 0.15 + np.random.random() * 0.35  # 15% - 50%

    return {
        "success": True,
        "data": {
            "symbol": symbol.upper(),
            "volatility": float(volatility),
            "days": days,
        }
    }


# ============================================================================
# 投资组合分析端点
# ============================================================================

@router.get("/portfolio/analyze", response_model=dict)
async def analyze_portfolio():
    """
    投资组合分析

    分析当前投资组合的健康度、风险和分散度
    """
    try:
        # 模拟数据（实际应从 broker 获取真实数据）
        np.random.seed(42)

        # 生成示例持仓
        positions = [
            {"symbol": "AAPL", "weight": 0.25, "pnl_pct": 15.5},
            {"symbol": "MSFT", "weight": 0.20, "pnl_pct": 12.3},
            {"symbol": "GOOGL", "weight": 0.15, "pnl_pct": -3.2},
            {"symbol": "AMZN", "weight": 0.12, "pnl_pct": 8.7},
            {"symbol": "NVDA", "weight": 0.10, "pnl_pct": 45.2},
            {"symbol": "TSLA", "weight": 0.08, "pnl_pct": -12.5},
            {"symbol": "META", "weight": 0.05, "pnl_pct": 22.1},
            {"symbol": "JPM", "weight": 0.05, "pnl_pct": 5.3},
        ]

        # 计算权重
        weights = [p["weight"] for p in positions]

        # HHI 指数
        hhi = sum(w**2 for w in weights)

        # 集中度
        sorted_weights = sorted(weights, reverse=True)
        top_holding_weight = sorted_weights[0]
        top_3_weight = sum(sorted_weights[:3])
        top_5_weight = sum(sorted_weights[:5])

        # 分散度评分 (0-100)
        diversification_score = min(100, max(0, (1 - hhi) * 120))

        # 风险指标
        portfolio_volatility = 0.22
        var_95 = 10000 * 1.65 * portfolio_volatility / np.sqrt(252)
        var_99 = 10000 * 2.33 * portfolio_volatility / np.sqrt(252)

        risk_metrics = {
            "var_95": float(var_95),
            "var_99": float(var_99),
            "cvar_95": float(var_95 * 1.2),
            "cvar_99": float(var_99 * 1.2),
            "volatility": float(portfolio_volatility),
            "sharpe_ratio": 1.25,
            "max_drawdown": 0.18,
            "beta": 1.15,
        }

        # 风险评分 (0-100, 越高风险越大)
        risk_score = min(100, max(0,
            hhi * 100 +  # 集中度风险
            portfolio_volatility * 100 +  # 波动风险
            risk_metrics["beta"] * 10  # 市场风险
        ))

        # 健康评分 (0-100)
        health_score = min(100, max(0,
            100 - risk_score * 0.3 +
            diversification_score * 0.3 +
            (1 if risk_metrics["sharpe_ratio"] > 1 else 0.5) * 20 +
            (1 if risk_metrics["max_drawdown"] < 0.2 else 0.5) * 20
        ))

        # 生成建议
        recommendations = []
        for p in positions:
            if p["weight"] > 0.2:
                recommendations.append({
                    "symbol": p["symbol"],
                    "action": "reduce",
                    "reason": f"持仓占比 {p['weight']*100:.1f}% 过高，建议减持以降低集中风险",
                    "priority": "high",
                    "current_weight": p["weight"],
                    "suggested_weight": 0.15,
                })
            elif p["pnl_pct"] < -10:
                recommendations.append({
                    "symbol": p["symbol"],
                    "action": "watch",
                    "reason": f"亏损 {abs(p['pnl_pct']):.1f}%，需要关注止损位",
                    "priority": "medium",
                    "current_weight": p["weight"],
                })
            elif p["pnl_pct"] > 40:
                recommendations.append({
                    "symbol": p["symbol"],
                    "action": "reduce",
                    "reason": f"盈利 {p['pnl_pct']:.1f}%，可考虑部分止盈",
                    "priority": "low",
                    "current_weight": p["weight"],
                })

        return {
            "success": True,
            "data": {
                "health_score": float(health_score),
                "risk_score": float(risk_score),
                "diversification_score": float(diversification_score),
                "risk_metrics": risk_metrics,
                "recommendations": recommendations,
                "concentration_risk": {
                    "top_holding_weight": float(top_holding_weight),
                    "top_3_weight": float(top_3_weight),
                    "top_5_weight": float(top_5_weight),
                    "hhi_index": float(hhi),
                },
                "analysis_timestamp": datetime.utcnow().isoformat(),
            }
        }

    except Exception as e:
        logger.error(f"Portfolio analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/portfolio/risk")
async def get_portfolio_risk():
    """获取投资组合风险指标"""
    return {
        "success": True,
        "data": {
            "var_95": 1523.45,
            "var_99": 2145.67,
            "cvar_95": 1828.14,
            "cvar_99": 2574.80,
            "volatility": 0.22,
            "sharpe_ratio": 1.25,
            "max_drawdown": 0.18,
            "beta": 1.15,
        }
    }


@router.get("/portfolio/recommendations")
async def get_portfolio_recommendations():
    """获取持仓建议"""
    return {
        "success": True,
        "data": {
            "recommendations": [
                {
                    "symbol": "AAPL",
                    "action": "hold",
                    "reason": "持仓比例适中，基本面稳健",
                    "priority": "low",
                    "current_weight": 0.25,
                },
                {
                    "symbol": "NVDA",
                    "action": "reduce",
                    "reason": "涨幅较大，可考虑部分止盈",
                    "priority": "medium",
                    "current_weight": 0.10,
                    "suggested_weight": 0.07,
                },
            ]
        }
    }
