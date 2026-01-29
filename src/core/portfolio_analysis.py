"""
FinMind - 投资组合上下文分析

结合券商持仓数据，提供个性化的投资分析和建议。

特性：
- 持仓感知的分析建议
- 投资组合优化建议
- 相关性分析
- 风险暴露分析
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np
import logging

from ..brokers.base import Position, PortfolioSummary
from .monte_carlo import MonteCarloSimulator, SimulationConfig

logger = logging.getLogger(__name__)


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class PositionContext:
    """持仓上下文"""
    symbol: str
    quantity: float
    market_value: float
    portfolio_weight: float           # 占投资组合比例
    avg_cost: float
    current_price: float
    unrealized_pnl: float
    unrealized_pnl_percent: float
    # 分析标记
    is_concentrated: bool = False     # 是否过度集中
    is_profitable: bool = False       # 是否盈利
    is_underwater: bool = False       # 是否亏损超过阈值
    days_held: Optional[int] = None   # 持有天数


@dataclass
class PortfolioRiskMetrics:
    """投资组合风险指标"""
    total_value: float
    cash_ratio: float                 # 现金比例
    # 集中度
    concentration_score: float        # 集中度评分 (0-100)
    top_holding_weight: float         # 最大持仓占比
    top_5_weight: float               # 前5大持仓占比
    # 多样化
    unique_positions: int             # 持仓数量
    market_diversity: Dict[str, float]  # 市场分布
    sector_diversity: Dict[str, float]  # 行业分布
    # 风险
    var_95: float                     # 95% VaR
    var_99: float                     # 99% VaR
    expected_volatility: float        # 预期波动率
    beta: float                       # 组合 Beta
    # 收益
    total_unrealized_pnl: float
    total_unrealized_pnl_percent: float
    winning_positions: int
    losing_positions: int
    win_rate: float


@dataclass
class PositionRecommendation:
    """持仓建议"""
    symbol: str
    action: str                       # hold, add, reduce, close
    reason: str
    urgency: str                      # low, medium, high
    target_weight: Optional[float] = None
    suggested_quantity_change: Optional[float] = None


@dataclass
class PortfolioAnalysisResult:
    """投资组合分析结果"""
    # 基本信息
    analysis_date: datetime
    total_value: float
    cash: float
    # 风险指标
    risk_metrics: PortfolioRiskMetrics
    # 持仓分析
    position_contexts: List[PositionContext]
    # 建议
    recommendations: List[PositionRecommendation]
    portfolio_recommendations: List[str]
    # 评分
    health_score: float               # 组合健康评分 (0-100)
    risk_score: float                 # 风险评分 (0-100)
    diversification_score: float      # 分散化评分 (0-100)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "analysis_date": self.analysis_date.isoformat(),
            "total_value": self.total_value,
            "cash": self.cash,
            "risk_metrics": {
                "total_value": self.risk_metrics.total_value,
                "cash_ratio": self.risk_metrics.cash_ratio,
                "concentration_score": self.risk_metrics.concentration_score,
                "top_holding_weight": self.risk_metrics.top_holding_weight,
                "top_5_weight": self.risk_metrics.top_5_weight,
                "unique_positions": self.risk_metrics.unique_positions,
                "var_95": self.risk_metrics.var_95,
                "var_99": self.risk_metrics.var_99,
                "expected_volatility": self.risk_metrics.expected_volatility,
                "total_unrealized_pnl": self.risk_metrics.total_unrealized_pnl,
                "win_rate": self.risk_metrics.win_rate,
            },
            "position_count": len(self.position_contexts),
            "recommendations": [
                {
                    "symbol": r.symbol,
                    "action": r.action,
                    "reason": r.reason,
                    "urgency": r.urgency,
                }
                for r in self.recommendations
            ],
            "portfolio_recommendations": self.portfolio_recommendations,
            "health_score": self.health_score,
            "risk_score": self.risk_score,
            "diversification_score": self.diversification_score,
        }


# =============================================================================
# 投资组合分析器
# =============================================================================

class PortfolioAnalyzer:
    """
    投资组合分析器

    分析用户的实际持仓，提供个性化建议。

    Example:
        ```python
        analyzer = PortfolioAnalyzer()

        # 从券商数据分析
        portfolio_summary = await broker.get_portfolio_summary()
        result = analyzer.analyze(portfolio_summary)

        print(f"Portfolio Health: {result.health_score}/100")
        for rec in result.recommendations:
            print(f"{rec.symbol}: {rec.action} - {rec.reason}")
        ```
    """

    def __init__(
        self,
        concentration_threshold: float = 0.20,  # 单一持仓超过20%视为集中
        underwater_threshold: float = -0.15,    # 亏损超过15%
        min_positions: int = 5,                 # 最少持仓数量
        max_single_weight: float = 0.25,        # 单一持仓最大权重
    ):
        self.concentration_threshold = concentration_threshold
        self.underwater_threshold = underwater_threshold
        self.min_positions = min_positions
        self.max_single_weight = max_single_weight
        self._simulator = MonteCarloSimulator()

    def analyze(
        self,
        portfolio: PortfolioSummary,
        market_data: Optional[Dict[str, Dict]] = None,
    ) -> PortfolioAnalysisResult:
        """
        分析投资组合

        Args:
            portfolio: 投资组合摘要
            market_data: 可选的市场数据（包含波动率等）

        Returns:
            PortfolioAnalysisResult: 分析结果
        """
        # 计算持仓上下文
        position_contexts = self._analyze_positions(portfolio)

        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(portfolio, position_contexts, market_data)

        # 生成建议
        position_recommendations = self._generate_position_recommendations(position_contexts)
        portfolio_recommendations = self._generate_portfolio_recommendations(
            portfolio, risk_metrics, position_contexts
        )

        # 计算评分
        health_score = self._calculate_health_score(risk_metrics, position_contexts)
        risk_score = self._calculate_risk_score(risk_metrics)
        diversification_score = self._calculate_diversification_score(risk_metrics)

        return PortfolioAnalysisResult(
            analysis_date=datetime.now(),
            total_value=portfolio.balance.total_assets,
            cash=portfolio.balance.cash,
            risk_metrics=risk_metrics,
            position_contexts=position_contexts,
            recommendations=position_recommendations,
            portfolio_recommendations=portfolio_recommendations,
            health_score=health_score,
            risk_score=risk_score,
            diversification_score=diversification_score,
        )

    def _analyze_positions(self, portfolio: PortfolioSummary) -> List[PositionContext]:
        """分析各个持仓"""
        total_value = portfolio.balance.total_assets
        contexts = []

        for pos in portfolio.positions:
            weight = pos.market_value / total_value if total_value > 0 else 0

            ctx = PositionContext(
                symbol=pos.symbol,
                quantity=pos.quantity,
                market_value=pos.market_value,
                portfolio_weight=weight,
                avg_cost=pos.avg_cost,
                current_price=pos.current_price,
                unrealized_pnl=pos.unrealized_pnl,
                unrealized_pnl_percent=pos.unrealized_pnl_percent,
                is_concentrated=weight > self.concentration_threshold,
                is_profitable=pos.unrealized_pnl > 0,
                is_underwater=pos.unrealized_pnl_percent / 100 < self.underwater_threshold,
            )
            contexts.append(ctx)

        return contexts

    def _calculate_risk_metrics(
        self,
        portfolio: PortfolioSummary,
        contexts: List[PositionContext],
        market_data: Optional[Dict] = None,
    ) -> PortfolioRiskMetrics:
        """计算风险指标"""
        total_value = portfolio.balance.total_assets
        cash = portfolio.balance.cash

        # 集中度
        sorted_contexts = sorted(contexts, key=lambda x: x.market_value, reverse=True)
        top_weight = sorted_contexts[0].portfolio_weight if sorted_contexts else 0
        top_5_weight = sum(c.portfolio_weight for c in sorted_contexts[:5])

        # 集中度评分 (HHI 指数转换)
        hhi = sum(c.portfolio_weight ** 2 for c in contexts)
        concentration_score = min(hhi * 10000, 100)  # 转换为0-100

        # 市场和行业分布
        market_dist: Dict[str, float] = {}
        for pos in portfolio.positions:
            market = pos.market.value
            market_dist[market] = market_dist.get(market, 0) + pos.market_value

        market_diversity = {k: v / total_value * 100 for k, v in market_dist.items()} if total_value > 0 else {}

        # 盈亏统计
        winning = sum(1 for c in contexts if c.is_profitable)
        losing = len(contexts) - winning
        win_rate = winning / len(contexts) if contexts else 0

        total_pnl = sum(c.unrealized_pnl for c in contexts)
        total_cost = sum(c.quantity * c.avg_cost for c in contexts)
        total_pnl_percent = total_pnl / total_cost * 100 if total_cost > 0 else 0

        # VaR 计算（简化版，假设平均波动率）
        default_volatility = 0.25
        var_95 = total_value * default_volatility * 1.645 * np.sqrt(1/252)
        var_99 = total_value * default_volatility * 2.326 * np.sqrt(1/252)

        return PortfolioRiskMetrics(
            total_value=total_value,
            cash_ratio=cash / total_value * 100 if total_value > 0 else 0,
            concentration_score=concentration_score,
            top_holding_weight=top_weight * 100,
            top_5_weight=top_5_weight * 100,
            unique_positions=len(contexts),
            market_diversity=market_diversity,
            sector_diversity={},  # 需要行业数据
            var_95=var_95,
            var_99=var_99,
            expected_volatility=default_volatility,
            beta=1.0,  # 需要市场数据计算
            total_unrealized_pnl=total_pnl,
            total_unrealized_pnl_percent=total_pnl_percent,
            winning_positions=winning,
            losing_positions=losing,
            win_rate=win_rate * 100,
        )

    def _generate_position_recommendations(
        self,
        contexts: List[PositionContext],
    ) -> List[PositionRecommendation]:
        """生成持仓建议"""
        recommendations = []

        for ctx in contexts:
            # 检查过度集中
            if ctx.is_concentrated:
                recommendations.append(PositionRecommendation(
                    symbol=ctx.symbol,
                    action="reduce",
                    reason=f"持仓占比 {ctx.portfolio_weight*100:.1f}% 过高，建议降至 {self.max_single_weight*100:.0f}% 以下",
                    urgency="medium",
                    target_weight=self.max_single_weight,
                ))

            # 检查深度亏损
            elif ctx.is_underwater:
                recommendations.append(PositionRecommendation(
                    symbol=ctx.symbol,
                    action="review",
                    reason=f"亏损 {ctx.unrealized_pnl_percent:.1f}%，建议重新评估投资逻辑",
                    urgency="high",
                ))

            # 检查大幅盈利（可能需要止盈）
            elif ctx.unrealized_pnl_percent > 50:
                recommendations.append(PositionRecommendation(
                    symbol=ctx.symbol,
                    action="consider_taking_profit",
                    reason=f"盈利 {ctx.unrealized_pnl_percent:.1f}%，可考虑部分止盈锁定利润",
                    urgency="low",
                ))

        return recommendations

    def _generate_portfolio_recommendations(
        self,
        portfolio: PortfolioSummary,
        risk_metrics: PortfolioRiskMetrics,
        contexts: List[PositionContext],
    ) -> List[str]:
        """生成投资组合级别建议"""
        recommendations = []

        # 检查持仓数量
        if risk_metrics.unique_positions < self.min_positions:
            recommendations.append(
                f"持仓数量（{risk_metrics.unique_positions}）较少，建议增加分散化程度，持有至少 {self.min_positions} 只股票"
            )

        # 检查现金比例
        if risk_metrics.cash_ratio > 30:
            recommendations.append(
                f"现金比例（{risk_metrics.cash_ratio:.1f}%）较高，可考虑适当增加投资以提高资金效率"
            )
        elif risk_metrics.cash_ratio < 5:
            recommendations.append(
                f"现金比例（{risk_metrics.cash_ratio:.1f}%）较低，建议保留一定现金作为应急储备"
            )

        # 检查集中度
        if risk_metrics.top_holding_weight > 30:
            recommendations.append(
                f"最大持仓占比（{risk_metrics.top_holding_weight:.1f}%）过高，单一股票风险敞口较大"
            )

        if risk_metrics.top_5_weight > 80:
            recommendations.append(
                f"前5大持仓占比（{risk_metrics.top_5_weight:.1f}%）过高，建议增加持仓分散度"
            )

        # 检查市场分布
        if len(risk_metrics.market_diversity) == 1:
            market = list(risk_metrics.market_diversity.keys())[0]
            recommendations.append(
                f"所有持仓集中在 {market} 市场，可考虑配置其他市场以分散地域风险"
            )

        # 检查胜率
        if risk_metrics.win_rate < 40 and risk_metrics.unique_positions >= 5:
            recommendations.append(
                f"盈利持仓占比（{risk_metrics.win_rate:.0f}%）较低，建议审视选股策略"
            )

        return recommendations

    def _calculate_health_score(
        self,
        risk_metrics: PortfolioRiskMetrics,
        contexts: List[PositionContext],
    ) -> float:
        """计算组合健康评分"""
        score = 100.0

        # 扣分项
        # 集中度过高
        if risk_metrics.top_holding_weight > 25:
            score -= (risk_metrics.top_holding_weight - 25) * 0.5

        # 持仓数量不足
        if risk_metrics.unique_positions < self.min_positions:
            score -= (self.min_positions - risk_metrics.unique_positions) * 5

        # 深度亏损持仓
        underwater_count = sum(1 for c in contexts if c.is_underwater)
        score -= underwater_count * 5

        # 现金比例异常
        if risk_metrics.cash_ratio > 40 or risk_metrics.cash_ratio < 3:
            score -= 10

        return max(0, min(100, score))

    def _calculate_risk_score(self, risk_metrics: PortfolioRiskMetrics) -> float:
        """计算风险评分（越高越有风险）"""
        score = 50.0  # 基准

        # 集中度
        score += risk_metrics.concentration_score * 0.3

        # 波动率
        score += (risk_metrics.expected_volatility - 0.20) * 100

        # 现金比例（低现金 = 高风险）
        if risk_metrics.cash_ratio < 5:
            score += 15
        elif risk_metrics.cash_ratio < 10:
            score += 5

        return max(0, min(100, score))

    def _calculate_diversification_score(self, risk_metrics: PortfolioRiskMetrics) -> float:
        """计算分散化评分"""
        score = 0.0

        # 持仓数量
        if risk_metrics.unique_positions >= 15:
            score += 30
        elif risk_metrics.unique_positions >= 10:
            score += 25
        elif risk_metrics.unique_positions >= 5:
            score += 15
        else:
            score += risk_metrics.unique_positions * 3

        # 集中度（反向）
        score += max(0, 30 - risk_metrics.concentration_score * 0.3)

        # 市场分布
        market_count = len(risk_metrics.market_diversity)
        score += min(market_count * 10, 20)

        # 前5大占比（反向）
        score += max(0, 20 - (risk_metrics.top_5_weight - 50) * 0.4)

        return max(0, min(100, score))

    def get_position_aware_analysis(
        self,
        symbol: str,
        portfolio: PortfolioSummary,
        base_analysis: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        获取结合持仓的分析

        Args:
            symbol: 股票代码
            portfolio: 投资组合
            base_analysis: 基础分析结果

        Returns:
            Dict: 增强的分析结果
        """
        # 查找持仓
        position = None
        for pos in portfolio.positions:
            if pos.symbol.upper() == symbol.upper():
                position = pos
                break

        enhanced = base_analysis.copy()
        enhanced["position_context"] = None

        if position:
            total_value = portfolio.balance.total_assets
            weight = position.market_value / total_value if total_value > 0 else 0

            enhanced["position_context"] = {
                "has_position": True,
                "quantity": position.quantity,
                "avg_cost": position.avg_cost,
                "market_value": position.market_value,
                "portfolio_weight": weight * 100,
                "unrealized_pnl": position.unrealized_pnl,
                "unrealized_pnl_percent": position.unrealized_pnl_percent,
                "is_concentrated": weight > self.concentration_threshold,
            }

            # 调整建议
            if weight > self.concentration_threshold:
                enhanced["position_aware_advice"] = (
                    f"您当前持有 {position.quantity} 股，占投资组合 {weight*100:.1f}%。"
                    f"持仓比例偏高，即使看好也不建议继续加仓。"
                )
            elif position.unrealized_pnl_percent < self.underwater_threshold * 100:
                enhanced["position_aware_advice"] = (
                    f"您当前持仓亏损 {position.unrealized_pnl_percent:.1f}%。"
                    f"建议重新评估投资逻辑，考虑是否止损或摊低成本。"
                )
            elif position.unrealized_pnl_percent > 30:
                enhanced["position_aware_advice"] = (
                    f"您当前持仓盈利 {position.unrealized_pnl_percent:.1f}%。"
                    f"可考虑部分止盈以锁定利润。"
                )
            else:
                enhanced["position_aware_advice"] = (
                    f"您当前持有 {position.quantity} 股，成本 ${position.avg_cost:.2f}。"
                    f"可根据分析结果决定是否调整仓位。"
                )
        else:
            enhanced["position_context"] = {
                "has_position": False,
            }
            enhanced["position_aware_advice"] = "您当前未持有该股票。"

        return enhanced
