"""
FinMind - 蒙特卡洛模拟

提供股票价格和投资组合的蒙特卡洛模拟功能。

特性：
- 几何布朗运动 (GBM) 价格模拟
- VaR (Value at Risk) 计算
- 投资组合风险分析
- 多情景模拟
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# 数据类
# =============================================================================

@dataclass
class SimulationConfig:
    """模拟配置"""
    num_simulations: int = 10000       # 模拟次数
    time_horizon: int = 252            # 时间范围（交易日）
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    random_seed: Optional[int] = None  # 随机种子（用于复现）


@dataclass
class PriceSimulationResult:
    """价格模拟结果"""
    symbol: str
    current_price: float
    simulated_paths: np.ndarray        # shape: (num_simulations, time_horizon)
    final_prices: np.ndarray           # shape: (num_simulations,)
    # 统计数据
    mean_price: float
    median_price: float
    std_price: float
    min_price: float
    max_price: float
    # 百分位
    percentiles: Dict[int, float]      # {5: price, 25: price, 50: price, 75: price, 95: price}
    # VaR
    var_values: Dict[float, float]     # {0.95: var_value, 0.99: var_value}
    expected_return: float
    # 元数据
    config: SimulationConfig
    timestamp: datetime

    def to_dict(self) -> Dict:
        """转换为字典（不包含大数组）"""
        return {
            "symbol": self.symbol,
            "current_price": self.current_price,
            "mean_price": self.mean_price,
            "median_price": self.median_price,
            "std_price": self.std_price,
            "min_price": self.min_price,
            "max_price": self.max_price,
            "percentiles": self.percentiles,
            "var_values": self.var_values,
            "expected_return": self.expected_return,
            "num_simulations": self.config.num_simulations,
            "time_horizon": self.config.time_horizon,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PortfolioSimulationResult:
    """投资组合模拟结果"""
    portfolio_value: float
    simulated_values: np.ndarray       # shape: (num_simulations,)
    # 统计数据
    mean_value: float
    median_value: float
    std_value: float
    min_value: float
    max_value: float
    # 风险指标
    var_values: Dict[float, float]     # VaR
    cvar_values: Dict[float, float]    # CVaR (Expected Shortfall)
    sharpe_ratio: float
    max_drawdown: float
    # 资产贡献
    asset_contributions: Dict[str, float]
    # 元数据
    config: SimulationConfig
    timestamp: datetime

    def to_dict(self) -> Dict:
        return {
            "portfolio_value": self.portfolio_value,
            "mean_value": self.mean_value,
            "median_value": self.median_value,
            "std_value": self.std_value,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "var_values": self.var_values,
            "cvar_values": self.cvar_values,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "asset_contributions": self.asset_contributions,
            "num_simulations": self.config.num_simulations,
            "time_horizon": self.config.time_horizon,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# 蒙特卡洛模拟器
# =============================================================================

class MonteCarloSimulator:
    """
    蒙特卡洛模拟器

    使用几何布朗运动 (GBM) 模型模拟股票价格。

    GBM 公式:
    dS = μS dt + σS dW
    其中:
    - S: 股票价格
    - μ: 漂移率 (expected return)
    - σ: 波动率
    - dW: 维纳过程增量

    Example:
        ```python
        simulator = MonteCarloSimulator()

        # 单股票模拟
        result = simulator.simulate_price(
            symbol="AAPL",
            current_price=175.0,
            annual_return=0.10,
            annual_volatility=0.25,
        )
        print(f"Mean future price: ${result.mean_price:.2f}")
        print(f"95% VaR: ${result.var_values[0.95]:.2f}")

        # 投资组合模拟
        portfolio_result = simulator.simulate_portfolio(
            holdings=[
                {"symbol": "AAPL", "value": 50000, "return": 0.10, "volatility": 0.25},
                {"symbol": "MSFT", "value": 30000, "return": 0.12, "volatility": 0.22},
                {"symbol": "GOOGL", "value": 20000, "return": 0.08, "volatility": 0.28},
            ],
            correlations={
                ("AAPL", "MSFT"): 0.7,
                ("AAPL", "GOOGL"): 0.6,
                ("MSFT", "GOOGL"): 0.65,
            },
        )
        ```
    """

    def __init__(self, config: Optional[SimulationConfig] = None):
        self.config = config or SimulationConfig()
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def simulate_price(
        self,
        symbol: str,
        current_price: float,
        annual_return: float,
        annual_volatility: float,
        config: Optional[SimulationConfig] = None,
    ) -> PriceSimulationResult:
        """
        模拟单只股票的未来价格

        Args:
            symbol: 股票代码
            current_price: 当前价格
            annual_return: 年化预期收益率
            annual_volatility: 年化波动率
            config: 可选的模拟配置

        Returns:
            PriceSimulationResult: 模拟结果
        """
        cfg = config or self.config
        num_sims = cfg.num_simulations
        time_horizon = cfg.time_horizon

        # 转换为日频参数
        dt = 1 / 252  # 日时间步长
        daily_return = annual_return * dt
        daily_volatility = annual_volatility * np.sqrt(dt)

        # 生成随机数
        random_shocks = np.random.standard_normal((num_sims, time_horizon))

        # GBM 模拟
        # S(t+dt) = S(t) * exp((μ - σ²/2)dt + σ√dt * Z)
        drift = (annual_return - 0.5 * annual_volatility ** 2) * dt
        diffusion = annual_volatility * np.sqrt(dt) * random_shocks

        # 累积对数收益
        log_returns = drift + diffusion
        cumulative_log_returns = np.cumsum(log_returns, axis=1)

        # 价格路径
        price_paths = current_price * np.exp(cumulative_log_returns)

        # 最终价格
        final_prices = price_paths[:, -1]

        # 计算统计数据
        percentiles = {
            5: np.percentile(final_prices, 5),
            25: np.percentile(final_prices, 25),
            50: np.percentile(final_prices, 50),
            75: np.percentile(final_prices, 75),
            95: np.percentile(final_prices, 95),
        }

        # 计算 VaR（损失视角）
        returns = (final_prices - current_price) / current_price
        var_values = {}
        for cl in cfg.confidence_levels:
            var_values[cl] = -np.percentile(returns, (1 - cl) * 100) * current_price

        return PriceSimulationResult(
            symbol=symbol,
            current_price=current_price,
            simulated_paths=price_paths,
            final_prices=final_prices,
            mean_price=np.mean(final_prices),
            median_price=np.median(final_prices),
            std_price=np.std(final_prices),
            min_price=np.min(final_prices),
            max_price=np.max(final_prices),
            percentiles=percentiles,
            var_values=var_values,
            expected_return=(np.mean(final_prices) - current_price) / current_price,
            config=cfg,
            timestamp=datetime.now(),
        )

    def simulate_portfolio(
        self,
        holdings: List[Dict],
        correlations: Optional[Dict[Tuple[str, str], float]] = None,
        config: Optional[SimulationConfig] = None,
        risk_free_rate: float = 0.05,
    ) -> PortfolioSimulationResult:
        """
        模拟投资组合的未来价值

        Args:
            holdings: 持仓列表，每个元素包含:
                - symbol: 股票代码
                - value: 当前市值
                - return: 年化预期收益率
                - volatility: 年化波动率
            correlations: 资产间相关系数，键为 (symbol1, symbol2) 元组
            config: 可选的模拟配置
            risk_free_rate: 无风险利率

        Returns:
            PortfolioSimulationResult: 模拟结果
        """
        cfg = config or self.config
        num_sims = cfg.num_simulations
        time_horizon = cfg.time_horizon
        n_assets = len(holdings)

        if n_assets == 0:
            raise ValueError("Holdings cannot be empty")

        # 提取参数
        symbols = [h["symbol"] for h in holdings]
        values = np.array([h["value"] for h in holdings])
        returns = np.array([h["return"] for h in holdings])
        volatilities = np.array([h["volatility"] for h in holdings])

        total_value = np.sum(values)
        weights = values / total_value

        # 构建相关矩阵
        corr_matrix = np.eye(n_assets)
        if correlations:
            for i, s1 in enumerate(symbols):
                for j, s2 in enumerate(symbols):
                    if i != j:
                        key = (s1, s2) if (s1, s2) in correlations else (s2, s1)
                        if key in correlations:
                            corr_matrix[i, j] = correlations[key]

        # 构建协方差矩阵
        cov_matrix = np.outer(volatilities, volatilities) * corr_matrix

        # Cholesky 分解
        L = np.linalg.cholesky(cov_matrix)

        # 生成相关的随机数
        dt = 1 / 252
        uncorrelated_shocks = np.random.standard_normal((num_sims, time_horizon, n_assets))

        # 模拟每个资产的收益
        portfolio_values = np.zeros((num_sims, time_horizon + 1))
        portfolio_values[:, 0] = total_value

        for t in range(time_horizon):
            # 应用相关性
            correlated_shocks = uncorrelated_shocks[:, t, :] @ L.T

            # 计算每个资产的日收益
            asset_returns = (returns - 0.5 * volatilities ** 2) * dt + \
                           volatilities * np.sqrt(dt) * correlated_shocks

            # 更新投资组合价值
            asset_values = portfolio_values[:, t:t+1] * weights * np.exp(asset_returns)
            portfolio_values[:, t + 1] = np.sum(asset_values, axis=1)

        final_values = portfolio_values[:, -1]

        # 计算风险指标
        portfolio_returns = (final_values - total_value) / total_value

        # VaR
        var_values = {}
        for cl in cfg.confidence_levels:
            var_values[cl] = -np.percentile(portfolio_returns, (1 - cl) * 100) * total_value

        # CVaR (Expected Shortfall)
        cvar_values = {}
        for cl in cfg.confidence_levels:
            var_threshold = np.percentile(portfolio_returns, (1 - cl) * 100)
            tail_returns = portfolio_returns[portfolio_returns <= var_threshold]
            cvar_values[cl] = -np.mean(tail_returns) * total_value if len(tail_returns) > 0 else var_values[cl]

        # Sharpe Ratio
        mean_return = np.mean(portfolio_returns)
        std_return = np.std(portfolio_returns)
        annualized_return = mean_return * (252 / time_horizon)
        annualized_vol = std_return * np.sqrt(252 / time_horizon)
        sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else 0

        # Max Drawdown
        cumulative_max = np.maximum.accumulate(portfolio_values, axis=1)
        drawdowns = (cumulative_max - portfolio_values) / cumulative_max
        max_drawdown = np.max(drawdowns)

        # 资产贡献（基于边际 VaR）
        asset_contributions = {}
        for i, symbol in enumerate(symbols):
            # 简化计算：按权重分配 VaR
            asset_contributions[symbol] = weights[i] * var_values.get(0.95, 0)

        return PortfolioSimulationResult(
            portfolio_value=total_value,
            simulated_values=final_values,
            mean_value=np.mean(final_values),
            median_value=np.median(final_values),
            std_value=np.std(final_values),
            min_value=np.min(final_values),
            max_value=np.max(final_values),
            var_values=var_values,
            cvar_values=cvar_values,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            asset_contributions=asset_contributions,
            config=cfg,
            timestamp=datetime.now(),
        )

    def calculate_historical_volatility(
        self,
        prices: List[float],
        window: int = 252,
    ) -> float:
        """
        计算历史波动率

        Args:
            prices: 价格序列
            window: 计算窗口（交易日）

        Returns:
            float: 年化波动率
        """
        if len(prices) < 2:
            return 0.0

        prices_arr = np.array(prices)
        log_returns = np.log(prices_arr[1:] / prices_arr[:-1])

        if len(log_returns) < window:
            window = len(log_returns)

        daily_vol = np.std(log_returns[-window:])
        annual_vol = daily_vol * np.sqrt(252)

        return annual_vol

    def estimate_parameters_from_history(
        self,
        prices: List[float],
        window: int = 252,
    ) -> Tuple[float, float]:
        """
        从历史价格估计收益率和波动率

        Args:
            prices: 价格序列
            window: 计算窗口

        Returns:
            Tuple[float, float]: (年化收益率, 年化波动率)
        """
        if len(prices) < 2:
            return 0.0, 0.0

        prices_arr = np.array(prices)
        log_returns = np.log(prices_arr[1:] / prices_arr[:-1])

        if len(log_returns) < window:
            window = len(log_returns)

        recent_returns = log_returns[-window:]

        daily_return = np.mean(recent_returns)
        daily_vol = np.std(recent_returns)

        annual_return = daily_return * 252
        annual_vol = daily_vol * np.sqrt(252)

        return annual_return, annual_vol


# =============================================================================
# 便捷函数
# =============================================================================

def quick_price_simulation(
    symbol: str,
    current_price: float,
    annual_volatility: float,
    annual_return: float = 0.08,
    num_simulations: int = 10000,
    days: int = 252,
) -> Dict:
    """
    快速价格模拟

    Args:
        symbol: 股票代码
        current_price: 当前价格
        annual_volatility: 年化波动率
        annual_return: 年化预期收益率
        num_simulations: 模拟次数
        days: 模拟天数

    Returns:
        Dict: 模拟结果摘要
    """
    config = SimulationConfig(
        num_simulations=num_simulations,
        time_horizon=days,
    )

    simulator = MonteCarloSimulator(config)
    result = simulator.simulate_price(
        symbol=symbol,
        current_price=current_price,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
    )

    return result.to_dict()


def quick_portfolio_var(
    holdings: List[Dict],
    confidence_level: float = 0.95,
    num_simulations: int = 10000,
    days: int = 252,
) -> Dict:
    """
    快速计算投资组合 VaR

    Args:
        holdings: 持仓列表
        confidence_level: 置信水平
        num_simulations: 模拟次数
        days: 模拟天数

    Returns:
        Dict: VaR 结果
    """
    config = SimulationConfig(
        num_simulations=num_simulations,
        time_horizon=days,
        confidence_levels=[confidence_level],
    )

    simulator = MonteCarloSimulator(config)
    result = simulator.simulate_portfolio(holdings)

    return {
        "portfolio_value": result.portfolio_value,
        "var": result.var_values.get(confidence_level, 0),
        "cvar": result.cvar_values.get(confidence_level, 0),
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
    }
