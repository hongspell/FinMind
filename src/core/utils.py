"""
FinMind - 工具函数模块

提供通用工具函数：
- 日期处理
- 数值格式化
- 财务计算
- 数据验证
"""

import math
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from decimal import Decimal, ROUND_HALF_UP
import re


# =============================================================================
# 日期工具
# =============================================================================

def get_fiscal_quarter(date: datetime) -> str:
    """获取财务季度"""
    month = date.month
    year = date.year
    
    if month <= 3:
        return f"{year}Q1"
    elif month <= 6:
        return f"{year}Q2"
    elif month <= 9:
        return f"{year}Q3"
    else:
        return f"{year}Q4"


def parse_quarter(quarter_str: str) -> Tuple[int, int]:
    """解析季度字符串，返回(年份, 季度)"""
    match = re.match(r"(\d{4})Q([1-4])", quarter_str)
    if not match:
        raise ValueError(f"Invalid quarter format: {quarter_str}")
    return int(match.group(1)), int(match.group(2))


def get_quarter_dates(year: int, quarter: int) -> Tuple[datetime, datetime]:
    """获取季度的开始和结束日期"""
    start_months = {1: 1, 2: 4, 3: 7, 4: 10}
    end_months = {1: 3, 2: 6, 3: 9, 4: 12}
    end_days = {1: 31, 2: 30, 3: 30, 4: 31}
    
    start = datetime(year, start_months[quarter], 1)
    end = datetime(year, end_months[quarter], end_days[quarter])
    
    return start, end


def trading_days_between(start: datetime, end: datetime) -> int:
    """计算两个日期之间的交易日数量（简化版）"""
    total_days = (end - start).days
    weeks = total_days // 7
    remaining = total_days % 7
    
    # 假设每周5个交易日
    trading_days = weeks * 5
    
    # 处理剩余天数
    current_day = start.weekday()
    for _ in range(remaining):
        current_day = (current_day + 1) % 7
        if current_day < 5:  # 周一到周五
            trading_days += 1
    
    return trading_days


# =============================================================================
# 数值格式化
# =============================================================================

def format_currency(value: float, currency: str = "USD", precision: int = 2) -> str:
    """格式化货币"""
    symbols = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "CNY": "¥",
        "JPY": "¥"
    }
    symbol = symbols.get(currency, currency)
    
    if abs(value) >= 1e12:
        return f"{symbol}{value/1e12:.{precision}f}T"
    elif abs(value) >= 1e9:
        return f"{symbol}{value/1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"{symbol}{value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{symbol}{value/1e3:.{precision}f}K"
    else:
        return f"{symbol}{value:.{precision}f}"


def format_percentage(value: float, precision: int = 1) -> str:
    """格式化百分比"""
    return f"{value * 100:.{precision}f}%"


def format_multiple(value: float, precision: int = 1) -> str:
    """格式化倍数"""
    return f"{value:.{precision}f}x"


def format_large_number(value: float, precision: int = 2) -> str:
    """格式化大数字"""
    if abs(value) >= 1e12:
        return f"{value/1e12:.{precision}f}T"
    elif abs(value) >= 1e9:
        return f"{value/1e9:.{precision}f}B"
    elif abs(value) >= 1e6:
        return f"{value/1e6:.{precision}f}M"
    elif abs(value) >= 1e3:
        return f"{value/1e3:.{precision}f}K"
    else:
        return f"{value:.{precision}f}"


# =============================================================================
# 财务计算
# =============================================================================

def calculate_cagr(
    start_value: float, 
    end_value: float, 
    years: float
) -> float:
    """计算复合年增长率"""
    if start_value <= 0 or years <= 0:
        return 0.0
    
    return (end_value / start_value) ** (1 / years) - 1


def calculate_wacc(
    equity_weight: float,
    debt_weight: float,
    cost_of_equity: float,
    cost_of_debt: float,
    tax_rate: float
) -> float:
    """计算加权平均资本成本"""
    return (equity_weight * cost_of_equity + 
            debt_weight * cost_of_debt * (1 - tax_rate))


def calculate_capm(
    risk_free_rate: float,
    beta: float,
    market_risk_premium: float,
    size_premium: float = 0.0,
    company_specific_risk: float = 0.0
) -> float:
    """计算资本资产定价模型的预期回报"""
    return (risk_free_rate + 
            beta * market_risk_premium + 
            size_premium + 
            company_specific_risk)


def calculate_dcf_value(
    cash_flows: List[float],
    discount_rate: float,
    terminal_value: float = 0.0
) -> float:
    """计算现金流折现值"""
    pv = 0.0
    for i, cf in enumerate(cash_flows, 1):
        pv += cf / ((1 + discount_rate) ** i)
    
    # 添加终值的现值
    if terminal_value > 0:
        n = len(cash_flows)
        pv += terminal_value / ((1 + discount_rate) ** n)
    
    return pv


def calculate_gordon_growth_terminal_value(
    final_fcf: float,
    terminal_growth_rate: float,
    discount_rate: float
) -> float:
    """使用Gordon增长模型计算终值"""
    if discount_rate <= terminal_growth_rate:
        raise ValueError("折现率必须大于终值增长率")
    
    return (final_fcf * (1 + terminal_growth_rate)) / (discount_rate - terminal_growth_rate)


def calculate_ev_to_equity(
    enterprise_value: float,
    total_debt: float,
    cash: float,
    minority_interest: float = 0.0,
    preferred_stock: float = 0.0
) -> float:
    """从企业价值计算股权价值"""
    return (enterprise_value - total_debt - minority_interest - 
            preferred_stock + cash)


def calculate_implied_growth(
    pe_ratio: float,
    cost_of_equity: float,
    payout_ratio: float = 0.3
) -> float:
    """从P/E倍数反推隐含增长率"""
    # 使用Gordon模型反推: P/E = payout / (r - g)
    # 解出: g = r - payout / PE
    return cost_of_equity - payout_ratio / pe_ratio


# =============================================================================
# 统计计算
# =============================================================================

def calculate_percentile(values: List[float], percentile: float) -> float:
    """计算百分位数"""
    if not values:
        return 0.0
    
    sorted_values = sorted(values)
    n = len(sorted_values)
    
    if percentile >= 100:
        return sorted_values[-1]
    if percentile <= 0:
        return sorted_values[0]
    
    k = (n - 1) * percentile / 100
    f = math.floor(k)
    c = math.ceil(k)
    
    if f == c:
        return sorted_values[int(k)]
    
    return sorted_values[f] * (c - k) + sorted_values[c] * (k - f)


def calculate_zscore(value: float, mean: float, std_dev: float) -> float:
    """计算Z分数"""
    if std_dev == 0:
        return 0.0
    return (value - mean) / std_dev


def winsorize(values: List[float], lower: float = 5, upper: float = 95) -> List[float]:
    """Winsorization处理极端值"""
    lower_bound = calculate_percentile(values, lower)
    upper_bound = calculate_percentile(values, upper)
    
    return [
        max(lower_bound, min(upper_bound, v)) 
        for v in values
    ]


def calculate_rolling_stats(
    values: List[float], 
    window: int
) -> Dict[str, List[float]]:
    """计算滚动统计量"""
    if len(values) < window:
        return {"mean": [], "std": [], "min": [], "max": []}
    
    means = []
    stds = []
    mins = []
    maxs = []
    
    for i in range(len(values) - window + 1):
        window_values = values[i:i + window]
        mean = sum(window_values) / window
        std = math.sqrt(sum((v - mean) ** 2 for v in window_values) / window)
        
        means.append(mean)
        stds.append(std)
        mins.append(min(window_values))
        maxs.append(max(window_values))
    
    return {"mean": means, "std": stds, "min": mins, "max": maxs}


# =============================================================================
# 数据验证
# =============================================================================

def validate_ticker(ticker: str) -> bool:
    """验证股票代码格式"""
    # 美股：1-5个字母
    # A股：6位数字+后缀
    us_pattern = r"^[A-Z]{1,5}$"
    cn_pattern = r"^\d{6}\.(SH|SZ|BJ)$"
    
    ticker_upper = ticker.upper()
    return bool(re.match(us_pattern, ticker_upper) or 
                re.match(cn_pattern, ticker_upper))


def validate_date_range(
    start_date: datetime, 
    end_date: datetime,
    max_days: int = 3650
) -> bool:
    """验证日期范围"""
    if start_date >= end_date:
        return False
    if (end_date - start_date).days > max_days:
        return False
    return True


def sanitize_financial_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """清理财务数据"""
    cleaned = {}
    
    for key, value in data.items():
        if value is None:
            continue
        
        if isinstance(value, (int, float)):
            # 处理无穷大和NaN
            if math.isnan(value) or math.isinf(value):
                continue
            cleaned[key] = value
        elif isinstance(value, str):
            cleaned[key] = value.strip()
        elif isinstance(value, dict):
            cleaned[key] = sanitize_financial_data(value)
        elif isinstance(value, list):
            cleaned[key] = [
                sanitize_financial_data(v) if isinstance(v, dict) else v
                for v in value
                if v is not None
            ]
        else:
            cleaned[key] = value
    
    return cleaned


# =============================================================================
# 文本处理
# =============================================================================

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """截断文本"""
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def extract_numbers(text: str) -> List[float]:
    """从文本中提取数字"""
    pattern = r"-?\d+\.?\d*"
    matches = re.findall(pattern, text)
    return [float(m) for m in matches]


def normalize_company_name(name: str) -> str:
    """标准化公司名称"""
    # 移除常见后缀
    suffixes = [
        "Inc.", "Inc", "Corp.", "Corp", "Corporation",
        "Ltd.", "Ltd", "Limited", "LLC", "L.L.C.",
        "PLC", "plc", "Co.", "Co", "Company"
    ]
    
    normalized = name.strip()
    for suffix in suffixes:
        if normalized.endswith(suffix):
            normalized = normalized[:-len(suffix)].strip()
            break
    
    # 移除标点
    normalized = re.sub(r"[,.]$", "", normalized)
    
    return normalized.strip()


# =============================================================================
# 辅助函数
# =============================================================================

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法"""
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, min_value: float, max_value: float) -> float:
    """限制值在范围内"""
    return max(min_value, min(max_value, value))


def interpolate(
    x: float, 
    x1: float, 
    y1: float, 
    x2: float, 
    y2: float
) -> float:
    """线性插值"""
    if x2 == x1:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


def round_to_significant(value: float, digits: int = 2) -> float:
    """四舍五入到有效数字"""
    if value == 0:
        return 0
    
    magnitude = math.floor(math.log10(abs(value)))
    return round(value, -int(magnitude) + digits - 1)
