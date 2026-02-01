"""
FinMind - Quantitative Backtest Module

Pure quantitative backtesting: runs technical indicators + DCF valuation
on historical data and compares predictions against actual outcomes.
No LLM calls.
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def run_backtest(
    symbol: str,
    backtest_date: str,
    forward_days: int = 90,
) -> Dict[str, Any]:
    """
    Run a quantitative backtest for *symbol* as of *backtest_date*.

    1. Fetch historical price data up to backtest_date.
    2. Compute technical indicators (MA crossover, RSI, MACD).
    3. Run DCF valuation using historical financials.
    4. Fetch actual prices for *forward_days* after backtest_date.
    5. Compare prediction vs. reality.

    Returns a dict with prediction, actual outcome, and accuracy metrics.
    """
    import yfinance as yf
    import numpy as np

    bt_dt = datetime.strptime(backtest_date, "%Y-%m-%d")
    end_dt = bt_dt + timedelta(days=forward_days)
    # Fetch extended history for indicator calculation (need ~260 trading days before)
    fetch_start = bt_dt - timedelta(days=400)

    ticker = yf.Ticker(symbol)
    hist = ticker.history(start=fetch_start.strftime("%Y-%m-%d"), end=end_dt.strftime("%Y-%m-%d"))

    if hist.empty:
        return {"error": f"No historical data available for {symbol}"}

    # Split into before / after backtest date
    before = hist[hist.index <= bt_dt.strftime("%Y-%m-%d")]
    after = hist[hist.index > bt_dt.strftime("%Y-%m-%d")]

    if before.empty or len(before) < 50:
        return {"error": "Insufficient historical data before backtest date"}

    closes = before["Close"].values
    bt_price = float(closes[-1])

    # ------ Technical Indicators ------
    tech_signals = _compute_technical_signals(closes)

    # ------ DCF Valuation ------
    dcf_result = _run_historical_dcf(ticker, bt_price)

    # ------ Aggregate prediction ------
    # Simple scoring: +1 bullish, -1 bearish per signal
    score = 0
    score += 1 if tech_signals["ma_crossover"] == "bullish" else -1
    score += 1 if tech_signals["rsi_signal"] == "bullish" else (-1 if tech_signals["rsi_signal"] == "bearish" else 0)
    score += 1 if tech_signals["macd_signal"] == "bullish" else -1

    if dcf_result.get("upside_pct") is not None:
        if dcf_result["upside_pct"] > 10:
            score += 2
        elif dcf_result["upside_pct"] > 0:
            score += 1
        elif dcf_result["upside_pct"] < -10:
            score -= 2
        else:
            score -= 1

    predicted_direction = "bullish" if score > 0 else ("bearish" if score < 0 else "neutral")

    # ------ Actual outcome ------
    actual_result = _compute_actual_outcome(after, bt_price, forward_days)

    # ------ Accuracy ------
    direction_correct = None
    if actual_result.get("actual_direction") and predicted_direction != "neutral":
        direction_correct = predicted_direction == actual_result["actual_direction"]

    return {
        "symbol": symbol,
        "backtest_date": backtest_date,
        "forward_days": forward_days,
        "backtest_price": round(bt_price, 2),
        "prediction": {
            "direction": predicted_direction,
            "score": score,
            "technical_signals": tech_signals,
            "dcf": dcf_result,
        },
        "actual": actual_result,
        "accuracy": {
            "direction_correct": direction_correct,
            "predicted_direction": predicted_direction,
            "actual_direction": actual_result.get("actual_direction"),
        },
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_technical_signals(closes) -> Dict[str, Any]:
    """Compute MA crossover, RSI, MACD from a numpy array of close prices."""
    import numpy as np

    result: Dict[str, Any] = {}

    # --- MA crossover (50 / 200) ---
    if len(closes) >= 200:
        ma50 = float(np.mean(closes[-50:]))
        ma200 = float(np.mean(closes[-200:]))
        result["ma50"] = round(ma50, 2)
        result["ma200"] = round(ma200, 2)
        result["ma_crossover"] = "bullish" if ma50 > ma200 else "bearish"
    else:
        ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else float(closes[-1])
        ma50 = float(np.mean(closes[-50:])) if len(closes) >= 50 else float(closes[-1])
        result["ma20"] = round(ma20, 2)
        result["ma50"] = round(ma50, 2)
        result["ma_crossover"] = "bullish" if ma20 > ma50 else "bearish"

    # --- RSI (14-period) ---
    if len(closes) >= 15:
        deltas = np.diff(closes[-15:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = float(np.mean(gains))
        avg_loss = float(np.mean(losses))
        if avg_loss == 0:
            rsi = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))
        result["rsi"] = round(rsi, 2)
        result["rsi_signal"] = "bearish" if rsi > 70 else ("bullish" if rsi < 30 else "neutral")
    else:
        result["rsi"] = None
        result["rsi_signal"] = "neutral"

    # --- MACD (12, 26, 9) ---
    if len(closes) >= 35:
        ema12 = _ema(closes, 12)
        ema26 = _ema(closes, 26)
        macd_line = ema12 - ema26
        # signal line = 9-period EMA of MACD line (approximate with last 9 MACD values)
        macd_values = []
        for i in range(9):
            idx = len(closes) - 9 + i
            e12 = _ema(closes[: idx + 1], 12)
            e26 = _ema(closes[: idx + 1], 26)
            macd_values.append(e12 - e26)
        signal_line = _ema_from_values(macd_values, 9)
        result["macd"] = round(macd_line, 4)
        result["macd_signal_line"] = round(signal_line, 4)
        result["macd_signal"] = "bullish" if macd_line > signal_line else "bearish"
    else:
        result["macd"] = None
        result["macd_signal"] = "neutral"

    return result


def _ema(data, period: int) -> float:
    """Compute the EMA of *data* with given *period*. Returns the last value."""
    import numpy as np

    if len(data) < period:
        return float(np.mean(data))
    k = 2.0 / (period + 1)
    ema = float(data[-period])
    for val in data[-period + 1 :]:
        ema = float(val) * k + ema * (1 - k)
    return ema


def _ema_from_values(values, period: int) -> float:
    """EMA from a plain list of floats."""
    if len(values) < period:
        return sum(values) / len(values) if values else 0.0
    k = 2.0 / (period + 1)
    ema = values[0]
    for v in values[1:]:
        ema = v * k + ema * (1 - k)
    return ema


def _run_historical_dcf(ticker, current_price: float) -> Dict[str, Any]:
    """Run a simple DCF using ticker.info fundamentals available at query time."""
    try:
        from src.agents.valuation_agent import calculate_dcf

        info = ticker.info
        fcf = info.get("freeCashflow") or 0
        if not fcf or fcf <= 0:
            return {"error": "No positive FCF available", "fair_value": None, "upside_pct": None}

        shares = info.get("sharesOutstanding") or 1
        total_debt = info.get("totalDebt") or 0
        total_cash = info.get("totalCash") or 0
        net_debt = total_debt - total_cash
        beta = info.get("beta") or 1.0
        growth = info.get("revenueGrowth") or 0.05

        discount_rate = 0.04 + beta * 0.06  # simplified CAPM
        terminal_growth = 0.025

        result = calculate_dcf(
            fcf=fcf,
            growth_rate=growth,
            discount_rate=discount_rate,
            terminal_growth=terminal_growth,
            projection_years=10,
            net_debt=net_debt,
            shares=shares,
        )

        fair_value = result["fair_value"]
        upside = ((fair_value - current_price) / current_price) * 100 if current_price else None

        return {
            "fair_value": round(fair_value, 2),
            "upside_pct": round(upside, 2) if upside is not None else None,
            "discount_rate": round(discount_rate, 4),
            "growth_rate": round(growth, 4),
        }
    except Exception as e:
        logger.warning(f"DCF backtest failed: {e}")
        return {"error": str(e), "fair_value": None, "upside_pct": None}


def _compute_actual_outcome(after_df, bt_price: float, forward_days: int) -> Dict[str, Any]:
    """Compute what actually happened after the backtest date."""
    if after_df.empty:
        return {
            "final_price": None,
            "return_pct": None,
            "max_price": None,
            "min_price": None,
            "actual_direction": None,
            "trading_days": 0,
        }

    closes = after_df["Close"].values
    final_price = float(closes[-1])
    max_price = float(after_df["High"].max())
    min_price = float(after_df["Low"].min())
    return_pct = ((final_price - bt_price) / bt_price) * 100

    return {
        "final_price": round(final_price, 2),
        "return_pct": round(return_pct, 2),
        "max_price": round(max_price, 2),
        "min_price": round(min_price, 2),
        "max_drawup_pct": round(((max_price - bt_price) / bt_price) * 100, 2),
        "max_drawdown_pct": round(((min_price - bt_price) / bt_price) * 100, 2),
        "actual_direction": "bullish" if return_pct > 0 else "bearish",
        "trading_days": len(closes),
    }
