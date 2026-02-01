"""
FinMind - Quote Service

Centralized yfinance quote fetching with cache integration.
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.core.yfinance_utils import get_ticker_info, get_ticker_history

logger = logging.getLogger(__name__)


def _fetch_single_quote(symbol: str) -> Optional[Dict[str, Any]]:
    """Fetch a single stock quote via yfinance (blocking, for use in thread pool)."""
    try:
        info = get_ticker_info(symbol.upper())

        current_price = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        if not current_price:
            return None

        prev_close = info.get("previousClose", current_price)
        change = current_price - prev_close
        change_percent = (change / prev_close * 100) if prev_close else 0

        return {
            "symbol": symbol.upper(),
            "price": current_price,
            "change": round(change, 4),
            "change_percent": round(change_percent, 4),
            "volume": info.get("volume", 0),
            "timestamp": datetime.now().isoformat(),
        }
    except Exception:
        logger.debug(f"Failed to fetch quote for {symbol}")
        return None


def fetch_quotes_batch(symbols: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple quotes in parallel using a thread pool."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    quotes = []
    max_workers = min(5, len(symbols))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_fetch_single_quote, s): s for s in symbols}
        for future in as_completed(futures):
            result = future.result()
            if result:
                quotes.append(result)

    # Restore original order
    symbol_order = {s.upper(): i for i, s in enumerate(symbols)}
    quotes.sort(key=lambda q: symbol_order.get(q["symbol"], 999))
    return quotes


def fetch_price_history(symbol: str, period: str = "1y") -> Optional[Dict[str, Any]]:
    """Fetch historical price data for a symbol."""
    try:
        hist = get_ticker_history(symbol.upper(), period=period)

        if hist.empty:
            return None

        return {
            "symbol": symbol.upper(),
            "period": period,
            "dates": [d.strftime("%Y-%m-%d") for d in hist.index],
            "open": [round(v, 4) for v in hist["Open"].tolist()],
            "high": [round(v, 4) for v in hist["High"].tolist()],
            "low": [round(v, 4) for v in hist["Low"].tolist()],
            "close": [round(v, 4) for v in hist["Close"].tolist()],
            "volume": [int(v) for v in hist["Volume"].tolist()],
        }
    except Exception as e:
        logger.debug(f"Failed to fetch history for {symbol}: {e}")
        return None


def search_symbol(query: str) -> List[Dict[str, Any]]:
    """Search for a ticker symbol by validating it via yfinance."""
    try:
        info = get_ticker_info(query.upper())

        # yfinance doesn't have a true search API; we validate the ticker exists
        name = info.get("shortName") or info.get("longName")
        if not name:
            return []

        return [
            {
                "symbol": query.upper(),
                "name": name,
                "exchange": info.get("exchange", ""),
                "type": info.get("quoteType", "EQUITY"),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
            }
        ]
    except Exception:
        return []


def fetch_volatility(symbol: str, days: int = 252) -> Optional[Dict[str, Any]]:
    """Calculate annualized and daily volatility from historical data."""
    try:
        import numpy as np

        hist = get_ticker_history(symbol.upper(), period=f"{max(days, 30)}d")

        if hist.empty or len(hist) < 20:
            return None

        closes = hist["Close"].values
        log_returns = np.diff(np.log(closes))

        daily_vol = float(np.std(log_returns))
        annual_vol = daily_vol * np.sqrt(252)

        return {
            "symbol": symbol.upper(),
            "daily_volatility": round(daily_vol, 6),
            "annual_volatility": round(annual_vol, 6),
            "data_points": len(log_returns),
            "period_days": days,
        }
    except Exception as e:
        logger.debug(f"Failed to calculate volatility for {symbol}: {e}")
        return None
