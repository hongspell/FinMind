"""
FinMind - yfinance Retry Utilities

Shared retry-enabled wrappers for all yfinance calls.
Uses tenacity for exponential backoff on transient network errors.
"""

import logging
import socket
from typing import Any, Dict, List, Optional

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

# Transient exceptions that warrant a retry
RETRYABLE_EXCEPTIONS = (
    ConnectionError,
    TimeoutError,
    socket.error,
    OSError,
)


def yf_retry(max_attempts: int = 3, min_wait: int = 1, max_wait: int = 10):
    """yfinance retry decorator factory."""
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=1, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )


@yf_retry()
def get_ticker_info(symbol: str) -> Dict[str, Any]:
    """Fetch ticker.info with retry."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    return ticker.info


@yf_retry()
def get_ticker_history(symbol: str, period: str = "1y", start=None, end=None) -> Any:
    """Fetch ticker.history with retry. Returns a pandas DataFrame."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    if start:
        return ticker.history(start=start, end=end)
    return ticker.history(period=period)


@yf_retry()
def get_ticker_financials(symbol: str) -> Dict[str, Any]:
    """Fetch financial statements (income_stmt, balance_sheet, cashflow) with retry."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    return {
        "income_stmt": ticker.income_stmt,
        "balance_sheet": ticker.balance_sheet,
        "cashflow": ticker.cashflow,
    }


@yf_retry()
def get_ticker_news(symbol: str, max_items: int = 15) -> List[Dict]:
    """Fetch ticker news with retry."""
    import yfinance as yf

    ticker = yf.Ticker(symbol)
    news = ticker.news or []
    return news[:max_items]
