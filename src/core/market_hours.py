"""
FinMind - Market Hours & Extended Trading Module

Handles market session detection and extended hours price weighting.
Supports pre-market, regular hours, and after-hours trading data.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from enum import Enum
from typing import Dict, Optional, Tuple
import pytz


class MarketSession(Enum):
    """Market trading session types"""
    PRE_MARKET = "pre_market"        # 4:00 AM - 9:30 AM ET
    REGULAR = "regular"               # 9:30 AM - 4:00 PM ET
    AFTER_HOURS = "after_hours"       # 4:00 PM - 8:00 PM ET
    CLOSED = "closed"                 # 8:00 PM - 4:00 AM ET
    WEEKEND = "weekend"               # Saturday & Sunday
    HOLIDAY = "holiday"               # Market holidays


@dataclass
class ExtendedPrice:
    """Extended hours price data with weighting"""
    regular_price: Optional[float]           # Regular market close/current price
    pre_market_price: Optional[float]        # Pre-market price
    after_hours_price: Optional[float]       # After-hours price
    pre_market_volume: Optional[int]         # Pre-market volume
    after_hours_volume: Optional[int]        # After-hours volume
    regular_volume: Optional[int]            # Regular market volume
    weighted_price: float                    # Calculated weighted price
    session: MarketSession                   # Current session
    price_source: str                        # Description of price source
    confidence: float                        # Confidence in the weighted price (0-1)
    timestamp: datetime                      # When data was fetched


class MarketStatusDetector:
    """
    Detects current market session and calculates appropriate price weighting.

    US Market Hours (Eastern Time):
    - Pre-market:   4:00 AM - 9:30 AM ET
    - Regular:      9:30 AM - 4:00 PM ET
    - After-hours:  4:00 PM - 8:00 PM ET
    - Closed:       8:00 PM - 4:00 AM ET
    """

    # US Market time boundaries (Eastern Time)
    PRE_MARKET_START = time(4, 0)    # 4:00 AM
    MARKET_OPEN = time(9, 30)         # 9:30 AM
    MARKET_CLOSE = time(16, 0)        # 4:00 PM
    AFTER_HOURS_END = time(20, 0)     # 8:00 PM

    # Major US market holidays (2024-2026) - dates when market is closed
    HOLIDAYS = {
        # 2024
        datetime(2024, 1, 1).date(),    # New Year's Day
        datetime(2024, 1, 15).date(),   # MLK Day
        datetime(2024, 2, 19).date(),   # Presidents Day
        datetime(2024, 3, 29).date(),   # Good Friday
        datetime(2024, 5, 27).date(),   # Memorial Day
        datetime(2024, 6, 19).date(),   # Juneteenth
        datetime(2024, 7, 4).date(),    # Independence Day
        datetime(2024, 9, 2).date(),    # Labor Day
        datetime(2024, 11, 28).date(),  # Thanksgiving
        datetime(2024, 12, 25).date(),  # Christmas
        # 2025
        datetime(2025, 1, 1).date(),
        datetime(2025, 1, 20).date(),
        datetime(2025, 2, 17).date(),
        datetime(2025, 4, 18).date(),
        datetime(2025, 5, 26).date(),
        datetime(2025, 6, 19).date(),
        datetime(2025, 7, 4).date(),
        datetime(2025, 9, 1).date(),
        datetime(2025, 11, 27).date(),
        datetime(2025, 12, 25).date(),
        # 2026
        datetime(2026, 1, 1).date(),
        datetime(2026, 1, 19).date(),
        datetime(2026, 2, 16).date(),
        datetime(2026, 4, 3).date(),
        datetime(2026, 5, 25).date(),
        datetime(2026, 6, 19).date(),
        datetime(2026, 7, 3).date(),   # Observed
        datetime(2026, 9, 7).date(),
        datetime(2026, 11, 26).date(),
        datetime(2026, 12, 25).date(),
    }

    def __init__(self):
        self.eastern_tz = pytz.timezone('US/Eastern')

    def get_eastern_time(self, dt: Optional[datetime] = None) -> datetime:
        """Convert datetime to US Eastern time"""
        if dt is None:
            dt = datetime.now(pytz.UTC)
        elif dt.tzinfo is None:
            # Assume local time, convert to UTC first
            dt = pytz.UTC.localize(dt)
        return dt.astimezone(self.eastern_tz)

    def get_current_session(self, dt: Optional[datetime] = None) -> MarketSession:
        """
        Determine the current market session.

        Args:
            dt: Datetime to check (defaults to now)

        Returns:
            MarketSession enum value
        """
        et_time = self.get_eastern_time(dt)
        current_time = et_time.time()
        current_date = et_time.date()

        # Check weekend
        if et_time.weekday() >= 5:  # Saturday = 5, Sunday = 6
            return MarketSession.WEEKEND

        # Check holidays
        if current_date in self.HOLIDAYS:
            return MarketSession.HOLIDAY

        # Check time-based sessions
        if self.PRE_MARKET_START <= current_time < self.MARKET_OPEN:
            return MarketSession.PRE_MARKET
        elif self.MARKET_OPEN <= current_time < self.MARKET_CLOSE:
            return MarketSession.REGULAR
        elif self.MARKET_CLOSE <= current_time < self.AFTER_HOURS_END:
            return MarketSession.AFTER_HOURS
        else:
            return MarketSession.CLOSED

    def get_session_info(self, dt: Optional[datetime] = None) -> Dict:
        """
        Get detailed information about the current session.

        Returns:
            Dict with session info including time to next session
        """
        et_time = self.get_eastern_time(dt)
        session = self.get_current_session(dt)

        info = {
            'session': session,
            'session_name': session.value,
            'eastern_time': et_time.strftime('%Y-%m-%d %H:%M:%S %Z'),
            'is_trading': session in [MarketSession.PRE_MARKET, MarketSession.REGULAR, MarketSession.AFTER_HOURS],
            'is_regular_hours': session == MarketSession.REGULAR,
        }

        # Calculate time to next session
        current_time = et_time.time()
        if session == MarketSession.PRE_MARKET:
            next_session = "Regular Market Open"
            minutes_until = self._minutes_between_times(current_time, self.MARKET_OPEN)
        elif session == MarketSession.REGULAR:
            next_session = "Market Close"
            minutes_until = self._minutes_between_times(current_time, self.MARKET_CLOSE)
        elif session == MarketSession.AFTER_HOURS:
            next_session = "After-Hours End"
            minutes_until = self._minutes_between_times(current_time, self.AFTER_HOURS_END)
        else:
            next_session = "Pre-Market Open"
            if current_time < self.PRE_MARKET_START:
                minutes_until = self._minutes_between_times(current_time, self.PRE_MARKET_START)
            else:
                # After 8 PM, next pre-market is tomorrow
                minutes_until = self._minutes_between_times(current_time, time(23, 59)) + \
                               self._minutes_between_times(time(0, 0), self.PRE_MARKET_START) + 1

        info['next_session'] = next_session
        info['minutes_until_next'] = minutes_until

        return info

    def _minutes_between_times(self, t1: time, t2: time) -> int:
        """Calculate minutes between two times"""
        minutes1 = t1.hour * 60 + t1.minute
        minutes2 = t2.hour * 60 + t2.minute
        return minutes2 - minutes1


class ExtendedHoursPriceCalculator:
    """
    Calculates weighted price based on extended trading hours data.

    Weighting Strategy:
    - Regular hours: Use regular price (highest confidence)
    - Pre-market: Weight based on volume and time proximity to open
    - After-hours: Weight based on volume and time since close
    - Closed: Use most recent available price
    """

    def __init__(self):
        self.detector = MarketStatusDetector()

    def calculate_weighted_price(
        self,
        regular_price: Optional[float],
        pre_market_price: Optional[float] = None,
        after_hours_price: Optional[float] = None,
        pre_market_volume: Optional[int] = None,
        after_hours_volume: Optional[int] = None,
        regular_volume: Optional[int] = None,
        dt: Optional[datetime] = None
    ) -> ExtendedPrice:
        """
        Calculate the best representative price based on current session.

        Args:
            regular_price: Regular market price (close or current)
            pre_market_price: Pre-market trading price
            after_hours_price: After-hours trading price
            pre_market_volume: Pre-market trading volume
            after_hours_volume: After-hours trading volume
            regular_volume: Regular session volume (for comparison)
            dt: Datetime to evaluate (defaults to now)

        Returns:
            ExtendedPrice with weighted price and metadata
        """
        session = self.detector.get_current_session(dt)
        session_info = self.detector.get_session_info(dt)

        # Default values
        weighted_price = regular_price or 0.0
        price_source = "Regular market price"
        confidence = 1.0

        if session == MarketSession.REGULAR:
            # During regular hours, use regular price with full confidence
            weighted_price = regular_price or 0.0
            price_source = "Real-time market price"
            confidence = 1.0

        elif session == MarketSession.PRE_MARKET:
            weighted_price, price_source, confidence = self._calculate_pre_market_weight(
                regular_price, pre_market_price, pre_market_volume, regular_volume, session_info
            )

        elif session == MarketSession.AFTER_HOURS:
            weighted_price, price_source, confidence = self._calculate_after_hours_weight(
                regular_price, after_hours_price, after_hours_volume, regular_volume, session_info
            )

        elif session in [MarketSession.CLOSED, MarketSession.WEEKEND, MarketSession.HOLIDAY]:
            # Market closed - use most recent price with context
            weighted_price, price_source, confidence = self._calculate_closed_market_price(
                regular_price, after_hours_price, pre_market_price
            )

        return ExtendedPrice(
            regular_price=regular_price,
            pre_market_price=pre_market_price,
            after_hours_price=after_hours_price,
            pre_market_volume=pre_market_volume,
            after_hours_volume=after_hours_volume,
            regular_volume=regular_volume,
            weighted_price=weighted_price,
            session=session,
            price_source=price_source,
            confidence=confidence,
            timestamp=datetime.now(pytz.UTC)
        )

    def _calculate_pre_market_weight(
        self,
        regular_price: Optional[float],
        pre_market_price: Optional[float],
        pre_market_volume: Optional[int],
        regular_volume: Optional[int],
        session_info: Dict
    ) -> Tuple[float, str, float]:
        """Calculate weighted price during pre-market session"""

        if not regular_price:
            if pre_market_price:
                return pre_market_price, "Pre-market price (no regular price available)", 0.6
            return 0.0, "No price data available", 0.0

        if not pre_market_price:
            return regular_price, "Previous close (no pre-market data)", 0.8

        # Calculate weight based on:
        # 1. Volume ratio (higher pre-market volume = more significant)
        # 2. Time proximity to open (closer to open = pre-market more relevant)

        minutes_to_open = session_info.get('minutes_until_next', 330)  # Max 5.5 hours
        time_weight = 1 - (minutes_to_open / 330)  # 0 at 4 AM, 1 at 9:30 AM
        time_weight = max(0.1, min(0.9, time_weight))  # Clamp between 0.1 and 0.9

        # Volume-based adjustment
        volume_weight = 0.5  # Default
        if pre_market_volume and regular_volume and regular_volume > 0:
            volume_ratio = pre_market_volume / regular_volume
            # If pre-market volume is > 10% of regular, give more weight
            if volume_ratio > 0.1:
                volume_weight = min(0.8, 0.5 + volume_ratio)
            elif volume_ratio < 0.01:
                volume_weight = 0.2

        # Combine weights
        pre_market_weight = (time_weight * 0.6 + volume_weight * 0.4)
        pre_market_weight = max(0.1, min(0.7, pre_market_weight))  # Cap at 70%

        # Calculate weighted price
        weighted_price = (regular_price * (1 - pre_market_weight) +
                         pre_market_price * pre_market_weight)

        # Calculate price change
        change_pct = ((pre_market_price - regular_price) / regular_price * 100) if regular_price else 0

        # Confidence based on volume and price stability
        confidence = 0.7 + (volume_weight * 0.2)

        price_source = (f"Weighted: {(1-pre_market_weight)*100:.0f}% prev close + "
                       f"{pre_market_weight*100:.0f}% pre-market "
                       f"(Pre-mkt: ${pre_market_price:.2f}, {change_pct:+.2f}%)")

        return weighted_price, price_source, confidence

    def _calculate_after_hours_weight(
        self,
        regular_price: Optional[float],
        after_hours_price: Optional[float],
        after_hours_volume: Optional[int],
        regular_volume: Optional[int],
        session_info: Dict
    ) -> Tuple[float, str, float]:
        """Calculate weighted price during after-hours session"""

        if not regular_price:
            if after_hours_price:
                return after_hours_price, "After-hours price (no regular price)", 0.6
            return 0.0, "No price data available", 0.0

        if not after_hours_price:
            return regular_price, "Today's close (no after-hours data)", 0.9

        # Time since close affects weight (more time = AH more relevant for next day)
        minutes_since_close = 240 - session_info.get('minutes_until_next', 240)  # 4 hours max
        time_weight = minutes_since_close / 240  # 0 at close, 1 at 8 PM
        time_weight = max(0.1, min(0.6, time_weight))  # Clamp

        # Volume-based adjustment
        volume_weight = 0.3
        if after_hours_volume and regular_volume and regular_volume > 0:
            volume_ratio = after_hours_volume / regular_volume
            if volume_ratio > 0.05:
                volume_weight = min(0.6, 0.3 + volume_ratio * 2)

        # Combine weights - after-hours typically less weight than pre-market
        ah_weight = (time_weight * 0.5 + volume_weight * 0.5)
        ah_weight = max(0.1, min(0.5, ah_weight))  # Cap at 50%

        # Calculate weighted price
        weighted_price = regular_price * (1 - ah_weight) + after_hours_price * ah_weight

        # Calculate price change
        change_pct = ((after_hours_price - regular_price) / regular_price * 100) if regular_price else 0

        confidence = 0.75 + (volume_weight * 0.15)

        price_source = (f"Weighted: {(1-ah_weight)*100:.0f}% close + "
                       f"{ah_weight*100:.0f}% after-hours "
                       f"(AH: ${after_hours_price:.2f}, {change_pct:+.2f}%)")

        return weighted_price, price_source, confidence

    def _calculate_closed_market_price(
        self,
        regular_price: Optional[float],
        after_hours_price: Optional[float],
        pre_market_price: Optional[float]
    ) -> Tuple[float, str, float]:
        """Determine best price when market is closed"""

        # Priority: after-hours > regular > pre-market
        if after_hours_price and regular_price:
            # Use after-hours as it's more recent
            change_pct = ((after_hours_price - regular_price) / regular_price * 100) if regular_price else 0
            return (after_hours_price,
                   f"After-hours close: ${after_hours_price:.2f} ({change_pct:+.2f}% from close)",
                   0.85)
        elif regular_price:
            return regular_price, f"Market close: ${regular_price:.2f}", 0.9
        elif pre_market_price:
            return pre_market_price, f"Pre-market: ${pre_market_price:.2f} (stale)", 0.5
        else:
            return 0.0, "No price data available", 0.0


def get_market_status_summary(dt: Optional[datetime] = None) -> str:
    """Get a human-readable market status summary"""
    detector = MarketStatusDetector()
    info = detector.get_session_info(dt)

    session = info['session']
    et_time = info['eastern_time']

    status_map = {
        MarketSession.PRE_MARKET: "🌅 Pre-Market Trading",
        MarketSession.REGULAR: "📈 Market Open",
        MarketSession.AFTER_HOURS: "🌙 After-Hours Trading",
        MarketSession.CLOSED: "🔒 Market Closed",
        MarketSession.WEEKEND: "📅 Weekend - Market Closed",
        MarketSession.HOLIDAY: "🎉 Holiday - Market Closed",
    }

    summary = f"{status_map.get(session, 'Unknown')} | ET: {et_time}"

    if info['is_trading']:
        summary += f" | {info['minutes_until_next']} min until {info['next_session']}"

    return summary
