"""
Technical Analysis Agent
技术分析Agent - 价格行为和技术指标分析
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import statistics


class TrendDirection(Enum):
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


class SignalStrength(Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    NEUTRAL = "neutral"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class Timeframe(Enum):
    """时间框架"""
    SHORT_TERM = "short_term"      # 短期: 1-2周
    MEDIUM_TERM = "medium_term"    # 中期: 1-3月
    LONG_TERM = "long_term"        # 长期: 3-12月


@dataclass
class TimeframeAnalysis:
    """单个时间框架的分析结果"""
    timeframe: Timeframe
    timeframe_label: str           # 显示名称，如"短期(1-2周)"
    signal: SignalStrength
    trend: TrendDirection
    trend_strength: float          # 0-1
    confidence: float              # 0-1
    key_indicators: List[str]      # 该时间框架的关键指标描述
    description: str               # 分析描述


@dataclass
class TechnicalIndicator:
    """技术指标结果"""
    name: str
    value: float
    signal: SignalStrength
    description: str
    weight: float = 1.0


@dataclass
class SupportResistance:
    """支撑/阻力位"""
    level: float
    type: str  # "support" or "resistance"
    strength: float  # 0-1
    touches: int  # 触及次数
    last_touch: Optional[datetime] = None


@dataclass
class PatternMatch:
    """形态识别结果"""
    pattern_name: str
    confidence: float
    direction: TrendDirection
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    description: str = ""


@dataclass
class TechnicalAnalysisResult:
    """技术分析结果"""
    symbol: str
    timestamp: datetime
    current_price: float

    # 趋势分析（综合）
    trend: TrendDirection
    trend_strength: float  # 0-1

    # 技术指标
    indicators: List[TechnicalIndicator]

    # 支撑阻力
    support_levels: List[SupportResistance]
    resistance_levels: List[SupportResistance]

    # 形态识别
    patterns: List[PatternMatch]

    # 综合信号
    overall_signal: SignalStrength
    signal_confidence: float

    # 多时间框架分析
    timeframe_analyses: List[TimeframeAnalysis] = field(default_factory=list)

    # 关键价位
    key_levels: Dict[str, float] = field(default_factory=dict)

    # 分析描述
    summary: str = ""
    key_observations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class TechnicalCalculator:
    """技术指标计算器"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> Optional[float]:
        """简单移动平均"""
        if len(prices) < period:
            return None
        return sum(prices[-period:]) / period
    
    @staticmethod
    def ema(prices: List[float], period: int) -> Optional[float]:
        """指数移动平均"""
        if len(prices) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = prices[0]
        for price in prices[1:]:
            ema = (price - ema) * multiplier + ema
        return ema
    
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> Optional[float]:
        """相对强弱指数"""
        if len(prices) < period + 1:
            return None
        
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Optional[Dict]:
        """MACD指标"""
        if len(prices) < slow + signal:
            return None
        
        calc = TechnicalCalculator
        
        # 计算快线和慢线EMA
        ema_fast = calc.ema(prices, fast)
        ema_slow = calc.ema(prices, slow)
        
        if ema_fast is None or ema_slow is None:
            return None
        
        macd_line = ema_fast - ema_slow
        
        # 计算历史MACD线用于signal
        macd_history = []
        for i in range(slow + signal, len(prices) + 1):
            subset = prices[:i]
            ef = calc.ema(subset, fast)
            es = calc.ema(subset, slow)
            if ef and es:
                macd_history.append(ef - es)
        
        signal_line = calc.ema(macd_history, signal) if len(macd_history) >= signal else None
        histogram = macd_line - signal_line if signal_line else None
        
        return {
            "macd": macd_line,
            "signal": signal_line,
            "histogram": histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Optional[Dict]:
        """布林带"""
        if len(prices) < period:
            return None
        
        sma = TechnicalCalculator.sma(prices, period)
        std = statistics.stdev(prices[-period:])
        
        return {
            "upper": sma + std_dev * std,
            "middle": sma,
            "lower": sma - std_dev * std,
            "bandwidth": (std_dev * std * 2) / sma * 100
        }
    
    @staticmethod
    def stochastic(highs: List[float], lows: List[float], closes: List[float], 
                   k_period: int = 14, d_period: int = 3) -> Optional[Dict]:
        """随机指标"""
        if len(closes) < k_period:
            return None
        
        highest_high = max(highs[-k_period:])
        lowest_low = min(lows[-k_period:])
        
        if highest_high == lowest_low:
            k = 50
        else:
            k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
        
        # 计算%D (K的SMA)
        k_values = []
        for i in range(d_period):
            if len(closes) >= k_period + i:
                hh = max(highs[-(k_period+i):len(highs)-i] if i > 0 else highs[-k_period:])
                ll = min(lows[-(k_period+i):len(lows)-i] if i > 0 else lows[-k_period:])
                c = closes[-(i+1)]
                if hh != ll:
                    k_values.append(((c - ll) / (hh - ll)) * 100)
        
        d = sum(k_values) / len(k_values) if k_values else k
        
        return {"k": k, "d": d}
    
    @staticmethod
    def atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
        """平均真实范围"""
        if len(closes) < period + 1:
            return None
        
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close = abs(highs[i] - closes[i-1])
            low_close = abs(lows[i] - closes[i-1])
            true_ranges.append(max(high_low, high_close, low_close))
        
        return sum(true_ranges[-period:]) / period
    
    @staticmethod
    def adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[Dict]:
        """平均趋向指数"""
        if len(closes) < period * 2:
            return None
        
        plus_dm = []
        minus_dm = []
        tr = []
        
        for i in range(1, len(closes)):
            high_diff = highs[i] - highs[i-1]
            low_diff = lows[i-1] - lows[i]
            
            plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
            minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)
            
            tr.append(max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1])
            ))
        
        atr = sum(tr[-period:]) / period
        plus_di = (sum(plus_dm[-period:]) / period) / atr * 100 if atr > 0 else 0
        minus_di = (sum(minus_dm[-period:]) / period) / atr * 100 if atr > 0 else 0
        
        dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
        
        return {
            "adx": dx,  # 简化版，实际应该是DX的平均
            "plus_di": plus_di,
            "minus_di": minus_di
        }
    
    @staticmethod
    def obv(closes: List[float], volumes: List[float]) -> Optional[float]:
        """能量潮"""
        if len(closes) < 2:
            return None
        
        obv = 0
        for i in range(1, len(closes)):
            if closes[i] > closes[i-1]:
                obv += volumes[i]
            elif closes[i] < closes[i-1]:
                obv -= volumes[i]
        
        return obv
    
    @staticmethod
    def vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Optional[float]:
        """成交量加权平均价"""
        if not volumes or sum(volumes) == 0:
            return None
        
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
        return sum(tp * v for tp, v in zip(typical_prices, volumes)) / sum(volumes)


class PatternRecognizer:
    """形态识别器"""
    
    @staticmethod
    def detect_head_and_shoulders(highs: List[float], lows: List[float], closes: List[float]) -> Optional[PatternMatch]:
        """头肩顶/底形态"""
        if len(closes) < 30:
            return None
        
        # 简化的头肩识别逻辑
        # 实际应用需要更复杂的峰值检测
        peaks = []
        for i in range(2, len(highs) - 2):
            if highs[i] > highs[i-1] and highs[i] > highs[i-2] and \
               highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                peaks.append((i, highs[i]))
        
        if len(peaks) >= 3:
            # 检查是否形成头肩形态
            for i in range(len(peaks) - 2):
                left, head, right = peaks[i:i+3]
                if head[1] > left[1] and head[1] > right[1]:
                    if abs(left[1] - right[1]) / head[1] < 0.05:  # 两肩高度相近
                        neckline = min(lows[left[0]:right[0]+1])
                        target = neckline - (head[1] - neckline)
                        
                        return PatternMatch(
                            pattern_name="Head and Shoulders",
                            confidence=0.7,
                            direction=TrendDirection.BEARISH,
                            target_price=target,
                            stop_loss=head[1] * 1.02,
                            description="看跌反转形态，颈线突破确认"
                        )
        
        return None
    
    @staticmethod
    def detect_double_top_bottom(highs: List[float], lows: List[float], closes: List[float]) -> Optional[PatternMatch]:
        """双顶/双底形态"""
        if len(closes) < 20:
            return None
        
        # 检测双顶
        recent_highs = highs[-20:]
        recent_lows = lows[-20:]

        if len(recent_highs) < 20:
            return None

        # 找前半段和后半段的最高点
        first_half = recent_highs[:10]
        second_half = recent_highs[10:]

        max_val1 = max(first_half)
        max_val2 = max(second_half)
        max_idx1 = first_half.index(max_val1)
        max_idx2 = second_half.index(max_val2) + 10

        if recent_highs[max_idx1] > 0 and abs(max_val1 - max_val2) / recent_highs[max_idx1] < 0.03:
            neckline = min(recent_lows[max_idx1:max_idx2+1]) if max_idx2 > max_idx1 else min(recent_lows)
            if closes[-1] < neckline:
                target = neckline - (max_val1 - neckline)
                return PatternMatch(
                    pattern_name="Double Top",
                    confidence=0.65,
                    direction=TrendDirection.BEARISH,
                    target_price=target,
                    description="双顶形态确认，趋势可能反转"
                )

        # 检测双底
        first_half_lows = recent_lows[:10]
        second_half_lows = recent_lows[10:]

        min_val1 = min(first_half_lows)
        min_val2 = min(second_half_lows)
        min_idx1 = first_half_lows.index(min_val1)
        min_idx2 = second_half_lows.index(min_val2) + 10

        if min_val1 > 0 and abs(min_val1 - min_val2) / min_val1 < 0.03:
            neckline = max(recent_highs[min_idx1:min_idx2+1]) if min_idx2 > min_idx1 else max(recent_highs)
            if closes[-1] > neckline:
                target = neckline + (neckline - min_val1)
                return PatternMatch(
                    pattern_name="Double Bottom",
                    confidence=0.65,
                    direction=TrendDirection.BULLISH,
                    target_price=target,
                    description="双底形态确认，趋势可能反转"
                )

        return None
    
    @staticmethod
    def detect_triangle(highs: List[float], lows: List[float], closes: List[float]) -> Optional[PatternMatch]:
        """三角形整理形态"""
        if len(closes) < 15:
            return None
        
        # 计算高点和低点的趋势
        recent_highs = highs[-15:]
        recent_lows = lows[-15:]
        
        high_slope = (recent_highs[-1] - recent_highs[0]) / len(recent_highs)
        low_slope = (recent_lows[-1] - recent_lows[0]) / len(recent_lows)
        
        # 收敛三角形
        if high_slope < 0 and low_slope > 0:
            avg_price = (closes[-1] + closes[0]) / 2
            return PatternMatch(
                pattern_name="Symmetrical Triangle",
                confidence=0.6,
                direction=TrendDirection.NEUTRAL,
                target_price=avg_price * 1.05 if closes[-1] > avg_price else avg_price * 0.95,
                description="对称三角形，等待突破方向确认"
            )
        
        # 上升三角形
        if abs(high_slope) < 0.001 and low_slope > 0:
            resistance = max(recent_highs)
            return PatternMatch(
                pattern_name="Ascending Triangle",
                confidence=0.65,
                direction=TrendDirection.BULLISH,
                target_price=resistance * 1.05,
                description="上升三角形，通常看涨突破"
            )
        
        # 下降三角形
        if high_slope < 0 and abs(low_slope) < 0.001:
            support = min(recent_lows)
            return PatternMatch(
                pattern_name="Descending Triangle",
                confidence=0.65,
                direction=TrendDirection.BEARISH,
                target_price=support * 0.95,
                description="下降三角形，通常看跌突破"
            )
        
        return None


class TechnicalAgent:
    """技术分析Agent"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.calculator = TechnicalCalculator()
        self.pattern_recognizer = PatternRecognizer()

    async def analyze_price_action(self, context, inputs: Dict) -> 'TechnicalAnalysisResult':
        """ChainExecutor 调用的适配方法"""
        target = context.target if hasattr(context, 'target') else 'UNKNOWN'

        # 从 inputs 中提取市场数据
        market_data = inputs.get('market_data', {})

        # 支持两种数据格式
        if isinstance(market_data, dict) and 'price_history' in market_data:
            price_data = market_data['price_history']
        elif isinstance(market_data, dict):
            price_data = market_data
        else:
            # 尝试从 market_data 对象中提取
            price_data = getattr(market_data, 'data', {}) or {}
            if 'price_history' in price_data:
                price_data = price_data['price_history']

        return await self.analyze(target, price_data)

    async def analyze(
        self,
        symbol: str,
        price_data: Dict[str, List[float]],
        volume_data: Optional[List[float]] = None
    ) -> TechnicalAnalysisResult:
        """执行完整技术分析"""
        
        opens = price_data.get("open", [])
        highs = price_data.get("high", [])
        lows = price_data.get("low", [])
        closes = price_data.get("close", [])
        volumes = volume_data or []
        
        current_price = closes[-1] if closes else 0
        
        # 计算所有技术指标
        indicators = self._calculate_indicators(closes, highs, lows, volumes)
        
        # 识别支撑阻力
        support_levels, resistance_levels = self._find_support_resistance(highs, lows, closes)
        
        # 形态识别
        patterns = self._detect_patterns(highs, lows, closes)
        
        # 确定趋势
        trend, trend_strength = self._determine_trend(closes, indicators)
        
        # 综合信号
        overall_signal, signal_confidence = self._calculate_overall_signal(indicators, trend, patterns)

        # 多时间框架分析
        timeframe_analyses = self._analyze_timeframes(closes, highs, lows, volumes)

        # 关键价位
        key_levels = self._identify_key_levels(current_price, support_levels, resistance_levels, indicators)

        # 生成分析摘要
        summary, observations, warnings = self._generate_analysis_summary(
            symbol, current_price, trend, indicators, patterns, support_levels, resistance_levels
        )

        return TechnicalAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            current_price=current_price,
            trend=trend,
            trend_strength=trend_strength,
            indicators=indicators,
            support_levels=support_levels,
            resistance_levels=resistance_levels,
            patterns=patterns,
            overall_signal=overall_signal,
            signal_confidence=signal_confidence,
            timeframe_analyses=timeframe_analyses,
            key_levels=key_levels,
            summary=summary,
            key_observations=observations,
            warnings=warnings
        )
    
    def _calculate_indicators(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float]
    ) -> List[TechnicalIndicator]:
        """计算所有技术指标"""
        indicators = []
        calc = self.calculator
        
        # Moving Averages
        sma_20 = calc.sma(closes, 20)
        sma_50 = calc.sma(closes, 50)
        sma_200 = calc.sma(closes, 200)
        ema_12 = calc.ema(closes, 12)
        ema_26 = calc.ema(closes, 26)
        
        current = closes[-1] if closes else 0
        
        if sma_20:
            signal = SignalStrength.BUY if current > sma_20 else SignalStrength.SELL
            indicators.append(TechnicalIndicator(
                name="SMA(20)",
                value=sma_20,
                signal=signal,
                description=f"价格{'高于' if current > sma_20 else '低于'}20日均线",
                weight=1.0
            ))
        
        if sma_50:
            signal = SignalStrength.BUY if current > sma_50 else SignalStrength.SELL
            indicators.append(TechnicalIndicator(
                name="SMA(50)",
                value=sma_50,
                signal=signal,
                description=f"价格{'高于' if current > sma_50 else '低于'}50日均线",
                weight=1.2
            ))
        
        if sma_200:
            signal = SignalStrength.STRONG_BUY if current > sma_200 else SignalStrength.STRONG_SELL
            indicators.append(TechnicalIndicator(
                name="SMA(200)",
                value=sma_200,
                signal=signal,
                description=f"价格{'高于' if current > sma_200 else '低于'}200日均线（长期趋势）",
                weight=1.5
            ))
        
        # 金叉/死叉
        if sma_50 and sma_200:
            if sma_50 > sma_200:
                indicators.append(TechnicalIndicator(
                    name="Golden Cross",
                    value=1,
                    signal=SignalStrength.STRONG_BUY,
                    description="50日均线在200日均线之上（金叉）",
                    weight=2.0
                ))
            else:
                indicators.append(TechnicalIndicator(
                    name="Death Cross",
                    value=-1,
                    signal=SignalStrength.STRONG_SELL,
                    description="50日均线在200日均线之下（死叉）",
                    weight=2.0
                ))
        
        # RSI
        rsi = calc.rsi(closes)
        if rsi is not None:
            if rsi > 70:
                signal = SignalStrength.SELL
                desc = f"RSI={rsi:.1f}，超买区域"
            elif rsi < 30:
                signal = SignalStrength.BUY
                desc = f"RSI={rsi:.1f}，超卖区域"
            else:
                signal = SignalStrength.NEUTRAL
                desc = f"RSI={rsi:.1f}，中性区域"
            
            indicators.append(TechnicalIndicator(
                name="RSI(14)",
                value=rsi,
                signal=signal,
                description=desc,
                weight=1.3
            ))
        
        # MACD
        macd = calc.macd(closes)
        if macd and macd.get("histogram") is not None:
            hist = macd["histogram"]
            if hist > 0:
                signal = SignalStrength.BUY
                desc = "MACD柱状图为正，动能向上"
            else:
                signal = SignalStrength.SELL
                desc = "MACD柱状图为负，动能向下"
            
            indicators.append(TechnicalIndicator(
                name="MACD",
                value=hist,
                signal=signal,
                description=desc,
                weight=1.4
            ))
        
        # Bollinger Bands
        bb = calc.bollinger_bands(closes)
        if bb:
            if current > bb["upper"]:
                signal = SignalStrength.SELL
                desc = "价格突破布林带上轨，可能超买"
            elif current < bb["lower"]:
                signal = SignalStrength.BUY
                desc = "价格跌破布林带下轨，可能超卖"
            else:
                pct = (current - bb["lower"]) / (bb["upper"] - bb["lower"])
                signal = SignalStrength.NEUTRAL
                desc = f"价格在布林带内，位于{pct*100:.0f}%位置"
            
            indicators.append(TechnicalIndicator(
                name="Bollinger Bands",
                value=bb["bandwidth"],
                signal=signal,
                description=desc,
                weight=1.2
            ))
        
        # Stochastic
        if highs and lows:
            stoch = calc.stochastic(highs, lows, closes)
            if stoch:
                k, d = stoch["k"], stoch["d"]
                if k > 80 and d > 80:
                    signal = SignalStrength.SELL
                    desc = f"KD指标超买 (K={k:.1f}, D={d:.1f})"
                elif k < 20 and d < 20:
                    signal = SignalStrength.BUY
                    desc = f"KD指标超卖 (K={k:.1f}, D={d:.1f})"
                else:
                    signal = SignalStrength.NEUTRAL
                    desc = f"KD指标中性 (K={k:.1f}, D={d:.1f})"
                
                indicators.append(TechnicalIndicator(
                    name="Stochastic",
                    value=k,
                    signal=signal,
                    description=desc,
                    weight=1.1
                ))
        
        # ATR (波动率)
        if highs and lows:
            atr = calc.atr(highs, lows, closes)
            if atr:
                atr_pct = (atr / current) * 100
                indicators.append(TechnicalIndicator(
                    name="ATR(14)",
                    value=atr,
                    signal=SignalStrength.NEUTRAL,
                    description=f"平均真实范围={atr:.2f} ({atr_pct:.1f}%)",
                    weight=0.5
                ))
        
        # ADX (趋势强度)
        if highs and lows:
            adx = calc.adx(highs, lows, closes)
            if adx:
                adx_val = adx["adx"]
                if adx_val > 25:
                    if adx["plus_di"] > adx["minus_di"]:
                        signal = SignalStrength.BUY
                        desc = f"强势上升趋势 (ADX={adx_val:.1f})"
                    else:
                        signal = SignalStrength.SELL
                        desc = f"强势下降趋势 (ADX={adx_val:.1f})"
                else:
                    signal = SignalStrength.NEUTRAL
                    desc = f"趋势较弱 (ADX={adx_val:.1f})"
                
                indicators.append(TechnicalIndicator(
                    name="ADX",
                    value=adx_val,
                    signal=signal,
                    description=desc,
                    weight=1.3
                ))
        
        # Volume indicators
        if volumes:
            obv = calc.obv(closes, volumes)
            if obv is not None:
                indicators.append(TechnicalIndicator(
                    name="OBV",
                    value=obv,
                    signal=SignalStrength.NEUTRAL,
                    description=f"能量潮指标值: {obv:,.0f}",
                    weight=0.8
                ))
            
            if highs and lows:
                vwap = calc.vwap(highs, lows, closes, volumes)
                if vwap:
                    signal = SignalStrength.BUY if current > vwap else SignalStrength.SELL
                    indicators.append(TechnicalIndicator(
                        name="VWAP",
                        value=vwap,
                        signal=signal,
                        description=f"VWAP={vwap:.2f}，价格{'高于' if current > vwap else '低于'}成交量加权均价",
                        weight=1.0
                    ))
        
        return indicators
    
    def _find_support_resistance(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float]
    ) -> Tuple[List[SupportResistance], List[SupportResistance]]:
        """识别支撑阻力位"""
        supports = []
        resistances = []
        
        if len(closes) < 20:
            return supports, resistances
        
        current = closes[-1]
        
        # 使用局部极值识别关键价位
        window = 5
        for i in range(window, len(closes) - window):
            # 局部高点 -> 阻力
            if highs[i] == max(highs[i-window:i+window+1]):
                level = highs[i]
                if level > current:
                    touches = sum(1 for h in highs if abs(h - level) / level < 0.01)
                    resistances.append(SupportResistance(
                        level=level,
                        type="resistance",
                        strength=min(touches / 5, 1.0),
                        touches=touches
                    ))
            
            # 局部低点 -> 支撑
            if lows[i] == min(lows[i-window:i+window+1]):
                level = lows[i]
                if level < current:
                    touches = sum(1 for l in lows if abs(l - level) / level < 0.01)
                    supports.append(SupportResistance(
                        level=level,
                        type="support",
                        strength=min(touches / 5, 1.0),
                        touches=touches
                    ))
        
        # 合并相近的价位
        supports = self._merge_levels(supports)
        resistances = self._merge_levels(resistances)
        
        # 按距离当前价格排序
        supports.sort(key=lambda x: current - x.level)
        resistances.sort(key=lambda x: x.level - current)
        
        return supports[:5], resistances[:5]
    
    def _merge_levels(self, levels: List[SupportResistance], threshold: float = 0.02) -> List[SupportResistance]:
        """合并相近的支撑/阻力位"""
        if not levels:
            return []
        
        levels.sort(key=lambda x: x.level)
        merged = [levels[0]]
        
        for level in levels[1:]:
            if abs(level.level - merged[-1].level) / merged[-1].level < threshold:
                # 合并：取平均价位，累加强度
                merged[-1] = SupportResistance(
                    level=(merged[-1].level + level.level) / 2,
                    type=level.type,
                    strength=min(merged[-1].strength + level.strength, 1.0),
                    touches=merged[-1].touches + level.touches
                )
            else:
                merged.append(level)
        
        return merged
    
    def _detect_patterns(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float]
    ) -> List[PatternMatch]:
        """识别技术形态"""
        patterns = []
        
        # 头肩形态
        hs = self.pattern_recognizer.detect_head_and_shoulders(highs, lows, closes)
        if hs:
            patterns.append(hs)
        
        # 双顶/双底
        dt = self.pattern_recognizer.detect_double_top_bottom(highs, lows, closes)
        if dt:
            patterns.append(dt)
        
        # 三角形
        tri = self.pattern_recognizer.detect_triangle(highs, lows, closes)
        if tri:
            patterns.append(tri)
        
        return patterns
    
    def _determine_trend(
        self,
        closes: List[float],
        indicators: List[TechnicalIndicator]
    ) -> Tuple[TrendDirection, float]:
        """确定趋势方向和强度"""
        
        if len(closes) < 50:
            return TrendDirection.NEUTRAL, 0.5
        
        # 短期、中期、长期趋势
        short_change = (closes[-1] - closes[-20]) / closes[-20] if closes[-20] else 0
        medium_change = (closes[-1] - closes[-50]) / closes[-50] if closes[-50] else 0
        
        # 根据均线位置
        sma_signals = [ind for ind in indicators if "SMA" in ind.name]
        bullish_count = sum(1 for s in sma_signals if s.signal in [SignalStrength.BUY, SignalStrength.STRONG_BUY])
        bearish_count = sum(1 for s in sma_signals if s.signal in [SignalStrength.SELL, SignalStrength.STRONG_SELL])
        
        # 综合判断
        trend_score = 0
        
        # 价格变化贡献
        if short_change > 0.05:
            trend_score += 2
        elif short_change > 0.02:
            trend_score += 1
        elif short_change < -0.05:
            trend_score -= 2
        elif short_change < -0.02:
            trend_score -= 1
        
        if medium_change > 0.1:
            trend_score += 2
        elif medium_change > 0.05:
            trend_score += 1
        elif medium_change < -0.1:
            trend_score -= 2
        elif medium_change < -0.05:
            trend_score -= 1
        
        # 均线贡献
        trend_score += (bullish_count - bearish_count)
        
        # 确定趋势
        if trend_score >= 4:
            trend = TrendDirection.STRONG_BULLISH
            strength = min(0.9, 0.6 + abs(trend_score) * 0.05)
        elif trend_score >= 2:
            trend = TrendDirection.BULLISH
            strength = 0.6 + abs(trend_score) * 0.05
        elif trend_score <= -4:
            trend = TrendDirection.STRONG_BEARISH
            strength = min(0.9, 0.6 + abs(trend_score) * 0.05)
        elif trend_score <= -2:
            trend = TrendDirection.BEARISH
            strength = 0.6 + abs(trend_score) * 0.05
        else:
            trend = TrendDirection.NEUTRAL
            strength = 0.5
        
        return trend, strength
    
    def _calculate_overall_signal(
        self,
        indicators: List[TechnicalIndicator],
        trend: TrendDirection,
        patterns: List[PatternMatch]
    ) -> Tuple[SignalStrength, float]:
        """计算综合信号"""
        
        # 加权计算指标信号
        signal_scores = {
            SignalStrength.STRONG_BUY: 2,
            SignalStrength.BUY: 1,
            SignalStrength.NEUTRAL: 0,
            SignalStrength.SELL: -1,
            SignalStrength.STRONG_SELL: -2
        }
        
        total_weight = 0
        weighted_score = 0
        
        for ind in indicators:
            weighted_score += signal_scores[ind.signal] * ind.weight
            total_weight += ind.weight
        
        # 趋势贡献
        trend_scores = {
            TrendDirection.STRONG_BULLISH: 2,
            TrendDirection.BULLISH: 1,
            TrendDirection.NEUTRAL: 0,
            TrendDirection.BEARISH: -1,
            TrendDirection.STRONG_BEARISH: -2
        }
        weighted_score += trend_scores[trend] * 2
        total_weight += 2
        
        # 形态贡献
        for pattern in patterns:
            pattern_score = 0
            if pattern.direction == TrendDirection.STRONG_BULLISH:
                pattern_score = 2
            elif pattern.direction == TrendDirection.BULLISH:
                pattern_score = 1
            elif pattern.direction == TrendDirection.BEARISH:
                pattern_score = -1
            elif pattern.direction == TrendDirection.STRONG_BEARISH:
                pattern_score = -2
            
            weighted_score += pattern_score * pattern.confidence * 1.5
            total_weight += pattern.confidence * 1.5
        
        avg_score = weighted_score / total_weight if total_weight > 0 else 0
        
        # 确定信号
        if avg_score >= 1.5:
            signal = SignalStrength.STRONG_BUY
        elif avg_score >= 0.5:
            signal = SignalStrength.BUY
        elif avg_score <= -1.5:
            signal = SignalStrength.STRONG_SELL
        elif avg_score <= -0.5:
            signal = SignalStrength.SELL
        else:
            signal = SignalStrength.NEUTRAL
        
        # 信心度基于所有信号源的一致性（指标 + 趋势 + 形态）
        signal_counts = {}

        # 统计指标信号
        for ind in indicators:
            signal_counts[ind.signal] = signal_counts.get(ind.signal, 0) + ind.weight

        # 统计趋势信号（将TrendDirection映射到SignalStrength）
        trend_to_signal = {
            TrendDirection.STRONG_BULLISH: SignalStrength.STRONG_BUY,
            TrendDirection.BULLISH: SignalStrength.BUY,
            TrendDirection.NEUTRAL: SignalStrength.NEUTRAL,
            TrendDirection.BEARISH: SignalStrength.SELL,
            TrendDirection.STRONG_BEARISH: SignalStrength.STRONG_SELL
        }
        trend_signal = trend_to_signal[trend]
        signal_counts[trend_signal] = signal_counts.get(trend_signal, 0) + 2

        # 统计形态信号
        for pattern in patterns:
            pattern_signal = trend_to_signal.get(pattern.direction, SignalStrength.NEUTRAL)
            pattern_weight = pattern.confidence * 1.5
            signal_counts[pattern_signal] = signal_counts.get(pattern_signal, 0) + pattern_weight

        # 计算信号一致性置信度 - 考虑精确匹配和方向性一致
        if signal_counts:
            # 1. 精确信号一致性
            max_weight = max(signal_counts.values())
            exact_consistency = max_weight / total_weight if total_weight > 0 else 0.5

            # 2. 方向性一致性（看涨 vs 看跌 vs 中性）
            bullish_weight = signal_counts.get(SignalStrength.STRONG_BUY, 0) + signal_counts.get(SignalStrength.BUY, 0)
            bearish_weight = signal_counts.get(SignalStrength.STRONG_SELL, 0) + signal_counts.get(SignalStrength.SELL, 0)
            neutral_weight = signal_counts.get(SignalStrength.NEUTRAL, 0)

            direction_weights = [bullish_weight, bearish_weight, neutral_weight]
            max_direction_weight = max(direction_weights)
            direction_consistency = max_direction_weight / total_weight if total_weight > 0 else 0.5

            # 3. 综合置信度：方向一致性更重要（权重60%），精确一致性次之（权重40%）
            combined_consistency = direction_consistency * 0.6 + exact_consistency * 0.4

            # 4. 检查计算出的信号是否与主导方向一致
            strongest_signal = max(signal_counts, key=signal_counts.get)
            is_bullish_signal = signal in [SignalStrength.STRONG_BUY, SignalStrength.BUY]
            is_bearish_signal = signal in [SignalStrength.STRONG_SELL, SignalStrength.SELL]
            dominant_direction_is_bullish = bullish_weight == max_direction_weight
            dominant_direction_is_bearish = bearish_weight == max_direction_weight

            direction_match = (
                (is_bullish_signal and dominant_direction_is_bullish) or
                (is_bearish_signal and dominant_direction_is_bearish) or
                (signal == SignalStrength.NEUTRAL and neutral_weight == max_direction_weight)
            )

            if direction_match and strongest_signal == signal:
                # 完美一致：方向+精确信号都匹配
                confidence = combined_consistency * 0.65 + 0.35
            elif direction_match:
                # 方向一致但信号强度不同
                confidence = combined_consistency * 0.6 + 0.3
            else:
                # 方向不一致（边界情况，信号可能在切换）
                confidence = combined_consistency * 0.5 + 0.2
        else:
            confidence = 0.5

        return signal, min(confidence, 0.95)

    def _analyze_timeframes(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        volumes: List[float]
    ) -> List[TimeframeAnalysis]:
        """
        分析多个时间框架的技术信号
        - 短期 (1-2周): 基于5日、10日、20日数据
        - 中期 (1-3月): 基于20日、50日数据
        - 长期 (3-12月): 基于50日、200日数据
        """
        timeframe_results = []
        calc = self.calculator
        current = closes[-1] if closes else 0

        # ========== 短期分析 (1-2周) ==========
        short_term = self._analyze_short_term(closes, highs, lows, calc, current)
        timeframe_results.append(short_term)

        # ========== 中期分析 (1-3月) ==========
        medium_term = self._analyze_medium_term(closes, highs, lows, calc, current)
        timeframe_results.append(medium_term)

        # ========== 长期分析 (3-12月) ==========
        long_term = self._analyze_long_term(closes, highs, lows, calc, current)
        timeframe_results.append(long_term)

        return timeframe_results

    def _analyze_short_term(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        calc: 'TechnicalCalculator',
        current: float
    ) -> TimeframeAnalysis:
        """短期分析 (1-2周)"""
        key_indicators = []
        signals = []
        weights = []

        # 1. 5日与10日均线关系
        sma_5 = calc.sma(closes, 5)
        sma_10 = calc.sma(closes, 10)
        sma_20 = calc.sma(closes, 20)

        if sma_5 and sma_10:
            if sma_5 > sma_10:
                signals.append(SignalStrength.BUY)
                key_indicators.append("5日均线在10日均线之上")
            else:
                signals.append(SignalStrength.SELL)
                key_indicators.append("5日均线在10日均线之下")
            weights.append(1.5)

        # 2. 价格与20日均线
        if sma_20:
            if current > sma_20 * 1.02:
                signals.append(SignalStrength.BUY)
                key_indicators.append(f"价格高于20日均线 {((current/sma_20-1)*100):.1f}%")
            elif current < sma_20 * 0.98:
                signals.append(SignalStrength.SELL)
                key_indicators.append(f"价格低于20日均线 {((1-current/sma_20)*100):.1f}%")
            else:
                signals.append(SignalStrength.NEUTRAL)
                key_indicators.append("价格接近20日均线")
            weights.append(1.2)

        # 3. RSI(7) 短期超买超卖
        rsi_7 = calc.rsi(closes, 7)
        if rsi_7 is not None:
            if rsi_7 > 75:
                signals.append(SignalStrength.STRONG_SELL)
                key_indicators.append(f"RSI(7)={rsi_7:.0f} 严重超买")
            elif rsi_7 > 65:
                signals.append(SignalStrength.SELL)
                key_indicators.append(f"RSI(7)={rsi_7:.0f} 超买")
            elif rsi_7 < 25:
                signals.append(SignalStrength.STRONG_BUY)
                key_indicators.append(f"RSI(7)={rsi_7:.0f} 严重超卖")
            elif rsi_7 < 35:
                signals.append(SignalStrength.BUY)
                key_indicators.append(f"RSI(7)={rsi_7:.0f} 超卖")
            else:
                signals.append(SignalStrength.NEUTRAL)
            weights.append(1.3)

        # 4. 最近5天价格变化
        if len(closes) >= 5:
            change_5d = (closes[-1] - closes[-5]) / closes[-5]
            if change_5d > 0.05:
                signals.append(SignalStrength.STRONG_BUY)
                key_indicators.append(f"5日涨幅 +{change_5d*100:.1f}%")
            elif change_5d > 0.02:
                signals.append(SignalStrength.BUY)
                key_indicators.append(f"5日涨幅 +{change_5d*100:.1f}%")
            elif change_5d < -0.05:
                signals.append(SignalStrength.STRONG_SELL)
                key_indicators.append(f"5日跌幅 {change_5d*100:.1f}%")
            elif change_5d < -0.02:
                signals.append(SignalStrength.SELL)
                key_indicators.append(f"5日跌幅 {change_5d*100:.1f}%")
            else:
                signals.append(SignalStrength.NEUTRAL)
            weights.append(1.0)

        # 5. Stochastic 随机指标
        stoch = calc.stochastic(closes, highs, lows)
        if stoch:
            k = stoch.get('k', 50)
            if k > 80:
                signals.append(SignalStrength.SELL)
                key_indicators.append(f"随机指标K={k:.0f} 超买")
            elif k < 20:
                signals.append(SignalStrength.BUY)
                key_indicators.append(f"随机指标K={k:.0f} 超卖")
            else:
                signals.append(SignalStrength.NEUTRAL)
            weights.append(1.0)

        # 计算综合信号和置信度
        signal, trend, confidence = self._calculate_timeframe_signal(signals, weights)

        return TimeframeAnalysis(
            timeframe=Timeframe.SHORT_TERM,
            timeframe_label="短期 (1-2周)",
            signal=signal,
            trend=trend,
            trend_strength=confidence,
            confidence=confidence,
            key_indicators=key_indicators[:4],  # 最多显示4个关键指标
            description=self._generate_timeframe_description(signal, trend, "短期")
        )

    def _analyze_medium_term(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        calc: 'TechnicalCalculator',
        current: float
    ) -> TimeframeAnalysis:
        """中期分析 (1-3月)"""
        key_indicators = []
        signals = []
        weights = []

        sma_20 = calc.sma(closes, 20)
        sma_50 = calc.sma(closes, 50)

        # 1. 20日与50日均线关系
        if sma_20 and sma_50:
            if sma_20 > sma_50:
                signals.append(SignalStrength.BUY)
                key_indicators.append("20日均线在50日均线之上")
            else:
                signals.append(SignalStrength.SELL)
                key_indicators.append("20日均线在50日均线之下")
            weights.append(2.0)

        # 2. 价格与50日均线
        if sma_50:
            diff_pct = (current / sma_50 - 1) * 100
            if current > sma_50 * 1.05:
                signals.append(SignalStrength.STRONG_BUY)
                key_indicators.append(f"价格高于50日均线 +{diff_pct:.1f}%")
            elif current > sma_50:
                signals.append(SignalStrength.BUY)
                key_indicators.append(f"价格高于50日均线 +{diff_pct:.1f}%")
            elif current < sma_50 * 0.95:
                signals.append(SignalStrength.STRONG_SELL)
                key_indicators.append(f"价格低于50日均线 {diff_pct:.1f}%")
            else:
                signals.append(SignalStrength.SELL)
                key_indicators.append(f"价格低于50日均线 {diff_pct:.1f}%")
            weights.append(1.5)

        # 3. RSI(14) 标准周期
        rsi_14 = calc.rsi(closes, 14)
        if rsi_14 is not None:
            if rsi_14 > 70:
                signals.append(SignalStrength.SELL)
                key_indicators.append(f"RSI(14)={rsi_14:.0f} 超买")
            elif rsi_14 < 30:
                signals.append(SignalStrength.BUY)
                key_indicators.append(f"RSI(14)={rsi_14:.0f} 超卖")
            else:
                signals.append(SignalStrength.NEUTRAL)
                key_indicators.append(f"RSI(14)={rsi_14:.0f} 中性")
            weights.append(1.3)

        # 4. MACD
        macd = calc.macd(closes)
        if macd and macd.get("histogram") is not None:
            hist = macd["histogram"]
            macd_line = macd.get("macd", 0)
            signal_line = macd.get("signal", 0)

            if hist > 0 and macd_line > signal_line:
                signals.append(SignalStrength.BUY)
                key_indicators.append("MACD金叉，动能向上")
            elif hist < 0 and macd_line < signal_line:
                signals.append(SignalStrength.SELL)
                key_indicators.append("MACD死叉，动能向下")
            else:
                signals.append(SignalStrength.NEUTRAL)
            weights.append(1.5)

        # 5. 20日价格变化
        if len(closes) >= 20:
            change_20d = (closes[-1] - closes[-20]) / closes[-20]
            if change_20d > 0.1:
                signals.append(SignalStrength.STRONG_BUY)
                key_indicators.append(f"20日涨幅 +{change_20d*100:.1f}%")
            elif change_20d > 0.03:
                signals.append(SignalStrength.BUY)
            elif change_20d < -0.1:
                signals.append(SignalStrength.STRONG_SELL)
                key_indicators.append(f"20日跌幅 {change_20d*100:.1f}%")
            elif change_20d < -0.03:
                signals.append(SignalStrength.SELL)
            else:
                signals.append(SignalStrength.NEUTRAL)
            weights.append(1.0)

        # 6. ADX 趋势强度
        adx = calc.adx(highs, lows, closes)
        if adx:
            adx_value = adx.get('adx', 0)
            plus_di = adx.get('plus_di', 0)
            minus_di = adx.get('minus_di', 0)

            if adx_value > 25:  # 有明显趋势
                if plus_di > minus_di:
                    signals.append(SignalStrength.BUY)
                    key_indicators.append(f"ADX={adx_value:.0f} 上升趋势")
                else:
                    signals.append(SignalStrength.SELL)
                    key_indicators.append(f"ADX={adx_value:.0f} 下降趋势")
            else:
                signals.append(SignalStrength.NEUTRAL)
                key_indicators.append(f"ADX={adx_value:.0f} 趋势不明")
            weights.append(1.2)

        signal, trend, confidence = self._calculate_timeframe_signal(signals, weights)

        return TimeframeAnalysis(
            timeframe=Timeframe.MEDIUM_TERM,
            timeframe_label="中期 (1-3月)",
            signal=signal,
            trend=trend,
            trend_strength=confidence,
            confidence=confidence,
            key_indicators=key_indicators[:4],
            description=self._generate_timeframe_description(signal, trend, "中期")
        )

    def _analyze_long_term(
        self,
        closes: List[float],
        highs: List[float],
        lows: List[float],
        calc: 'TechnicalCalculator',
        current: float
    ) -> TimeframeAnalysis:
        """长期分析 (3-12月)"""
        key_indicators = []
        signals = []
        weights = []

        sma_50 = calc.sma(closes, 50)
        sma_200 = calc.sma(closes, 200)

        # 1. 金叉/死叉 (50日 vs 200日)
        if sma_50 and sma_200:
            if sma_50 > sma_200:
                signals.append(SignalStrength.STRONG_BUY)
                key_indicators.append("金叉：50日均线在200日均线之上")
                weights.append(2.5)
            else:
                signals.append(SignalStrength.STRONG_SELL)
                key_indicators.append("死叉：50日均线在200日均线之下")
                weights.append(2.5)

        # 2. 价格与200日均线
        if sma_200:
            diff_pct = (current / sma_200 - 1) * 100
            if current > sma_200 * 1.1:
                signals.append(SignalStrength.STRONG_BUY)
                key_indicators.append(f"价格远高于200日均线 +{diff_pct:.1f}%")
            elif current > sma_200:
                signals.append(SignalStrength.BUY)
                key_indicators.append(f"价格高于200日均线 +{diff_pct:.1f}%")
            elif current < sma_200 * 0.9:
                signals.append(SignalStrength.STRONG_SELL)
                key_indicators.append(f"价格远低于200日均线 {diff_pct:.1f}%")
            else:
                signals.append(SignalStrength.SELL)
                key_indicators.append(f"价格低于200日均线 {diff_pct:.1f}%")
            weights.append(2.0)

        # 3. 50日价格变化
        if len(closes) >= 50:
            change_50d = (closes[-1] - closes[-50]) / closes[-50]
            if change_50d > 0.15:
                signals.append(SignalStrength.STRONG_BUY)
                key_indicators.append(f"50日涨幅 +{change_50d*100:.1f}%")
            elif change_50d > 0.05:
                signals.append(SignalStrength.BUY)
                key_indicators.append(f"50日涨幅 +{change_50d*100:.1f}%")
            elif change_50d < -0.15:
                signals.append(SignalStrength.STRONG_SELL)
                key_indicators.append(f"50日跌幅 {change_50d*100:.1f}%")
            elif change_50d < -0.05:
                signals.append(SignalStrength.SELL)
                key_indicators.append(f"50日跌幅 {change_50d*100:.1f}%")
            else:
                signals.append(SignalStrength.NEUTRAL)
            weights.append(1.5)

        # 4. 200日趋势（200日均线斜率）
        if len(closes) >= 220 and sma_200:
            sma_200_20_days_ago = calc.sma(closes[:-20], 200)
            if sma_200_20_days_ago:
                slope = (sma_200 - sma_200_20_days_ago) / sma_200_20_days_ago
                if slope > 0.02:
                    signals.append(SignalStrength.BUY)
                    key_indicators.append("200日均线上升趋势")
                elif slope < -0.02:
                    signals.append(SignalStrength.SELL)
                    key_indicators.append("200日均线下降趋势")
                else:
                    signals.append(SignalStrength.NEUTRAL)
                    key_indicators.append("200日均线走平")
                weights.append(1.5)

        # 5. 距离52周高低点
        if len(closes) >= 252:
            high_52w = max(highs[-252:]) if highs else current
            low_52w = min(lows[-252:]) if lows else current
            range_52w = high_52w - low_52w

            if range_52w > 0:
                position = (current - low_52w) / range_52w
                if position > 0.8:
                    signals.append(SignalStrength.BUY)
                    key_indicators.append(f"接近52周高点 (位置: {position*100:.0f}%)")
                elif position < 0.2:
                    signals.append(SignalStrength.SELL)
                    key_indicators.append(f"接近52周低点 (位置: {position*100:.0f}%)")
                else:
                    signals.append(SignalStrength.NEUTRAL)
                weights.append(1.0)

        signal, trend, confidence = self._calculate_timeframe_signal(signals, weights)

        return TimeframeAnalysis(
            timeframe=Timeframe.LONG_TERM,
            timeframe_label="长期 (3-12月)",
            signal=signal,
            trend=trend,
            trend_strength=confidence,
            confidence=confidence,
            key_indicators=key_indicators[:4],
            description=self._generate_timeframe_description(signal, trend, "长期")
        )

    def _calculate_timeframe_signal(
        self,
        signals: List[SignalStrength],
        weights: List[float]
    ) -> Tuple[SignalStrength, TrendDirection, float]:
        """计算单个时间框架的综合信号、趋势和置信度"""
        if not signals:
            return SignalStrength.NEUTRAL, TrendDirection.NEUTRAL, 0.5

        signal_scores = {
            SignalStrength.STRONG_BUY: 2,
            SignalStrength.BUY: 1,
            SignalStrength.NEUTRAL: 0,
            SignalStrength.SELL: -1,
            SignalStrength.STRONG_SELL: -2
        }

        # 计算加权得分
        total_weight = sum(weights)
        weighted_score = sum(signal_scores[s] * w for s, w in zip(signals, weights))
        avg_score = weighted_score / total_weight if total_weight > 0 else 0

        # 确定信号
        if avg_score >= 1.2:
            signal = SignalStrength.STRONG_BUY
        elif avg_score >= 0.4:
            signal = SignalStrength.BUY
        elif avg_score <= -1.2:
            signal = SignalStrength.STRONG_SELL
        elif avg_score <= -0.4:
            signal = SignalStrength.SELL
        else:
            signal = SignalStrength.NEUTRAL

        # 确定趋势
        if avg_score >= 1.0:
            trend = TrendDirection.STRONG_BULLISH
        elif avg_score >= 0.3:
            trend = TrendDirection.BULLISH
        elif avg_score <= -1.0:
            trend = TrendDirection.STRONG_BEARISH
        elif avg_score <= -0.3:
            trend = TrendDirection.BEARISH
        else:
            trend = TrendDirection.NEUTRAL

        # 计算置信度（基于信号一致性）
        bullish_count = sum(1 for s in signals if s in [SignalStrength.STRONG_BUY, SignalStrength.BUY])
        bearish_count = sum(1 for s in signals if s in [SignalStrength.STRONG_SELL, SignalStrength.SELL])
        neutral_count = sum(1 for s in signals if s == SignalStrength.NEUTRAL)

        max_count = max(bullish_count, bearish_count, neutral_count)
        consistency = max_count / len(signals) if signals else 0.5

        # 置信度 = 一致性 * 0.6 + 基础分 0.3 + 信号强度奖励
        strength_bonus = 0.1 if signal in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL] else 0
        confidence = min(0.95, consistency * 0.6 + 0.3 + strength_bonus)

        return signal, trend, confidence

    def _generate_timeframe_description(
        self,
        signal: SignalStrength,
        trend: TrendDirection,
        timeframe_name: str
    ) -> str:
        """生成时间框架描述"""
        signal_desc = {
            SignalStrength.STRONG_BUY: "强烈看涨",
            SignalStrength.BUY: "看涨",
            SignalStrength.NEUTRAL: "中性",
            SignalStrength.SELL: "看跌",
            SignalStrength.STRONG_SELL: "强烈看跌"
        }

        trend_desc = {
            TrendDirection.STRONG_BULLISH: "强势上涨",
            TrendDirection.BULLISH: "温和上涨",
            TrendDirection.NEUTRAL: "横盘震荡",
            TrendDirection.BEARISH: "温和下跌",
            TrendDirection.STRONG_BEARISH: "强势下跌"
        }

        return f"{timeframe_name}技术面{signal_desc[signal]}，趋势呈{trend_desc[trend]}态势"

    def _identify_key_levels(
        self,
        current_price: float,
        supports: List[SupportResistance],
        resistances: List[SupportResistance],
        indicators: List[TechnicalIndicator]
    ) -> Dict[str, float]:
        """识别关键价位"""
        levels = {}
        
        # 最近支撑/阻力
        if supports:
            levels["nearest_support"] = supports[0].level
            levels["strong_support"] = max(supports, key=lambda x: x.strength).level if supports else None
        
        if resistances:
            levels["nearest_resistance"] = resistances[0].level
            levels["strong_resistance"] = max(resistances, key=lambda x: x.strength).level if resistances else None
        
        # 均线关键位
        for ind in indicators:
            if "SMA(50)" in ind.name:
                levels["sma_50"] = ind.value
            elif "SMA(200)" in ind.name:
                levels["sma_200"] = ind.value
        
        # 布林带
        for ind in indicators:
            if "Bollinger" in ind.name:
                # 从description中提取，这里简化处理
                pass
        
        return {k: v for k, v in levels.items() if v is not None}
    
    def _generate_analysis_summary(
        self,
        symbol: str,
        current_price: float,
        trend: TrendDirection,
        indicators: List[TechnicalIndicator],
        patterns: List[PatternMatch],
        supports: List[SupportResistance],
        resistances: List[SupportResistance]
    ) -> Tuple[str, List[str], List[str]]:
        """生成分析摘要"""
        
        observations = []
        warnings = []
        
        # 趋势描述
        trend_desc = {
            TrendDirection.STRONG_BULLISH: "强势上涨",
            TrendDirection.BULLISH: "温和上涨",
            TrendDirection.NEUTRAL: "震荡整理",
            TrendDirection.BEARISH: "温和下跌",
            TrendDirection.STRONG_BEARISH: "强势下跌"
        }
        
        summary = f"{symbol} 当前价格 {current_price:.2f}，整体呈{trend_desc[trend]}趋势。"
        
        # 关键指标观察
        for ind in indicators:
            if ind.signal in [SignalStrength.STRONG_BUY, SignalStrength.STRONG_SELL]:
                observations.append(ind.description)
        
        # 形态观察
        for pattern in patterns:
            observations.append(f"识别到{pattern.pattern_name}形态，置信度{pattern.confidence*100:.0f}%")
            if pattern.target_price:
                observations.append(f"形态目标价: {pattern.target_price:.2f}")
        
        # 支撑阻力观察
        if supports:
            observations.append(f"最近支撑位: {supports[0].level:.2f}")
        if resistances:
            observations.append(f"最近阻力位: {resistances[0].level:.2f}")
        
        # 警告
        rsi_ind = next((i for i in indicators if "RSI" in i.name), None)
        if rsi_ind:
            if rsi_ind.value > 80:
                warnings.append("RSI严重超买，注意回调风险")
            elif rsi_ind.value < 20:
                warnings.append("RSI严重超卖，可能存在反弹机会")
        
        # 指标分歧
        buy_count = sum(1 for i in indicators if i.signal in [SignalStrength.BUY, SignalStrength.STRONG_BUY])
        sell_count = sum(1 for i in indicators if i.signal in [SignalStrength.SELL, SignalStrength.STRONG_SELL])
        
        if buy_count > 0 and sell_count > 0:
            if abs(buy_count - sell_count) < 2:
                warnings.append("技术指标存在分歧，建议谨慎操作")
        
        return summary, observations, warnings
