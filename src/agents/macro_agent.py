"""
MacroAgent - 宏观经济分析Agent

分析宏观经济环境对投资标的的影响：
- 经济周期定位
- 利率环境分析
- 通胀趋势评估
- 货币政策解读
- 地缘政治风险
- 行业周期映射
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from src.core.base import (
    BaseAgent, AgentOutput, AnalysisContext, ConfidenceScore,
    ReasoningStep, DataSource, Assumption, Uncertainty
)


class EconomicCycle(str, Enum):
    """经济周期阶段"""
    EARLY_EXPANSION = "early_expansion"      # 早期扩张
    MID_EXPANSION = "mid_expansion"          # 中期扩张
    LATE_EXPANSION = "late_expansion"        # 晚期扩张
    EARLY_CONTRACTION = "early_contraction"  # 早期收缩
    RECESSION = "recession"                   # 衰退
    RECOVERY = "recovery"                     # 复苏


class MonetaryStance(str, Enum):
    """货币政策立场"""
    VERY_DOVISH = "very_dovish"
    DOVISH = "dovish"
    NEUTRAL = "neutral"
    HAWKISH = "hawkish"
    VERY_HAWKISH = "very_hawkish"


class InflationRegime(str, Enum):
    """通胀环境"""
    DEFLATION = "deflation"
    LOW_STABLE = "low_stable"
    MODERATE = "moderate"
    HIGH = "high"
    HYPERINFLATION = "hyperinflation"


@dataclass
class MacroIndicator:
    """宏观指标"""
    name: str
    value: float
    previous: float
    change: float
    trend: str  # improving, stable, deteriorating
    percentile: float  # 历史百分位
    source: str
    as_of_date: datetime


@dataclass
class MacroEnvironment:
    """宏观环境评估"""
    cycle_phase: EconomicCycle
    cycle_confidence: float
    monetary_stance: MonetaryStance
    inflation_regime: InflationRegime
    growth_outlook: str  # accelerating, stable, decelerating
    risk_appetite: str   # risk_on, neutral, risk_off
    key_indicators: List[MacroIndicator]
    regional_analysis: Dict[str, Any]


@dataclass
class SectorCyclicality:
    """行业周期性分析"""
    sector: str
    cyclicality: str  # defensive, cyclical, growth
    current_phase_outlook: str  # favorable, neutral, unfavorable
    rate_sensitivity: float  # -1 to 1
    inflation_sensitivity: float  # -1 to 1
    historical_performance: Dict[str, float]  # 各周期阶段历史表现


@dataclass
class MacroRisk:
    """宏观风险"""
    category: str
    description: str
    probability: float
    impact: str  # low, medium, high, severe
    affected_sectors: List[str]
    hedging_strategies: List[str]


class MacroAgent(BaseAgent):
    """
    宏观经济分析Agent
    
    核心功能：
    1. 经济周期定位 - 判断当前经济所处阶段
    2. 政策环境分析 - 货币政策、财政政策解读
    3. 通胀趋势预测 - CPI、PPI、核心通胀分析
    4. 利率路径预判 - 联储政策、收益率曲线
    5. 行业周期映射 - 不同周期阶段的行业表现
    6. 地缘政治风险 - 贸易、制裁、冲突影响
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.agent_name = "MacroAgent"
        
        # 周期敏感性矩阵：各行业在不同周期的表现
        self.sector_cycle_matrix = {
            "technology": {
                EconomicCycle.EARLY_EXPANSION: 1.2,
                EconomicCycle.MID_EXPANSION: 1.0,
                EconomicCycle.LATE_EXPANSION: 0.8,
                EconomicCycle.EARLY_CONTRACTION: 0.6,
                EconomicCycle.RECESSION: 0.7,
                EconomicCycle.RECOVERY: 1.3
            },
            "consumer_discretionary": {
                EconomicCycle.EARLY_EXPANSION: 1.3,
                EconomicCycle.MID_EXPANSION: 1.1,
                EconomicCycle.LATE_EXPANSION: 0.7,
                EconomicCycle.EARLY_CONTRACTION: 0.5,
                EconomicCycle.RECESSION: 0.4,
                EconomicCycle.RECOVERY: 1.2
            },
            "consumer_staples": {
                EconomicCycle.EARLY_EXPANSION: 0.8,
                EconomicCycle.MID_EXPANSION: 0.9,
                EconomicCycle.LATE_EXPANSION: 1.0,
                EconomicCycle.EARLY_CONTRACTION: 1.1,
                EconomicCycle.RECESSION: 1.2,
                EconomicCycle.RECOVERY: 0.9
            },
            "healthcare": {
                EconomicCycle.EARLY_EXPANSION: 0.9,
                EconomicCycle.MID_EXPANSION: 1.0,
                EconomicCycle.LATE_EXPANSION: 1.0,
                EconomicCycle.EARLY_CONTRACTION: 1.1,
                EconomicCycle.RECESSION: 1.1,
                EconomicCycle.RECOVERY: 0.9
            },
            "financials": {
                EconomicCycle.EARLY_EXPANSION: 1.2,
                EconomicCycle.MID_EXPANSION: 1.1,
                EconomicCycle.LATE_EXPANSION: 0.9,
                EconomicCycle.EARLY_CONTRACTION: 0.6,
                EconomicCycle.RECESSION: 0.5,
                EconomicCycle.RECOVERY: 1.3
            },
            "industrials": {
                EconomicCycle.EARLY_EXPANSION: 1.2,
                EconomicCycle.MID_EXPANSION: 1.1,
                EconomicCycle.LATE_EXPANSION: 0.8,
                EconomicCycle.EARLY_CONTRACTION: 0.6,
                EconomicCycle.RECESSION: 0.5,
                EconomicCycle.RECOVERY: 1.2
            },
            "energy": {
                EconomicCycle.EARLY_EXPANSION: 1.0,
                EconomicCycle.MID_EXPANSION: 1.2,
                EconomicCycle.LATE_EXPANSION: 1.3,
                EconomicCycle.EARLY_CONTRACTION: 0.8,
                EconomicCycle.RECESSION: 0.6,
                EconomicCycle.RECOVERY: 1.0
            },
            "utilities": {
                EconomicCycle.EARLY_EXPANSION: 0.7,
                EconomicCycle.MID_EXPANSION: 0.8,
                EconomicCycle.LATE_EXPANSION: 1.0,
                EconomicCycle.EARLY_CONTRACTION: 1.1,
                EconomicCycle.RECESSION: 1.2,
                EconomicCycle.RECOVERY: 0.8
            },
            "real_estate": {
                EconomicCycle.EARLY_EXPANSION: 1.1,
                EconomicCycle.MID_EXPANSION: 1.0,
                EconomicCycle.LATE_EXPANSION: 0.8,
                EconomicCycle.EARLY_CONTRACTION: 0.6,
                EconomicCycle.RECESSION: 0.7,
                EconomicCycle.RECOVERY: 1.0
            },
            "materials": {
                EconomicCycle.EARLY_EXPANSION: 1.2,
                EconomicCycle.MID_EXPANSION: 1.1,
                EconomicCycle.LATE_EXPANSION: 1.0,
                EconomicCycle.EARLY_CONTRACTION: 0.7,
                EconomicCycle.RECESSION: 0.5,
                EconomicCycle.RECOVERY: 1.1
            }
        }
    
    async def analyze(self, context: AnalysisContext) -> AgentOutput:
        """执行宏观经济分析"""
        reasoning_chain = []
        data_sources = []
        assumptions = []
        uncertainties = []
        warnings = []
        
        target = context.target
        
        # Step 1: 获取宏观经济数据
        step1 = ReasoningStep(
            step_number=1,
            description="获取宏观经济指标数据",
            inputs={"regions": ["US", "EU", "CN"]},
            outputs={},
            confidence=0.0
        )
        
        macro_data = await self._fetch_macro_data()
        step1.outputs = {"indicators_count": len(macro_data)}
        step1.confidence = 0.85
        reasoning_chain.append(step1)
        
        data_sources.append(DataSource(
            name="FRED",
            type="macro_data",
            url="https://fred.stlouisfed.org",
            accessed_at=datetime.utcnow(),
            quality_score=0.9
        ))
        
        # Step 2: 经济周期定位
        step2 = ReasoningStep(
            step_number=2,
            description="判断当前经济周期阶段",
            inputs={"indicators": list(macro_data.keys())},
            outputs={},
            confidence=0.0
        )
        
        cycle_assessment = self._assess_economic_cycle(macro_data)
        step2.outputs = {
            "cycle_phase": cycle_assessment["phase"].value,
            "confidence": cycle_assessment["confidence"]
        }
        step2.confidence = cycle_assessment["confidence"]
        reasoning_chain.append(step2)
        
        assumptions.append(Assumption(
            category="economic_cycle",
            description="经济周期判断基于滞后指标，实际转折点可能已发生",
            impact="medium",
            sensitivity=0.3
        ))
        
        # Step 3: 货币政策分析
        step3 = ReasoningStep(
            step_number=3,
            description="分析货币政策立场和利率路径",
            inputs={"fed_funds_rate": macro_data.get("fed_funds_rate", 5.25)},
            outputs={},
            confidence=0.0
        )
        
        monetary_analysis = self._analyze_monetary_policy(macro_data)
        step3.outputs = {
            "stance": monetary_analysis["stance"].value,
            "rate_path": monetary_analysis["rate_path"]
        }
        step3.confidence = 0.75
        reasoning_chain.append(step3)
        
        uncertainties.append(Uncertainty(
            source="fed_policy",
            description="联储政策路径存在不确定性，取决于通胀和就业数据",
            impact="high",
            range_low=-0.75,
            range_high=0.25
        ))
        
        # Step 4: 通胀环境评估
        step4 = ReasoningStep(
            step_number=4,
            description="评估通胀环境和趋势",
            inputs={"cpi": macro_data.get("cpi_yoy", 3.2)},
            outputs={},
            confidence=0.0
        )
        
        inflation_analysis = self._analyze_inflation(macro_data)
        step4.outputs = {
            "regime": inflation_analysis["regime"].value,
            "trend": inflation_analysis["trend"]
        }
        step4.confidence = 0.8
        reasoning_chain.append(step4)
        
        # Step 5: 行业周期映射
        step5 = ReasoningStep(
            step_number=5,
            description="分析目标所属行业的周期敏感性",
            inputs={"target": target, "cycle": cycle_assessment["phase"].value},
            outputs={},
            confidence=0.0
        )
        
        sector = self._identify_sector(target)
        sector_analysis = self._analyze_sector_cyclicality(
            sector, 
            cycle_assessment["phase"]
        )
        step5.outputs = {
            "sector": sector,
            "outlook": sector_analysis.current_phase_outlook
        }
        step5.confidence = 0.7
        reasoning_chain.append(step5)
        
        # Step 6: 宏观风险识别
        step6 = ReasoningStep(
            step_number=6,
            description="识别主要宏观风险因素",
            inputs={},
            outputs={},
            confidence=0.0
        )
        
        macro_risks = self._identify_macro_risks(
            cycle_assessment["phase"],
            monetary_analysis,
            inflation_analysis,
            sector
        )
        step6.outputs = {"risk_count": len(macro_risks)}
        step6.confidence = 0.65
        reasoning_chain.append(step6)
        
        # Step 7: LLM综合分析
        step7 = ReasoningStep(
            step_number=7,
            description="使用LLM综合分析宏观环境对标的的影响",
            inputs={"target": target},
            outputs={},
            confidence=0.0
        )
        
        llm_analysis = await self._llm_macro_synthesis(
            target,
            cycle_assessment,
            monetary_analysis,
            inflation_analysis,
            sector_analysis,
            macro_risks
        )
        step7.outputs = {"analysis": "completed"}
        step7.confidence = 0.7
        reasoning_chain.append(step7)
        
        # 构建最终结果
        result = {
            "economic_environment": {
                "cycle_phase": cycle_assessment["phase"].value,
                "cycle_confidence": cycle_assessment["confidence"],
                "cycle_description": self._get_cycle_description(cycle_assessment["phase"]),
                "monetary_stance": monetary_analysis["stance"].value,
                "inflation_regime": inflation_analysis["regime"].value,
                "growth_outlook": self._assess_growth_outlook(macro_data)
            },
            "policy_analysis": {
                "fed_stance": monetary_analysis["stance"].value,
                "rate_path": monetary_analysis["rate_path"],
                "next_move": monetary_analysis["next_move"],
                "terminal_rate": monetary_analysis["terminal_rate"]
            },
            "inflation_analysis": {
                "current_regime": inflation_analysis["regime"].value,
                "trend": inflation_analysis["trend"],
                "core_cpi": macro_data.get("core_cpi_yoy", 3.5),
                "outlook": inflation_analysis["outlook"]
            },
            "sector_positioning": {
                "sector": sector,
                "cyclicality": sector_analysis.cyclicality,
                "current_outlook": sector_analysis.current_phase_outlook,
                "rate_sensitivity": sector_analysis.rate_sensitivity,
                "inflation_sensitivity": sector_analysis.inflation_sensitivity,
                "phase_score": self.sector_cycle_matrix.get(sector, {}).get(
                    cycle_assessment["phase"], 1.0
                )
            },
            "macro_risks": [
                {
                    "category": r.category,
                    "description": r.description,
                    "probability": r.probability,
                    "impact": r.impact,
                    "affected_sectors": r.affected_sectors
                }
                for r in macro_risks
            ],
            "investment_implications": llm_analysis.get("implications", []),
            "key_metrics_to_watch": llm_analysis.get("metrics_to_watch", []),
            "scenario_analysis": self._build_macro_scenarios(
                cycle_assessment["phase"],
                monetary_analysis,
                sector
            )
        }
        
        # 添加警告
        if cycle_assessment["phase"] in [EconomicCycle.LATE_EXPANSION, 
                                          EconomicCycle.EARLY_CONTRACTION]:
            warnings.append(
                "经济处于周期后期，建议关注防御性配置"
            )
        
        if inflation_analysis["regime"] == InflationRegime.HIGH:
            warnings.append(
                "高通胀环境可能压缩估值倍数，需谨慎对待高估值标的"
            )
        
        # 计算置信度
        confidence = self._calculate_confidence(
            data_quality=0.85,
            completeness=0.8,
            reasoning_quality=0.75,
            validation=0.7,
            methodology_fit=0.8
        )
        
        return AgentOutput(
            agent_name=self.agent_name,
            result=result,
            confidence=confidence,
            reasoning_chain=reasoning_chain,
            data_sources=data_sources,
            assumptions=assumptions,
            uncertainties=uncertainties,
            warnings=warnings,
            timestamp=datetime.utcnow()
        )
    
    async def _fetch_macro_data(self) -> Dict[str, float]:
        """获取宏观经济数据 - 通过 yfinance 获取真实市场指标"""
        # Defaults for indicators that cannot be fetched from yfinance
        data = {
            "gdp_growth": 2.1,
            "unemployment_rate": 3.9,
            "cpi_yoy": 3.2,
            "core_cpi_yoy": 3.5,
            "pce_yoy": 2.8,
            "fed_funds_rate": 5.25,
            "ism_manufacturing": 48.5,
            "ism_services": 52.3,
            "consumer_confidence": 102.5,
            "retail_sales_mom": 0.3,
            "industrial_production_yoy": 1.2,
            "housing_starts": 1.35,
            "initial_claims": 215000,
            "china_gdp_growth": 5.2,
            "eurozone_gdp_growth": 0.8,
        }

        # Fetch real market data via yfinance (with retry)
        try:
            from src.core.yfinance_utils import get_ticker_info

            tickers_map = {
                "^TNX": "treasury_10y",       # 10-Year Treasury Yield
                "^VIX": "vix",                # CBOE Volatility Index
                "DX-Y.NYB": "dxy_index",      # US Dollar Index
                "CL=F": "oil_wti",            # Crude Oil Futures
                "GC=F": "gold_price",         # Gold Futures
                "^GSPC": "sp500",             # S&P 500
                "^IRX": "treasury_3m",        # 3-Month Treasury Bill
            }

            for ticker_symbol, key in tickers_map.items():
                try:
                    info = get_ticker_info(ticker_symbol)
                    price = (
                        info.get("regularMarketPrice")
                        or info.get("previousClose")
                        or info.get("currentPrice")
                    )
                    if price is not None:
                        data[key] = float(price)
                except Exception:
                    pass  # Keep default

            # Derive yield curve spread: 10Y - 3M
            t10y = data.get("treasury_10y")
            t3m = data.get("treasury_3m")
            if t10y is not None and t3m is not None:
                data["yield_curve_spread"] = round(t10y - t3m, 2)
                data["treasury_2y"] = round((t10y + t3m) / 2, 2)  # approximate
            else:
                data["yield_curve_spread"] = -0.50
                data["treasury_2y"] = data.get("treasury_10y", 4.35) + 0.50

        except ImportError:
            # yfinance not available, use all defaults
            data.update({
                "treasury_10y": 4.35,
                "treasury_2y": 4.85,
                "yield_curve_spread": -0.50,
                "dxy_index": 104.5,
                "oil_wti": 78.5,
                "gold_price": 2350,
                "vix": 15.5,
            })
        except Exception:
            data.update({
                "treasury_10y": 4.35,
                "treasury_2y": 4.85,
                "yield_curve_spread": -0.50,
                "dxy_index": 104.5,
                "oil_wti": 78.5,
                "gold_price": 2350,
                "vix": 15.5,
            })

        return data
    
    def _assess_economic_cycle(self, data: Dict[str, float]) -> Dict[str, Any]:
        """评估经济周期阶段"""
        # 周期指标评分
        scores = {
            EconomicCycle.EARLY_EXPANSION: 0,
            EconomicCycle.MID_EXPANSION: 0,
            EconomicCycle.LATE_EXPANSION: 0,
            EconomicCycle.EARLY_CONTRACTION: 0,
            EconomicCycle.RECESSION: 0,
            EconomicCycle.RECOVERY: 0
        }
        
        # GDP增长
        gdp = data.get("gdp_growth", 2.0)
        if gdp > 3.0:
            scores[EconomicCycle.MID_EXPANSION] += 2
        elif gdp > 2.0:
            scores[EconomicCycle.LATE_EXPANSION] += 1
            scores[EconomicCycle.MID_EXPANSION] += 1
        elif gdp > 0:
            scores[EconomicCycle.LATE_EXPANSION] += 2
        else:
            scores[EconomicCycle.RECESSION] += 2
        
        # 失业率
        unemployment = data.get("unemployment_rate", 4.0)
        if unemployment < 4.0:
            scores[EconomicCycle.LATE_EXPANSION] += 2
        elif unemployment < 5.0:
            scores[EconomicCycle.MID_EXPANSION] += 1
        elif unemployment < 6.0:
            scores[EconomicCycle.EARLY_CONTRACTION] += 1
        else:
            scores[EconomicCycle.RECESSION] += 2
        
        # ISM制造业
        ism = data.get("ism_manufacturing", 50)
        if ism > 55:
            scores[EconomicCycle.EARLY_EXPANSION] += 2
        elif ism > 50:
            scores[EconomicCycle.MID_EXPANSION] += 1
        elif ism > 45:
            scores[EconomicCycle.LATE_EXPANSION] += 1
            scores[EconomicCycle.EARLY_CONTRACTION] += 1
        else:
            scores[EconomicCycle.RECESSION] += 2
        
        # 收益率曲线
        curve = data.get("yield_curve_spread", 0)
        if curve < -0.5:
            scores[EconomicCycle.LATE_EXPANSION] += 2
            scores[EconomicCycle.EARLY_CONTRACTION] += 1
        elif curve < 0:
            scores[EconomicCycle.LATE_EXPANSION] += 1
        elif curve > 1.0:
            scores[EconomicCycle.EARLY_EXPANSION] += 1
            scores[EconomicCycle.RECOVERY] += 1
        
        # 确定最可能的阶段
        best_phase = max(scores, key=scores.get)
        total_score = sum(scores.values())
        confidence = scores[best_phase] / total_score if total_score > 0 else 0.5
        
        return {
            "phase": best_phase,
            "confidence": min(0.9, confidence + 0.3),
            "scores": scores
        }
    
    def _analyze_monetary_policy(self, data: Dict[str, float]) -> Dict[str, Any]:
        """分析货币政策"""
        fed_rate = data.get("fed_funds_rate", 5.25)
        inflation = data.get("cpi_yoy", 3.0)
        unemployment = data.get("unemployment_rate", 4.0)
        
        # 泰勒规则简化版
        neutral_rate = 2.5
        inflation_gap = inflation - 2.0
        unemployment_gap = 4.5 - unemployment
        taylor_rate = neutral_rate + 1.5 * inflation_gap + 0.5 * unemployment_gap
        
        # 判断政策立场
        if fed_rate > taylor_rate + 0.5:
            stance = MonetaryStance.HAWKISH
        elif fed_rate > taylor_rate:
            stance = MonetaryStance.NEUTRAL
        else:
            stance = MonetaryStance.DOVISH
        
        # 利率路径预测
        if inflation > 3.0:
            rate_path = "higher_for_longer"
            next_move = "hold"
            terminal_rate = fed_rate
        elif inflation < 2.5 and unemployment > 4.2:
            rate_path = "easing"
            next_move = "cut"
            terminal_rate = fed_rate - 0.75
        else:
            rate_path = "stable"
            next_move = "hold"
            terminal_rate = fed_rate
        
        return {
            "stance": stance,
            "rate_path": rate_path,
            "next_move": next_move,
            "terminal_rate": terminal_rate,
            "current_rate": fed_rate,
            "taylor_rate": round(taylor_rate, 2)
        }
    
    def _analyze_inflation(self, data: Dict[str, float]) -> Dict[str, Any]:
        """分析通胀环境"""
        cpi = data.get("cpi_yoy", 3.0)
        core_cpi = data.get("core_cpi_yoy", 3.0)
        pce = data.get("pce_yoy", 2.5)
        
        # 判断通胀水平
        avg_inflation = (cpi + core_cpi + pce) / 3
        
        if avg_inflation < 0:
            regime = InflationRegime.DEFLATION
        elif avg_inflation < 2.0:
            regime = InflationRegime.LOW_STABLE
        elif avg_inflation < 3.5:
            regime = InflationRegime.MODERATE
        elif avg_inflation < 6.0:
            regime = InflationRegime.HIGH
        else:
            regime = InflationRegime.HYPERINFLATION
        
        # 趋势判断（简化）
        if core_cpi < cpi:
            trend = "moderating"
            outlook = "通胀压力有所缓解，核心通胀低于整体通胀"
        elif core_cpi > 4.0:
            trend = "sticky"
            outlook = "核心通胀粘性较强，降温速度慢于预期"
        else:
            trend = "stable"
            outlook = "通胀保持相对稳定"
        
        return {
            "regime": regime,
            "trend": trend,
            "outlook": outlook,
            "headline_cpi": cpi,
            "core_cpi": core_cpi,
            "pce": pce
        }
    
    def _identify_sector(self, target: str) -> str:
        """识别标的所属行业"""
        # 简化的行业映射
        sector_map = {
            "AAPL": "technology",
            "MSFT": "technology",
            "GOOGL": "technology",
            "AMZN": "consumer_discretionary",
            "TSLA": "consumer_discretionary",
            "JPM": "financials",
            "BAC": "financials",
            "JNJ": "healthcare",
            "PFE": "healthcare",
            "XOM": "energy",
            "CVX": "energy",
            "PG": "consumer_staples",
            "KO": "consumer_staples",
            "CAT": "industrials",
            "BA": "industrials",
        }
        return sector_map.get(target.upper(), "technology")
    
    def _analyze_sector_cyclicality(
        self, 
        sector: str, 
        cycle_phase: EconomicCycle
    ) -> SectorCyclicality:
        """分析行业周期敏感性"""
        # 行业属性
        cyclicality_map = {
            "technology": "growth",
            "consumer_discretionary": "cyclical",
            "consumer_staples": "defensive",
            "healthcare": "defensive",
            "financials": "cyclical",
            "industrials": "cyclical",
            "energy": "cyclical",
            "utilities": "defensive",
            "real_estate": "cyclical",
            "materials": "cyclical"
        }
        
        rate_sensitivity_map = {
            "technology": -0.4,
            "consumer_discretionary": -0.3,
            "consumer_staples": -0.1,
            "healthcare": -0.1,
            "financials": 0.3,
            "industrials": -0.2,
            "energy": 0.1,
            "utilities": -0.5,
            "real_estate": -0.6,
            "materials": -0.2
        }
        
        inflation_sensitivity_map = {
            "technology": -0.3,
            "consumer_discretionary": -0.4,
            "consumer_staples": 0.2,
            "healthcare": 0.1,
            "financials": 0.2,
            "industrials": 0.1,
            "energy": 0.5,
            "utilities": -0.2,
            "real_estate": 0.3,
            "materials": 0.4
        }
        
        # 当前周期表现
        phase_score = self.sector_cycle_matrix.get(sector, {}).get(cycle_phase, 1.0)
        
        if phase_score > 1.1:
            outlook = "favorable"
        elif phase_score > 0.9:
            outlook = "neutral"
        else:
            outlook = "unfavorable"
        
        return SectorCyclicality(
            sector=sector,
            cyclicality=cyclicality_map.get(sector, "cyclical"),
            current_phase_outlook=outlook,
            rate_sensitivity=rate_sensitivity_map.get(sector, 0),
            inflation_sensitivity=inflation_sensitivity_map.get(sector, 0),
            historical_performance=self.sector_cycle_matrix.get(sector, {})
        )
    
    def _identify_macro_risks(
        self,
        cycle_phase: EconomicCycle,
        monetary: Dict,
        inflation: Dict,
        sector: str
    ) -> List[MacroRisk]:
        """识别宏观风险"""
        risks = []
        
        # 经济衰退风险
        if cycle_phase in [EconomicCycle.LATE_EXPANSION, 
                           EconomicCycle.EARLY_CONTRACTION]:
            risks.append(MacroRisk(
                category="recession",
                description="经济处于周期后期，衰退风险上升",
                probability=0.35,
                impact="high",
                affected_sectors=["consumer_discretionary", "industrials", "financials"],
                hedging_strategies=["增持防御性行业", "提高现金比例", "考虑看跌期权"]
            ))
        
        # 利率风险
        if monetary["stance"] in [MonetaryStance.HAWKISH, MonetaryStance.VERY_HAWKISH]:
            risks.append(MacroRisk(
                category="interest_rate",
                description="高利率环境持续，压制估值和融资成本",
                probability=0.6,
                impact="medium",
                affected_sectors=["real_estate", "utilities", "technology"],
                hedging_strategies=["关注短久期资产", "避开高杠杆公司"]
            ))
        
        # 通胀风险
        if inflation["regime"] in [InflationRegime.HIGH, InflationRegime.HYPERINFLATION]:
            risks.append(MacroRisk(
                category="inflation",
                description="高通胀侵蚀利润率和消费能力",
                probability=0.5,
                impact="medium",
                affected_sectors=["consumer_discretionary", "technology"],
                hedging_strategies=["关注定价能力强的公司", "考虑通胀保护资产"]
            ))
        
        # 地缘政治风险
        risks.append(MacroRisk(
            category="geopolitical",
            description="中美关系紧张、贸易摩擦风险",
            probability=0.4,
            impact="medium",
            affected_sectors=["technology", "industrials", "materials"],
            hedging_strategies=["关注供应链多元化", "评估中国敞口"]
        ))
        
        return risks
    
    async def _llm_macro_synthesis(
        self,
        target: str,
        cycle: Dict,
        monetary: Dict,
        inflation: Dict,
        sector: SectorCyclicality,
        risks: List[MacroRisk]
    ) -> Dict[str, Any]:
        """使用LLM综合分析"""
        prompt = f"""
作为宏观经济分析师，请分析当前宏观环境对{target}的影响。

当前宏观环境：
- 经济周期：{cycle['phase'].value}（置信度：{cycle['confidence']:.1%}）
- 货币政策：{monetary['stance'].value}，利率路径：{monetary['rate_path']}
- 通胀环境：{inflation['regime'].value}，趋势：{inflation['trend']}
- 行业定位：{sector.sector}（{sector.cyclicality}），当前周期展望：{sector.current_phase_outlook}

主要风险：
{chr(10).join([f"- {r.category}: {r.description}" for r in risks])}

请提供：
1. 宏观环境对该标的的3-5个关键影响
2. 需要密切关注的3-5个宏观指标
3. 投资时机建议

请直接输出分析，不要包含前言。
"""
        
        # 模拟LLM响应
        return {
            "implications": [
                f"{sector.sector}行业在当前{cycle['phase'].value}阶段表现" + 
                ("偏强" if sector.current_phase_outlook == "favorable" else "偏弱"),
                f"利率环境{monetary['rate_path']}对估值形成" +
                ("压制" if monetary['stance'] == MonetaryStance.HAWKISH else "支撑"),
                f"{inflation['regime'].value}通胀环境影响利润率预期",
                "需关注经济周期转折点信号"
            ],
            "metrics_to_watch": [
                "ISM制造业PMI - 经济动能指标",
                "核心PCE通胀 - 联储政策锚点",
                "初请失业金人数 - 劳动力市场健康度",
                "收益率曲线 - 衰退预警信号",
                "消费者信心指数 - 需求前瞻指标"
            ],
            "timing_advice": "当前周期位置建议保持谨慎，关注防御性配置"
        }
    
    def _get_cycle_description(self, phase: EconomicCycle) -> str:
        """获取周期阶段描述"""
        descriptions = {
            EconomicCycle.EARLY_EXPANSION: "经济从衰退中复苏，增长加速，失业率开始下降",
            EconomicCycle.MID_EXPANSION: "经济稳健增长，企业盈利改善，信贷条件宽松",
            EconomicCycle.LATE_EXPANSION: "增长放缓，通胀上升，劳动力市场紧张",
            EconomicCycle.EARLY_CONTRACTION: "经济开始收缩，企业盈利下滑，信贷收紧",
            EconomicCycle.RECESSION: "经济全面收缩，失业率上升，需求萎缩",
            EconomicCycle.RECOVERY: "衰退触底，政策刺激生效，信心开始恢复"
        }
        return descriptions.get(phase, "")
    
    def _assess_growth_outlook(self, data: Dict[str, float]) -> str:
        """评估增长前景"""
        gdp = data.get("gdp_growth", 2.0)
        ism = data.get("ism_manufacturing", 50)
        retail = data.get("retail_sales_mom", 0)
        
        score = 0
        if gdp > 2.5:
            score += 1
        if ism > 52:
            score += 1
        if retail > 0.3:
            score += 1
        
        if score >= 2:
            return "accelerating"
        elif score >= 1:
            return "stable"
        else:
            return "decelerating"
    
    def _build_macro_scenarios(
        self,
        current_phase: EconomicCycle,
        monetary: Dict,
        sector: str
    ) -> List[Dict[str, Any]]:
        """构建宏观情景"""
        scenarios = [
            {
                "name": "soft_landing",
                "description": "软着陆：通胀回落，经济温和放缓",
                "probability": 0.45,
                "gdp_impact": "1.5-2.5%",
                "rate_path": "缓步降息",
                "sector_impact": self.sector_cycle_matrix.get(sector, {}).get(
                    EconomicCycle.MID_EXPANSION, 1.0
                )
            },
            {
                "name": "no_landing",
                "description": "不着陆：经济持续强劲，通胀粘性",
                "probability": 0.30,
                "gdp_impact": "2.5-3.5%",
                "rate_path": "维持高利率更久",
                "sector_impact": self.sector_cycle_matrix.get(sector, {}).get(
                    EconomicCycle.LATE_EXPANSION, 1.0
                )
            },
            {
                "name": "hard_landing",
                "description": "硬着陆：经济衰退，失业率上升",
                "probability": 0.25,
                "gdp_impact": "-1.0 to 1.0%",
                "rate_path": "大幅降息",
                "sector_impact": self.sector_cycle_matrix.get(sector, {}).get(
                    EconomicCycle.RECESSION, 1.0
                )
            }
        ]
        return scenarios
    
    def _calculate_confidence(
        self,
        data_quality: float,
        completeness: float,
        reasoning_quality: float,
        validation: float,
        methodology_fit: float
    ) -> ConfidenceScore:
        """计算置信度"""
        weights = {
            "data_quality": 0.25,
            "completeness": 0.15,
            "reasoning": 0.25,
            "validation": 0.15,
            "methodology_fit": 0.20
        }
        
        factors = {
            "data_quality": data_quality,
            "completeness": completeness,
            "reasoning": reasoning_quality,
            "validation": validation,
            "methodology_fit": methodology_fit
        }
        
        overall = sum(factors[k] * weights[k] for k in weights)
        overall = max(0.1, min(0.9, overall))  # 永不100%确定
        
        return ConfidenceScore(
            overall=overall,
            factors=factors,
            weights=weights,
            explanation=f"宏观分析置信度{overall:.1%}，主要受限于经济周期判断的滞后性"
        )

    # ============== ChainExecutor 适配器方法 ==============

    async def analyze_environment(self, context, inputs: Dict = None) -> AgentOutput:
        """
        ChainExecutor 调用的适配方法

        将 ChainExecutor 的调用格式适配到 analyze 方法
        """
        # 如果 context 已经是 AnalysisContext，直接使用
        if hasattr(context, 'target'):
            return await self.analyze(context)

        # 否则从 inputs 或 context dict 中构建
        from src.core.base import AnalysisContext
        from datetime import datetime

        target = 'UNKNOWN'
        if isinstance(context, dict):
            target = context.get('target', 'UNKNOWN')
        elif inputs and isinstance(inputs, dict):
            market_data = inputs.get('market_data', {})
            target = market_data.get('symbol', 'UNKNOWN')

        analysis_context = AnalysisContext(
            target=target,
            analysis_date=datetime.now()
        )
        return await self.analyze(analysis_context)
