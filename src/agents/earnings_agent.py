"""
Earnings Analysis Agent

Comprehensive earnings quality and financial statement analysis:
- Revenue quality and sustainability
- Margin analysis and trends
- Cash flow quality
- Balance sheet health
- Capital allocation effectiveness
- Earnings manipulation detection
- Forward guidance analysis
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import statistics
import math


class EarningsQuality(Enum):
    """Overall earnings quality rating"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CONCERNING = "concerning"


class TrendDirection(Enum):
    """Trend direction for metrics"""
    STRONG_IMPROVING = "strong_improving"
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    STRONG_DECLINING = "strong_declining"


@dataclass
class RevenueAnalysis:
    """Revenue quality analysis"""
    total_revenue: float
    yoy_growth: float
    qoq_growth: float
    revenue_trend: TrendDirection
    
    # Revenue quality metrics
    recurring_revenue_pct: float  # Subscription, contracts, etc.
    customer_concentration: float  # Top 10 customers %
    geographic_diversification: float  # HHI of geographic revenue
    
    # Revenue drivers
    volume_contribution: float  # % from volume growth
    price_contribution: float  # % from price increases
    mix_contribution: float  # % from product mix
    
    # Sustainability indicators
    backlog_growth: Optional[float]  # Order backlog change
    deferred_revenue_growth: Optional[float]
    customer_retention_rate: Optional[float]
    
    quality_score: float  # 0-100
    key_observations: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)


@dataclass
class MarginAnalysis:
    """Margin analysis and trends"""
    # Current margins
    gross_margin: float
    operating_margin: float
    ebitda_margin: float
    net_margin: float
    fcf_margin: float
    
    # Margin trends (YoY change in bps)
    gross_margin_change: float
    operating_margin_change: float
    net_margin_change: float
    
    # Margin drivers
    cogs_to_revenue: float
    sga_to_revenue: float
    rd_to_revenue: float
    
    # Margin quality
    margin_stability: float  # Std dev of margins over time
    margin_trend: TrendDirection
    operating_leverage: float  # Revenue growth to operating income growth
    
    # Peer comparison
    vs_industry_gross: float  # Difference from industry avg
    vs_industry_operating: float
    
    quality_score: float
    key_observations: List[str] = field(default_factory=list)
    concerns: List[str] = field(default_factory=list)


@dataclass
class CashFlowAnalysis:
    """Cash flow quality analysis"""
    # Cash flow metrics
    operating_cash_flow: float
    free_cash_flow: float
    fcf_per_share: float
    
    # Cash flow quality
    ocf_to_net_income: float  # Should be > 1
    fcf_to_net_income: float
    accrual_ratio: float  # (Net Income - OCF) / Total Assets
    
    # Cash conversion
    cash_conversion_cycle: float  # Days
    dso_trend: TrendDirection  # Days sales outstanding
    dio_trend: TrendDirection  # Days inventory outstanding
    dpo_trend: TrendDirection  # Days payable outstanding
    
    # FCF sustainability
    capex_to_depreciation: float  # < 1 may indicate underinvestment
    maintenance_capex_estimate: float
    growth_capex_estimate: float
    
    # Working capital
    working_capital_change: float
    working_capital_to_revenue: float
    
    quality_score: float
    red_flags: List[str] = field(default_factory=list)
    key_observations: List[str] = field(default_factory=list)


@dataclass
class BalanceSheetAnalysis:
    """Balance sheet health analysis"""
    # Liquidity
    current_ratio: float
    quick_ratio: float
    cash_to_current_liabilities: float
    
    # Leverage
    debt_to_equity: float
    debt_to_ebitda: float
    net_debt_to_ebitda: float
    interest_coverage: float
    
    # Asset quality
    goodwill_to_assets: float
    intangibles_to_assets: float
    receivables_to_revenue: float
    inventory_to_cogs: float
    
    # Capital structure
    equity_ratio: float
    debt_maturity_profile: Dict[str, float]  # Years to maturity buckets
    weighted_avg_debt_cost: float
    
    # Trends
    leverage_trend: TrendDirection
    liquidity_trend: TrendDirection
    
    health_score: float
    risks: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)


@dataclass
class CapitalAllocationAnalysis:
    """Capital allocation effectiveness"""
    # Returns
    roic: float
    roe: float
    roa: float
    roic_vs_wacc: float  # Spread
    
    # Capital deployment
    capex_to_revenue: float
    rd_to_revenue: float
    ma_spending: float  # M&A as % of cash flow
    
    # Shareholder returns
    dividend_payout_ratio: float
    dividend_yield: float
    buyback_yield: float
    total_shareholder_return: float
    
    # Reinvestment
    reinvestment_rate: float
    organic_growth_investment: float
    
    # M&A track record
    ma_success_rate: Optional[float]
    goodwill_impairments: float
    
    # Trends
    roic_trend: TrendDirection
    capital_efficiency_trend: TrendDirection
    
    effectiveness_score: float
    observations: List[str] = field(default_factory=list)


@dataclass
class EarningsManipulationFlags:
    """Earnings manipulation detection"""
    # Beneish M-Score components
    dsri: float  # Days Sales in Receivables Index
    gmi: float  # Gross Margin Index
    aqi: float  # Asset Quality Index
    sgi: float  # Sales Growth Index
    depi: float  # Depreciation Index
    sgai: float  # SG&A Index
    lvgi: float  # Leverage Index
    tata: float  # Total Accruals to Total Assets
    
    m_score: float  # Beneish M-Score (> -1.78 suggests manipulation)
    manipulation_probability: float
    
    # Additional red flags
    revenue_recognition_concerns: List[str]
    expense_capitalization_flags: List[str]
    reserve_manipulation_flags: List[str]
    related_party_concerns: List[str]
    
    # Audit quality
    auditor_changes: int
    restatement_history: int
    material_weaknesses: int
    
    overall_risk: str  # Low, Medium, High
    detailed_flags: List[str] = field(default_factory=list)


@dataclass
class GuidanceAnalysis:
    """Forward guidance analysis"""
    # Current guidance
    revenue_guidance_low: Optional[float]
    revenue_guidance_high: Optional[float]
    eps_guidance_low: Optional[float]
    eps_guidance_high: Optional[float]
    
    # Guidance vs consensus
    revenue_vs_consensus: float  # % difference
    eps_vs_consensus: float
    
    # Guidance track record
    historical_beat_rate: float
    avg_beat_magnitude: float
    guidance_accuracy: float  # How close to actual
    
    # Guidance changes
    guidance_revision: str  # Raised, Maintained, Lowered
    revision_magnitude: float
    
    # Tone analysis
    management_tone: str  # Optimistic, Cautious, Neutral
    key_themes: List[str]
    risks_highlighted: List[str]
    
    credibility_score: float
    observations: List[str] = field(default_factory=list)


@dataclass
class EarningsAnalysisResult:
    """Complete earnings analysis result"""
    symbol: str
    analysis_date: datetime
    fiscal_period: str
    
    # Component analyses
    revenue: RevenueAnalysis
    margins: MarginAnalysis
    cash_flow: CashFlowAnalysis
    balance_sheet: BalanceSheetAnalysis
    capital_allocation: CapitalAllocationAnalysis
    manipulation_flags: EarningsManipulationFlags
    guidance: Optional[GuidanceAnalysis]
    
    # Overall assessment
    overall_quality: EarningsQuality
    quality_score: float  # 0-100
    confidence: float
    
    # Key metrics summary
    key_metrics: Dict[str, float]
    
    # Synthesis
    strengths: List[str]
    weaknesses: List[str]
    key_risks: List[str]
    catalysts: List[str]
    
    # Investment implications
    earnings_power: str  # Strong, Stable, Weak, Deteriorating
    sustainability_assessment: str
    
    summary: str
    detailed_notes: List[str] = field(default_factory=list)


class FinancialCalculator:
    """Financial ratio and metric calculations"""
    
    @staticmethod
    def calculate_growth(current: float, previous: float) -> float:
        """Calculate growth rate"""
        if previous == 0:
            return 0.0
        return (current - previous) / abs(previous)
    
    @staticmethod
    def calculate_cagr(start: float, end: float, years: int) -> float:
        """Calculate compound annual growth rate"""
        if start <= 0 or years == 0:
            return 0.0
        return (end / start) ** (1 / years) - 1
    
    @staticmethod
    def calculate_trend(values: List[float]) -> TrendDirection:
        """Determine trend direction from time series"""
        if len(values) < 3:
            return TrendDirection.STABLE
        
        # Simple linear regression
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return TrendDirection.STABLE
        
        slope = numerator / denominator
        
        # Normalize slope by mean value
        if y_mean != 0:
            normalized_slope = slope / abs(y_mean)
        else:
            normalized_slope = slope
        
        if normalized_slope > 0.1:
            return TrendDirection.STRONG_IMPROVING
        elif normalized_slope > 0.03:
            return TrendDirection.IMPROVING
        elif normalized_slope < -0.1:
            return TrendDirection.STRONG_DECLINING
        elif normalized_slope < -0.03:
            return TrendDirection.DECLINING
        else:
            return TrendDirection.STABLE
    
    @staticmethod
    def calculate_volatility(values: List[float]) -> float:
        """Calculate coefficient of variation"""
        if len(values) < 2:
            return 0.0
        mean = statistics.mean(values)
        if mean == 0:
            return 0.0
        return statistics.stdev(values) / abs(mean)


class BeneishMScore:
    """Beneish M-Score for earnings manipulation detection"""
    
    @staticmethod
    def calculate_dsri(receivables_t: float, revenue_t: float,
                       receivables_t1: float, revenue_t1: float) -> float:
        """Days Sales in Receivables Index"""
        if revenue_t == 0 or revenue_t1 == 0 or receivables_t1 == 0:
            return 1.0
        dsr_t = receivables_t / revenue_t
        dsr_t1 = receivables_t1 / revenue_t1
        return dsr_t / dsr_t1 if dsr_t1 != 0 else 1.0
    
    @staticmethod
    def calculate_gmi(gross_margin_t1: float, gross_margin_t: float) -> float:
        """Gross Margin Index"""
        if gross_margin_t == 0:
            return 1.0
        return gross_margin_t1 / gross_margin_t
    
    @staticmethod
    def calculate_aqi(assets_t: float, ppe_t: float, ca_t: float,
                      assets_t1: float, ppe_t1: float, ca_t1: float) -> float:
        """Asset Quality Index"""
        if assets_t == 0 or assets_t1 == 0:
            return 1.0
        aq_t = 1 - (ca_t + ppe_t) / assets_t
        aq_t1 = 1 - (ca_t1 + ppe_t1) / assets_t1
        return aq_t / aq_t1 if aq_t1 != 0 else 1.0
    
    @staticmethod
    def calculate_sgi(revenue_t: float, revenue_t1: float) -> float:
        """Sales Growth Index"""
        if revenue_t1 == 0:
            return 1.0
        return revenue_t / revenue_t1
    
    @staticmethod
    def calculate_depi(depreciation_t1: float, ppe_t1: float,
                       depreciation_t: float, ppe_t: float) -> float:
        """Depreciation Index"""
        if ppe_t == 0 or ppe_t1 == 0:
            return 1.0
        dep_rate_t1 = depreciation_t1 / (ppe_t1 + depreciation_t1) if (ppe_t1 + depreciation_t1) != 0 else 0
        dep_rate_t = depreciation_t / (ppe_t + depreciation_t) if (ppe_t + depreciation_t) != 0 else 0
        return dep_rate_t1 / dep_rate_t if dep_rate_t != 0 else 1.0
    
    @staticmethod
    def calculate_sgai(sga_t: float, revenue_t: float,
                       sga_t1: float, revenue_t1: float) -> float:
        """SG&A Index"""
        if revenue_t == 0 or revenue_t1 == 0:
            return 1.0
        sgai_t = sga_t / revenue_t
        sgai_t1 = sga_t1 / revenue_t1
        return sgai_t / sgai_t1 if sgai_t1 != 0 else 1.0
    
    @staticmethod
    def calculate_lvgi(debt_t: float, assets_t: float,
                       debt_t1: float, assets_t1: float) -> float:
        """Leverage Index"""
        if assets_t == 0 or assets_t1 == 0:
            return 1.0
        lev_t = debt_t / assets_t
        lev_t1 = debt_t1 / assets_t1
        return lev_t / lev_t1 if lev_t1 != 0 else 1.0
    
    @staticmethod
    def calculate_tata(net_income: float, ocf: float, assets: float) -> float:
        """Total Accruals to Total Assets"""
        if assets == 0:
            return 0.0
        return (net_income - ocf) / assets
    
    @classmethod
    def calculate_m_score(cls, dsri: float, gmi: float, aqi: float, sgi: float,
                          depi: float, sgai: float, lvgi: float, tata: float) -> float:
        """Calculate Beneish M-Score"""
        return (-4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 
                0.892 * sgi + 0.115 * depi - 0.172 * sgai + 
                4.679 * tata - 0.327 * lvgi)


class EarningsAgent:
    """
    Agent for comprehensive earnings and financial statement analysis
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.calculator = FinancialCalculator()
        self.beneish = BeneishMScore()
    
    async def analyze(self, context: Dict[str, Any]) -> EarningsAnalysisResult:
        """
        Perform comprehensive earnings analysis
        
        Args:
            context: Analysis context with financial data
            
        Returns:
            Complete earnings analysis result
        """
        symbol = context.get("symbol", "UNKNOWN")
        
        # Get financial data
        financials = context.get("financials", {})
        income_stmt = financials.get("income_statement", [])
        balance_sheet = financials.get("balance_sheet", [])
        cash_flow_stmt = financials.get("cash_flow", [])
        
        # If no real data, use mock data for demonstration
        if not income_stmt:
            financials = self._get_mock_financials(symbol)
            income_stmt = financials["income_statement"]
            balance_sheet = financials["balance_sheet"]
            cash_flow_stmt = financials["cash_flow"]
        
        # Perform component analyses
        revenue_analysis = self._analyze_revenue(income_stmt, context)
        margin_analysis = self._analyze_margins(income_stmt, context)
        cash_flow_analysis = self._analyze_cash_flow(cash_flow_stmt, income_stmt, balance_sheet)
        balance_sheet_analysis = self._analyze_balance_sheet(balance_sheet, income_stmt)
        capital_allocation = self._analyze_capital_allocation(
            income_stmt, balance_sheet, cash_flow_stmt, context
        )
        manipulation_flags = self._detect_manipulation(
            income_stmt, balance_sheet, cash_flow_stmt
        )
        guidance_analysis = self._analyze_guidance(context)
        
        # Calculate overall quality score
        quality_score = self._calculate_overall_quality(
            revenue_analysis, margin_analysis, cash_flow_analysis,
            balance_sheet_analysis, capital_allocation, manipulation_flags
        )
        
        # Determine quality rating
        overall_quality = self._determine_quality_rating(quality_score)
        
        # Extract key metrics
        key_metrics = self._extract_key_metrics(
            revenue_analysis, margin_analysis, cash_flow_analysis,
            balance_sheet_analysis, capital_allocation
        )
        
        # Synthesize findings
        strengths, weaknesses, risks, catalysts = self._synthesize_findings(
            revenue_analysis, margin_analysis, cash_flow_analysis,
            balance_sheet_analysis, capital_allocation, manipulation_flags
        )
        
        # Assess earnings power
        earnings_power = self._assess_earnings_power(
            margin_analysis, cash_flow_analysis, capital_allocation
        )
        
        # Generate summary
        summary = self._generate_summary(
            symbol, overall_quality, quality_score, key_metrics,
            strengths, weaknesses, risks
        )
        
        return EarningsAnalysisResult(
            symbol=symbol,
            analysis_date=datetime.now(),
            fiscal_period=context.get("fiscal_period", "FY2024"),
            revenue=revenue_analysis,
            margins=margin_analysis,
            cash_flow=cash_flow_analysis,
            balance_sheet=balance_sheet_analysis,
            capital_allocation=capital_allocation,
            manipulation_flags=manipulation_flags,
            guidance=guidance_analysis,
            overall_quality=overall_quality,
            quality_score=quality_score,
            confidence=0.75,
            key_metrics=key_metrics,
            strengths=strengths,
            weaknesses=weaknesses,
            key_risks=risks,
            catalysts=catalysts,
            earnings_power=earnings_power,
            sustainability_assessment=self._assess_sustainability(
                revenue_analysis, margin_analysis, cash_flow_analysis
            ),
            summary=summary
        )
    
    def _analyze_revenue(self, income_stmt: List[Dict], 
                         context: Dict) -> RevenueAnalysis:
        """Analyze revenue quality and sustainability"""
        if not income_stmt:
            return self._get_default_revenue_analysis()
        
        current = income_stmt[0]
        previous = income_stmt[1] if len(income_stmt) > 1 else current
        
        revenue = current.get("revenue", 0)
        prev_revenue = previous.get("revenue", 0)
        
        yoy_growth = self.calculator.calculate_growth(revenue, prev_revenue)
        
        # Extract revenue data for multiple periods
        revenues = [stmt.get("revenue", 0) for stmt in income_stmt[:8]]
        revenue_trend = self.calculator.calculate_trend(revenues)
        
        # Estimate QoQ if quarterly data available
        qoq_growth = yoy_growth / 4  # Simplified
        
        # Revenue quality metrics (would come from detailed segment data)
        recurring_pct = context.get("recurring_revenue_pct", 0.65)
        customer_concentration = context.get("customer_concentration", 0.25)
        geo_diversification = context.get("geographic_hhi", 0.35)
        
        # Revenue driver decomposition (estimated)
        volume_contribution = 0.6 * yoy_growth if yoy_growth > 0 else 0.8 * yoy_growth
        price_contribution = 0.3 * yoy_growth if yoy_growth > 0 else 0.1 * yoy_growth
        mix_contribution = 0.1 * yoy_growth if yoy_growth > 0 else 0.1 * yoy_growth
        
        # Calculate quality score
        quality_score = self._calculate_revenue_quality_score(
            yoy_growth, revenue_trend, recurring_pct,
            customer_concentration, geo_diversification
        )
        
        observations = []
        risks = []
        
        if yoy_growth > 0.15:
            observations.append(f"Strong revenue growth of {yoy_growth:.1%}")
        elif yoy_growth < 0:
            risks.append(f"Revenue decline of {yoy_growth:.1%}")
        
        if recurring_pct > 0.7:
            observations.append(f"High recurring revenue ({recurring_pct:.0%}) provides stability")
        
        if customer_concentration > 0.4:
            risks.append(f"High customer concentration ({customer_concentration:.0%})")
        
        return RevenueAnalysis(
            total_revenue=revenue,
            yoy_growth=yoy_growth,
            qoq_growth=qoq_growth,
            revenue_trend=revenue_trend,
            recurring_revenue_pct=recurring_pct,
            customer_concentration=customer_concentration,
            geographic_diversification=1 - geo_diversification,
            volume_contribution=volume_contribution,
            price_contribution=price_contribution,
            mix_contribution=mix_contribution,
            backlog_growth=context.get("backlog_growth"),
            deferred_revenue_growth=context.get("deferred_revenue_growth"),
            customer_retention_rate=context.get("customer_retention_rate"),
            quality_score=quality_score,
            key_observations=observations,
            risks=risks
        )
    
    def _calculate_revenue_quality_score(self, growth: float, trend: TrendDirection,
                                         recurring: float, concentration: float,
                                         geo_hhi: float) -> float:
        """Calculate revenue quality score 0-100"""
        score = 50  # Base score
        
        # Growth contribution
        if growth > 0.20:
            score += 15
        elif growth > 0.10:
            score += 10
        elif growth > 0.05:
            score += 5
        elif growth < -0.05:
            score -= 10
        elif growth < -0.10:
            score -= 20
        
        # Trend contribution
        if trend == TrendDirection.STRONG_IMPROVING:
            score += 10
        elif trend == TrendDirection.IMPROVING:
            score += 5
        elif trend == TrendDirection.DECLINING:
            score -= 5
        elif trend == TrendDirection.STRONG_DECLINING:
            score -= 10
        
        # Recurring revenue
        score += (recurring - 0.5) * 20  # 0.5 is neutral
        
        # Concentration penalty
        score -= max(0, (concentration - 0.3) * 30)
        
        # Geographic diversification bonus
        score += (1 - geo_hhi) * 10
        
        return max(0, min(100, score))
    
    def _analyze_margins(self, income_stmt: List[Dict],
                         context: Dict) -> MarginAnalysis:
        """Analyze margin trends and quality"""
        if not income_stmt:
            return self._get_default_margin_analysis()
        
        current = income_stmt[0]
        previous = income_stmt[1] if len(income_stmt) > 1 else current
        
        revenue = current.get("revenue", 1)
        
        # Calculate margins
        gross_profit = current.get("gross_profit", revenue * 0.4)
        operating_income = current.get("operating_income", revenue * 0.15)
        ebitda = current.get("ebitda", operating_income * 1.2)
        net_income = current.get("net_income", revenue * 0.10)
        
        gross_margin = gross_profit / revenue if revenue else 0
        operating_margin = operating_income / revenue if revenue else 0
        ebitda_margin = ebitda / revenue if revenue else 0
        net_margin = net_income / revenue if revenue else 0
        
        # Calculate FCF margin
        fcf = context.get("free_cash_flow", net_income * 1.1)
        fcf_margin = fcf / revenue if revenue else 0
        
        # Previous margins
        prev_revenue = previous.get("revenue", 1)
        prev_gross = previous.get("gross_profit", prev_revenue * 0.4) / prev_revenue
        prev_operating = previous.get("operating_income", prev_revenue * 0.15) / prev_revenue
        prev_net = previous.get("net_income", prev_revenue * 0.10) / prev_revenue
        
        # Margin changes in basis points
        gross_change = (gross_margin - prev_gross) * 10000
        operating_change = (operating_margin - prev_operating) * 10000
        net_change = (net_margin - prev_net) * 10000
        
        # Cost structure
        cogs = current.get("cost_of_revenue", revenue * 0.6)
        sga = current.get("sga", revenue * 0.15)
        rd = current.get("rd", revenue * 0.05)
        
        # Margin stability
        gross_margins = [stmt.get("gross_profit", 0) / max(stmt.get("revenue", 1), 1) 
                        for stmt in income_stmt[:8]]
        margin_stability = 1 - self.calculator.calculate_volatility(gross_margins)
        
        # Operating leverage
        rev_growth = self.calculator.calculate_growth(revenue, prev_revenue)
        oi_growth = self.calculator.calculate_growth(
            operating_income, previous.get("operating_income", 1)
        )
        operating_leverage = oi_growth / rev_growth if rev_growth != 0 else 1.0
        
        # Trend
        margin_trend = self.calculator.calculate_trend(gross_margins)
        
        # Industry comparison (would come from sector data)
        industry_gross = context.get("industry_gross_margin", 0.35)
        industry_operating = context.get("industry_operating_margin", 0.12)
        
        quality_score = self._calculate_margin_quality_score(
            gross_margin, operating_margin, margin_trend,
            margin_stability, operating_leverage
        )
        
        observations = []
        concerns = []
        
        if gross_margin > industry_gross + 0.10:
            observations.append(f"Gross margin {(gross_margin - industry_gross)*100:.0f}bps above industry")
        if operating_margin > industry_operating + 0.05:
            observations.append("Strong operating leverage vs peers")
        
        if gross_change < -100:
            concerns.append(f"Gross margin contracted {abs(gross_change):.0f}bps YoY")
        if operating_leverage < 0 and rev_growth > 0:
            concerns.append("Negative operating leverage despite revenue growth")
        
        return MarginAnalysis(
            gross_margin=gross_margin,
            operating_margin=operating_margin,
            ebitda_margin=ebitda_margin,
            net_margin=net_margin,
            fcf_margin=fcf_margin,
            gross_margin_change=gross_change,
            operating_margin_change=operating_change,
            net_margin_change=net_change,
            cogs_to_revenue=cogs / revenue if revenue else 0,
            sga_to_revenue=sga / revenue if revenue else 0,
            rd_to_revenue=rd / revenue if revenue else 0,
            margin_stability=margin_stability,
            margin_trend=margin_trend,
            operating_leverage=operating_leverage,
            vs_industry_gross=gross_margin - industry_gross,
            vs_industry_operating=operating_margin - industry_operating,
            quality_score=quality_score,
            key_observations=observations,
            concerns=concerns
        )
    
    def _calculate_margin_quality_score(self, gross: float, operating: float,
                                        trend: TrendDirection, stability: float,
                                        leverage: float) -> float:
        """Calculate margin quality score"""
        score = 50
        
        # Absolute margin levels
        if gross > 0.5:
            score += 15
        elif gross > 0.35:
            score += 10
        elif gross < 0.20:
            score -= 10
        
        if operating > 0.20:
            score += 10
        elif operating > 0.10:
            score += 5
        elif operating < 0:
            score -= 15
        
        # Trend
        if trend == TrendDirection.STRONG_IMPROVING:
            score += 10
        elif trend == TrendDirection.IMPROVING:
            score += 5
        elif trend == TrendDirection.DECLINING:
            score -= 5
        elif trend == TrendDirection.STRONG_DECLINING:
            score -= 10
        
        # Stability bonus
        score += stability * 10
        
        # Operating leverage
        if leverage > 1.5:
            score += 5
        elif leverage < 0:
            score -= 5
        
        return max(0, min(100, score))
    
    def _analyze_cash_flow(self, cash_flow_stmt: List[Dict],
                           income_stmt: List[Dict],
                           balance_sheet: List[Dict]) -> CashFlowAnalysis:
        """Analyze cash flow quality"""
        if not cash_flow_stmt or not income_stmt:
            return self._get_default_cash_flow_analysis()
        
        cf_current = cash_flow_stmt[0]
        is_current = income_stmt[0]
        bs_current = balance_sheet[0] if balance_sheet else {}
        
        ocf = cf_current.get("operating_cash_flow", 0)
        capex = abs(cf_current.get("capex", 0))
        fcf = ocf - capex
        
        net_income = is_current.get("net_income", 1)
        total_assets = bs_current.get("total_assets", 1)
        shares = is_current.get("shares_outstanding", 1)
        
        # Cash flow quality metrics
        ocf_to_ni = ocf / net_income if net_income != 0 else 1.0
        fcf_to_ni = fcf / net_income if net_income != 0 else 1.0
        accrual_ratio = (net_income - ocf) / total_assets if total_assets != 0 else 0
        
        # Cash conversion cycle components
        revenue = is_current.get("revenue", 1)
        cogs = is_current.get("cost_of_revenue", revenue * 0.6)
        receivables = bs_current.get("receivables", revenue * 0.1)
        inventory = bs_current.get("inventory", cogs * 0.15)
        payables = bs_current.get("payables", cogs * 0.1)
        
        dso = (receivables / revenue) * 365 if revenue else 0
        dio = (inventory / cogs) * 365 if cogs else 0
        dpo = (payables / cogs) * 365 if cogs else 0
        ccc = dso + dio - dpo
        
        # Capex analysis
        depreciation = cf_current.get("depreciation", capex * 0.8)
        capex_to_dep = capex / depreciation if depreciation != 0 else 1.0
        maintenance_capex = depreciation * 0.8  # Estimate
        growth_capex = max(0, capex - maintenance_capex)
        
        # Working capital
        wc_change = cf_current.get("working_capital_change", 0)
        wc_to_revenue = wc_change / revenue if revenue else 0
        
        # Calculate quality score
        quality_score = self._calculate_cash_flow_quality(
            ocf_to_ni, accrual_ratio, ccc, capex_to_dep
        )
        
        red_flags = []
        observations = []
        
        if ocf_to_ni < 0.8:
            red_flags.append(f"Low cash conversion: OCF/NI = {ocf_to_ni:.2f}")
        if accrual_ratio > 0.10:
            red_flags.append(f"High accruals ({accrual_ratio:.1%}) may indicate low earnings quality")
        if capex_to_dep < 0.7:
            red_flags.append("Potential underinvestment (CapEx < Depreciation)")
        
        if ocf_to_ni > 1.2:
            observations.append("Strong cash conversion (OCF > Net Income)")
        if fcf > net_income:
            observations.append("FCF exceeds reported earnings")
        
        return CashFlowAnalysis(
            operating_cash_flow=ocf,
            free_cash_flow=fcf,
            fcf_per_share=fcf / shares if shares else 0,
            ocf_to_net_income=ocf_to_ni,
            fcf_to_net_income=fcf_to_ni,
            accrual_ratio=accrual_ratio,
            cash_conversion_cycle=ccc,
            dso_trend=TrendDirection.STABLE,  # Would calculate from history
            dio_trend=TrendDirection.STABLE,
            dpo_trend=TrendDirection.STABLE,
            capex_to_depreciation=capex_to_dep,
            maintenance_capex_estimate=maintenance_capex,
            growth_capex_estimate=growth_capex,
            working_capital_change=wc_change,
            working_capital_to_revenue=wc_to_revenue,
            quality_score=quality_score,
            red_flags=red_flags,
            key_observations=observations
        )
    
    def _calculate_cash_flow_quality(self, ocf_to_ni: float, accrual: float,
                                     ccc: float, capex_dep: float) -> float:
        """Calculate cash flow quality score"""
        score = 50
        
        # OCF to NI ratio
        if ocf_to_ni > 1.2:
            score += 20
        elif ocf_to_ni > 1.0:
            score += 10
        elif ocf_to_ni < 0.7:
            score -= 15
        elif ocf_to_ni < 0.5:
            score -= 25
        
        # Accrual quality
        if abs(accrual) < 0.03:
            score += 10
        elif abs(accrual) > 0.10:
            score -= 15
        
        # Cash conversion cycle
        if ccc < 30:
            score += 10
        elif ccc > 90:
            score -= 10
        
        # Reinvestment
        if 0.8 <= capex_dep <= 1.5:
            score += 5  # Healthy reinvestment
        elif capex_dep < 0.5:
            score -= 10  # Underinvestment
        
        return max(0, min(100, score))
    
    def _analyze_balance_sheet(self, balance_sheet: List[Dict],
                               income_stmt: List[Dict]) -> BalanceSheetAnalysis:
        """Analyze balance sheet health"""
        if not balance_sheet:
            return self._get_default_balance_sheet_analysis()
        
        bs = balance_sheet[0]
        is_current = income_stmt[0] if income_stmt else {}
        
        # Extract values
        current_assets = bs.get("current_assets", 0)
        current_liabilities = bs.get("current_liabilities", 1)
        cash = bs.get("cash", 0)
        inventory = bs.get("inventory", 0)
        receivables = bs.get("receivables", 0)
        
        total_assets = bs.get("total_assets", 1)
        total_debt = bs.get("total_debt", 0)
        total_equity = bs.get("total_equity", 1)
        
        goodwill = bs.get("goodwill", 0)
        intangibles = bs.get("intangibles", 0)
        
        revenue = is_current.get("revenue", 1)
        cogs = is_current.get("cost_of_revenue", revenue * 0.6)
        ebitda = is_current.get("ebitda", revenue * 0.15)
        interest_expense = is_current.get("interest_expense", total_debt * 0.05)
        
        # Calculate ratios
        current_ratio = current_assets / current_liabilities if current_liabilities else 2.0
        quick_ratio = (current_assets - inventory) / current_liabilities if current_liabilities else 1.5
        cash_to_cl = cash / current_liabilities if current_liabilities else 0.5
        
        debt_to_equity = total_debt / total_equity if total_equity else 0
        debt_to_ebitda = total_debt / ebitda if ebitda else 0
        net_debt = total_debt - cash
        net_debt_to_ebitda = net_debt / ebitda if ebitda else 0
        interest_coverage = ebitda / interest_expense if interest_expense else 10.0
        
        goodwill_to_assets = goodwill / total_assets if total_assets else 0
        intangibles_to_assets = intangibles / total_assets if total_assets else 0
        receivables_to_revenue = receivables / revenue if revenue else 0
        inventory_to_cogs = inventory / cogs if cogs else 0
        
        equity_ratio = total_equity / total_assets if total_assets else 0.5
        
        # Calculate health score
        health_score = self._calculate_balance_sheet_health(
            current_ratio, debt_to_equity, debt_to_ebitda,
            interest_coverage, goodwill_to_assets
        )
        
        risks = []
        strengths = []
        
        if current_ratio < 1.0:
            risks.append(f"Low liquidity (current ratio: {current_ratio:.2f})")
        if debt_to_ebitda > 4.0:
            risks.append(f"High leverage (Debt/EBITDA: {debt_to_ebitda:.1f}x)")
        if goodwill_to_assets > 0.3:
            risks.append(f"High goodwill ({goodwill_to_assets:.0%} of assets)")
        
        if net_debt < 0:
            strengths.append("Net cash position")
        if interest_coverage > 10:
            strengths.append(f"Strong interest coverage ({interest_coverage:.1f}x)")
        if current_ratio > 2.0:
            strengths.append("Excellent liquidity")
        
        return BalanceSheetAnalysis(
            current_ratio=current_ratio,
            quick_ratio=quick_ratio,
            cash_to_current_liabilities=cash_to_cl,
            debt_to_equity=debt_to_equity,
            debt_to_ebitda=debt_to_ebitda,
            net_debt_to_ebitda=net_debt_to_ebitda,
            interest_coverage=interest_coverage,
            goodwill_to_assets=goodwill_to_assets,
            intangibles_to_assets=intangibles_to_assets,
            receivables_to_revenue=receivables_to_revenue,
            inventory_to_cogs=inventory_to_cogs,
            equity_ratio=equity_ratio,
            debt_maturity_profile={"0-1yr": 0.2, "1-3yr": 0.3, "3-5yr": 0.3, "5yr+": 0.2},
            weighted_avg_debt_cost=0.05,
            leverage_trend=TrendDirection.STABLE,
            liquidity_trend=TrendDirection.STABLE,
            health_score=health_score,
            risks=risks,
            strengths=strengths
        )
    
    def _calculate_balance_sheet_health(self, current: float, de: float,
                                        debt_ebitda: float, coverage: float,
                                        goodwill: float) -> float:
        """Calculate balance sheet health score"""
        score = 50
        
        # Liquidity
        if current > 2.0:
            score += 15
        elif current > 1.5:
            score += 10
        elif current < 1.0:
            score -= 15
        
        # Leverage
        if de < 0.3:
            score += 10
        elif de > 1.5:
            score -= 10
        elif de > 2.5:
            score -= 20
        
        if debt_ebitda > 4:
            score -= 15
        elif debt_ebitda < 2:
            score += 10
        
        # Coverage
        if coverage > 10:
            score += 10
        elif coverage < 3:
            score -= 15
        
        # Asset quality
        if goodwill > 0.3:
            score -= 10
        
        return max(0, min(100, score))
    
    def _analyze_capital_allocation(self, income_stmt: List[Dict],
                                    balance_sheet: List[Dict],
                                    cash_flow: List[Dict],
                                    context: Dict) -> CapitalAllocationAnalysis:
        """Analyze capital allocation effectiveness"""
        is_current = income_stmt[0] if income_stmt else {}
        bs = balance_sheet[0] if balance_sheet else {}
        cf = cash_flow[0] if cash_flow else {}
        
        net_income = is_current.get("net_income", 0)
        revenue = is_current.get("revenue", 1)
        total_equity = bs.get("total_equity", 1)
        total_assets = bs.get("total_assets", 1)
        invested_capital = total_equity + bs.get("total_debt", 0) - bs.get("cash", 0)
        
        # Returns
        roe = net_income / total_equity if total_equity else 0
        roa = net_income / total_assets if total_assets else 0
        
        # ROIC calculation
        nopat = is_current.get("operating_income", net_income) * 0.75  # Tax-adjusted
        roic = nopat / invested_capital if invested_capital else 0
        
        wacc = context.get("wacc", 0.10)
        roic_spread = roic - wacc
        
        # Capital deployment
        capex = abs(cf.get("capex", 0))
        rd = is_current.get("rd", 0)
        
        # Shareholder returns
        dividends = cf.get("dividends_paid", 0)
        buybacks = cf.get("share_repurchases", 0)
        
        dividend_payout = abs(dividends) / net_income if net_income else 0
        dividend_yield = context.get("dividend_yield", 0.02)
        buyback_yield = abs(buybacks) / context.get("market_cap", 1)
        
        # Reinvestment
        reinvestment = capex + rd
        reinvestment_rate = reinvestment / nopat if nopat else 0
        
        effectiveness_score = self._calculate_capital_effectiveness(
            roic, roic_spread, dividend_payout, reinvestment_rate
        )
        
        observations = []
        if roic > wacc:
            observations.append(f"ROIC ({roic:.1%}) exceeds WACC ({wacc:.1%}), creating value")
        else:
            observations.append(f"ROIC ({roic:.1%}) below WACC ({wacc:.1%}), destroying value")
        
        if dividend_payout > 0.6:
            observations.append("High dividend payout may limit reinvestment")
        
        return CapitalAllocationAnalysis(
            roic=roic,
            roe=roe,
            roa=roa,
            roic_vs_wacc=roic_spread,
            capex_to_revenue=capex / revenue if revenue else 0,
            rd_to_revenue=rd / revenue if revenue else 0,
            ma_spending=context.get("ma_spending", 0),
            dividend_payout_ratio=dividend_payout,
            dividend_yield=dividend_yield,
            buyback_yield=buyback_yield,
            total_shareholder_return=dividend_yield + buyback_yield,
            reinvestment_rate=reinvestment_rate,
            organic_growth_investment=reinvestment / revenue if revenue else 0,
            ma_success_rate=context.get("ma_success_rate"),
            goodwill_impairments=0,
            roic_trend=TrendDirection.STABLE,
            capital_efficiency_trend=TrendDirection.STABLE,
            effectiveness_score=effectiveness_score,
            observations=observations
        )
    
    def _calculate_capital_effectiveness(self, roic: float, spread: float,
                                         payout: float, reinvest: float) -> float:
        """Calculate capital allocation effectiveness score"""
        score = 50
        
        if roic > 0.20:
            score += 20
        elif roic > 0.12:
            score += 10
        elif roic < 0.05:
            score -= 15
        
        if spread > 0.05:
            score += 15
        elif spread < -0.03:
            score -= 20
        
        # Balanced payout
        if 0.3 <= payout <= 0.5:
            score += 5
        elif payout > 0.8:
            score -= 5
        
        return max(0, min(100, score))
    
    def _detect_manipulation(self, income_stmt: List[Dict],
                            balance_sheet: List[Dict],
                            cash_flow: List[Dict]) -> EarningsManipulationFlags:
        """Detect potential earnings manipulation using Beneish M-Score"""
        if len(income_stmt) < 2 or len(balance_sheet) < 2:
            return self._get_default_manipulation_flags()
        
        is_t = income_stmt[0]
        is_t1 = income_stmt[1]
        bs_t = balance_sheet[0]
        bs_t1 = balance_sheet[1]
        cf_t = cash_flow[0] if cash_flow else {}
        
        # Calculate M-Score components
        dsri = self.beneish.calculate_dsri(
            bs_t.get("receivables", 0), is_t.get("revenue", 1),
            bs_t1.get("receivables", 0), is_t1.get("revenue", 1)
        )
        
        gmi = self.beneish.calculate_gmi(
            is_t1.get("gross_profit", 1) / max(is_t1.get("revenue", 1), 1),
            is_t.get("gross_profit", 1) / max(is_t.get("revenue", 1), 1)
        )
        
        aqi = self.beneish.calculate_aqi(
            bs_t.get("total_assets", 1), bs_t.get("ppe", 0), bs_t.get("current_assets", 0),
            bs_t1.get("total_assets", 1), bs_t1.get("ppe", 0), bs_t1.get("current_assets", 0)
        )
        
        sgi = self.beneish.calculate_sgi(
            is_t.get("revenue", 0), is_t1.get("revenue", 1)
        )
        
        depi = self.beneish.calculate_depi(
            is_t1.get("depreciation", 0), bs_t1.get("ppe", 1),
            is_t.get("depreciation", 0), bs_t.get("ppe", 1)
        )
        
        sgai = self.beneish.calculate_sgai(
            is_t.get("sga", 0), is_t.get("revenue", 1),
            is_t1.get("sga", 0), is_t1.get("revenue", 1)
        )
        
        lvgi = self.beneish.calculate_lvgi(
            bs_t.get("total_debt", 0), bs_t.get("total_assets", 1),
            bs_t1.get("total_debt", 0), bs_t1.get("total_assets", 1)
        )
        
        tata = self.beneish.calculate_tata(
            is_t.get("net_income", 0),
            cf_t.get("operating_cash_flow", 0),
            bs_t.get("total_assets", 1)
        )
        
        # Calculate M-Score
        m_score = self.beneish.calculate_m_score(
            dsri, gmi, aqi, sgi, depi, sgai, lvgi, tata
        )
        
        # Probability of manipulation
        manipulation_prob = 0.1  # Base probability
        if m_score > -1.78:
            manipulation_prob = 0.5 + (m_score + 1.78) * 0.1
        manipulation_prob = min(0.95, max(0.05, manipulation_prob))
        
        # Additional flags
        flags = []
        revenue_flags = []
        expense_flags = []
        reserve_flags = []
        
        if dsri > 1.2:
            revenue_flags.append("Receivables growing faster than revenue")
        if tata > 0.10:
            flags.append("High accruals relative to assets")
        if sgi > 1.5:
            flags.append("Unusually high sales growth warrants scrutiny")
        if gmi > 1.1:
            flags.append("Gross margin deterioration")
        
        # Overall risk
        if m_score > -1.78:
            risk = "High"
        elif m_score > -2.2:
            risk = "Medium"
        else:
            risk = "Low"
        
        return EarningsManipulationFlags(
            dsri=dsri,
            gmi=gmi,
            aqi=aqi,
            sgi=sgi,
            depi=depi,
            sgai=sgai,
            lvgi=lvgi,
            tata=tata,
            m_score=m_score,
            manipulation_probability=manipulation_prob,
            revenue_recognition_concerns=revenue_flags,
            expense_capitalization_flags=expense_flags,
            reserve_manipulation_flags=reserve_flags,
            related_party_concerns=[],
            auditor_changes=0,
            restatement_history=0,
            material_weaknesses=0,
            overall_risk=risk,
            detailed_flags=flags
        )
    
    def _analyze_guidance(self, context: Dict) -> Optional[GuidanceAnalysis]:
        """Analyze forward guidance if available"""
        guidance = context.get("guidance", {})
        if not guidance:
            return None
        
        return GuidanceAnalysis(
            revenue_guidance_low=guidance.get("revenue_low"),
            revenue_guidance_high=guidance.get("revenue_high"),
            eps_guidance_low=guidance.get("eps_low"),
            eps_guidance_high=guidance.get("eps_high"),
            revenue_vs_consensus=guidance.get("revenue_vs_consensus", 0),
            eps_vs_consensus=guidance.get("eps_vs_consensus", 0),
            historical_beat_rate=guidance.get("beat_rate", 0.7),
            avg_beat_magnitude=guidance.get("avg_beat", 0.02),
            guidance_accuracy=guidance.get("accuracy", 0.85),
            guidance_revision=guidance.get("revision", "Maintained"),
            revision_magnitude=guidance.get("revision_magnitude", 0),
            management_tone=guidance.get("tone", "Neutral"),
            key_themes=guidance.get("themes", []),
            risks_highlighted=guidance.get("risks", []),
            credibility_score=guidance.get("credibility", 70),
            observations=[]
        )
    
    def _calculate_overall_quality(self, revenue: RevenueAnalysis,
                                   margins: MarginAnalysis,
                                   cash_flow: CashFlowAnalysis,
                                   balance_sheet: BalanceSheetAnalysis,
                                   capital: CapitalAllocationAnalysis,
                                   manipulation: EarningsManipulationFlags) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            "revenue": 0.20,
            "margins": 0.20,
            "cash_flow": 0.25,
            "balance_sheet": 0.15,
            "capital": 0.15,
            "manipulation": 0.05
        }
        
        # Manipulation score (invert - higher M-score is worse)
        manipulation_score = max(0, min(100, 50 - manipulation.m_score * 20))
        
        weighted_score = (
            revenue.quality_score * weights["revenue"] +
            margins.quality_score * weights["margins"] +
            cash_flow.quality_score * weights["cash_flow"] +
            balance_sheet.health_score * weights["balance_sheet"] +
            capital.effectiveness_score * weights["capital"] +
            manipulation_score * weights["manipulation"]
        )
        
        return weighted_score
    
    def _determine_quality_rating(self, score: float) -> EarningsQuality:
        """Determine earnings quality rating from score"""
        if score >= 80:
            return EarningsQuality.EXCELLENT
        elif score >= 65:
            return EarningsQuality.GOOD
        elif score >= 50:
            return EarningsQuality.FAIR
        elif score >= 35:
            return EarningsQuality.POOR
        else:
            return EarningsQuality.CONCERNING
    
    def _extract_key_metrics(self, revenue: RevenueAnalysis,
                             margins: MarginAnalysis,
                             cash_flow: CashFlowAnalysis,
                             balance_sheet: BalanceSheetAnalysis,
                             capital: CapitalAllocationAnalysis) -> Dict[str, float]:
        """Extract key metrics summary"""
        return {
            "revenue_growth": revenue.yoy_growth,
            "recurring_revenue_pct": revenue.recurring_revenue_pct,
            "gross_margin": margins.gross_margin,
            "operating_margin": margins.operating_margin,
            "net_margin": margins.net_margin,
            "fcf_margin": margins.fcf_margin,
            "ocf_to_net_income": cash_flow.ocf_to_net_income,
            "accrual_ratio": cash_flow.accrual_ratio,
            "current_ratio": balance_sheet.current_ratio,
            "debt_to_ebitda": balance_sheet.debt_to_ebitda,
            "interest_coverage": balance_sheet.interest_coverage,
            "roic": capital.roic,
            "roe": capital.roe,
            "roic_vs_wacc": capital.roic_vs_wacc,
            "dividend_yield": capital.dividend_yield
        }
    
    def _synthesize_findings(self, revenue: RevenueAnalysis,
                            margins: MarginAnalysis,
                            cash_flow: CashFlowAnalysis,
                            balance_sheet: BalanceSheetAnalysis,
                            capital: CapitalAllocationAnalysis,
                            manipulation: EarningsManipulationFlags
                            ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """Synthesize findings into strengths, weaknesses, risks, catalysts"""
        strengths = []
        weaknesses = []
        risks = []
        catalysts = []
        
        # Revenue
        if revenue.yoy_growth > 0.10:
            strengths.append(f"Strong revenue growth ({revenue.yoy_growth:.1%})")
            catalysts.append("Continued revenue momentum")
        elif revenue.yoy_growth < 0:
            weaknesses.append(f"Revenue decline ({revenue.yoy_growth:.1%})")
            risks.append("Revenue trajectory uncertainty")
        
        if revenue.recurring_revenue_pct > 0.7:
            strengths.append("High recurring revenue provides visibility")
        
        # Margins
        if margins.gross_margin > 0.5:
            strengths.append(f"Premium gross margins ({margins.gross_margin:.1%})")
        if margins.operating_margin > 0.20:
            strengths.append("Excellent operating leverage")
        
        if margins.gross_margin_change < -200:
            weaknesses.append("Margin compression")
            risks.append("Continued margin pressure")
        
        # Cash flow
        if cash_flow.ocf_to_net_income > 1.2:
            strengths.append("Strong cash conversion")
        elif cash_flow.ocf_to_net_income < 0.7:
            weaknesses.append("Weak cash conversion")
            risks.append("Earnings quality concerns")
        
        strengths.extend(cash_flow.key_observations)
        risks.extend(cash_flow.red_flags)
        
        # Balance sheet
        strengths.extend(balance_sheet.strengths)
        risks.extend(balance_sheet.risks)
        
        # Capital allocation
        if capital.roic_vs_wacc > 0.05:
            strengths.append("Strong value creation (ROIC > WACC)")
        elif capital.roic_vs_wacc < -0.02:
            weaknesses.append("Value destruction (ROIC < WACC)")
        
        # Manipulation
        if manipulation.overall_risk == "High":
            risks.append("Elevated earnings manipulation risk")
        
        risks.extend(manipulation.detailed_flags)
        
        return strengths, weaknesses, risks, catalysts
    
    def _assess_earnings_power(self, margins: MarginAnalysis,
                               cash_flow: CashFlowAnalysis,
                               capital: CapitalAllocationAnalysis) -> str:
        """Assess overall earnings power"""
        score = 0
        
        if margins.operating_margin > 0.15:
            score += 2
        elif margins.operating_margin > 0.08:
            score += 1
        elif margins.operating_margin < 0:
            score -= 2
        
        if cash_flow.ocf_to_net_income > 1.0:
            score += 1
        elif cash_flow.ocf_to_net_income < 0.7:
            score -= 1
        
        if capital.roic > 0.12:
            score += 1
        elif capital.roic < 0.05:
            score -= 1
        
        if margins.margin_trend in [TrendDirection.IMPROVING, TrendDirection.STRONG_IMPROVING]:
            score += 1
        elif margins.margin_trend in [TrendDirection.DECLINING, TrendDirection.STRONG_DECLINING]:
            score -= 1
        
        if score >= 4:
            return "Strong"
        elif score >= 2:
            return "Stable"
        elif score >= 0:
            return "Moderate"
        else:
            return "Weak"
    
    def _assess_sustainability(self, revenue: RevenueAnalysis,
                               margins: MarginAnalysis,
                               cash_flow: CashFlowAnalysis) -> str:
        """Assess earnings sustainability"""
        score = 0
        
        if revenue.recurring_revenue_pct > 0.7:
            score += 2
        if margins.margin_stability > 0.8:
            score += 1
        if cash_flow.ocf_to_net_income > 1.0:
            score += 1
        if revenue.revenue_trend in [TrendDirection.STABLE, TrendDirection.IMPROVING]:
            score += 1
        
        if score >= 4:
            return "High sustainability - predictable earnings stream"
        elif score >= 2:
            return "Moderate sustainability - some variability expected"
        else:
            return "Low sustainability - earnings may be volatile"
    
    def _generate_summary(self, symbol: str, quality: EarningsQuality,
                         score: float, metrics: Dict[str, float],
                         strengths: List[str], weaknesses: List[str],
                         risks: List[str]) -> str:
        """Generate summary paragraph"""
        quality_desc = {
            EarningsQuality.EXCELLENT: "demonstrates excellent earnings quality",
            EarningsQuality.GOOD: "shows good earnings quality",
            EarningsQuality.FAIR: "exhibits fair earnings quality",
            EarningsQuality.POOR: "shows poor earnings quality",
            EarningsQuality.CONCERNING: "raises concerning earnings quality flags"
        }
        
        summary = f"{symbol} {quality_desc[quality]} with an overall score of {score:.0f}/100. "
        
        if strengths:
            summary += f"Key strengths include {strengths[0].lower()}. "
        
        if weaknesses:
            summary += f"Notable concerns include {weaknesses[0].lower()}. "
        
        summary += f"Operating margin stands at {metrics['operating_margin']:.1%} "
        summary += f"with ROIC of {metrics['roic']:.1%}. "
        
        if metrics['ocf_to_net_income'] > 1.0:
            summary += "Cash conversion is strong with OCF exceeding net income."
        elif metrics['ocf_to_net_income'] < 0.8:
            summary += "Cash conversion warrants monitoring as OCF trails net income."
        
        return summary
    
    # Default/mock data methods
    def _get_mock_financials(self, symbol: str) -> Dict[str, List[Dict]]:
        """Generate mock financial data for demonstration"""
        base_revenue = {"AAPL": 380e9, "MSFT": 210e9, "GOOGL": 280e9}.get(symbol, 50e9)
        
        income_stmt = []
        for i in range(8):
            growth = 1 - i * 0.02  # Slight growth deceleration
            revenue = base_revenue * growth
            income_stmt.append({
                "revenue": revenue,
                "cost_of_revenue": revenue * 0.58,
                "gross_profit": revenue * 0.42,
                "operating_income": revenue * 0.28,
                "ebitda": revenue * 0.35,
                "net_income": revenue * 0.23,
                "sga": revenue * 0.08,
                "rd": revenue * 0.06,
                "depreciation": revenue * 0.04,
                "interest_expense": base_revenue * 0.005,
                "shares_outstanding": 15e9
            })
        
        balance_sheet = []
        for i in range(8):
            assets = base_revenue * 0.9
            balance_sheet.append({
                "total_assets": assets,
                "current_assets": assets * 0.35,
                "cash": assets * 0.15,
                "receivables": base_revenue * 0.08,
                "inventory": base_revenue * 0.02,
                "ppe": assets * 0.25,
                "goodwill": assets * 0.10,
                "intangibles": assets * 0.08,
                "current_liabilities": assets * 0.20,
                "payables": base_revenue * 0.05,
                "total_debt": assets * 0.15,
                "total_equity": assets * 0.55
            })
        
        cash_flow = []
        for i in range(8):
            ni = income_stmt[i]["net_income"]
            cash_flow.append({
                "operating_cash_flow": ni * 1.15,
                "capex": -ni * 0.25,
                "depreciation": income_stmt[i]["depreciation"],
                "working_capital_change": -ni * 0.05,
                "dividends_paid": -ni * 0.25,
                "share_repurchases": -ni * 0.40
            })
        
        return {
            "income_statement": income_stmt,
            "balance_sheet": balance_sheet,
            "cash_flow": cash_flow
        }
    
    def _get_default_revenue_analysis(self) -> RevenueAnalysis:
        return RevenueAnalysis(
            total_revenue=0, yoy_growth=0, qoq_growth=0,
            revenue_trend=TrendDirection.STABLE,
            recurring_revenue_pct=0.5, customer_concentration=0.3,
            geographic_diversification=0.5,
            volume_contribution=0, price_contribution=0, mix_contribution=0,
            backlog_growth=None, deferred_revenue_growth=None,
            customer_retention_rate=None,
            quality_score=50
        )
    
    def _get_default_margin_analysis(self) -> MarginAnalysis:
        return MarginAnalysis(
            gross_margin=0.35, operating_margin=0.10, ebitda_margin=0.15,
            net_margin=0.08, fcf_margin=0.10,
            gross_margin_change=0, operating_margin_change=0, net_margin_change=0,
            cogs_to_revenue=0.65, sga_to_revenue=0.15, rd_to_revenue=0.05,
            margin_stability=0.8, margin_trend=TrendDirection.STABLE,
            operating_leverage=1.0,
            vs_industry_gross=0, vs_industry_operating=0,
            quality_score=50
        )
    
    def _get_default_cash_flow_analysis(self) -> CashFlowAnalysis:
        return CashFlowAnalysis(
            operating_cash_flow=0, free_cash_flow=0, fcf_per_share=0,
            ocf_to_net_income=1.0, fcf_to_net_income=0.8, accrual_ratio=0,
            cash_conversion_cycle=45,
            dso_trend=TrendDirection.STABLE,
            dio_trend=TrendDirection.STABLE,
            dpo_trend=TrendDirection.STABLE,
            capex_to_depreciation=1.0,
            maintenance_capex_estimate=0, growth_capex_estimate=0,
            working_capital_change=0, working_capital_to_revenue=0,
            quality_score=50
        )
    
    def _get_default_balance_sheet_analysis(self) -> BalanceSheetAnalysis:
        return BalanceSheetAnalysis(
            current_ratio=1.5, quick_ratio=1.2, cash_to_current_liabilities=0.5,
            debt_to_equity=0.5, debt_to_ebitda=2.0, net_debt_to_ebitda=1.5,
            interest_coverage=8.0,
            goodwill_to_assets=0.1, intangibles_to_assets=0.1,
            receivables_to_revenue=0.1, inventory_to_cogs=0.1,
            equity_ratio=0.5,
            debt_maturity_profile={},
            weighted_avg_debt_cost=0.05,
            leverage_trend=TrendDirection.STABLE,
            liquidity_trend=TrendDirection.STABLE,
            health_score=50
        )
    
    def _get_default_manipulation_flags(self) -> EarningsManipulationFlags:
        return EarningsManipulationFlags(
            dsri=1.0, gmi=1.0, aqi=1.0, sgi=1.0,
            depi=1.0, sgai=1.0, lvgi=1.0, tata=0.0,
            m_score=-2.5, manipulation_probability=0.1,
            revenue_recognition_concerns=[],
            expense_capitalization_flags=[],
            reserve_manipulation_flags=[],
            related_party_concerns=[],
            auditor_changes=0, restatement_history=0, material_weaknesses=0,
            overall_risk="Low"
        )


# Convenience function
async def analyze_earnings(symbol: str, context: Optional[Dict] = None) -> EarningsAnalysisResult:
    """Quick earnings analysis"""
    agent = EarningsAgent()
    ctx = context or {}
    ctx["symbol"] = symbol
    return await agent.analyze(ctx)


# ============== EarningsAgent ChainExecutor 适配器方法 ==============

async def _earnings_analyze_fundamentals(self, context, inputs: Dict = None) -> EarningsAnalysisResult:
    """
    ChainExecutor 调用的适配方法

    将 ChainExecutor 的调用格式适配到 analyze 方法
    """
    # 从 context 获取 symbol
    symbol = 'UNKNOWN'
    if hasattr(context, 'target'):
        symbol = context.target
    elif isinstance(context, dict):
        symbol = context.get('target', context.get('symbol', 'UNKNOWN'))

    # 从 inputs 中提取财务数据
    inputs = inputs or {}
    financial_data = inputs.get('financial_data', {})
    analyst_data = inputs.get('analyst_data', {})

    # 构建分析上下文
    analysis_context = {
        'symbol': symbol,
        'financials': {
            'income_statement': financial_data.get('income_statement', []),
            'balance_sheet': financial_data.get('balance_sheet', []),
            'cash_flow': financial_data.get('cash_flow', [])
        },
        'fiscal_period': inputs.get('fiscal_period', 'FY2024'),
        'wacc': financial_data.get('wacc', 0.10),
        'market_cap': financial_data.get('market_cap', 0),
        'guidance': analyst_data.get('guidance', {})
    }

    return await self.analyze(analysis_context)


# 为 EarningsAgent 类添加适配方法
EarningsAgent.analyze_fundamentals = _earnings_analyze_fundamentals
