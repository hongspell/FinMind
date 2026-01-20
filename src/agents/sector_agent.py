"""
SectorAgent - 行业分析Agent

分析目标公司在行业中的竞争地位：
- 行业结构分析（波特五力）
- 竞争格局评估
- 市场份额分析
- 同业对比（财务/估值）
- 行业趋势和驱动因素
- 竞争优势评估
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from src.core.base import (
    BaseAgent, AgentOutput, AnalysisContext, ConfidenceScore,
    ReasoningStep, DataSource, Assumption, Uncertainty
)


class CompetitivePosition(str, Enum):
    """竞争地位"""
    DOMINANT = "dominant"           # 主导地位
    STRONG = "strong"               # 强势地位
    FAVORABLE = "favorable"         # 有利地位
    TENABLE = "tenable"            # 可守地位
    WEAK = "weak"                   # 弱势地位


class IndustryLifecycle(str, Enum):
    """行业生命周期"""
    EMERGING = "emerging"           # 新兴期
    GROWTH = "growth"               # 成长期
    MATURE = "mature"               # 成熟期
    DECLINING = "declining"         # 衰退期


class MoatType(str, Enum):
    """护城河类型"""
    NETWORK_EFFECTS = "network_effects"
    SWITCHING_COSTS = "switching_costs"
    INTANGIBLE_ASSETS = "intangible_assets"
    COST_ADVANTAGE = "cost_advantage"
    EFFICIENT_SCALE = "efficient_scale"


@dataclass
class PorterForce:
    """波特五力单项"""
    force_name: str
    intensity: str  # low, medium, high
    score: float    # 1-5
    key_factors: List[str]
    trend: str      # increasing, stable, decreasing


@dataclass
class PorterFiveForces:
    """波特五力分析"""
    supplier_power: PorterForce
    buyer_power: PorterForce
    competitive_rivalry: PorterForce
    threat_of_substitutes: PorterForce
    threat_of_new_entrants: PorterForce
    overall_attractiveness: float  # 1-5
    analysis_summary: str


@dataclass
class CompetitorProfile:
    """竞争对手概况"""
    ticker: str
    name: str
    market_cap: float
    revenue: float
    market_share: float
    growth_rate: float
    profitability: Dict[str, float]  # margins
    valuation: Dict[str, float]      # multiples
    strengths: List[str]
    weaknesses: List[str]


@dataclass
class CompetitiveAdvantage:
    """竞争优势"""
    moat_type: MoatType
    strength: str  # wide, narrow, none
    durability: str  # strong, moderate, weak
    evidence: List[str]
    threats: List[str]


@dataclass
class IndustryMetrics:
    """行业指标"""
    market_size: float
    growth_rate: float
    avg_margin: float
    avg_roe: float
    avg_pe: float
    concentration: float  # HHI或CR4
    capex_intensity: float


class SectorAgent(BaseAgent):
    """
    行业分析Agent
    
    核心功能：
    1. 波特五力分析 - 评估行业竞争强度和吸引力
    2. 竞争格局分析 - 市场份额、竞争态势
    3. 同业对比 - 财务和估值对比
    4. 竞争优势评估 - 护城河分析
    5. 行业趋势 - 驱动因素和变革力量
    6. 相对估值参考 - 同业倍数
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config or {})
        self.agent_name = "SectorAgent"
        
        # 行业分类映射
        self.sector_taxonomy = {
            "AAPL": {"sector": "Technology", "industry": "Consumer Electronics", 
                     "peers": ["MSFT", "GOOGL", "SAMSUNG"]},
            "MSFT": {"sector": "Technology", "industry": "Software", 
                     "peers": ["AAPL", "GOOGL", "ORCL", "CRM"]},
            "GOOGL": {"sector": "Technology", "industry": "Internet Services", 
                      "peers": ["META", "MSFT", "AMZN"]},
            "AMZN": {"sector": "Consumer Discretionary", "industry": "E-Commerce", 
                     "peers": ["WMT", "TGT", "BABA", "JD"]},
            "TSLA": {"sector": "Consumer Discretionary", "industry": "Auto Manufacturers", 
                     "peers": ["GM", "F", "RIVN", "NIO"]},
            "JPM": {"sector": "Financials", "industry": "Banks", 
                    "peers": ["BAC", "WFC", "C", "GS"]},
            "JNJ": {"sector": "Healthcare", "industry": "Pharmaceuticals", 
                    "peers": ["PFE", "MRK", "ABBV", "LLY"]},
            "XOM": {"sector": "Energy", "industry": "Oil & Gas", 
                    "peers": ["CVX", "COP", "EOG", "SLB"]},
        }
    
    async def analyze(self, context: AnalysisContext) -> AgentOutput:
        """执行行业分析"""
        reasoning_chain = []
        data_sources = []
        assumptions = []
        uncertainties = []
        warnings = []
        
        target = context.target
        
        # Step 1: 识别行业和竞争对手
        step1 = ReasoningStep(
            step_number=1,
            description="识别目标所属行业和主要竞争对手",
            inputs={"target": target},
            outputs={},
            confidence=0.0
        )
        
        sector_info = self._identify_sector(target)
        step1.outputs = sector_info
        step1.confidence = 0.9
        reasoning_chain.append(step1)
        
        # Step 2: 波特五力分析
        step2 = ReasoningStep(
            step_number=2,
            description="执行波特五力分析评估行业吸引力",
            inputs={"industry": sector_info["industry"]},
            outputs={},
            confidence=0.0
        )
        
        porter_analysis = self._analyze_porter_forces(sector_info["industry"])
        step2.outputs = {"overall_attractiveness": porter_analysis.overall_attractiveness}
        step2.confidence = 0.7
        reasoning_chain.append(step2)
        
        assumptions.append(Assumption(
            category="porter_analysis",
            description="波特五力基于当前行业结构，未考虑颠覆性变化",
            impact="medium",
            sensitivity=0.25
        ))
        
        # Step 3: 获取竞争对手数据
        step3 = ReasoningStep(
            step_number=3,
            description="获取竞争对手财务和估值数据",
            inputs={"peers": sector_info["peers"]},
            outputs={},
            confidence=0.0
        )
        
        competitors = await self._fetch_competitor_data(
            target, 
            sector_info["peers"]
        )
        step3.outputs = {"competitors_analyzed": len(competitors)}
        step3.confidence = 0.85
        reasoning_chain.append(step3)
        
        data_sources.append(DataSource(
            name="Company Financials",
            type="fundamental_data",
            url="",
            accessed_at=datetime.utcnow(),
            quality_score=0.85
        ))
        
        # Step 4: 竞争地位评估
        step4 = ReasoningStep(
            step_number=4,
            description="评估目标公司竞争地位",
            inputs={"target": target},
            outputs={},
            confidence=0.0
        )
        
        target_profile = self._get_target_profile(target, competitors)
        competitive_position = self._assess_competitive_position(
            target_profile, 
            competitors
        )
        step4.outputs = {"position": competitive_position["position"].value}
        step4.confidence = 0.75
        reasoning_chain.append(step4)
        
        # Step 5: 竞争优势评估
        step5 = ReasoningStep(
            step_number=5,
            description="评估竞争优势和护城河",
            inputs={"target": target},
            outputs={},
            confidence=0.0
        )
        
        moat_analysis = self._analyze_competitive_advantages(
            target,
            sector_info["industry"]
        )
        step5.outputs = {
            "moat_types": [m.moat_type.value for m in moat_analysis],
            "overall_moat": self._determine_overall_moat(moat_analysis)
        }
        step5.confidence = 0.7
        reasoning_chain.append(step5)
        
        uncertainties.append(Uncertainty(
            source="moat_durability",
            description="护城河持久性受技术变革和竞争动态影响",
            impact="medium",
            range_low=-0.2,
            range_high=0.2
        ))
        
        # Step 6: 行业趋势分析
        step6 = ReasoningStep(
            step_number=6,
            description="分析行业趋势和驱动因素",
            inputs={"industry": sector_info["industry"]},
            outputs={},
            confidence=0.0
        )
        
        industry_trends = self._analyze_industry_trends(sector_info["industry"])
        step6.outputs = {"trend_count": len(industry_trends["drivers"])}
        step6.confidence = 0.7
        reasoning_chain.append(step6)
        
        # Step 7: 同业估值对比
        step7 = ReasoningStep(
            step_number=7,
            description="同业估值倍数对比",
            inputs={"target": target, "peers": sector_info["peers"]},
            outputs={},
            confidence=0.0
        )
        
        valuation_comparison = self._compare_valuations(
            target_profile, 
            competitors
        )
        step7.outputs = valuation_comparison["summary"]
        step7.confidence = 0.8
        reasoning_chain.append(step7)
        
        # Step 8: LLM综合分析
        step8 = ReasoningStep(
            step_number=8,
            description="使用LLM综合行业竞争分析",
            inputs={"target": target},
            outputs={},
            confidence=0.0
        )
        
        llm_analysis = await self._llm_sector_synthesis(
            target,
            sector_info,
            porter_analysis,
            competitive_position,
            moat_analysis,
            industry_trends
        )
        step8.outputs = {"analysis": "completed"}
        step8.confidence = 0.7
        reasoning_chain.append(step8)
        
        # 构建最终结果
        result = {
            "sector_overview": {
                "sector": sector_info["sector"],
                "industry": sector_info["industry"],
                "lifecycle": self._determine_lifecycle(sector_info["industry"]).value,
                "market_size": self._estimate_market_size(sector_info["industry"]),
                "growth_outlook": industry_trends["growth_outlook"]
            },
            "porter_five_forces": {
                "supplier_power": {
                    "intensity": porter_analysis.supplier_power.intensity,
                    "score": porter_analysis.supplier_power.score,
                    "factors": porter_analysis.supplier_power.key_factors
                },
                "buyer_power": {
                    "intensity": porter_analysis.buyer_power.intensity,
                    "score": porter_analysis.buyer_power.score,
                    "factors": porter_analysis.buyer_power.key_factors
                },
                "competitive_rivalry": {
                    "intensity": porter_analysis.competitive_rivalry.intensity,
                    "score": porter_analysis.competitive_rivalry.score,
                    "factors": porter_analysis.competitive_rivalry.key_factors
                },
                "threat_of_substitutes": {
                    "intensity": porter_analysis.threat_of_substitutes.intensity,
                    "score": porter_analysis.threat_of_substitutes.score,
                    "factors": porter_analysis.threat_of_substitutes.key_factors
                },
                "threat_of_new_entrants": {
                    "intensity": porter_analysis.threat_of_new_entrants.intensity,
                    "score": porter_analysis.threat_of_new_entrants.score,
                    "factors": porter_analysis.threat_of_new_entrants.key_factors
                },
                "overall_attractiveness": porter_analysis.overall_attractiveness,
                "summary": porter_analysis.analysis_summary
            },
            "competitive_position": {
                "position": competitive_position["position"].value,
                "market_share": competitive_position["market_share"],
                "relative_strength": competitive_position["relative_strength"],
                "key_advantages": competitive_position["advantages"],
                "key_disadvantages": competitive_position["disadvantages"]
            },
            "moat_analysis": {
                "overall_moat": self._determine_overall_moat(moat_analysis),
                "moat_types": [
                    {
                        "type": m.moat_type.value,
                        "strength": m.strength,
                        "durability": m.durability,
                        "evidence": m.evidence,
                        "threats": m.threats
                    }
                    for m in moat_analysis
                ]
            },
            "peer_comparison": {
                "peers": [
                    {
                        "ticker": c.ticker,
                        "name": c.name,
                        "market_cap_bn": c.market_cap / 1e9,
                        "market_share": c.market_share,
                        "revenue_growth": c.growth_rate,
                        "gross_margin": c.profitability.get("gross_margin", 0),
                        "operating_margin": c.profitability.get("operating_margin", 0),
                        "pe_ratio": c.valuation.get("pe", 0),
                        "ev_sales": c.valuation.get("ev_sales", 0)
                    }
                    for c in competitors
                ],
                "relative_valuation": valuation_comparison
            },
            "industry_trends": {
                "drivers": industry_trends["drivers"],
                "headwinds": industry_trends["headwinds"],
                "disruption_risks": industry_trends["disruption_risks"],
                "growth_outlook": industry_trends["growth_outlook"]
            },
            "investment_implications": llm_analysis.get("implications", []),
            "key_metrics_to_watch": llm_analysis.get("metrics_to_watch", [])
        }
        
        # 添加警告
        if porter_analysis.competitive_rivalry.intensity == "high":
            warnings.append("行业竞争激烈，可能压缩利润率")
        
        if self._determine_overall_moat(moat_analysis) == "none":
            warnings.append("未识别到明显护城河，竞争优势可能不持久")
        
        if competitive_position["position"] in [CompetitivePosition.TENABLE, 
                                                  CompetitivePosition.WEAK]:
            warnings.append("竞争地位较弱，需关注市场份额变化")
        
        # 计算置信度
        confidence = self._calculate_confidence(
            data_quality=0.8,
            completeness=0.75,
            reasoning_quality=0.7,
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
    
    def _identify_sector(self, target: str) -> Dict[str, Any]:
        """识别行业和竞争对手"""
        target_upper = target.upper()
        if target_upper in self.sector_taxonomy:
            return self.sector_taxonomy[target_upper]
        
        # 默认返回科技行业
        return {
            "sector": "Technology",
            "industry": "Software & Services",
            "peers": ["MSFT", "GOOGL", "CRM", "ORCL"]
        }
    
    def _analyze_porter_forces(self, industry: str) -> PorterFiveForces:
        """波特五力分析"""
        # 行业特定的五力评估（简化示例）
        industry_forces = {
            "Consumer Electronics": {
                "supplier": (2.5, "medium", ["少数关键组件供应商", "供应链集中"]),
                "buyer": (3.5, "medium", ["消费者价格敏感", "产品差异化"]),
                "rivalry": (4.0, "high", ["激烈价格竞争", "创新压力大"]),
                "substitutes": (2.0, "low", ["产品功能整合", "生态系统锁定"]),
                "entrants": (2.5, "medium", ["高研发投入", "品牌壁垒"])
            },
            "Software": {
                "supplier": (2.0, "low", ["人才稀缺", "云服务依赖"]),
                "buyer": (2.5, "medium", ["企业客户粘性", "切换成本"]),
                "rivalry": (3.5, "medium", ["快速创新", "市场细分"]),
                "substitutes": (3.0, "medium", ["开源替代", "云原生方案"]),
                "entrants": (3.0, "medium", ["低资本需求", "技术壁垒"])
            },
            "default": {
                "supplier": (2.5, "medium", ["供应商分散"]),
                "buyer": (3.0, "medium", ["买方议价力一般"]),
                "rivalry": (3.5, "medium", ["行业竞争激烈"]),
                "substitutes": (2.5, "medium", ["替代品威胁中等"]),
                "entrants": (3.0, "medium", ["进入壁垒中等"])
            }
        }
        
        forces = industry_forces.get(industry, industry_forces["default"])
        
        supplier = PorterForce(
            force_name="供应商议价能力",
            intensity=forces["supplier"][1],
            score=forces["supplier"][0],
            key_factors=forces["supplier"][2],
            trend="stable"
        )
        
        buyer = PorterForce(
            force_name="买方议价能力",
            intensity=forces["buyer"][1],
            score=forces["buyer"][0],
            key_factors=forces["buyer"][2],
            trend="stable"
        )
        
        rivalry = PorterForce(
            force_name="行业竞争强度",
            intensity=forces["rivalry"][1],
            score=forces["rivalry"][0],
            key_factors=forces["rivalry"][2],
            trend="increasing"
        )
        
        substitutes = PorterForce(
            force_name="替代品威胁",
            intensity=forces["substitutes"][1],
            score=forces["substitutes"][0],
            key_factors=forces["substitutes"][2],
            trend="stable"
        )
        
        entrants = PorterForce(
            force_name="新进入者威胁",
            intensity=forces["entrants"][1],
            score=forces["entrants"][0],
            key_factors=forces["entrants"][2],
            trend="stable"
        )
        
        # 计算整体吸引力（分数越低越有吸引力）
        avg_score = (supplier.score + buyer.score + rivalry.score + 
                     substitutes.score + entrants.score) / 5
        attractiveness = 5 - avg_score + 1  # 转换为吸引力评分
        
        return PorterFiveForces(
            supplier_power=supplier,
            buyer_power=buyer,
            competitive_rivalry=rivalry,
            threat_of_substitutes=substitutes,
            threat_of_new_entrants=entrants,
            overall_attractiveness=round(attractiveness, 1),
            analysis_summary=f"行业整体吸引力{attractiveness:.1f}/5，" +
                           f"主要挑战来自{'竞争激烈' if rivalry.score > 3.5 else '买方压力'}"
        )
    
    async def _fetch_competitor_data(
        self, 
        target: str, 
        peers: List[str]
    ) -> List[CompetitorProfile]:
        """获取竞争对手数据（模拟）"""
        # 模拟数据
        mock_data = {
            "AAPL": CompetitorProfile(
                ticker="AAPL", name="Apple Inc.",
                market_cap=2.8e12, revenue=385e9, market_share=0.28,
                growth_rate=0.03,
                profitability={"gross_margin": 0.44, "operating_margin": 0.30, "net_margin": 0.25},
                valuation={"pe": 28, "ev_sales": 7.2, "ev_ebitda": 21},
                strengths=["品牌价值", "生态系统", "服务增长"],
                weaknesses=["iPhone依赖", "中国市场风险"]
            ),
            "MSFT": CompetitorProfile(
                ticker="MSFT", name="Microsoft Corp.",
                market_cap=2.9e12, revenue=212e9, market_share=0.22,
                growth_rate=0.12,
                profitability={"gross_margin": 0.69, "operating_margin": 0.42, "net_margin": 0.36},
                valuation={"pe": 34, "ev_sales": 12, "ev_ebitda": 24},
                strengths=["云增长", "AI布局", "企业客户"],
                weaknesses=["消费者业务", "反垄断风险"]
            ),
            "GOOGL": CompetitorProfile(
                ticker="GOOGL", name="Alphabet Inc.",
                market_cap=1.7e12, revenue=307e9, market_share=0.18,
                growth_rate=0.08,
                profitability={"gross_margin": 0.56, "operating_margin": 0.27, "net_margin": 0.22},
                valuation={"pe": 24, "ev_sales": 5.5, "ev_ebitda": 14},
                strengths=["搜索主导", "YouTube", "AI能力"],
                weaknesses=["广告依赖", "监管压力"]
            ),
            "AMZN": CompetitorProfile(
                ticker="AMZN", name="Amazon.com Inc.",
                market_cap=1.5e12, revenue=575e9, market_share=0.38,
                growth_rate=0.10,
                profitability={"gross_margin": 0.45, "operating_margin": 0.06, "net_margin": 0.04},
                valuation={"pe": 65, "ev_sales": 2.8, "ev_ebitda": 18},
                strengths=["规模优势", "AWS", "Prime会员"],
                weaknesses=["低利润率", "劳动力成本"]
            ),
            "META": CompetitorProfile(
                ticker="META", name="Meta Platforms Inc.",
                market_cap=1.2e12, revenue=135e9, market_share=0.12,
                growth_rate=0.15,
                profitability={"gross_margin": 0.81, "operating_margin": 0.35, "net_margin": 0.28},
                valuation={"pe": 26, "ev_sales": 8.5, "ev_ebitda": 15},
                strengths=["用户规模", "广告技术", "Reels增长"],
                weaknesses=["元宇宙投入", "年轻用户流失"]
            )
        }
        
        # 返回竞争对手数据
        competitors = []
        for peer in peers:
            if peer.upper() in mock_data:
                competitors.append(mock_data[peer.upper()])
        
        return competitors
    
    def _get_target_profile(
        self, 
        target: str, 
        competitors: List[CompetitorProfile]
    ) -> CompetitorProfile:
        """获取目标公司概况"""
        # 查找目标公司
        for c in competitors:
            if c.ticker.upper() == target.upper():
                return c
        
        # 默认概况
        return CompetitorProfile(
            ticker=target.upper(),
            name=f"{target.upper()} Inc.",
            market_cap=500e9,
            revenue=100e9,
            market_share=0.15,
            growth_rate=0.08,
            profitability={"gross_margin": 0.45, "operating_margin": 0.20, "net_margin": 0.15},
            valuation={"pe": 25, "ev_sales": 5, "ev_ebitda": 15},
            strengths=["待分析"],
            weaknesses=["待分析"]
        )
    
    def _assess_competitive_position(
        self, 
        target: CompetitorProfile,
        competitors: List[CompetitorProfile]
    ) -> Dict[str, Any]:
        """评估竞争地位"""
        # 计算相对指标
        all_companies = [target] + [c for c in competitors if c.ticker != target.ticker]
        
        # 市场份额排名
        sorted_by_share = sorted(all_companies, key=lambda x: x.market_share, reverse=True)
        share_rank = next(i for i, c in enumerate(sorted_by_share) if c.ticker == target.ticker) + 1
        
        # 市值排名
        sorted_by_cap = sorted(all_companies, key=lambda x: x.market_cap, reverse=True)
        cap_rank = next(i for i, c in enumerate(sorted_by_cap) if c.ticker == target.ticker) + 1
        
        # 确定竞争地位
        total = len(all_companies)
        if share_rank == 1 and target.market_share > 0.3:
            position = CompetitivePosition.DOMINANT
        elif share_rank <= 2:
            position = CompetitivePosition.STRONG
        elif share_rank <= total * 0.5:
            position = CompetitivePosition.FAVORABLE
        elif share_rank <= total * 0.75:
            position = CompetitivePosition.TENABLE
        else:
            position = CompetitivePosition.WEAK
        
        # 相对优势分析
        avg_margin = sum(c.profitability.get("operating_margin", 0) for c in all_companies) / len(all_companies)
        relative_strength = (target.profitability.get("operating_margin", 0) - avg_margin) / avg_margin if avg_margin > 0 else 0
        
        return {
            "position": position,
            "market_share": target.market_share,
            "share_rank": share_rank,
            "cap_rank": cap_rank,
            "relative_strength": round(relative_strength, 2),
            "advantages": target.strengths,
            "disadvantages": target.weaknesses
        }
    
    def _analyze_competitive_advantages(
        self, 
        target: str,
        industry: str
    ) -> List[CompetitiveAdvantage]:
        """分析竞争优势"""
        # 公司特定的护城河分析
        moat_analysis = {
            "AAPL": [
                CompetitiveAdvantage(
                    moat_type=MoatType.SWITCHING_COSTS,
                    strength="wide",
                    durability="strong",
                    evidence=["iOS生态系统锁定", "iCloud数据迁移成本", "配件和服务捆绑"],
                    threats=["跨平台应用增多", "监管要求互操作性"]
                ),
                CompetitiveAdvantage(
                    moat_type=MoatType.INTANGIBLE_ASSETS,
                    strength="wide",
                    durability="strong",
                    evidence=["全球最具价值品牌之一", "设计专利组合", "零售体验"],
                    threats=["品牌老化风险", "中国市场民族主义"]
                ),
                CompetitiveAdvantage(
                    moat_type=MoatType.NETWORK_EFFECTS,
                    strength="narrow",
                    durability="moderate",
                    evidence=["App Store开发者生态", "iMessage网络效应"],
                    threats=["监管开放侧载", "跨平台通信工具"]
                )
            ],
            "MSFT": [
                CompetitiveAdvantage(
                    moat_type=MoatType.SWITCHING_COSTS,
                    strength="wide",
                    durability="strong",
                    evidence=["企业IT基础设施依赖", "Office套件标准化", "Azure集成"],
                    threats=["云原生替代方案", "开源竞争"]
                ),
                CompetitiveAdvantage(
                    moat_type=MoatType.NETWORK_EFFECTS,
                    strength="narrow",
                    durability="moderate",
                    evidence=["LinkedIn职业网络", "Teams协作平台", "GitHub开发者社区"],
                    threats=["竞争产品增长", "用户分散化"]
                )
            ],
            "GOOGL": [
                CompetitiveAdvantage(
                    moat_type=MoatType.INTANGIBLE_ASSETS,
                    strength="wide",
                    durability="strong",
                    evidence=["搜索算法领先", "AI/ML技术积累", "海量训练数据"],
                    threats=["AI颠覆搜索入口", "数据隐私限制"]
                ),
                CompetitiveAdvantage(
                    moat_type=MoatType.NETWORK_EFFECTS,
                    strength="wide",
                    durability="moderate",
                    evidence=["YouTube创作者生态", "Android设备网络", "广告主网络"],
                    threats=["TikTok竞争", "反垄断拆分"]
                )
            ]
        }
        
        return moat_analysis.get(target.upper(), [
            CompetitiveAdvantage(
                moat_type=MoatType.SWITCHING_COSTS,
                strength="narrow",
                durability="moderate",
                evidence=["客户关系", "产品集成"],
                threats=["竞争加剧", "技术变革"]
            )
        ])
    
    def _determine_overall_moat(self, moat_analysis: List[CompetitiveAdvantage]) -> str:
        """确定整体护城河评级"""
        if not moat_analysis:
            return "none"
        
        wide_count = sum(1 for m in moat_analysis if m.strength == "wide")
        narrow_count = sum(1 for m in moat_analysis if m.strength == "narrow")
        
        if wide_count >= 2:
            return "wide"
        elif wide_count >= 1 or narrow_count >= 2:
            return "narrow"
        else:
            return "none"
    
    def _analyze_industry_trends(self, industry: str) -> Dict[str, Any]:
        """分析行业趋势"""
        trends = {
            "Consumer Electronics": {
                "drivers": ["AI集成", "可穿戴设备增长", "服务收入扩展", "AR/VR发展"],
                "headwinds": ["智能手机饱和", "经济周期敏感", "供应链风险"],
                "disruption_risks": ["折叠屏技术", "AI助手替代App", "隐私法规"],
                "growth_outlook": "moderate"
            },
            "Software": {
                "drivers": ["云迁移加速", "AI/ML采用", "数字化转型", "远程办公常态化"],
                "headwinds": ["客户预算收紧", "竞争加剧", "技术债务"],
                "disruption_risks": ["AI代码生成", "低代码平台", "开源替代"],
                "growth_outlook": "strong"
            },
            "Internet Services": {
                "drivers": ["数字广告增长", "视频流媒体", "电商渗透", "AI应用"],
                "headwinds": ["隐私法规", "反垄断审查", "广告周期性"],
                "disruption_risks": ["AI改变搜索", "Web3去中心化", "新兴平台"],
                "growth_outlook": "moderate"
            },
            "default": {
                "drivers": ["数字化转型", "效率提升", "新市场开拓"],
                "headwinds": ["经济不确定性", "竞争压力", "监管风险"],
                "disruption_risks": ["技术变革", "商业模式创新"],
                "growth_outlook": "moderate"
            }
        }
        
        return trends.get(industry, trends["default"])
    
    def _compare_valuations(
        self, 
        target: CompetitorProfile,
        competitors: List[CompetitorProfile]
    ) -> Dict[str, Any]:
        """同业估值对比"""
        all_companies = [target] + [c for c in competitors if c.ticker != target.ticker]
        
        # 计算行业平均估值
        avg_pe = sum(c.valuation.get("pe", 0) for c in all_companies) / len(all_companies)
        avg_ev_sales = sum(c.valuation.get("ev_sales", 0) for c in all_companies) / len(all_companies)
        avg_ev_ebitda = sum(c.valuation.get("ev_ebitda", 0) for c in all_companies) / len(all_companies)
        
        target_pe = target.valuation.get("pe", 0)
        target_ev_sales = target.valuation.get("ev_sales", 0)
        target_ev_ebitda = target.valuation.get("ev_ebitda", 0)
        
        # 计算溢价/折价
        pe_premium = (target_pe - avg_pe) / avg_pe if avg_pe > 0 else 0
        ev_sales_premium = (target_ev_sales - avg_ev_sales) / avg_ev_sales if avg_ev_sales > 0 else 0
        
        # 估值判断
        if pe_premium > 0.2:
            valuation_status = "premium"
            valuation_comment = "估值高于同业平均，需要更高增长支撑"
        elif pe_premium < -0.2:
            valuation_status = "discount"
            valuation_comment = "估值低于同业平均，可能存在价值机会"
        else:
            valuation_status = "inline"
            valuation_comment = "估值与同业基本一致"
        
        return {
            "summary": {
                "valuation_status": valuation_status,
                "pe_vs_peers": round(pe_premium * 100, 1),
                "ev_sales_vs_peers": round(ev_sales_premium * 100, 1)
            },
            "target_multiples": {
                "pe": target_pe,
                "ev_sales": target_ev_sales,
                "ev_ebitda": target_ev_ebitda
            },
            "peer_averages": {
                "pe": round(avg_pe, 1),
                "ev_sales": round(avg_ev_sales, 1),
                "ev_ebitda": round(avg_ev_ebitda, 1)
            },
            "valuation_comment": valuation_comment
        }
    
    def _determine_lifecycle(self, industry: str) -> IndustryLifecycle:
        """确定行业生命周期"""
        lifecycle_map = {
            "Consumer Electronics": IndustryLifecycle.MATURE,
            "Software": IndustryLifecycle.GROWTH,
            "Internet Services": IndustryLifecycle.MATURE,
            "E-Commerce": IndustryLifecycle.GROWTH,
            "Electric Vehicles": IndustryLifecycle.GROWTH,
            "Banks": IndustryLifecycle.MATURE,
            "Pharmaceuticals": IndustryLifecycle.MATURE,
            "Oil & Gas": IndustryLifecycle.DECLINING
        }
        return lifecycle_map.get(industry, IndustryLifecycle.MATURE)
    
    def _estimate_market_size(self, industry: str) -> Dict[str, Any]:
        """估算市场规模"""
        market_sizes = {
            "Consumer Electronics": {"size_bn": 1100, "growth_rate": 0.05},
            "Software": {"size_bn": 650, "growth_rate": 0.12},
            "Internet Services": {"size_bn": 800, "growth_rate": 0.08},
            "E-Commerce": {"size_bn": 5500, "growth_rate": 0.10},
            "Banks": {"size_bn": 8000, "growth_rate": 0.03},
            "Pharmaceuticals": {"size_bn": 1500, "growth_rate": 0.06},
        }
        return market_sizes.get(industry, {"size_bn": 500, "growth_rate": 0.05})
    
    async def _llm_sector_synthesis(
        self,
        target: str,
        sector_info: Dict,
        porter: PorterFiveForces,
        position: Dict,
        moat: List[CompetitiveAdvantage],
        trends: Dict
    ) -> Dict[str, Any]:
        """LLM综合分析"""
        # 模拟LLM响应
        moat_str = self._determine_overall_moat(moat)
        return {
            "implications": [
                f"{target}在{sector_info['industry']}行业处于{position['position'].value}地位",
                f"护城河评级为{moat_str}，" + 
                ("竞争优势较为持久" if moat_str == "wide" else "需关注竞争动态"),
                f"行业增长前景{trends['growth_outlook']}",
                f"主要驱动因素：{', '.join(trends['drivers'][:2])}"
            ],
            "metrics_to_watch": [
                "市场份额变化 - 竞争地位指标",
                "毛利率趋势 - 定价能力验证",
                "客户留存率 - 切换成本验证",
                "研发投入占比 - 创新能力指标",
                "新产品收入占比 - 增长动能"
            ]
        }
    
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
            "completeness": 0.20,
            "reasoning": 0.20,
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
        overall = max(0.1, min(0.9, overall))
        
        return ConfidenceScore(
            overall=overall,
            factors=factors,
            weights=weights,
            explanation=f"行业分析置信度{overall:.1%}，受限于定性判断的主观性"
        )

    # ============== ChainExecutor 适配器方法 ==============

    async def compare_with_peers(self, context, inputs: Dict = None) -> AgentOutput:
        """
        ChainExecutor 调用的适配方法 - 同业对比

        将 ChainExecutor 的调用格式适配到 analyze 方法
        """
        # 如果 context 已经是 AnalysisContext，直接使用
        if hasattr(context, 'target'):
            return await self.analyze(context)

        # 否则从 inputs 或 context dict 中构建
        from src.core.base import AnalysisContext
        from datetime import datetime

        target = 'UNKNOWN'
        if hasattr(context, 'target'):
            target = context.target
        elif isinstance(context, dict):
            target = context.get('target', 'UNKNOWN')

        analysis_context = AnalysisContext(
            target=target,
            analysis_date=datetime.now()
        )
        return await self.analyze(analysis_context)

    async def analyze_competitive_position(self, context, inputs: Dict = None) -> AgentOutput:
        """
        ChainExecutor 调用的适配方法 - 竞争地位分析

        将 ChainExecutor 的调用格式适配到 analyze 方法
        主要关注竞争地位和护城河分析
        """
        # 如果 context 已经是 AnalysisContext，直接使用
        if hasattr(context, 'target'):
            return await self.analyze(context)

        # 否则从 inputs 或 context dict 中构建
        from src.core.base import AnalysisContext
        from datetime import datetime

        target = 'UNKNOWN'
        if hasattr(context, 'target'):
            target = context.target
        elif isinstance(context, dict):
            target = context.get('target', 'UNKNOWN')

        analysis_context = AnalysisContext(
            target=target,
            analysis_date=datetime.now()
        )
        return await self.analyze(analysis_context)
