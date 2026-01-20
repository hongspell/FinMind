"""
FinMind - Valuation Agent
估值分析 Agent 完整实现
"""
from typing import Any, Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass
import json

from src.core.base import (
    BaseAgent,
    AnalysisType,
    AnalysisContext,
    AgentOutput,
    DataSource,
    ReasoningStep,
    Uncertainty,
    LLMGateway,
    ConfidenceScorer
)


@dataclass
class ValuationResult:
    """估值结果"""
    method: str
    fair_value: float
    fair_value_range: tuple  # (low, high)
    upside_downside: float   # 相对当前价格
    key_assumptions: List[str]
    sensitivity: Dict[str, Any]


class ValuationAgent(BaseAgent):
    """
    估值分析 Agent
    
    支持的估值方法:
    - DCF (Discounted Cash Flow)
    - Comparable Companies
    - Historical Multiples
    - Sum of Parts
    - Dividend Discount Model
    """
    
    @property
    def analysis_type(self) -> AnalysisType:
        return AnalysisType.VALUATION

    async def comprehensive_valuation(self, context, inputs: Dict) -> AgentOutput:
        """
        ChainExecutor 调用的适配方法

        将 ChainExecutor 的调用格式适配到 analyze 方法
        """
        # 从 inputs 中提取数据
        market_data = inputs.get('market_data', {})
        financial_data = inputs.get('financial_data', {})
        macro_view = inputs.get('macro_view')

        # 构建 inputs dict
        analysis_inputs = {
            'financial_data': financial_data,
            'market_data': market_data,
            'macro_view': macro_view,
            'custom_assumptions': inputs.get('custom_assumptions', {})
        }

        return await self.analyze(context, analysis_inputs)

    async def analyze(
        self,
        context: AnalysisContext,
        inputs: Dict[str, Any]
    ) -> AgentOutput:
        """
        执行估值分析
        
        Args:
            context: 分析上下文
            inputs: 包含以下数据
                - financial_data: 财务数据
                - market_data: 市场数据
                - macro_view: 宏观分析结果（可选）
                - custom_assumptions: 自定义假设（可选）
        """
        reasoning_steps = []
        data_sources = []
        uncertainties = []
        warnings = []
        
        # ========== Step 1: 数据验证与准备 ==========
        financial_data = inputs.get('financial_data', {})
        market_data = inputs.get('market_data', {})
        macro_view = inputs.get('macro_view')
        custom_assumptions = inputs.get('custom_assumptions', {})
        
        # 记录数据源
        if financial_data:
            data_sources.append(DataSource(
                name="financial_statements",
                type="api",
                timestamp=datetime.now(),
                quality_score=self._assess_data_quality(financial_data)
            ))
        
        if market_data:
            data_sources.append(DataSource(
                name="market_data",
                type="api",
                timestamp=datetime.now(),
                quality_score=0.9  # 市场数据通常质量较高
            ))
        
        # 数据完整性检查
        data_completeness = self._check_data_completeness(financial_data, market_data)
        if data_completeness < 0.7:
            warnings.append("财务数据不完整，估值可能存在较大误差")
            uncertainties.append(Uncertainty(
                factor="data_incompleteness",
                description="关键财务数据缺失",
                impact="high",
                mitigation="建议补充近3年完整财报数据"
            ))
        
        self._add_reasoning_step(
            reasoning_steps,
            "数据验证与准备",
            ["financial_data", "market_data"],
            f"数据完整度: {data_completeness:.0%}",
            confidence=data_completeness
        )
        
        # ========== Step 2: 确定适用的估值方法 ==========
        applicable_methods = await self._determine_valuation_methods(
            context, financial_data, market_data
        )
        
        self._add_reasoning_step(
            reasoning_steps,
            "选择估值方法",
            ["company_type", "data_availability"],
            f"选定方法: {', '.join(applicable_methods)}",
            confidence=0.8
        )
        
        # ========== Step 3: 执行各种估值方法 ==========
        valuation_results: List[ValuationResult] = []
        
        current_price = market_data.get('current_price', 0)
        
        # 3.1 DCF 估值
        if 'dcf' in applicable_methods:
            dcf_result = await self._dcf_valuation(
                context, financial_data, market_data, macro_view, custom_assumptions
            )
            if dcf_result:
                valuation_results.append(dcf_result)
                self._add_reasoning_step(
                    reasoning_steps,
                    "DCF 估值分析",
                    ["free_cash_flow", "growth_rate", "discount_rate"],
                    f"DCF 公允价值: ${dcf_result.fair_value:.2f}",
                    confidence=0.7,
                    supporting_data=dcf_result.sensitivity
                )
        
        # 3.2 可比公司估值
        if 'comparables' in applicable_methods:
            comp_result = await self._comparable_valuation(
                context, financial_data, market_data
            )
            if comp_result:
                valuation_results.append(comp_result)
                self._add_reasoning_step(
                    reasoning_steps,
                    "可比公司估值",
                    ["peer_multiples", "company_financials"],
                    f"可比公司法公允价值: ${comp_result.fair_value:.2f}",
                    confidence=0.75
                )
        
        # 3.3 历史估值
        if 'historical' in applicable_methods:
            hist_result = await self._historical_valuation(
                context, financial_data, market_data
            )
            if hist_result:
                valuation_results.append(hist_result)
                self._add_reasoning_step(
                    reasoning_steps,
                    "历史估值分析",
                    ["historical_multiples", "current_fundamentals"],
                    f"历史法公允价值: ${hist_result.fair_value:.2f}",
                    confidence=0.65
                )
        
        # ========== Step 4: 综合各方法结果 ==========
        if not valuation_results:
            warnings.append("无法执行任何估值方法，数据可能严重不足")
            return self._create_failed_output(context, warnings)
        
        synthesis = await self._synthesize_valuations(
            context, valuation_results, current_price
        )
        
        self._add_reasoning_step(
            reasoning_steps,
            "估值综合",
            [r.method for r in valuation_results],
            f"综合公允价值区间: ${synthesis['fair_value_range'][0]:.2f} - ${synthesis['fair_value_range'][1]:.2f}",
            confidence=synthesis['synthesis_confidence']
        )
        
        # ========== Step 5: 使用 LLM 生成深度分析 ==========
        llm_analysis = await self._generate_llm_analysis(
            context, valuation_results, synthesis, market_data, financial_data
        )
        
        data_sources.append(DataSource(
            name="llm_analysis",
            type="llm_inference",
            timestamp=datetime.now(),
            quality_score=0.7
        ))
        
        # ========== Step 6: 风险与不确定性评估 ==========
        uncertainties.extend(self._assess_valuation_uncertainties(
            synthesis, valuation_results, financial_data
        ))
        
        # ========== 构建输出 ==========
        key_assumptions = []
        for result in valuation_results:
            key_assumptions.extend(result.key_assumptions)
        key_assumptions = list(set(key_assumptions))  # 去重
        
        # 计算置信度输入
        avg_data_quality = sum(ds.quality_score for ds in data_sources) / len(data_sources)
        
        return self._create_output(
            result={
                'fair_value_range': synthesis['fair_value_range'],
                'fair_value_midpoint': synthesis['fair_value_midpoint'],
                'current_price': current_price,
                'upside_potential': synthesis['upside_potential'],
                'valuation_stance': synthesis['stance'],  # undervalued/fairly_valued/overvalued
                'conviction': synthesis['conviction'],     # low/medium/high
                'method_results': [
                    {
                        'method': r.method,
                        'fair_value': r.fair_value,
                        'range': r.fair_value_range,
                        'weight': self._get_method_weight(r.method)
                    }
                    for r in valuation_results
                ],
                'sensitivity_analysis': synthesis.get('sensitivity', {}),
                'key_catalysts': llm_analysis.get('catalysts', []),
                'key_risks': llm_analysis.get('risks', []),
            },
            summary=self._generate_summary(synthesis, current_price),
            reasoning_chain=reasoning_steps,
            data_sources=data_sources,
            assumptions=key_assumptions,
            uncertainties=uncertainties,
            confidence_inputs={
                'data_quality': avg_data_quality,
                'data_completeness': data_completeness,
                'reasoning_strength': synthesis['synthesis_confidence'],
                'external_validation': 0.6 if len(valuation_results) > 1 else 0.4,
                'methodology_fit': synthesis.get('methodology_fit', 0.7)
            },
            warnings=warnings
        )
    
    # ============== 私有方法：估值实现 ==============
    
    async def _determine_valuation_methods(
        self,
        context: AnalysisContext,
        financial_data: Dict,
        market_data: Dict
    ) -> List[str]:
        """确定适用的估值方法"""
        methods = []
        
        # 检查 DCF 适用性：需要正的 FCF 历史
        fcf_history = financial_data.get('free_cash_flow_history', [])
        if fcf_history and sum(1 for f in fcf_history if f > 0) >= 3:
            methods.append('dcf')
        
        # 可比公司法：几乎总是适用
        methods.append('comparables')
        
        # 历史估值：需要足够的历史数据
        if financial_data.get('historical_pe') or financial_data.get('historical_ps'):
            methods.append('historical')
        
        # 股息折现：需要有股息历史
        if financial_data.get('dividend_history'):
            methods.append('ddm')
        
        return methods
    
    async def _dcf_valuation(
        self,
        context: AnalysisContext,
        financial_data: Dict,
        market_data: Dict,
        macro_view: Optional[Dict],
        custom_assumptions: Dict
    ) -> Optional[ValuationResult]:
        """DCF 估值"""
        try:
            # 获取基础数据
            fcf = financial_data.get('latest_fcf', 0)
            growth_rate = custom_assumptions.get('growth_rate') or \
                          financial_data.get('estimated_growth', 0.10)
            
            # 折现率（从宏观分析获取或使用默认值）
            discount_rate = custom_assumptions.get('discount_rate')
            if not discount_rate:
                if macro_view and macro_view.get('risk_free_rate'):
                    risk_free = macro_view['risk_free_rate']
                    equity_premium = 0.055
                    beta = financial_data.get('beta', 1.0)
                    discount_rate = risk_free + beta * equity_premium
                else:
                    discount_rate = 0.10  # 默认 10%
            
            terminal_growth = custom_assumptions.get('terminal_growth', 0.025)
            projection_years = custom_assumptions.get('projection_years', 10)
            
            # 预测现金流
            projected_fcf = []
            current_fcf = fcf
            for year in range(1, projection_years + 1):
                # 增长率逐年递减到终值增长率
                year_growth = growth_rate - (growth_rate - terminal_growth) * (year / projection_years) ** 2
                current_fcf = current_fcf * (1 + year_growth)
                projected_fcf.append(current_fcf)
            
            # 计算现值
            pv_fcf = sum(
                cf / (1 + discount_rate) ** (i + 1)
                for i, cf in enumerate(projected_fcf)
            )
            
            # 终值
            terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
            pv_terminal = terminal_value / (1 + discount_rate) ** projection_years
            
            # 企业价值
            enterprise_value = pv_fcf + pv_terminal
            
            # 转换为每股价值
            net_debt = financial_data.get('net_debt', 0)
            shares = financial_data.get('shares_outstanding', 1)
            equity_value = enterprise_value - net_debt
            fair_value = equity_value / shares
            
            # 敏感性分析
            sensitivity = self._run_dcf_sensitivity(
                fcf, growth_rate, discount_rate, terminal_growth,
                projection_years, net_debt, shares
            )
            
            # 计算价值区间（基于敏感性）
            value_range = (
                fair_value * 0.85,  # Bear case
                fair_value * 1.15   # Bull case
            )
            
            return ValuationResult(
                method='DCF',
                fair_value=fair_value,
                fair_value_range=value_range,
                upside_downside=(fair_value / market_data.get('current_price', fair_value) - 1),
                key_assumptions=[
                    f"折现率: {discount_rate:.1%}",
                    f"5年增长率: {growth_rate:.1%}",
                    f"永续增长率: {terminal_growth:.1%}",
                    f"预测期: {projection_years}年"
                ],
                sensitivity=sensitivity
            )
            
        except Exception as e:
            print(f"DCF valuation failed: {e}")
            return None
    
    async def _comparable_valuation(
        self,
        context: AnalysisContext,
        financial_data: Dict,
        market_data: Dict
    ) -> Optional[ValuationResult]:
        """可比公司估值"""
        try:
            # 获取可比公司倍数
            peer_pe = financial_data.get('peer_pe_median', 20)
            peer_ps = financial_data.get('peer_ps_median', 3)
            peer_ev_ebitda = financial_data.get('peer_ev_ebitda_median', 12)
            
            # 公司数据
            eps = financial_data.get('eps', 0)
            revenue_per_share = financial_data.get('revenue_per_share', 0)
            ebitda_per_share = financial_data.get('ebitda_per_share', 0)
            
            values = []
            
            if eps > 0 and peer_pe:
                values.append(('P/E', eps * peer_pe))
            if revenue_per_share > 0 and peer_ps:
                values.append(('P/S', revenue_per_share * peer_ps))
            if ebitda_per_share > 0 and peer_ev_ebitda:
                values.append(('EV/EBITDA', ebitda_per_share * peer_ev_ebitda))
            
            if not values:
                return None
            
            # 加权平均
            fair_value = sum(v[1] for v in values) / len(values)
            
            return ValuationResult(
                method='Comparable Companies',
                fair_value=fair_value,
                fair_value_range=(fair_value * 0.8, fair_value * 1.2),
                upside_downside=(fair_value / market_data.get('current_price', fair_value) - 1),
                key_assumptions=[
                    f"同业 P/E 中位数: {peer_pe:.1f}x",
                    f"同业 P/S 中位数: {peer_ps:.1f}x",
                    f"同业 EV/EBITDA 中位数: {peer_ev_ebitda:.1f}x"
                ],
                sensitivity={'methods_used': [v[0] for v in values]}
            )
            
        except Exception as e:
            print(f"Comparable valuation failed: {e}")
            return None
    
    async def _historical_valuation(
        self,
        context: AnalysisContext,
        financial_data: Dict,
        market_data: Dict
    ) -> Optional[ValuationResult]:
        """历史估值分析"""
        try:
            # 历史估值区间
            hist_pe_range = financial_data.get('historical_pe_range', (15, 25))
            current_eps = financial_data.get('eps', 0)
            
            if current_eps <= 0:
                return None
            
            fair_value_low = current_eps * hist_pe_range[0]
            fair_value_high = current_eps * hist_pe_range[1]
            fair_value_mid = (fair_value_low + fair_value_high) / 2
            
            return ValuationResult(
                method='Historical Multiples',
                fair_value=fair_value_mid,
                fair_value_range=(fair_value_low, fair_value_high),
                upside_downside=(fair_value_mid / market_data.get('current_price', fair_value_mid) - 1),
                key_assumptions=[
                    f"历史 P/E 区间: {hist_pe_range[0]:.1f}x - {hist_pe_range[1]:.1f}x",
                    f"当前 EPS: ${current_eps:.2f}"
                ],
                sensitivity={'pe_range': hist_pe_range}
            )
            
        except Exception as e:
            print(f"Historical valuation failed: {e}")
            return None
    
    async def _synthesize_valuations(
        self,
        context: AnalysisContext,
        results: List[ValuationResult],
        current_price: float
    ) -> Dict[str, Any]:
        """综合各估值方法结果"""
        
        # 加权平均
        total_weight = 0
        weighted_value = 0
        
        for result in results:
            weight = self._get_method_weight(result.method)
            weighted_value += result.fair_value * weight
            total_weight += weight
        
        fair_value_midpoint = weighted_value / total_weight if total_weight > 0 else 0
        
        # 计算区间
        all_lows = [r.fair_value_range[0] for r in results]
        all_highs = [r.fair_value_range[1] for r in results]
        fair_value_range = (
            sum(all_lows) / len(all_lows),
            sum(all_highs) / len(all_highs)
        )
        
        # 判断估值状态
        upside = (fair_value_midpoint / current_price - 1) if current_price > 0 else 0
        
        if upside > 0.2:
            stance = 'undervalued'
            conviction = 'high' if len(results) >= 2 and upside > 0.3 else 'medium'
        elif upside < -0.15:
            stance = 'overvalued'
            conviction = 'high' if len(results) >= 2 and upside < -0.25 else 'medium'
        else:
            stance = 'fairly_valued'
            conviction = 'medium'
        
        # 如果各方法分歧大，降低信念度
        values = [r.fair_value for r in results]
        if len(values) > 1:
            dispersion = (max(values) - min(values)) / (sum(values) / len(values))
            if dispersion > 0.3:
                conviction = 'low'
        
        return {
            'fair_value_midpoint': fair_value_midpoint,
            'fair_value_range': fair_value_range,
            'upside_potential': upside,
            'stance': stance,
            'conviction': conviction,
            'synthesis_confidence': 0.7 if len(results) >= 2 else 0.5,
            'methodology_fit': 0.8 if len(results) >= 2 else 0.6
        }
    
    async def _generate_llm_analysis(
        self,
        context: AnalysisContext,
        results: List[ValuationResult],
        synthesis: Dict,
        market_data: Dict,
        financial_data: Dict
    ) -> Dict[str, Any]:
        """使用 LLM 生成深度分析"""
        
        prompt = f"""基于以下估值分析结果，提供投资洞察：

## 估值结果汇总
- 综合公允价值区间: ${synthesis['fair_value_range'][0]:.2f} - ${synthesis['fair_value_range'][1]:.2f}
- 当前股价: ${market_data.get('current_price', 0):.2f}
- 潜在涨幅: {synthesis['upside_potential']:.1%}
- 估值判断: {synthesis['stance']}

## 各方法详情
{json.dumps([{'method': r.method, 'fair_value': r.fair_value, 'assumptions': r.key_assumptions} for r in results], ensure_ascii=False, indent=2)}

## 公司关键指标
- 收入增长: {financial_data.get('revenue_growth', 'N/A')}
- 毛利率: {financial_data.get('gross_margin', 'N/A')}
- 净利率: {financial_data.get('net_margin', 'N/A')}

请分析：
1. 估值的主要驱动因素
2. 最大的上行催化剂（2-3个）
3. 最大的下行风险（2-3个）
4. 投资者应该关注的关键指标

请用 JSON 格式返回，包含 drivers, catalysts, risks, key_metrics 字段。
"""
        
        try:
            response = await self._call_llm(
                context,
                prompt,
                task_type='deep_analysis',
                temperature=0.3
            )
            
            # 解析 JSON 响应
            # 实际实现中需要更robust的解析
            return json.loads(response)
        except:
            return {
                'drivers': [],
                'catalysts': ['需要进一步分析'],
                'risks': ['估值假设可能过于乐观或悲观'],
                'key_metrics': []
            }
    
    # ============== 辅助方法 ==============
    
    def _assess_data_quality(self, data: Dict) -> float:
        """评估数据质量"""
        required_fields = [
            'revenue', 'net_income', 'free_cash_flow',
            'total_assets', 'total_debt', 'shares_outstanding'
        ]
        present = sum(1 for f in required_fields if data.get(f) is not None)
        return present / len(required_fields)
    
    def _check_data_completeness(self, financial_data: Dict, market_data: Dict) -> float:
        """检查数据完整性"""
        scores = []
        
        # 财务数据完整性
        fin_fields = ['revenue', 'net_income', 'eps', 'free_cash_flow']
        fin_score = sum(1 for f in fin_fields if financial_data.get(f)) / len(fin_fields)
        scores.append(fin_score)
        
        # 市场数据完整性
        mkt_fields = ['current_price', 'market_cap', 'volume']
        mkt_score = sum(1 for f in mkt_fields if market_data.get(f)) / len(mkt_fields)
        scores.append(mkt_score)
        
        return sum(scores) / len(scores)
    
    def _get_method_weight(self, method: str) -> float:
        """获取估值方法权重"""
        weights = {
            'DCF': 0.35,
            'Comparable Companies': 0.35,
            'Historical Multiples': 0.20,
            'DDM': 0.10
        }
        return weights.get(method, 0.25)
    
    def _run_dcf_sensitivity(
        self,
        fcf: float,
        growth: float,
        discount: float,
        terminal: float,
        years: int,
        net_debt: float,
        shares: float
    ) -> Dict[str, Any]:
        """DCF 敏感性分析"""
        # 简化的敏感性矩阵
        sensitivities = {}
        
        for dr in [-0.01, 0, 0.01]:
            for tg in [-0.005, 0, 0.005]:
                key = f"DR{discount+dr:.1%}_TG{terminal+tg:.1%}"
                # 这里应该重新计算 DCF，简化起见使用近似
                adjustment = (1 - dr * 10) * (1 + tg * 20)
                sensitivities[key] = adjustment
        
        return {'sensitivity_matrix': sensitivities}
    
    def _assess_valuation_uncertainties(
        self,
        synthesis: Dict,
        results: List[ValuationResult],
        financial_data: Dict
    ) -> List[Uncertainty]:
        """评估估值不确定性"""
        uncertainties = []
        
        # 检查方法间分歧
        values = [r.fair_value for r in results]
        if len(values) > 1:
            spread = (max(values) - min(values)) / (sum(values) / len(values))
            if spread > 0.25:
                uncertainties.append(Uncertainty(
                    factor="method_disagreement",
                    description=f"不同估值方法结果差异较大 ({spread:.0%})",
                    impact="medium",
                    mitigation="建议深入分析各方法假设的合理性"
                ))
        
        # 检查依赖假设
        if any('增长率' in a for r in results for a in r.key_assumptions):
            uncertainties.append(Uncertainty(
                factor="growth_assumption",
                description="估值高度依赖增长率假设",
                impact="high",
                mitigation="关注公司实际增长趋势，设置情景分析"
            ))
        
        return uncertainties
    
    def _generate_summary(self, synthesis: Dict, current_price: float) -> str:
        """生成摘要"""
        stance_text = {
            'undervalued': '低估',
            'fairly_valued': '合理',
            'overvalued': '高估'
        }
        
        conviction_text = {
            'low': '低',
            'medium': '中等',
            'high': '高'
        }
        
        return (
            f"综合估值区间 ${synthesis['fair_value_range'][0]:.2f}-${synthesis['fair_value_range'][1]:.2f}，"
            f"相对当前价格 ${current_price:.2f} "
            f"{'上涨' if synthesis['upside_potential'] > 0 else '下跌'}空间 {abs(synthesis['upside_potential']):.1%}，"
            f"判断为{stance_text[synthesis['stance']]}，"
            f"置信度{conviction_text[synthesis['conviction']]}。"
        )
    
    def _create_failed_output(
        self,
        context: AnalysisContext,
        warnings: List[str]
    ) -> AgentOutput:
        """创建失败输出"""
        return self._create_output(
            result={'status': 'failed', 'reason': 'insufficient_data'},
            summary="由于数据不足，无法完成估值分析",
            reasoning_chain=[],
            data_sources=[],
            assumptions=[],
            uncertainties=[Uncertainty(
                factor="analysis_failed",
                description="无法执行估值分析",
                impact="critical"
            )],
            confidence_inputs={
                'data_quality': 0.2,
                'data_completeness': 0.2,
                'reasoning_strength': 0.1,
                'external_validation': 0.0,
                'methodology_fit': 0.1
            },
            warnings=warnings
        )


# ============== 使用示例 ==============

if __name__ == "__main__":
    import asyncio
    
    async def demo():
        # 初始化 (实际使用中需要配置 LLM)
        # agent = ValuationAgent(
        #     config_path="config/agents/valuation_agent.yaml",
        #     llm_gateway=llm_gateway
        # )
        
        # 示例数据
        context = AnalysisContext(
            target="AAPL",
            analysis_date=datetime.now()
        )
        
        inputs = {
            'financial_data': {
                'latest_fcf': 100_000_000_000,  # 1000亿美元
                'eps': 6.50,
                'revenue_per_share': 25.0,
                'shares_outstanding': 15_500_000_000,
                'estimated_growth': 0.08,
                'beta': 1.2,
                'net_debt': -50_000_000_000,  # 净现金
                'peer_pe_median': 28,
                'peer_ps_median': 7,
                'historical_pe_range': (20, 35)
            },
            'market_data': {
                'current_price': 185.0,
                'market_cap': 2_800_000_000_000
            }
        }
        
        # result = await agent.analyze(context, inputs)
        # print(result.to_dict())
        
        print("Demo completed - see code for implementation details")
    
    asyncio.run(demo())
