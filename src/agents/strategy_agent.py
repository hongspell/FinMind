"""
Strategy Agent - 投资策略综合代理

职责：
1. 综合所有其他代理的分析结果
2. 生成投资建议（买入/持有/卖出）
3. 制定短中长期行动计划
4. 识别关键催化剂和风险
5. 确定投资时机和仓位建议
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from datetime import datetime, timedelta
import json

# 假设从 core 导入基础类
# from core.base import BaseAgent, AgentOutput, AnalysisContext, LLMGateway


class Recommendation(Enum):
    """投资建议枚举"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    REDUCE = "reduce"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


class TimeHorizon(Enum):
    """投资时间范围"""
    SHORT_TERM = "short_term"      # 1-3 个月
    MEDIUM_TERM = "medium_term"    # 3-12 个月
    LONG_TERM = "long_term"        # 1-3 年


class RiskTolerance(Enum):
    """风险承受能力"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class ActionItem:
    """行动项"""
    action: str                      # 具体行动
    priority: str                    # high/medium/low
    time_horizon: TimeHorizon        # 时间范围
    trigger_condition: str           # 触发条件
    target_price: Optional[float]    # 目标价格
    stop_loss: Optional[float]       # 止损价格
    position_size: str               # 建议仓位
    rationale: str                   # 理由


@dataclass
class Catalyst:
    """催化剂/风险事件"""
    event: str                       # 事件描述
    expected_date: Optional[str]     # 预期日期
    probability: float               # 发生概率
    impact: str                      # positive/negative/uncertain
    magnitude: str                   # high/medium/low
    description: str                 # 详细描述


@dataclass
class ScenarioAnalysis:
    """情景分析"""
    scenario_name: str               # bull/base/bear
    probability: float               # 概率
    target_price: float              # 目标价
    upside_downside: float           # 涨跌幅
    key_assumptions: List[str]       # 关键假设
    triggers: List[str]              # 触发因素


@dataclass
class PositionRecommendation:
    """仓位建议"""
    current_action: str              # 当前行动建议
    position_size_pct: Tuple[float, float]  # 建议仓位范围 (min%, max%)
    entry_strategy: str              # 入场策略
    exit_strategy: str               # 出场策略
    rebalance_triggers: List[str]    # 再平衡触发条件


@dataclass
class StrategyOutput:
    """策略代理输出"""
    # 核心建议
    recommendation: Recommendation
    conviction: str                  # high/medium/low
    time_horizon: TimeHorizon
    
    # 投资论点
    investment_thesis: str           # 核心投资论点
    thesis_pillars: List[str]        # 论点支柱
    
    # 价值评估综合
    fair_value_range: Tuple[float, float]
    current_price: float
    upside_potential: Tuple[float, float]  # 上涨空间范围
    
    # 情景分析
    scenarios: List[ScenarioAnalysis]
    
    # 行动计划
    action_items: Dict[TimeHorizon, List[ActionItem]]
    position_recommendation: PositionRecommendation
    
    # 催化剂与风险
    catalysts: List[Catalyst]
    key_risks: List[Catalyst]
    
    # 综合信号
    signal_summary: Dict[str, Any]   # 各代理信号汇总
    conflicts: List[str]             # 信号冲突
    
    # 元数据
    confidence_score: float
    reasoning_chain: List[Dict]
    data_sources: List[str]
    assumptions: List[str]
    uncertainties: List[str]
    warnings: List[str]
    generated_at: str


class StrategyAgent:
    """
    投资策略综合代理
    
    整合所有分析结果，生成可操作的投资建议
    """
    
    def __init__(self, config: Dict[str, Any], llm_gateway=None):
        self.config = config
        self.llm = llm_gateway
        
        # 加载配置
        self.weights = config.get('signal_weights', {
            'valuation': 0.25,
            'earnings': 0.25,
            'risk': 0.20,
            'technical': 0.10,
            'sentiment': 0.10,
            'macro': 0.10
        })
        
        self.guardrails = config.get('guardrails', [])
        self.thresholds = config.get('thresholds', {})
    
    async def analyze(
        self,
        context: Dict[str, Any],
        agent_outputs: Dict[str, Any]
    ) -> StrategyOutput:
        """
        综合分析并生成策略建议
        
        Args:
            context: 分析上下文（目标、时间、用户偏好等）
            agent_outputs: 所有其他代理的输出
        """
        target = context.get('target', 'Unknown')
        reasoning_chain = []
        warnings = []
        
        # Step 1: 提取各代理核心信号
        reasoning_chain.append({
            'step': 'extract_signals',
            'description': '提取各代理核心信号',
            'timestamp': datetime.now().isoformat()
        })
        
        signals = self._extract_signals(agent_outputs)
        
        # Step 2: 检测信号冲突
        reasoning_chain.append({
            'step': 'detect_conflicts',
            'description': '检测代理信号冲突'
        })
        
        conflicts = self._detect_conflicts(signals)
        if conflicts:
            warnings.append(f"检测到 {len(conflicts)} 个信号冲突，需谨慎决策")
        
        # Step 3: 综合估值分析
        reasoning_chain.append({
            'step': 'synthesize_valuation',
            'description': '综合估值结论'
        })
        
        valuation_synthesis = self._synthesize_valuation(
            agent_outputs.get('valuation'),
            agent_outputs.get('earnings')
        )
        
        # Step 4: 评估风险调整后收益
        reasoning_chain.append({
            'step': 'risk_adjusted_analysis',
            'description': '计算风险调整后收益预期'
        })
        
        risk_adjusted = self._calculate_risk_adjusted_return(
            valuation_synthesis,
            agent_outputs.get('risk'),
            signals
        )
        
        # Step 5: 生成情景分析
        reasoning_chain.append({
            'step': 'scenario_analysis',
            'description': '生成牛/基准/熊情景'
        })
        
        scenarios = self._build_scenarios(
            valuation_synthesis,
            agent_outputs.get('risk'),
            agent_outputs.get('macro')
        )
        
        # Step 6: 确定核心建议
        reasoning_chain.append({
            'step': 'determine_recommendation',
            'description': '基于综合分析确定投资建议'
        })
        
        recommendation, conviction = self._determine_recommendation(
            signals,
            risk_adjusted,
            conflicts,
            context.get('risk_tolerance', RiskTolerance.MODERATE)
        )
        
        # Step 7: 制定行动计划
        reasoning_chain.append({
            'step': 'create_action_plan',
            'description': '制定短中长期行动计划'
        })
        
        action_items = self._create_action_plan(
            recommendation,
            valuation_synthesis,
            agent_outputs.get('technical'),
            scenarios
        )
        
        # Step 8: 识别催化剂和风险
        reasoning_chain.append({
            'step': 'identify_catalysts',
            'description': '识别关键催化剂和风险事件'
        })
        
        catalysts, key_risks = self._identify_catalysts_and_risks(
            agent_outputs.get('earnings'),
            agent_outputs.get('sentiment'),
            agent_outputs.get('macro'),
            agent_outputs.get('risk')
        )
        
        # Step 9: 生成仓位建议
        reasoning_chain.append({
            'step': 'position_sizing',
            'description': '生成仓位和入场策略建议'
        })
        
        position_rec = self._generate_position_recommendation(
            recommendation,
            conviction,
            risk_adjusted,
            context.get('portfolio_context', {})
        )
        
        # Step 10: 使用 LLM 生成投资论点
        reasoning_chain.append({
            'step': 'generate_thesis',
            'description': '使用 LLM 生成投资论点叙述'
        })
        
        thesis = await self._generate_investment_thesis(
            target,
            signals,
            recommendation,
            valuation_synthesis,
            catalysts,
            key_risks
        )
        
        # Step 11: 应用护栏检查
        reasoning_chain.append({
            'step': 'apply_guardrails',
            'description': '应用策略护栏检查'
        })
        
        guardrail_warnings = self._apply_guardrails(
            recommendation,
            conviction,
            signals,
            conflicts
        )
        warnings.extend(guardrail_warnings)
        
        # Step 12: 计算综合置信度
        confidence = self._calculate_confidence(
            signals,
            conflicts,
            agent_outputs
        )
        
        # 构建输出
        current_price = valuation_synthesis.get('current_price', 0)
        fair_value_range = valuation_synthesis.get('fair_value_range', (0, 0))
        
        return StrategyOutput(
            recommendation=recommendation,
            conviction=conviction,
            time_horizon=self._determine_time_horizon(context, recommendation),
            
            investment_thesis=thesis['narrative'],
            thesis_pillars=thesis['pillars'],
            
            fair_value_range=fair_value_range,
            current_price=current_price,
            upside_potential=self._calculate_upside(current_price, fair_value_range),
            
            scenarios=scenarios,
            
            action_items=action_items,
            position_recommendation=position_rec,
            
            catalysts=catalysts,
            key_risks=key_risks,
            
            signal_summary=signals,
            conflicts=conflicts,
            
            confidence_score=confidence,
            reasoning_chain=reasoning_chain,
            data_sources=self._collect_data_sources(agent_outputs),
            assumptions=self._collect_assumptions(agent_outputs),
            uncertainties=self._identify_uncertainties(signals, conflicts),
            warnings=warnings,
            generated_at=datetime.now().isoformat()
        )
    
    def _extract_signals(self, agent_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """从各代理输出提取核心信号"""
        signals = {}
        
        # 估值信号
        if valuation := agent_outputs.get('valuation'):
            fair_value = valuation.get('fair_value_range', (0, 0))
            current = valuation.get('current_price', 0)
            if current > 0:
                upside_mid = ((fair_value[0] + fair_value[1]) / 2 - current) / current
                signals['valuation'] = {
                    'signal': self._upside_to_signal(upside_mid),
                    'upside': upside_mid,
                    'confidence': valuation.get('confidence', 0.5),
                    'fair_value_range': fair_value
                }
        
        # 盈利质量信号
        if earnings := agent_outputs.get('earnings'):
            signals['earnings'] = {
                'signal': earnings.get('overall_signal', 'neutral'),
                'quality_score': earnings.get('quality_score', 0.5),
                'growth_trend': earnings.get('growth_trend', 'stable'),
                'confidence': earnings.get('confidence', 0.5)
            }
        
        # 风险信号
        if risk := agent_outputs.get('risk'):
            signals['risk'] = {
                'signal': self._risk_to_signal(risk.get('overall_risk_level', 'medium')),
                'risk_score': risk.get('composite_risk_score', 0.5),
                'critical_risks': risk.get('critical_risks', []),
                'confidence': risk.get('confidence', 0.5)
            }
        
        # 技术信号
        if technical := agent_outputs.get('technical'):
            signals['technical'] = {
                'signal': technical.get('trend_signal', 'neutral'),
                'trend': technical.get('trend', 'sideways'),
                'momentum': technical.get('momentum', 'neutral'),
                'support': technical.get('support_level'),
                'resistance': technical.get('resistance_level'),
                'confidence': technical.get('confidence', 0.5)
            }
        
        # 情绪信号
        if sentiment := agent_outputs.get('sentiment'):
            signals['sentiment'] = {
                'signal': sentiment.get('overall_signal', 'neutral'),
                'score': sentiment.get('composite_score', 0.5),
                'news_sentiment': sentiment.get('news_sentiment', 'neutral'),
                'social_sentiment': sentiment.get('social_sentiment', 'neutral'),
                'confidence': sentiment.get('confidence', 0.5)
            }
        
        # 宏观信号
        if macro := agent_outputs.get('macro'):
            signals['macro'] = {
                'signal': macro.get('market_signal', 'neutral'),
                'economic_cycle': macro.get('cycle_phase', 'mid'),
                'sector_outlook': macro.get('sector_outlook', 'neutral'),
                'confidence': macro.get('confidence', 0.5)
            }
        
        return signals
    
    def _upside_to_signal(self, upside: float) -> str:
        """将上涨空间转换为信号"""
        if upside > 0.30:
            return 'strong_bullish'
        elif upside > 0.15:
            return 'bullish'
        elif upside > 0.05:
            return 'slightly_bullish'
        elif upside > -0.05:
            return 'neutral'
        elif upside > -0.15:
            return 'slightly_bearish'
        elif upside > -0.30:
            return 'bearish'
        else:
            return 'strong_bearish'
    
    def _risk_to_signal(self, risk_level: str) -> str:
        """将风险等级转换为信号"""
        mapping = {
            'very_low': 'bullish',
            'low': 'slightly_bullish',
            'medium': 'neutral',
            'high': 'slightly_bearish',
            'very_high': 'bearish',
            'critical': 'strong_bearish'
        }
        return mapping.get(risk_level, 'neutral')
    
    def _detect_conflicts(self, signals: Dict[str, Any]) -> List[str]:
        """检测信号之间的冲突"""
        conflicts = []
        
        signal_values = {
            'strong_bullish': 2, 'bullish': 1, 'slightly_bullish': 0.5,
            'neutral': 0,
            'slightly_bearish': -0.5, 'bearish': -1, 'strong_bearish': -2
        }
        
        # 提取各信号数值
        signal_nums = {}
        for agent, data in signals.items():
            if 'signal' in data:
                signal_nums[agent] = signal_values.get(data['signal'], 0)
        
        # 检测主要冲突
        if 'valuation' in signal_nums and 'technical' in signal_nums:
            val_sig = signal_nums['valuation']
            tech_sig = signal_nums['technical']
            if abs(val_sig - tech_sig) > 1.5:
                conflicts.append(
                    f"估值信号({signals['valuation']['signal']})与技术信号({signals['technical']['signal']})存在显著分歧"
                )
        
        if 'valuation' in signal_nums and 'risk' in signal_nums:
            val_sig = signal_nums['valuation']
            risk_sig = signal_nums['risk']
            if val_sig > 0.5 and risk_sig < -0.5:
                conflicts.append(
                    "估值显示低估但风险评估偏高，需要权衡上行空间与下行风险"
                )
        
        if 'sentiment' in signal_nums and 'valuation' in signal_nums:
            sent_sig = signal_nums['sentiment']
            val_sig = signal_nums['valuation']
            if abs(sent_sig - val_sig) > 1.5:
                conflicts.append(
                    f"市场情绪({signals['sentiment']['signal']})与基本面估值({signals['valuation']['signal']})背离"
                )
        
        if 'macro' in signal_nums and 'earnings' in signal_nums:
            macro_sig = signal_nums['macro']
            earn_sig = signal_nums.get('earnings', 0)
            if abs(macro_sig - earn_sig) > 1.5:
                conflicts.append(
                    f"宏观环境信号与公司财报表现存在背离"
                )

        return conflicts

    # ============== ChainExecutor 适配器方法 ==============

    async def synthesize_recommendation(self, context, inputs: Dict) -> StrategyOutput:
        """
        ChainExecutor 调用的适配方法

        将 ChainExecutor 的调用格式适配到 analyze 方法
        """
        # 从 context 获取基础信息
        target = context.target if hasattr(context, 'target') else 'UNKNOWN'

        # 构建 context dict
        analysis_context = {
            'target': target,
            'analysis_date': context.analysis_date.isoformat() if hasattr(context, 'analysis_date') else datetime.now().isoformat(),
            'risk_tolerance': inputs.get('risk_tolerance', RiskTolerance.MODERATE),
            'portfolio_context': inputs.get('portfolio_context', {})
        }

        # 从 inputs 中提取各代理的输出
        agent_outputs = {
            'valuation': inputs.get('valuation_analysis'),
            'earnings': inputs.get('earnings_analysis'),
            'risk': inputs.get('risk_assessment'),
            'technical': inputs.get('technical_analysis'),
            'sentiment': inputs.get('sentiment_analysis'),
            'macro': inputs.get('macro_analysis')
        }

        return await self.analyze(analysis_context, agent_outputs)

    # ============== 辅助方法实现 ==============

    def _synthesize_valuation(
        self,
        valuation_output: Optional[Dict],
        earnings_output: Optional[Dict]
    ) -> Dict[str, Any]:
        """综合估值分析"""
        result = {
            'fair_value_range': (0, 0),
            'current_price': 0,
            'upside_potential': 0,
            'conviction': 'low'
        }

        if valuation_output:
            result['fair_value_range'] = valuation_output.get('fair_value_range', (0, 0))
            result['current_price'] = valuation_output.get('current_price', 0)

            fv_mid = (result['fair_value_range'][0] + result['fair_value_range'][1]) / 2
            if result['current_price'] > 0:
                result['upside_potential'] = (fv_mid - result['current_price']) / result['current_price']

            result['conviction'] = valuation_output.get('conviction', 'medium')

        # 用盈利数据调整
        if earnings_output:
            quality = earnings_output.get('quality_score', 0.5)
            if quality > 0.7:
                # 高质量盈利支持估值
                pass
            elif quality < 0.4:
                # 低质量盈利降低信心
                result['conviction'] = 'low'

        return result

    def _calculate_risk_adjusted_return(
        self,
        valuation_synthesis: Dict,
        risk_output: Optional[Dict],
        signals: Dict
    ) -> Dict[str, Any]:
        """计算风险调整后收益"""
        base_upside = valuation_synthesis.get('upside_potential', 0)

        risk_adjustment = 1.0
        if risk_output:
            risk_score = risk_output.get('composite_risk_score', 0.5)
            risk_adjustment = 1.0 - (risk_score * 0.5)  # 风险越高，调整越大

        return {
            'raw_upside': base_upside,
            'risk_adjustment': risk_adjustment,
            'adjusted_upside': base_upside * risk_adjustment,
            'sharpe_estimate': base_upside / max(risk_adjustment, 0.1)
        }

    def _build_scenarios(
        self,
        valuation_synthesis: Dict,
        risk_output: Optional[Dict],
        macro_output: Optional[Dict]
    ) -> List[ScenarioAnalysis]:
        """构建情景分析"""
        fv_range = valuation_synthesis.get('fair_value_range', (0, 0))
        current = valuation_synthesis.get('current_price', 1)

        if current == 0:
            current = 1

        # Bull case
        bull_target = fv_range[1] * 1.1
        bull_scenario = ScenarioAnalysis(
            scenario_name='bull',
            probability=0.25,
            target_price=bull_target,
            upside_downside=(bull_target - current) / current,
            key_assumptions=['经济持续增长', '公司超预期表现', '估值扩张'],
            triggers=['超预期财报', '新产品成功', '利好政策']
        )

        # Base case
        base_target = (fv_range[0] + fv_range[1]) / 2
        base_scenario = ScenarioAnalysis(
            scenario_name='base',
            probability=0.50,
            target_price=base_target,
            upside_downside=(base_target - current) / current,
            key_assumptions=['经济温和增长', '公司符合预期'],
            triggers=['正常经营节奏']
        )

        # Bear case
        bear_target = fv_range[0] * 0.9
        bear_scenario = ScenarioAnalysis(
            scenario_name='bear',
            probability=0.25,
            target_price=bear_target,
            upside_downside=(bear_target - current) / current,
            key_assumptions=['经济放缓', '竞争加剧', '估值收缩'],
            triggers=['低于预期财报', '宏观风险', '行业逆风']
        )

        return [bull_scenario, base_scenario, bear_scenario]

    def _determine_recommendation(
        self,
        signals: Dict,
        risk_adjusted: Dict,
        conflicts: List[str],
        risk_tolerance: RiskTolerance
    ) -> Tuple[Recommendation, str]:
        """确定投资建议"""
        adjusted_upside = risk_adjusted.get('adjusted_upside', 0)

        # 基础建议
        if adjusted_upside > 0.25:
            rec = Recommendation.STRONG_BUY
        elif adjusted_upside > 0.15:
            rec = Recommendation.BUY
        elif adjusted_upside > 0.05:
            rec = Recommendation.BUY if risk_tolerance == RiskTolerance.AGGRESSIVE else Recommendation.HOLD
        elif adjusted_upside > -0.05:
            rec = Recommendation.HOLD
        elif adjusted_upside > -0.15:
            rec = Recommendation.REDUCE
        elif adjusted_upside > -0.25:
            rec = Recommendation.SELL
        else:
            rec = Recommendation.STRONG_SELL

        # 计算信念度
        conviction = 'high'
        if len(conflicts) > 2:
            conviction = 'low'
        elif len(conflicts) > 0:
            conviction = 'medium'

        return rec, conviction

    def _create_action_plan(
        self,
        recommendation: Recommendation,
        valuation_synthesis: Dict,
        technical_output: Optional[Dict],
        scenarios: List[ScenarioAnalysis]
    ) -> Dict[TimeHorizon, List[ActionItem]]:
        """创建行动计划"""
        current_price = valuation_synthesis.get('current_price', 0)
        fv_range = valuation_synthesis.get('fair_value_range', (0, 0))

        actions = {
            TimeHorizon.SHORT_TERM: [],
            TimeHorizon.MEDIUM_TERM: [],
            TimeHorizon.LONG_TERM: []
        }

        if recommendation in [Recommendation.STRONG_BUY, Recommendation.BUY]:
            actions[TimeHorizon.SHORT_TERM].append(ActionItem(
                action='建立初始仓位',
                priority='high',
                time_horizon=TimeHorizon.SHORT_TERM,
                trigger_condition='当前价格',
                target_price=fv_range[0],
                stop_loss=current_price * 0.9,
                position_size='30-50%目标仓位',
                rationale='估值具有吸引力'
            ))

            actions[TimeHorizon.MEDIUM_TERM].append(ActionItem(
                action='逢低加仓',
                priority='medium',
                time_horizon=TimeHorizon.MEDIUM_TERM,
                trigger_condition=f'价格回调至 ${current_price * 0.95:.2f}',
                target_price=(fv_range[0] + fv_range[1]) / 2,
                stop_loss=current_price * 0.85,
                position_size='增加20-30%仓位',
                rationale='利用市场波动增加敞口'
            ))

        elif recommendation in [Recommendation.SELL, Recommendation.STRONG_SELL]:
            actions[TimeHorizon.SHORT_TERM].append(ActionItem(
                action='减仓或清仓',
                priority='high',
                time_horizon=TimeHorizon.SHORT_TERM,
                trigger_condition='当前价格',
                target_price=None,
                stop_loss=None,
                position_size='减少50-100%仓位',
                rationale='估值偏高或风险过大'
            ))

        return actions

    def _identify_catalysts_and_risks(
        self,
        earnings_output: Optional[Dict],
        sentiment_output: Optional[Dict],
        macro_output: Optional[Dict],
        risk_output: Optional[Dict]
    ) -> Tuple[List[Catalyst], List[Catalyst]]:
        """识别催化剂和风险"""
        catalysts = []
        risks = []

        # 财报催化剂
        if earnings_output:
            catalysts.append(Catalyst(
                event='季度财报发布',
                expected_date='下个财季',
                probability=1.0,
                impact='uncertain',
                magnitude='high',
                description='财报表现将影响短期股价走势'
            ))

        # 情绪相关
        if sentiment_output:
            sentiment_score = sentiment_output.get('composite_score', 0.5)
            if sentiment_score > 0.7:
                catalysts.append(Catalyst(
                    event='市场情绪积极',
                    expected_date=None,
                    probability=0.7,
                    impact='positive',
                    magnitude='medium',
                    description='当前市场情绪正面，可能推动股价上涨'
                ))
            elif sentiment_score < 0.3:
                risks.append(Catalyst(
                    event='市场情绪消极',
                    expected_date=None,
                    probability=0.7,
                    impact='negative',
                    magnitude='medium',
                    description='当前市场情绪负面，可能压制股价'
                ))

        # 风险评估
        if risk_output:
            critical_risks = risk_output.get('critical_risks', [])
            for cr in critical_risks[:3]:
                risks.append(Catalyst(
                    event=cr.get('name', '未知风险'),
                    expected_date=None,
                    probability=0.5,
                    impact='negative',
                    magnitude='high',
                    description=cr.get('description', '')
                ))

        return catalysts, risks

    def _generate_position_recommendation(
        self,
        recommendation: Recommendation,
        conviction: str,
        risk_adjusted: Dict,
        portfolio_context: Dict
    ) -> PositionRecommendation:
        """生成仓位建议"""
        # 基于建议确定仓位范围
        position_map = {
            Recommendation.STRONG_BUY: (5, 10),
            Recommendation.BUY: (3, 7),
            Recommendation.HOLD: (0, 3),
            Recommendation.REDUCE: (0, 2),
            Recommendation.SELL: (0, 0),
            Recommendation.STRONG_SELL: (0, 0)
        }

        base_range = position_map.get(recommendation, (0, 3))

        # 根据信念度调整
        if conviction == 'high':
            position_range = (base_range[0], base_range[1] * 1.2)
        elif conviction == 'low':
            position_range = (base_range[0], base_range[1] * 0.7)
        else:
            position_range = base_range

        return PositionRecommendation(
            current_action=recommendation.value,
            position_size_pct=position_range,
            entry_strategy='分批建仓，分散入场时点' if recommendation in [Recommendation.BUY, Recommendation.STRONG_BUY] else '逐步减持',
            exit_strategy='设置止盈止损点，动态调整',
            rebalance_triggers=['价格偏离目标20%', '基本面重大变化', '宏观环境转变']
        )

    async def _generate_investment_thesis(
        self,
        target: str,
        signals: Dict,
        recommendation: Recommendation,
        valuation_synthesis: Dict,
        catalysts: List[Catalyst],
        key_risks: List[Catalyst]
    ) -> Dict[str, Any]:
        """生成投资论点"""
        # 构建论点支柱
        pillars = []

        if signals.get('valuation', {}).get('signal', '').endswith('bullish'):
            pillars.append('估值具有吸引力')
        if signals.get('earnings', {}).get('quality_score', 0) > 0.6:
            pillars.append('盈利质量较高')
        if signals.get('technical', {}).get('signal', '') == 'bullish':
            pillars.append('技术面支撑')
        if signals.get('sentiment', {}).get('signal', '') == 'bullish':
            pillars.append('市场情绪积极')

        # 生成叙述
        rec_text = {
            Recommendation.STRONG_BUY: '强烈建议买入',
            Recommendation.BUY: '建议买入',
            Recommendation.HOLD: '建议持有',
            Recommendation.REDUCE: '建议减持',
            Recommendation.SELL: '建议卖出',
            Recommendation.STRONG_SELL: '强烈建议卖出'
        }

        upside = valuation_synthesis.get('upside_potential', 0)
        narrative = (
            f"对于 {target}，我们{rec_text.get(recommendation, '保持中性')}。"
            f"基于综合分析，预计潜在{'上涨' if upside > 0 else '下跌'}空间约为 {abs(upside):.1%}。"
        )

        if pillars:
            narrative += f"主要支撑因素包括：{'、'.join(pillars)}。"

        if key_risks:
            narrative += f"需要关注的主要风险包括：{key_risks[0].event if key_risks else '市场波动'}。"

        return {
            'narrative': narrative,
            'pillars': pillars if pillars else ['综合分析']
        }

    def _apply_guardrails(
        self,
        recommendation: Recommendation,
        conviction: str,
        signals: Dict,
        conflicts: List[str]
    ) -> List[str]:
        """应用护栏检查"""
        warnings = []

        # 检查信号冲突警告
        if len(conflicts) > 2 and recommendation in [Recommendation.STRONG_BUY, Recommendation.STRONG_SELL]:
            warnings.append('存在多个信号冲突，强烈建议可能存在偏差，请谨慎决策')

        # 检查置信度警告
        if conviction == 'low' and recommendation != Recommendation.HOLD:
            warnings.append('当前置信度较低，建议降低仓位或等待更多确认信号')

        # 检查极端建议
        if recommendation in [Recommendation.STRONG_BUY, Recommendation.STRONG_SELL]:
            warnings.append('这是一个强烈的投资建议，请确保已充分理解相关风险')

        return warnings

    def _calculate_confidence(
        self,
        signals: Dict,
        conflicts: List[str],
        agent_outputs: Dict
    ) -> float:
        """计算综合置信度"""
        base_confidence = 0.6

        # 信号一致性加分
        signal_types = [s.get('signal', 'neutral') for s in signals.values() if isinstance(s, dict)]
        bullish_count = sum(1 for s in signal_types if 'bullish' in s)
        bearish_count = sum(1 for s in signal_types if 'bearish' in s)

        if len(signal_types) > 0:
            consistency = max(bullish_count, bearish_count) / len(signal_types)
            base_confidence += consistency * 0.2

        # 冲突扣分
        base_confidence -= len(conflicts) * 0.05

        # 数据完整性加分
        data_count = sum(1 for v in agent_outputs.values() if v is not None)
        base_confidence += (data_count / 6) * 0.1

        return max(0.1, min(0.95, base_confidence))

    def _determine_time_horizon(
        self,
        context: Dict,
        recommendation: Recommendation
    ) -> TimeHorizon:
        """确定投资时间范围"""
        user_horizon = context.get('time_horizon')
        if user_horizon:
            return user_horizon

        # 基于建议类型推断
        if recommendation in [Recommendation.STRONG_BUY, Recommendation.STRONG_SELL]:
            return TimeHorizon.MEDIUM_TERM
        else:
            return TimeHorizon.LONG_TERM

    def _calculate_upside(
        self,
        current_price: float,
        fair_value_range: Tuple[float, float]
    ) -> Tuple[float, float]:
        """计算上涨空间范围"""
        if current_price == 0:
            return (0, 0)

        upside_low = (fair_value_range[0] - current_price) / current_price
        upside_high = (fair_value_range[1] - current_price) / current_price
        return (upside_low, upside_high)

    def _collect_data_sources(self, agent_outputs: Dict) -> List[str]:
        """收集数据源"""
        sources = []
        for agent_name, output in agent_outputs.items():
            if output:
                sources.append(f"{agent_name}_analysis")
        return sources

    def _collect_assumptions(self, agent_outputs: Dict) -> List[str]:
        """收集假设"""
        assumptions = []
        for agent_name, output in agent_outputs.items():
            if output and isinstance(output, dict):
                agent_assumptions = output.get('assumptions', [])
                assumptions.extend(agent_assumptions[:2])  # 每个代理最多2个
        return assumptions

    def _identify_uncertainties(
        self,
        signals: Dict,
        conflicts: List[str]
    ) -> List[str]:
        """识别不确定性"""
        uncertainties = []

        if conflicts:
            uncertainties.append('代理信号存在冲突，增加分析不确定性')

        # 检查低置信度信号
        for agent, data in signals.items():
            if isinstance(data, dict) and data.get('confidence', 1) < 0.5:
                uncertainties.append(f'{agent} 分析置信度较低')

        return uncertainties