"""
Sentiment Agent & Risk Agent
情绪分析和风险评估Agent
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import re
import math


# ============ Sentiment Agent ============

class SentimentLevel(Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


@dataclass
class NewsItem:
    """新闻条目"""
    title: str
    source: str
    published_at: datetime
    url: Optional[str] = None
    summary: Optional[str] = None
    sentiment_score: float = 0.0  # -1 to 1
    relevance_score: float = 1.0  # 0 to 1
    topics: List[str] = field(default_factory=list)


@dataclass
class SocialSentiment:
    """社交媒体情绪"""
    platform: str
    mention_count: int
    positive_ratio: float
    negative_ratio: float
    neutral_ratio: float
    trending_topics: List[str]
    influential_posts: List[Dict[str, Any]]
    volume_change_pct: float  # 相比前期的变化


@dataclass
class AnalystSentiment:
    """分析师情绪"""
    total_analysts: int
    strong_buy: int
    buy: int
    hold: int
    sell: int
    strong_sell: int
    average_target: float
    high_target: float
    low_target: float
    recent_upgrades: int
    recent_downgrades: int


@dataclass
class SentimentAnalysisResult:
    """情绪分析结果"""
    symbol: str
    timestamp: datetime
    
    # 综合情绪
    overall_sentiment: SentimentLevel
    sentiment_score: float  # -1 to 1
    confidence: float
    
    # 新闻情绪
    news_sentiment: float
    news_volume: int
    key_news: List[NewsItem]
    
    # 社交媒体情绪
    social_sentiment: Optional[SocialSentiment]
    
    # 分析师情绪
    analyst_sentiment: Optional[AnalystSentiment]
    
    # 情绪趋势
    sentiment_trend: str  # "improving", "stable", "deteriorating"
    sentiment_momentum: float  # 情绪变化速度
    
    # 关键发现
    key_drivers: List[str]  # 情绪驱动因素
    risks_identified: List[str]
    opportunities_identified: List[str]
    
    # 分析摘要
    summary: str


class SentimentAnalyzer:
    """情绪分析器"""
    
    # 正面词汇
    POSITIVE_WORDS = {
        'beat', 'beats', 'exceed', 'exceeds', 'exceeded', 'outperform', 'outperforms',
        'upgrade', 'upgraded', 'positive', 'strong', 'growth', 'profit', 'gain', 'gains',
        'surge', 'surges', 'rally', 'rallies', 'bullish', 'optimistic', 'record',
        'breakthrough', 'success', 'successful', 'innovation', 'innovative', 'expand',
        'expansion', 'partnership', 'deal', 'acquisition', 'launch', 'launched',
        'improve', 'improved', 'improvement', 'higher', 'increase', 'increased',
        'boost', 'boosted', 'momentum', 'opportunity', 'opportunities', 'upside',
        '超预期', '增长', '上涨', '利好', '突破', '创新高', '看涨', '买入', '强势'
    }
    
    # 负面词汇
    NEGATIVE_WORDS = {
        'miss', 'misses', 'missed', 'below', 'underperform', 'underperforms',
        'downgrade', 'downgraded', 'negative', 'weak', 'weakness', 'loss', 'losses',
        'decline', 'declines', 'declining', 'fall', 'falls', 'fell', 'drop', 'drops',
        'plunge', 'plunges', 'crash', 'bearish', 'pessimistic', 'concern', 'concerns',
        'risk', 'risks', 'warning', 'warns', 'cut', 'cuts', 'reduce', 'reduced',
        'lower', 'layoff', 'layoffs', 'lawsuit', 'investigation', 'probe',
        'scandal', 'fraud', 'bankruptcy', 'default', 'recession', 'slowdown',
        '低于预期', '下跌', '利空', '跌破', '创新低', '看跌', '卖出', '弱势', '亏损'
    }
    
    # 高影响词汇（加权）
    HIGH_IMPACT_POSITIVE = {'breakthrough', 'record', 'acquisition', 'beat', 'surge'}
    HIGH_IMPACT_NEGATIVE = {'crash', 'fraud', 'bankruptcy', 'investigation', 'scandal'}
    
    def analyze_text(self, text: str) -> Tuple[float, float]:
        """分析文本情绪
        
        Returns:
            (sentiment_score, confidence)
        """
        if not text:
            return 0.0, 0.0
        
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = 0
        negative_count = 0
        high_impact_bonus = 0
        
        for word in words:
            if word in self.POSITIVE_WORDS:
                positive_count += 1
                if word in self.HIGH_IMPACT_POSITIVE:
                    high_impact_bonus += 0.5
            elif word in self.NEGATIVE_WORDS:
                negative_count += 1
                if word in self.HIGH_IMPACT_NEGATIVE:
                    high_impact_bonus -= 0.5
        
        total = positive_count + negative_count
        if total == 0:
            return 0.0, 0.3  # 无情绪词，低置信度
        
        # 计算基础情绪分数
        base_score = (positive_count - negative_count) / total
        
        # 应用高影响词汇加成
        adjusted_score = base_score + high_impact_bonus * 0.2
        
        # 限制范围
        score = max(-1.0, min(1.0, adjusted_score))
        
        # 置信度基于情绪词密度
        word_count = len(words)
        sentiment_density = total / word_count if word_count > 0 else 0
        confidence = min(0.9, sentiment_density * 5 + 0.3)
        
        return score, confidence
    
    def aggregate_sentiments(
        self,
        sentiments: List[Tuple[float, float, float]]  # (score, confidence, weight)
    ) -> Tuple[float, float]:
        """聚合多个情绪分数"""
        if not sentiments:
            return 0.0, 0.0
        
        total_weight = 0
        weighted_score = 0
        weighted_confidence = 0
        
        for score, confidence, weight in sentiments:
            effective_weight = weight * confidence
            weighted_score += score * effective_weight
            weighted_confidence += confidence * weight
            total_weight += effective_weight
        
        if total_weight == 0:
            return 0.0, 0.0
        
        final_score = weighted_score / total_weight
        final_confidence = weighted_confidence / sum(w for _, _, w in sentiments)
        
        return final_score, final_confidence


class SentimentAgent:
    """情绪分析Agent"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.analyzer = SentimentAnalyzer()

    async def analyze_sentiment(self, context, inputs: Dict) -> 'SentimentAnalysisResult':
        """ChainExecutor 调用的适配方法"""
        target = context.target if hasattr(context, 'target') else 'UNKNOWN'

        # 从 inputs 中提取新闻数据
        news_data = inputs.get('news_data', {})
        if isinstance(news_data, dict):
            news_items = news_data.get('news', [])
        else:
            news_items = []

        # 提取分析师数据
        analyst_data = inputs.get('analyst_data', {})

        # 提取市场数据中的当前价格
        market_data = inputs.get('market_data', {})
        current_price = market_data.get('current_price') if isinstance(market_data, dict) else None

        return await self.analyze(target, news_items, analyst_data=analyst_data, current_price=current_price)

    async def analyze(
        self,
        symbol: str,
        news_items: List[Dict[str, Any]],
        social_data: Optional[Dict[str, Any]] = None,
        analyst_data: Optional[Dict[str, Any]] = None,
        current_price: Optional[float] = None
    ) -> SentimentAnalysisResult:
        """执行情绪分析"""
        
        # 分析新闻情绪
        analyzed_news = self._analyze_news(news_items)
        news_sentiment = self._calculate_news_sentiment(analyzed_news)
        
        # 分析社交媒体
        social_sentiment = self._analyze_social(social_data) if social_data else None
        
        # 分析分析师观点
        analyst_sentiment = self._analyze_analysts(analyst_data) if analyst_data else None
        
        # 计算综合情绪
        overall_score, confidence = self._calculate_overall_sentiment(
            news_sentiment,
            analyzed_news,
            social_sentiment,
            analyst_sentiment
        )
        
        # 确定情绪级别
        sentiment_level = self._score_to_level(overall_score)
        
        # 分析情绪趋势
        sentiment_trend, momentum = self._analyze_trend(analyzed_news)
        
        # 提取关键驱动因素
        key_drivers = self._extract_key_drivers(analyzed_news, social_sentiment)
        
        # 识别风险和机会
        risks, opportunities = self._identify_risks_opportunities(
            analyzed_news, analyst_sentiment, overall_score
        )
        
        # 生成摘要
        summary = self._generate_summary(
            symbol, sentiment_level, overall_score, confidence,
            key_drivers, risks, opportunities
        )
        
        return SentimentAnalysisResult(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_sentiment=sentiment_level,
            sentiment_score=overall_score,
            confidence=confidence,
            news_sentiment=news_sentiment,
            news_volume=len(analyzed_news),
            key_news=analyzed_news[:10],  # 前10条重要新闻
            social_sentiment=social_sentiment,
            analyst_sentiment=analyst_sentiment,
            sentiment_trend=sentiment_trend,
            sentiment_momentum=momentum,
            key_drivers=key_drivers,
            risks_identified=risks,
            opportunities_identified=opportunities,
            summary=summary
        )
    
    def _analyze_news(self, news_items: List[Dict]) -> List[NewsItem]:
        """分析新闻列表"""
        analyzed = []
        
        for item in news_items:
            title = item.get("title", "")
            summary = item.get("summary", item.get("description", ""))
            
            # 分析标题和摘要的情绪
            title_score, title_conf = self.analyzer.analyze_text(title)
            summary_score, summary_conf = self.analyzer.analyze_text(summary)
            
            # 标题权重更高
            combined_score = title_score * 0.6 + summary_score * 0.4
            
            # 解析时间
            pub_time = item.get("published_at") or item.get("publishedAt")
            if isinstance(pub_time, str):
                try:
                    pub_time = datetime.fromisoformat(pub_time.replace("Z", "+00:00"))
                except:
                    pub_time = datetime.now()
            elif not pub_time:
                pub_time = datetime.now()
            
            analyzed.append(NewsItem(
                title=title,
                source=item.get("source", {}).get("name", "Unknown") if isinstance(item.get("source"), dict) else item.get("source", "Unknown"),
                published_at=pub_time,
                url=item.get("url"),
                summary=summary,
                sentiment_score=combined_score,
                relevance_score=item.get("relevance", 1.0),
                topics=item.get("topics", [])
            ))
        
        # 按时间排序
        analyzed.sort(key=lambda x: x.published_at, reverse=True)
        
        return analyzed
    
    def _calculate_news_sentiment(self, news: List[NewsItem]) -> float:
        """计算新闻整体情绪"""
        if not news:
            return 0.0
        
        # 时间衰减加权
        now = datetime.now()
        weighted_sum = 0
        weight_sum = 0
        
        for item in news:
            # 时间衰减：24小时内权重最高
            age_hours = (now - item.published_at.replace(tzinfo=None)).total_seconds() / 3600
            time_weight = math.exp(-age_hours / 48)  # 48小时半衰期
            
            # 相关性权重
            relevance_weight = item.relevance_score
            
            total_weight = time_weight * relevance_weight
            weighted_sum += item.sentiment_score * total_weight
            weight_sum += total_weight
        
        return weighted_sum / weight_sum if weight_sum > 0 else 0.0
    
    def _analyze_social(self, data: Dict) -> SocialSentiment:
        """分析社交媒体数据"""
        return SocialSentiment(
            platform=data.get("platform", "mixed"),
            mention_count=data.get("mention_count", 0),
            positive_ratio=data.get("positive_ratio", 0.33),
            negative_ratio=data.get("negative_ratio", 0.33),
            neutral_ratio=data.get("neutral_ratio", 0.34),
            trending_topics=data.get("trending_topics", []),
            influential_posts=data.get("influential_posts", []),
            volume_change_pct=data.get("volume_change_pct", 0)
        )
    
    def _analyze_analysts(self, data: Dict) -> AnalystSentiment:
        """分析分析师数据"""
        return AnalystSentiment(
            total_analysts=data.get("total", 0),
            strong_buy=data.get("strong_buy", 0),
            buy=data.get("buy", 0),
            hold=data.get("hold", 0),
            sell=data.get("sell", 0),
            strong_sell=data.get("strong_sell", 0),
            average_target=data.get("target_mean", 0),
            high_target=data.get("target_high", 0),
            low_target=data.get("target_low", 0),
            recent_upgrades=data.get("upgrades_30d", 0),
            recent_downgrades=data.get("downgrades_30d", 0)
        )
    
    def _calculate_overall_sentiment(
        self,
        news_sentiment: float,
        news: List[NewsItem],
        social: Optional[SocialSentiment],
        analyst: Optional[AnalystSentiment]
    ) -> Tuple[float, float]:
        """计算综合情绪"""
        
        sentiments = []
        
        # 新闻情绪 (权重 40%)
        news_confidence = min(0.9, len(news) / 20)  # 20条新闻达到最高置信度
        sentiments.append((news_sentiment, news_confidence, 0.4))
        
        # 社交媒体情绪 (权重 30%)
        if social:
            social_score = social.positive_ratio - social.negative_ratio
            social_confidence = min(0.8, social.mention_count / 1000)
            sentiments.append((social_score, social_confidence, 0.3))
        
        # 分析师情绪 (权重 30%)
        if analyst and analyst.total_analysts > 0:
            # 计算分析师评分
            total = analyst.total_analysts
            score = (
                analyst.strong_buy * 1.0 +
                analyst.buy * 0.5 +
                analyst.hold * 0 +
                analyst.sell * -0.5 +
                analyst.strong_sell * -1.0
            ) / total
            
            analyst_confidence = min(0.85, analyst.total_analysts / 15)
            sentiments.append((score, analyst_confidence, 0.3))
        
        return self.analyzer.aggregate_sentiments(sentiments)
    
    def _score_to_level(self, score: float) -> SentimentLevel:
        """将分数转换为情绪级别"""
        if score >= 0.6:
            return SentimentLevel.VERY_BULLISH
        elif score >= 0.2:
            return SentimentLevel.BULLISH
        elif score <= -0.6:
            return SentimentLevel.VERY_BEARISH
        elif score <= -0.2:
            return SentimentLevel.BEARISH
        else:
            return SentimentLevel.NEUTRAL
    
    def _analyze_trend(self, news: List[NewsItem]) -> Tuple[str, float]:
        """分析情绪趋势"""
        if len(news) < 5:
            return "stable", 0.0
        
        now = datetime.now()
        
        # 分成最近和较早两组
        recent = [n for n in news if (now - n.published_at.replace(tzinfo=None)).days < 3]
        older = [n for n in news if 3 <= (now - n.published_at.replace(tzinfo=None)).days < 7]
        
        if not recent or not older:
            return "stable", 0.0
        
        recent_avg = sum(n.sentiment_score for n in recent) / len(recent)
        older_avg = sum(n.sentiment_score for n in older) / len(older)
        
        diff = recent_avg - older_avg
        
        if diff > 0.2:
            return "improving", diff
        elif diff < -0.2:
            return "deteriorating", diff
        else:
            return "stable", diff
    
    def _extract_key_drivers(
        self,
        news: List[NewsItem],
        social: Optional[SocialSentiment]
    ) -> List[str]:
        """提取关键情绪驱动因素"""
        drivers = []
        
        # 从高情绪新闻提取
        for item in news[:5]:
            if abs(item.sentiment_score) > 0.5:
                drivers.append(item.title)
        
        # 从热门话题提取
        if social and social.trending_topics:
            for topic in social.trending_topics[:3]:
                drivers.append(f"热门话题: {topic}")
        
        return drivers[:5]
    
    def _identify_risks_opportunities(
        self,
        news: List[NewsItem],
        analyst: Optional[AnalystSentiment],
        overall_score: float
    ) -> Tuple[List[str], List[str]]:
        """识别风险和机会"""
        risks = []
        opportunities = []
        
        # 从负面新闻识别风险
        negative_news = [n for n in news if n.sentiment_score < -0.3]
        for item in negative_news[:3]:
            risks.append(f"负面报道: {item.title[:50]}...")
        
        # 从正面新闻识别机会
        positive_news = [n for n in news if n.sentiment_score > 0.3]
        for item in positive_news[:3]:
            opportunities.append(f"正面消息: {item.title[:50]}...")
        
        # 分析师评级相关
        if analyst:
            if analyst.recent_downgrades > analyst.recent_upgrades:
                risks.append(f"近期分析师下调评级({analyst.recent_downgrades}次)")
            if analyst.recent_upgrades > analyst.recent_downgrades:
                opportunities.append(f"近期分析师上调评级({analyst.recent_upgrades}次)")
        
        return risks, opportunities
    
    def _generate_summary(
        self,
        symbol: str,
        level: SentimentLevel,
        score: float,
        confidence: float,
        drivers: List[str],
        risks: List[str],
        opportunities: List[str]
    ) -> str:
        """生成分析摘要"""
        
        level_desc = {
            SentimentLevel.VERY_BULLISH: "非常看涨",
            SentimentLevel.BULLISH: "看涨",
            SentimentLevel.NEUTRAL: "中性",
            SentimentLevel.BEARISH: "看跌",
            SentimentLevel.VERY_BEARISH: "非常看跌"
        }
        
        summary = f"{symbol} 当前市场情绪{level_desc[level]}（情绪分数: {score:.2f}，置信度: {confidence:.0%}）。"
        
        if drivers:
            summary += f" 主要情绪驱动因素包括: {drivers[0][:30]}..."
        
        if risks:
            summary += f" 需关注风险: 共识别{len(risks)}个潜在风险点。"
        
        if opportunities:
            summary += f" 潜在机会: 共识别{len(opportunities)}个积极信号。"
        
        return summary


# ============ Risk Agent ============

class RiskCategory(Enum):
    MARKET = "market"
    BUSINESS = "business"
    FINANCIAL = "financial"
    REGULATORY = "regulatory"
    OPERATIONAL = "operational"
    STRATEGIC = "strategic"
    ESG = "esg"
    GEOPOLITICAL = "geopolitical"


class RiskSeverity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class RiskFactor:
    """风险因子"""
    name: str
    category: RiskCategory
    severity: RiskSeverity
    probability: float  # 0-1
    impact: float  # 0-1
    description: str
    mitigation: Optional[str] = None
    monitoring_metrics: List[str] = field(default_factory=list)


@dataclass
class StressScenario:
    """压力测试场景"""
    name: str
    description: str
    probability: float
    
    # 影响
    price_impact_pct: float
    revenue_impact_pct: float
    margin_impact_pct: float
    
    # 触发条件
    triggers: List[str]
    
    # 历史参考
    historical_precedent: Optional[str] = None


@dataclass
class RiskAssessmentResult:
    """风险评估结果"""
    symbol: str
    timestamp: datetime
    
    # 综合风险评分
    overall_risk_score: float  # 0-100
    risk_rating: str  # "Very High", "High", "Moderate", "Low", "Very Low"
    
    # 各类风险
    risk_factors: List[RiskFactor]
    risk_by_category: Dict[RiskCategory, float]
    
    # 压力测试
    stress_scenarios: List[StressScenario]
    
    # 风险指标
    beta: Optional[float]
    volatility_30d: Optional[float]
    max_drawdown: Optional[float]
    var_95: Optional[float]  # 95% Value at Risk
    sharpe_ratio: Optional[float]
    
    # 财务风险指标
    debt_to_equity: Optional[float]
    current_ratio: Optional[float]
    interest_coverage: Optional[float]
    
    # 关键风险摘要
    top_risks: List[str]
    risk_mitigation_suggestions: List[str]
    
    # 分析摘要
    summary: str


class RiskAgent:
    """风险评估Agent"""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}

    async def comprehensive_risk_assessment(self, context, inputs: Dict) -> 'RiskAssessmentResult':
        """ChainExecutor 调用的适配方法"""
        target = context.target if hasattr(context, 'target') else 'UNKNOWN'

        # 从 inputs 中提取数据
        market_data = inputs.get('market_data', {})
        financial_data = inputs.get('financial_data', {})
        news_data = inputs.get('news_data', {})

        # 提取价格历史数据
        if isinstance(market_data, dict) and 'price_history' in market_data:
            price_data = market_data['price_history']
        else:
            price_data = {'close': [], 'high': [], 'low': []}

        # 提取新闻列表
        news_items = news_data.get('news', []) if isinstance(news_data, dict) else []

        return await self.analyze(target, price_data, financial_data, market_data, news_items)

    async def analyze(
        self,
        symbol: str,
        price_data: Dict[str, List[float]],
        financial_data: Optional[Dict] = None,
        market_data: Optional[Dict] = None,
        news_data: Optional[List[Dict]] = None
    ) -> RiskAssessmentResult:
        """执行风险评估"""
        
        # 计算风险指标
        risk_metrics = self._calculate_risk_metrics(price_data)
        
        # 分析财务风险
        financial_risks = self._analyze_financial_risks(financial_data) if financial_data else []
        
        # 分析市场风险
        market_risks = self._analyze_market_risks(price_data, market_data)
        
        # 分析业务风险
        business_risks = self._analyze_business_risks(financial_data, news_data)
        
        # 合并所有风险因子
        all_risks = financial_risks + market_risks + business_risks
        
        # 按类别汇总风险
        risk_by_category = self._aggregate_risks_by_category(all_risks)
        
        # 生成压力测试场景
        stress_scenarios = self._generate_stress_scenarios(
            symbol, price_data, financial_data
        )
        
        # 计算综合风险评分
        overall_score = self._calculate_overall_risk_score(all_risks, risk_metrics)
        risk_rating = self._score_to_rating(overall_score)
        
        # 提取关键风险
        top_risks = [r.description for r in sorted(
            all_risks,
            key=lambda x: x.probability * x.impact,
            reverse=True
        )[:5]]
        
        # 生成缓解建议
        mitigation_suggestions = self._generate_mitigation_suggestions(all_risks)
        
        # 生成摘要
        summary = self._generate_summary(
            symbol, overall_score, risk_rating, top_risks
        )
        
        return RiskAssessmentResult(
            symbol=symbol,
            timestamp=datetime.now(),
            overall_risk_score=overall_score,
            risk_rating=risk_rating,
            risk_factors=all_risks,
            risk_by_category=risk_by_category,
            stress_scenarios=stress_scenarios,
            beta=risk_metrics.get("beta"),
            volatility_30d=risk_metrics.get("volatility"),
            max_drawdown=risk_metrics.get("max_drawdown"),
            var_95=risk_metrics.get("var_95"),
            sharpe_ratio=risk_metrics.get("sharpe"),
            debt_to_equity=financial_data.get("debt_to_equity") if financial_data else None,
            current_ratio=financial_data.get("current_ratio") if financial_data else None,
            interest_coverage=financial_data.get("interest_coverage") if financial_data else None,
            top_risks=top_risks,
            risk_mitigation_suggestions=mitigation_suggestions,
            summary=summary
        )
    
    def _calculate_risk_metrics(self, price_data: Dict[str, List[float]]) -> Dict[str, float]:
        """计算风险指标"""
        closes = price_data.get("close", [])
        
        if len(closes) < 30:
            return {}
        
        # 计算收益率
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        
        # 波动率 (年化)
        import statistics
        volatility = statistics.stdev(returns[-30:]) * (252 ** 0.5) if len(returns) >= 30 else None
        
        # 最大回撤
        peak = closes[0]
        max_drawdown = 0
        for price in closes:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # VaR (95%)
        if returns:
            sorted_returns = sorted(returns)
            var_idx = int(len(sorted_returns) * 0.05)
            var_95 = -sorted_returns[var_idx] if var_idx < len(sorted_returns) else None
        else:
            var_95 = None
        
        # Sharpe Ratio (假设无风险利率4%)
        rf_daily = 0.04 / 252
        if returns:
            excess_returns = [r - rf_daily for r in returns]
            mean_excess = statistics.mean(excess_returns)
            std_excess = statistics.stdev(excess_returns) if len(excess_returns) > 1 else 1
            sharpe = (mean_excess / std_excess) * (252 ** 0.5) if std_excess > 0 else 0
        else:
            sharpe = None
        
        return {
            "volatility": volatility,
            "max_drawdown": max_drawdown,
            "var_95": var_95,
            "sharpe": sharpe
        }
    
    def _analyze_financial_risks(self, data: Dict) -> List[RiskFactor]:
        """分析财务风险"""
        risks = []
        
        # 杠杆风险
        de_ratio = data.get("debt_to_equity")
        if de_ratio is not None:
            if de_ratio > 2:
                risks.append(RiskFactor(
                    name="High Leverage",
                    category=RiskCategory.FINANCIAL,
                    severity=RiskSeverity.HIGH,
                    probability=0.8,
                    impact=0.7,
                    description=f"负债权益比过高({de_ratio:.1f}x)，财务杠杆风险显著",
                    mitigation="关注公司去杠杆计划和现金流改善情况",
                    monitoring_metrics=["debt_to_equity", "interest_coverage"]
                ))
            elif de_ratio > 1:
                risks.append(RiskFactor(
                    name="Moderate Leverage",
                    category=RiskCategory.FINANCIAL,
                    severity=RiskSeverity.MEDIUM,
                    probability=0.5,
                    impact=0.4,
                    description=f"负债权益比适中({de_ratio:.1f}x)，杠杆风险可控",
                    monitoring_metrics=["debt_to_equity"]
                ))
        
        # 流动性风险
        current_ratio = data.get("current_ratio")
        if current_ratio is not None:
            if current_ratio < 1:
                risks.append(RiskFactor(
                    name="Liquidity Risk",
                    category=RiskCategory.FINANCIAL,
                    severity=RiskSeverity.HIGH,
                    probability=0.7,
                    impact=0.8,
                    description=f"流动比率低于1({current_ratio:.2f})，短期偿债能力不足",
                    mitigation="需密切关注公司融资安排和现金管理",
                    monitoring_metrics=["current_ratio", "quick_ratio", "cash"]
                ))
            elif current_ratio < 1.5:
                risks.append(RiskFactor(
                    name="Moderate Liquidity",
                    category=RiskCategory.FINANCIAL,
                    severity=RiskSeverity.MEDIUM,
                    probability=0.4,
                    impact=0.5,
                    description=f"流动比率偏低({current_ratio:.2f})，流动性需关注",
                    monitoring_metrics=["current_ratio"]
                ))
        
        # 盈利能力风险
        profit_margin = data.get("profit_margin")
        if profit_margin is not None and profit_margin < 0:
            risks.append(RiskFactor(
                name="Profitability Risk",
                category=RiskCategory.FINANCIAL,
                severity=RiskSeverity.HIGH,
                probability=0.9,
                impact=0.6,
                description=f"公司处于亏损状态(利润率{profit_margin:.1%})，盈利能力存疑",
                mitigation="需评估亏损原因和扭亏时间表",
                monitoring_metrics=["profit_margin", "operating_margin", "revenue_growth"]
            ))
        
        return risks
    
    def _analyze_market_risks(
        self,
        price_data: Dict[str, List[float]],
        market_data: Optional[Dict]
    ) -> List[RiskFactor]:
        """分析市场风险"""
        risks = []
        closes = price_data.get("close", [])
        
        if len(closes) < 50:
            return risks
        
        # 波动率风险
        import statistics
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))]
        volatility = statistics.stdev(returns[-30:]) * (252 ** 0.5) if len(returns) >= 30 else 0
        
        if volatility > 0.5:  # 年化波动率 > 50%
            risks.append(RiskFactor(
                name="High Volatility",
                category=RiskCategory.MARKET,
                severity=RiskSeverity.HIGH,
                probability=0.9,
                impact=0.6,
                description=f"股价波动剧烈(年化波动率{volatility:.1%})，市场风险较高",
                mitigation="考虑仓位管理和止损策略",
                monitoring_metrics=["volatility", "atr"]
            ))
        elif volatility > 0.3:
            risks.append(RiskFactor(
                name="Moderate Volatility",
                category=RiskCategory.MARKET,
                severity=RiskSeverity.MEDIUM,
                probability=0.7,
                impact=0.4,
                description=f"股价波动性中等(年化波动率{volatility:.1%})",
                monitoring_metrics=["volatility"]
            ))
        
        # 趋势风险
        sma_50 = sum(closes[-50:]) / 50
        sma_200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None
        current = closes[-1]
        
        if sma_200 and current < sma_200:
            risks.append(RiskFactor(
                name="Downtrend Risk",
                category=RiskCategory.MARKET,
                severity=RiskSeverity.MEDIUM,
                probability=0.6,
                impact=0.5,
                description="股价低于200日均线，处于长期下降趋势",
                monitoring_metrics=["sma_200", "price_trend"]
            ))
        
        # 估值风险（如果有市场数据）
        if market_data:
            pe_ratio = market_data.get("pe_ratio")
            sector_pe = market_data.get("sector_pe", 20)
            
            if pe_ratio and pe_ratio > sector_pe * 1.5:
                risks.append(RiskFactor(
                    name="Valuation Risk",
                    category=RiskCategory.MARKET,
                    severity=RiskSeverity.MEDIUM,
                    probability=0.5,
                    impact=0.6,
                    description=f"市盈率({pe_ratio:.1f}x)显著高于行业平均({sector_pe:.1f}x)，估值偏高",
                    mitigation="需评估高估值是否有成长性支撑",
                    monitoring_metrics=["pe_ratio", "peg_ratio"]
                ))
        
        return risks
    
    def _analyze_business_risks(
        self,
        financial_data: Optional[Dict],
        news_data: Optional[List[Dict]]
    ) -> List[RiskFactor]:
        """分析业务风险"""
        risks = []
        
        # 增长风险
        if financial_data:
            revenue_growth = financial_data.get("revenue_growth")
            if revenue_growth is not None and revenue_growth < 0:
                risks.append(RiskFactor(
                    name="Growth Decline",
                    category=RiskCategory.BUSINESS,
                    severity=RiskSeverity.MEDIUM,
                    probability=0.7,
                    impact=0.5,
                    description=f"营收同比下降({revenue_growth:.1%})，业务增长面临挑战",
                    monitoring_metrics=["revenue_growth", "customer_count"]
                ))
        
        # 从新闻识别风险
        if news_data:
            risk_keywords = {
                'lawsuit': ('法律诉讼风险', RiskCategory.REGULATORY),
                'investigation': ('调查风险', RiskCategory.REGULATORY),
                'recall': ('产品召回风险', RiskCategory.OPERATIONAL),
                'layoff': ('裁员重组风险', RiskCategory.OPERATIONAL),
                'competition': ('竞争加剧风险', RiskCategory.STRATEGIC),
                'regulation': ('监管政策风险', RiskCategory.REGULATORY)
            }
            
            for news in news_data[:20]:
                title = news.get("title", "").lower()
                for keyword, (risk_name, category) in risk_keywords.items():
                    if keyword in title:
                        risks.append(RiskFactor(
                            name=risk_name,
                            category=category,
                            severity=RiskSeverity.MEDIUM,
                            probability=0.5,
                            impact=0.4,
                            description=f"新闻报道涉及{risk_name}: {news.get('title', '')[:50]}...",
                            monitoring_metrics=["news_sentiment"]
                        ))
                        break
        
        return risks
    
    def _aggregate_risks_by_category(
        self,
        risks: List[RiskFactor]
    ) -> Dict[RiskCategory, float]:
        """按类别汇总风险"""
        category_scores = {}
        
        for category in RiskCategory:
            category_risks = [r for r in risks if r.category == category]
            if category_risks:
                # 计算该类别的加权风险分数
                score = sum(r.probability * r.impact for r in category_risks) / len(category_risks)
                category_scores[category] = score * 100
            else:
                category_scores[category] = 0
        
        return category_scores
    
    def _generate_stress_scenarios(
        self,
        symbol: str,
        price_data: Dict[str, List[float]],
        financial_data: Optional[Dict]
    ) -> List[StressScenario]:
        """生成压力测试场景"""
        scenarios = []
        
        # 场景1: 经济衰退
        scenarios.append(StressScenario(
            name="Economic Recession",
            description="宏观经济衰退，消费和投资大幅下降",
            probability=0.15,
            price_impact_pct=-35,
            revenue_impact_pct=-20,
            margin_impact_pct=-5,
            triggers=["GDP连续两季度负增长", "失业率超过7%", "消费者信心指数大幅下降"],
            historical_precedent="2008年金融危机期间，标普500下跌约50%"
        ))
        
        # 场景2: 行业危机
        scenarios.append(StressScenario(
            name="Sector Downturn",
            description="所在行业遭遇系统性下行压力",
            probability=0.2,
            price_impact_pct=-25,
            revenue_impact_pct=-15,
            margin_impact_pct=-8,
            triggers=["行业监管收紧", "主要竞争对手激进定价", "需求周期性下降"]
        ))
        
        # 场景3: 利率急剧上升
        scenarios.append(StressScenario(
            name="Rate Shock",
            description="利率快速上升导致估值压缩和融资成本上升",
            probability=0.1,
            price_impact_pct=-20,
            revenue_impact_pct=-5,
            margin_impact_pct=-3,
            triggers=["联储加息超预期", "通胀失控", "10年期国债收益率突破5%"],
            historical_precedent="2022年加息周期中成长股普遍下跌30-50%"
        ))
        
        # 场景4: 公司特定危机
        scenarios.append(StressScenario(
            name="Company Crisis",
            description="公司遭遇重大负面事件（财务造假、产品问题、管理层丑闻等）",
            probability=0.05,
            price_impact_pct=-50,
            revenue_impact_pct=-30,
            margin_impact_pct=-15,
            triggers=["财报重大负面意外", "产品安全问题", "关键高管离职", "重大诉讼"]
        ))
        
        return scenarios
    
    def _calculate_overall_risk_score(
        self,
        risks: List[RiskFactor],
        metrics: Dict[str, float]
    ) -> float:
        """计算综合风险评分"""
        
        # 基于风险因子计算
        if risks:
            risk_score = sum(r.probability * r.impact for r in risks) / len(risks) * 100
        else:
            risk_score = 30  # 基础风险
        
        # 基于波动率调整
        volatility = metrics.get("volatility")
        if volatility:
            vol_adjustment = min(20, volatility * 30)  # 波动率贡献最多20分
            risk_score += vol_adjustment
        
        # 基于最大回撤调整
        max_dd = metrics.get("max_drawdown")
        if max_dd:
            dd_adjustment = min(15, max_dd * 30)  # 回撤贡献最多15分
            risk_score += dd_adjustment
        
        return min(100, max(0, risk_score))
    
    def _score_to_rating(self, score: float) -> str:
        """将分数转换为评级"""
        if score >= 80:
            return "Very High"
        elif score >= 60:
            return "High"
        elif score >= 40:
            return "Moderate"
        elif score >= 20:
            return "Low"
        else:
            return "Very Low"
    
    def _generate_mitigation_suggestions(self, risks: List[RiskFactor]) -> List[str]:
        """生成风险缓解建议"""
        suggestions = []
        
        high_risks = [r for r in risks if r.severity in [RiskSeverity.CRITICAL, RiskSeverity.HIGH]]
        
        if high_risks:
            suggestions.append("建议控制仓位，避免过度集中")
            suggestions.append("设置合理止损位，控制下行风险")
        
        # 根据具体风险类型给出建议
        categories = set(r.category for r in high_risks)
        
        if RiskCategory.FINANCIAL in categories:
            suggestions.append("密切关注公司现金流和债务情况")
        
        if RiskCategory.MARKET in categories:
            suggestions.append("考虑使用期权等工具对冲市场风险")
        
        if RiskCategory.REGULATORY in categories:
            suggestions.append("跟踪监管政策变化和合规动态")
        
        return suggestions[:5]
    
    def _generate_summary(
        self,
        symbol: str,
        score: float,
        rating: str,
        top_risks: List[str]
    ) -> str:
        """生成风险评估摘要"""
        
        summary = f"{symbol} 综合风险评分为 {score:.0f}/100，风险等级: {rating}。"
        
        if top_risks:
            summary += f" 主要风险因素包括: {top_risks[0][:40]}..."
        
        if rating in ["Very High", "High"]:
            summary += " 投资需谨慎，建议严格控制风险敞口。"
        elif rating == "Moderate":
            summary += " 风险水平适中，建议保持合理仓位并设置止损。"
        else:
            summary += " 风险水平较低，但仍需持续监控市场变化。"
        
        return summary
