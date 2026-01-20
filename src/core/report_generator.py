"""
Report Generator Module

Generates professional terminal summaries and Markdown reports with bilingual support (en/zh).
Focuses on accuracy, depth, and clear data source attribution.
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import math


# Bilingual text templates
TEXTS = {
    'en': {
        # Header
        'title': 'Investment Analysis Report',
        'executive_summary': 'Executive Summary',
        'analysis_summary': 'Analysis Summary',
        'target': 'Target',
        'analysis_date': 'Analysis Date',
        'analysis_chain': 'Analysis Chain',
        'data_sources': 'Data Sources',

        # Market Data
        'market_data': 'Market Overview',
        'price_performance': 'Price Performance',
        'current_price': 'Current Price',
        'market_cap': 'Market Cap',
        'enterprise_value': 'Enterprise Value',
        'pe_ratio': 'P/E Ratio (TTM)',
        'forward_pe': 'Forward P/E',
        'peg_ratio': 'PEG Ratio',
        'ps_ratio': 'P/S Ratio',
        'pb_ratio': 'P/B Ratio',
        'ev_ebitda': 'EV/EBITDA',
        'dividend_yield': 'Dividend Yield',
        '52w_high': '52-Week High',
        '52w_low': '52-Week Low',
        '52w_range': '52-Week Range',
        'from_52w_high': 'From 52W High',
        'from_52w_low': 'From 52W Low',
        'avg_volume': 'Avg Volume (10D)',
        'beta': 'Beta',
        'shares_outstanding': 'Shares Outstanding',

        # Price changes
        'price_change_1d': '1 Day',
        'price_change_1w': '1 Week',
        'price_change_1m': '1 Month',
        'price_change_3m': '3 Months',
        'price_change_ytd': 'YTD',
        'price_change_1y': '1 Year',

        # Financial Data
        'financial_data': 'Financial Highlights',
        'income_metrics': 'Income Metrics',
        'revenue': 'Revenue (TTM)',
        'revenue_growth': 'Revenue Growth (YoY)',
        'gross_profit': 'Gross Profit',
        'gross_margin': 'Gross Margin',
        'operating_income': 'Operating Income',
        'operating_margin': 'Operating Margin',
        'net_income': 'Net Income',
        'net_margin': 'Net Margin',
        'eps': 'EPS (TTM)',
        'eps_growth': 'EPS Growth (YoY)',

        # Cash Flow
        'cash_flow': 'Cash Flow',
        'operating_cf': 'Operating Cash Flow',
        'fcf': 'Free Cash Flow',
        'fcf_yield': 'FCF Yield',
        'capex': 'Capital Expenditure',

        # Balance Sheet
        'balance_sheet': 'Balance Sheet',
        'total_cash': 'Cash & Equivalents',
        'total_debt': 'Total Debt',
        'net_debt': 'Net Debt',
        'debt_equity': 'Debt/Equity',
        'current_ratio': 'Current Ratio',
        'book_value': 'Book Value/Share',

        # Valuation
        'valuation_analysis': 'Valuation Analysis',
        'valuation_metrics': 'Valuation Metrics',
        'fair_value': 'Estimated Fair Value',
        'upside_downside': 'Upside/Downside',
        'valuation_rating': 'Valuation Assessment',
        'undervalued': 'Potentially Undervalued',
        'fairly_valued': 'Fairly Valued',
        'overvalued': 'Potentially Overvalued',
        'sector_comparison': 'vs Sector Average',

        # Technical Analysis
        'technical_analysis': 'Technical Analysis',
        'signal': 'Overall Signal',
        'trend': 'Trend Direction',
        'confidence': 'Signal Confidence',
        'support': 'Support Levels',
        'resistance': 'Resistance Levels',
        'moving_averages': 'Moving Averages',
        'sma_20': 'SMA (20)',
        'sma_50': 'SMA (50)',
        'sma_200': 'SMA (200)',
        'ema_12': 'EMA (12)',
        'ema_26': 'EMA (26)',
        'technical_indicators': 'Technical Indicators',
        'rsi': 'RSI (14)',
        'rsi_interpretation': 'RSI Status',
        'macd': 'MACD',
        'macd_signal': 'MACD Signal',
        'macd_histogram': 'MACD Histogram',
        'bollinger_upper': 'Bollinger Upper',
        'bollinger_lower': 'Bollinger Lower',
        'atr': 'ATR (14)',
        'volume_trend': 'Volume Trend',

        # Signal/Trend translations
        'STRONG_BUY': 'STRONG BUY',
        'BUY': 'BUY',
        'NEUTRAL': 'NEUTRAL',
        'SELL': 'SELL',
        'STRONG_SELL': 'STRONG SELL',
        'STRONG_BULLISH': 'STRONG BULLISH',
        'BULLISH': 'BULLISH',
        'SIDEWAYS': 'SIDEWAYS',
        'BEARISH': 'BEARISH',
        'STRONG_BEARISH': 'STRONG BEARISH',

        # RSI interpretations
        'rsi_oversold': 'Oversold (<30)',
        'rsi_neutral': 'Neutral (30-70)',
        'rsi_overbought': 'Overbought (>70)',

        # Risk Assessment
        'risk_assessment': 'Risk Assessment',
        'overall_risk': 'Overall Risk Level',
        'risk_factors': 'Key Risk Factors',
        'risk_high': 'HIGH',
        'risk_medium': 'MEDIUM',
        'risk_low': 'LOW',
        'volatility_risk': 'Volatility Risk',
        'liquidity_risk': 'Liquidity Risk',
        'market_risk': 'Market Risk',

        # Investment Considerations
        'investment_considerations': 'Investment Considerations',
        'strengths': 'Strengths',
        'weaknesses': 'Weaknesses',
        'opportunities': 'Opportunities',
        'threats': 'Threats',

        # Recommendation
        'recommendation': 'Investment Thesis',
        'action': 'Recommended Action',
        'position_size': 'Position Sizing',
        'time_horizon': 'Time Horizon',
        'price_targets': 'Price Targets',
        'stop_loss': 'Stop Loss',

        # Footer
        'report_saved': 'Full report saved to',
        'disclaimer': 'DISCLAIMER: This report is generated by an automated system for educational and informational purposes only. It does not constitute investment advice, financial advice, or any recommendation to buy, sell, or hold any security. Always conduct your own research and consult with a qualified financial advisor before making investment decisions. Past performance does not guarantee future results.',
        'generated_by': 'Generated by FinMind',
        'data_source_note': 'Market data provided by Yahoo Finance. Data may be delayed.',

        # Confidence interpretation
        'confidence_high': 'High confidence - Strong signal alignment',
        'confidence_medium': 'Medium confidence - Some indicator divergence',
        'confidence_low': 'Low confidence - Significant indicator conflicts',
        'confidence_very_low': 'Very low confidence - Highly uncertain market conditions',
    },
    'zh': {
        # Header
        'title': '投资分析报告',
        'executive_summary': '摘要',
        'analysis_summary': '分析摘要',
        'target': '标的',
        'analysis_date': '分析日期',
        'analysis_chain': '分析链',
        'data_sources': '数据来源',

        # Market Data
        'market_data': '市场概览',
        'price_performance': '价格表现',
        'current_price': '当前价格',
        'market_cap': '市值',
        'enterprise_value': '企业价值',
        'pe_ratio': '市盈率 (TTM)',
        'forward_pe': '预期市盈率',
        'peg_ratio': 'PEG比率',
        'ps_ratio': '市销率',
        'pb_ratio': '市净率',
        'ev_ebitda': 'EV/EBITDA',
        'dividend_yield': '股息率',
        '52w_high': '52周最高',
        '52w_low': '52周最低',
        '52w_range': '52周区间',
        'from_52w_high': '距52周高点',
        'from_52w_low': '距52周低点',
        'avg_volume': '平均成交量 (10日)',
        'beta': 'Beta系数',
        'shares_outstanding': '流通股数',

        # Price changes
        'price_change_1d': '1日',
        'price_change_1w': '1周',
        'price_change_1m': '1月',
        'price_change_3m': '3月',
        'price_change_ytd': '年初至今',
        'price_change_1y': '1年',

        # Financial Data
        'financial_data': '财务亮点',
        'income_metrics': '收入指标',
        'revenue': '营收 (TTM)',
        'revenue_growth': '营收增长 (同比)',
        'gross_profit': '毛利润',
        'gross_margin': '毛利率',
        'operating_income': '营业利润',
        'operating_margin': '营业利润率',
        'net_income': '净利润',
        'net_margin': '净利率',
        'eps': '每股收益 (TTM)',
        'eps_growth': 'EPS增长 (同比)',

        # Cash Flow
        'cash_flow': '现金流',
        'operating_cf': '经营现金流',
        'fcf': '自由现金流',
        'fcf_yield': 'FCF收益率',
        'capex': '资本支出',

        # Balance Sheet
        'balance_sheet': '资产负债表',
        'total_cash': '现金及等价物',
        'total_debt': '总债务',
        'net_debt': '净债务',
        'debt_equity': '资产负债率',
        'current_ratio': '流动比率',
        'book_value': '每股账面价值',

        # Valuation
        'valuation_analysis': '估值分析',
        'valuation_metrics': '估值指标',
        'fair_value': '估算公允价值',
        'upside_downside': '上涨/下跌空间',
        'valuation_rating': '估值评估',
        'undervalued': '可能被低估',
        'fairly_valued': '估值合理',
        'overvalued': '可能被高估',
        'sector_comparison': '对比行业平均',

        # Technical Analysis
        'technical_analysis': '技术分析',
        'signal': '综合信号',
        'trend': '趋势方向',
        'confidence': '信号置信度',
        'support': '支撑位',
        'resistance': '阻力位',
        'moving_averages': '均线系统',
        'sma_20': 'SMA (20日)',
        'sma_50': 'SMA (50日)',
        'sma_200': 'SMA (200日)',
        'ema_12': 'EMA (12日)',
        'ema_26': 'EMA (26日)',
        'technical_indicators': '技术指标',
        'rsi': 'RSI (14)',
        'rsi_interpretation': 'RSI状态',
        'macd': 'MACD',
        'macd_signal': 'MACD信号线',
        'macd_histogram': 'MACD柱状图',
        'bollinger_upper': '布林上轨',
        'bollinger_lower': '布林下轨',
        'atr': 'ATR (14)',
        'volume_trend': '成交量趋势',

        # Signal/Trend translations
        'STRONG_BUY': '强烈买入',
        'BUY': '买入',
        'NEUTRAL': '中性',
        'SELL': '卖出',
        'STRONG_SELL': '强烈卖出',
        'STRONG_BULLISH': '强烈看涨',
        'BULLISH': '看涨',
        'SIDEWAYS': '横盘震荡',
        'BEARISH': '看跌',
        'STRONG_BEARISH': '强烈看跌',

        # RSI interpretations
        'rsi_oversold': '超卖 (<30)',
        'rsi_neutral': '中性 (30-70)',
        'rsi_overbought': '超买 (>70)',

        # Risk Assessment
        'risk_assessment': '风险评估',
        'overall_risk': '整体风险等级',
        'risk_factors': '关键风险因素',
        'risk_high': '高',
        'risk_medium': '中',
        'risk_low': '低',
        'volatility_risk': '波动性风险',
        'liquidity_risk': '流动性风险',
        'market_risk': '市场风险',

        # Investment Considerations
        'investment_considerations': '投资考量',
        'strengths': '优势',
        'weaknesses': '劣势',
        'opportunities': '机会',
        'threats': '威胁',

        # Recommendation
        'recommendation': '投资论点',
        'action': '建议操作',
        'position_size': '仓位建议',
        'time_horizon': '投资期限',
        'price_targets': '目标价位',
        'stop_loss': '止损位',

        # Footer
        'report_saved': '完整报告已保存至',
        'disclaimer': '免责声明：本报告由自动化系统生成，仅供教育和信息参考。不构成投资建议、财务建议或任何买卖证券的推荐。在做出投资决策前，请务必进行独立研究并咨询合格的财务顾问。过往表现不代表未来收益。',
        'generated_by': '由 FinMind 生成',
        'data_source_note': '市场数据由 Yahoo Finance 提供，数据可能存在延迟。',

        # Confidence interpretation
        'confidence_high': '高置信度 - 指标信号一致',
        'confidence_medium': '中等置信度 - 部分指标存在分歧',
        'confidence_low': '低置信度 - 指标信号明显冲突',
        'confidence_very_low': '极低置信度 - 市场状况高度不确定',
    }
}


def get_text(key: str, lang: str = 'en') -> str:
    """Get translated text"""
    return TEXTS.get(lang, TEXTS['en']).get(key, key)


def format_number(value: float, prefix: str = '', suffix: str = '', decimals: int = 2) -> str:
    """Format number with optional prefix/suffix"""
    if value is None:
        return 'N/A'
    if abs(value) >= 1e12:
        return f"{prefix}{value/1e12:.{decimals}f}T{suffix}"
    elif abs(value) >= 1e9:
        return f"{prefix}{value/1e9:.{decimals}f}B{suffix}"
    elif abs(value) >= 1e6:
        return f"{prefix}{value/1e6:.{decimals}f}M{suffix}"
    elif abs(value) >= 1e3:
        return f"{prefix}{value/1e3:.{decimals}f}K{suffix}"
    else:
        return f"{prefix}{value:,.{decimals}f}{suffix}"


def format_percent(value: float, decimals: int = 2, show_sign: bool = True) -> str:
    """Format as percentage"""
    if value is None:
        return 'N/A'
    sign = '+' if value > 0 and show_sign else ''
    return f"{sign}{value:.{decimals}f}%"


def format_price(value: float) -> str:
    """Format as price"""
    if value is None:
        return 'N/A'
    return f"${value:,.2f}"


def translate_signal(signal: str, lang: str = 'en') -> str:
    """Translate signal strength"""
    if hasattr(signal, 'value'):
        signal = signal.value
    elif hasattr(signal, 'name'):
        signal = signal.name
    signal_str = str(signal).upper()

    signal_map = {
        'SIGNALSTRENGTH.STRONG_BUY': 'STRONG_BUY',
        'SIGNALSTRENGTH.BUY': 'BUY',
        'SIGNALSTRENGTH.NEUTRAL': 'NEUTRAL',
        'SIGNALSTRENGTH.SELL': 'SELL',
        'SIGNALSTRENGTH.STRONG_SELL': 'STRONG_SELL',
    }

    for k, v in signal_map.items():
        if k in signal_str:
            signal_str = v
            break

    return get_text(signal_str, lang)


def translate_trend(trend: str, lang: str = 'en') -> str:
    """Translate trend direction"""
    if hasattr(trend, 'value'):
        trend = trend.value
    elif hasattr(trend, 'name'):
        trend = trend.name
    trend_str = str(trend).upper()

    trend_map = {
        'TRENDDIRECTION.STRONG_BULLISH': 'STRONG_BULLISH',
        'TRENDDIRECTION.BULLISH': 'BULLISH',
        'TRENDDIRECTION.SIDEWAYS': 'SIDEWAYS',
        'TRENDDIRECTION.BEARISH': 'BEARISH',
        'TRENDDIRECTION.STRONG_BEARISH': 'STRONG_BEARISH',
    }

    for k, v in trend_map.items():
        if k in trend_str:
            trend_str = v
            break

    return get_text(trend_str, lang)


def get_confidence_interpretation(confidence: float, lang: str = 'en') -> str:
    """Get interpretation of confidence score"""
    if confidence >= 0.7:
        return get_text('confidence_high', lang)
    elif confidence >= 0.5:
        return get_text('confidence_medium', lang)
    elif confidence >= 0.4:
        return get_text('confidence_low', lang)
    else:
        return get_text('confidence_very_low', lang)


def get_rsi_interpretation(rsi: float, lang: str = 'en') -> str:
    """Get RSI interpretation"""
    if rsi is None:
        return 'N/A'
    if rsi < 30:
        return get_text('rsi_oversold', lang)
    elif rsi > 70:
        return get_text('rsi_overbought', lang)
    else:
        return get_text('rsi_neutral', lang)


def calculate_price_position(current: float, high: float, low: float) -> Dict[str, float]:
    """Calculate price position relative to 52-week range"""
    if not all([current, high, low]) or high == low:
        return {'from_high': None, 'from_low': None, 'position': None}

    from_high = ((current - high) / high) * 100
    from_low = ((current - low) / low) * 100
    position = ((current - low) / (high - low)) * 100

    return {
        'from_high': from_high,
        'from_low': from_low,
        'position': position
    }


class ReportGenerator:
    """Generate professional analysis reports"""

    def __init__(self, lang: str = 'en'):
        self.lang = lang

    def t(self, key: str) -> str:
        """Shortcut for translation"""
        return get_text(key, self.lang)

    def extract_data(self, result) -> Dict[str, Any]:
        """Extract comprehensive data from analysis result"""
        data = {
            'target': 'UNKNOWN',
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'market_data': {},
            'financial_data': {},
            'technical': {},
            'indicators': {},
            'risk': {},
            'valuation': {},
            'recommendation': {}
        }

        if not hasattr(result, 'raw'):
            return data

        results = result.raw.get('results', {})

        # Market data
        market = results.get('market_data')
        if market:
            if hasattr(market, 'data') and market.data:
                data['market_data'] = market.data
            elif isinstance(market, dict):
                data['market_data'] = market
            elif hasattr(market, 'result') and market.result:
                data['market_data'] = market.result

        # Financial data
        fin = results.get('financial_data')
        if fin:
            if hasattr(fin, 'data') and fin.data:
                data['financial_data'] = fin.data
            elif isinstance(fin, dict):
                data['financial_data'] = fin
            elif hasattr(fin, 'result') and fin.result:
                data['financial_data'] = fin.result

        # Technical analysis
        tech = results.get('technical_view')
        if tech:
            data['technical'] = {
                'signal': getattr(tech, 'overall_signal', None),
                'trend': getattr(tech, 'trend', None),
                'confidence': getattr(tech, 'signal_confidence', None),
                'support': getattr(tech, 'support_levels', []),
                'resistance': getattr(tech, 'resistance_levels', []),
            }

            # Extract technical indicators if available
            if hasattr(tech, 'indicators'):
                data['indicators'] = tech.indicators
            elif hasattr(tech, 'rsi'):
                data['indicators'] = {
                    'rsi': getattr(tech, 'rsi', None),
                    'macd': getattr(tech, 'macd', None),
                    'macd_signal': getattr(tech, 'macd_signal', None),
                    'macd_histogram': getattr(tech, 'macd_histogram', None),
                    'sma_20': getattr(tech, 'sma_20', None),
                    'sma_50': getattr(tech, 'sma_50', None),
                    'sma_200': getattr(tech, 'sma_200', None),
                }

        # Risk assessment
        risk = results.get('risk_view')
        if risk:
            data['risk'] = {
                'overall': getattr(risk, 'overall_risk', None),
                'score': getattr(risk, 'composite_score', None),
                'factors': getattr(risk, 'risk_factors', [])
            }

        return data

    def generate_terminal_summary(self, result, target: str, chain: str = 'full_analysis') -> str:
        """Generate concise terminal summary"""
        data = self.extract_data(result)
        lines = []

        # Header
        lines.append("")
        lines.append("=" * 68)
        lines.append(f"  {target} - {self.t('analysis_summary')}")
        lines.append("=" * 68)

        # Market Data Row
        md = data['market_data']
        price = md.get('current_price')
        mcap = md.get('market_cap')
        pe = md.get('pe_ratio')

        price_str = format_price(price)
        mcap_str = format_number(mcap, '$') if mcap else "N/A"
        pe_str = f"{pe:.2f}" if pe else "N/A"

        lines.append("")
        lines.append(f"  {self.t('current_price')}: {price_str}    {self.t('market_cap')}: {mcap_str}    {self.t('pe_ratio')}: {pe_str}")

        # 52-week position
        high_52w = md.get('fifty_two_week_high')
        low_52w = md.get('fifty_two_week_low')
        if price and high_52w and low_52w:
            pos = calculate_price_position(price, high_52w, low_52w)
            if pos['from_high'] is not None:
                lines.append(f"  {self.t('52w_range')}: {format_price(low_52w)} - {format_price(high_52w)} | {self.t('from_52w_high')}: {format_percent(pos['from_high'])}")

        # Technical Analysis
        tech = data['technical']
        if tech.get('signal'):
            lines.append("")
            lines.append(f"  {self.t('technical_analysis')}:")

            signal = translate_signal(tech['signal'], self.lang)
            trend = translate_trend(tech['trend'], self.lang) if tech.get('trend') else 'N/A'
            conf = tech.get('confidence')

            lines.append(f"    {self.t('signal')}: {signal}")
            lines.append(f"    {self.t('trend')}: {trend}")

            if conf is not None:
                conf_pct = f"{conf:.1%}" if conf < 1 else f"{conf:.1f}%"
                conf_interp = get_confidence_interpretation(conf if conf < 1 else conf/100, self.lang)
                lines.append(f"    {self.t('confidence')}: {conf_pct} ({conf_interp})")

        # Risk Assessment
        risk = data['risk']
        if risk.get('overall'):
            lines.append("")
            lines.append(f"  {self.t('risk_assessment')}:")
            lines.append(f"    {self.t('overall_risk')}: {risk['overall']}")

        # Footer with data source
        lines.append("")
        lines.append("-" * 68)
        lines.append(f"  {self.t('data_source_note')}")
        lines.append(f"  {self.t('analysis_date')}: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 68)

        return "\n".join(lines)

    def generate_markdown_report(self, result, target: str, chain: str = 'full_analysis') -> str:
        """Generate comprehensive professional Markdown report"""
        data = self.extract_data(result)
        lines = []

        # Header
        lines.append(f"# {target} - {self.t('title')}")
        lines.append("")
        lines.append(f"**{self.t('analysis_date')}**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Executive Summary Box
        lines.append(f"## {self.t('executive_summary')}")
        lines.append("")
        md = data['market_data']
        tech = data['technical']

        if md.get('current_price'):
            lines.append(f"**{target}** is currently trading at **{format_price(md.get('current_price'))}** "
                        f"with a market capitalization of **{format_number(md.get('market_cap'), '$')}**.")

            if tech.get('signal'):
                signal = translate_signal(tech['signal'], self.lang)
                trend = translate_trend(tech['trend'], self.lang) if tech.get('trend') else ''
                conf = tech.get('confidence', 0)
                conf_val = conf if conf < 1 else conf / 100

                lines.append("")
                lines.append(f"Technical analysis indicates a **{signal}** signal with a **{trend}** trend direction. "
                            f"Signal confidence is **{conf_val:.1%}** ({get_confidence_interpretation(conf_val, self.lang).lower()}).")
        lines.append("")
        lines.append("---")
        lines.append("")

        # Market Overview Section
        if md:
            lines.append(f"## {self.t('market_data')}")
            lines.append("")

            # Price and Valuation Table
            lines.append("### Price & Valuation Metrics")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|:-------|------:|")

            metrics = [
                ('current_price', md.get('current_price'), format_price),
                ('market_cap', md.get('market_cap'), lambda x: format_number(x, '$')),
                ('pe_ratio', md.get('pe_ratio'), lambda x: f"{x:.2f}" if x else 'N/A'),
                ('forward_pe', md.get('forward_pe'), lambda x: f"{x:.2f}" if x else 'N/A'),
                ('ps_ratio', md.get('ps_ratio'), lambda x: f"{x:.2f}" if x else 'N/A'),
                ('pb_ratio', md.get('pb_ratio'), lambda x: f"{x:.2f}" if x else 'N/A'),
                ('dividend_yield', md.get('dividend_yield'), lambda x: f"{x:.2f}%" if x else 'N/A'),
                ('beta', md.get('beta'), lambda x: f"{x:.3f}" if x else 'N/A'),
            ]

            for key, value, formatter in metrics:
                if value is not None:
                    lines.append(f"| {self.t(key)} | {formatter(value)} |")

            lines.append("")

            # 52-Week Range
            high_52w = md.get('fifty_two_week_high')
            low_52w = md.get('fifty_two_week_low')
            price = md.get('current_price')

            if all([high_52w, low_52w, price]):
                lines.append("### 52-Week Trading Range")
                lines.append("")
                pos = calculate_price_position(price, high_52w, low_52w)

                lines.append(f"| Metric | Value |")
                lines.append("|:-------|------:|")
                lines.append(f"| {self.t('52w_high')} | {format_price(high_52w)} |")
                lines.append(f"| {self.t('52w_low')} | {format_price(low_52w)} |")
                lines.append(f"| {self.t('from_52w_high')} | {format_percent(pos['from_high'])} |")
                lines.append(f"| {self.t('from_52w_low')} | {format_percent(pos['from_low'])} |")
                lines.append(f"| Range Position | {pos['position']:.1f}% |")
                lines.append("")

        # Financial Data Section
        fd = data['financial_data']
        if fd:
            lines.append(f"## {self.t('financial_data')}")
            lines.append("")
            lines.append("| Metric | Value |")
            lines.append("|:-------|------:|")

            fin_metrics = [
                ('revenue', fd.get('revenue'), lambda x: format_number(x, '$')),
                ('eps', fd.get('eps'), lambda x: f"${x:.2f}" if x else 'N/A'),
                ('fcf', fd.get('latest_fcf'), lambda x: format_number(x, '$')),
                ('beta', fd.get('beta'), lambda x: f"{x:.3f}" if x else 'N/A'),
            ]

            for key, value, formatter in fin_metrics:
                if value is not None:
                    lines.append(f"| {self.t(key)} | {formatter(value)} |")

            lines.append("")

        # Technical Analysis Section
        tech = data['technical']
        if tech.get('signal'):
            lines.append(f"## {self.t('technical_analysis')}")
            lines.append("")

            # Signal Summary
            lines.append("### Signal Summary")
            lines.append("")
            lines.append("| Indicator | Value | Interpretation |")
            lines.append("|:----------|:-----:|:---------------|")

            signal = translate_signal(tech['signal'], self.lang)
            lines.append(f"| **{self.t('signal')}** | {signal} | Primary trading signal |")

            if tech.get('trend'):
                trend = translate_trend(tech['trend'], self.lang)
                lines.append(f"| **{self.t('trend')}** | {trend} | Overall price direction |")

            conf = tech.get('confidence')
            if conf is not None:
                conf_val = conf if conf < 1 else conf / 100
                conf_pct = f"{conf_val:.1%}"
                conf_interp = get_confidence_interpretation(conf_val, self.lang)
                lines.append(f"| **{self.t('confidence')}** | {conf_pct} | {conf_interp} |")

            lines.append("")

            # Support/Resistance
            if tech.get('support') or tech.get('resistance'):
                lines.append(f"### {self.t('support')} & {self.t('resistance')}")
                lines.append("")

                if tech.get('support'):
                    try:
                        support_list = tech['support'][:3] if isinstance(tech['support'], list) else []
                        support_vals = []
                        for s in support_list:
                            if hasattr(s, 'price'):
                                support_vals.append(f"${s.price:.2f}")
                            elif isinstance(s, (int, float)):
                                support_vals.append(f"${s:.2f}")
                        if support_vals:
                            lines.append(f"- **{self.t('support')}**: {', '.join(support_vals)}")
                    except Exception:
                        pass

                if tech.get('resistance'):
                    try:
                        resistance_list = tech['resistance'][:3] if isinstance(tech['resistance'], list) else []
                        resistance_vals = []
                        for r in resistance_list:
                            if hasattr(r, 'price'):
                                resistance_vals.append(f"${r.price:.2f}")
                            elif isinstance(r, (int, float)):
                                resistance_vals.append(f"${r:.2f}")
                        if resistance_vals:
                            lines.append(f"- **{self.t('resistance')}**: {', '.join(resistance_vals)}")
                    except Exception:
                        pass
                lines.append("")

        # Risk Assessment Section
        risk = data['risk']
        if risk.get('overall'):
            lines.append(f"## {self.t('risk_assessment')}")
            lines.append("")
            lines.append(f"**{self.t('overall_risk')}**: {risk['overall']}")
            lines.append("")

        # Data Sources Section
        lines.append(f"## {self.t('data_sources')}")
        lines.append("")
        lines.append("| Source | Data Type | Notes |")
        lines.append("|:-------|:----------|:------|")
        lines.append("| Yahoo Finance | Market Data, Financials | Real-time quotes may be delayed 15-20 minutes |")
        lines.append("| Company Filings | Financial Statements | Based on latest available SEC filings |")
        lines.append("| Technical Calculations | Indicators | Computed from historical price data |")
        lines.append("")

        # Disclaimer
        lines.append("---")
        lines.append("")
        lines.append(f"### {self.t('disclaimer').split(':')[0]}")
        lines.append("")
        lines.append(f"_{self.t('disclaimer')}_")
        lines.append("")
        lines.append(f"---")
        lines.append("")
        lines.append(f"*{self.t('generated_by')} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

        return "\n".join(lines)

    def save_markdown_report(self, result, target: str, output_dir: str = 'reports',
                            chain: str = 'full_analysis') -> str:
        """Save Markdown report to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        date_str = datetime.now().strftime('%Y-%m-%d')
        filename = f"{target}_{date_str}.md"
        filepath = output_path / filename

        content = self.generate_markdown_report(result, target, chain)
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        return str(filepath)
