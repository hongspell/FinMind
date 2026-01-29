// FinMind 分析结果类型定义

export type SignalStrength = 'strong_buy' | 'buy' | 'neutral' | 'sell' | 'strong_sell';

export type TrendDirection = 'strong_bullish' | 'bullish' | 'neutral' | 'bearish' | 'strong_bearish';

export type Timeframe = 'short_term' | 'medium_term' | 'long_term';

export interface TimeframeAnalysis {
  timeframe: Timeframe;
  timeframe_label: string;
  signal: SignalStrength;
  trend: TrendDirection;
  trend_strength: number;
  confidence: number;
  key_indicators: string[];
  description: string;
}

export interface MarketData {
  current_price: number;
  regular_price?: number;
  pre_market_price?: number;
  post_market_price?: number;
  price_source?: string;
  market_session?: string;
  market_status?: string;
  market_cap?: number;
  pe_ratio?: number;
  forward_pe?: number;
  ps_ratio?: number;
  pb_ratio?: number;
  beta?: number;
  fifty_two_week_high?: number;
  fifty_two_week_low?: number;
  avg_volume?: number;
  volume?: number;
  previous_close?: number;
  change?: number;
  change_percent?: number;
}

export interface PriceHistory {
  dates: string[];
  open: number[];
  high: number[];
  low: number[];
  close: number[];
  volume: number[];
}

export interface TechnicalAnalysis {
  overall_signal: SignalStrength;
  trend: TrendDirection;
  signal_confidence: number;
  timeframe_analyses: TimeframeAnalysis[];
  support_levels: number[];
  resistance_levels: number[];
}

export interface FinancialData {
  revenue?: number;
  revenue_growth?: number;
  net_income?: number;
  eps?: number;
  free_cash_flow?: number;
  gross_margin?: number;
  operating_margin?: number;
  net_margin?: number;
  debt_to_equity?: number;
  current_ratio?: number;
}

export interface RiskAssessment {
  overall_risk: 'low' | 'medium' | 'high' | 'critical';
  risk_score: number;
  risk_factors: {
    name: string;
    level: string;
    description: string;
  }[];
}

export interface AnalysisResult {
  symbol: string;
  timestamp: string;
  market_data: MarketData;
  price_history?: PriceHistory;
  technical_analysis: TechnicalAnalysis;
  financial_data?: FinancialData;
  risk_assessment?: RiskAssessment;
  recommendation?: {
    action: string;
    target_price?: number;
    confidence: number;
    rationale: string;
  };
}

// API 响应类型
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: string;
}

// 股票搜索结果
export interface StockSearchResult {
  symbol: string;
  name: string;
  exchange: string;
  type: string;
}

// 实时报价
export interface Quote {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  timestamp: string;
}
