// 券商集成类型定义

export type BrokerType = 'ibkr' | 'futu' | 'tiger';

export interface BrokerConfig {
  broker_type: BrokerType;
  // IBKR
  ibkr_host?: string;
  ibkr_port?: number;
  ibkr_client_id?: number;
  // Futu
  futu_host?: string;
  futu_port?: number;
  // Tiger
  tiger_id?: string;
  tiger_account?: string;
  tiger_private_key?: string;
}

export interface BrokerStatus {
  broker_type: BrokerType;
  connected: boolean;
  account_id?: string;
  last_sync?: string;
  error?: string;
}

export interface Position {
  symbol: string;
  quantity: number;
  avg_cost: number;
  market_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  currency: string;
  broker?: BrokerType;
}

export interface Trade {
  symbol: string;
  market: string;
  action: 'buy' | 'sell';
  quantity: number;
  price: number;
  total_value: number;
  commission: number;
  currency: string;
  trade_time?: string;
  order_id?: string;
  execution_id?: string;
  realized_pnl?: number;
}

export interface AccountBalance {
  total_value: number;
  cash: number;
  buying_power: number;
  currency: string;
  margin_used?: number;
  day_pnl?: number;
  day_pnl_percent?: number;
}

export interface BrokerSummary {
  broker_type: BrokerType;
  connected: boolean;
  balance?: AccountBalance;
  positions?: Position[];
  total_positions?: number;
  total_market_value?: number;
  error?: string;
}

export interface PortfolioHolding {
  symbol: string;
  market?: string;
  quantity: number;
  avg_cost: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  weight: number;
}

export interface UnifiedPortfolio {
  total_value?: number;
  total_assets: number;
  total_cash: number;
  total_market_value: number;
  total_unrealized_pnl: number;
  total_unrealized_pnl_percent?: number;
  total_realized_pnl?: number;
  broker_allocation?: Record<string, number>;
  market_allocation?: Record<string, number>;
  currency_exposure?: Record<string, number>;
  top_holdings: PortfolioHolding[];
  broker_count?: number;
  position_count: number;
  brokers?: BrokerSummary[];
  positions?: Position[];
  positions_by_symbol?: Record<string, Position[]>;
  last_updated: string;
}

// 投资组合分析类型
export interface PortfolioRiskMetrics {
  var_95: number;
  var_99: number;
  cvar_95: number;
  cvar_99: number;
  volatility: number;
  sharpe_ratio: number;
  max_drawdown: number;
  beta: number;
}

export interface PositionRecommendation {
  symbol: string;
  action: 'hold' | 'reduce' | 'increase' | 'sell' | 'watch';
  reason: string;
  priority: 'low' | 'medium' | 'high';
  current_weight: number;
  suggested_weight?: number;
}

export interface PortfolioAnalysis {
  health_score: number;
  risk_score: number;
  diversification_score: number;
  risk_metrics: PortfolioRiskMetrics;
  recommendations: PositionRecommendation[];
  concentration_risk: {
    top_holding_weight: number;
    top_3_weight: number;
    top_5_weight: number;
    hhi_index: number;
  };
  sector_allocation?: Record<string, number>;
  analysis_timestamp: string;
}

// 蒙特卡洛模拟类型
export interface MonteCarloParams {
  symbol?: string;
  current_price?: number;
  annual_return?: number;
  annual_volatility?: number;
  days?: number;
  simulations?: number;
  confidence_levels?: number[];
}

export interface PriceSimulationResult {
  symbol: string;
  current_price: number;
  simulations: number;
  days: number;
  paths: number[][]; // 模拟路径（采样）
  final_prices: {
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
    percentiles: Record<number, number>;
  };
  var_values: Record<number, number>;
  cvar_values: Record<number, number>;
  probability_of_profit: number;
  expected_return: number;
}

export interface PortfolioSimulationResult {
  initial_value: number;
  simulations: number;
  days: number;
  final_values: {
    mean: number;
    median: number;
    std: number;
    min: number;
    max: number;
    percentiles: Record<number, number>;
  };
  var_values: Record<number, number>;
  cvar_values: Record<number, number>;
  sharpe_ratio: number;
  max_drawdown: number;
  probability_of_profit: number;
}
