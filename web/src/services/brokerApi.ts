import axios from 'axios';
import type { ApiResponse } from '../types/analysis';
import type {
  BrokerType,
  BrokerConfig,
  BrokerStatus,
  BrokerSummary,
  UnifiedPortfolio,
  Position,
  PortfolioAnalysis,
  MonteCarloParams,
  PriceSimulationResult,
  PortfolioSimulationResult,
} from '../types/broker';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.detail || error.response?.data?.error || error.message || 'An error occurred';
    return Promise.reject(new Error(message));
  }
);

// 券商 API
export const brokerApi = {
  // 获取支持的券商列表
  getSupportedBrokers: async (): Promise<ApiResponse<{ brokers: Array<{ type: BrokerType; name: string; description: string }> }>> => {
    return api.get('/broker/supported');
  },

  // 连接券商
  connect: async (config: BrokerConfig): Promise<ApiResponse<{ connected: boolean; account_id?: string }>> => {
    return api.post('/broker/connect', config);
  },

  // 断开券商连接
  disconnect: async (brokerType: BrokerType): Promise<ApiResponse<{ disconnected: boolean }>> => {
    return api.delete(`/broker/disconnect/${brokerType}`);
  },

  // 获取所有券商状态
  getStatus: async (): Promise<ApiResponse<{ brokers: BrokerStatus[] }>> => {
    return api.get('/broker/status');
  },

  // 获取单个券商摘要
  getSummary: async (brokerType: BrokerType): Promise<ApiResponse<BrokerSummary>> => {
    return api.get(`/broker/summary/${brokerType}`);
  },

  // 获取券商持仓
  getPositions: async (brokerType: BrokerType): Promise<ApiResponse<{ positions: Position[] }>> => {
    return api.get(`/broker/positions/${brokerType}`);
  },

  // 获取交易历史
  getTrades: async (brokerType: BrokerType, days: number = 7): Promise<any> => {
    return api.get(`/broker/trades/${brokerType}`, { params: { days } });
  },

  // 获取统一投资组合
  getUnifiedPortfolio: async (): Promise<ApiResponse<UnifiedPortfolio>> => {
    return api.get('/broker/unified');
  },

  // 获取跨券商持仓
  getPositionAcrossBrokers: async (symbol: string): Promise<ApiResponse<{ positions: Record<BrokerType, Position | null> }>> => {
    return api.get(`/broker/position/${symbol}`);
  },

  // 设置演示环境
  setupDemo: async (): Promise<ApiResponse<{ message: string }>> => {
    return api.post('/broker/demo/setup');
  },

  // 清理演示环境
  teardownDemo: async (): Promise<ApiResponse<{ message: string }>> => {
    return api.post('/broker/demo/teardown');
  },
};

// 投资组合分析 API
export const portfolioApi = {
  // 获取投资组合分析
  analyze: async (): Promise<ApiResponse<PortfolioAnalysis>> => {
    return api.get('/portfolio/analyze');
  },

  // 获取投资组合风险指标
  getRiskMetrics: async (): Promise<ApiResponse<PortfolioAnalysis['risk_metrics']>> => {
    return api.get('/portfolio/risk');
  },

  // 获取持仓建议
  getRecommendations: async (): Promise<ApiResponse<{ recommendations: PortfolioAnalysis['recommendations'] }>> => {
    return api.get('/portfolio/recommendations');
  },

  // 获取完整的投资组合数据（合并端点 - 推荐使用）
  getFullData: async (): Promise<ApiResponse<{
    portfolio: UnifiedPortfolio;
    analysis: PortfolioAnalysis | null;
    message?: string;
  }>> => {
    return api.get('/portfolio/full');
  },
};

// 蒙特卡洛模拟 API
export const monteCarloApi = {
  // 单股票价格模拟
  simulatePrice: async (params: MonteCarloParams): Promise<ApiResponse<PriceSimulationResult>> => {
    return api.post('/monte-carlo/price', params);
  },

  // 投资组合模拟
  simulatePortfolio: async (params?: { days?: number; simulations?: number }): Promise<ApiResponse<PortfolioSimulationResult>> => {
    return api.post('/monte-carlo/portfolio', params || {});
  },

  // 获取股票历史波动率
  getVolatility: async (symbol: string, days?: number): Promise<ApiResponse<{ symbol: string; volatility: number; days: number }>> => {
    return api.get(`/monte-carlo/volatility/${symbol}`, { params: { days } });
  },
};

export default {
  broker: brokerApi,
  portfolio: portfolioApi,
  monteCarlo: monteCarloApi,
};
