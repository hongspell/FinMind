import axios from 'axios';
import type { AnalysisResult, ApiResponse, Quote, StockSearchResult } from '../types/analysis';

const api = axios.create({
  baseURL: '/api',
  timeout: 120000, // 分析可能需要较长时间
  headers: {
    'Content-Type': 'application/json',
  },
});

// 请求拦截器
api.interceptors.request.use(
  (config) => {
    // 可以在这里添加 token 等
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// 响应拦截器
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const message = error.response?.data?.error || error.message || 'An error occurred';
    return Promise.reject(new Error(message));
  }
);

// 分析 API
export const analysisApi = {
  // 执行股票分析
  analyze: async (symbol: string): Promise<ApiResponse<AnalysisResult>> => {
    return api.post('/analyze', { symbol });
  },

  // 获取快速报价
  getQuote: async (symbol: string): Promise<ApiResponse<Quote>> => {
    return api.get(`/quote/${symbol}`);
  },

  // 获取多个股票报价
  getQuotes: async (symbols: string[]): Promise<ApiResponse<Quote[]>> => {
    return api.post('/quotes', { symbols });
  },

  // 搜索股票
  searchStocks: async (query: string): Promise<ApiResponse<StockSearchResult[]>> => {
    return api.get('/search', { params: { q: query } });
  },

  // 获取价格历史
  getPriceHistory: async (
    symbol: string,
    period: string = '1y'
  ): Promise<ApiResponse<{ dates: string[]; prices: number[] }>> => {
    return api.get(`/history/${symbol}`, { params: { period } });
  },
};

// 市场 API
export const marketApi = {
  // 获取市场状态
  getMarketStatus: async (): Promise<ApiResponse<{
    status: string;
    session: string;
    next_open: string;
    next_close: string;
  }>> => {
    return api.get('/market/status');
  },

  // 获取热门股票
  getTrending: async (): Promise<ApiResponse<Quote[]>> => {
    return api.get('/market/trending');
  },

  // 获取市场指数
  getIndices: async (): Promise<ApiResponse<Quote[]>> => {
    return api.get('/market/indices');
  },
};

// 配置项类型
export interface ConfigItem {
  key: string;
  value: string;
  hasValue: boolean;
  category: string;
  description: string;
  is_secret: boolean;
}

// 配置管理 API
export const configApi = {
  // 获取所有配置
  getConfig: async (): Promise<ApiResponse<{ configs: ConfigItem[] }>> => {
    return api.get('/config');
  },

  // 更新配置
  updateConfig: async (configs: { key: string; value: string }[]): Promise<ApiResponse<{ message: string }>> => {
    return api.post('/config', { configs });
  },
};

export default api;
