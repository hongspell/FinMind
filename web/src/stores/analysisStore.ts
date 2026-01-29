import { create } from 'zustand';
import type { AnalysisResult, Quote } from '../types/analysis';
import { analysisApi } from '../services/api';

interface AnalysisState {
  // 当前分析结果
  currentAnalysis: AnalysisResult | null;
  isLoading: boolean;
  error: string | null;

  // 最近分析的股票
  recentSymbols: string[];

  // 监控列表
  watchlist: Quote[];

  // Actions
  analyze: (symbol: string) => Promise<void>;
  clearAnalysis: () => void;
  addToRecent: (symbol: string) => void;
  addToWatchlist: (symbol: string) => Promise<void>;
  removeFromWatchlist: (symbol: string) => void;
  refreshWatchlist: () => Promise<void>;
  isInWatchlist: (symbol: string) => boolean;
  toggleWatchlist: (symbol: string) => Promise<void>;
}

export const useAnalysisStore = create<AnalysisState>((set, get) => ({
  currentAnalysis: null,
  isLoading: false,
  error: null,
  recentSymbols: [],
  watchlist: [],

  analyze: async (symbol: string) => {
    set({ isLoading: true, error: null });

    try {
      const response = await analysisApi.analyze(symbol.toUpperCase());

      if (response.success && response.data) {
        set({ currentAnalysis: response.data, isLoading: false });
        get().addToRecent(symbol.toUpperCase());
      } else {
        set({ error: response.error || 'Analysis failed', isLoading: false });
      }
    } catch (error) {
      set({
        error: error instanceof Error ? error.message : 'Analysis failed',
        isLoading: false,
      });
    }
  },

  clearAnalysis: () => {
    set({ currentAnalysis: null, error: null });
  },

  addToRecent: (symbol: string) => {
    const { recentSymbols } = get();
    const updated = [symbol, ...recentSymbols.filter((s) => s !== symbol)].slice(0, 10);
    set({ recentSymbols: updated });

    // 持久化到 localStorage
    localStorage.setItem('finmind_recent', JSON.stringify(updated));
  },

  addToWatchlist: async (symbol: string) => {
    const upperSymbol = symbol.toUpperCase();
    const { watchlist } = get();

    // 如果已存在，不重复添加
    if (watchlist.find((q) => q.symbol === upperSymbol)) {
      return;
    }

    // 先添加占位数据到 watchlist 和 localStorage，确保即使 API 失败也能保存
    const placeholderQuote: Quote = {
      symbol: upperSymbol,
      price: 0,
      change: 0,
      change_percent: 0,
      volume: 0,
      timestamp: new Date().toISOString(),
    };
    const updatedWithPlaceholder = [...watchlist, placeholderQuote];
    set({ watchlist: updatedWithPlaceholder });
    localStorage.setItem('finmind_watchlist', JSON.stringify(updatedWithPlaceholder.map((q) => q.symbol)));

    // 然后尝试获取实际数据
    try {
      const response = await analysisApi.getQuote(upperSymbol);
      if (response.success && response.data) {
        // 用真实数据替换占位数据
        const updatedWithRealData = get().watchlist.map((q) =>
          q.symbol === upperSymbol ? response.data! : q
        );
        set({ watchlist: updatedWithRealData });
      }
    } catch (error) {
      console.error('Failed to fetch quote for watchlist:', error);
      // API 失败时，占位数据仍然保留在 watchlist 中
    }
  },

  removeFromWatchlist: (symbol: string) => {
    const { watchlist } = get();
    const updated = watchlist.filter((q) => q.symbol !== symbol.toUpperCase());
    set({ watchlist: updated });
    localStorage.setItem('finmind_watchlist', JSON.stringify(updated.map((q) => q.symbol)));
  },

  refreshWatchlist: async () => {
    const { watchlist } = get();
    if (watchlist.length === 0) return;

    try {
      const symbols = watchlist.map((q) => q.symbol);
      const response = await analysisApi.getQuotes(symbols);
      // 只有当返回的数据非空时才更新 watchlist，避免数据丢失
      if (response.success && response.data && response.data.length > 0) {
        // 合并数据：保留 localStorage 中的 symbol，但更新价格等信息
        const updatedWatchlist = symbols.map((symbol) => {
          const newData = response.data?.find((q) => q.symbol === symbol);
          const oldData = watchlist.find((q) => q.symbol === symbol);
          return newData || oldData;
        }).filter((q): q is Quote => q !== undefined);

        set({ watchlist: updatedWatchlist });
      }
    } catch (error) {
      console.error('Failed to refresh watchlist:', error);
      // 出错时保留现有数据，不做任何修改
    }
  },

  isInWatchlist: (symbol: string) => {
    const { watchlist } = get();
    return watchlist.some((q) => q.symbol === symbol.toUpperCase());
  },

  toggleWatchlist: async (symbol: string) => {
    const { isInWatchlist, addToWatchlist, removeFromWatchlist } = get();
    if (isInWatchlist(symbol)) {
      removeFromWatchlist(symbol);
    } else {
      await addToWatchlist(symbol);
    }
  },
}));

// 初始化时从 localStorage 加载
const initStore = async () => {
  try {
    const recent = localStorage.getItem('finmind_recent');
    if (recent) {
      useAnalysisStore.setState({ recentSymbols: JSON.parse(recent) });
    }

    // 加载 watchlist
    const watchlistSymbols = localStorage.getItem('finmind_watchlist');
    if (watchlistSymbols) {
      const symbols = JSON.parse(watchlistSymbols) as string[];
      if (symbols.length > 0) {
        // 先设置占位数据，确保即使 API 失败也能显示股票列表
        const placeholderData: Quote[] = symbols.map((symbol) => ({
          symbol,
          price: 0,
          change: 0,
          change_percent: 0,
          volume: 0,
          timestamp: new Date().toISOString(),
        }));
        useAnalysisStore.setState({ watchlist: placeholderData });

        // 获取报价数据
        try {
          const response = await analysisApi.getQuotes(symbols);
          if (response.success && response.data) {
            useAnalysisStore.setState({ watchlist: response.data });
          }
        } catch {
          console.error('Failed to load watchlist quotes, showing placeholder data');
        }
      }
    }
  } catch (e) {
    console.error('Failed to load store:', e);
  }
};

initStore();
