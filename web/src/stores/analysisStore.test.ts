import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useAnalysisStore } from './analysisStore';

// Mock the API
vi.mock('../services/api', () => ({
  analysisApi: {
    analyze: vi.fn(),
    getQuote: vi.fn(),
    getQuotes: vi.fn(),
  },
}));

describe('analysisStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useAnalysisStore.setState({
      currentAnalysis: null,
      isLoading: false,
      error: null,
      recentSymbols: [],
      watchlist: [],
    });

    // Clear localStorage mock
    vi.mocked(window.localStorage.getItem).mockReturnValue(null);
    vi.mocked(window.localStorage.setItem).mockClear();
  });

  describe('analyze', () => {
    it('should set loading state when analyzing', async () => {
      const { analysisApi } = await import('../services/api');
      vi.mocked(analysisApi.analyze).mockResolvedValue({
        success: true,
        timestamp: new Date().toISOString(),
        data: {
          symbol: 'AAPL',
          timestamp: new Date().toISOString(),
          market_data: { current_price: 175 },
          technical_analysis: {
            overall_signal: 'buy',
            trend: 'bullish',
            signal_confidence: 0.8,
            support_levels: [170],
            resistance_levels: [180],
            timeframe_analyses: [],
          },
          price_history: { dates: [], open: [], high: [], low: [], close: [], volume: [] },
        },
      });

      const { analyze } = useAnalysisStore.getState();

      // Start analysis
      const analyzePromise = analyze('AAPL');

      // Should be loading
      expect(useAnalysisStore.getState().isLoading).toBe(true);

      await analyzePromise;

      // Should not be loading after completion
      expect(useAnalysisStore.getState().isLoading).toBe(false);
      expect(useAnalysisStore.getState().currentAnalysis?.symbol).toBe('AAPL');
    });

    it('should handle analysis errors', async () => {
      const { analysisApi } = await import('../services/api');
      vi.mocked(analysisApi.analyze).mockResolvedValue({
        success: false,
        error: 'Analysis failed',
        timestamp: new Date().toISOString(),
      });

      const { analyze } = useAnalysisStore.getState();
      await analyze('INVALID');

      expect(useAnalysisStore.getState().error).toBe('Analysis failed');
      expect(useAnalysisStore.getState().isLoading).toBe(false);
    });
  });

  describe('clearAnalysis', () => {
    it('should clear current analysis and error', () => {
      useAnalysisStore.setState({
        currentAnalysis: { symbol: 'AAPL' } as any,
        error: 'Some error',
      });

      const { clearAnalysis } = useAnalysisStore.getState();
      clearAnalysis();

      expect(useAnalysisStore.getState().currentAnalysis).toBeNull();
      expect(useAnalysisStore.getState().error).toBeNull();
    });
  });

  describe('addToRecent', () => {
    it('should add symbol to recent list', () => {
      const { addToRecent } = useAnalysisStore.getState();
      addToRecent('AAPL');

      expect(useAnalysisStore.getState().recentSymbols).toContain('AAPL');
    });

    it('should move existing symbol to front', () => {
      useAnalysisStore.setState({ recentSymbols: ['TSLA', 'GOOGL', 'AAPL'] });

      const { addToRecent } = useAnalysisStore.getState();
      addToRecent('AAPL');

      const { recentSymbols } = useAnalysisStore.getState();
      expect(recentSymbols[0]).toBe('AAPL');
      expect(recentSymbols.length).toBe(3);
    });

    it('should limit to 10 recent symbols', () => {
      const symbols = Array.from({ length: 12 }, (_, i) => `SYM${i}`);
      useAnalysisStore.setState({ recentSymbols: symbols.slice(0, 10) });

      const { addToRecent } = useAnalysisStore.getState();
      addToRecent('NEW');

      expect(useAnalysisStore.getState().recentSymbols.length).toBe(10);
      expect(useAnalysisStore.getState().recentSymbols[0]).toBe('NEW');
    });

    it('should persist to localStorage', () => {
      const { addToRecent } = useAnalysisStore.getState();
      addToRecent('AAPL');

      expect(window.localStorage.setItem).toHaveBeenCalledWith(
        'finmind_recent',
        expect.any(String)
      );
    });
  });

  describe('watchlist operations', () => {
    it('should add to watchlist with placeholder data', async () => {
      const { analysisApi } = await import('../services/api');
      vi.mocked(analysisApi.getQuote).mockResolvedValue({
        success: true,
        timestamp: new Date().toISOString(),
        data: {
          symbol: 'AAPL',
          price: 175,
          change: 2.5,
          change_percent: 1.45,
          volume: 50000000,
          timestamp: new Date().toISOString(),
        },
      });

      const { addToWatchlist } = useAnalysisStore.getState();
      await addToWatchlist('AAPL');

      const { watchlist } = useAnalysisStore.getState();
      expect(watchlist.length).toBe(1);
      expect(watchlist[0].symbol).toBe('AAPL');
    });

    it('should not add duplicate symbols', async () => {
      useAnalysisStore.setState({
        watchlist: [{ symbol: 'AAPL', price: 175, change: 0, change_percent: 0, volume: 0, timestamp: '' }],
      });

      const { addToWatchlist } = useAnalysisStore.getState();
      await addToWatchlist('AAPL');

      expect(useAnalysisStore.getState().watchlist.length).toBe(1);
    });

    it('should remove from watchlist', () => {
      useAnalysisStore.setState({
        watchlist: [
          { symbol: 'AAPL', price: 175, change: 0, change_percent: 0, volume: 0, timestamp: '' },
          { symbol: 'TSLA', price: 250, change: 0, change_percent: 0, volume: 0, timestamp: '' },
        ],
      });

      const { removeFromWatchlist } = useAnalysisStore.getState();
      removeFromWatchlist('AAPL');

      const { watchlist } = useAnalysisStore.getState();
      expect(watchlist.length).toBe(1);
      expect(watchlist[0].symbol).toBe('TSLA');
    });

    it('should check if symbol is in watchlist', () => {
      useAnalysisStore.setState({
        watchlist: [{ symbol: 'AAPL', price: 175, change: 0, change_percent: 0, volume: 0, timestamp: '' }],
      });

      const { isInWatchlist } = useAnalysisStore.getState();

      expect(isInWatchlist('AAPL')).toBe(true);
      expect(isInWatchlist('aapl')).toBe(true); // Should be case-insensitive
      expect(isInWatchlist('TSLA')).toBe(false);
    });

    it('should toggle watchlist', async () => {
      const { analysisApi } = await import('../services/api');
      vi.mocked(analysisApi.getQuote).mockResolvedValue({
        success: true,
        timestamp: new Date().toISOString(),
        data: {
          symbol: 'AAPL',
          price: 175,
          change: 2.5,
          change_percent: 1.45,
          volume: 50000000,
          timestamp: new Date().toISOString(),
        },
      });

      const { toggleWatchlist } = useAnalysisStore.getState();

      // Add
      await toggleWatchlist('AAPL');
      expect(useAnalysisStore.getState().watchlist.length).toBe(1);

      // Remove
      await toggleWatchlist('AAPL');
      expect(useAnalysisStore.getState().watchlist.length).toBe(0);
    });

    it('should persist watchlist to localStorage immediately', async () => {
      const { analysisApi } = await import('../services/api');
      vi.mocked(analysisApi.getQuote).mockImplementation(() =>
        new Promise(resolve => setTimeout(() => resolve({ success: false, timestamp: new Date().toISOString() }), 100))
      );

      const { addToWatchlist } = useAnalysisStore.getState();
      const promise = addToWatchlist('AAPL');

      // Should save to localStorage immediately (before API response)
      expect(window.localStorage.setItem).toHaveBeenCalledWith(
        'finmind_watchlist',
        expect.stringContaining('AAPL')
      );

      await promise;
    });
  });

  describe('refreshWatchlist', () => {
    it('should update watchlist with new data', async () => {
      const { analysisApi } = await import('../services/api');

      useAnalysisStore.setState({
        watchlist: [
          { symbol: 'AAPL', price: 170, change: 0, change_percent: 0, volume: 0, timestamp: '' },
        ],
      });

      vi.mocked(analysisApi.getQuotes).mockResolvedValue({
        success: true,
        timestamp: new Date().toISOString(),
        data: [
          { symbol: 'AAPL', price: 175, change: 5, change_percent: 2.94, volume: 50000000, timestamp: '' },
        ],
      });

      const { refreshWatchlist } = useAnalysisStore.getState();
      await refreshWatchlist();

      const { watchlist } = useAnalysisStore.getState();
      expect(watchlist[0].price).toBe(175);
    });

    it('should not update watchlist with empty response', async () => {
      const { analysisApi } = await import('../services/api');

      useAnalysisStore.setState({
        watchlist: [
          { symbol: 'AAPL', price: 170, change: 0, change_percent: 0, volume: 0, timestamp: '' },
        ],
      });

      vi.mocked(analysisApi.getQuotes).mockResolvedValue({
        success: true,
        timestamp: new Date().toISOString(),
        data: [],
      });

      const { refreshWatchlist } = useAnalysisStore.getState();
      await refreshWatchlist();

      // Should preserve original data
      const { watchlist } = useAnalysisStore.getState();
      expect(watchlist[0].price).toBe(170);
    });

    it('should handle API errors gracefully', async () => {
      const { analysisApi } = await import('../services/api');

      useAnalysisStore.setState({
        watchlist: [
          { symbol: 'AAPL', price: 170, change: 0, change_percent: 0, volume: 0, timestamp: '' },
        ],
      });

      vi.mocked(analysisApi.getQuotes).mockRejectedValue(new Error('Network error'));

      const { refreshWatchlist } = useAnalysisStore.getState();
      await refreshWatchlist();

      // Should preserve original data
      const { watchlist } = useAnalysisStore.getState();
      expect(watchlist.length).toBe(1);
      expect(watchlist[0].symbol).toBe('AAPL');
    });
  });
});
