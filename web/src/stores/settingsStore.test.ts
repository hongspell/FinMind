import { describe, it, expect, beforeEach, vi } from 'vitest';
import { useSettingsStore } from './settingsStore';

describe('settingsStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useSettingsStore.setState({
      theme: 'dark',
      chartStyle: 'candle',
      priceAlerts: false,
      signalAlerts: false,
    });

    // Clear localStorage mock
    vi.mocked(window.localStorage.setItem).mockClear();
    vi.mocked(window.localStorage.getItem).mockReturnValue(null);
  });

  describe('theme', () => {
    it('should default to dark theme', () => {
      expect(useSettingsStore.getState().theme).toBe('dark');
    });

    it('should set theme and persist to localStorage', () => {
      const { setTheme } = useSettingsStore.getState();
      setTheme('light');

      expect(useSettingsStore.getState().theme).toBe('light');
      expect(window.localStorage.setItem).toHaveBeenCalledWith('finmind_theme', 'light');
    });

    it('should toggle between themes', () => {
      const { setTheme } = useSettingsStore.getState();

      setTheme('light');
      expect(useSettingsStore.getState().theme).toBe('light');

      setTheme('dark');
      expect(useSettingsStore.getState().theme).toBe('dark');
    });
  });

  describe('chartStyle', () => {
    it('should default to candle chart', () => {
      expect(useSettingsStore.getState().chartStyle).toBe('candle');
    });

    it('should set chart style and persist to localStorage', () => {
      const { setChartStyle } = useSettingsStore.getState();
      setChartStyle('line');

      expect(useSettingsStore.getState().chartStyle).toBe('line');
      expect(window.localStorage.setItem).toHaveBeenCalledWith('finmind_chartStyle', 'line');
    });

    it('should support all chart styles', () => {
      const { setChartStyle } = useSettingsStore.getState();

      setChartStyle('candle');
      expect(useSettingsStore.getState().chartStyle).toBe('candle');

      setChartStyle('line');
      expect(useSettingsStore.getState().chartStyle).toBe('line');

      setChartStyle('area');
      expect(useSettingsStore.getState().chartStyle).toBe('area');
    });
  });

  describe('priceAlerts', () => {
    it('should default to false', () => {
      expect(useSettingsStore.getState().priceAlerts).toBe(false);
    });

    it('should set price alerts and persist to localStorage', () => {
      const { setPriceAlerts } = useSettingsStore.getState();
      setPriceAlerts(true);

      expect(useSettingsStore.getState().priceAlerts).toBe(true);
      expect(window.localStorage.setItem).toHaveBeenCalledWith('finmind_priceAlerts', 'true');
    });

    it('should toggle price alerts', () => {
      const { setPriceAlerts } = useSettingsStore.getState();

      setPriceAlerts(true);
      expect(useSettingsStore.getState().priceAlerts).toBe(true);

      setPriceAlerts(false);
      expect(useSettingsStore.getState().priceAlerts).toBe(false);
    });
  });

  describe('signalAlerts', () => {
    it('should default to false', () => {
      expect(useSettingsStore.getState().signalAlerts).toBe(false);
    });

    it('should set signal alerts and persist to localStorage', () => {
      const { setSignalAlerts } = useSettingsStore.getState();
      setSignalAlerts(true);

      expect(useSettingsStore.getState().signalAlerts).toBe(true);
      expect(window.localStorage.setItem).toHaveBeenCalledWith('finmind_signalAlerts', 'true');
    });
  });
});
