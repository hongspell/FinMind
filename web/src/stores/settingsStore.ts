import { create } from 'zustand';

export type ChartStyle = 'candle' | 'line' | 'area';
export type Theme = 'dark' | 'light';

interface SettingsState {
  // 显示设置
  theme: Theme;
  chartStyle: ChartStyle;

  // 通知设置
  priceAlerts: boolean;
  signalAlerts: boolean;

  // Actions
  setTheme: (theme: Theme) => void;
  setChartStyle: (style: ChartStyle) => void;
  setPriceAlerts: (enabled: boolean) => void;
  setSignalAlerts: (enabled: boolean) => void;
}

export const useSettingsStore = create<SettingsState>((set) => ({
  theme: 'dark',
  chartStyle: 'candle',
  priceAlerts: false,
  signalAlerts: false,

  setTheme: (theme) => {
    set({ theme });
    localStorage.setItem('finmind_theme', theme);
  },

  setChartStyle: (chartStyle) => {
    set({ chartStyle });
    localStorage.setItem('finmind_chartStyle', chartStyle);
  },

  setPriceAlerts: (priceAlerts) => {
    set({ priceAlerts });
    localStorage.setItem('finmind_priceAlerts', String(priceAlerts));
  },

  setSignalAlerts: (signalAlerts) => {
    set({ signalAlerts });
    localStorage.setItem('finmind_signalAlerts', String(signalAlerts));
  },
}));

// 初始化时从 localStorage 加载
const initSettings = () => {
  try {
    const theme = localStorage.getItem('finmind_theme') as Theme;
    const chartStyle = localStorage.getItem('finmind_chartStyle') as ChartStyle;
    const priceAlerts = localStorage.getItem('finmind_priceAlerts');
    const signalAlerts = localStorage.getItem('finmind_signalAlerts');

    useSettingsStore.setState({
      theme: theme || 'dark',
      chartStyle: chartStyle || 'candle',
      priceAlerts: priceAlerts === 'true',
      signalAlerts: signalAlerts === 'true',
    });
  } catch (e) {
    console.error('Failed to load settings:', e);
  }
};

initSettings();
