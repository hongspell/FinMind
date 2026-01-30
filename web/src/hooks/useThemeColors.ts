import { useMemo } from 'react';
import { useSettingsStore } from '../stores/settingsStore';

export interface ThemeColors {
  isDark: boolean;
  textColor: string;
  secondaryColor: string;
  borderColor: string;
  cardBg: string;
  siderBg: string;
}

export function useThemeColors(): ThemeColors {
  const { theme } = useSettingsStore();

  return useMemo(() => {
    const isDark = theme === 'dark';
    return {
      isDark,
      textColor: isDark ? '#e6edf3' : '#1f1f1f',
      secondaryColor: isDark ? '#8b949e' : '#595959',
      borderColor: isDark ? '#30363d' : '#e8e8e8',
      cardBg: isDark ? '#1c2128' : '#ffffff',
      siderBg: isDark ? '#161b22' : '#ffffff',
    };
  }, [theme]);
}
