import type { ThemeConfig } from 'antd';
import { theme } from 'antd';

// FinMind 专业金融深色主题
export const darkTheme: ThemeConfig = {
  algorithm: theme.darkAlgorithm,
  token: {
    // 主色调 - 专业蓝
    colorPrimary: '#1890ff',
    colorInfo: '#1890ff',

    // 成功/上涨 - 金融绿
    colorSuccess: '#00d4aa',

    // 错误/下跌 - 金融红
    colorError: '#ff4d6a',

    // 警告
    colorWarning: '#faad14',

    // 背景色
    colorBgBase: '#0d1117',
    colorBgContainer: '#161b22',
    colorBgElevated: '#1c2128',
    colorBgLayout: '#0d1117',

    // 边框
    colorBorder: '#30363d',
    colorBorderSecondary: '#21262d',

    // 文字
    colorText: '#e6edf3',
    colorTextSecondary: '#8b949e',
    colorTextTertiary: '#6e7681',
    colorTextQuaternary: '#484f58',

    // 圆角
    borderRadius: 6,
    borderRadiusLG: 8,
    borderRadiusSM: 4,

    // 字体
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    fontSize: 14,

    // 阴影
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.3)',
    boxShadowSecondary: '0 4px 16px rgba(0, 0, 0, 0.4)',
  },
  components: {
    Layout: {
      headerBg: '#161b22',
      siderBg: '#161b22',
      bodyBg: '#0d1117',
      headerHeight: 56,
    },
    Menu: {
      darkItemBg: '#161b22',
      darkItemSelectedBg: '#1f6feb33',
      darkItemHoverBg: '#1f6feb1a',
    },
    Card: {
      colorBgContainer: '#161b22',
      colorBorderSecondary: '#30363d',
    },
    Table: {
      colorBgContainer: '#161b22',
      headerBg: '#1c2128',
      rowHoverBg: '#1f6feb1a',
    },
    Input: {
      colorBgContainer: '#0d1117',
      colorBorder: '#30363d',
      activeBorderColor: '#1890ff',
      hoverBorderColor: '#484f58',
    },
    Select: {
      colorBgContainer: '#0d1117',
      colorBgElevated: '#1c2128',
    },
    Button: {
      primaryShadow: '0 2px 0 rgba(24, 144, 255, 0.1)',
    },
    Statistic: {
      contentFontSize: 24,
    },
  },
};

// 保持向后兼容
export const finmindTheme = darkTheme;

// FinMind 浅色主题
export const lightTheme: ThemeConfig = {
  algorithm: theme.defaultAlgorithm,
  token: {
    // 主色调 - 专业蓝
    colorPrimary: '#1890ff',
    colorInfo: '#1890ff',

    // 成功/上涨 - 金融绿
    colorSuccess: '#00a67d',

    // 错误/下跌 - 金融红
    colorError: '#e64545',

    // 警告
    colorWarning: '#faad14',

    // 背景色 - 浅色
    colorBgBase: '#ffffff',
    colorBgContainer: '#ffffff',
    colorBgElevated: '#fafafa',
    colorBgLayout: '#f5f5f5',

    // 边框
    colorBorder: '#d9d9d9',
    colorBorderSecondary: '#f0f0f0',

    // 文字
    colorText: '#1f1f1f',
    colorTextSecondary: '#595959',
    colorTextTertiary: '#8c8c8c',
    colorTextQuaternary: '#bfbfbf',

    // 圆角
    borderRadius: 6,
    borderRadiusLG: 8,
    borderRadiusSM: 4,

    // 字体
    fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    fontSize: 14,

    // 阴影
    boxShadow: '0 2px 8px rgba(0, 0, 0, 0.08)',
    boxShadowSecondary: '0 4px 16px rgba(0, 0, 0, 0.12)',
  },
  components: {
    Layout: {
      headerBg: '#ffffff',
      siderBg: '#ffffff',
      bodyBg: '#f5f5f5',
      headerHeight: 56,
    },
    Menu: {
      itemBg: '#ffffff',
      itemSelectedBg: '#e6f4ff',
      itemHoverBg: '#f5f5f5',
    },
    Card: {
      colorBgContainer: '#ffffff',
      colorBorderSecondary: '#f0f0f0',
    },
    Table: {
      colorBgContainer: '#ffffff',
      headerBg: '#fafafa',
      rowHoverBg: '#f5f5f5',
    },
    Input: {
      colorBgContainer: '#ffffff',
      colorBorder: '#d9d9d9',
      activeBorderColor: '#1890ff',
      hoverBorderColor: '#69b1ff',
    },
    Select: {
      colorBgContainer: '#ffffff',
      colorBgElevated: '#ffffff',
    },
    Button: {
      primaryShadow: '0 2px 0 rgba(24, 144, 255, 0.1)',
    },
    Statistic: {
      contentFontSize: 24,
    },
  },
};

// 涨跌颜色
export const colors = {
  up: '#00d4aa',      // 上涨绿
  down: '#ff4d6a',    // 下跌红
  neutral: '#8b949e', // 中性灰

  // 信号颜色
  strongBuy: '#00d4aa',
  buy: '#52c41a',
  hold: '#faad14',
  sell: '#ff7a45',
  strongSell: '#ff4d6a',

  // 置信度颜色
  highConfidence: '#00d4aa',
  mediumConfidence: '#faad14',
  lowConfidence: '#ff4d6a',

  // 图表颜色
  chart: {
    primary: '#1890ff',
    secondary: '#722ed1',
    tertiary: '#13c2c2',
    grid: '#30363d',
    crosshair: '#8b949e',
  },
};

// 信号映射
export const signalColors: Record<string, string> = {
  strong_buy: colors.strongBuy,
  buy: colors.buy,
  neutral: colors.neutral,
  sell: colors.sell,
  strong_sell: colors.strongSell,
};

// 趋势映射
export const trendColors: Record<string, string> = {
  strong_bullish: colors.up,
  bullish: colors.buy,
  neutral: colors.neutral,
  bearish: colors.sell,
  strong_bearish: colors.down,
};
