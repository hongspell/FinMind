import React, { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Card, Space, Radio, Typography, Segmented } from 'antd';
import { LineChartOutlined, AreaChartOutlined, BarChartOutlined } from '@ant-design/icons';
import { createChart, IChartApi, ISeriesApi, CandlestickData, LineData, AreaData } from 'lightweight-charts';
import { useTranslation } from 'react-i18next';
import { colors } from '../../styles/theme';
import { useSettingsStore, ChartStyle } from '../../stores/settingsStore';

const { Text } = Typography;

interface PriceChartProps {
  symbol: string;
  data?: {
    dates: string[];
    open: number[];
    high: number[];
    low: number[];
    close: number[];
    volume: number[];
  };
}

type TimeRange = '1m' | '3m' | '6m' | '1y' | '5y';

const getDataForRange = (
  data: PriceChartProps['data'],
  range: TimeRange
): PriceChartProps['data'] | undefined => {
  if (!data || data.dates.length === 0) return data;

  const now = new Date();
  let startDate: Date;

  switch (range) {
    case '1m':
      startDate = new Date(now.setMonth(now.getMonth() - 1));
      break;
    case '3m':
      startDate = new Date(now.setMonth(now.getMonth() - 3));
      break;
    case '6m':
      startDate = new Date(now.setMonth(now.getMonth() - 6));
      break;
    case '1y':
      startDate = new Date(now.setFullYear(now.getFullYear() - 1));
      break;
    case '5y':
      startDate = new Date(now.setFullYear(now.getFullYear() - 5));
      break;
    default:
      return data;
  }

  const startIdx = data.dates.findIndex((d) => new Date(d) >= startDate);
  if (startIdx === -1) return data;

  return {
    dates: data.dates.slice(startIdx),
    open: data.open.slice(startIdx),
    high: data.high.slice(startIdx),
    low: data.low.slice(startIdx),
    close: data.close.slice(startIdx),
    volume: data.volume.slice(startIdx),
  };
};

const PriceChart: React.FC<PriceChartProps> = ({ symbol, data }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const mainSeriesRef = useRef<ISeriesApi<'Candlestick'> | ISeriesApi<'Line'> | ISeriesApi<'Area'> | null>(null);
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null);
  const isDisposedRef = useRef(false);
  const { t } = useTranslation();
  const [timeRange, setTimeRange] = useState<TimeRange>('1y');
  const { chartStyle, setChartStyle, theme } = useSettingsStore();

  const filteredData = useMemo(() => getDataForRange(data, timeRange), [data, timeRange]);

  // 获取主题相关的图表颜色
  const getChartColors = useCallback(() => {
    const isDark = theme === 'dark';
    return {
      background: isDark ? '#161b22' : '#ffffff',
      textColor: isDark ? '#8b949e' : '#595959',
      gridColor: isDark ? '#30363d' : '#e8e8e8',
      crosshairLabelBg: isDark ? '#1c2128' : '#ffffff',
    };
  }, [theme]);

  // 创建图表
  useEffect(() => {
    if (!chartContainerRef.current) return;

    isDisposedRef.current = false;
    const chartColors = getChartColors();

    // 创建图表
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { color: chartColors.background },
        textColor: chartColors.textColor,
      },
      grid: {
        vertLines: { color: chartColors.gridColor },
        horzLines: { color: chartColors.gridColor },
      },
      crosshair: {
        mode: 1,
        vertLine: {
          color: chartColors.textColor,
          width: 1,
          style: 2,
          labelBackgroundColor: chartColors.crosshairLabelBg,
        },
        horzLine: {
          color: chartColors.textColor,
          width: 1,
          style: 2,
          labelBackgroundColor: chartColors.crosshairLabelBg,
        },
      },
      rightPriceScale: {
        borderColor: chartColors.gridColor,
      },
      timeScale: {
        borderColor: chartColors.gridColor,
        timeVisible: true,
        secondsVisible: false,
      },
      handleScroll: {
        mouseWheel: true,
        pressedMouseMove: true,
      },
      handleScale: {
        axisPressedMouseMove: true,
        mouseWheel: true,
        pinch: true,
      },
    });

    // 根据图表样式创建不同的系列
    let mainSeries: ISeriesApi<'Candlestick'> | ISeriesApi<'Line'> | ISeriesApi<'Area'>;

    if (chartStyle === 'candle') {
      mainSeries = chart.addCandlestickSeries({
        upColor: colors.up,
        downColor: colors.down,
        borderUpColor: colors.up,
        borderDownColor: colors.down,
        wickUpColor: colors.up,
        wickDownColor: colors.down,
      });
    } else if (chartStyle === 'line') {
      mainSeries = chart.addLineSeries({
        color: colors.chart.primary,
        lineWidth: 2,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
        crosshairMarkerBorderColor: colors.chart.primary,
        crosshairMarkerBackgroundColor: chartColors.background,
      });
    } else {
      // area
      mainSeries = chart.addAreaSeries({
        topColor: `${colors.chart.primary}50`,
        bottomColor: `${colors.chart.primary}10`,
        lineColor: colors.chart.primary,
        lineWidth: 2,
        crosshairMarkerVisible: true,
        crosshairMarkerRadius: 4,
      });
    }

    // 成交量系列
    const volumeSeries = chart.addHistogramSeries({
      color: colors.chart.primary,
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: '',
    });

    volumeSeries.priceScale().applyOptions({
      scaleMargins: {
        top: 0.8,
        bottom: 0,
      },
    });

    chartRef.current = chart;
    mainSeriesRef.current = mainSeries;
    volumeSeriesRef.current = volumeSeries;

    // 响应式
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current && !isDisposedRef.current) {
        try {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
          });
        } catch (e) {
          // Chart might be disposed
        }
      }
    };

    window.addEventListener('resize', handleResize);

    // 清理函数
    return () => {
      window.removeEventListener('resize', handleResize);
      isDisposedRef.current = true;

      // 先清空 refs
      mainSeriesRef.current = null;
      volumeSeriesRef.current = null;

      // 然后移除图表
      if (chartRef.current) {
        try {
          chartRef.current.remove();
        } catch (e) {
          // Chart might already be disposed
        }
        chartRef.current = null;
      }
    };
  }, [chartStyle, theme, getChartColors]);

  // 更新数据 - 使用单独的 effect 来设置数据
  useEffect(() => {
    if (isDisposedRef.current) return;
    if (!filteredData) return;
    if (!chartRef.current || !mainSeriesRef.current || !volumeSeriesRef.current) return;

    try {
      if (chartStyle === 'candle') {
        // K线数据
        const candleData: CandlestickData[] = filteredData.dates.map((date, i) => ({
          time: date as unknown as import('lightweight-charts').Time,
          open: filteredData.open[i],
          high: filteredData.high[i],
          low: filteredData.low[i],
          close: filteredData.close[i],
        }));
        (mainSeriesRef.current as ISeriesApi<'Candlestick'>).setData(candleData);
      } else if (chartStyle === 'line') {
        // 折线数据
        const lineData: LineData[] = filteredData.dates.map((date, i) => ({
          time: date as unknown as import('lightweight-charts').Time,
          value: filteredData.close[i],
        }));
        (mainSeriesRef.current as ISeriesApi<'Line'>).setData(lineData);
      } else {
        // 面积图数据
        const areaData: AreaData[] = filteredData.dates.map((date, i) => ({
          time: date as unknown as import('lightweight-charts').Time,
          value: filteredData.close[i],
        }));
        (mainSeriesRef.current as ISeriesApi<'Area'>).setData(areaData);
      }

      const volumeData = filteredData.dates.map((date, i) => ({
        time: date as unknown as import('lightweight-charts').Time,
        value: filteredData.volume[i],
        color: filteredData.close[i] >= filteredData.open[i] ? `${colors.up}80` : `${colors.down}80`,
      }));

      volumeSeriesRef.current.setData(volumeData);

      // 自动适配
      if (chartRef.current && !isDisposedRef.current) {
        chartRef.current.timeScale().fitContent();
      }
    } catch (e) {
      // Chart might be disposed during data update
      console.warn('Chart update failed:', e);
    }
  }, [filteredData, chartStyle, theme]); // 添加 theme 作为依赖，确保数据在图表重建后更新

  const chartStyleOptions = [
    { value: 'candle', icon: <BarChartOutlined />, label: t('settings.candlestick') || 'K线' },
    { value: 'line', icon: <LineChartOutlined />, label: t('settings.line') || '折线' },
    { value: 'area', icon: <AreaChartOutlined />, label: t('settings.area') || '面积' },
  ];

  return (
    <Card
      title={
        <Space>
          {chartStyle === 'candle' ? <BarChartOutlined /> : chartStyle === 'line' ? <LineChartOutlined /> : <AreaChartOutlined />}
          <span>{t('analysis.priceChart', { symbol })}</span>
        </Space>
      }
      extra={
        <Space size={16}>
          {/* 图表样式切换 */}
          <Segmented
            size="small"
            value={chartStyle}
            onChange={(value) => setChartStyle(value as ChartStyle)}
            options={chartStyleOptions.map((opt) => ({
              value: opt.value,
              icon: opt.icon,
            }))}
          />
          {/* 时间范围切换 */}
          <Radio.Group
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
            size="small"
          >
            <Radio.Button value="1m">{t('timeframes.1M')}</Radio.Button>
            <Radio.Button value="3m">{t('timeframes.3M')}</Radio.Button>
            <Radio.Button value="6m">{t('timeframes.6M')}</Radio.Button>
            <Radio.Button value="1y">{t('timeframes.1Y')}</Radio.Button>
            <Radio.Button value="5y">{t('timeframes.5Y')}</Radio.Button>
          </Radio.Group>
        </Space>
      }
      styles={{ body: { padding: 0 } }}
    >
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: 400,
        }}
      />
      {!data && (
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: '50%',
            transform: 'translate(-50%, -50%)',
          }}
        >
          <Text type="secondary">{t('common.loading')}</Text>
        </div>
      )}
    </Card>
  );
};

export default PriceChart;
