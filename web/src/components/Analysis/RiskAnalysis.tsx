import React, { useEffect, useState, useRef, useCallback } from 'react';
import { Card, Row, Col, Statistic, Space, Button, Spin, Alert, Tooltip, Select, Typography } from 'antd';
import {
  WarningOutlined,
  ReloadOutlined,
  InfoCircleOutlined,
  RiseOutlined,
  FallOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useThemeColors } from '../../hooks/useThemeColors';
import { monteCarloApi } from '../../services/brokerApi';
import type { PriceSimulationResult } from '../../types/broker';

// echarts 按需加载（减少约 500KB 包大小）
import * as echarts from 'echarts/core';
import { LineChart } from 'echarts/charts';
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
} from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { LineSeriesOption } from 'echarts/charts';
import type { ComposeOption } from 'echarts/core';
import type {
  GridComponentOption,
  TooltipComponentOption,
  LegendComponentOption,
} from 'echarts/components';

type ECOption = ComposeOption<
  LineSeriesOption | GridComponentOption | TooltipComponentOption | LegendComponentOption
>;

echarts.use([LineChart, GridComponent, TooltipComponent, LegendComponent, CanvasRenderer]);

const { Text } = Typography;

interface RiskAnalysisProps {
  symbol: string;
  currentPrice: number;
}

const RiskAnalysis: React.FC<RiskAnalysisProps> = ({ symbol, currentPrice }) => {
  const { i18n } = useTranslation();
  const isZh = i18n.language?.startsWith('zh');
  const { isDark, borderColor, textColor, secondaryColor } = useThemeColors();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [simulation, setSimulation] = useState<PriceSimulationResult | null>(null);
  const [days, setDays] = useState(30);

  const chartRef = useRef<HTMLDivElement>(null);
  const chartInstance = useRef<echarts.ECharts | null>(null);

  // 加载模拟数据
  const loadSimulation = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await monteCarloApi.simulatePrice({
        symbol,
        current_price: currentPrice,
        days,
        simulations: 1000,
        confidence_levels: [0.95, 0.99],
      });
      if (response.success && response.data) {
        setSimulation(response.data);
      } else {
        setError(isZh ? '模拟服务暂不可用，请稍后重试' : 'Simulation service unavailable, please try again later');
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : '';
      const errorMsg = message.includes('404') || message.includes('Not Found')
        ? (isZh ? '后端服务未启动，请先启动 API 服务器' : 'Backend service not running. Please start the API server first.')
        : (isZh ? '模拟服务暂不可用' : 'Simulation service unavailable');
      setError(errorMsg);
    } finally {
      setLoading(false);
    }
  }, [symbol, currentPrice, days, isZh]);

  useEffect(() => {
    if (symbol && currentPrice > 0) {
      loadSimulation();
    }
  }, [symbol, currentPrice, days, loadSimulation]);

  // 渲染图表
  useEffect(() => {
    if (!chartRef.current || !simulation || !simulation.paths) return;

    if (!chartInstance.current) {
      chartInstance.current = echarts.init(chartRef.current);
    }

    const chart = chartInstance.current;

    // 准备路径数据 - 取前20条路径展示
    const pathsToShow = simulation.paths.slice(0, 20);
    const xAxisData = Array.from({ length: simulation.days + 1 }, (_, i) => i);

    const series: LineSeriesOption[] = pathsToShow.map((path, idx) => ({
      name: `Path ${idx + 1}`,
      type: 'line',
      data: path,
      showSymbol: false,
      lineStyle: {
        width: 1,
        opacity: 0.3,
      },
      emphasis: {
        lineStyle: {
          width: 2,
          opacity: 1,
        },
      },
    }));

    // 添加均值线
    if (simulation.final_prices) {
      const meanPath = Array.from({ length: simulation.days + 1 }, (_, i) => {
        const progress = i / simulation.days;
        return currentPrice + (simulation.final_prices.mean - currentPrice) * progress;
      });
      series.push({
        name: isZh ? '预期均值' : 'Expected Mean',
        type: 'line',
        data: meanPath,
        showSymbol: false,
        lineStyle: {
          width: 2,
          color: '#1890ff',
          type: 'dashed',
        },
      });
    }

    // 添加当前价格线
    series.push({
      name: isZh ? '当前价格' : 'Current Price',
      type: 'line',
      data: Array(simulation.days + 1).fill(currentPrice),
      showSymbol: false,
      lineStyle: {
        width: 1,
        color: '#52c41a',
        type: 'dotted',
      },
    });

    const option: ECOption = {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'axis',
        backgroundColor: isDark ? '#1c2128' : '#fff',
        borderColor: borderColor,
        textStyle: { color: textColor },
        formatter: (params: any) => {
          if (!Array.isArray(params) || params.length === 0) return '';
          const day = params[0].dataIndex;
          return `${isZh ? '第' : 'Day '}${day}${isZh ? '天' : ''}<br/>` +
            params.slice(0, 5).map((p: any) => `${p.seriesName}: $${p.value?.toFixed(2) || '-'}`).join('<br/>');
        },
      },
      legend: {
        show: false,
      },
      grid: {
        left: 60,
        right: 20,
        top: 20,
        bottom: 40,
      },
      xAxis: {
        type: 'category',
        data: xAxisData,
        name: isZh ? '交易日' : 'Trading Days',
        nameLocation: 'middle',
        nameGap: 25,
        axisLine: { lineStyle: { color: borderColor } },
        axisLabel: { color: secondaryColor },
      },
      yAxis: {
        type: 'value',
        name: isZh ? '价格 ($)' : 'Price ($)',
        axisLine: { lineStyle: { color: borderColor } },
        axisLabel: {
          color: secondaryColor,
          formatter: (val: number) => `$${val.toFixed(0)}`,
        },
        splitLine: { lineStyle: { color: borderColor, opacity: 0.5 } },
      },
      series,
    };

    chart.setOption(option, true);
  }, [simulation, isDark, isZh, currentPrice, borderColor, textColor, secondaryColor]);

  // resize 监听器：只挂载一次
  useEffect(() => {
    const handleResize = () => {
      chartInstance.current?.resize();
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
    };
  }, []);

  // 清理图表
  useEffect(() => {
    return () => {
      chartInstance.current?.dispose();
      chartInstance.current = null;
    };
  }, []);

  if (loading && !simulation) {
    return (
      <Card
        title={
          <Space>
            <WarningOutlined />
            <span>{isZh ? '风险分析' : 'Risk Analysis'}</span>
          </Space>
        }
        size="small"
      >
        <div style={{ display: 'flex', justifyContent: 'center', padding: 40 }}>
          <Spin />
        </div>
      </Card>
    );
  }

  if (error) {
    return (
      <Card
        title={
          <Space>
            <WarningOutlined />
            <span>{isZh ? '风险分析' : 'Risk Analysis'}</span>
          </Space>
        }
        size="small"
      >
        <Alert
          message={error}
          type="error"
          showIcon
          action={
            <Button size="small" onClick={loadSimulation}>
              {isZh ? '重试' : 'Retry'}
            </Button>
          }
        />
      </Card>
    );
  }

  return (
    <Card
      title={
        <Space>
          <WarningOutlined />
          <span>{isZh ? '风险分析 - 蒙特卡洛模拟' : 'Risk Analysis - Monte Carlo Simulation'}</span>
        </Space>
      }
      size="small"
      extra={
        <Space>
          <Select
            value={days}
            onChange={setDays}
            size="small"
            style={{ width: 100 }}
            options={[
              { value: 7, label: isZh ? '7天' : '7 Days' },
              { value: 14, label: isZh ? '14天' : '14 Days' },
              { value: 30, label: isZh ? '30天' : '30 Days' },
              { value: 60, label: isZh ? '60天' : '60 Days' },
              { value: 90, label: isZh ? '90天' : '90 Days' },
            ]}
          />
          <Button
            icon={<ReloadOutlined spin={loading} />}
            size="small"
            onClick={loadSimulation}
            loading={loading}
          >
            {isZh ? '刷新' : 'Refresh'}
          </Button>
        </Space>
      }
    >
      {simulation && (
        <>
          {/* VaR/CVaR Metrics */}
          <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
            <Col xs={12} sm={6}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '95%置信度下的最大日损失' : 'Maximum daily loss at 95% confidence'}>
                    <Space>
                      VaR (95%)
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={simulation.var_values?.[0.95] || 0}
                precision={2}
                prefix="$"
                valueStyle={{ color: '#ff4d4f', fontSize: 18 }}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '99%置信度下的最大日损失' : 'Maximum daily loss at 99% confidence'}>
                    <Space>
                      VaR (99%)
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={simulation.var_values?.[0.99] || 0}
                precision={2}
                prefix="$"
                valueStyle={{ color: '#ff4d4f', fontSize: 18 }}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '盈利概率' : 'Probability of profit'}>
                    <Space>
                      {isZh ? '盈利概率' : 'Profit Prob.'}
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={(simulation.probability_of_profit || 0) * 100}
                precision={1}
                suffix="%"
                valueStyle={{ color: simulation.probability_of_profit >= 0.5 ? '#52c41a' : '#ff4d4f', fontSize: 18 }}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '预期收益率' : 'Expected return'}>
                    <Space>
                      {isZh ? '预期收益' : 'Exp. Return'}
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={(simulation.expected_return || 0) * 100}
                precision={2}
                suffix="%"
                prefix={simulation.expected_return >= 0 ? <RiseOutlined /> : <FallOutlined />}
                valueStyle={{ color: simulation.expected_return >= 0 ? '#52c41a' : '#ff4d4f', fontSize: 18 }}
              />
            </Col>
          </Row>

          {/* Price Distribution */}
          {simulation.final_prices && (
            <Row gutter={[16, 16]} style={{ marginBottom: 16 }}>
              <Col xs={8} sm={4}>
                <Statistic
                  title={isZh ? '最低价' : 'Min'}
                  value={simulation.final_prices.min}
                  precision={2}
                  prefix="$"
                  valueStyle={{ fontSize: 14 }}
                />
              </Col>
              <Col xs={8} sm={4}>
                <Statistic
                  title={isZh ? '5%分位' : '5th Pctl'}
                  value={simulation.final_prices.percentiles?.[5] || 0}
                  precision={2}
                  prefix="$"
                  valueStyle={{ fontSize: 14 }}
                />
              </Col>
              <Col xs={8} sm={4}>
                <Statistic
                  title={isZh ? '中位数' : 'Median'}
                  value={simulation.final_prices.median}
                  precision={2}
                  prefix="$"
                  valueStyle={{ fontSize: 14, color: '#1890ff' }}
                />
              </Col>
              <Col xs={8} sm={4}>
                <Statistic
                  title={isZh ? '均值' : 'Mean'}
                  value={simulation.final_prices.mean}
                  precision={2}
                  prefix="$"
                  valueStyle={{ fontSize: 14 }}
                />
              </Col>
              <Col xs={8} sm={4}>
                <Statistic
                  title={isZh ? '95%分位' : '95th Pctl'}
                  value={simulation.final_prices.percentiles?.[95] || 0}
                  precision={2}
                  prefix="$"
                  valueStyle={{ fontSize: 14 }}
                />
              </Col>
              <Col xs={8} sm={4}>
                <Statistic
                  title={isZh ? '最高价' : 'Max'}
                  value={simulation.final_prices.max}
                  precision={2}
                  prefix="$"
                  valueStyle={{ fontSize: 14 }}
                />
              </Col>
            </Row>
          )}

          {/* Monte Carlo Chart */}
          <div
            ref={chartRef}
            style={{
              width: '100%',
              height: 300,
              marginTop: 16,
            }}
          />

          <Text type="secondary" style={{ display: 'block', marginTop: 8, fontSize: 12 }}>
            {isZh
              ? `基于 ${simulation.simulations} 次蒙特卡洛模拟，使用几何布朗运动模型。图表显示 20 条采样路径。`
              : `Based on ${simulation.simulations} Monte Carlo simulations using Geometric Brownian Motion. Chart shows 20 sample paths.`}
          </Text>
        </>
      )}
    </Card>
  );
};

export default RiskAnalysis;
