import React, { useState, useCallback } from 'react';
import { Card, DatePicker, InputNumber, Button, Row, Col, Typography, Statistic, Tag, Space, Spin, Alert, Table } from 'antd';
import { ExperimentOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { backtestApi } from '../../services/api';
import dayjs from 'dayjs';

const { Text, Title } = Typography;

interface BacktestProps {
  symbol: string;
}

interface BacktestResult {
  symbol: string;
  backtest_date: string;
  forward_days: number;
  backtest_price: number;
  prediction: {
    direction: string;
    score: number;
    technical_signals: Record<string, any>;
    dcf: Record<string, any>;
  };
  actual: {
    final_price: number | null;
    return_pct: number | null;
    max_price: number | null;
    min_price: number | null;
    max_drawup_pct: number | null;
    max_drawdown_pct: number | null;
    actual_direction: string | null;
    trading_days: number;
  };
  accuracy: {
    direction_correct: boolean | null;
    predicted_direction: string;
    actual_direction: string | null;
  };
}

const BacktestPanel: React.FC<BacktestProps> = ({ symbol }) => {
  const { i18n } = useTranslation();
  const isZh = i18n.language?.startsWith('zh');

  const [backtestDate, setBacktestDate] = useState<string>('2025-01-01');
  const [forwardDays, setForwardDays] = useState<number>(90);
  const [result, setResult] = useState<BacktestResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runBacktest = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await backtestApi.runBacktest({
        symbol,
        backtest_date: backtestDate,
        forward_days: forwardDays,
      });
      setResult(res as unknown as BacktestResult);
    } catch (e: any) {
      setError(e.message || 'Backtest failed');
    } finally {
      setLoading(false);
    }
  }, [symbol, backtestDate, forwardDays]);

  const directionColor = (dir: string | null) => {
    if (dir === 'bullish') return '#52c41a';
    if (dir === 'bearish') return '#ff4d4f';
    return '#d9d9d9';
  };

  const directionLabel = (dir: string | null) => {
    if (!dir) return '-';
    if (dir === 'bullish') return isZh ? '看涨' : 'Bullish';
    if (dir === 'bearish') return isZh ? '看跌' : 'Bearish';
    return isZh ? '中性' : 'Neutral';
  };

  const signalRows = result
    ? [
        {
          key: 'ma',
          indicator: isZh ? 'MA 交叉' : 'MA Crossover',
          value: result.prediction.technical_signals.ma_crossover || '-',
        },
        {
          key: 'rsi',
          indicator: 'RSI',
          value: result.prediction.technical_signals.rsi
            ? `${result.prediction.technical_signals.rsi} (${result.prediction.technical_signals.rsi_signal})`
            : '-',
        },
        {
          key: 'macd',
          indicator: 'MACD',
          value: result.prediction.technical_signals.macd_signal || '-',
        },
        {
          key: 'dcf',
          indicator: isZh ? 'DCF 估值' : 'DCF Valuation',
          value: result.prediction.dcf.fair_value
            ? `$${result.prediction.dcf.fair_value} (${result.prediction.dcf.upside_pct > 0 ? '+' : ''}${result.prediction.dcf.upside_pct}%)`
            : result.prediction.dcf.error || '-',
        },
      ]
    : [];

  return (
    <Card
      title={
        <Space>
          <ExperimentOutlined />
          <span>{isZh ? '量化回测' : 'Quantitative Backtest'}</span>
        </Space>
      }
      size="small"
    >
      {/* Controls */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }} align="middle">
        <Col>
          <Text strong>{isZh ? '回测日期' : 'Backtest Date'}:</Text>
          <DatePicker
            value={dayjs(backtestDate)}
            onChange={(_, dateStr) => setBacktestDate(dateStr as string)}
            style={{ marginLeft: 8, width: 150 }}
            disabledDate={(current) => current && current > dayjs().subtract(7, 'day')}
          />
        </Col>
        <Col>
          <Text strong>{isZh ? '预测天数' : 'Forward Days'}:</Text>
          <InputNumber
            min={5}
            max={365}
            value={forwardDays}
            onChange={(v) => v && setForwardDays(v)}
            style={{ marginLeft: 8, width: 80 }}
          />
        </Col>
        <Col>
          <Button type="primary" onClick={runBacktest} loading={loading}>
            {isZh ? '运行回测' : 'Run Backtest'}
          </Button>
        </Col>
      </Row>

      {error && <Alert message={error} type="error" showIcon closable style={{ marginBottom: 16 }} />}

      {loading && !result && (
        <div style={{ textAlign: 'center', padding: 32 }}>
          <Spin size="large" />
        </div>
      )}

      {result && (
        <>
          {/* Direction comparison */}
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={12} sm={6}>
              <Statistic
                title={isZh ? '回测日价格' : 'Backtest Price'}
                value={result.backtest_price}
                precision={2}
                prefix="$"
              />
            </Col>
            <Col xs={12} sm={6}>
              <Card size="small" bodyStyle={{ textAlign: 'center' }}>
                <Text type="secondary">{isZh ? '预测方向' : 'Predicted'}</Text>
                <div>
                  <Tag
                    color={directionColor(result.accuracy.predicted_direction)}
                    style={{ fontSize: 16, padding: '4px 12px', marginTop: 4 }}
                  >
                    {directionLabel(result.accuracy.predicted_direction)}
                  </Tag>
                </div>
              </Card>
            </Col>
            <Col xs={12} sm={6}>
              <Card size="small" bodyStyle={{ textAlign: 'center' }}>
                <Text type="secondary">{isZh ? '实际方向' : 'Actual'}</Text>
                <div>
                  <Tag
                    color={directionColor(result.accuracy.actual_direction)}
                    style={{ fontSize: 16, padding: '4px 12px', marginTop: 4 }}
                  >
                    {directionLabel(result.accuracy.actual_direction)}
                  </Tag>
                </div>
              </Card>
            </Col>
            <Col xs={12} sm={6}>
              <Card size="small" bodyStyle={{ textAlign: 'center' }}>
                <Text type="secondary">{isZh ? '预测结果' : 'Result'}</Text>
                <div style={{ marginTop: 4 }}>
                  {result.accuracy.direction_correct === null ? (
                    <Tag>N/A</Tag>
                  ) : result.accuracy.direction_correct ? (
                    <Tag color="success" style={{ fontSize: 16, padding: '4px 12px' }}>
                      {isZh ? '正确' : 'Correct'}
                    </Tag>
                  ) : (
                    <Tag color="error" style={{ fontSize: 16, padding: '4px 12px' }}>
                      {isZh ? '错误' : 'Wrong'}
                    </Tag>
                  )}
                </div>
              </Card>
            </Col>
          </Row>

          {/* Actual outcome */}
          {result.actual.final_price !== null && (
            <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
              <Col xs={12} sm={6}>
                <Statistic
                  title={isZh ? '最终价格' : 'Final Price'}
                  value={result.actual.final_price}
                  precision={2}
                  prefix="$"
                />
              </Col>
              <Col xs={12} sm={6}>
                <Statistic
                  title={isZh ? '回报率' : 'Return'}
                  value={result.actual.return_pct!}
                  precision={2}
                  suffix="%"
                  prefix={result.actual.return_pct! >= 0 ? '+' : ''}
                  valueStyle={{ color: result.actual.return_pct! >= 0 ? '#52c41a' : '#ff4d4f' }}
                />
              </Col>
              <Col xs={12} sm={6}>
                <Statistic
                  title={isZh ? '最高涨幅' : 'Max Drawup'}
                  value={result.actual.max_drawup_pct!}
                  precision={2}
                  suffix="%"
                  prefix="+"
                  valueStyle={{ color: '#52c41a' }}
                />
              </Col>
              <Col xs={12} sm={6}>
                <Statistic
                  title={isZh ? '最大回撤' : 'Max Drawdown'}
                  value={result.actual.max_drawdown_pct!}
                  precision={2}
                  suffix="%"
                  valueStyle={{ color: '#ff4d4f' }}
                />
              </Col>
            </Row>
          )}

          {/* Signal breakdown */}
          <Title level={5}>{isZh ? '信号明细' : 'Signal Breakdown'}</Title>
          <Table
            dataSource={signalRows}
            columns={[
              {
                title: isZh ? '指标' : 'Indicator',
                dataIndex: 'indicator',
                key: 'indicator',
                width: 150,
              },
              {
                title: isZh ? '信号值' : 'Signal',
                dataIndex: 'value',
                key: 'value',
              },
            ]}
            pagination={false}
            size="small"
            style={{ marginBottom: 8 }}
          />
          <Text type="secondary" style={{ fontSize: 12 }}>
            {isZh
              ? `综合评分: ${result.prediction.score}（正数=看涨, 负数=看跌）| 交易天数: ${result.actual.trading_days}`
              : `Aggregate score: ${result.prediction.score} (positive=bullish, negative=bearish) | Trading days: ${result.actual.trading_days}`}
          </Text>
        </>
      )}
    </Card>
  );
};

export default BacktestPanel;
