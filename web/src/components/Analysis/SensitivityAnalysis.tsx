import React, { useState, useCallback, useEffect, useRef } from 'react';
import { Card, Slider, Row, Col, Typography, Statistic, Spin, Alert, Space } from 'antd';
import { useTranslation } from 'react-i18next';
import { valuationApi } from '../../services/api';

const { Text, Title } = Typography;

interface SensitivityProps {
  symbol: string;
  currentPrice: number;
}

interface SensitivityResult {
  fair_value: number;
  enterprise_value: number;
  upside_downside: number;
  company_name: string;
  beta: number;
  parameters: {
    discount_rate: number;
    growth_rate: number;
    terminal_growth: number;
    projection_years: number;
  };
  sensitivity_matrix: {
    discount_rates: number[];
    terminal_growths: number[];
    fair_values: (number | null)[][];
  };
}

const SensitivityAnalysis: React.FC<SensitivityProps> = ({ symbol, currentPrice }) => {
  const { i18n } = useTranslation();
  const isZh = i18n.language?.startsWith('zh');

  const [discountRate, setDiscountRate] = useState(10);
  const [growthRate, setGrowthRate] = useState(8);
  const [terminalGrowth, setTerminalGrowth] = useState(2.5);
  const [projectionYears, setProjectionYears] = useState(10);
  const [result, setResult] = useState<SensitivityResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const debounceRef = useRef<ReturnType<typeof setTimeout>>();

  const fetchSensitivity = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await valuationApi.sensitivityAnalysis({
        symbol,
        discount_rate: discountRate / 100,
        growth_rate: growthRate / 100,
        terminal_growth: terminalGrowth / 100,
        projection_years: projectionYears,
      });
      setResult(res as unknown as SensitivityResult);
    } catch (e: any) {
      setError(e.message || 'Failed to run sensitivity analysis');
    } finally {
      setLoading(false);
    }
  }, [symbol, discountRate, growthRate, terminalGrowth, projectionYears]);

  // Debounced fetch on parameter change
  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(fetchSensitivity, 300);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [fetchSensitivity]);

  const getCellColor = (value: number | null): string => {
    if (value === null) return '#f0f0f0';
    const diff = ((value - currentPrice) / currentPrice) * 100;
    if (diff > 30) return '#95de64';
    if (diff > 10) return '#b7eb8f';
    if (diff > 0) return '#d9f7be';
    if (diff > -10) return '#fff1f0';
    if (diff > -30) return '#ffa39e';
    return '#ff7875';
  };

  return (
    <Card
      title={
        <Space>
          <span>{isZh ? 'DCF 敏感度分析' : 'DCF Sensitivity Analysis'}</span>
          {loading && <Spin size="small" />}
        </Space>
      }
      size="small"
    >
      {error && (
        <Alert
          message={error}
          type="warning"
          showIcon
          closable
          style={{ marginBottom: 16 }}
        />
      )}

      {/* Sliders */}
      <Row gutter={[24, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} md={6}>
          <Text strong>{isZh ? '折现率' : 'Discount Rate'}: {discountRate}%</Text>
          <Slider
            min={5}
            max={15}
            step={0.5}
            value={discountRate}
            onChange={setDiscountRate}
            marks={{ 5: '5%', 10: '10%', 15: '15%' }}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Text strong>{isZh ? '增长率' : 'Growth Rate'}: {growthRate}%</Text>
          <Slider
            min={0}
            max={25}
            step={0.5}
            value={growthRate}
            onChange={setGrowthRate}
            marks={{ 0: '0%', 12: '12%', 25: '25%' }}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Text strong>{isZh ? '终值增长率' : 'Terminal Growth'}: {terminalGrowth}%</Text>
          <Slider
            min={1}
            max={4}
            step={0.25}
            value={terminalGrowth}
            onChange={setTerminalGrowth}
            marks={{ 1: '1%', 2.5: '2.5%', 4: '4%' }}
          />
        </Col>
        <Col xs={24} sm={12} md={6}>
          <Text strong>{isZh ? '预测年数' : 'Projection Years'}: {projectionYears}</Text>
          <Slider
            min={5}
            max={15}
            step={1}
            value={projectionYears}
            onChange={setProjectionYears}
            marks={{ 5: '5', 10: '10', 15: '15' }}
          />
        </Col>
      </Row>

      {/* Fair Value Display */}
      {result && (
        <>
          <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
            <Col xs={12} sm={6}>
              <Statistic
                title={isZh ? '公允价值' : 'Fair Value'}
                value={result.fair_value}
                precision={2}
                prefix="$"
                valueStyle={{ color: '#1890ff', fontWeight: 'bold' }}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title={isZh ? '当前价格' : 'Current Price'}
                value={currentPrice}
                precision={2}
                prefix="$"
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title={isZh ? '上涨/下跌空间' : 'Upside/Downside'}
                value={result.upside_downside}
                precision={2}
                suffix="%"
                valueStyle={{ color: result.upside_downside >= 0 ? '#52c41a' : '#ff4d4f' }}
                prefix={result.upside_downside >= 0 ? '+' : ''}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title={isZh ? '企业价值' : 'Enterprise Value'}
                value={result.enterprise_value / 1e9}
                precision={2}
                suffix="B"
                prefix="$"
              />
            </Col>
          </Row>

          {/* Sensitivity Heatmap */}
          <Title level={5} style={{ marginBottom: 8 }}>
            {isZh ? '敏感性矩阵（折现率 × 终值增长率）' : 'Sensitivity Matrix (Discount Rate × Terminal Growth)'}
          </Title>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ borderCollapse: 'collapse', width: '100%', minWidth: 500, fontSize: 13 }}>
              <thead>
                <tr>
                  <th style={{ padding: '6px 8px', border: '1px solid #d9d9d9', background: '#fafafa' }}>
                    DR \ TG
                  </th>
                  {result.sensitivity_matrix.terminal_growths.map((tg) => (
                    <th
                      key={tg}
                      style={{
                        padding: '6px 8px',
                        border: '1px solid #d9d9d9',
                        background: '#fafafa',
                        textAlign: 'center',
                      }}
                    >
                      {tg}%
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {result.sensitivity_matrix.discount_rates.map((dr, i) => (
                  <tr key={dr}>
                    <td
                      style={{
                        padding: '6px 8px',
                        border: '1px solid #d9d9d9',
                        background: '#fafafa',
                        fontWeight: 'bold',
                      }}
                    >
                      {dr}%
                    </td>
                    {result.sensitivity_matrix.fair_values[i].map((fv, j) => (
                      <td
                        key={j}
                        style={{
                          padding: '6px 8px',
                          border: '1px solid #d9d9d9',
                          textAlign: 'center',
                          background: getCellColor(fv),
                          fontWeight:
                            dr === result.parameters.discount_rate * 100 &&
                            result.sensitivity_matrix.terminal_growths[j] === result.parameters.terminal_growth * 100
                              ? 'bold'
                              : 'normal',
                        }}
                      >
                        {fv !== null ? `$${fv.toFixed(0)}` : 'N/A'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <Text type="secondary" style={{ display: 'block', marginTop: 8, fontSize: 12 }}>
            {isZh
              ? '绿色 = 低估（公允价值 > 当前价格），红色 = 高估（公允价值 < 当前价格）'
              : 'Green = undervalued (fair value > current price), Red = overvalued (fair value < current price)'}
          </Text>
        </>
      )}
    </Card>
  );
};

export default SensitivityAnalysis;
