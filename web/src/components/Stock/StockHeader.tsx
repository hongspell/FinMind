import React from 'react';
import { Card, Typography, Space, Tag, Statistic, Row, Col, Tooltip } from 'antd';
import {
  ArrowUpOutlined,
  ArrowDownOutlined,
  InfoCircleOutlined,
  StarOutlined,
  StarFilled,
} from '@ant-design/icons';
import type { MarketData } from '../../types/analysis';
import { colors } from '../../styles/theme';
import { useSettingsStore } from '../../stores/settingsStore';

const { Title, Text } = Typography;

interface StockHeaderProps {
  symbol: string;
  marketData: MarketData;
  isInWatchlist?: boolean;
  onToggleWatchlist?: () => void;
}

const StockHeader: React.FC<StockHeaderProps> = ({
  symbol,
  marketData,
  isInWatchlist = false,
  onToggleWatchlist,
}) => {
  const {
    current_price,
    market_cap,
    pe_ratio,
    forward_pe,
    fifty_two_week_high,
    fifty_two_week_low,
    market_status,
    price_source,
    beta,
    change,
    change_percent,
  } = marketData;

  // 使用 trailing PE，如果没有则使用 forward PE
  const displayPE = pe_ratio || forward_pe;

  // 价格变动
  const priceChange = change || 0;
  const priceChangePercent = change_percent || 0;
  const isUp = priceChange >= 0;

  // 52周位置
  const range52w = fifty_two_week_high && fifty_two_week_low
    ? ((current_price - fifty_two_week_low) / (fifty_two_week_high - fifty_two_week_low) * 100).toFixed(1)
    : null;

  const formatMarketCap = (value?: number) => {
    if (!value) return 'N/A';
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return `$${value.toLocaleString()}`;
  };

  // 主题相关颜色
  const { theme } = useSettingsStore();
  const isDark = theme === 'dark';
  const cardBg = isDark
    ? 'linear-gradient(135deg, #161b22 0%, #1c2128 100%)'
    : 'linear-gradient(135deg, #ffffff 0%, #f5f5f5 100%)';
  const secondaryColor = isDark ? '#8b949e' : '#595959';

  return (
    <Card
      style={{
        marginBottom: 24,
        background: cardBg,
      }}
      styles={{ body: { padding: 24 } }}
    >
      <Row gutter={[24, 24]} align="middle">
        {/* 股票信息 */}
        <Col xs={24} md={8}>
          <Space direction="vertical" size={4}>
            <Space align="center">
              <Title level={2} style={{ margin: 0 }}>
                {symbol}
              </Title>
              <Tag
                color={isInWatchlist ? 'gold' : 'default'}
                style={{ cursor: 'pointer' }}
                onClick={onToggleWatchlist}
              >
                {isInWatchlist ? <StarFilled /> : <StarOutlined />}
              </Tag>
            </Space>

            {market_status && (
              <Space size={8}>
                <Tag color="blue">{market_status}</Tag>
                {price_source && (
                  <Tooltip title={price_source}>
                    <InfoCircleOutlined style={{ color: secondaryColor, cursor: 'help' }} />
                  </Tooltip>
                )}
              </Space>
            )}
          </Space>
        </Col>

        {/* 价格 */}
        <Col xs={24} md={8}>
          <Space direction="vertical" size={0} style={{ textAlign: 'center', width: '100%' }}>
            <Statistic
              value={current_price}
              precision={2}
              prefix="$"
              valueStyle={{
                fontSize: 36,
                fontWeight: 700,
              }}
            />
            <Space>
              <Text
                style={{
                  color: isUp ? colors.up : colors.down,
                  fontSize: 16,
                }}
              >
                {isUp ? <ArrowUpOutlined /> : <ArrowDownOutlined />}
                {isUp ? '+' : ''}{priceChange.toFixed(2)} ({isUp ? '+' : ''}{priceChangePercent.toFixed(2)}%)
              </Text>
            </Space>
          </Space>
        </Col>

        {/* 关键指标 */}
        <Col xs={24} md={8}>
          <Row gutter={[16, 8]}>
            <Col span={12}>
              <Space direction="vertical" size={0}>
                <Text type="secondary" style={{ fontSize: 12 }}>Market Cap</Text>
                <Text strong>{formatMarketCap(market_cap)}</Text>
              </Space>
            </Col>
            <Col span={12}>
              <Space direction="vertical" size={0}>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {pe_ratio ? 'P/E Ratio' : forward_pe ? 'Forward P/E' : 'P/E Ratio'}
                </Text>
                <Text strong>{displayPE?.toFixed(2) || 'N/A'}</Text>
              </Space>
            </Col>
            <Col span={12}>
              <Space direction="vertical" size={0}>
                <Text type="secondary" style={{ fontSize: 12 }}>Beta</Text>
                <Text strong>{beta?.toFixed(3) || 'N/A'}</Text>
              </Space>
            </Col>
            <Col span={12}>
              <Space direction="vertical" size={0}>
                <Text type="secondary" style={{ fontSize: 12 }}>52W Range</Text>
                <Text strong>{range52w}%</Text>
              </Space>
            </Col>
          </Row>
        </Col>
      </Row>
    </Card>
  );
};

export default StockHeader;
