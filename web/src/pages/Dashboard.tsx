import React from 'react';
import { Card, Row, Col, Typography, Space, Input, List, Tag, Empty } from 'antd';
import {
  SearchOutlined,
  LineChartOutlined,
  RiseOutlined,
  ClockCircleOutlined,
  StarOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAnalysisStore } from '../stores/analysisStore';
import { useSettingsStore } from '../stores/settingsStore';

const { Title, Text, Paragraph } = Typography;

const Dashboard: React.FC = () => {
  const navigate = useNavigate();
  const { recentSymbols } = useAnalysisStore();
  const { t } = useTranslation();
  const { theme } = useSettingsStore();

  // Theme-aware colors
  const isDark = theme === 'dark';
  const textColor = isDark ? '#e6edf3' : '#1f1f1f';
  const secondaryColor = isDark ? '#8b949e' : '#595959';
  const cardBg = isDark ? '#1c2128' : '#ffffff';
  const borderColor = isDark ? '#30363d' : '#e8e8e8';

  const handleSearch = (value: string) => {
    if (value.trim()) {
      navigate(`/analysis/${value.trim().toUpperCase()}`);
    }
  };

  // 热门股票示例数据
  const popularStocks = [
    { symbol: 'AAPL', name: 'Apple Inc.', price: 246.43, change: -0.29 },
    { symbol: 'TSLA', name: 'Tesla Inc.', price: 420.09, change: 0.52 },
    { symbol: 'NVDA', name: 'NVIDIA Corp.', price: 142.65, change: 1.23 },
    { symbol: 'MSFT', name: 'Microsoft Corp.', price: 428.50, change: -0.15 },
    { symbol: 'GOOGL', name: 'Alphabet Inc.', price: 321.58, change: -0.37 },
    { symbol: 'AMZN', name: 'Amazon.com Inc.', price: 234.85, change: 0.85 },
  ];

  return (
    <div>
      {/* Hero Section */}
      <Card
        style={{
          marginBottom: 24,
          background: isDark
            ? 'linear-gradient(135deg, #1890ff15 0%, #722ed115 100%)'
            : 'linear-gradient(135deg, #1890ff10 0%, #722ed110 100%)',
          border: '1px solid #1890ff30',
        }}
      >
        <Row gutter={24} align="middle">
          <Col xs={24} md={16}>
            <Space direction="vertical" size={16}>
              <Title level={2} style={{ margin: 0, color: textColor }}>
                <LineChartOutlined style={{ marginRight: 12, color: '#1890ff' }} />
                {t('dashboard.title')}
              </Title>
              <Paragraph style={{ fontSize: 16, color: secondaryColor, margin: 0 }}>
                {t('dashboard.subtitle')}
              </Paragraph>
              <Input.Search
                placeholder={t('dashboard.searchPlaceholder')}
                allowClear
                enterButton={t('dashboard.analyze')}
                size="large"
                onSearch={handleSearch}
                style={{ maxWidth: 500 }}
                prefix={<SearchOutlined style={{ color: secondaryColor }} />}
              />
            </Space>
          </Col>
          <Col xs={0} md={8} style={{ textAlign: 'center' }}>
            <LineChartOutlined style={{ fontSize: 120, color: '#1890ff20' }} />
          </Col>
        </Row>
      </Card>

      <Row gutter={24}>
        {/* Popular Stocks */}
        <Col xs={24} lg={16}>
          <Card
            title={
              <Space>
                <RiseOutlined />
                <span>{t('dashboard.popularStocks')}</span>
              </Space>
            }
          >
            <Row gutter={[16, 16]}>
              {popularStocks.map((stock) => (
                <Col xs={12} sm={8} md={6} lg={8} xl={6} key={stock.symbol}>
                  <Card
                    hoverable
                    size="small"
                    onClick={() => navigate(`/analysis/${stock.symbol}`)}
                    style={{
                      background: cardBg,
                      border: `1px solid ${borderColor}`,
                    }}
                  >
                    <Space direction="vertical" size={4} style={{ width: '100%' }}>
                      <Text strong style={{ color: textColor, fontSize: 16 }}>
                        {stock.symbol}
                      </Text>
                      <Text type="secondary" ellipsis style={{ fontSize: 12 }}>
                        {stock.name}
                      </Text>
                      <Space style={{ width: '100%', justifyContent: 'space-between' }}>
                        <Text style={{ color: textColor }}>
                          ${stock.price.toFixed(2)}
                        </Text>
                        <Tag
                          color={stock.change >= 0 ? 'success' : 'error'}
                          style={{ margin: 0 }}
                        >
                          {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}%
                        </Tag>
                      </Space>
                    </Space>
                  </Card>
                </Col>
              ))}
            </Row>
          </Card>
        </Col>

        {/* Recent & Watchlist */}
        <Col xs={24} lg={8}>
          <Space direction="vertical" size={24} style={{ width: '100%' }}>
            {/* Recent Searches */}
            <Card
              title={
                <Space>
                  <ClockCircleOutlined />
                  <span>{t('dashboard.recentSearches')}</span>
                </Space>
              }
              size="small"
            >
              {recentSymbols.length > 0 ? (
                <List
                  size="small"
                  dataSource={recentSymbols.slice(0, 5)}
                  renderItem={(symbol) => (
                    <List.Item
                      style={{ cursor: 'pointer', padding: '8px 0' }}
                      onClick={() => navigate(`/analysis/${symbol}`)}
                    >
                      <Text style={{ color: textColor }}>{symbol}</Text>
                    </List.Item>
                  )}
                />
              ) : (
                <Empty
                  image={Empty.PRESENTED_IMAGE_SIMPLE}
                  description={t('dashboard.noRecentSearches')}
                />
              )}
            </Card>

            {/* Quick Tips */}
            <Card
              title={
                <Space>
                  <StarOutlined />
                  <span>{t('dashboard.quickTips')}</span>
                </Space>
              }
              size="small"
            >
              <Space direction="vertical" size={8}>
                <Text type="secondary" style={{ fontSize: 13 }}>
                  • {t('dashboard.tip1')}
                </Text>
                <Text type="secondary" style={{ fontSize: 13 }}>
                  • {t('dashboard.tip2')}
                </Text>
                <Text type="secondary" style={{ fontSize: 13 }}>
                  • {t('dashboard.tip3')}
                </Text>
                <Text type="secondary" style={{ fontSize: 13 }}>
                  • {t('dashboard.tip4')}
                </Text>
              </Space>
            </Card>
          </Space>
        </Col>
      </Row>
    </div>
  );
};

export default Dashboard;
