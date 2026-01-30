import React, { useEffect, useCallback } from 'react';
import { Card, Table, Tag, Button, Space, Typography, Empty, Popconfirm, Skeleton } from 'antd';
import {
  StarOutlined,
  DeleteOutlined,
  LineChartOutlined,
  ReloadOutlined,
} from '@ant-design/icons';
import { useNavigate } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { useAnalysisStore } from '../stores/analysisStore';
import type { Quote } from '../types/analysis';

const { Title, Text } = Typography;

const AUTO_REFRESH_INTERVAL = 60000; // 60 秒自动刷新

const WatchlistPage: React.FC = () => {
  const { t } = useTranslation();
  const navigate = useNavigate();
  const { watchlist, removeFromWatchlist, refreshWatchlist } = useAnalysisStore();
  const [loading, setLoading] = React.useState(false);

  const handleRefresh = useCallback(async () => {
    setLoading(true);
    try {
      await refreshWatchlist();
    } finally {
      setLoading(false);
    }
  }, [refreshWatchlist]);

  // 挂载时刷新一次
  useEffect(() => {
    if (watchlist.length > 0) {
      handleRefresh();
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // 自动定时刷新
  useEffect(() => {
    if (watchlist.length === 0) return;

    const timer = setInterval(() => {
      handleRefresh();
    }, AUTO_REFRESH_INTERVAL);

    return () => clearInterval(timer);
  }, [watchlist.length, handleRefresh]);


  const columns = [
    {
      title: t('market.symbol') || 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (symbol: string) => (
        <Button
          type="link"
          onClick={() => navigate(`/analysis/${symbol}`)}
          style={{ padding: 0, fontWeight: 600 }}
        >
          {symbol}
        </Button>
      ),
    },
    {
      title: t('market.price'),
      dataIndex: 'price',
      key: 'price',
      render: (price: number) => (
        <Text strong>${price.toFixed(2)}</Text>
      ),
    },
    {
      title: t('market.change'),
      dataIndex: 'change',
      key: 'change',
      render: (_: number, record: Quote) => {
        const isPositive = record.change >= 0;
        return (
          <Tag color={isPositive ? 'success' : 'error'}>
            {isPositive ? '+' : ''}{record.change.toFixed(2)} ({isPositive ? '+' : ''}{record.change_percent.toFixed(2)}%)
          </Tag>
        );
      },
    },
    {
      title: t('market.volume'),
      dataIndex: 'volume',
      key: 'volume',
      render: (volume: number) => (
        <Text type="secondary">{volume?.toLocaleString() || '-'}</Text>
      ),
    },
    {
      title: '',
      key: 'actions',
      width: 120,
      render: (_: unknown, record: Quote) => (
        <Space>
          <Button
            type="text"
            icon={<LineChartOutlined />}
            onClick={() => navigate(`/analysis/${record.symbol}`)}
          />
          <Popconfirm
            title={t('common.confirm')}
            onConfirm={() => removeFromWatchlist(record.symbol)}
          >
            <Button type="text" danger icon={<DeleteOutlined />} />
          </Popconfirm>
        </Space>
      ),
    },
  ];

  return (
    <div>
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 24,
        }}
      >
        <Title level={3} style={{ margin: 0 }}>
          <StarOutlined style={{ marginRight: 12 }} />
          {t('nav.watchlist')}
        </Title>
        <Button
          icon={<ReloadOutlined />}
          onClick={handleRefresh}
          loading={loading}
        >
          {t('common.refresh')}
        </Button>
      </div>

      <Card>
        {loading && watchlist.length === 0 ? (
          // 首次加载时显示骨架屏
          <Skeleton active paragraph={{ rows: 4 }} />
        ) : watchlist.length > 0 ? (
          <Table
            dataSource={watchlist}
            columns={columns}
            rowKey="symbol"
            pagination={false}
            loading={loading}
          />
        ) : (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={
              <Space direction="vertical">
                <Text type="secondary">{t('watchlist.empty') || 'No stocks in watchlist'}</Text>
                <Text type="secondary" style={{ fontSize: 12 }}>
                  {t('watchlist.addHint') || 'Search for a stock and click the star icon to add it'}
                </Text>
              </Space>
            }
          />
        )}
      </Card>
    </div>
  );
};

export default WatchlistPage;
