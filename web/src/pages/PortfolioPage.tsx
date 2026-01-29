import React, { useEffect, useState, useMemo } from 'react';
import {
  Card,
  Row,
  Col,
  Typography,
  Space,
  Spin,
  Alert,
  Table,
  Tag,
  Progress,
  Statistic,
  Button,
  Tooltip,
  Empty,
} from 'antd';
import {
  PieChartOutlined,
  LineChartOutlined,
  SafetyOutlined,
  ReloadOutlined,
  BankOutlined,
  WarningOutlined,
  CheckCircleOutlined,
  InfoCircleOutlined,
  SettingOutlined,
  SwapOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import { useNavigate } from 'react-router-dom';
import { useSettingsStore } from '../stores/settingsStore';
import { brokerApi, portfolioApi } from '../services/brokerApi';
import type { UnifiedPortfolio, PortfolioAnalysis, Trade } from '../types/broker';

const { Title, Text } = Typography;

// 健康评分颜色
const getScoreColor = (score: number): string => {
  if (score >= 80) return '#52c41a';
  if (score >= 60) return '#1890ff';
  if (score >= 40) return '#faad14';
  return '#ff4d4f';
};

// 风险等级标签
const getRiskLevel = (score: number, isZh: boolean): { level: string; color: string } => {
  if (score <= 30) return { level: isZh ? '低风险' : 'Low Risk', color: 'success' };
  if (score <= 50) return { level: isZh ? '中等风险' : 'Medium Risk', color: 'processing' };
  if (score <= 70) return { level: isZh ? '较高风险' : 'High Risk', color: 'warning' };
  return { level: isZh ? '高风险' : 'Very High Risk', color: 'error' };
};

// 行动建议颜色
const getActionColor = (action: string): string => {
  switch (action) {
    case 'hold': return 'default';
    case 'increase': return 'success';
    case 'reduce': return 'warning';
    case 'sell': return 'error';
    case 'watch': return 'processing';
    default: return 'default';
  }
};

const PortfolioPage: React.FC = () => {
  const { i18n } = useTranslation();
  const navigate = useNavigate();
  const { theme } = useSettingsStore();
  const isZh = i18n.language?.startsWith('zh');
  const isDark = theme === 'dark';

  // 状态
  const [loading, setLoading] = useState(true);
  const [portfolio, setPortfolio] = useState<UnifiedPortfolio | null>(null);
  const [analysis, setAnalysis] = useState<PortfolioAnalysis | null>(null);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [tradesLoading, setTradesLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // 主题颜色
  const secondaryColor = isDark ? '#8b949e' : '#595959';

  // 加载数据
  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [portfolioRes, analysisRes] = await Promise.all([
        brokerApi.getUnifiedPortfolio().catch(() => null),
        portfolioApi.analyze().catch(() => null),
      ]);

      // broker API 直接返回数据
      if (portfolioRes && (portfolioRes as any).total_assets !== undefined) {
        setPortfolio(portfolioRes as any);
      }
      // portfolio analyze API 返回 {success, data} 格式
      if ((analysisRes as any)?.success && (analysisRes as any)?.data) {
        setAnalysis((analysisRes as any).data);
      }
      // 如果两个 API 都失败，不设置错误，而是显示空状态提示用户连接券商
    } catch (err: any) {
      // 忽略错误，显示空状态
      console.error('Portfolio load error:', err);
    } finally {
      setLoading(false);
    }
  };

  // 加载交易历史
  const loadTrades = async (days: number = 7) => {
    setTradesLoading(true);
    try {
      // 获取所有已连接的券商
      const statusRes = await brokerApi.getStatus().catch(() => null);
      const connectedBrokers = (statusRes as any)?.brokers?.filter((b: any) => b.connected) || [];

      // 从所有已连接券商获取交易历史
      const allTrades: Trade[] = [];
      for (const broker of connectedBrokers) {
        try {
          const tradesRes = await brokerApi.getTrades(broker.broker_type, days);
          if (Array.isArray(tradesRes)) {
            allTrades.push(...tradesRes);
          }
        } catch (err) {
          console.error(`Failed to load trades from ${broker.broker_type}:`, err);
        }
      }

      // 按交易时间排序（最新在前）
      allTrades.sort((a, b) => {
        const timeA = a.trade_time ? new Date(a.trade_time).getTime() : 0;
        const timeB = b.trade_time ? new Date(b.trade_time).getTime() : 0;
        return timeB - timeA;
      });

      setTrades(allTrades);
    } catch (err) {
      console.error('Failed to load trades:', err);
    } finally {
      setTradesLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    loadTrades();
  }, []);

  // 持仓表格列 - 匹配 API 返回的 top_holdings 结构
  const positionColumns = useMemo(() => [
    {
      title: isZh ? '股票' : 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (symbol: string) => (
        <Button type="link" size="small" onClick={() => navigate(`/analysis/${symbol}`)}>
          {symbol}
        </Button>
      ),
    },
    {
      title: isZh ? '数量' : 'Quantity',
      dataIndex: 'total_quantity',
      key: 'total_quantity',
      align: 'right' as const,
      render: (val: number) => val?.toLocaleString() || '-',
    },
    {
      title: isZh ? '成本价' : 'Avg Cost',
      dataIndex: 'avg_cost',
      key: 'avg_cost',
      align: 'right' as const,
      render: (val: number) => val ? `$${val.toFixed(2)}` : '-',
    },
    {
      title: isZh ? '现价' : 'Price',
      dataIndex: 'current_price',
      key: 'current_price',
      align: 'right' as const,
      render: (val: number) => val ? `$${val.toFixed(2)}` : '-',
    },
    {
      title: isZh ? '市值' : 'Value',
      dataIndex: 'total_market_value',
      key: 'total_market_value',
      align: 'right' as const,
      render: (val: number) => val ? `$${val.toLocaleString(undefined, { minimumFractionDigits: 2 })}` : '-',
      sorter: (a: any, b: any) => (a.total_market_value || 0) - (b.total_market_value || 0),
      defaultSortOrder: 'descend' as const,
    },
    {
      title: isZh ? '盈亏' : 'P&L',
      dataIndex: 'total_unrealized_pnl',
      key: 'total_unrealized_pnl',
      align: 'right' as const,
      render: (val: number, record: any) => (
        <Space direction="vertical" size={0} style={{ textAlign: 'right' }}>
          <Text style={{ color: (val || 0) >= 0 ? '#52c41a' : '#ff4d4f' }}>
            {(val || 0) >= 0 ? '+' : ''}{(val || 0).toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </Text>
          <Text type="secondary" style={{ fontSize: 12 }}>
            {(record.unrealized_pnl_percent || 0) >= 0 ? '+' : ''}{(record.unrealized_pnl_percent || 0).toFixed(2)}%
          </Text>
        </Space>
      ),
      sorter: (a: any, b: any) => (a.total_unrealized_pnl || 0) - (b.total_unrealized_pnl || 0),
    },
    {
      title: isZh ? '券商' : 'Broker',
      dataIndex: 'brokers',
      key: 'brokers',
      render: (brokers: string[]) => {
        const names: Record<string, string> = {
          IBKR: isZh ? '盈透' : 'IBKR',
          Futu: isZh ? '富途' : 'Futu',
          Tiger: isZh ? '老虎' : 'Tiger',
        };
        return brokers?.map(b => <Tag key={b}>{names[b] || b}</Tag>) || '-';
      },
    },
  ], [isZh, navigate]);

  // 建议表格列
  const recommendationColumns = useMemo(() => [
    {
      title: isZh ? '股票' : 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (symbol: string) => (
        <Button type="link" size="small" onClick={() => navigate(`/analysis/${symbol}`)}>
          {symbol}
        </Button>
      ),
    },
    {
      title: isZh ? '建议' : 'Action',
      dataIndex: 'action',
      key: 'action',
      render: (action: string) => {
        const actionLabels: Record<string, { en: string; zh: string }> = {
          hold: { en: 'Hold', zh: '持有' },
          increase: { en: 'Increase', zh: '加仓' },
          reduce: { en: 'Reduce', zh: '减仓' },
          sell: { en: 'Sell', zh: '卖出' },
          watch: { en: 'Watch', zh: '观察' },
        };
        return (
          <Tag color={getActionColor(action)}>
            {isZh ? actionLabels[action]?.zh : actionLabels[action]?.en || action}
          </Tag>
        );
      },
    },
    {
      title: isZh ? '当前权重' : 'Current Weight',
      dataIndex: 'current_weight',
      key: 'current_weight',
      align: 'right' as const,
      render: (val: number) => `${(val * 100).toFixed(1)}%`,
    },
    {
      title: isZh ? '优先级' : 'Priority',
      dataIndex: 'priority',
      key: 'priority',
      render: (priority: string) => {
        const colors: Record<string, string> = { low: 'default', medium: 'processing', high: 'error' };
        const labels: Record<string, { en: string; zh: string }> = {
          low: { en: 'Low', zh: '低' },
          medium: { en: 'Medium', zh: '中' },
          high: { en: 'High', zh: '高' },
        };
        return <Tag color={colors[priority]}>{isZh ? labels[priority]?.zh : labels[priority]?.en}</Tag>;
      },
    },
    {
      title: isZh ? '理由' : 'Reason',
      dataIndex: 'reason',
      key: 'reason',
      ellipsis: true,
    },
  ], [isZh, navigate]);

  // 交易历史表格列
  const tradeColumns = useMemo(() => [
    {
      title: isZh ? '时间' : 'Time',
      dataIndex: 'trade_time',
      key: 'trade_time',
      width: 160,
      render: (val: string) => val ? new Date(val).toLocaleString() : '-',
    },
    {
      title: isZh ? '股票' : 'Symbol',
      dataIndex: 'symbol',
      key: 'symbol',
      render: (symbol: string) => (
        <Button type="link" size="small" onClick={() => navigate(`/analysis/${symbol}`)}>
          {symbol}
        </Button>
      ),
    },
    {
      title: isZh ? '操作' : 'Action',
      dataIndex: 'action',
      key: 'action',
      render: (action: string) => (
        <Tag color={action === 'buy' ? 'success' : 'error'}>
          {action === 'buy' ? (isZh ? '买入' : 'BUY') : (isZh ? '卖出' : 'SELL')}
        </Tag>
      ),
    },
    {
      title: isZh ? '数量' : 'Quantity',
      dataIndex: 'quantity',
      key: 'quantity',
      align: 'right' as const,
      render: (val: number) => val?.toLocaleString() || '-',
    },
    {
      title: isZh ? '价格' : 'Price',
      dataIndex: 'price',
      key: 'price',
      align: 'right' as const,
      render: (val: number) => val ? `$${val.toFixed(2)}` : '-',
    },
    {
      title: isZh ? '金额' : 'Total',
      dataIndex: 'total_value',
      key: 'total_value',
      align: 'right' as const,
      render: (val: number) => val ? `$${val.toLocaleString(undefined, { minimumFractionDigits: 2 })}` : '-',
    },
    {
      title: isZh ? '佣金' : 'Commission',
      dataIndex: 'commission',
      key: 'commission',
      align: 'right' as const,
      render: (val: number) => val ? `$${val.toFixed(2)}` : '-',
    },
    {
      title: isZh ? '已实现盈亏' : 'Realized P&L',
      dataIndex: 'realized_pnl',
      key: 'realized_pnl',
      align: 'right' as const,
      render: (val: number | null) => {
        if (val === null || val === undefined) return '-';
        return (
          <Text style={{ color: val >= 0 ? '#52c41a' : '#ff4d4f' }}>
            {val >= 0 ? '+' : ''}{val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
          </Text>
        );
      },
    },
  ], [isZh, navigate]);

  // 如果没有连接任何券商
  const hasNoBrokers = !loading && (!portfolio || (portfolio as any).broker_count === 0);

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: 400 }}>
        <Spin size="large" />
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        message={isZh ? '加载失败' : 'Failed to load'}
        description={error}
        type="error"
        showIcon
        action={
          <Button size="small" onClick={loadData}>
            {isZh ? '重试' : 'Retry'}
          </Button>
        }
      />
    );
  }

  if (hasNoBrokers) {
    return (
      <div>
        <Title level={3} style={{ marginBottom: 24 }}>
          <PieChartOutlined style={{ marginRight: 12 }} />
          {isZh ? '投资组合' : 'Portfolio'}
        </Title>
        <Card>
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={
              <Space direction="vertical" size={8}>
                <Text>{isZh ? '尚未连接任何券商账户' : 'No broker accounts connected'}</Text>
                <Text type="secondary">
                  {isZh ? '连接您的券商以查看投资组合分析' : 'Connect your broker to view portfolio analysis'}
                </Text>
              </Space>
            }
          >
            <Button
              type="primary"
              icon={<SettingOutlined />}
              onClick={() => navigate('/settings')}
            >
              {isZh ? '前往设置' : 'Go to Settings'}
            </Button>
          </Empty>
        </Card>
      </div>
    );
  }

  const riskLevel = analysis ? getRiskLevel(analysis.risk_score, isZh) : null;

  return (
    <div>
      {/* Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
        <Title level={3} style={{ margin: 0 }}>
          <PieChartOutlined style={{ marginRight: 12 }} />
          {isZh ? '投资组合' : 'Portfolio'}
        </Title>
        <Space>
          <Text type="secondary">
            {isZh ? '最后更新' : 'Updated'}: {portfolio?.last_updated ? new Date(portfolio.last_updated).toLocaleString() : '-'}
          </Text>
          <Button icon={<ReloadOutlined />} onClick={loadData}>
            {isZh ? '刷新' : 'Refresh'}
          </Button>
        </Space>
      </div>

      {/* Summary Cards */}
      <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title={isZh ? '总资产' : 'Total Value'}
              value={(portfolio as any)?.total_assets || 0}
              precision={2}
              prefix="$"
              valueStyle={{ color: '#1890ff' }}
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title={isZh ? '持仓市值' : 'Market Value'}
              value={portfolio?.total_market_value || 0}
              precision={2}
              prefix="$"
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title={isZh ? '未实现盈亏' : 'Unrealized P&L'}
              value={portfolio?.total_unrealized_pnl || 0}
              precision={2}
              prefix={portfolio?.total_unrealized_pnl && portfolio.total_unrealized_pnl >= 0 ? '+$' : '$'}
              valueStyle={{ color: (portfolio?.total_unrealized_pnl || 0) >= 0 ? '#52c41a' : '#ff4d4f' }}
              suffix={
                portfolio?.total_market_value ? (
                  <Text type="secondary" style={{ fontSize: 14 }}>
                    ({(((portfolio?.total_unrealized_pnl || 0) / portfolio.total_market_value) * 100).toFixed(2)}%)
                  </Text>
                ) : null
              }
            />
          </Card>
        </Col>
        <Col xs={24} sm={12} lg={6}>
          <Card size="small">
            <Statistic
              title={isZh ? '现金余额' : 'Cash'}
              value={portfolio?.total_cash || 0}
              precision={2}
              prefix="$"
              valueStyle={{ color: secondaryColor }}
            />
          </Card>
        </Col>
      </Row>

      {/* Scores and Risk Metrics */}
      {analysis && (
        <Row gutter={[16, 16]} style={{ marginBottom: 24 }}>
          <Col xs={24} lg={8}>
            <Card
              title={
                <Space>
                  <SafetyOutlined />
                  <span>{isZh ? '健康评分' : 'Health Score'}</span>
                </Space>
              }
              size="small"
            >
              <div style={{ textAlign: 'center', padding: '16px 0' }}>
                <Progress
                  type="dashboard"
                  percent={analysis.health_score}
                  strokeColor={getScoreColor(analysis.health_score)}
                  format={(percent) => (
                    <div>
                      <div style={{ fontSize: 28, fontWeight: 'bold' }}>{percent}</div>
                      <div style={{ fontSize: 12, color: secondaryColor }}>/ 100</div>
                    </div>
                  )}
                />
                <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
                  {analysis.health_score >= 80
                    ? (isZh ? '投资组合状态良好' : 'Portfolio is healthy')
                    : analysis.health_score >= 60
                    ? (isZh ? '投资组合状态一般' : 'Portfolio needs attention')
                    : (isZh ? '投资组合需要优化' : 'Portfolio needs optimization')}
                </Text>
              </div>
            </Card>
          </Col>

          <Col xs={24} lg={8}>
            <Card
              title={
                <Space>
                  <WarningOutlined />
                  <span>{isZh ? '风险评分' : 'Risk Score'}</span>
                </Space>
              }
              size="small"
            >
              <div style={{ textAlign: 'center', padding: '16px 0' }}>
                <Progress
                  type="dashboard"
                  percent={analysis.risk_score}
                  strokeColor={getScoreColor(100 - analysis.risk_score)}
                  format={(percent) => (
                    <div>
                      <div style={{ fontSize: 28, fontWeight: 'bold' }}>{percent}</div>
                      <div style={{ fontSize: 12, color: secondaryColor }}>/ 100</div>
                    </div>
                  )}
                />
                {riskLevel && (
                  <Tag color={riskLevel.color} style={{ marginTop: 8 }}>
                    {riskLevel.level}
                  </Tag>
                )}
              </div>
            </Card>
          </Col>

          <Col xs={24} lg={8}>
            <Card
              title={
                <Space>
                  <PieChartOutlined />
                  <span>{isZh ? '分散度评分' : 'Diversification'}</span>
                </Space>
              }
              size="small"
            >
              <div style={{ textAlign: 'center', padding: '16px 0' }}>
                <Progress
                  type="dashboard"
                  percent={analysis.diversification_score}
                  strokeColor={getScoreColor(analysis.diversification_score)}
                  format={(percent) => (
                    <div>
                      <div style={{ fontSize: 28, fontWeight: 'bold' }}>{percent}</div>
                      <div style={{ fontSize: 12, color: secondaryColor }}>/ 100</div>
                    </div>
                  )}
                />
                <Text type="secondary" style={{ display: 'block', marginTop: 8 }}>
                  {analysis.diversification_score >= 70
                    ? (isZh ? '分散化良好' : 'Well diversified')
                    : (isZh ? '集中度较高' : 'Concentrated portfolio')}
                </Text>
              </div>
            </Card>
          </Col>
        </Row>
      )}

      {/* Risk Metrics */}
      {analysis?.risk_metrics && (
        <Card
          title={
            <Space>
              <LineChartOutlined />
              <span>{isZh ? '风险指标' : 'Risk Metrics'}</span>
            </Space>
          }
          size="small"
          style={{ marginBottom: 24 }}
        >
          <Row gutter={[24, 16]}>
            <Col xs={12} sm={8} md={4}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '95%置信度下的最大损失' : 'Maximum loss at 95% confidence'}>
                    <Space>
                      VaR (95%)
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={analysis.risk_metrics.var_95}
                precision={2}
                prefix="$"
                valueStyle={{ color: '#ff4d4f', fontSize: 18 }}
              />
            </Col>
            <Col xs={12} sm={8} md={4}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '99%置信度下的最大损失' : 'Maximum loss at 99% confidence'}>
                    <Space>
                      VaR (99%)
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={analysis.risk_metrics.var_99}
                precision={2}
                prefix="$"
                valueStyle={{ color: '#ff4d4f', fontSize: 18 }}
              />
            </Col>
            <Col xs={12} sm={8} md={4}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '年化波动率' : 'Annualized volatility'}>
                    <Space>
                      {isZh ? '波动率' : 'Volatility'}
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={(analysis.risk_metrics.volatility * 100)}
                precision={1}
                suffix="%"
                valueStyle={{ fontSize: 18 }}
              />
            </Col>
            <Col xs={12} sm={8} md={4}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '风险调整后收益' : 'Risk-adjusted return'}>
                    <Space>
                      {isZh ? '夏普比率' : 'Sharpe'}
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={analysis.risk_metrics.sharpe_ratio}
                precision={2}
                valueStyle={{ color: analysis.risk_metrics.sharpe_ratio >= 1 ? '#52c41a' : secondaryColor, fontSize: 18 }}
              />
            </Col>
            <Col xs={12} sm={8} md={4}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '最大回撤幅度' : 'Maximum drawdown'}>
                    <Space>
                      {isZh ? '最大回撤' : 'Max DD'}
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={(analysis.risk_metrics.max_drawdown * 100)}
                precision={1}
                suffix="%"
                valueStyle={{ color: '#ff4d4f', fontSize: 18 }}
              />
            </Col>
            <Col xs={12} sm={8} md={4}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '相对于市场的波动性' : 'Volatility relative to market'}>
                    <Space>
                      Beta
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={analysis.risk_metrics.beta}
                precision={2}
                valueStyle={{ fontSize: 18 }}
              />
            </Col>
          </Row>
        </Card>
      )}

      {/* Positions Table */}
      <Card
        title={
          <Space>
            <BankOutlined />
            <span>{isZh ? '持仓明细' : 'Positions'}</span>
            <Tag>{(portfolio as any)?.position_count || 0}</Tag>
          </Space>
        }
        size="small"
        style={{ marginBottom: 24 }}
      >
        <Table
          dataSource={(portfolio as any)?.top_holdings || []}
          columns={positionColumns}
          rowKey={(record) => `${record.symbol}-${record.market || 'US'}`}
          pagination={{ pageSize: 10, showSizeChanger: true }}
          size="small"
          scroll={{ x: 800 }}
        />
      </Card>

      {/* Trade History */}
      <Card
        title={
          <Space>
            <SwapOutlined />
            <span>{isZh ? '交易历史' : 'Trade History'}</span>
            <Tag>{trades.length}</Tag>
          </Space>
        }
        size="small"
        style={{ marginBottom: 24 }}
        extra={
          <Button
            size="small"
            icon={<ReloadOutlined />}
            loading={tradesLoading}
            onClick={() => loadTrades()}
          >
            {isZh ? '刷新' : 'Refresh'}
          </Button>
        }
      >
        {tradesLoading ? (
          <div style={{ textAlign: 'center', padding: 24 }}>
            <Spin />
          </div>
        ) : trades.length === 0 ? (
          <Empty
            image={Empty.PRESENTED_IMAGE_SIMPLE}
            description={isZh ? '暂无交易记录' : 'No trade history'}
          />
        ) : (
          <Table
            dataSource={trades}
            columns={tradeColumns}
            rowKey={(record, index) => `${record.execution_id || record.order_id || index}`}
            pagination={{ pageSize: 10, showSizeChanger: true }}
            size="small"
            scroll={{ x: 900 }}
          />
        )}
      </Card>

      {/* Recommendations */}
      {analysis?.recommendations && analysis.recommendations.length > 0 && (
        <Card
          title={
            <Space>
              <CheckCircleOutlined />
              <span>{isZh ? '持仓建议' : 'Recommendations'}</span>
            </Space>
          }
          size="small"
        >
          <Table
            dataSource={analysis.recommendations}
            columns={recommendationColumns}
            rowKey="symbol"
            pagination={false}
            size="small"
          />
        </Card>
      )}

      {/* Concentration Risk */}
      {analysis?.concentration_risk && (
        <Card
          title={
            <Space>
              <WarningOutlined />
              <span>{isZh ? '集中度风险' : 'Concentration Risk'}</span>
            </Space>
          }
          size="small"
          style={{ marginTop: 24 }}
        >
          <Row gutter={[24, 16]}>
            <Col xs={12} sm={6}>
              <Statistic
                title={isZh ? '最大持仓占比' : 'Top Holding'}
                value={(analysis.concentration_risk.top_holding_weight * 100)}
                precision={1}
                suffix="%"
                valueStyle={{
                  color: analysis.concentration_risk.top_holding_weight > 0.3 ? '#ff4d4f' : undefined,
                }}
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title={isZh ? '前三大持仓' : 'Top 3 Holdings'}
                value={(analysis.concentration_risk.top_3_weight * 100)}
                precision={1}
                suffix="%"
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title={isZh ? '前五大持仓' : 'Top 5 Holdings'}
                value={(analysis.concentration_risk.top_5_weight * 100)}
                precision={1}
                suffix="%"
              />
            </Col>
            <Col xs={12} sm={6}>
              <Statistic
                title={
                  <Tooltip title={isZh ? '赫芬达尔指数，越低表示越分散' : 'Herfindahl Index, lower means more diversified'}>
                    <Space>
                      HHI
                      <InfoCircleOutlined style={{ fontSize: 12 }} />
                    </Space>
                  </Tooltip>
                }
                value={(analysis.concentration_risk.hhi_index * 100)}
                precision={1}
                suffix="%"
              />
            </Col>
          </Row>
        </Card>
      )}
    </div>
  );
};

export default PortfolioPage;
