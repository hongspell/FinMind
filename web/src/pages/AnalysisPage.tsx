import React, { useEffect, useState, useCallback } from 'react';
import { useParams } from 'react-router-dom';
import { Row, Col, Spin, Alert, Space, Typography, Button, message, Tabs } from 'antd';
import { ReloadOutlined, DownloadOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import StockHeader from '../components/Stock/StockHeader';
import TechnicalOverview from '../components/Analysis/TechnicalOverview';
import TimeframeAnalysis from '../components/Analysis/TimeframeAnalysis';
import RiskAnalysis from '../components/Analysis/RiskAnalysis';
import SensitivityAnalysis from '../components/Analysis/SensitivityAnalysis';
import BacktestPanel from '../components/Analysis/BacktestPanel';
import PriceChart from '../components/Charts/PriceChart';
import { useAnalysisStore } from '../stores/analysisStore';
import { useThemeColors } from '../hooks/useThemeColors';

const { Title, Text } = Typography;

const AnalysisPage: React.FC = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const { currentAnalysis, isLoading, error, analyze, clearAnalysis, isInWatchlist, toggleWatchlist } = useAnalysisStore();
  const [isRefreshing, setIsRefreshing] = useState(false);
  const { t, i18n } = useTranslation();
  const { secondaryColor } = useThemeColors();

  useEffect(() => {
    if (symbol) {
      const upperSymbol = symbol.toUpperCase();
      if (currentAnalysis && currentAnalysis.symbol !== upperSymbol) {
        clearAnalysis();
      }
      if (!currentAnalysis || currentAnalysis.symbol !== upperSymbol) {
        analyze(symbol);
      }
    }
  }, [symbol, currentAnalysis, clearAnalysis, analyze]);

  const handleRefresh = useCallback(async () => {
    if (!symbol) return;
    setIsRefreshing(true);
    try {
      await analyze(symbol);
    } finally {
      setIsRefreshing(false);
    }
  }, [symbol, analyze]);

  const handleToggleWatchlist = async () => {
    if (symbol) {
      const wasInWatchlist = isInWatchlist(symbol);
      await toggleWatchlist(symbol);
      message.success(
        wasInWatchlist
          ? t('watchlist.removed') || 'Removed from watchlist'
          : t('watchlist.added') || 'Added to watchlist'
      );
    }
  };

  const handleExportReport = () => {
    if (!currentAnalysis) return;

    const { market_data, technical_analysis } = currentAnalysis;
    const isZh = i18n.language?.startsWith('zh');

    const report = `
================================================================================
${isZh ? 'FinMind 股票分析报告' : 'FinMind Stock Analysis Report'}
================================================================================

${isZh ? '股票代码' : 'Symbol'}: ${currentAnalysis.symbol}
${isZh ? '生成时间' : 'Generated'}: ${new Date(currentAnalysis.timestamp).toLocaleString()}

--------------------------------------------------------------------------------
${isZh ? '市场数据' : 'Market Data'}
--------------------------------------------------------------------------------
${isZh ? '当前价格' : 'Current Price'}: $${market_data.current_price.toFixed(2)}
${isZh ? '市值' : 'Market Cap'}: ${market_data.market_cap ? `$${(market_data.market_cap / 1e9).toFixed(2)}B` : 'N/A'}
${isZh ? '市盈率' : 'P/E Ratio'}: ${market_data.pe_ratio?.toFixed(2) || market_data.forward_pe?.toFixed(2) || 'N/A'}
Beta: ${market_data.beta?.toFixed(3) || 'N/A'}
${isZh ? '52周最高' : '52W High'}: $${market_data.fifty_two_week_high?.toFixed(2) || 'N/A'}
${isZh ? '52周最低' : '52W Low'}: $${market_data.fifty_two_week_low?.toFixed(2) || 'N/A'}
${isZh ? '交易量' : 'Volume'}: ${market_data.volume?.toLocaleString() || 'N/A'}

--------------------------------------------------------------------------------
${isZh ? '技术分析' : 'Technical Analysis'}
--------------------------------------------------------------------------------
${isZh ? '综合信号' : 'Overall Signal'}: ${technical_analysis.overall_signal.toUpperCase()}
${isZh ? '趋势方向' : 'Trend'}: ${technical_analysis.trend}
${isZh ? '置信度' : 'Confidence'}: ${(technical_analysis.signal_confidence * 100).toFixed(1)}%

${isZh ? '支撑位' : 'Support Levels'}: ${technical_analysis.support_levels.map(l => `$${l.toFixed(2)}`).join(', ') || 'N/A'}
${isZh ? '阻力位' : 'Resistance Levels'}: ${technical_analysis.resistance_levels.map(l => `$${l.toFixed(2)}`).join(', ') || 'N/A'}

--------------------------------------------------------------------------------
${isZh ? '多时间框架分析' : 'Multi-Timeframe Analysis'}
--------------------------------------------------------------------------------
${technical_analysis.timeframe_analyses.map(tf => `
${tf.timeframe_label}:
  ${isZh ? '信号' : 'Signal'}: ${tf.signal.toUpperCase()}
  ${isZh ? '趋势' : 'Trend'}: ${tf.trend}
  ${isZh ? '置信度' : 'Confidence'}: ${(tf.confidence * 100).toFixed(1)}%
  ${isZh ? '关键指标' : 'Key Indicators'}: ${tf.key_indicators.join('; ')}
  ${isZh ? '描述' : 'Description'}: ${tf.description}
`).join('')}

================================================================================
${isZh ? '免责声明：本报告仅供参考，不构成投资建议。' : 'Disclaimer: This report is for reference only and does not constitute investment advice.'}
================================================================================
`;

    // 下载文件
    const blob = new Blob([report], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentAnalysis.symbol}_analysis_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    message.success(t('analysis.exportSuccess') || 'Report exported successfully');
  };

  if (isLoading && !currentAnalysis) {
    return (
      <div
        style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          minHeight: 400,
        }}
      >
        <Spin size="large" />
        <Text style={{ marginTop: 24, color: secondaryColor }}>
          {t('analysis.analyzing', { symbol: symbol?.toUpperCase() })}
        </Text>
        <Text type="secondary" style={{ marginTop: 8 }}>
          {t('analysis.analyzingHint')}
        </Text>
      </div>
    );
  }

  if (error) {
    return (
      <Alert
        message={t('analysis.failed')}
        description={error}
        type="error"
        showIcon
        action={
          <Button size="small" onClick={handleRefresh}>
            {t('common.retry')}
          </Button>
        }
      />
    );
  }

  if (!currentAnalysis) {
    return (
      <Alert
        message={t('analysis.noData')}
        description={t('analysis.noDataHint')}
        type="info"
        showIcon
      />
    );
  }

  const { market_data, technical_analysis, price_history } = currentAnalysis;

  return (
    <div>
      {/* Header with refresh */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 16,
        }}
      >
        <Title level={4} style={{ margin: 0 }}>
          {t('analysis.title', { symbol: currentAnalysis.symbol })}
        </Title>
        <Space>
          <Text type="secondary">
            {t('analysis.lastUpdated')}: {new Date(currentAnalysis.timestamp).toLocaleString()}
          </Text>
          <Button
            icon={<ReloadOutlined spin={isRefreshing} />}
            onClick={handleRefresh}
            loading={isLoading}
          >
            {t('common.refresh')}
          </Button>
          <Button icon={<DownloadOutlined />} onClick={handleExportReport}>
            {t('common.export')}
          </Button>
        </Space>
      </div>

      {/* Stock Header */}
      <StockHeader
        symbol={currentAnalysis.symbol}
        marketData={market_data}
        isInWatchlist={isInWatchlist(currentAnalysis.symbol)}
        onToggleWatchlist={handleToggleWatchlist}
      />

      <Row gutter={[24, 24]}>
        {/* Price Chart */}
        <Col xs={24} xl={16}>
          <PriceChart
            symbol={currentAnalysis.symbol}
            data={price_history}
          />
        </Col>

        {/* Technical Overview */}
        <Col xs={24} xl={8}>
          <TechnicalOverview technical={technical_analysis} />
        </Col>

        {/* Multi-Timeframe Analysis */}
        <Col span={24}>
          <TimeframeAnalysis analyses={technical_analysis.timeframe_analyses} />
        </Col>

        {/* Risk Analysis - Monte Carlo Simulation */}
        <Col span={24}>
          <RiskAnalysis
            symbol={currentAnalysis.symbol}
            currentPrice={market_data.current_price}
          />
        </Col>

        {/* DCF Sensitivity & Backtest */}
        <Col span={24}>
          <Tabs
            items={[
              {
                key: 'sensitivity',
                label: t('analysis.sensitivity') || (i18n.language?.startsWith('zh') ? 'DCF 敏感度' : 'DCF Sensitivity'),
                children: (
                  <SensitivityAnalysis
                    symbol={currentAnalysis.symbol}
                    currentPrice={market_data.current_price}
                  />
                ),
              },
              {
                key: 'backtest',
                label: t('analysis.backtest') || (i18n.language?.startsWith('zh') ? '量化回测' : 'Backtest'),
                children: (
                  <BacktestPanel symbol={currentAnalysis.symbol} />
                ),
              },
            ]}
          />
        </Col>
      </Row>
    </div>
  );
};

export default AnalysisPage;
