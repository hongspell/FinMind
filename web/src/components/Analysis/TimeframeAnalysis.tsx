import React from 'react';
import { Card, Table, Tag, Progress, Typography, Space, Tooltip } from 'antd';
import {
  RiseOutlined,
  FallOutlined,
  MinusOutlined,
  ClockCircleOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import type { TimeframeAnalysis as TimeframeAnalysisType, SignalStrength, TrendDirection } from '../../types/analysis';
import { colors, signalColors, trendColors } from '../../styles/theme';
import { useThemeColors } from '../../hooks/useThemeColors';

const { Title, Text } = Typography;

interface TimeframeAnalysisProps {
  analyses: TimeframeAnalysisType[];
}

const trendIcons: Record<TrendDirection, React.ReactNode> = {
  strong_bullish: <RiseOutlined style={{ color: colors.up }} />,
  bullish: <RiseOutlined style={{ color: colors.buy }} />,
  neutral: <MinusOutlined style={{ color: colors.neutral }} />,
  bearish: <FallOutlined style={{ color: colors.sell }} />,
  strong_bearish: <FallOutlined style={{ color: colors.down }} />,
};

const TimeframeAnalysis: React.FC<TimeframeAnalysisProps> = ({ analyses }) => {
  const { t } = useTranslation();
  const { textColor, secondaryColor, borderColor } = useThemeColors();

  const columns = [
    {
      title: (
        <Space>
          <ClockCircleOutlined />
          <span>{t('analysis.timeframe')}</span>
        </Space>
      ),
      dataIndex: 'timeframe',
      key: 'timeframe',
      width: 150,
      render: (timeframe: string) => (
        <Text strong style={{ color: textColor }}>
          {t(`timeframes.${timeframe}`)} {t(`timeframes.${timeframe}_period`)}
        </Text>
      ),
    },
    {
      title: t('analysis.signal'),
      dataIndex: 'signal',
      key: 'signal',
      width: 140,
      render: (signal: SignalStrength) => (
        <Tag
          style={{
            color: signalColors[signal],
            background: `${signalColors[signal]}15`,
            border: `1px solid ${signalColors[signal]}40`,
            fontWeight: 600,
          }}
        >
          {t(`signals.${signal}`)}
        </Tag>
      ),
    },
    {
      title: t('analysis.trend'),
      dataIndex: 'trend',
      key: 'trend',
      width: 160,
      render: (trend: TrendDirection) => (
        <Space>
          {trendIcons[trend]}
          <Text style={{ color: trendColors[trend] }}>
            {t(`trends.${trend}`)}
          </Text>
        </Space>
      ),
    },
    {
      title: t('analysis.confidence'),
      dataIndex: 'confidence',
      key: 'confidence',
      width: 180,
      render: (confidence: number) => {
        const percent = Math.round(confidence * 100);
        const color = percent >= 70 ? colors.up : percent >= 50 ? colors.hold : colors.down;
        return (
          <Space direction="vertical" size={2} style={{ width: '100%' }}>
            <Progress
              percent={percent}
              size="small"
              strokeColor={color}
              trailColor={borderColor}
              showInfo={false}
            />
            <Text style={{ color, fontSize: 12 }}>{percent}%</Text>
          </Space>
        );
      },
    },
    {
      title: t('analysis.keyIndicators'),
      dataIndex: 'key_indicators',
      key: 'indicators',
      render: (indicators: string[]) => (
        <Space direction="vertical" size={2}>
          {indicators.slice(0, 2).map((ind, idx) => (
            <Tooltip title={ind} key={idx}>
              <Text
                type="secondary"
                ellipsis
                style={{ maxWidth: 200, fontSize: 12 }}
              >
                • {ind}
              </Text>
            </Tooltip>
          ))}
        </Space>
      ),
    },
  ];

  return (
    <Card
      title={
        <Space>
          <ClockCircleOutlined />
          <span>{t('analysis.multiTimeframe')}</span>
        </Space>
      }
      styles={{ body: { padding: 0 } }}
    >
      <Table
        dataSource={analyses}
        columns={columns}
        rowKey="timeframe"
        pagination={false}
        size="middle"
        style={{ background: 'transparent' }}
      />

      {/* 分析说明 */}
      <div style={{ padding: '16px 24px', borderTop: `1px solid ${borderColor}` }}>
        <Title level={5} style={{ marginBottom: 12, color: secondaryColor }}>
          {t('analysis.description')}
        </Title>
        <Space direction="vertical" size={4}>
          {analyses.map((tf) => (
            <Text key={tf.timeframe} type="secondary" style={{ fontSize: 13 }}>
              • {t(`timeframeDescriptions.${tf.timeframe}`, {
                signal: t(`signals.${tf.signal}`),
                trend: t(`trends.${tf.trend}`),
                confidence: Math.round(tf.confidence * 100)
              })}
            </Text>
          ))}
        </Space>
      </div>
    </Card>
  );
};

export default TimeframeAnalysis;
