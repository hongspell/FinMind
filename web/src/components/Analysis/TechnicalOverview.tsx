import React from 'react';
import { Card, Row, Col, Progress, Typography, Space, Tag, Divider } from 'antd';
import {
  ThunderboltOutlined,
  RiseOutlined,
  SafetyCertificateOutlined,
} from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import type { TechnicalAnalysis } from '../../types/analysis';
import { colors, signalColors, trendColors } from '../../styles/theme';
import { useSettingsStore } from '../../stores/settingsStore';

const { Title, Text } = Typography;

interface TechnicalOverviewProps {
  technical: TechnicalAnalysis;
}

const TechnicalOverview: React.FC<TechnicalOverviewProps> = ({ technical }) => {
  const { overall_signal, trend, signal_confidence, support_levels, resistance_levels } = technical;
  const { t } = useTranslation();
  const { theme } = useSettingsStore();

  const isDark = theme === 'dark';
  const cardBg = isDark ? '#1c2128' : '#fafafa';
  const borderColor = isDark ? '#30363d' : '#e8e8e8';

  const confidencePercent = Math.round(signal_confidence * 100);
  const confidenceColor = confidencePercent >= 70 ? colors.up : confidencePercent >= 50 ? colors.hold : colors.down;
  const confidenceKey = confidencePercent >= 70 ? 'high' : confidencePercent >= 50 ? 'medium' : 'low';

  return (
    <Card>
      <Row gutter={[24, 24]}>
        {/* Overall Signal */}
        <Col xs={24} md={8}>
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 24,
              background: `${signalColors[overall_signal]}10`,
              borderRadius: 8,
              border: `1px solid ${signalColors[overall_signal]}30`,
              minHeight: 180,
            }}
          >
            <ThunderboltOutlined
              style={{
                fontSize: 32,
                color: signalColors[overall_signal],
                marginBottom: 12,
              }}
            />
            <Title level={4} style={{ margin: 0, textAlign: 'center' }}>
              {t('analysis.overallSignal')}
            </Title>
            <div style={{ marginTop: 12, display: 'flex', justifyContent: 'center' }}>
              <Tag
                style={{
                  padding: '4px 16px',
                  fontSize: 14,
                  fontWeight: 700,
                  color: signalColors[overall_signal],
                  background: `${signalColors[overall_signal]}20`,
                  border: `2px solid ${signalColors[overall_signal]}`,
                  margin: 0,
                }}
              >
                {t(`signals.${overall_signal}`)}
              </Tag>
            </div>
          </div>
        </Col>

        {/* Trend Direction */}
        <Col xs={24} md={8}>
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 24,
              background: `${trendColors[trend]}10`,
              borderRadius: 8,
              border: `1px solid ${trendColors[trend]}30`,
              minHeight: 180,
            }}
          >
            <RiseOutlined
              style={{
                fontSize: 32,
                color: trendColors[trend],
                marginBottom: 12,
              }}
            />
            <Title level={4} style={{ margin: 0, textAlign: 'center' }}>
              {t('analysis.trendDirection')}
            </Title>
            <Text
              style={{
                display: 'block',
                marginTop: 12,
                fontSize: 18,
                fontWeight: 600,
                color: trendColors[trend],
                textAlign: 'center',
              }}
            >
              {t(`trends.${trend}`)}
            </Text>
          </div>
        </Col>

        {/* Confidence */}
        <Col xs={24} md={8}>
          <div
            style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: 24,
              background: cardBg,
              borderRadius: 8,
              border: `1px solid ${borderColor}`,
              minHeight: 180,
            }}
          >
            <SafetyCertificateOutlined
              style={{
                fontSize: 32,
                color: confidenceColor,
                marginBottom: 12,
              }}
            />
            <Title level={4} style={{ margin: 0, textAlign: 'center' }}>
              {t('analysis.signalConfidence')}
            </Title>
            <div style={{ marginTop: 12, display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
              <Progress
                type="circle"
                percent={confidencePercent}
                size={80}
                strokeColor={confidenceColor}
                trailColor={borderColor}
                format={(percent) => (
                  <span style={{ color: confidenceColor, fontWeight: 700 }}>
                    {percent}%
                  </span>
                )}
              />
              <Text
                style={{
                  display: 'block',
                  marginTop: 8,
                  color: confidenceColor,
                  fontWeight: 500,
                  textAlign: 'center',
                }}
              >
                {t(`confidence.${confidenceKey}`)} {t('analysis.confidence')}
              </Text>
            </div>
          </div>
        </Col>
      </Row>

      {/* Support & Resistance */}
      {(support_levels?.length > 0 || resistance_levels?.length > 0) && (
        <>
          <Divider style={{ borderColor: borderColor }} />
          <Row gutter={24}>
            {support_levels?.length > 0 && (
              <Col xs={24} md={12}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text type="secondary">{t('analysis.supportLevels')}</Text>
                  <Space wrap>
                    {support_levels.slice(0, 3).map((level, idx) => (
                      <Tag
                        key={idx}
                        color="green"
                        style={{ fontSize: 14, padding: '4px 12px' }}
                      >
                        ${level.toFixed(2)}
                      </Tag>
                    ))}
                  </Space>
                </Space>
              </Col>
            )}
            {resistance_levels?.length > 0 && (
              <Col xs={24} md={12}>
                <Space direction="vertical" style={{ width: '100%' }}>
                  <Text type="secondary">{t('analysis.resistanceLevels')}</Text>
                  <Space wrap>
                    {resistance_levels.slice(0, 3).map((level, idx) => (
                      <Tag
                        key={idx}
                        color="red"
                        style={{ fontSize: 14, padding: '4px 12px' }}
                      >
                        ${level.toFixed(2)}
                      </Tag>
                    ))}
                  </Space>
                </Space>
              </Col>
            )}
          </Row>
        </>
      )}
    </Card>
  );
};

export default TechnicalOverview;
