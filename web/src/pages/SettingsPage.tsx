import React, { useEffect, useState } from 'react';
import { Card, Form, Select, Switch, Typography, Space, Divider, Button, message, Popconfirm, Row, Col, Tag, Input, Collapse, Spin, Badge, InputNumber, Modal, Alert } from 'antd';
import {
  SettingOutlined,
  GlobalOutlined,
  BgColorsOutlined,
  BellOutlined,
  DatabaseOutlined,
  ClockCircleOutlined,
  StarOutlined,
  ExclamationCircleOutlined,
  ApiOutlined,
  KeyOutlined,
  CloudServerOutlined,
  RobotOutlined,
  SaveOutlined,
  ReloadOutlined,
  CheckCircleOutlined,
  CloseCircleOutlined,
  BankOutlined,
  LinkOutlined,
  DisconnectOutlined,
  PlayCircleOutlined,
} from '@ant-design/icons';
import { brokerApi } from '../services/brokerApi';
import type { BrokerType, BrokerStatus, BrokerConfig } from '../types/broker';
import { useTranslation } from 'react-i18next';
import { useSettingsStore } from '../stores/settingsStore';
import { useAnalysisStore } from '../stores/analysisStore';
import { configApi, ConfigItem } from '../services/api';

const { Title, Text } = Typography;

// 分类配置
const CATEGORY_INFO: Record<string, { icon: React.ReactNode; label: string; labelZh: string }> = {
  llm: { icon: <RobotOutlined />, label: 'LLM API Keys', labelZh: 'AI 模型 API' },
  data: { icon: <CloudServerOutlined />, label: 'Data Sources', labelZh: '数据源 API' },
  database: { icon: <DatabaseOutlined />, label: 'Database', labelZh: '数据库配置' },
  app: { icon: <SettingOutlined />, label: 'Application', labelZh: '应用设置' },
  local_llm: { icon: <RobotOutlined />, label: 'Local LLM', labelZh: '本地 AI 模型' },
  china: { icon: <GlobalOutlined />, label: 'China Market', labelZh: 'A股数据' },
};

const SettingsPage: React.FC = () => {
  const { t, i18n } = useTranslation();
  const {
    theme,
    chartStyle,
    priceAlerts,
    signalAlerts,
    setTheme,
    setChartStyle,
    setPriceAlerts,
    setSignalAlerts,
  } = useSettingsStore();

  const { recentSymbols, watchlist } = useAnalysisStore();

  // API 配置状态
  const [configs, setConfigs] = useState<ConfigItem[]>([]);
  const [configLoading, setConfigLoading] = useState(false);
  const [configUpdates, setConfigUpdates] = useState<Record<string, string>>({});
  const [saving, setSaving] = useState(false);

  // 券商配置状态
  const [brokerStatuses, setBrokerStatuses] = useState<BrokerStatus[]>([]);
  const [brokerLoading, setBrokerLoading] = useState(false);
  const [connectingBroker, setConnectingBroker] = useState<BrokerType | null>(null);
  const [brokerModalVisible, setBrokerModalVisible] = useState(false);
  const [selectedBrokerType, setSelectedBrokerType] = useState<BrokerType | null>(null);
  const [brokerForm] = Form.useForm();

  const isZh = i18n.language?.startsWith('zh');

  // 加载配置
  const loadConfigs = async () => {
    setConfigLoading(true);
    try {
      const response = await configApi.getConfig();
      if (response.success && response.data) {
        setConfigs(response.data.configs);
      }
    } catch (error) {
      console.error('Failed to load configs:', error);
    } finally {
      setConfigLoading(false);
    }
  };

  useEffect(() => {
    loadConfigs();
    loadBrokerStatus();
  }, []);

  // 加载券商状态
  const loadBrokerStatus = async () => {
    setBrokerLoading(true);
    try {
      const response = await brokerApi.getStatus() as any;
      // API 直接返回 {brokers: [...]}
      if (response.brokers) {
        setBrokerStatuses(response.brokers);
      }
    } catch (error) {
      console.error('Failed to load broker status:', error);
    } finally {
      setBrokerLoading(false);
    }
  };

  // 连接券商
  const handleConnectBroker = async (values: any) => {
    if (!selectedBrokerType) return;

    setConnectingBroker(selectedBrokerType);
    try {
      const config: BrokerConfig = {
        broker_type: selectedBrokerType,
        ...values,
      };
      const response = await brokerApi.connect(config) as any;
      // API 直接返回 {connected, account_id, error}，不是包装在 success/data 中
      if (response.connected) {
        message.success(isZh ? '券商连接成功' : 'Broker connected successfully');
        setBrokerModalVisible(false);
        brokerForm.resetFields();
        loadBrokerStatus();
      } else {
        message.error(response.error || (isZh ? '连接失败' : 'Connection failed'));
      }
    } catch (error: any) {
      message.error(error.message || (isZh ? '连接失败' : 'Connection failed'));
    } finally {
      setConnectingBroker(null);
    }
  };

  // 断开券商连接
  const handleDisconnectBroker = async (brokerType: BrokerType) => {
    try {
      await brokerApi.disconnect(brokerType);
      message.success(isZh ? '已断开连接' : 'Disconnected');
      loadBrokerStatus();
    } catch (error: any) {
      message.error(error.message);
    }
  };

  // 设置演示模式
  const handleSetupDemo = async () => {
    setBrokerLoading(true);
    try {
      await brokerApi.setupDemo();
      message.success(isZh ? '演示环境已启用' : 'Demo environment enabled');
      loadBrokerStatus();
    } catch (error: any) {
      message.error(error.message);
    } finally {
      setBrokerLoading(false);
    }
  };

  // 打开连接对话框
  const openConnectModal = (brokerType: BrokerType) => {
    setSelectedBrokerType(brokerType);
    setBrokerModalVisible(true);
    brokerForm.resetFields();
    // 设置默认值
    if (brokerType === 'ibkr') {
      brokerForm.setFieldsValue({ ibkr_host: '127.0.0.1', ibkr_port: 4001, ibkr_client_id: 1 });
    } else if (brokerType === 'futu') {
      brokerForm.setFieldsValue({ futu_host: '127.0.0.1', futu_port: 11111 });
    }
  };

  // 获取券商信息
  const getBrokerInfo = (type: BrokerType) => {
    const info: Record<BrokerType, { name: string; nameZh: string; description: string; descriptionZh: string }> = {
      ibkr: {
        name: 'Interactive Brokers',
        nameZh: '盈透证券',
        description: 'Connect via TWS or IB Gateway',
        descriptionZh: '通过 TWS 或 IB Gateway 连接',
      },
      futu: {
        name: 'Futu Securities',
        nameZh: '富途证券',
        description: 'Connect via OpenD',
        descriptionZh: '通过 OpenD 连接',
      },
      tiger: {
        name: 'Tiger Brokers',
        nameZh: '老虎证券',
        description: 'Connect via Tiger Open API',
        descriptionZh: '通过 Tiger Open API 连接',
      },
    };
    return info[type];
  };

  // 保存配置
  const handleSaveConfigs = async () => {
    const updates = Object.entries(configUpdates)
      .filter(([_, value]) => value && !value.startsWith('*'))
      .map(([key, value]) => ({ key, value }));

    if (updates.length === 0) {
      message.warning(isZh ? '没有需要保存的更改' : 'No changes to save');
      return;
    }

    setSaving(true);
    try {
      const response = await configApi.updateConfig(updates);
      if (response.success) {
        message.success(isZh ? '配置已保存，部分设置可能需要重启服务生效' : 'Configuration saved. Some settings may require service restart.');
        setConfigUpdates({});
        loadConfigs();
      } else {
        message.error(response.error || 'Failed to save');
      }
    } catch (error) {
      message.error(isZh ? '保存失败' : 'Failed to save');
    } finally {
      setSaving(false);
    }
  };

  const handleLanguageChange = (value: string) => {
    i18n.changeLanguage(value);
    message.success(t('settings.saved') || 'Settings saved');
  };

  const handleThemeChange = (value: 'dark' | 'light') => {
    setTheme(value);
    if (value === 'light') {
      message.info(t('settings.lightThemeHint') || 'Light theme applied (some components may need refresh)');
    } else {
      message.success(t('settings.saved') || 'Settings saved');
    }
  };

  const handleClearRecent = () => {
    localStorage.removeItem('finmind_recent');
    useAnalysisStore.setState({ recentSymbols: [] });
    message.success(t('settings.cleared') || 'Data cleared');
  };

  const handleClearWatchlist = () => {
    localStorage.removeItem('finmind_watchlist');
    useAnalysisStore.setState({ watchlist: [] });
    message.success(t('settings.cleared') || 'Data cleared');
  };

  // 主题相关样式
  const isDark = theme === 'dark';
  const cardBg = isDark ? '#1c2128' : '#fafafa';
  const borderColor = isDark ? '#30363d' : '#e8e8e8';
  const secondaryColor = isDark ? '#8b949e' : '#595959';

  // 按分类分组配置
  const configsByCategory = configs.reduce((acc, config) => {
    if (!acc[config.category]) {
      acc[config.category] = [];
    }
    acc[config.category].push(config);
    return acc;
  }, {} as Record<string, ConfigItem[]>);

  // 渲染配置项
  const renderConfigItem = (config: ConfigItem) => {
    const currentValue = configUpdates[config.key] ?? config.value;
    const hasUpdate = configUpdates[config.key] !== undefined;

    return (
      <div
        key={config.key}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '12px 0',
          borderBottom: `1px solid ${borderColor}`,
        }}
      >
        <div style={{ flex: 1, marginRight: 16 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Text strong style={{ fontFamily: 'monospace' }}>{config.key}</Text>
            {config.hasValue ? (
              <CheckCircleOutlined style={{ color: '#52c41a', fontSize: 12 }} />
            ) : (
              <CloseCircleOutlined style={{ color: '#8b949e', fontSize: 12 }} />
            )}
          </div>
          <Text type="secondary" style={{ fontSize: 12 }}>{config.description}</Text>
        </div>
        <div style={{ width: 300 }}>
          <Input.Password
            value={currentValue}
            onChange={(e) => setConfigUpdates({ ...configUpdates, [config.key]: e.target.value })}
            placeholder={config.is_secret ? '••••••••' : (isZh ? '未设置' : 'Not set')}
            visibilityToggle={config.is_secret}
            style={{
              fontFamily: 'monospace',
              borderColor: hasUpdate ? '#1890ff' : undefined,
            }}
          />
        </div>
      </div>
    );
  };

  // 生成 Collapse items
  const collapseItems = Object.entries(configsByCategory).map(([category, items]) => {
    const info = CATEGORY_INFO[category] || { icon: <KeyOutlined />, label: category, labelZh: category };
    const configuredCount = items.filter(i => i.hasValue).length;

    return {
      key: category,
      label: (
        <Space>
          {info.icon}
          <span>{isZh ? info.labelZh : info.label}</span>
          <Badge
            count={`${configuredCount}/${items.length}`}
            style={{
              backgroundColor: configuredCount === items.length ? '#52c41a' : '#8b949e',
            }}
          />
        </Space>
      ),
      children: items.map(renderConfigItem),
    };
  });

  return (
    <div>
      <Title level={3} style={{ marginBottom: 24 }}>
        <SettingOutlined style={{ marginRight: 12 }} />
        {t('nav.settings')}
      </Title>

      <Space direction="vertical" size={24} style={{ width: '100%', maxWidth: 900 }}>
        {/* Language Settings */}
        <Card
          title={
            <Space>
              <GlobalOutlined />
              <span>{t('settings.language') || 'Language'}</span>
            </Space>
          }
        >
          <Form layout="vertical">
            <Form.Item
              label={t('settings.displayLanguage') || 'Display Language'}
              style={{ marginBottom: 0 }}
            >
              <Select
                value={i18n.language?.startsWith('zh') ? 'zh' : 'en'}
                onChange={handleLanguageChange}
                style={{ width: 200 }}
                options={[
                  { value: 'en', label: 'English' },
                  { value: 'zh', label: '中文' },
                ]}
              />
            </Form.Item>
          </Form>
        </Card>

        {/* Display Settings */}
        <Card
          title={
            <Space>
              <BgColorsOutlined />
              <span>{t('settings.display') || 'Display'}</span>
            </Space>
          }
        >
          <Form layout="vertical">
            <Form.Item
              label={t('settings.theme') || 'Theme'}
              style={{ marginBottom: 16 }}
            >
              <Select
                value={theme}
                onChange={handleThemeChange}
                style={{ width: 200 }}
                options={[
                  { value: 'dark', label: `${t('settings.darkTheme') || 'Dark'}` },
                  { value: 'light', label: `${t('settings.lightTheme') || 'Light'}` },
                ]}
              />
              {theme === 'light' && (
                <Text type="warning" style={{ display: 'block', marginTop: 8, fontSize: 12 }}>
                  <ExclamationCircleOutlined style={{ marginRight: 4 }} />
                  {t('settings.lightThemeBeta') || 'Light theme is in beta, some areas may not display correctly'}
                </Text>
              )}
            </Form.Item>

            <Divider style={{ borderColor: borderColor, margin: '16px 0' }} />

            <Form.Item
              label={t('settings.chartStyle') || 'Chart Style'}
              style={{ marginBottom: 0 }}
            >
              <Select
                value={chartStyle}
                onChange={(value) => {
                  setChartStyle(value);
                  message.success(t('settings.saved') || 'Settings saved');
                }}
                style={{ width: 200 }}
                options={[
                  { value: 'candle', label: `${t('settings.candlestick') || 'Candlestick'}` },
                  { value: 'line', label: `${t('settings.line') || 'Line'}` },
                  { value: 'area', label: `${t('settings.area') || 'Area'}` },
                ]}
              />
            </Form.Item>
          </Form>
        </Card>

        {/* Broker Configuration */}
        <Card
          title={
            <Space>
              <BankOutlined />
              <span>{isZh ? '券商连接' : 'Broker Connections'}</span>
            </Space>
          }
          extra={
            <Button
              icon={<PlayCircleOutlined />}
              onClick={handleSetupDemo}
              loading={brokerLoading}
            >
              {isZh ? '启用演示模式' : 'Enable Demo Mode'}
            </Button>
          }
        >
          <Alert
            message={isZh ? '连接您的券商账户以获取实时持仓和个性化分析建议' : 'Connect your broker accounts for real-time positions and personalized analysis'}
            type="info"
            showIcon
            style={{ marginBottom: 16 }}
          />

          <Row gutter={[16, 16]}>
            {(['ibkr', 'futu', 'tiger'] as BrokerType[]).map((brokerType) => {
              const info = getBrokerInfo(brokerType);
              const status = brokerStatuses.find((s) => s.broker_type === brokerType);
              const isConnected = status?.connected || false;

              return (
                <Col xs={24} md={8} key={brokerType}>
                  <div
                    style={{
                      padding: 16,
                      background: cardBg,
                      borderRadius: 8,
                      border: `1px solid ${isConnected ? '#52c41a' : borderColor}`,
                      height: '100%',
                    }}
                  >
                    <Space direction="vertical" size={12} style={{ width: '100%' }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Text strong style={{ fontSize: 16 }}>
                          {isZh ? info.nameZh : info.name}
                        </Text>
                        {isConnected ? (
                          <Tag color="success" icon={<CheckCircleOutlined />}>
                            {isZh ? '已连接' : 'Connected'}
                          </Tag>
                        ) : (
                          <Tag color="default">{isZh ? '未连接' : 'Not Connected'}</Tag>
                        )}
                      </div>

                      <Text type="secondary" style={{ fontSize: 12 }}>
                        {isZh ? info.descriptionZh : info.description}
                      </Text>

                      {isConnected && status?.account_id && (
                        <Text type="secondary" style={{ fontSize: 12 }}>
                          {isZh ? '账户' : 'Account'}: {status.account_id}
                        </Text>
                      )}

                      <div style={{ marginTop: 8 }}>
                        {isConnected ? (
                          <Popconfirm
                            title={isZh ? '确定断开连接？' : 'Disconnect this broker?'}
                            onConfirm={() => handleDisconnectBroker(brokerType)}
                            okText={isZh ? '确定' : 'Yes'}
                            cancelText={isZh ? '取消' : 'No'}
                          >
                            <Button
                              danger
                              icon={<DisconnectOutlined />}
                              size="small"
                              block
                            >
                              {isZh ? '断开连接' : 'Disconnect'}
                            </Button>
                          </Popconfirm>
                        ) : (
                          <Button
                            type="primary"
                            icon={<LinkOutlined />}
                            size="small"
                            block
                            onClick={() => openConnectModal(brokerType)}
                          >
                            {isZh ? '连接' : 'Connect'}
                          </Button>
                        )}
                      </div>
                    </Space>
                  </div>
                </Col>
              );
            })}
          </Row>
        </Card>

        {/* API Configuration */}
        <Card
          title={
            <Space>
              <ApiOutlined />
              <span>{isZh ? 'API 配置' : 'API Configuration'}</span>
            </Space>
          }
          extra={
            <Space>
              <Button
                icon={<ReloadOutlined />}
                onClick={loadConfigs}
                loading={configLoading}
              >
                {isZh ? '刷新' : 'Refresh'}
              </Button>
              <Button
                type="primary"
                icon={<SaveOutlined />}
                onClick={handleSaveConfigs}
                loading={saving}
                disabled={Object.keys(configUpdates).length === 0}
              >
                {isZh ? '保存更改' : 'Save Changes'}
              </Button>
            </Space>
          }
        >
          {configLoading ? (
            <div style={{ textAlign: 'center', padding: 40 }}>
              <Spin />
            </div>
          ) : (
            <>
              <Text type="secondary" style={{ display: 'block', marginBottom: 16 }}>
                {isZh
                  ? '在此配置各类 API 密钥。敏感信息已加密显示，输入新值后点击保存生效。'
                  : 'Configure API keys here. Sensitive values are masked. Enter new values and click Save to apply.'}
              </Text>
              <Collapse
                items={collapseItems}
                defaultActiveKey={['llm', 'data']}
                style={{ background: 'transparent' }}
              />
            </>
          )}
        </Card>

        {/* Notification Settings */}
        <Card
          title={
            <Space>
              <BellOutlined />
              <span>{t('settings.notifications') || 'Notifications'}</span>
            </Space>
          }
        >
          <Space direction="vertical" size={16} style={{ width: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <Text strong>
                  {t('settings.priceAlerts') || 'Price Alerts'}
                </Text>
                <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>
                  {t('settings.priceAlertsHint') || 'Get notified when price targets are hit'}
                </Text>
              </div>
              <Switch
                checked={priceAlerts}
                onChange={(checked) => {
                  setPriceAlerts(checked);
                  message.success(t('settings.saved') || 'Settings saved');
                }}
              />
            </div>

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div>
                <Text strong>
                  {t('settings.signalAlerts') || 'Signal Alerts'}
                </Text>
                <Text type="secondary" style={{ display: 'block', fontSize: 12 }}>
                  {t('settings.signalAlertsHint') || 'Get notified when signals change'}
                </Text>
              </div>
              <Switch
                checked={signalAlerts}
                onChange={(checked) => {
                  setSignalAlerts(checked);
                  message.success(t('settings.saved') || 'Settings saved');
                }}
              />
            </div>

            <Text type="secondary" style={{ fontSize: 12 }}>
              {t('settings.notificationsHint') || 'Note: Browser notifications require permission'}
            </Text>
          </Space>
        </Card>

        {/* Data Management */}
        <Card
          title={
            <Space>
              <DatabaseOutlined />
              <span>{t('settings.dataManagement') || 'Data Management'}</span>
            </Space>
          }
        >
          <Row gutter={[16, 16]}>
            {/* Recent Searches Card */}
            <Col xs={24} sm={12}>
              <div
                style={{
                  padding: 16,
                  background: cardBg,
                  borderRadius: 8,
                  border: `1px solid ${borderColor}`,
                }}
              >
                <Space direction="vertical" size={12} style={{ width: '100%' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <ClockCircleOutlined style={{ color: secondaryColor, fontSize: 18 }} />
                    <Text strong>
                      {t('settings.recentSearches') || 'Recent Searches'}
                    </Text>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Tag color={recentSymbols.length > 0 ? 'blue' : 'default'}>
                      {recentSymbols.length} {t('settings.items') || 'items'}
                    </Tag>
                    <Popconfirm
                      title={t('settings.confirmClear') || 'Clear all recent searches?'}
                      onConfirm={handleClearRecent}
                      okText={t('common.confirm') || 'Yes'}
                      cancelText={t('common.cancel') || 'No'}
                      disabled={recentSymbols.length === 0}
                    >
                      <Button
                        type="text"
                        size="small"
                        danger={recentSymbols.length > 0}
                        disabled={recentSymbols.length === 0}
                      >
                        {t('settings.clear') || 'Clear'}
                      </Button>
                    </Popconfirm>
                  </div>
                </Space>
              </div>
            </Col>

            {/* Watchlist Card */}
            <Col xs={24} sm={12}>
              <div
                style={{
                  padding: 16,
                  background: cardBg,
                  borderRadius: 8,
                  border: `1px solid ${borderColor}`,
                }}
              >
                <Space direction="vertical" size={12} style={{ width: '100%' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <StarOutlined style={{ color: secondaryColor, fontSize: 18 }} />
                    <Text strong>
                      {t('nav.watchlist') || 'Watchlist'}
                    </Text>
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <Tag color={watchlist.length > 0 ? 'gold' : 'default'}>
                      {watchlist.length} {t('settings.stocks') || 'stocks'}
                    </Tag>
                    <Popconfirm
                      title={t('settings.confirmClearWatchlist') || 'Clear all watchlist?'}
                      onConfirm={handleClearWatchlist}
                      okText={t('common.confirm') || 'Yes'}
                      cancelText={t('common.cancel') || 'No'}
                      disabled={watchlist.length === 0}
                    >
                      <Button
                        type="text"
                        size="small"
                        danger={watchlist.length > 0}
                        disabled={watchlist.length === 0}
                      >
                        {t('settings.clear') || 'Clear'}
                      </Button>
                    </Popconfirm>
                  </div>
                </Space>
              </div>
            </Col>
          </Row>
        </Card>
      </Space>

      {/* Broker Connection Modal */}
      <Modal
        title={
          <Space>
            <BankOutlined />
            <span>
              {isZh ? '连接' : 'Connect'}{' '}
              {selectedBrokerType && (isZh ? getBrokerInfo(selectedBrokerType).nameZh : getBrokerInfo(selectedBrokerType).name)}
            </span>
          </Space>
        }
        open={brokerModalVisible}
        onCancel={() => {
          setBrokerModalVisible(false);
          brokerForm.resetFields();
        }}
        footer={null}
        destroyOnClose
      >
        <Form
          form={brokerForm}
          layout="vertical"
          onFinish={handleConnectBroker}
        >
          {selectedBrokerType === 'ibkr' && (
            <>
              <Alert
                message={isZh ? '请确保 TWS 或 IB Gateway 已启动并启用 API 连接' : 'Make sure TWS or IB Gateway is running with API connections enabled'}
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
              <Form.Item
                name="ibkr_host"
                label={isZh ? '主机地址' : 'Host'}
                rules={[{ required: true }]}
              >
                <Input placeholder="127.0.0.1" />
              </Form.Item>
              <Form.Item
                name="ibkr_port"
                label={isZh ? '端口' : 'Port'}
                rules={[{ required: true }]}
                extra={isZh ? 'TWS: 7496/7497, IB Gateway: 4001/4002' : 'TWS: 7496/7497, IB Gateway: 4001/4002'}
              >
                <InputNumber style={{ width: '100%' }} min={1} max={65535} />
              </Form.Item>
              <Form.Item
                name="ibkr_client_id"
                label={isZh ? '客户端 ID' : 'Client ID'}
                rules={[{ required: true }]}
              >
                <InputNumber style={{ width: '100%' }} min={0} max={999} />
              </Form.Item>
            </>
          )}

          {selectedBrokerType === 'futu' && (
            <>
              <Alert
                message={isZh ? '请确保 OpenD 已启动并已登录' : 'Make sure OpenD is running and logged in'}
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
              <Form.Item
                name="futu_host"
                label={isZh ? '主机地址' : 'Host'}
                rules={[{ required: true }]}
              >
                <Input placeholder="127.0.0.1" />
              </Form.Item>
              <Form.Item
                name="futu_port"
                label={isZh ? '端口' : 'Port'}
                rules={[{ required: true }]}
              >
                <InputNumber style={{ width: '100%' }} min={1} max={65535} />
              </Form.Item>
            </>
          )}

          {selectedBrokerType === 'tiger' && (
            <>
              <Alert
                message={isZh ? '请在老虎开放平台获取 API 密钥' : 'Get API credentials from Tiger Open Platform'}
                type="info"
                showIcon
                style={{ marginBottom: 16 }}
              />
              <Form.Item
                name="tiger_id"
                label="Tiger ID"
                rules={[{ required: true }]}
              >
                <Input />
              </Form.Item>
              <Form.Item
                name="tiger_account"
                label={isZh ? '账户' : 'Account'}
                rules={[{ required: true }]}
              >
                <Input />
              </Form.Item>
              <Form.Item
                name="tiger_private_key"
                label={isZh ? '私钥' : 'Private Key'}
                rules={[{ required: true }]}
              >
                <Input.TextArea rows={4} placeholder="-----BEGIN RSA PRIVATE KEY-----" />
              </Form.Item>
            </>
          )}

          <Form.Item style={{ marginBottom: 0, marginTop: 24 }}>
            <Space style={{ width: '100%', justifyContent: 'flex-end' }}>
              <Button onClick={() => setBrokerModalVisible(false)}>
                {isZh ? '取消' : 'Cancel'}
              </Button>
              <Button
                type="primary"
                htmlType="submit"
                loading={connectingBroker === selectedBrokerType}
                icon={<LinkOutlined />}
              >
                {isZh ? '连接' : 'Connect'}
              </Button>
            </Space>
          </Form.Item>
        </Form>
      </Modal>
    </div>
  );
};

export default SettingsPage;
