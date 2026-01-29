import React, { useState } from 'react';
import { Layout, Menu, Button, Space, Typography, Dropdown, Avatar } from 'antd';
import {
  LineChartOutlined,
  DashboardOutlined,
  StarOutlined,
  SettingOutlined,
  BellOutlined,
  MenuFoldOutlined,
  MenuUnfoldOutlined,
  GithubOutlined,
  PieChartOutlined,
} from '@ant-design/icons';
import { useNavigate, useLocation } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { LanguageSwitcher } from '../Common';
import { useSettingsStore } from '../../stores/settingsStore';

const { Header, Sider, Content } = Layout;
const { Text } = Typography;

interface AppLayoutProps {
  children: React.ReactNode;
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const [collapsed, setCollapsed] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { t } = useTranslation();
  const { theme } = useSettingsStore();

  // 主题相关颜色
  const isDark = theme === 'dark';
  const borderColor = isDark ? '#30363d' : '#e8e8e8';
  const textColor = isDark ? '#e6edf3' : '#1f1f1f';
  const secondaryColor = isDark ? '#8b949e' : '#595959';
  const siderBg = isDark ? '#161b22' : '#ffffff';

  const menuItems = [
    {
      key: '/',
      icon: <DashboardOutlined />,
      label: t('nav.dashboard'),
    },
    {
      key: '/analysis',
      icon: <LineChartOutlined />,
      label: t('nav.analysis'),
    },
    {
      key: '/portfolio',
      icon: <PieChartOutlined />,
      label: t('nav.portfolio'),
    },
    {
      key: '/watchlist',
      icon: <StarOutlined />,
      label: t('nav.watchlist'),
    },
    {
      key: '/settings',
      icon: <SettingOutlined />,
      label: t('nav.settings'),
    },
  ];

  return (
    <Layout style={{ minHeight: '100vh' }}>
      {/* Sidebar */}
      <Sider
        trigger={null}
        collapsible
        collapsed={collapsed}
        width={220}
        style={{
          borderRight: `1px solid ${borderColor}`,
          background: siderBg,
        }}
      >
        {/* Logo */}
        <div
          style={{
            height: 56,
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'flex-start',
            padding: collapsed ? 0 : '0 16px',
            borderBottom: `1px solid ${borderColor}`,
          }}
        >
          <LineChartOutlined style={{ fontSize: 24, color: '#1890ff' }} />
          {!collapsed && (
            <Text
              strong
              style={{
                marginLeft: 12,
                fontSize: 18,
                color: textColor,
                letterSpacing: 1,
              }}
            >
              FinMind
            </Text>
          )}
        </div>

        {/* Navigation */}
        <Menu
          theme={isDark ? 'dark' : 'light'}
          mode="inline"
          selectedKeys={[location.pathname]}
          items={menuItems}
          onClick={({ key }) => navigate(key)}
          style={{ borderRight: 0, background: siderBg }}
        />

        {/* GitHub Link */}
        <div
          style={{
            position: 'absolute',
            bottom: 16,
            left: 0,
            right: 0,
            textAlign: 'center',
          }}
        >
          <Button
            type="text"
            icon={<GithubOutlined />}
            href="https://github.com/hongspell/FinMind"
            target="_blank"
            style={{ color: secondaryColor }}
          >
            {!collapsed && 'GitHub'}
          </Button>
        </div>
      </Sider>

      <Layout>
        {/* Header */}
        <Header
          style={{
            padding: '0 24px',
            height: 56,
            lineHeight: '56px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            borderBottom: `1px solid ${borderColor}`,
            background: siderBg,
          }}
        >
          {/* Left: Collapse Button */}
          <Button
            type="text"
            icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
            onClick={() => setCollapsed(!collapsed)}
            style={{ color: textColor }}
          />

          {/* Right: Actions */}
          <Space size="middle" align="center">
            <LanguageSwitcher />
            <Button
              type="text"
              icon={<BellOutlined />}
              style={{ color: secondaryColor }}
            />
            <Dropdown
              menu={{
                items: [
                  { key: 'profile', label: t('nav.profile') },
                  { key: 'settings', label: t('nav.settings') },
                  { type: 'divider' },
                  { key: 'logout', label: t('nav.logout') },
                ],
              }}
              placement="bottomRight"
            >
              <Avatar
                style={{
                  backgroundColor: '#1890ff',
                  cursor: 'pointer',
                }}
              >
                U
              </Avatar>
            </Dropdown>
          </Space>
        </Header>

        {/* Content */}
        <Content
          style={{
            padding: 24,
            minHeight: 280,
            overflow: 'auto',
            background: isDark ? '#0d1117' : '#f5f5f5',
          }}
        >
          {children}
        </Content>
      </Layout>
    </Layout>
  );
};

export default AppLayout;
