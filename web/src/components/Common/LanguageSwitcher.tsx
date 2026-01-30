import React from 'react';
import { Dropdown, Button } from 'antd';
import { GlobalOutlined } from '@ant-design/icons';
import { useTranslation } from 'react-i18next';
import type { MenuProps } from 'antd';

const LanguageSwitcher: React.FC = () => {
  const { i18n, t } = useTranslation();

  const currentLanguage = i18n.language?.startsWith('zh') ? 'zh' : 'en';

  const items: MenuProps['items'] = [
    {
      key: 'en',
      label: '🇺🇸 English',
      onClick: () => i18n.changeLanguage('en'),
    },
    {
      key: 'zh',
      label: '🇨🇳 中文',
      onClick: () => i18n.changeLanguage('zh'),
    },
  ];

  return (
    <Dropdown menu={{ items, selectedKeys: [currentLanguage] }} placement="bottomRight">
      <Button
        type="text"
        icon={<GlobalOutlined />}
        style={{ color: '#8b949e' }}
      >
        {t(`language.${currentLanguage}`)}
      </Button>
    </Dropdown>
  );
};

export default React.memo(LanguageSwitcher);
