import React, { useEffect, Component, ErrorInfo, ReactNode } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { ConfigProvider, Alert, Button } from 'antd';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { AppLayout } from './components/Layout';
import Dashboard from './pages/Dashboard';
import AnalysisPage from './pages/AnalysisPage';
import WatchlistPage from './pages/WatchlistPage';
import PortfolioPage from './pages/PortfolioPage';
import SettingsPage from './pages/SettingsPage';
import { darkTheme, lightTheme } from './styles/theme';
import { useSettingsStore } from './stores/settingsStore';
import './styles/global.css';

// 错误边界组件
interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('React Error Boundary caught an error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div style={{ padding: 40, background: '#fff', minHeight: '100vh' }}>
          <Alert
            message="应用程序错误 / Application Error"
            description={
              <div>
                <p>{this.state.error?.message}</p>
                <pre style={{ fontSize: 12, overflow: 'auto', maxHeight: 200 }}>
                  {this.state.error?.stack}
                </pre>
                <Button onClick={() => window.location.reload()}>
                  刷新页面 / Reload Page
                </Button>
              </div>
            }
            type="error"
            showIcon
          />
        </div>
      );
    }

    return this.props.children;
  }
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 2,
    },
  },
});

// 主题包装组件
const ThemedApp: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const { theme } = useSettingsStore();
  const currentTheme = theme === 'light' ? lightTheme : darkTheme;

  // 更新 body 背景色
  useEffect(() => {
    if (theme === 'light') {
      document.body.style.backgroundColor = '#f5f5f5';
      document.body.style.color = '#1f1f1f';
      document.documentElement.setAttribute('data-theme', 'light');
    } else {
      document.body.style.backgroundColor = '#0d1117';
      document.body.style.color = '#e6edf3';
      document.documentElement.setAttribute('data-theme', 'dark');
    }
  }, [theme]);

  return (
    <ConfigProvider theme={currentTheme}>
      {children}
    </ConfigProvider>
  );
};

const App: React.FC = () => {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <ThemedApp>
          <BrowserRouter>
            <AppLayout>
              <Routes>
                <Route path="/" element={<Dashboard />} />
                <Route path="/analysis" element={<Dashboard />} />
                <Route path="/analysis/:symbol" element={<AnalysisPage />} />
                <Route path="/watchlist" element={<WatchlistPage />} />
                <Route path="/portfolio" element={<PortfolioPage />} />
                <Route path="/settings" element={<SettingsPage />} />
              </Routes>
            </AppLayout>
          </BrowserRouter>
        </ThemedApp>
      </QueryClientProvider>
    </ErrorBoundary>
  );
};

export default App;
