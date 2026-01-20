"""
FinanceAI Pro - API模块

FastAPI REST API服务，提供：
- /api/v1/analyze: 创建分析任务
- /api/v1/scan: 批量扫描
- /api/v1/quote: 实时报价
- /api/v1/valuation: 快速估值
- /health: 健康检查
"""

from .main import app

__all__ = ["app"]
