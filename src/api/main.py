"""
FinMind - FastAPI REST API服务

提供RESTful API接口，支持：
- 股票分析请求
- 批量扫描
- 实时状态查询
- WebSocket推送
"""

import asyncio
import os
import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
import redis.asyncio as redis

# 导入路由
from .broker_routes import router as broker_router
from .analysis_routes import router as analysis_router

logger = logging.getLogger(__name__)

# ============================================================================
# Pydantic Models
# ============================================================================

class AnalysisStatus(str, Enum):
    """分析任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisRequest(BaseModel):
    """分析请求"""
    target: str = Field(..., description="分析目标，如 AAPL, 600519.SH")
    chain: str = Field(default="full_analysis", description="分析链名称")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="自定义参数")
    priority: int = Field(default=5, ge=1, le=10, description="优先级 1-10")
    
    class Config:
        json_schema_extra = {
            "example": {
                "target": "AAPL",
                "chain": "full_analysis",
                "parameters": {"scenarios": ["bull", "base", "bear"]},
                "priority": 5
            }
        }


class BatchScanRequest(BaseModel):
    """批量扫描请求"""
    targets: List[str] = Field(..., min_length=1, max_length=50)
    chain: str = Field(default="quick_scan")
    filters: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
    """分析响应"""
    task_id: str
    status: AnalysisStatus
    target: str
    chain: str
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress: float = Field(default=0.0, ge=0, le=1)
    current_stage: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, str]


class ValuationSummary(BaseModel):
    """估值摘要"""
    fair_value_low: float
    fair_value_mid: float
    fair_value_high: float
    current_price: float
    upside_percent: float
    confidence: float
    primary_method: str


class RecommendationResponse(BaseModel):
    """投资建议响应"""
    target: str
    recommendation: str  # strong_buy, buy, hold, sell, strong_sell
    conviction: str  # high, medium, low
    price_target: ValuationSummary
    key_catalysts: List[str]
    key_risks: List[str]
    time_horizon: str
    generated_at: datetime


# ============================================================================
# 任务存储（生产环境应使用Redis/数据库）
# ============================================================================

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """添加安全响应头"""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
        return response


class TaskStore:
    """任务存储管理"""

    _MAX_TASKS = 1000
    _TASK_TTL_SECONDS = 86400  # 24小时

    def __init__(self):
        self._tasks: Dict[str, AnalysisResponse] = {}
        self._redis: Optional[redis.Redis] = None
    
    async def connect_redis(self, url: str = "redis://localhost:6379"):
        """连接Redis"""
        try:
            self._redis = redis.from_url(url, decode_responses=True)
            await self._redis.ping()
        except Exception:
            # 连接失败时确保关闭已创建的连接
            if self._redis:
                try:
                    await self._redis.close()
                except Exception:
                    pass
            self._redis = None

    async def close_redis(self):
        """关闭Redis连接"""
        if self._redis:
            try:
                await self._redis.close()
            except Exception:
                pass
            self._redis = None
    
    def _cleanup_old_tasks(self):
        """清理已完成的旧任务，防止内存泄漏"""
        if len(self._tasks) < self._MAX_TASKS:
            return
        completed = [
            (tid, t) for tid, t in self._tasks.items()
            if t.status in (AnalysisStatus.COMPLETED, AnalysisStatus.FAILED, AnalysisStatus.CANCELLED)
        ]
        completed.sort(key=lambda x: x[1].created_at)
        to_remove = len(self._tasks) - self._MAX_TASKS + 100  # 清理出余量
        for tid, _ in completed[:to_remove]:
            del self._tasks[tid]

    async def create_task(self, request: AnalysisRequest) -> AnalysisResponse:
        """创建新任务"""
        self._cleanup_old_tasks()

        task_id = str(uuid.uuid4())
        task = AnalysisResponse(
            task_id=task_id,
            status=AnalysisStatus.PENDING,
            target=request.target,
            chain=request.chain,
            created_at=datetime.utcnow(),
            progress=0.0
        )
        self._tasks[task_id] = task
        
        if self._redis:
            await self._redis.hset(
                f"task:{task_id}",
                mapping=task.model_dump(mode="json")
            )
            await self._redis.expire(f"task:{task_id}", 86400)  # 24小时过期
        
        return task
    
    async def get_task(self, task_id: str) -> Optional[AnalysisResponse]:
        """获取任务"""
        if task_id in self._tasks:
            return self._tasks[task_id]

        if self._redis:
            try:
                data = await self._redis.hgetall(f"task:{task_id}")
                if data:
                    return AnalysisResponse(**data)
            except (ValueError, TypeError):
                pass

        return None
    
    async def update_task(self, task_id: str, **updates) -> Optional[AnalysisResponse]:
        """更新任务"""
        task = await self.get_task(task_id)
        if not task:
            return None
        
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)
        
        self._tasks[task_id] = task
        
        if self._redis:
            await self._redis.hset(
                f"task:{task_id}",
                mapping=task.model_dump(mode="json")
            )
        
        return task
    
    async def list_tasks(
        self, 
        status: Optional[AnalysisStatus] = None,
        limit: int = 20
    ) -> List[AnalysisResponse]:
        """列出任务"""
        tasks = list(self._tasks.values())
        
        if status:
            tasks = [t for t in tasks if t.status == status]
        
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]


# ============================================================================
# 全局状态
# ============================================================================

task_store = TaskStore()
start_time = datetime.utcnow()


# ============================================================================
# FastAPI 应用
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时
    await task_store.connect_redis()
    print("FinMind API 启动完成")
    yield
    # 关闭时 - 清理所有资源
    await task_store.close_redis()
    print("FinMind API 关闭")


_environment = os.environ.get("ENVIRONMENT", "development")
_is_production = _environment == "production"

app = FastAPI(
    title="FinMind API",
    description="模块化金融AI分析平台 REST API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if _is_production else "/docs",
    redoc_url=None if _is_production else "/redoc",
)

# 安全响应头中间件
app.add_middleware(SecurityHeadersMiddleware)

# CORS 配置 - 从环境变量读取白名单
_cors_origins_env = os.environ.get("CORS_ORIGINS", "")
_allowed_origins = (
    [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins_env
    else ["http://localhost:3000", "http://localhost:5173"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(broker_router)
app.include_router(analysis_router)


# 全局异常处理 - 错误信息脱敏
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ============================================================================
# 健康检查端点
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """系统健康检查"""
    uptime = (datetime.utcnow() - start_time).total_seconds()
    
    components = {
        "api": "healthy",
        "redis": "healthy" if task_store._redis else "unavailable",
        "llm_gateway": "healthy",  # TODO: 实际检查
        "data_providers": "healthy"  # TODO: 实际检查
    }
    
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=uptime,
        components=components
    )


@app.get("/", tags=["System"])
async def root():
    """API根路径"""
    return {
        "name": "FinMind API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }


# ============================================================================
# 分析端点
# ============================================================================

@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["Analysis"])
async def create_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    创建新的分析任务
    
    - **target**: 分析目标（股票代码）
    - **chain**: 使用的分析链
    - **parameters**: 自定义参数
    - **priority**: 任务优先级
    
    返回任务ID，可通过 /api/v1/analyze/{task_id} 查询进度
    """
    task = await task_store.create_task(request)
    
    # 添加后台任务执行分析
    background_tasks.add_task(run_analysis, task.task_id, request)
    
    return task


async def run_analysis(task_id: str, request: AnalysisRequest):
    """
    后台执行分析任务

    TODO: 集成真实的 FinanceAI 分析代理
    当前版本尚未完成分析代理集成，将返回错误状态
    """
    try:
        await task_store.update_task(
            task_id,
            status=AnalysisStatus.RUNNING,
            started_at=datetime.utcnow(),
            current_stage="initializing"
        )

        # TODO: 集成真实的分析代理系统
        # 当前版本分析代理尚未完成集成，返回错误状态
        await task_store.update_task(
            task_id,
            status=AnalysisStatus.FAILED,
            error="分析服务尚未完成配置。需要配置 LLM API 密钥和数据源才能进行完整分析。",
            result={
                "target": request.target,
                "error": "service_not_configured",
                "message": "完整股票分析功能需要配置以下内容：",
                "requirements": [
                    "LLM API 密钥（OpenAI/Anthropic/本地模型）",
                    "财务数据源（如 Polygon.io, Alpha Vantage）",
                    "市场数据源"
                ],
                "suggestion": "请查阅文档配置相关服务，或使用已连接券商的投资组合分析功能"
            }
        )

    except Exception as e:
        await task_store.update_task(
            task_id,
            status=AnalysisStatus.FAILED,
            error=str(e)
        )


@app.get("/api/v1/analyze/{task_id}", response_model=AnalysisResponse, tags=["Analysis"])
async def get_analysis(task_id: str):
    """获取分析任务状态和结果"""
    task = await task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.delete("/api/v1/analyze/{task_id}", tags=["Analysis"])
async def cancel_analysis(task_id: str):
    """取消分析任务"""
    task = await task_store.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.status in [AnalysisStatus.COMPLETED, AnalysisStatus.FAILED]:
        raise HTTPException(status_code=400, detail="Cannot cancel completed task")
    
    await task_store.update_task(task_id, status=AnalysisStatus.CANCELLED)
    return {"message": "Task cancelled", "task_id": task_id}


@app.get("/api/v1/tasks", response_model=List[AnalysisResponse], tags=["Analysis"])
async def list_tasks(
    status: Optional[AnalysisStatus] = Query(None),
    limit: int = Query(20, ge=1, le=100)
):
    """列出分析任务"""
    return await task_store.list_tasks(status=status, limit=limit)


# ============================================================================
# 批量扫描端点
# ============================================================================

@app.post("/api/v1/scan", tags=["Scanning"])
async def batch_scan(
    request: BatchScanRequest,
    background_tasks: BackgroundTasks
):
    """
    批量扫描多个股票
    
    适用于快速筛选，返回每个目标的简要评估
    """
    scan_id = str(uuid.uuid4())
    tasks = []
    
    for target in request.targets:
        analysis_request = AnalysisRequest(
            target=target,
            chain=request.chain,
            parameters=request.filters
        )
        task = await task_store.create_task(analysis_request)
        tasks.append(task)
        background_tasks.add_task(run_analysis, task.task_id, analysis_request)
    
    return {
        "scan_id": scan_id,
        "total_targets": len(request.targets),
        "tasks": [t.task_id for t in tasks]
    }


# ============================================================================
# 快速查询端点
# ============================================================================

@app.get("/api/v1/quote/{symbol}", tags=["Quick Query"])
async def get_quote(symbol: str):
    """
    获取实时报价

    注意：此接口需要外部数据源支持
    """
    # TODO: 集成真实的市场数据源（如 yfinance, polygon.io, Alpha Vantage 等）
    raise HTTPException(
        status_code=501,
        detail={
            "error": "data_source_not_configured",
            "message": f"获取 {symbol.upper()} 的实时报价需要集成外部数据源",
            "suggestion": "请配置市场数据提供商（如 Polygon.io, Alpha Vantage）"
        }
    )


@app.get("/api/v1/valuation/{symbol}", tags=["Quick Query"])
async def get_quick_valuation(symbol: str):
    """
    快速估值查询

    注意：此接口需要财务数据源和估值模型支持
    """
    # TODO: 集成真实的财务数据源和估值计算
    raise HTTPException(
        status_code=501,
        detail={
            "error": "valuation_service_not_available",
            "message": f"获取 {symbol.upper()} 的估值需要集成财务数据源和估值模型",
            "suggestion": "请使用完整分析功能或配置财务数据提供商"
        }
    )


@app.get("/api/v1/recommendation/{symbol}", tags=["Quick Query"])
async def get_recommendation(symbol: str):
    """
    获取投资建议摘要

    注意：此接口需要完成完整分析后才能提供建议
    """
    # TODO: 从缓存或数据库获取已完成的分析结果
    raise HTTPException(
        status_code=501,
        detail={
            "error": "recommendation_not_available",
            "message": f"获取 {symbol.upper()} 的投资建议需要先完成完整分析",
            "suggestion": "请使用 POST /api/v1/analyze 接口发起分析任务"
        }
    )


# ============================================================================
# 方法论配置端点
# ============================================================================

@app.get("/api/v1/config/chains", tags=["Configuration"])
async def list_chains():
    """列出可用的分析链"""
    return {
        "chains": [
            {
                "name": "full_analysis",
                "description": "完整分析，包含所有Agent",
                "estimated_time": "5-7 minutes"
            },
            {
                "name": "quick_scan",
                "description": "快速扫描，仅核心指标",
                "estimated_time": "1-2 minutes"
            },
            {
                "name": "valuation_only",
                "description": "仅估值分析",
                "estimated_time": "2-3 minutes"
            },
            {
                "name": "earnings_deep_dive",
                "description": "财报深度分析",
                "estimated_time": "3-5 minutes"
            }
        ]
    }


@app.get("/api/v1/config/agents", tags=["Configuration"])
async def list_agents():
    """列出可用的Agent"""
    return {
        "agents": [
            {"name": "valuation", "description": "估值分析"},
            {"name": "technical", "description": "技术分析"},
            {"name": "sentiment", "description": "情绪分析"},
            {"name": "risk", "description": "风险评估"},
            {"name": "earnings", "description": "财报分析"},
            {"name": "strategy", "description": "策略综合"},
            {"name": "sector", "description": "行业分析"},
            {"name": "macro", "description": "宏观分析"}
        ]
    }


# ============================================================================
# 流式响应端点
# ============================================================================

@app.get("/api/v1/analyze/{task_id}/stream", tags=["Analysis"])
async def stream_analysis(task_id: str):
    """
    流式获取分析进度（Server-Sent Events）
    
    适用于前端实时显示分析进度
    """
    max_poll_seconds = 3600  # 最多轮询 1 小时

    async def event_generator():
        elapsed = 0
        while elapsed < max_poll_seconds:
            task = await task_store.get_task(task_id)
            if not task:
                yield f"event: error\ndata: Task not found\n\n"
                break

            yield f"data: {task.model_dump_json()}\n\n"

            if task.status in [AnalysisStatus.COMPLETED,
                               AnalysisStatus.FAILED,
                               AnalysisStatus.CANCELLED]:
                break

            await asyncio.sleep(1)
            elapsed += 1
        else:
            yield f"event: timeout\ndata: Stream timeout after {max_poll_seconds}s\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# ============================================================================
# CLI入口
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
