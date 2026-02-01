"""
FinMind - TaskStore: Async Analysis Task Management

Provides task lifecycle management for async analysis jobs:
- In-memory task storage with optional Redis backing
- TTL-based cleanup to prevent memory leaks
- CRUD operations for analysis tasks
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ============================================================================
# Models
# ============================================================================

class AnalysisStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AnalysisRequest(BaseModel):
    target: str = Field(..., description="Analysis target, e.g. AAPL, 600519.SH")
    chain: str = Field(default="full_analysis", description="Analysis chain name")
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=5, ge=1, le=10)

    class Config:
        json_schema_extra = {
            "example": {
                "target": "AAPL",
                "chain": "full_analysis",
                "parameters": {"scenarios": ["bull", "base", "bear"]},
                "priority": 5,
            }
        }


class BatchScanRequest(BaseModel):
    targets: List[str] = Field(..., min_length=1, max_length=50)
    chain: str = Field(default="quick_scan")
    filters: Dict[str, Any] = Field(default_factory=dict)


class AnalysisResponse(BaseModel):
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


# ============================================================================
# TaskStore
# ============================================================================

class TaskStore:
    _MAX_TASKS = 1000
    _TASK_TTL_SECONDS = 86400  # 24 hours

    def __init__(self):
        self._tasks: Dict[str, AnalysisResponse] = {}
        self._redis = None

    async def connect_redis(self, url: str = "redis://localhost:6379"):
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(url, decode_responses=True)
            await self._redis.ping()
            logger.info("TaskStore connected to Redis")
        except Exception:
            if self._redis:
                try:
                    await self._redis.close()
                except Exception:
                    pass
            self._redis = None
            logger.info("TaskStore using in-memory storage (Redis unavailable)")

    async def close_redis(self):
        if self._redis:
            try:
                await self._redis.close()
            except Exception:
                pass
            self._redis = None

    def _cleanup_old_tasks(self):
        if len(self._tasks) < self._MAX_TASKS:
            return
        completed = [
            (tid, t)
            for tid, t in self._tasks.items()
            if t.status
            in (
                AnalysisStatus.COMPLETED,
                AnalysisStatus.FAILED,
                AnalysisStatus.CANCELLED,
            )
        ]
        completed.sort(key=lambda x: x[1].created_at)
        to_remove = len(self._tasks) - self._MAX_TASKS + 100
        for tid, _ in completed[:to_remove]:
            del self._tasks[tid]

    async def create_task(self, request: AnalysisRequest) -> AnalysisResponse:
        self._cleanup_old_tasks()
        task_id = str(uuid.uuid4())
        task = AnalysisResponse(
            task_id=task_id,
            status=AnalysisStatus.PENDING,
            target=request.target,
            chain=request.chain,
            created_at=datetime.utcnow(),
            progress=0.0,
        )
        self._tasks[task_id] = task

        if self._redis:
            try:
                await self._redis.hset(
                    f"task:{task_id}", mapping=task.model_dump(mode="json")
                )
                await self._redis.expire(f"task:{task_id}", self._TASK_TTL_SECONDS)
            except Exception as e:
                logger.warning(f"Failed to persist task to Redis: {e}")

        return task

    async def get_task(self, task_id: str) -> Optional[AnalysisResponse]:
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

    async def update_task(
        self, task_id: str, **updates
    ) -> Optional[AnalysisResponse]:
        task = await self.get_task(task_id)
        if not task:
            return None

        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        self._tasks[task_id] = task

        if self._redis:
            try:
                await self._redis.hset(
                    f"task:{task_id}", mapping=task.model_dump(mode="json")
                )
            except Exception as e:
                logger.warning(f"Failed to update task in Redis: {e}")

        return task

    async def list_tasks(
        self, status: Optional[AnalysisStatus] = None, limit: int = 20
    ) -> List[AnalysisResponse]:
        tasks = list(self._tasks.values())
        if status:
            tasks = [t for t in tasks if t.status == status]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]
