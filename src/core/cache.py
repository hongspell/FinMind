"""
FinMind - 缓存服务

提供统一的缓存抽象层，支持 Redis 和内存缓存。

特性：
- 自动序列化/反序列化
- TTL 支持
- 缓存装饰器
- 优雅降级（Redis 不可用时使用内存缓存）
"""

import asyncio
import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional, Any, TypeVar, Callable, Dict, Union
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T')


# =============================================================================
# 配置
# =============================================================================

@dataclass
class CacheConfig:
    """缓存配置"""
    redis_url: str = "redis://localhost:6379"
    default_ttl: int = 300  # 5分钟
    key_prefix: str = "finmind:"
    enable_memory_fallback: bool = True
    max_memory_items: int = 1000


# =============================================================================
# 缓存抽象基类
# =============================================================================

class CacheBackend(ABC):
    """缓存后端抽象基类"""

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存值"""
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """删除缓存"""
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        pass

    @abstractmethod
    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass


# =============================================================================
# 内存缓存
# =============================================================================

class MemoryCache(CacheBackend):
    """
    内存缓存实现

    使用字典存储，支持 TTL 和 LRU 淘汰。
    """

    def __init__(self, max_items: int = 1000):
        self._cache: Dict[str, tuple] = {}  # {key: (value, expire_time)}
        self._max_items = max_items
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None

            value, expire_time = self._cache[key]
            if expire_time and time.time() > expire_time:
                del self._cache[key]
                return None

            # LRU：访问时移动到末尾（最近使用）
            del self._cache[key]
            self._cache[key] = (value, expire_time)

            return value

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        async with self._lock:
            # 如果 key 已存在，先删除再插入以更新顺序
            if key in self._cache:
                del self._cache[key]
            elif len(self._cache) >= self._max_items:
                # LRU 淘汰：字典头部是最久未访问的
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]

            expire_time = time.time() + ttl if ttl else None
            self._cache[key] = (value, expire_time)
            return True

    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    async def exists(self, key: str) -> bool:
        value = await self.get(key)
        return value is not None

    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存（简单实现）"""
        async with self._lock:
            # 简单的前缀匹配
            prefix = pattern.rstrip('*')
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for k in keys_to_delete:
                del self._cache[k]
            return len(keys_to_delete)

    async def health_check(self) -> bool:
        return True

    @property
    def size(self) -> int:
        return len(self._cache)


# =============================================================================
# Redis 缓存
# =============================================================================

class RedisCache(CacheBackend):
    """
    Redis 缓存实现

    使用 redis.asyncio 库。
    """

    def __init__(self, url: str = "redis://localhost:6379"):
        self._url = url
        self._redis = None
        self._connected = False

    async def connect(self) -> bool:
        """连接到 Redis"""
        try:
            import redis.asyncio as aioredis

            self._redis = aioredis.from_url(
                self._url,
                encoding="utf-8",
                decode_responses=True,
            )
            await self._redis.ping()
            self._connected = True
            logger.info(f"Connected to Redis: {self._url}")
            return True
        except ImportError:
            logger.warning("redis library not installed")
            return False
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            return False

    async def disconnect(self) -> None:
        """断开连接"""
        if self._redis:
            await self._redis.close()
            self._connected = False

    async def get(self, key: str) -> Optional[Any]:
        if not self._connected:
            return None

        try:
            value = await self._redis.get(key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        if not self._connected:
            return False

        try:
            serialized = json.dumps(value, default=str)
            if ttl:
                await self._redis.setex(key, ttl, serialized)
            else:
                await self._redis.set(key, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    async def delete(self, key: str) -> bool:
        if not self._connected:
            return False

        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    async def exists(self, key: str) -> bool:
        if not self._connected:
            return False

        try:
            return await self._redis.exists(key) > 0
        except Exception:
            return False

    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        if not self._connected:
            return 0

        try:
            keys = []
            async for key in self._redis.scan_iter(match=pattern):
                keys.append(key)

            if keys:
                await self._redis.delete(*keys)
            return len(keys)
        except Exception as e:
            logger.error(f"Redis clear_pattern error: {e}")
            return 0

    async def health_check(self) -> bool:
        if not self._connected or not self._redis:
            return False

        try:
            await self._redis.ping()
            return True
        except Exception:
            return False


# =============================================================================
# 统一缓存服务
# =============================================================================

class CacheService:
    """
    统一缓存服务

    优先使用 Redis，不可用时降级到内存缓存。

    Example:
        ```python
        cache = CacheService()
        await cache.initialize()

        # 设置缓存
        await cache.set("user:123", {"name": "John"}, ttl=3600)

        # 获取缓存
        user = await cache.get("user:123")

        # 使用装饰器
        @cache.cached(ttl=300)
        async def get_stock_data(symbol: str):
            # 耗时操作
            return await fetch_from_api(symbol)
        ```
    """

    def __init__(self, config: Optional[CacheConfig] = None):
        self.config = config or CacheConfig()
        self._redis_cache = RedisCache(self.config.redis_url)
        self._memory_cache = MemoryCache(self.config.max_memory_items)
        self._initialized = False

    @property
    def backend(self) -> CacheBackend:
        """当前使用的缓存后端"""
        if self._redis_cache._connected:
            return self._redis_cache
        return self._memory_cache

    @property
    def backend_name(self) -> str:
        """当前后端名称"""
        if self._redis_cache._connected:
            return "redis"
        return "memory"

    async def initialize(self) -> None:
        """初始化缓存服务"""
        # 尝试连接 Redis
        redis_ok = await self._redis_cache.connect()

        if redis_ok:
            logger.info("Cache service initialized with Redis backend")
        elif self.config.enable_memory_fallback:
            logger.info("Cache service initialized with memory backend (Redis unavailable)")
        else:
            logger.warning("Cache service: no backend available")

        self._initialized = True

    async def shutdown(self) -> None:
        """关闭缓存服务"""
        await self._redis_cache.disconnect()
        self._initialized = False

    def _make_key(self, key: str) -> str:
        """生成带前缀的完整键"""
        return f"{self.config.key_prefix}{key}"

    async def get(self, key: str) -> Optional[Any]:
        """获取缓存"""
        full_key = self._make_key(key)
        return await self.backend.get(full_key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存"""
        full_key = self._make_key(key)
        ttl = ttl or self.config.default_ttl
        return await self.backend.set(full_key, value, ttl)

    async def delete(self, key: str) -> bool:
        """删除缓存"""
        full_key = self._make_key(key)
        return await self.backend.delete(full_key)

    async def exists(self, key: str) -> bool:
        """检查缓存是否存在"""
        full_key = self._make_key(key)
        return await self.backend.exists(full_key)

    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: Optional[int] = None,
    ) -> Any:
        """
        获取缓存，不存在则调用工厂函数生成并缓存

        Args:
            key: 缓存键
            factory: 工厂函数（无参数）
            ttl: 过期时间

        Returns:
            缓存的值
        """
        value = await self.get(key)
        if value is not None:
            return value

        # 调用工厂函数
        if asyncio.iscoroutinefunction(factory):
            value = await factory()
        else:
            value = factory()

        await self.set(key, value, ttl)
        return value

    async def clear_pattern(self, pattern: str) -> int:
        """清除匹配模式的缓存"""
        full_pattern = self._make_key(pattern)
        return await self.backend.clear_pattern(full_pattern)

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        redis_ok = await self._redis_cache.health_check()
        memory_ok = await self._memory_cache.health_check()

        return {
            "status": "healthy" if (redis_ok or memory_ok) else "unhealthy",
            "backend": self.backend_name,
            "redis": {
                "connected": redis_ok,
                "url": self.config.redis_url,
            },
            "memory": {
                "enabled": self.config.enable_memory_fallback,
                "items": self._memory_cache.size,
                "max_items": self.config.max_memory_items,
            },
        }

    # =========================================================================
    # 缓存装饰器
    # =========================================================================

    def cached(
        self,
        ttl: Optional[int] = None,
        key_prefix: str = "",
        key_builder: Optional[Callable[..., str]] = None,
    ):
        """
        缓存装饰器

        Args:
            ttl: 过期时间（秒）
            key_prefix: 键前缀
            key_builder: 自定义键生成函数

        Example:
            ```python
            @cache.cached(ttl=300, key_prefix="stock:")
            async def get_stock_price(symbol: str):
                return await api.get_price(symbol)
            ```
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                # 生成缓存键
                if key_builder:
                    cache_key = key_builder(*args, **kwargs)
                else:
                    cache_key = self._build_key(func, args, kwargs)

                if key_prefix:
                    cache_key = f"{key_prefix}{cache_key}"

                # 尝试从缓存获取
                cached_value = await self.get(cache_key)
                if cached_value is not None:
                    logger.debug(f"Cache hit: {cache_key}")
                    return cached_value

                # 调用原函数
                logger.debug(f"Cache miss: {cache_key}")
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # 缓存结果
                await self.set(cache_key, result, ttl)
                return result

            return wrapper
        return decorator

    def _build_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """生成缓存键"""
        # 函数名
        key_parts = [func.__module__, func.__name__]

        # 参数
        for arg in args:
            key_parts.append(str(arg))
        for k, v in sorted(kwargs.items()):
            key_parts.append(f"{k}={v}")

        # 生成哈希
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def invalidate(self, key_prefix: str = ""):
        """
        缓存失效装饰器

        在函数执行后清除匹配的缓存。

        Example:
            ```python
            @cache.invalidate(key_prefix="stock:")
            async def update_stock(symbol: str, data: dict):
                await db.update(symbol, data)
            ```
        """
        def decorator(func: Callable[..., T]) -> Callable[..., T]:
            @wraps(func)
            async def wrapper(*args, **kwargs) -> T:
                # 执行原函数
                if asyncio.iscoroutinefunction(func):
                    result = await func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)

                # 清除缓存
                pattern = f"{key_prefix}*"
                cleared = await self.clear_pattern(pattern)
                logger.debug(f"Invalidated {cleared} cache entries matching {pattern}")

                return result

            return wrapper
        return decorator


# =============================================================================
# 预定义的缓存键生成器
# =============================================================================

class CacheKeys:
    """预定义的缓存键"""

    @staticmethod
    def stock_quote(symbol: str) -> str:
        """股票报价缓存键"""
        return f"quote:{symbol.upper()}"

    @staticmethod
    def stock_analysis(symbol: str) -> str:
        """股票分析缓存键"""
        return f"analysis:{symbol.upper()}"

    @staticmethod
    def market_data(symbol: str, data_type: str) -> str:
        """市场数据缓存键"""
        return f"market:{symbol.upper()}:{data_type}"

    @staticmethod
    def user_portfolio(user_id: str) -> str:
        """用户投资组合缓存键"""
        return f"portfolio:{user_id}"

    @staticmethod
    def broker_positions(broker: str, account_id: str) -> str:
        """券商持仓缓存键"""
        return f"broker:{broker}:{account_id}:positions"


# =============================================================================
# 全局缓存实例
# =============================================================================

# 全局缓存服务（延迟初始化，线程安全）
_cache_service: Optional[CacheService] = None
_cache_lock = asyncio.Lock()


async def get_cache_service() -> CacheService:
    """获取全局缓存服务（并发安全）"""
    global _cache_service
    if _cache_service is not None:
        return _cache_service
    async with _cache_lock:
        # Double-check locking：锁内再次检查
        if _cache_service is None:
            service = CacheService()
            await service.initialize()
            _cache_service = service
    return _cache_service


async def shutdown_cache_service() -> None:
    """关闭全局缓存服务"""
    global _cache_service
    async with _cache_lock:
        if _cache_service:
            await _cache_service.shutdown()
            _cache_service = None
