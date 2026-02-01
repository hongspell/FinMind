"""
FinMind - 交易记录本地持久化组件

提供统一的 JSON 文件读写 + 去重逻辑，供所有券商适配器组合使用。
每个适配器通过构造参数指定文件前缀和去重字段即可，无需各自实现存储逻辑。

存储路径: ~/.finmind/{prefix}_{account_id}.json
"""

import json
import logging
import os
from typing import List, Sequence, Union

logger = logging.getLogger(__name__)

_STORE_DIR = os.path.expanduser("~/.finmind")


class TradeStore:
    """本地 JSON 文件交易记录存储

    用法:
        store = TradeStore("futu_trades", dedup_keys="deal_id")
        store.set_account("88888888")
        store.persist([{"deal_id": "D1", ...}])
        records = store.load()

    Args:
        prefix: 文件名前缀，如 "ibkr_trades"、"futu_trades"
        dedup_keys: 用于去重的字段名，字符串或字符串列表。
                    记录中任意一个 key 的值已存在即视为重复。
        max_records: 最多保留的记录数，防止文件无限增长。
    """

    def __init__(
        self,
        prefix: str,
        dedup_keys: Union[str, Sequence[str]],
        max_records: int = 5000,
    ):
        self._prefix = prefix
        self._dedup_keys: List[str] = (
            [dedup_keys] if isinstance(dedup_keys, str) else list(dedup_keys)
        )
        self._max_records = max_records
        self._path: str | None = None

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------

    def set_account(self, account_id: str) -> None:
        """设置账户 ID（决定存储文件路径），调用 load/persist 前必须先调用"""
        os.makedirs(_STORE_DIR, exist_ok=True)
        self._path = os.path.join(
            _STORE_DIR, f"{self._prefix}_{account_id}.json"
        )

    @property
    def path(self) -> str | None:
        return self._path

    def load(self) -> list:
        """从本地文件加载所有交易记录"""
        if not self._path or not os.path.exists(self._path):
            return []
        try:
            with open(self._path, "r") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load trade store {self._path}: {e}")
            return []

    def persist(self, new_records: list) -> int:
        """将新记录追加到本地文件，自动去重

        Args:
            new_records: 新的交易记录字典列表

        Returns:
            实际新增的记录数
        """
        if not self._path:
            logger.warning("TradeStore: account not set, skipping persist")
            return 0

        existing = self.load()
        existing_ids = self._build_id_set(existing)

        new_count = 0
        for record in new_records:
            record_ids = self._extract_ids(record)
            # 如果该记录的任何 dedup key 值已存在，跳过
            if record_ids & existing_ids:
                continue
            existing.append(record)
            existing_ids |= record_ids
            new_count += 1

        if new_count > 0:
            # 截断至最大记录数
            if len(existing) > self._max_records:
                existing = existing[-self._max_records:]
            self._save(existing)
            logger.info(f"Persisted {new_count} new records to {self._path}")

        return new_count

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _build_id_set(self, records: list) -> set:
        """从现有记录中提取所有 dedup key 的值"""
        ids = set()
        for record in records:
            ids |= self._extract_ids(record)
        return ids

    def _extract_ids(self, record: dict) -> set:
        """从单条记录中提取所有非空 dedup key 值"""
        ids = set()
        for key in self._dedup_keys:
            val = record.get(key)
            if val:
                ids.add(val)
        return ids

    def _save(self, records: list) -> None:
        """写入文件"""
        try:
            with open(self._path, "w") as f:
                json.dump(records, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save trade store {self._path}: {e}")
