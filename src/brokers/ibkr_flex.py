"""
FinMind - IBKR Flex Queries 历史交易导入

通过 Flex Web Service 导入完整的历史交易记录。
补充 TWS API 和 Client Portal 的不足。

流程：
1. SendRequest: 发起查询请求，获取 referenceCode
2. GetStatement: 轮询获取报表（XML 格式）
3. 解析 XML 提取交易数据
4. 存入本地持久化文件

使用前需要：
1. 在 IBKR 账户管理中创建 Flex Query (Activity Flex Query -> Trades)
2. 配置 Flex Web Service token
3. 获取 query_id
"""

import asyncio
import logging
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Optional

from .trade_store import TradeStore

logger = logging.getLogger(__name__)

_FLEX_BASE_URL = "https://ndcdyn.interactivebrokers.com/AccountManagement/FlexWebService"
_POLL_INTERVAL = 5  # seconds between GetStatement retries
_MAX_POLL_ATTEMPTS = 24  # max 2 minutes of polling


async def fetch_flex_statement(token: str, query_id: str) -> str:
    """获取 Flex 报表原始 XML

    两步流程：
    1. SendRequest -> 获取 referenceCode
    2. GetStatement -> 轮询获取 XML 内容

    Args:
        token: Flex Web Service token
        query_id: Flex Query ID

    Returns:
        XML 字符串

    Raises:
        Exception: 请求失败
    """
    try:
        import httpx
    except ImportError:
        raise RuntimeError("httpx library not installed. Install with: pip install httpx")

    async with httpx.AsyncClient(timeout=30) as client:
        # Step 1: SendRequest
        send_url = f"{_FLEX_BASE_URL}/SendRequest"
        resp = await client.get(
            send_url,
            params={"t": token, "q": query_id, "v": "3"},
        )

        if resp.status_code != 200:
            raise RuntimeError(f"Flex SendRequest failed: HTTP {resp.status_code}")

        # 解析 SendRequest 响应 XML
        send_xml = resp.text
        send_root = ET.fromstring(send_xml)

        status = send_root.findtext("Status")
        if status != "Success":
            error_msg = send_root.findtext("ErrorMessage") or "Unknown error"
            raise RuntimeError(f"Flex SendRequest failed: {error_msg}")

        reference_code = send_root.findtext("ReferenceCode")
        if not reference_code:
            raise RuntimeError("No ReferenceCode in SendRequest response")

        # Step 2: GetStatement (轮询)
        get_url = f"{_FLEX_BASE_URL}/GetStatement"

        for attempt in range(_MAX_POLL_ATTEMPTS):
            await asyncio.sleep(_POLL_INTERVAL)

            resp = await client.get(
                get_url,
                params={"t": token, "q": reference_code, "v": "3"},
            )

            if resp.status_code != 200:
                continue

            content = resp.text

            # 检查是否仍在生成中
            if content.strip().startswith("<"):
                try:
                    check_root = ET.fromstring(content)
                    check_status = check_root.findtext("Status")
                    if check_status == "Warn":
                        error_msg = check_root.findtext("ErrorMessage") or ""
                        if "please try again" in error_msg.lower():
                            logger.debug(f"Flex statement not ready, attempt {attempt + 1}")
                            continue
                        if check_status != "Success":
                            raise RuntimeError(f"Flex GetStatement failed: {error_msg}")
                except ET.ParseError:
                    pass

            # 如果包含 FlexQueryResponse，则成功
            if "FlexQueryResponse" in content or "FlexStatementResponse" in content:
                return content

        raise RuntimeError(
            f"Flex statement not ready after {_MAX_POLL_ATTEMPTS * _POLL_INTERVAL}s polling"
        )


def parse_flex_trades(xml_content: str) -> List[dict]:
    """解析 Flex XML 中的交易记录

    Args:
        xml_content: Flex 报表 XML 内容

    Returns:
        交易记录字典列表
    """
    root = ET.fromstring(xml_content)

    trades = []

    # 查找所有 Trade 元素（可能在不同层级）
    for trade_el in root.iter("Trade"):
        attrib = trade_el.attrib

        # 跳过非股票
        asset_category = attrib.get("assetCategory", "")
        if asset_category and asset_category not in ("STK", "ETF"):
            continue

        trade_time = None
        date_str = attrib.get("dateTime") or attrib.get("tradeDate")
        if date_str:
            try:
                # IBKR Flex 日期格式通常是 YYYYMMDD 或 YYYYMMDD;HHMMSS
                date_str = date_str.replace(";", " ")
                if len(date_str) == 8:
                    trade_time = datetime.strptime(date_str, "%Y%m%d").isoformat()
                elif len(date_str) >= 14:
                    trade_time = datetime.strptime(
                        date_str[:15], "%Y%m%d %H%M%S"
                    ).isoformat()
                else:
                    trade_time = date_str
            except Exception:
                trade_time = date_str

        # 构建交易记录字典（与 ibkr.py 的持久化格式兼容）
        side = attrib.get("buySell", "")
        exec_id = attrib.get("ibExecID") or attrib.get("tradeID") or ""
        trade_id = attrib.get("tradeID", "")

        trade_dict = {
            "exec_id": exec_id,
            "trade_id": trade_id,
            "symbol": attrib.get("symbol", ""),
            "sec_type": attrib.get("assetCategory", "STK"),
            "exchange": attrib.get("exchange", ""),
            "currency": attrib.get("currency", "USD"),
            "side": "BOT" if side in ("BUY", "BOT") else "SLD",
            "shares": abs(float(attrib.get("quantity", 0))),
            "price": float(attrib.get("tradePrice", 0)),
            "order_id": attrib.get("ibOrderID", ""),
            "time": trade_time,
            "commission": abs(float(attrib.get("ibCommission", 0))),
            "realized_pnl": (
                float(attrib.get("fifoPnlRealized", 0))
                if attrib.get("fifoPnlRealized")
                else None
            ),
            "source": "flex",
        }

        trades.append(trade_dict)

    return trades


def _make_store(account_id: str) -> TradeStore:
    """创建并初始化一个与 ibkr.py 共享存储文件的 TradeStore"""
    store = TradeStore("ibkr_trades", dedup_keys=["exec_id", "trade_id"])
    store.set_account(account_id)
    return store


def load_persisted_trades(account_id: str) -> list:
    """从本地文件加载持久化的交易记录"""
    return _make_store(account_id).load()


async def import_flex_trades(token: str, query_id: str, account_id: str) -> int:
    """导入 Flex 交易记录到本地存储

    Args:
        token: Flex Web Service token
        query_id: Flex Query ID
        account_id: IBKR 账户 ID（用于隔离存储文件）

    Returns:
        新导入的交易数量
    """
    # 获取 Flex 报表
    xml_content = await fetch_flex_statement(token, query_id)

    # 解析交易记录
    new_trades = parse_flex_trades(xml_content)
    if not new_trades:
        logger.info("No trades found in Flex statement")
        return 0

    # 通过 TradeStore 持久化（自动去重）
    store = _make_store(account_id)
    imported_count = store.persist(new_trades)

    if imported_count > 0:
        logger.info(f"Imported {imported_count} new trades from Flex statement")

    return imported_count
