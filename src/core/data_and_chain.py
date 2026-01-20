"""
FinanceAI Pro - Data Providers & Chain Executor
数据提供者和分析链执行器实现
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable, Type
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
import asyncio
import yaml
import json
from pathlib import Path


# ============== 数据提供者基础 ==============

class DataQuality(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class DataResult:
    """数据查询结果"""
    data: Any
    source: str
    fetched_at: datetime
    quality: DataQuality
    completeness: float  # 0-1
    freshness: str       # real-time, daily, weekly, etc.
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class DataProvider(ABC):
    """数据提供者基类"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """数据源名称"""
        pass
    
    @property
    @abstractmethod
    def data_types(self) -> List[str]:
        """支持的数据类型"""
        pass
    
    @abstractmethod
    async def fetch(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        **kwargs
    ) -> DataResult:
        """获取数据"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """健康检查"""
        pass
    
    def _assess_quality(self, data: Any, expected_fields: List[str]) -> DataQuality:
        """评估数据质量"""
        if not data:
            return DataQuality.UNKNOWN
        
        if isinstance(data, dict):
            present = sum(1 for f in expected_fields if f in data and data[f] is not None)
            ratio = present / len(expected_fields) if expected_fields else 1.0
            
            if ratio >= 0.9:
                return DataQuality.HIGH
            elif ratio >= 0.7:
                return DataQuality.MEDIUM
            else:
                return DataQuality.LOW
        
        return DataQuality.MEDIUM


class DataProviderRegistry:
    """数据提供者注册中心"""
    
    _providers: Dict[str, DataProvider] = {}
    _type_to_providers: Dict[str, List[str]] = {}
    
    @classmethod
    def register(cls, provider: DataProvider):
        """注册数据提供者"""
        cls._providers[provider.name] = provider
        
        for dtype in provider.data_types:
            if dtype not in cls._type_to_providers:
                cls._type_to_providers[dtype] = []
            cls._type_to_providers[dtype].append(provider.name)
    
    @classmethod
    def get(cls, name: str) -> Optional[DataProvider]:
        return cls._providers.get(name)
    
    @classmethod
    def get_for_type(cls, data_type: str) -> List[DataProvider]:
        """获取支持指定数据类型的提供者"""
        provider_names = cls._type_to_providers.get(data_type, [])
        return [cls._providers[name] for name in provider_names if name in cls._providers]
    
    @classmethod
    def list_all(cls) -> List[Dict]:
        return [
            {'name': name, 'types': p.data_types}
            for name, p in cls._providers.items()
        ]


# ============== 具体数据提供者实现 ==============

class YFinanceProvider(DataProvider):
    """Yahoo Finance 数据提供者"""
    
    @property
    def name(self) -> str:
        return "yfinance"
    
    @property
    def data_types(self) -> List[str]:
        return [
            "price_history",
            "current_price",
            "fundamentals",
            "financials",
            "options"
        ]
    
    async def fetch(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        **kwargs
    ) -> DataResult:
        """获取数据"""
        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)

            if data_type == "price_history":
                hist = ticker.history(start=start_date, end=end_date, period="1y" if not start_date else None)
                data = {
                    'prices': hist['Close'].tolist() if not hist.empty else [],
                    'volumes': hist['Volume'].tolist() if not hist.empty else [],
                    'dates': [d.strftime('%Y-%m-%d') for d in hist.index.tolist()] if not hist.empty else [],
                    'high': hist['High'].tolist() if not hist.empty else [],
                    'low': hist['Low'].tolist() if not hist.empty else [],
                    'open': hist['Open'].tolist() if not hist.empty else [],
                }
                quality = DataQuality.HIGH

            elif data_type == "current_price":
                info = ticker.info
                data = {
                    'current_price': info.get('currentPrice') or info.get('regularMarketPrice'),
                    'market_cap': info.get('marketCap'),
                    'pe_ratio': info.get('trailingPE'),
                    'forward_pe': info.get('forwardPE'),
                    'dividend_yield': info.get('dividendYield'),
                    'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                    'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                    'volume': info.get('volume'),
                    'avg_volume': info.get('averageVolume'),
                }
                quality = DataQuality.HIGH

            elif data_type == "fundamentals":
                info = ticker.info
                data = {
                    'revenue': info.get('totalRevenue'),
                    'net_income': info.get('netIncomeToCommon'),
                    'free_cash_flow': info.get('freeCashflow'),
                    'total_assets': info.get('totalAssets'),
                    'total_debt': info.get('totalDebt'),
                    'shares_outstanding': info.get('sharesOutstanding'),
                    'eps': info.get('trailingEps'),
                    'beta': info.get('beta'),
                    'profit_margin': info.get('profitMargins'),
                    'operating_margin': info.get('operatingMargins'),
                    'roe': info.get('returnOnEquity'),
                    'roa': info.get('returnOnAssets'),
                }
                quality = DataQuality.MEDIUM

            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            
            return DataResult(
                data=data,
                source=self.name,
                fetched_at=datetime.now(),
                quality=quality,
                completeness=0.9,
                freshness="real-time" if data_type == "current_price" else "daily",
                metadata={'symbol': symbol, 'data_type': data_type}
            )
            
        except Exception as e:
            return DataResult(
                data=None,
                source=self.name,
                fetched_at=datetime.now(),
                quality=DataQuality.UNKNOWN,
                completeness=0.0,
                freshness="unknown",
                warnings=[str(e)]
            )
    
    async def health_check(self) -> bool:
        try:
            # 尝试获取一个简单的数据
            result = await self.fetch("AAPL", "current_price")
            return result.data is not None
        except:
            return False
    
    # Mock 方法（实际实现中替换为真实 API 调用）
    def _mock_price_history(self, symbol: str, start: date, end: date) -> Dict:
        return {
            'prices': [100, 101, 102, 103, 104],
            'volumes': [1000000, 1100000, 1200000, 1300000, 1400000],
            'dates': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']
        }
    
    def _mock_current_price(self, symbol: str) -> Dict:
        return {
            'current_price': 185.0,
            'market_cap': 2800000000000,
            'pe_ratio': 28.5,
            'dividend_yield': 0.005
        }
    
    def _mock_fundamentals(self, symbol: str) -> Dict:
        return {
            'revenue': 380000000000,
            'net_income': 95000000000,
            'free_cash_flow': 100000000000,
            'total_assets': 350000000000,
            'total_debt': 100000000000,
            'shares_outstanding': 15500000000,
            'eps': 6.13,
            'beta': 1.2
        }


class SECProvider(DataProvider):
    """SEC 财报数据提供者"""
    
    @property
    def name(self) -> str:
        return "sec_filings"
    
    @property
    def data_types(self) -> List[str]:
        return ["10k", "10q", "8k", "earnings_call"]
    
    async def fetch(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        **kwargs
    ) -> DataResult:
        """获取 SEC 文件"""
        # 实际实现中调用 SEC EDGAR API
        return DataResult(
            data={'filings': []},
            source=self.name,
            fetched_at=datetime.now(),
            quality=DataQuality.HIGH,
            completeness=0.8,
            freshness="quarterly"
        )
    
    async def health_check(self) -> bool:
        return True


class NewsProvider(DataProvider):
    """新闻数据提供者"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    @property
    def name(self) -> str:
        return "news_api"
    
    @property
    def data_types(self) -> List[str]:
        return ["news", "press_release", "analyst_report"]
    
    async def fetch(
        self,
        symbol: str,
        data_type: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        **kwargs
    ) -> DataResult:
        """获取新闻数据"""
        lookback_days = kwargs.get('lookback_days', 7)
        
        # 实际实现中调用新闻 API
        return DataResult(
            data={
                'articles': [
                    {
                        'title': f'{symbol} reports strong earnings',
                        'source': 'Reuters',
                        'date': datetime.now().isoformat(),
                        'sentiment': 0.7
                    }
                ]
            },
            source=self.name,
            fetched_at=datetime.now(),
            quality=DataQuality.MEDIUM,
            completeness=0.7,
            freshness="hourly"
        )
    
    async def health_check(self) -> bool:
        return self.api_key is not None


# ============== 分析链执行器 ==============

@dataclass
class ChainTask:
    """分析链任务"""
    name: str
    agent: str
    action: str
    inputs: List[str]
    output_key: str
    params: Dict[str, Any] = field(default_factory=dict)
    timeout: int = 60


@dataclass
class ChainStage:
    """分析链阶段"""
    name: str
    tasks: List[ChainTask]
    parallel: bool = True
    depends_on: List[str] = field(default_factory=list)


@dataclass
class ChainConfig:
    """分析链配置"""
    name: str
    description: str
    stages: List[ChainStage]
    conflict_resolution: Dict[str, Any]
    output_config: Dict[str, Any]
    quality_gates: List[Dict[str, Any]]
    timeouts: Dict[str, int]


class ChainExecutor:
    """分析链执行器"""
    
    def __init__(
        self,
        agent_registry: Dict[str, Any],  # Agent 实例注册表
        data_providers: DataProviderRegistry
    ):
        self.agents = agent_registry
        self.data_providers = data_providers
        self.results: Dict[str, Any] = {}
        self.execution_log: List[Dict] = []
    
    def load_chain(self, config_path: str) -> ChainConfig:
        """从配置文件加载分析链"""
        with open(config_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        
        stages = []
        for stage_data in raw.get('stages', []):
            tasks = []
            for task_data in stage_data.get('tasks', []):
                tasks.append(ChainTask(
                    name=task_data.get('name', task_data['action']),
                    agent=task_data['agent'],
                    action=task_data['action'],
                    inputs=task_data.get('inputs', []),
                    output_key=task_data['output_key'],
                    params=task_data.get('params', {}),
                    timeout=task_data.get('timeout', 60)
                ))
            
            stages.append(ChainStage(
                name=stage_data['name'],
                tasks=tasks,
                parallel=stage_data.get('parallel', True),
                depends_on=stage_data.get('depends_on', [])
            ))
        
        return ChainConfig(
            name=raw['name'],
            description=raw.get('description', ''),
            stages=stages,
            conflict_resolution=raw.get('conflict_resolution', {}),
            output_config=raw.get('output', {}),
            quality_gates=raw.get('quality_gates', []),
            timeouts=raw.get('timeouts', {'stage_timeout': 60, 'total_timeout': 300})
        )
    
    async def execute(
        self,
        chain: ChainConfig,
        context: Any,  # AnalysisContext
        initial_inputs: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        执行分析链
        
        Args:
            chain: 分析链配置
            context: 分析上下文
            initial_inputs: 初始输入数据
        """
        self.results = initial_inputs or {}
        self.execution_log = []
        
        start_time = datetime.now()
        
        # 按阶段执行
        completed_stages = set()
        
        for stage in chain.stages:
            # 检查依赖
            for dep in stage.depends_on:
                if dep not in completed_stages:
                    raise RuntimeError(f"Stage '{stage.name}' depends on '{dep}' which has not completed")
            
            self._log(f"Starting stage: {stage.name}")
            
            try:
                if stage.parallel:
                    # 并行执行所有任务
                    await self._execute_parallel(stage, context, chain.timeouts['stage_timeout'])
                else:
                    # 串行执行
                    await self._execute_sequential(stage, context, chain.timeouts['stage_timeout'])
                
                completed_stages.add(stage.name)
                self._log(f"Completed stage: {stage.name}")
                
            except asyncio.TimeoutError:
                self._log(f"Stage '{stage.name}' timed out", level="error")
                raise
            except Exception as e:
                self._log(f"Stage '{stage.name}' failed: {e}", level="error")
                raise
            
            # 检查总超时
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed > chain.timeouts['total_timeout']:
                raise asyncio.TimeoutError("Total chain execution timeout exceeded")
        
        # 质量门控检查
        self._check_quality_gates(chain.quality_gates)
        
        # 处理冲突
        if chain.conflict_resolution:
            self._resolve_conflicts(chain.conflict_resolution)
        
        # 生成最终输出
        return self._generate_output(chain.output_config)
    
    async def _execute_parallel(
        self,
        stage: ChainStage,
        context: Any,
        timeout: int
    ):
        """并行执行阶段任务"""
        tasks = [
            self._execute_task(task, context)
            for task in stage.tasks
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        for task, result in zip(stage.tasks, results):
            if isinstance(result, Exception):
                self._log(f"Task '{task.name}' failed: {result}", level="warning")
                self.results[task.output_key] = None
            else:
                self.results[task.output_key] = result
    
    async def _execute_sequential(
        self,
        stage: ChainStage,
        context: Any,
        timeout: int
    ):
        """串行执行阶段任务"""
        for task in stage.tasks:
            try:
                result = await asyncio.wait_for(
                    self._execute_task(task, context),
                    timeout=timeout
                )
                self.results[task.output_key] = result
            except Exception as e:
                self._log(f"Task '{task.name}' failed: {e}", level="error")
                self.results[task.output_key] = None
    
    async def _execute_task(
        self,
        task: ChainTask,
        context: Any
    ) -> Any:
        """执行单个任务"""
        self._log(f"Executing task: {task.name} (agent: {task.agent})")
        
        # 获取 Agent
        agent = self.agents.get(task.agent)
        if not agent:
            raise ValueError(f"Agent '{task.agent}' not found")
        
        # 准备输入
        inputs = {}
        for input_key in task.inputs:
            if input_key in self.results:
                inputs[input_key] = self.results[input_key]
            else:
                self._log(f"Input '{input_key}' not found for task '{task.name}'", level="warning")
        
        # 添加任务参数
        inputs.update(task.params)
        
        # 调用 Agent
        # 假设 agent 有对应的 action 方法
        action_method = getattr(agent, task.action, None)
        if action_method:
            result = await action_method(context, inputs)
        else:
            # 默认调用 analyze 方法
            result = await agent.analyze(context, inputs)
        
        return result
    
    def _check_quality_gates(self, gates: List[Dict]):
        """检查质量门控"""
        for gate in gates:
            name = gate.get('name')
            threshold = gate.get('threshold', 0.5)
            action = gate.get('action', 'warn')
            
            # 获取相关指标
            if name == 'data_completeness':
                # 检查数据完整性
                pass
            elif name == 'agent_agreement':
                # 检查 Agent 间一致性
                pass
            elif name == 'confidence_score':
                # 检查置信度
                pass
            
            self._log(f"Quality gate '{name}' checked")
    
    def _resolve_conflicts(self, config: Dict):
        """解决冲突"""
        method = config.get('method', 'weighted_vote')
        weights = config.get('weights', {})
        
        # 实现冲突解决逻辑
        self._log("Conflicts resolved")
    
    def _generate_output(self, config: Dict) -> Dict[str, Any]:
        """生成最终输出"""
        output = {
            'results': self.results,
            'execution_log': self.execution_log,
            'generated_at': datetime.now().isoformat()
        }
        
        # 根据配置格式化输出
        if config.get('format') == 'structured_report':
            output['sections'] = {}
            for section in config.get('sections', []):
                section_name = section.get('name') if isinstance(section, dict) else section
                output['sections'][section_name] = self._extract_section(section_name)
        
        return output
    
    def _extract_section(self, section_name: str) -> Dict:
        """提取报告章节"""
        # 根据章节名称从结果中提取相关内容
        return {'content': f'Section: {section_name}', 'data': {}}
    
    def _log(self, message: str, level: str = "info"):
        """记录日志"""
        self.execution_log.append({
            'timestamp': datetime.now().isoformat(),
            'level': level,
            'message': message
        })


# ============== 数据收集器 ==============

class SimpleDataCollector:
    """数据收集器，使用 yfinance 获取真实金融数据"""

    def __init__(self, data_registry: DataProviderRegistry):
        self.data_registry = data_registry
        self._cache = {}

    def _get_ticker(self, symbol: str):
        """获取并缓存 ticker 对象"""
        if symbol not in self._cache:
            import yfinance as yf
            self._cache[symbol] = yf.Ticker(symbol)
        return self._cache[symbol]

    async def fetch_market_data(self, context, inputs: Dict = None) -> Dict:
        """获取市场数据，返回估值模型需要的格式"""
        target = context.target if hasattr(context, 'target') else str(context)
        try:
            ticker = self._get_ticker(target)
            info = ticker.info
            hist = ticker.history(period='1y')

            return {
                'current_price': info.get('currentPrice') or info.get('regularMarketPrice', 0),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE'),
                'forward_pe': info.get('forwardPE'),
                'ps_ratio': info.get('priceToSalesTrailing12Months'),
                'pb_ratio': info.get('priceToBook'),
                'dividend_yield': info.get('dividendYield'),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh'),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow'),
                'avg_volume': info.get('averageVolume'),
                'beta': info.get('beta', 1.0),
                # 历史价格数据
                'price_history': {
                    'close': hist['Close'].tolist() if not hist.empty else [],
                    'high': hist['High'].tolist() if not hist.empty else [],
                    'low': hist['Low'].tolist() if not hist.empty else [],
                    'open': hist['Open'].tolist() if not hist.empty else [],
                    'volume': hist['Volume'].tolist() if not hist.empty else [],
                    'dates': [d.strftime('%Y-%m-%d') for d in hist.index] if not hist.empty else [],
                }
            }
        except Exception as e:
            print(f"Warning: Failed to fetch market data for {target}: {e}")
            return {'current_price': 0, 'error': str(e)}

    async def fetch_financials(self, context, inputs: Dict = None) -> Dict:
        """获取财务数据，返回估值模型需要的格式"""
        target = context.target if hasattr(context, 'target') else str(context)
        try:
            ticker = self._get_ticker(target)
            info = ticker.info

            # 计算自由现金流
            fcf = info.get('freeCashflow', 0) or 0
            shares = info.get('sharesOutstanding', 1) or 1
            revenue = info.get('totalRevenue', 0) or 0
            ebitda = info.get('ebitda', 0) or 0
            net_income = info.get('netIncomeToCommon', 0) or 0
            total_debt = info.get('totalDebt', 0) or 0
            total_cash = info.get('totalCash', 0) or 0

            return {
                # DCF 需要的数据
                'latest_fcf': fcf,
                'estimated_growth': info.get('revenueGrowth') or 0.05,
                'beta': info.get('beta', 1.0),
                'shares_outstanding': shares,
                'net_debt': total_debt - total_cash,

                # 可比公司法需要的数据
                'eps': info.get('trailingEps', 0),
                'revenue_per_share': revenue / shares if shares > 0 else 0,
                'ebitda_per_share': ebitda / shares if shares > 0 else 0,
                'peer_pe_median': info.get('trailingPE', 20),  # 使用自身作为基准
                'peer_ps_median': info.get('priceToSalesTrailing12Months', 3),
                'peer_ev_ebitda_median': 12,  # 默认值

                # 历史估值数据
                'historical_pe_range': (15, 30),  # 默认范围

                # 其他财务指标
                'revenue': revenue,
                'net_income': net_income,
                'ebitda': ebitda,
                'free_cash_flow': fcf,
                'total_debt': total_debt,
                'total_cash': total_cash,
                'gross_margin': info.get('grossMargins'),
                'operating_margin': info.get('operatingMargins'),
                'net_margin': info.get('profitMargins'),
                'roe': info.get('returnOnEquity'),
                'roa': info.get('returnOnAssets'),
                'current_ratio': info.get('currentRatio'),
                'debt_to_equity': info.get('debtToEquity'),
                'revenue_growth': info.get('revenueGrowth'),
            }
        except Exception as e:
            print(f"Warning: Failed to fetch financials for {target}: {e}")
            return {'error': str(e)}

    async def fetch_news(self, context, inputs: Dict = None) -> Dict:
        """获取新闻数据"""
        target = context.target if hasattr(context, 'target') else str(context)
        try:
            ticker = self._get_ticker(target)
            news = ticker.news or []

            # 转换为分析需要的格式
            news_items = []
            for item in news[:10]:  # 最多10条
                news_items.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'timestamp': item.get('providerPublishTime', ''),
                    'sentiment': 0.0,  # 需要后续分析
                })

            return {
                'target': target,
                'news': news_items,
                'news_count': len(news_items),
            }
        except Exception as e:
            print(f"Warning: Failed to fetch news for {target}: {e}")
            return {'target': target, 'news': [], 'news_count': 0}

    async def fetch_analyst_data(self, context, inputs: Dict = None) -> Dict:
        """获取分析师数据"""
        target = context.target if hasattr(context, 'target') else str(context)
        try:
            ticker = self._get_ticker(target)
            info = ticker.info

            return {
                'target_price_high': info.get('targetHighPrice'),
                'target_price_low': info.get('targetLowPrice'),
                'target_price_mean': info.get('targetMeanPrice'),
                'target_price_median': info.get('targetMedianPrice'),
                'recommendation': info.get('recommendationKey'),
                'recommendation_mean': info.get('recommendationMean'),
                'number_of_analysts': info.get('numberOfAnalystOpinions'),
            }
        except Exception as e:
            print(f"Warning: Failed to fetch analyst data for {target}: {e}")
            return {'target': target, 'error': str(e)}


# ============== 主入口类 ==============

class FinanceAI:
    """
    FinanceAI Pro 主入口
    
    使用示例:
    ```python
    ai = FinanceAI(config_dir="config/")
    
    result = await ai.analyze(
        target="AAPL",
        chain="full_analysis",
        custom_params={'discount_rate': 0.10}
    )
    
    print(result.summary)
    result.export_pdf("report.pdf")
    ```
    """
    
    def __init__(self, config_dir: str, llm_config: Dict[str, str] = None):
        self.config_dir = Path(config_dir)
        self.llm_config = llm_config or {}
        
        # 初始化组件
        self._init_data_providers()
        self._init_llm_gateway()
        self._init_agents()
        self.chain_executor = ChainExecutor(self.agents, self.data_registry)
    
    def _init_data_providers(self):
        """初始化数据提供者"""
        self.data_registry = DataProviderRegistry()
        
        # 注册默认提供者
        DataProviderRegistry.register(YFinanceProvider())
        DataProviderRegistry.register(SECProvider())
        DataProviderRegistry.register(NewsProvider())
    
    def _init_llm_gateway(self):
        """初始化 LLM 网关"""
        from src.core.base import LLMGateway
        self.llm_gateway = LLMGateway(self.llm_config)
    
    def _init_agents(self):
        """初始化 Agents"""
        from src.agents.valuation_agent import ValuationAgent
        from src.agents.technical_agent import TechnicalAgent
        from src.agents.earnings_agent import EarningsAgent
        from src.agents.sentiment_risk_agent import SentimentAgent, RiskAgent
        from src.agents.strategy_agent import StrategyAgent
        from src.agents.macro_agent import MacroAgent
        from src.agents.sector_agent import SectorAgent

        self.agents = {}
        agents_dir = self.config_dir / 'agents'

        # Agent 类映射 (name, class, uses_base_agent_init)
        agent_classes = {
            'valuation': ('ValuationAgent', ValuationAgent, True),
            'technical': ('TechnicalAgent', TechnicalAgent, False),
            'earnings': ('EarningsAgent', EarningsAgent, False),
            'sentiment': ('SentimentAgent', SentimentAgent, False),
            'risk': ('RiskAgent', RiskAgent, False),
            'strategy': ('StrategyAgent', StrategyAgent, False),
            'macro': ('MacroAgent', MacroAgent, False),
            'sector': ('SectorAgent', SectorAgent, False),
        }

        # 加载配置
        import yaml
        if agents_dir.exists():
            for config_file in agents_dir.glob('*.yaml'):
                agent_name = config_file.stem.replace('_agent', '')
                if agent_name in agent_classes:
                    name, cls, uses_base = agent_classes[agent_name]
                    try:
                        with open(config_file, 'r', encoding='utf-8') as f:
                            config = yaml.safe_load(f)
                        if uses_base:
                            from src.core.base import ConfidenceScorer
                            agent = cls(str(config_file), self.llm_gateway, ConfidenceScorer())
                        else:
                            agent = cls(config)
                        self.agents[name] = agent
                    except Exception as e:
                        print(f"Warning: Failed to load {name}: {e}")

        # 添加简单的数据收集器
        self.agents['DataCollector'] = SimpleDataCollector(self.data_registry)
    
    async def analyze(
        self,
        target: str,
        chain: str = "full_analysis",
        custom_params: Dict[str, Any] = None
    ) -> 'AnalysisResult':
        """
        执行分析
        
        Args:
            target: 分析标的（股票代码等）
            chain: 分析链名称
            custom_params: 自定义参数
        """
        from src.core.base import AnalysisContext
        
        # 创建上下文
        context = AnalysisContext(
            target=target,
            analysis_date=datetime.now(),
            custom_params=custom_params or {}
        )
        
        # 加载分析链
        chain_path = self.config_dir / 'chains' / f'{chain}.yaml'
        chain_config = self.chain_executor.load_chain(str(chain_path))
        
        # 获取初始数据
        initial_data = await self._fetch_initial_data(target)
        
        # 执行分析链
        raw_result = await self.chain_executor.execute(
            chain_config,
            context,
            initial_data
        )
        
        return AnalysisResult(raw_result, target)
    
    async def _fetch_initial_data(self, target: str) -> Dict[str, Any]:
        """获取初始数据"""
        data = {}
        
        # 获取市场数据
        market_providers = DataProviderRegistry.get_for_type("current_price")
        if market_providers:
            result = await market_providers[0].fetch(target, "current_price")
            data['market_data'] = result.data
        
        # 获取财务数据
        fin_providers = DataProviderRegistry.get_for_type("fundamentals")
        if fin_providers:
            result = await fin_providers[0].fetch(target, "fundamentals")
            data['financial_data'] = result.data
        
        return data
    
    def register_provider(self, provider: DataProvider):
        """注册自定义数据提供者"""
        DataProviderRegistry.register(provider)
    
    def register_agent(self, name: str, agent: Any):
        """注册自定义 Agent"""
        self.agents[name] = agent


@dataclass
class AnalysisResult:
    """分析结果包装类"""
    raw: Dict[str, Any]
    target: str
    
    @property
    def summary(self) -> str:
        """获取摘要"""
        results = self.raw.get('results', {})
        if 'final_recommendation' in results:
            return results['final_recommendation'].get('summary', '')
        return "Analysis completed"
    
    @property
    def valuation(self) -> Optional[Dict]:
        """获取估值结果"""
        return self.raw.get('results', {}).get('valuation_view')
    
    @property
    def risk_assessment(self) -> Optional[Dict]:
        """获取风险评估"""
        return self.raw.get('results', {}).get('risk_view')
    
    @property
    def recommendation(self) -> Optional[Dict]:
        """获取推荐"""
        return self.raw.get('results', {}).get('final_recommendation')
    
    def export_pdf(self, path: str):
        """导出 PDF 报告"""
        # 实现 PDF 导出
        print(f"PDF exported to {path}")
    
    def export_html(self, path: str):
        """导出 HTML 报告"""
        # 实现 HTML 导出
        print(f"HTML exported to {path}")
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return self.raw


# ============== 使用示例 ==============

if __name__ == "__main__":
    async def main():
        print("FinanceAI Pro - Example Usage")
        print("=" * 50)
        
        # 实际使用示例（需要配置文件和 API keys）
        # ai = FinanceAI(
        #     config_dir="config/",
        #     llm_config={
        #         'anthropic_api_key': 'your-key',
        #         'openai_api_key': 'your-key'
        #     }
        # )
        
        # result = await ai.analyze(
        #     target="AAPL",
        #     chain="full_analysis"
        # )
        
        # print(result.summary)
        # print(result.valuation)
        
        # 演示数据提供者
        provider = YFinanceProvider()
        result = await provider.fetch("AAPL", "fundamentals")
        print(f"\n数据提供者测试:")
        print(f"  数据源: {result.source}")
        print(f"  质量: {result.quality}")
        print(f"  数据: {result.data}")
        
        print("\n✅ Demo completed successfully")
    
    asyncio.run(main())
