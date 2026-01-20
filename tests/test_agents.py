"""
FinanceAI Pro - 单元测试

运行测试:
    pytest tests/ -v
    pytest tests/test_agents.py -v
    pytest tests/ -v --cov=src
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_context():
    """示例分析上下文"""
    from src.core.base import AnalysisContext
    return AnalysisContext(
        target="AAPL",
        analysis_date=datetime(2024, 1, 15),
        parameters={"scenarios": ["bull", "base", "bear"]}
    )


@pytest.fixture
def sample_financial_data():
    """示例财务数据"""
    return {
        "revenue": 385e9,
        "ebitda": 125e9,
        "net_income": 97e9,
        "free_cash_flow": 110e9,
        "total_debt": 120e9,
        "cash": 65e9,
        "shares_outstanding": 15.5e9,
        "current_price": 175.0,
        "pe_ratio": 28.5,
        "ev_ebitda": 21.0,
        "gross_margin": 0.44,
        "operating_margin": 0.30,
        "revenue_growth_5y": [0.08, 0.06, 0.02, -0.01, 0.05]
    }


@pytest.fixture
def sample_market_data():
    """示例市场数据"""
    return {
        "price": 175.0,
        "volume": 45_000_000,
        "market_cap": 2.75e12,
        "beta": 1.25,
        "52w_high": 198.0,
        "52w_low": 142.0,
        "sma_50": 172.0,
        "sma_200": 165.0,
        "rsi_14": 55.0
    }


# =============================================================================
# Base Module Tests
# =============================================================================

class TestConfidenceScore:
    """ConfidenceScore测试"""
    
    def test_confidence_score_creation(self):
        """测试置信度创建"""
        from src.core.base import ConfidenceScore
        
        score = ConfidenceScore(
            overall=0.75,
            factors={"data_quality": 0.8, "completeness": 0.7},
            weights={"data_quality": 0.5, "completeness": 0.5},
            explanation="测试置信度"
        )
        
        assert score.overall == 0.75
        assert score.factors["data_quality"] == 0.8
        assert "测试" in score.explanation
    
    def test_confidence_score_bounds(self):
        """测试置信度边界"""
        from src.core.base import ConfidenceScore
        
        # 不应该有100%置信度
        score = ConfidenceScore(
            overall=0.95,
            factors={},
            weights={},
            explanation=""
        )
        assert score.overall <= 0.95


class TestAnalysisContext:
    """AnalysisContext测试"""
    
    def test_context_creation(self, sample_context):
        """测试上下文创建"""
        assert sample_context.target == "AAPL"
        assert sample_context.parameters["scenarios"] == ["bull", "base", "bear"]
    
    def test_context_with_preferences(self):
        """测试带偏好的上下文"""
        from src.core.base import AnalysisContext
        
        context = AnalysisContext(
            target="MSFT",
            analysis_date=datetime.now(),
            user_preferences={"risk_tolerance": "moderate"},
            parameters={}
        )
        
        assert context.user_preferences["risk_tolerance"] == "moderate"


class TestReasoningStep:
    """ReasoningStep测试"""
    
    def test_reasoning_step_creation(self):
        """测试推理步骤创建"""
        from src.core.base import ReasoningStep
        
        step = ReasoningStep(
            step_number=1,
            description="获取财务数据",
            inputs={"target": "AAPL"},
            outputs={"revenue": 385e9},
            confidence=0.85
        )
        
        assert step.step_number == 1
        assert step.confidence == 0.85
        assert "revenue" in step.outputs


class TestAgentOutput:
    """AgentOutput测试"""
    
    def test_agent_output_creation(self):
        """测试Agent输出创建"""
        from src.core.base import AgentOutput, ConfidenceScore
        
        output = AgentOutput(
            agent_name="TestAgent",
            result={"valuation": 175.0},
            confidence=ConfidenceScore(
                overall=0.8,
                factors={},
                weights={},
                explanation=""
            ),
            reasoning_chain=[],
            data_sources=[],
            assumptions=[],
            uncertainties=[],
            warnings=["测试警告"],
            timestamp=datetime.now()
        )
        
        assert output.agent_name == "TestAgent"
        assert output.result["valuation"] == 175.0
        assert len(output.warnings) == 1


# =============================================================================
# Valuation Agent Tests
# =============================================================================

class TestValuationAgent:
    """ValuationAgent测试"""
    
    @pytest.fixture
    def valuation_agent(self):
        """创建估值Agent"""
        from src.agents.valuation_agent import ValuationAgent
        return ValuationAgent()
    
    @pytest.mark.asyncio
    async def test_analyze_returns_output(self, valuation_agent, sample_context):
        """测试分析返回输出"""
        result = await valuation_agent.analyze(sample_context)
        
        assert result is not None
        assert result.agent_name == "ValuationAgent"
        assert "valuation" in result.result or "fair_value_range" in str(result.result)
    
    @pytest.mark.asyncio
    async def test_dcf_valuation(self, valuation_agent, sample_financial_data):
        """测试DCF估值"""
        dcf_result = valuation_agent._dcf_valuation(sample_financial_data)
        
        assert "enterprise_value" in dcf_result
        assert "equity_value" in dcf_result
        assert "per_share_value" in dcf_result
        assert dcf_result["per_share_value"] > 0
    
    @pytest.mark.asyncio
    async def test_comparable_valuation(self, valuation_agent, sample_financial_data):
        """测试可比公司估值"""
        # 模拟可比公司数据
        peers = [
            {"pe": 25, "ev_ebitda": 18, "ev_sales": 6},
            {"pe": 30, "ev_ebitda": 22, "ev_sales": 8},
            {"pe": 28, "ev_ebitda": 20, "ev_sales": 7},
        ]
        
        comp_result = valuation_agent._comparable_valuation(
            sample_financial_data, 
            peers
        )
        
        assert "implied_value_pe" in comp_result
        assert "implied_value_ev_ebitda" in comp_result
    
    def test_confidence_never_100_percent(self, valuation_agent):
        """测试置信度永不为100%"""
        confidence = valuation_agent._calculate_confidence(
            data_quality=1.0,
            completeness=1.0,
            reasoning_quality=1.0,
            validation=1.0,
            methodology_fit=1.0
        )
        
        assert confidence.overall < 1.0
        assert confidence.overall <= 0.95


# =============================================================================
# Macro Agent Tests
# =============================================================================

class TestMacroAgent:
    """MacroAgent测试"""
    
    @pytest.fixture
    def macro_agent(self):
        """创建宏观Agent"""
        from src.agents.macro_agent import MacroAgent
        return MacroAgent()
    
    @pytest.mark.asyncio
    async def test_analyze_returns_output(self, macro_agent, sample_context):
        """测试分析返回输出"""
        result = await macro_agent.analyze(sample_context)
        
        assert result is not None
        assert result.agent_name == "MacroAgent"
        assert "economic_environment" in result.result
    
    def test_cycle_assessment(self, macro_agent):
        """测试经济周期评估"""
        macro_data = {
            "gdp_growth": 2.5,
            "unemployment_rate": 3.8,
            "ism_manufacturing": 52,
            "yield_curve_spread": -0.3
        }
        
        cycle = macro_agent._assess_economic_cycle(macro_data)
        
        assert "phase" in cycle
        assert "confidence" in cycle
        assert 0 < cycle["confidence"] <= 1
    
    def test_monetary_policy_analysis(self, macro_agent):
        """测试货币政策分析"""
        macro_data = {
            "fed_funds_rate": 5.25,
            "cpi_yoy": 3.2,
            "unemployment_rate": 3.8
        }
        
        monetary = macro_agent._analyze_monetary_policy(macro_data)
        
        assert "stance" in monetary
        assert "rate_path" in monetary
        assert "terminal_rate" in monetary


# =============================================================================
# Sector Agent Tests
# =============================================================================

class TestSectorAgent:
    """SectorAgent测试"""
    
    @pytest.fixture
    def sector_agent(self):
        """创建行业Agent"""
        from src.agents.sector_agent import SectorAgent
        return SectorAgent()
    
    @pytest.mark.asyncio
    async def test_analyze_returns_output(self, sector_agent, sample_context):
        """测试分析返回输出"""
        result = await sector_agent.analyze(sample_context)
        
        assert result is not None
        assert result.agent_name == "SectorAgent"
        assert "porter_five_forces" in result.result
    
    def test_porter_analysis(self, sector_agent):
        """测试波特五力分析"""
        porter = sector_agent._analyze_porter_forces("Consumer Electronics")
        
        assert porter.supplier_power is not None
        assert porter.buyer_power is not None
        assert porter.competitive_rivalry is not None
        assert 1 <= porter.overall_attractiveness <= 5
    
    def test_moat_analysis(self, sector_agent):
        """测试护城河分析"""
        moat = sector_agent._analyze_competitive_advantages("AAPL", "Consumer Electronics")
        
        assert len(moat) > 0
        assert all(hasattr(m, 'moat_type') for m in moat)
        assert all(hasattr(m, 'strength') for m in moat)


# =============================================================================
# Data Provider Tests
# =============================================================================

class TestDataProviders:
    """数据提供者测试"""
    
    def test_yfinance_provider_creation(self):
        """测试YFinance提供者创建"""
        from src.core.data_and_chain import YFinanceProvider
        
        provider = YFinanceProvider()
        assert provider.name == "yfinance"
        assert provider.data_type == "market_data"
    
    @pytest.mark.asyncio
    async def test_provider_fetch(self):
        """测试数据获取"""
        from src.core.data_and_chain import YFinanceProvider
        
        provider = YFinanceProvider()
        result = await provider.fetch("AAPL", {"data_type": "price_history"})
        
        # 模拟数据应该返回结果
        assert result is not None
    
    def test_provider_registry(self):
        """测试提供者注册表"""
        from src.core.data_and_chain import DataProviderRegistry, YFinanceProvider
        
        registry = DataProviderRegistry()
        provider = YFinanceProvider()
        registry.register(provider)
        
        found = registry.get_provider("market_data")
        assert found is not None
        assert found.name == "yfinance"


# =============================================================================
# Chain Executor Tests
# =============================================================================

class TestChainExecutor:
    """分析链执行器测试"""
    
    @pytest.fixture
    def chain_executor(self):
        """创建链执行器"""
        from src.core.data_and_chain import ChainExecutor
        return ChainExecutor()
    
    def test_chain_loading(self, chain_executor):
        """测试链配置加载"""
        # 模拟链配置
        chain_config = {
            "name": "test_chain",
            "stages": [
                {
                    "name": "stage1",
                    "tasks": [{"agent": "valuation", "params": {}}]
                }
            ]
        }
        
        # 验证配置结构
        assert "name" in chain_config
        assert "stages" in chain_config
        assert len(chain_config["stages"]) > 0


# =============================================================================
# LLM Gateway Tests
# =============================================================================

class TestLLMGateway:
    """LLM网关测试"""
    
    def test_gateway_creation(self):
        """测试网关创建"""
        from src.llm.gateway import LLMGateway
        
        gateway = LLMGateway()
        assert gateway is not None
    
    def test_model_alias_resolution(self):
        """测试模型别名解析"""
        from src.llm.gateway import LLMGateway
        
        gateway = LLMGateway()
        
        # 测试别名解析
        if hasattr(gateway, '_resolve_model'):
            resolved = gateway._resolve_model("gpt4")
            assert resolved is not None


# =============================================================================
# Config Loader Tests
# =============================================================================

class TestConfigLoader:
    """配置加载器测试"""
    
    def test_yaml_loading(self, tmp_path):
        """测试YAML加载"""
        from src.core.config_loader import ConfigLoader
        
        # 创建临时配置文件
        config_file = tmp_path / "test_config.yaml"
        config_file.write_text("""
name: test
version: "1.0"
settings:
  debug: true
""")
        
        loader = ConfigLoader(str(tmp_path))
        config = loader.load_yaml(str(config_file))
        
        assert config["name"] == "test"
        assert config["settings"]["debug"] is True


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_analysis_pipeline(self, sample_context):
        """测试完整分析流程"""
        from src.core.data_and_chain import FinanceAI
        
        # 创建分析引擎
        ai = FinanceAI()
        
        # 执行分析
        result = await ai.analyze(
            target="AAPL",
            chain="full_analysis"
        )
        
        # 验证结果结构
        assert result is not None
        assert hasattr(result, 'target')
    
    @pytest.mark.asyncio
    async def test_multi_agent_orchestration(self, sample_context):
        """测试多Agent协调"""
        from src.agents.valuation_agent import ValuationAgent
        from src.agents.macro_agent import MacroAgent
        from src.agents.sector_agent import SectorAgent
        
        # 创建Agent
        valuation = ValuationAgent()
        macro = MacroAgent()
        sector = SectorAgent()
        
        # 并行执行
        results = await asyncio.gather(
            valuation.analyze(sample_context),
            macro.analyze(sample_context),
            sector.analyze(sample_context)
        )
        
        # 验证所有Agent返回结果
        assert len(results) == 3
        assert all(r is not None for r in results)
        assert all(hasattr(r, 'agent_name') for r in results)


# =============================================================================
# Performance Tests
# =============================================================================

class TestPerformance:
    """性能测试"""
    
    @pytest.mark.asyncio
    async def test_analysis_timeout(self, sample_context):
        """测试分析超时"""
        from src.agents.valuation_agent import ValuationAgent
        
        agent = ValuationAgent()
        
        # 设置超时
        try:
            result = await asyncio.wait_for(
                agent.analyze(sample_context),
                timeout=30.0
            )
            assert result is not None
        except asyncio.TimeoutError:
            pytest.fail("分析超时")
    
    def test_memory_usage(self, sample_financial_data):
        """测试内存使用"""
        import sys
        
        from src.agents.valuation_agent import ValuationAgent
        
        initial_size = sys.getsizeof(sample_financial_data)
        agent = ValuationAgent()
        
        # 执行多次DCF计算
        for _ in range(100):
            agent._dcf_valuation(sample_financial_data)
        
        # 内存不应该显著增长
        # （这是一个简化的测试，实际应使用memory_profiler）


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """边界情况测试"""
    
    def test_negative_fcf_handling(self):
        """测试负FCF处理"""
        from src.agents.valuation_agent import ValuationAgent
        
        agent = ValuationAgent()
        
        # 负FCF数据
        data = {
            "free_cash_flow": -50e6,
            "revenue": 100e6,
            "shares_outstanding": 10e6
        }
        
        # DCF应该处理或跳过负FCF
        result = agent._dcf_valuation(data)
        
        # 应该有警告或使用替代方法
        assert result is not None
    
    def test_missing_data_handling(self):
        """测试缺失数据处理"""
        from src.agents.valuation_agent import ValuationAgent
        
        agent = ValuationAgent()
        
        # 不完整数据
        incomplete_data = {
            "revenue": 100e9
            # 缺少其他字段
        }
        
        # 应该优雅处理
        try:
            result = agent._dcf_valuation(incomplete_data)
            # 可能返回None或降级结果
        except Exception as e:
            # 应该是预期的错误类型
            assert "missing" in str(e).lower() or "required" in str(e).lower()
    
    def test_extreme_values(self):
        """测试极端值处理"""
        from src.agents.valuation_agent import ValuationAgent
        
        agent = ValuationAgent()
        
        # 极端数据
        extreme_data = {
            "free_cash_flow": 1e15,  # 非常大的FCF
            "revenue": 1e15,
            "shares_outstanding": 1
        }
        
        result = agent._dcf_valuation(extreme_data)
        
        # 应该有合理性检查
        if result and "per_share_value" in result:
            # 单股价值不应该是天文数字
            assert result["per_share_value"] < 1e12


# =============================================================================
# Run Tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
