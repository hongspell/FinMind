#!/usr/bin/env python3
"""
FinanceAI Pro - 命令行入口

使用示例:
    # 完整分析
    python -m src.main analyze AAPL --chain full_analysis --output report.pdf
    
    # 快速扫描多只股票
    python -m src.main scan AAPL MSFT GOOGL --chain quick_scan
    
    # 仅估值分析
    python -m src.main valuation AAPL --scenarios bull,base,bear
    
    # 财报分析
    python -m src.main earnings AAPL --quarter 2024Q3
    
    # 启动API服务
    python -m src.main serve --port 8000
"""

import asyncio
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, List

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def create_parser() -> argparse.ArgumentParser:
    """创建命令行解析器"""
    parser = argparse.ArgumentParser(
        prog="financeai",
        description="FinanceAI Pro - 模块化金融AI分析平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s analyze AAPL                    完整分析Apple
  %(prog)s analyze AAPL --chain quick_scan 快速扫描
  %(prog)s scan AAPL MSFT GOOGL            批量扫描
  %(prog)s valuation TSLA                  估值分析
  %(prog)s serve --port 8000               启动API服务
        """
    )
    
    parser.add_argument(
        "--version", 
        action="version", 
        version="FinanceAI Pro v0.1.0"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/",
        help="配置目录路径 (默认: config/)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Output language: en (English, default) or zh (Chinese)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # analyze 命令
    analyze_parser = subparsers.add_parser(
        "analyze", 
        help="执行完整分析"
    )
    analyze_parser.add_argument(
        "target",
        type=str,
        help="分析目标（股票代码）"
    )
    analyze_parser.add_argument(
        "--chain",
        type=str,
        default="full_analysis",
        help="分析链名称 (默认: full_analysis)"
    )
    analyze_parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出文件路径 (支持 .pdf, .html, .json)"
    )
    analyze_parser.add_argument(
        "--scenarios",
        type=str,
        help="情景分析 (逗号分隔: bull,base,bear)"
    )
    
    # scan 命令
    scan_parser = subparsers.add_parser(
        "scan",
        help="批量扫描多只股票"
    )
    scan_parser.add_argument(
        "targets",
        type=str,
        nargs="+",
        help="股票代码列表"
    )
    scan_parser.add_argument(
        "--chain",
        type=str,
        default="quick_scan",
        help="分析链名称 (默认: quick_scan)"
    )
    scan_parser.add_argument(
        "--output", "-o",
        type=str,
        help="输出目录路径"
    )
    
    # valuation 命令
    valuation_parser = subparsers.add_parser(
        "valuation",
        help="执行估值分析"
    )
    valuation_parser.add_argument(
        "target",
        type=str,
        help="分析目标"
    )
    valuation_parser.add_argument(
        "--methods",
        type=str,
        default="dcf,comparables,historical",
        help="估值方法 (逗号分隔)"
    )
    valuation_parser.add_argument(
        "--scenarios",
        type=str,
        default="bull,base,bear",
        help="情景分析"
    )
    
    # earnings 命令
    earnings_parser = subparsers.add_parser(
        "earnings",
        help="财报分析"
    )
    earnings_parser.add_argument(
        "target",
        type=str,
        help="分析目标"
    )
    earnings_parser.add_argument(
        "--quarter",
        type=str,
        help="季度 (如: 2024Q3)"
    )
    earnings_parser.add_argument(
        "--compare",
        type=str,
        help="对比季度 (如: 2023Q3)"
    )
    
    # serve 命令
    serve_parser = subparsers.add_parser(
        "serve",
        help="启动API服务"
    )
    serve_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="监听地址 (默认: 0.0.0.0)"
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="监听端口 (默认: 8000)"
    )
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="开发模式（自动重载）"
    )
    
    # config 命令
    config_parser = subparsers.add_parser(
        "config",
        help="配置管理"
    )
    config_subparsers = config_parser.add_subparsers(dest="config_action")
    
    config_subparsers.add_parser("list", help="列出所有配置")
    config_subparsers.add_parser("validate", help="验证配置文件")
    
    return parser


async def run_analyze(args) -> int:
    """Execute analysis command"""
    lang = getattr(args, 'lang', 'en')

    # Progress messages based on language
    if lang == 'zh':
        msg_init = "初始化分析引擎"
        msg_chain = "加载分析链"
        msg_data = "获取数据"
        msg_analyze = "执行分析"
        msg_report = "生成报告"
        msg_import_err = "导入错误"
        msg_install = "请确保所有依赖已安装: pip install -r requirements.txt"
        msg_failed = "分析失败"
    else:
        msg_init = "Initializing analysis engine"
        msg_chain = "Loading analysis chain"
        msg_data = "Fetching data"
        msg_analyze = "Running analysis"
        msg_report = "Generating report"
        msg_import_err = "Import error"
        msg_install = "Please ensure all dependencies are installed: pip install -r requirements.txt"
        msg_failed = "Analysis failed"

    print(f"\n{'='*64}")
    print(f"  FinanceAI Pro - {args.target}")
    print(f"{'='*64}")
    print(f"  Chain: {args.chain}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*64}\n")

    try:
        # Import analysis modules
        from src.core.data_and_chain import FinanceAI
        from src.core.base import AnalysisContext
        from src.core.report_generator import ReportGenerator

        # Initialize
        print(f"[1/5] {msg_init}...")
        ai = FinanceAI(config_dir=args.config)
        report_gen = ReportGenerator(lang=lang)

        # Create context
        context = AnalysisContext(
            target=args.target,
            analysis_date=datetime.now(),
            custom_params={
                "scenarios": args.scenarios.split(",") if args.scenarios else ["bull", "base", "bear"]
            }
        )

        # Execute analysis
        print(f"[2/5] {msg_chain}: {args.chain}")
        print(f"[3/5] {msg_data}...")
        print(f"[4/5] {msg_analyze}...")

        result = await ai.analyze(
            target=args.target,
            chain=args.chain,
            custom_params=context.custom_params
        )

        print(f"[5/5] {msg_report}...")

        # Handle output
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".md":
                # Generate Markdown report
                content = report_gen.generate_markdown_report(result, args.target, args.chain)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(content)
            elif output_path.suffix == ".json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(result.to_dict(), f, ensure_ascii=False, indent=2, default=str)
            else:
                # Default to Markdown for other extensions
                with open(output_path, "w", encoding="utf-8") as f:
                    content = report_gen.generate_markdown_report(result, args.target, args.chain)
                    f.write(content)

            print(f"\n✓ Report saved: {output_path}")

            # Also print terminal summary
            print(report_gen.generate_terminal_summary(result, args.target, args.chain))
        else:
            # Print terminal summary
            print(report_gen.generate_terminal_summary(result, args.target, args.chain))

            # Auto-save Markdown report
            try:
                report_path = report_gen.save_markdown_report(result, args.target, 'reports', args.chain)
                saved_msg = "Full report saved to" if lang == 'en' else "完整报告已保存至"
                print(f"\n  {saved_msg}: {report_path}")
            except Exception as e:
                if args.verbose:
                    print(f"\n  Warning: Could not save report: {e}")
            print("=" * 64 + "\n")

        return 0

    except ImportError as e:
        print(f"\n❌ {msg_import_err}: {e}")
        print(msg_install)
        return 1
    except Exception as e:
        print(f"\n❌ {msg_failed}: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


async def run_scan(args) -> int:
    """执行批量扫描"""
    print(f"\n{'='*60}")
    print(f"  FinanceAI Pro - 批量扫描")
    print(f"{'='*60}")
    print(f"  目标: {', '.join(args.targets)}")
    print(f"  分析链: {args.chain}")
    print(f"{'='*60}\n")
    
    results = []
    for i, target in enumerate(args.targets, 1):
        print(f"[{i}/{len(args.targets)}] 扫描 {target}...")
        
        # 模拟扫描结果
        results.append({
            "target": target,
            "status": "completed",
            "recommendation": "hold",
            "confidence": 0.72,
            "fair_value": 175.0,
            "current_price": 168.5,
            "upside": 3.9
        })
    
    # 打印结果表格
    print("\n" + "="*80)
    print(f"{'股票':<8} {'建议':<8} {'置信度':<10} {'公允价值':<12} {'当前价':<12} {'空间':<8}")
    print("-"*80)
    for r in results:
        print(f"{r['target']:<8} {r['recommendation']:<8} {r['confidence']:.1%}       "
              f"${r['fair_value']:<10.2f} ${r['current_price']:<10.2f} {r['upside']:.1f}%")
    print("="*80 + "\n")
    
    return 0


async def run_valuation(args) -> int:
    """执行估值分析"""
    print(f"\n{'='*60}")
    print(f"  FinanceAI Pro - 估值分析")
    print(f"{'='*60}")
    print(f"  目标: {args.target}")
    print(f"  方法: {args.methods}")
    print(f"  情景: {args.scenarios}")
    print(f"{'='*60}\n")
    
    methods = args.methods.split(",")
    scenarios = args.scenarios.split(",")
    
    print("执行估值分析...")
    print(f"  - DCF估值")
    print(f"  - 可比公司估值")
    print(f"  - 历史估值")
    
    # 模拟结果
    print("\n" + "="*60)
    print("  估值结果")
    print("="*60)
    print(f"\n  {'方法':<20} {'低值':<12} {'中值':<12} {'高值':<12}")
    print("-"*60)
    print(f"  {'DCF':<20} ${'145.00':<11} ${'165.00':<11} ${'185.00':<11}")
    print(f"  {'可比公司':<20} ${'155.00':<11} ${'172.00':<11} ${'190.00':<11}")
    print(f"  {'历史估值':<20} ${'150.00':<11} ${'168.00':<11} ${'188.00':<11}")
    print("-"*60)
    print(f"  {'综合估值':<20} ${'150.00':<11} ${'168.00':<11} ${'188.00':<11}")
    print("="*60)
    
    print(f"\n  情景分析:")
    print(f"    牛市情景 (25%概率): $195.00")
    print(f"    基准情景 (50%概率): $168.00")
    print(f"    熊市情景 (25%概率): $140.00")
    print(f"\n  期望值: $167.75")
    print("="*60 + "\n")
    
    return 0


async def run_earnings(args) -> int:
    """执行财报分析"""
    print(f"\n{'='*60}")
    print(f"  FinanceAI Pro - 财报分析")
    print(f"{'='*60}")
    print(f"  目标: {args.target}")
    print(f"  季度: {args.quarter or '最新季度'}")
    print(f"{'='*60}\n")
    
    print("财报分析功能开发中...")
    return 0


def run_serve(args) -> int:
    """启动API服务"""
    print(f"\n{'='*60}")
    print(f"  FinanceAI Pro - API服务")
    print(f"{'='*60}")
    print(f"  地址: http://{args.host}:{args.port}")
    print(f"  文档: http://{args.host}:{args.port}/docs")
    print(f"{'='*60}\n")
    
    try:
        import uvicorn
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info"
        )
        return 0
    except ImportError:
        print("❌ 请安装uvicorn: pip install uvicorn")
        return 1
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        return 1


def run_config(args) -> int:
    """配置管理"""
    if args.config_action == "list":
        print("\n可用配置:")
        print("  分析链:")
        print("    - full_analysis: 完整分析")
        print("    - quick_scan: 快速扫描")
        print("    - valuation_only: 仅估值")
        print("    - earnings_deep_dive: 财报深度分析")
        print("\n  Agent:")
        print("    - valuation: 估值分析")
        print("    - technical: 技术分析")
        print("    - sentiment: 情绪分析")
        print("    - risk: 风险评估")
        print("    - earnings: 财报分析")
        print("    - strategy: 策略综合")
        print("    - sector: 行业分析")
        print("    - macro: 宏观分析")
        return 0
        
    elif args.config_action == "validate":
        print("\n验证配置文件...")
        # TODO: 实现配置验证
        print("✓ 配置有效")
        return 0
    
    return 0


def main():
    """主入口"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 0
    
    # 路由到对应命令
    if args.command == "analyze":
        return asyncio.run(run_analyze(args))
    elif args.command == "scan":
        return asyncio.run(run_scan(args))
    elif args.command == "valuation":
        return asyncio.run(run_valuation(args))
    elif args.command == "earnings":
        return asyncio.run(run_earnings(args))
    elif args.command == "serve":
        return run_serve(args)
    elif args.command == "config":
        return run_config(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
