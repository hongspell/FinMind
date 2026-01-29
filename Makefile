# FinMind Makefile
# 简化常用开发命令

.PHONY: help install dev test lint format clean docker-up docker-down serve web api

# 默认目标
help:
	@echo "FinMind - 可用命令:"
	@echo ""
	@echo "  make install     - 安装Python依赖"
	@echo "  make dev         - 安装开发依赖"
	@echo "  make web-install - 安装前端依赖"
	@echo "  make web         - 启动前端开发服务器"
	@echo "  make api         - 启动API服务器"
	@echo "  make start       - 同时启动前端和API"
	@echo "  make test        - 运行测试"
	@echo "  make lint        - 代码检查"
	@echo "  make format      - 代码格式化"
	@echo "  make clean       - 清理缓存文件"
	@echo "  make analyze     - 示例分析命令"

# 安装生产依赖
install:
	pip install -r requirements.txt

# 安装开发依赖
dev:
	pip install -r requirements.txt
	pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy

# 运行测试
test:
	pytest tests/ -v

# 运行测试并生成覆盖率报告
test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term-missing

# 代码检查
lint:
	flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503
	mypy src/ --ignore-missing-imports

# 代码格式化
format:
	black src/ tests/ --line-length=100
	isort src/ tests/

# 清理缓存
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true

# Docker命令
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

docker-build:
	docker-compose build

# ============ 前端命令 ============

# 安装前端依赖
web-install:
	cd web && npm install

# 启动前端开发服务器
web:
	cd web && npm run dev

# 构建前端
web-build:
	cd web && npm run build

# ============ API命令 ============

# 启动API服务器 (开发模式)
api:
	cd api && python main.py

# 启动API服务 (开发模式) - 旧命令兼容
serve:
	python -m src.main serve --port 8000 --reload

# 启动API服务 (生产模式)
serve-prod:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ============ 同时启动 ============

# 同时启动前端和API (需要两个终端)
start:
	@echo "请在两个终端分别运行:"
	@echo "  终端1: make api"
	@echo "  终端2: make web"
	@echo ""
	@echo "或者运行: ./scripts/start-dev.sh"

# 示例分析命令
analyze:
	python -m src.main analyze AAPL --chain full_analysis

# 快速扫描示例
scan:
	python -m src.main scan AAPL MSFT GOOGL TSLA --chain quick_scan

# 验证配置
validate-config:
	python -m src.main config validate

# 生成文档
docs:
	@echo "TODO: 生成API文档"

# 数据库迁移
db-init:
	docker-compose exec timescaledb psql -U financeai -d financeai -f /docker-entrypoint-initdb.d/init-db.sql

# 检查所有内容
check: lint test
	@echo "所有检查通过!"
