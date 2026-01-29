#!/bin/bash

# FinMind Development Server Startup Script
# 启动前端和后端开发服务器

echo "🚀 Starting FinMind Development Servers..."
echo ""

# 检查是否在项目根目录
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# 启动 API 服务器
echo "📡 Starting API server on http://localhost:8000..."
cd api && python main.py &
API_PID=$!

# 等待 API 服务器启动
sleep 3

# 启动前端开发服务器
echo "🎨 Starting frontend dev server on http://localhost:3000..."
cd ../web && npm run dev &
WEB_PID=$!

echo ""
echo "✅ Development servers started!"
echo ""
echo "   Frontend: http://localhost:3000"
echo "   API:      http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop all servers"

# 等待用户中断
trap "kill $API_PID $WEB_PID 2>/dev/null; exit 0" SIGINT SIGTERM
wait
