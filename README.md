# 五子棋 AlphaZero AI

基于 AlphaZero 算法（MCTS + ResNet CNN）的五子棋项目，现已同时支持：

- 训练与监控
- Web 版对弈与研究控制台
- Web 版模型对战
- 保留 legacy `pygame` GUI

## 当前架构

```text
Gomoku/
├── train.py                      # 核心模型、MCTS、训练逻辑
├── backend/                      # FastAPI + WebSocket + SQLite 会话持久化
├── frontend/                     # React + TypeScript + Vite 单页应用
├── game_gui.py                   # Legacy: pygame 对弈与研究界面
├── model_battle.py               # Legacy: pygame 模型对战
├── ai_battle.py                  # 批量模型评估脚本
├── training_monitor.py           # 训练监控
├── gomoku_cnn_strong/            # 强化训练模型目录
└── requirements.txt              # Python 依赖
```

## 快速开始

### Python 环境

```bash
conda run -n base python -m pip install -r requirements.txt
```

### 启动 Web 后端

```bash
cd backend
conda run -n base uvicorn main:app --reload
```

### 启动 Web 前端

```bash
cd frontend
npm install
npm run dev
```

### 一键启动（macOS `.command`）

```bash
./start_project.command
```

也可以直接在 Finder 里双击 [start_project.command](/Users/gordongauerk/Projects/Neutron's_machine_learning/Gomoku/start_project.command)。

默认情况下：

- 后端地址：`http://127.0.0.1:8000`
- 前端地址：`http://127.0.0.1:5173`

## Web 功能

### Play Console

- 模型选择 + MCTS 模拟次数配置
- 新建/恢复持久化对局
- 人类落子、AI 落子、悔棋、前进
- 研究模式 WebSocket：每 10 次模拟推送一次热力图
- 自动对弈 WebSocket：持续推进到终局
- NN 概率/价值矩阵可视化

### Battle Console

- 选择黑白双方模型
- 创建持久化对战会话
- Battle WebSocket 流式推进模型对战
- 刷新页面后自动尝试恢复最近一次 Battle 会话

### SQLite 持久化

- 数据库文件：`backend/gomoku_web.sqlite3`
- 持久化内容：棋盘、走子历史、模型路径、模拟次数、终局结果
- 不持久化 `MCTS` 搜索树；服务重启后从当前局面重新搜索

## Legacy GUI

如果你仍然想用本地 `pygame` 界面：

```bash
conda run -n base python game_gui.py
conda run -n base python model_battle.py
```

### 操作键位

- 鼠标点击：落子
- `R`：研究模式
- `A`：自动对弈
- `P`：AI 走一步
- `B`：悔棋
- `F`：前进一步
- `S`：显示神经网络评估

## 训练

```bash
conda run -n base python train.py
conda run -n base python training_monitor.py
```

当前默认模型是 `gomoku_cnn_4090_test/model_4090_trained.pth`。正式模型目录仍然是 `gomoku_cnn_strong/`，模型命名为轮次，例如 `55.pth`。当前模型发现逻辑会优先推荐：

1. 4090 旧基线模型
2. 第 55 轮
3. 第 50 轮
4. 第 49 轮
5. 第 48 轮
6. 第 46 轮

当前保留的实验目录：

- `gomoku_cnn_4090_test/`
  集中保留 4090 基线与它的派生实验模型：
  - `model_4090_trained.pth`
  - `legacy20.pth`
  - `online_refine_latest.pth`

## API 概览

### REST

- `GET /api/models`
- `POST /api/game/new`
- `GET /api/game/{id}`
- `POST /api/game/{id}/move`
- `POST /api/game/{id}/ai-move`
- `POST /api/game/{id}/undo`
- `POST /api/game/{id}/redo`
- `GET /api/game/{id}/nn`
- `POST /api/battle/new`
- `GET /api/battle/{id}`

### WebSocket

- `/ws/game/{id}/research`
- `/ws/game/{id}/autoplay`
- `/ws/battle/{id}`

## 构建检查

```bash
conda run -n base python -m compileall train.py backend game_gui.py model_battle.py
cd frontend && npm run build
```

### 自定义训练

1. **调整棋盘大小**
   ```python
   board_size = 11  # 或 13, 15, 19
   ```

2. **修改网络架构**
   ```python
   Config.channel = 128      # 增加网络容量
   # 在 ResNet 类中添加更多残差块
   ```

3. **调整 MCTS 参数**
   ```python
   Config.train_simulation = 100  # 增加搜索深度
   ```

## 🔧 硬件要求

### 最低配置
- CPU: Intel i5 / AMD Ryzen 5 或同等性能
- 内存: 8GB RAM
- 存储: 1GB 可用空间

### 推荐配置 (训练)
- GPU: RTX 3060 / RTX 4060 或更高
- 内存: 16GB+ RAM
- 存储: 5GB+ 可用空间

### Apple Silicon
- 芯片: M1 / M2 / M3 / M4 任意型号
- 内存: 16GB+ (统一内存)

## 📊 模型性能

 - **基础模型**: `gomoku_cnn_4090_test/model_4090_trained.pth` (RTX 4090训练)
 - **当前默认模型**: `gomoku_cnn_4090_test/model_4090_trained.pth`
 - **独立实验模型**:
  - `gomoku_cnn_4090_test/legacy20.pth`
  - `gomoku_cnn_4090_test/online_refine_latest.pth`
- **棋力评估**: 接近业余高段水平

## 🛠️ 故障排除

### 常见问题

1. **CUDA 内存不足**
   ```python
   # 减小批大小
   Config.batch_size = 64
   ```

2. **MPS 不可用**
   ```bash
   # 检查 PyTorch MPS 支持
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

3. **训练中断恢复**
   - 训练会自动保存检查点
   - 重新运行 `train.py` 会从最新检查点继续

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

MIT License

## 🙏 致谢

- AlphaZero 论文作者
- PyTorch 团队
- 开源社区贡献者
