# 五子棋 AI - AlphaZero 算法实现

基于 AlphaZero 算法（MCTS + CNN）的五子棋人工智能项目，支持训练、游戏对弈和AI分析。

## 🎯 项目特性

- **强大的AI算法**: 使用 MCTS (蒙特卡洛树搜索) + ResNet CNN 架构
- **跨平台训练**: 支持 NVIDIA GPU (CUDA) 和 Apple Silicon (MPS) 训练
- **可视化界面**: 提供图形化游戏界面和AI决策分析
- **训练监控**: 实时监控训练进度和模型性能
- **模型对比**: AI模型之间的对弈评估工具

## 📁 项目结构

```
gomoku-alphazero-ai/
├── train.py                    # 核心训练脚本 (AlphaZero算法)
├── game_gui.py                 # 图形化游戏界面
├── ai_battle.py                # AI对弈评估工具
├── training_monitor.py         # 训练进度监控
├── model_4090_trained.pth      # 预训练模型 (RTX 4090训练)
├── gomoku_cnn_strong/          # 强化训练模型目录
├── requirements.txt            # 项目依赖
├── LICENSE                     # MIT许可证
└── README.md                   # 项目说明
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/Proton1917/gomoku-alphazero-ai.git
cd gomoku-alphazero-ai

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行游戏界面

```bash
python game_gui.py
```

## 🎮 游戏操作

- **鼠标点击**: 落子
- **R键**: 开启/关闭AI思考可视化
- **A键**: 自动对弈模式
- **P键**: AI走一步
- **B键**: 悔棋
- **F键**: 前进一步
- **S键**: 显示神经网络评估

## 🏋️ 训练说明

### 支持的训练平台

1. **NVIDIA GPU** (推荐)
   - RTX 3060 或更高
   - CUDA 11.8+
   - 训练速度: ~1-2天完成50轮

2. **Apple Silicon** (M1/M2/M3/M4)
   - 使用 MPS 加速
   - 训练速度: 比NVIDIA GPU慢约2-3倍

3. **CPU训练** (不推荐)
   - 训练时间极长，仅用于测试

### 开始训练

```bash
# 从头开始训练
python train.py

# 监控训练进度
python training_monitor.py
```

### 训练配置

编辑 `train.py` 中的 `Config` 类来调整训练参数：

```python
class Config:
    batch_size = 128        # 批大小
    num_epochs = 5          # 每轮训练轮数
    learning_rate = 2e-4    # 学习率
    num_samples = 200       # 每轮生成样本数
    train_simulation = 50   # MCTS模拟次数
    channel = 64           # CNN通道数
```

### 训练流程

1. **数据生成**: AI自我对弈生成训练数据
2. **模型训练**: 使用生成的数据训练神经网络
3. **模型评估**: 新模型与旧模型对弈测试
4. **迭代更新**: 重复上述过程50轮

## 📚 详细训练教程

### 第一次训练 (从零开始)

1. **检查环境**
   ```bash
   # 检查 CUDA 支持 (NVIDIA GPU)
   python -c "import torch; print(torch.cuda.is_available())"
   
   # 检查 MPS 支持 (Apple Silicon)
   python -c "import torch; print(torch.backends.mps.is_available())"
   ```

2. **配置训练参数**
   
   根据你的硬件调整 `train.py` 中的配置：
   
   ```python
   # 高端GPU (RTX 4070+)
   class Config:
       batch_size = 128
       num_samples = 200
       train_simulation = 50
       channel = 64
   
   # 中端GPU (RTX 3060/4060) 或 Apple Silicon
   class Config:
       batch_size = 64
       num_samples = 150
       train_simulation = 40
       channel = 48
   
   # 低端GPU 或 CPU
   class Config:
       batch_size = 32
       num_samples = 100
       train_simulation = 30
       channel = 32
   ```

3. **开始训练**
   ```bash
   python train.py
   ```

4. **监控进度**
   ```bash
   # 在另一个终端窗口运行
   python training_monitor.py
   ```

### 继续训练 (基于已有模型)

如果要基于 `model_4090_trained.pth` 继续强化训练：

1. **修改配置**
   ```python
   # 在 train.py 中设置基础模型
   class Config:
       base_path = 'model_4090_trained.pth'  # 设置基础模型路径
       model_path = 'gomoku_cnn_strong'      # 输出目录
   ```

2. **开始强化训练**
   ```bash
   python train.py
   ```

### 训练时间估算

| 硬件配置 | 每轮训练时间 | 总时间(50轮) |
|---------|-------------|-------------|
| RTX 4090 | ~1-2小时 | 2-4天 |
| RTX 4070 | ~2-3小时 | 4-6天 |
| RTX 3060 | ~3-4小时 | 6-8天 |
| M3 Max | ~4-6小时 | 8-12天 |
| M2 Pro | ~6-8小时 | 12-16天 |
| M1 | ~8-12小时 | 16-25天 |

### 训练优化技巧

1. **GPU 内存优化**
   ```python
   # 如果遇到 CUDA OOM 错误，减小批大小
   Config.batch_size = 32  # 或更小
   ```

2. **训练稳定性**
   ```python
   # 降低学习率提高稳定性
   Config.learning_rate = 1e-4
   ```

3. **加速训练**
   ```python
   # 减少每轮样本数但增加训练轮数
   Config.num_samples = 100
   Config.num_epochs = 8
   ```

### 训练监控指标

运行 `training_monitor.py` 可以看到：

- **训练进度**: 当前完成轮数/总轮数
- **模型性能**: 新模型vs基础模型的胜率
- **训练时间**: 每轮耗时和预计完成时间
- **资源使用**: GPU/CPU/内存使用情况

### 训练中断与恢复

训练支持自动断点续传：

1. **意外中断**: 直接重新运行 `python train.py`
2. **手动停止**: Ctrl+C 后重新运行
3. **检查点**: 模型自动保存在 `gomoku_cnn_strong/` 目录

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

- **基础模型**: `model_4090_trained.pth` (RTX 4090训练)
- **强化模型**: `gomoku_cnn_strong/32.pth` (最新强化训练)
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
