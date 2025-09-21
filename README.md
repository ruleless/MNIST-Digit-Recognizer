# MNIST 手写数字识别项目

基于 PyTorch 实现的手写数字识别程序，支持 CNN 和简单神经网络两种模型架构。

## 项目结构

```
mnist_recognition/
├── __init__.py          # 包初始化文件
├── data_loader.py       # 数据加载和预处理模块
├── model.py            # 神经网络模型定义
├── trainer.py          # 训练逻辑模块
├── evaluator.py        # 测试和评估模块
└── main.py             # 主程序入口
requirements.txt        # 项目依赖
README.md              # 项目说明文档
```

## 功能特性

- **数据加载**: 自动下载和预处理 MNIST 数据集
- **模型架构**:
  - CNN 模型：包含卷积层、池化层、Dropout 和全连接层
  - 简单模型：仅包含全连接层的神经网络
- **训练功能**: 支持多轮训练，自动保存最佳模型
- **评估功能**: 提供详细的性能评估报告和可视化
- **预测功能**: 支持单张图像预测

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 训练模型

```bash
# 使用 CNN 模型训练 10 轮
python mnist_recognition/main.py --mode train --model_type cnn --epochs 10

# 使用简单模型训练
python mnist_recognition/main.py --mode train --model_type simple --epochs 10

# 自定义参数
python mnist_recognition/main.py --mode train --model_type cnn --epochs 15 --batch_size 128 --lr 0.0005
```

### 2. 评估模型

```bash
# 评估训练好的模型
python mnist_recognition/main.py --mode eval --model_path models/mnist_model.pth

# 指定模型文件
python mnist_recognition/main.py --mode eval --model_path path/to/your/model.pth
```

### 3. 预测示例

```bash
# 使用训练好的模型进行预测
python mnist_recognition/main.py --mode predict --model_path models/mnist_model.pth
```

### 4. 可视化数据

```bash
# 可视化数据样本
python mnist_recognition/main.py --mode train --visualize --epochs 1
```

## 参数说明

- `--mode`: 运行模式
  - `train`: 训练模式
  - `eval`: 评估模式
  - `predict`: 预测模式
- `--model_type`: 模型类型
  - `cnn`: CNN 模型（默认）
  - `simple`: 简单神经网络模型
- `--epochs`: 训练轮数（默认：10）
- `--batch_size`: 批次大小（默认：64）
- `--lr`: 学习率（默认：0.001）
- `--model_path`: 模型保存/加载路径（默认：models/mnist_model.pth）
- `--device`: 设备选择
  - `auto`: 自动选择（默认）
  - `cpu`: 使用 CPU
  - `cuda`: 使用 GPU
- `--download`: 下载数据集
- `--visualize`: 可视化数据样本

## 模型架构

### CNN 模型

- 输入层：28x28 灰度图像
- 卷积层 1：32 个 3x3 卷积核，ReLU 激活
- 池化层 1：2x2 最大池化
- 卷积层 2：64 个 3x3 卷积核，ReLU 激活
- 池化层 2：2x2 最大池化
- Dropout 层 1：0.25
- 全连接层 1：128 个神经元，ReLU 激活
- Dropout 层 2：0.5
- 输出层：10 个神经元（对应 0-9 数字）

### 简单模型

- 输入层：28x28 = 784 个神经元
- 隐藏层 1：256 个神经元，ReLU 激活
- Dropout 层：0.2
- 隐藏层 2：128 个神经元，ReLU 激活
- Dropout 层：0.2
- 输出层：10 个神经元（对应 0-9 数字）

## 训练过程

训练过程包含以下步骤：

1. **数据加载**：自动下载 MNIST 数据集并进行预处理
2. **模型初始化**：根据选择的模型类型创建神经网络
3. **训练循环**：
   - 前向传播计算预测结果
   - 计算损失函数（负对数似然损失）
   - 反向传播更新参数
   - 定期在测试集上评估性能
4. **模型保存**：自动保存性能最佳的模型

## 评估功能

评估模块提供以下功能：

- **准确率计算**：总体准确率和各类别准确率
- **混淆矩阵**：可视化分类结果
- **分类报告**：精确率、召回率、F1 分数等指标
- **错误分析**：显示错误分类的样本
- **可视化**：生成各种性能图表

## 注意事项

1. **数据集**：首次运行时会自动下载 MNIST 数据集，需要网络连接
2. **GPU 支持**：程序会自动检测是否有可用的 GPU，优先使用 GPU 训练
3. **模型保存**：训练好的模型会保存在 `models/` 目录下
4. **内存使用**：CNN 模型需要较多内存，建议使用 GPU 训练

## 常见问题

### Q: 如何解决 CUDA 内存不足的问题？
A: 可以减小批次大小（`--batch_size 32`）或使用 CPU 训练（`--device cpu`）

### Q: 如何提高模型准确率？
A: 可以尝试：
- 增加训练轮数（`--epochs 20`）
- 调整学习率（`--lr 0.0005`）
- 使用更大的批次大小（如果有足够内存）

### Q: 如何使用自己的数据？
A: 需要修改 `data_loader.py` 文件，实现自定义数据集的加载逻辑

## 许可证

本项目采用 MIT 许可证。
