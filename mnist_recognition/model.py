"""MNIST 神经网络模型模块"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTNet(nn.Module):
    """MNIST 手写数字识别CNN模型"""

    # 模型参数常量
    INPUT_CHANNELS = 1
    CONV1_OUT_CHANNELS = 32
    CONV2_OUT_CHANNELS = 64
    FC1_OUT_FEATURES = 128
    FC2_OUT_FEATURES = 10
    DROPOUT1_RATE = 0.25
    DROPOUT2_RATE = 0.5

    def __init__(self) -> None:
        """初始化MNIST神经网络模型"""
        super().__init__()

        # 第一个卷积层: 输入1通道, 输出32通道, 卷积核3x3
        self.conv1 = nn.Conv2d(
            self.INPUT_CHANNELS,
            self.CONV1_OUT_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 第二个卷积层: 输入32通道, 输出64通道, 卷积核3x3
        self.conv2 = nn.Conv2d(
            self.CONV1_OUT_CHANNELS,
            self.CONV2_OUT_CHANNELS,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # 池化层: 2x2 最大池化
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout层: 防止过拟合
        self.dropout1 = nn.Dropout2d(self.DROPOUT1_RATE)
        self.dropout2 = nn.Dropout2d(self.DROPOUT2_RATE)

        # 全连接层1: 输入7*7*64, 输出128
        self.fc1 = nn.Linear(7 * 7 * self.CONV2_OUT_CHANNELS, self.FC1_OUT_FEATURES)
        # 全连接层2: 输入128, 输出10 (10个数字类别)
        self.fc2 = nn.Linear(self.FC1_OUT_FEATURES, self.FC2_OUT_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量, 形状为 (batch_size, 1, 28, 28)

        Returns:
            输出张量, 形状为 (batch_size, 10)
        """
        # 第一个卷积块: 卷积 -> ReLU -> 池化
        x = F.relu(self.conv1(x))  # 输出: (batch_size, 32, 28, 28)
        x = self.pool(x)           # 输出: (batch_size, 32, 14, 14)

        # 第二个卷积块: 卷积 -> ReLU -> 池化
        x = F.relu(self.conv2(x))  # 输出: (batch_size, 64, 14, 14)
        x = self.pool(x)           # 输出: (batch_size, 64, 7, 7)

        # Dropout
        x = self.dropout1(x)

        # 展平张量
        x = torch.flatten(x, 1)    # 输出: (batch_size, 7*7*64)

        # 全连接层
        x = F.relu(self.fc1(x))    # 输出: (batch_size, 128)
        x = self.dropout2(x)
        x = self.fc2(x)           # 输出: (batch_size, 10)

        # 应用 softmax 获取概率分布
        output = F.log_softmax(x, dim=1)

        return output


class SimpleMNISTNet(nn.Module):
    """简化的 MNIST 全连接神经网络模型"""

    # 模型参数常量
    INPUT_SIZE = 28 * 28
    FC1_OUT_FEATURES = 256
    FC2_OUT_FEATURES = 128
    FC3_OUT_FEATURES = 10
    DROPOUT_RATE = 0.2

    def __init__(self) -> None:
        """初始化简化的MNIST神经网络模型"""
        super().__init__()

        # 全连接层
        self.fc1 = nn.Linear(self.INPUT_SIZE, self.FC1_OUT_FEATURES)
        self.fc2 = nn.Linear(self.FC1_OUT_FEATURES, self.FC2_OUT_FEATURES)
        self.fc3 = nn.Linear(self.FC2_OUT_FEATURES, self.FC3_OUT_FEATURES)

        # Dropout层
        self.dropout = nn.Dropout(self.DROPOUT_RATE)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x: 输入张量, 形状为 (batch_size, 1, 28, 28)

        Returns:
            输出张量, 形状为 (batch_size, 10)
        """
        # 展平输入
        x = torch.flatten(x, 1)    # 输出: (batch_size, 28*28)

        # 全连接层 + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        # 应用 softmax 获取概率分布
        output = F.log_softmax(x, dim=1)

        return output


def get_model(model_type: str = 'cnn') -> nn.Module:
    """
    获取模型实例

    Args:
        model_type: 模型类型, 'cnn' 或 'simple'，默认为'cnn'

    Returns:
        模型实例

    Raises:
        ValueError: 当模型类型不支持时
    """
    if model_type == 'cnn':
        return MNISTNet()
    if model_type == 'simple':
        return SimpleMNISTNet()

    raise ValueError(f"不支持的模型类型: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """
    计算模型参数数量

    Args:
        model: 神经网络模型

    Returns:
        模型参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
