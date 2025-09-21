"""MNIST 手写数字识别

使用示例:
    python mnist_recognition/main.py --mode train --epochs 10
    python mnist_recognition/main.py --mode eval --model_path models/mnist_model.pth
"""

__version__ = "1.0.0"

# 导入主要类和函数，方便外部使用
from .data_loader import MNISTDataLoader
from .evaluator import MNISTEvaluator
from .model import MNISTNet, SimpleMNISTNet, get_model, count_parameters
from .trainer import MNISTTrainer

__all__ = [
    'MNISTDataLoader',
    'MNISTEvaluator',
    'MNISTNet',
    'SimpleMNISTNet',
    'get_model',
    'count_parameters',
    'MNISTTrainer'
]
