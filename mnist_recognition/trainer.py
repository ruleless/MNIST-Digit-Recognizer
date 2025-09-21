"""MNIST Model Trainer Module"""

import os
import time
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


class MNISTTrainer:
    """MNIST model trainer for training and evaluating neural network models."""

    # 默认学习率
    DEFAULT_LEARNING_RATE = 0.001

    def __init__(self, model: nn.Module, train_loader: DataLoader,
                 test_loader: DataLoader, device: str = 'cpu',
                 lr: float = DEFAULT_LEARNING_RATE) -> None:
        """
        Initialize trainer

        Args:
            model: Neural network model
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device ('cpu' or 'cuda'), defaults to 'cpu'
            lr: Learning rate, defaults to 0.001
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

        # 定义损失函数和优化器
        self.criterion = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 训练历史记录
        self.train_losses: List[float] = []
        self.train_accuracies: List[float] = []
        self.test_losses: List[float] = []
        self.test_accuracies: List[float] = []

    def train_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        Train one epoch

        Args:
            epoch: Current epoch number

        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 使用 tqdm 显示进度条
        pbar = tqdm(self.train_loader, desc=f'Training Epoch {epoch}')
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            # 清零梯度
            self.optimizer.zero_grad()

            # 前向传播
            output = self.model(data)
            loss = self.criterion(output, target)

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 统计信息
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = running_loss / len(self.train_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def test(self) -> Tuple[float, float]:
        """
        Test model performance

        Returns:
            tuple: (average loss, accuracy)
        """
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += self.criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = test_loss / len(self.test_loader)
        accuracy = 100. * correct / total

        return avg_loss, accuracy

    def train(self, epochs: int, save_model: bool = True,
              model_path: str = 'mnist_model.pth') -> None:
        """
        Train model

        Args:
            epochs: Number of training epochs
            save_model: Whether to save model, defaults to True
            model_path: Model save path, defaults to 'mnist_model.pth'
        """
        if epochs <= 0:
            raise ValueError("Number of training epochs must be greater than 0")

        print(f"Start training model, device: {self.device}")
        print(f"Training epochs: {epochs}")

        best_accuracy = 0.0

        for epoch in range(1, epochs + 1):
            start_time = time.time()

            # 训练一个 epoch
            train_loss, train_acc = self.train_epoch(epoch)

            # 测试模型
            test_loss, test_acc = self.test()

            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)

            # 打印结果
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch}/{epochs} - '
                  f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_acc:.2f}% - '
                  f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}% - '
                  f'Cost Time: {epoch_time:.2f}s')

            # 保存最佳模型
            if test_acc > best_accuracy and save_model:
                best_accuracy = test_acc
                self.save_model(model_path)
                print(f'Model saved, test accuracy: {best_accuracy:.2f}%')

            print('-' * 80)

        print(f'Training completed! Best test accuracy: {best_accuracy:.2f}%')

    def save_model(self, path: str) -> None:
        """
        Save model

        Args:
            path: Save path
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(path), exist_ok=True)

        # 保存模型状态字典
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies
        }, path)

    def load_model(self, path: str) -> None:
        """
        Load model

        Args:
            path: Model path
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file {path} does not exist")

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.train_accuracies = checkpoint['train_accuracies']
        self.test_losses = checkpoint['test_losses']
        self.test_accuracies = checkpoint['test_accuracies']
        print(f'Model loaded from {path}')

    def get_training_history(self) -> Dict[str, List[float]]:
        """
        Get training history

        Returns:
            Dictionary of training history data
        """
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'test_losses': self.test_losses,
            'test_accuracies': self.test_accuracies
        }
