"""MNIST 数据加载器模块"""

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class MNISTDataLoader:
    """MNIST 数据加载器类，用于加载和预处理 MNIST 数据集。"""

    # MNIST 数据集的标准化参数
    MNIST_MEAN = 0.1307
    MNIST_STD = 0.3081
    NUM_CLASSES = 10

    def __init__(self, batch_size: int = 64, download: bool = True,
                 data_root: str = './data') -> None:
        """
        初始化数据加载器

        Args:
            batch_size: 批次大小，默认为64
            download: 是否下载数据集，默认为True
            data_root: 数据集存储根目录，默认为'./data'
        """
        self.batch_size = batch_size
        self.download = download
        self.data_root = data_root

        # 定义数据预处理转换
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((self.MNIST_MEAN,), (self.MNIST_STD,))
        ])

        # 加载训练集和测试集
        self._load_datasets()

        # 创建数据加载器
        self._create_data_loaders()

    def _load_datasets(self) -> None:
        """加载训练集和测试集"""
        self.train_dataset: Dataset = datasets.MNIST(
            root=self.data_root,
            train=True,
            download=self.download,
            transform=self.transform
        )

        self.test_dataset: Dataset = datasets.MNIST(
            root=self.data_root,
            train=False,
            download=self.download,
            transform=self.transform
        )

    def _create_data_loaders(self) -> None:
        """创建数据加载器"""
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )

    def get_train_loader(self) -> DataLoader:
        """获取训练数据加载器

        Returns:
            DataLoader: 训练数据加载器
        """
        return self.train_loader

    def get_test_loader(self) -> DataLoader:
        """获取测试数据加载器

        Returns:
            DataLoader: 测试数据加载器
        """
        return self.test_loader

    def get_dataset_info(self) -> Dict[str, int]:
        """获取数据集信息

        Returns:
            Dict[str, int]: 包含数据集信息的字典
        """
        return {
            'train_size': len(self.train_dataset),
            'test_size': len(self.test_dataset),
            'batch_size': self.batch_size,
            'num_classes': self.NUM_CLASSES
        }

    def visualize_samples(self, num_samples: int = 6) -> None:
        """可视化样本数据

        Args:
            num_samples: 要显示的样本数量，默认为6
        """
        if num_samples <= 0:
            raise ValueError("Number of samples must be greater than 0")

        # 获取一个批次的数据
        dataiter = iter(self.train_loader)
        images, labels = next(dataiter)

        # 创建图形
        fig, axes = plt.subplots(1, num_samples, figsize=(12, 2))
        fig.suptitle('MNIST Sample Data')

        for i in range(min(num_samples, len(images))):
            # 显示图像
            axes[i].imshow(images[i].squeeze(), cmap='gray')
            axes[i].set_title(f'label: {labels[i].item()}')
            axes[i].axis('off')

        plt.tight_layout()
        plt.show()

    def get_class_distribution(self) -> Dict[str, np.ndarray]:
        """获取类别分布

        Returns:
            Dict[str, np.ndarray]: 包含训练集和测试集类别分布的字典
        """
        # 安全地获取标签数据
        train_labels = self._get_dataset_labels(self.train_dataset)
        test_labels = self._get_dataset_labels(self.test_dataset)

        train_dist = np.bincount(train_labels, minlength=self.NUM_CLASSES)
        test_dist = np.bincount(test_labels, minlength=self.NUM_CLASSES)

        return {
            'train_distribution': train_dist,
            'test_distribution': test_dist
        }

    def _get_dataset_labels(self, dataset: Dataset) -> np.ndarray:
        """安全地获取数据集标签

        Args:
            dataset: 数据集对象

        Returns:
            np.ndarray: 标签数组
        """
        if hasattr(dataset, 'targets'):
            return dataset.targets.numpy()

        # 如果数据集没有targets属性，则遍历数据集获取标签
        labels = []
        for _, label in dataset:
            labels.append(label)
        return np.array(labels)
