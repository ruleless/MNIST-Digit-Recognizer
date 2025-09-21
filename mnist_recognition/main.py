"""MNIST 手写数字识别主程序"""

import argparse
import os
import random
import sys
import torch

# 将项目根目录添加到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from mnist_recognition.data_loader import MNISTDataLoader
from mnist_recognition.evaluator import MNISTEvaluator
from mnist_recognition.model import count_parameters, get_model
from mnist_recognition.trainer import MNISTTrainer


def parse_arguments() -> argparse.Namespace:
    """解析命令行参数

    Returns:
        解析后的参数对象
    """
    parser = argparse.ArgumentParser(
        description="""MNIST 手写数字识别程序

这是一个基于 PyTorch 的 MNIST 手写数字识别系统，支持训练、评估和预测功能。
程序提供两种模型类型：简单的神经网络和卷积神经网络(CNN)。

使用示例:
  python mnist_recognition/main.py --mode train --model_type cnn --epochs 10
  python mnist_recognition/main.py --mode eval --model_path models/mnist_model.pth
  python mnist_recognition/main.py --mode predict --visualize
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="更多信息请参考项目文档或使用 --help 参数查看详细帮助。",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "predict"],
        help="运行模式: train(训练模型), eval(评估模型性能), predict(预测数字)",
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='cnn',
        choices=['cnn', 'simple'],
        help='模型类型: cnn(卷积神经网络，准确率更高), simple(简单神经网络，训练更快)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=5,
        help='训练轮数 (建议值: 5-20，更多轮数通常能获得更好效果但训练时间更长)'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='批次大小 (影响训练速度和内存使用，建议值: 32, 64, 128)'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='学习率 (控制模型更新步长，建议值: 0.0001-0.01)'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/mnist_model.pth',
        help='模型保存/加载路径 (训练时保存模型，评估/预测时加载模型)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='设备选择: auto(自动选择最佳设备), cpu(CPU计算), cuda(GPU加速，需要CUDA支持)'
    )
    parser.add_argument(
        '--download',
        action='store_true',
        help='下载数据集 (首次运行或需要重新下载数据时使用)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='可视化数据样本 (显示训练数据集中的样本图像，帮助理解数据格式)'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='MNIST Digit Recognizer 1.0.0',
        help='显示程序版本信息'
    )

    return parser.parse_args()


def get_device(device_option: str) -> str:
    """获取计算设备

    Args:
        device_option: 设备选项 ('auto', 'cpu', 'cuda')

    Returns:
        设备字符串
    """
    if device_option == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_option


def setup_data_loaders(args: argparse.Namespace) -> tuple:
    """设置数据加载器

    Args:
        args: 命令行参数

    Returns:
        数据加载器和数据加载器的元组
    """
    print("Loading dataset...")
    data_loader = MNISTDataLoader(
        batch_size=args.batch_size,
        download=args.download
    )
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()

    # 显示数据集信息
    dataset_info = data_loader.get_dataset_info()
    print("Dataset information:")
    print(f"  Training set size: {dataset_info['train_size']}")
    print(f"  Test set size: {dataset_info['test_size']}")
    print(f"  Batch size: {dataset_info['batch_size']}")
    print(f"  Number of classes: {dataset_info['num_classes']}")

    return data_loader, train_loader, test_loader


def train_mode(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader,
    test_loader,
    device: str,
) -> None:
    """训练模式

    Args:
        args: 命令行参数
        model: 神经网络模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 计算设备
    """
    print("Starting training...")
    trainer = MNISTTrainer(model, train_loader, test_loader, device=device, lr=args.lr)
    trainer.train(epochs=args.epochs, save_model=True, model_path=args.model_path)


def eval_mode(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader,
    test_loader,
    device: str,
) -> None:
    """评估模式

    Args:
        args: 命令行参数
        model: 神经网络模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 计算设备
    """
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return

    print("Loading model for evaluation...")
    trainer = MNISTTrainer(model, train_loader, test_loader, device=device, lr=args.lr)
    trainer.load_model(args.model_path)

    # 创建评估器
    evaluator = MNISTEvaluator(model, test_loader, device=device)

    # 评估模型
    print("Evaluating model performance...")
    results = evaluator.evaluate()
    evaluator.print_evaluation_report(results)

    # 绘制混淆矩阵
    print("Plotting confusion matrix...")
    evaluator.plot_confusion_matrix(results["confusion_matrix"])

    # 绘制错误分类示例
    print("Plotting misclassified examples...")
    evaluator.plot_misclassified_examples(num_examples=5)

    # 绘制各类别准确率
    print("Plotting class accuracy...")
    evaluator.plot_class_accuracy(results["classification_report"])


def predict_mode(
    args: argparse.Namespace,
    model: torch.nn.Module,
    train_loader,
    test_loader,
    device: str,
) -> None:
    """预测模式

    Args:
        args: 命令行参数
        model: 神经网络模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        device: 计算设备
    """
    if not os.path.exists(args.model_path):
        print(f"Error: Model file {args.model_path} does not exist")
        return

    print("Loading model for prediction...")
    trainer = MNISTTrainer(model, train_loader, test_loader, device=device, lr=args.lr)
    trainer.load_model(args.model_path)

    # 创建评估器
    evaluator = MNISTEvaluator(model, test_loader, device=device)

    # 从测试集中随机选择一些样本进行预测
    dataiter = iter(test_loader)
    images, labels = next(dataiter)

    # 随机选择5个样本
    indices = random.sample(range(len(images)), min(5, len(images)))

    print("Prediction results:")
    for idx in indices:
        image = images[idx]
        true_label = labels[idx].item()

        # 预测
        predicted_label, confidence = evaluator.predict_single_image(image)

        print(
            f"True label: {true_label}, "
            f"Predicted label: {predicted_label}, "
            f"Confidence: {confidence:.4f}"
        )


def main() -> None:
    """主函数"""
    args = parse_arguments()

    # 设备选择
    device = get_device(args.device)
    print(f"Using device: {device}")

    # 设置数据加载器
    data_loader, train_loader, test_loader = setup_data_loaders(args)

    # 可视化数据样本
    if args.visualize:
        print("Visualizing data samples...")
        data_loader.visualize_samples()
        return

    # 创建模型
    print(f"Creating {args.model_type} model...")
    model = get_model(args.model_type)
    print(f"Number of model parameters: {count_parameters(model):,}")

    # 根据模式执行相应操作
    if args.mode == 'train':
        train_mode(args, model, train_loader, test_loader, device)
    elif args.mode == 'eval':
        eval_mode(args, model, train_loader, test_loader, device)
    elif args.mode == 'predict':
        predict_mode(args, model, train_loader, test_loader, device)
    else:
        print(f"Error: Unsupported mode {args.mode}")


if __name__ == '__main__':
    main()
