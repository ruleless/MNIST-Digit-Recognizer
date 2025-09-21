"""MNIST Model Evaluator Module"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader


class MNISTEvaluator:
    """MNIST model evaluator for evaluating model performance and visualizing results."""

    def __init__(self, model: nn.Module, test_loader: DataLoader,
                 device: str = 'cpu') -> None:
        """
        Initialize the evaluator
        
        Args:
            model: Neural network model
            test_loader: Test data loader
            device: Device ('cpu' or 'cuda'), defaults to 'cpu'
        """
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device
        self.model.eval()

    def evaluate(self) -> Dict[str, Union[float, np.ndarray, Dict, List]]:
        """
        Evaluate model performance
        
        Returns:
            Dictionary containing evaluation results including accuracy, confusion matrix,
            classification report, etc.
        """
        all_predictions = []
        all_targets = []
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                # 获取预测结果
                _, predicted = torch.max(output.data, dim=1)

                # 收集预测和真实标签
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

                # 计算准确率
                total += target.size(0)
                correct += (predicted == target).sum().item()

        # 计算总体准确率
        accuracy = 100. * correct / total

        # 计算混淆矩阵
        cm = confusion_matrix(all_targets, all_predictions)

        # 计算分类报告
        class_names = [str(i) for i in range(10)]
        class_report = classification_report(
            all_targets, all_predictions,
            target_names=class_names,
            output_dict=True
        )

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': all_predictions,
            'targets': all_targets
        }

    def plot_confusion_matrix(
        self, cm: np.ndarray, save_path: Optional[str] = None
    ) -> None:
        """
        Plot confusion matrix

        Args:
            cm: Confusion matrix
            save_path: Save path, optional
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(10),
            yticklabels=range(10),
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        if save_path:
            plt.savefig(save_path)
            print(f"Confusion matrix saved to: {save_path}")

        plt.show()

    def plot_misclassified_examples(
        self, num_examples: int = 5, save_path: Optional[str] = None
    ) -> None:
        """
        Plot misclassified examples

        Args:
            num_examples: Number of examples to display, defaults to 5
            save_path: Save path, optional
        """
        if num_examples <= 0:
            raise ValueError("Number of examples must be greater than 0")

        misclassified = []

        with torch.no_grad():
            for data, target in self.test_loader:
                # If we have enough examples, break early
                if len(misclassified) >= num_examples:
                    break

                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)

                # 找到错误分类的样本
                mask = predicted != target
                if not mask.any():
                    continue

                # Get indices of misclassified examples
                misclassified_indices = [
                    i for i, is_misclassified in enumerate(mask) if is_misclassified
                ]

                # Add misclassified examples until we reach the desired number
                for i in misclassified_indices:
                    if len(misclassified) >= num_examples:
                        break

                    misclassified.append(
                        {
                            "image": data[i].cpu(),
                            "true_label": target[i].cpu().item(),
                            "predicted_label": predicted[i].cpu().item(),
                        }
                    )

        if not misclassified:
            print("No misclassified examples found")
            return

        # 绘制错误分类的示例
        _, axes = plt.subplots(1, len(misclassified), figsize=(15, 3))
        if len(misclassified) == 1:
            axes = [axes]

        for i, example in enumerate(misclassified):
            axes[i].imshow(example["image"].squeeze(), cmap="gray")
            axes[i].set_title(
                f'True: {example["true_label"]}\n'
                f'Predicted: {example["predicted_label"]}'
            )
            axes[i].axis("off")

        plt.suptitle("Misclassified Examples")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Misclassified examples saved to: {save_path}")

        plt.show()

    def plot_class_accuracy(
        self, class_report: Dict, save_path: Optional[str] = None
    ) -> None:
        """
        Plot accuracy for each class

        Args:
            class_report: Classification report
            save_path: Save path, optional
        """
        classes = [str(i) for i in range(10)]
        accuracies = [class_report[cls]["precision"] for cls in classes]

        plt.figure(figsize=(10, 6))
        bars = plt.bar(classes, accuracies, color="skyblue")
        plt.title("Accuracy by Class")
        plt.xlabel("Digit Class")
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)

        # 在柱状图上显示数值
        for bar_obj, acc in zip(bars, accuracies):
            plt.text(
                bar_obj.get_x() + bar_obj.get_width() / 2,
                bar_obj.get_height() + 0.01,
                f"{acc:.3f}",
                ha="center",
                va="bottom",
            )

        if save_path:
            plt.savefig(save_path)
            print(f"Class accuracy plot saved to: {save_path}")

        plt.show()

    def print_evaluation_report(self, results: Dict) -> None:
        """
        Print evaluation report

        Args:
            results: Evaluation results
        """
        print("=" * 60)
        print("Model Evaluation Report")
        print("=" * 60)
        print(f"Overall Accuracy: {results['accuracy']:.2f}%")
        print()

        print("Classification Report:")
        print("-" * 40)
        for i in range(10):
            cls = str(i)
            report = results["classification_report"][cls]
            print(f"Class {cls}:")
            print(f"  Precision: {report['precision']:.3f}")
            print(f"  Recall: {report['recall']:.3f}")
            print(f"  F1-Score: {report['f1-score']:.3f}")
            print(f"  Support: {report['support']}")
            print()

        print("Macro Average:")
        macro_avg = results["classification_report"]["macro avg"]
        print(f"  Precision: {macro_avg['precision']:.3f}")
        print(f"  Recall: {macro_avg['recall']:.3f}")
        print(f"  F1-Score: {macro_avg['f1-score']:.3f}")
        print()

        print("Weighted Average:")
        weighted_avg = results["classification_report"]["weighted avg"]
        print(f"  Precision: {weighted_avg['precision']:.3f}")
        print(f"  Recall: {weighted_avg['recall']:.3f}")
        print(f"  F1-Score: {weighted_avg['f1-score']:.3f}")
        print("=" * 60)

    def predict_single_image(self, image: torch.Tensor) -> Tuple[int, float]:
        """
        Predict a single image

        Args:
            image: Input image tensor

        Returns:
            tuple: (predicted_class, confidence)
        """
        self.model.eval()
        with torch.no_grad():
            image = image.to(self.device)
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # 添加批次维度

            output = self.model(image)
            probabilities = torch.exp(output)
            confidence, predicted = torch.max(probabilities, 1)

            return predicted.item(), confidence.item()
