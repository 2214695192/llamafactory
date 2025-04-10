import json
import os
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import time
from typing import Dict, List, Any, Tuple
import numpy as np

class EvaluationMetrics:
    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        self.response_times = []
        self.predictions = []
        self.true_labels = []
        self.results_detail = []
        self.correct_count = 0

    def add_prediction(self, pred: str, true_label: str, question: Dict[str, Any], response_time: float = 0.0):
        """添加一个预测结果"""
        self.predictions.append(pred)
        self.true_labels.append(true_label)
        self.response_times.append(response_time)
        
        is_correct = pred == true_label
        if is_correct:
            self.correct_count += 1
        
        self.results_detail.append({
            'id': question['id'],
            'question': question['question'],
            'predicted': pred,
            'correct': true_label,
            'is_correct': is_correct,
            'response_time': response_time
        })

    def calculate_metrics(self) -> Dict[str, Any]:
        """计算所有评估指标"""
        accuracy = accuracy_score(self.true_labels, self.predictions)
        f1 = f1_score(self.true_labels, self.predictions, average='macro', labels=['A', 'B', 'C', 'D'])
        conf_matrix = confusion_matrix(self.true_labels, self.predictions, labels=['A', 'B', 'C', 'D'])
        
        # 计算效率指标
        avg_response_time = np.mean(self.response_times)
        median_response_time = np.median(self.response_times)
        p95_response_time = np.percentile(self.response_times, 95)
        
        return {
            'accuracy': accuracy,
            'f1_score': f1,
            'confusion_matrix': conf_matrix,
            'efficiency_metrics': {
                'average_response_time': avg_response_time,
                'median_response_time': median_response_time,
                'p95_response_time': p95_response_time,
                'response_times': self.response_times
            },
            'predictions': self.predictions,
            'true_labels': self.true_labels,
            'detail': self.results_detail
        }

    def save_results(self, results: Dict[str, Any]):
        """保存评估结果"""
        # 创建结果目录
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 保存详细结果
        results_file = os.path.join(self.save_dir, 'evaluation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'accuracy': results['accuracy'],
                'f1_score': results['f1_score'],
                'efficiency_metrics': results['efficiency_metrics'],
                'detail': results['detail']
            }, f, ensure_ascii=False, indent=4)
        
        # 保存混淆矩阵图
        self._save_confusion_matrix(results['confusion_matrix'])
        
        # 保存响应时间分布图
        self._save_response_time_distribution(results['efficiency_metrics']['response_times'])

    def _save_confusion_matrix(self, conf_matrix):
        """保存混淆矩阵图"""
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        
        labels = ['A', 'B', 'C', 'D']
        plt.xticks(range(len(labels)), labels)
        plt.yticks(range(len(labels)), labels)
        
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                plt.text(j, i, str(conf_matrix[i, j]),
                        horizontalalignment="center",
                        color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(os.path.join(self.save_dir, 'confusion_matrix.png'))
        plt.close()

    def _save_response_time_distribution(self, response_times):
        """保存响应时间分布图"""
        plt.figure(figsize=(10, 6))
        plt.hist(response_times, bins=30, edgecolor='black')
        plt.title('Response Time Distribution')
        plt.xlabel('Response Time (seconds)')
        plt.ylabel('Count')
        plt.savefig(os.path.join(self.save_dir, 'response_time_distribution.png'))
        plt.close()

    def print_results(self, results: Dict[str, Any]):
        """打印评估结果"""
        print(f"\n评估结果:")
        print(f"准确率: {results['accuracy']:.2%}")
        print(f"宏平均F1分数: {results['f1_score']:.4f}")
        print("\n效率指标:")
        print(f"平均响应时间: {results['efficiency_metrics']['average_response_time']:.2f}秒")
        print(f"中位数响应时间: {results['efficiency_metrics']['median_response_time']:.2f}秒")
        print(f"95分位响应时间: {results['efficiency_metrics']['p95_response_time']:.2f}秒")
        print(f"\n详细结果已保存至: {self.save_dir}") 