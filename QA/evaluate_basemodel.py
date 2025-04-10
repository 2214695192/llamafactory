from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
from metrics import QAMetrics
from config import *
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import numpy as np
import time

class QAEvaluator:
    def __init__(self):
        """初始化评估器，减少模型加载时的输出"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("初始化系统...")
        
        # 加载模型和tokenizer
        print(f"正在从本地加载模型: {MODEL_PATH}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True,
            use_fast=False,
            local_files_only=True  # 强制使用本地文件
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True  # 强制使用本地文件
        )
        self.model.eval()
        
        # 初始化评估指标
        self.metrics = QAMetrics()
        
        print("系统初始化完成!")
    def generate_answer(self, question):
        """生成答案"""
        try:
            response, history = self.model.chat(
                self.tokenizer,
                question,
                history=None,
                temperature=0.1
            )
            return response
        except Exception as e:
            print(f"生成答案时出错: {str(e)}")
            return ""

    def evaluate_single_qa(self, question, reference):
        """评估单个问答对"""
        eval_start_time = time.time()  # 记录评估开始时间
        
        # 生成答案
        prediction = self.generate_answer(question)
        
        # 计算各项指标
        results = {
            'question': question,
            'prediction': prediction,
            'reference': reference,
            'metrics': {
                'bleu': self.metrics.calculate_bleu(prediction, reference),
                'medical_f1': self.metrics.calculate_medical_f1(prediction, reference),
                'bert_score': self.metrics.calculate_bert_score(prediction, reference),
                'moverscore': self.metrics.calculate_moverscore(prediction, reference),
                'factcc': self.metrics.check_factual_consistency(prediction, reference),
                'safety': self.metrics.check_safety(prediction)
            }
        }
        
        # 计算并记录总评估时间
        eval_time = time.time() - eval_start_time
        self.metrics.add_evaluation_time(eval_time)
        results['evaluation_time'] = eval_time
        
        return results

    def evaluate(self):
        """评估所有问答对"""
        # 重置评估时间
        self.metrics.reset_evaluation_times()
        
        # 创建保存目录
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # 加载QA数据
        with open(QA_DATA_PATH, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        
        # 评估每个问答对
        all_results = []
        for qa in tqdm(qa_data, desc="评估进度"):
            result = self.evaluate_single_qa(qa['question'], qa['answer'])
            all_results.append(result)
        
        # 计算平均分数
        avg_scores = {
            'bleu': np.mean([r['metrics']['bleu'] for r in all_results]),
            'medical_f1': np.mean([r['metrics']['medical_f1'] for r in all_results]),
            'bert_score': np.mean([r['metrics']['bert_score']['f1'] for r in all_results]),
            'moverscore': np.mean([r['metrics']['moverscore'] for r in all_results]),
            'factcc': np.mean([r['metrics']['factcc'] for r in all_results]),
            'safety': np.mean([r['metrics']['safety']['score'] for r in all_results]),
            'avg_evaluation_time': self.metrics.get_average_evaluation_time()
        }
        
        # 保存结果
        self.save_results(all_results, avg_scores)
        
        return all_results, avg_scores

    def save_results(self, all_results, avg_scores):
        """保存评估结果"""
        # 保存详细结果
        with open(os.path.join(SAVE_DIR, 'detailed_results.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        
        # 保存平均分数
        with open(os.path.join(SAVE_DIR, 'average_scores.json'), 'w', encoding='utf-8') as f:
            json.dump(avg_scores, f, ensure_ascii=False, indent=4)
        
        # 创建评分可视化
        scores_df = pd.DataFrame(avg_scores.items(), columns=['Metric', 'Score'])
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Score', y='Metric', data=scores_df)
        plt.title('Average Scores by Metric')
        plt.savefig(os.path.join(SAVE_DIR, 'scores_visualization.png'))
        plt.close()

    def generate_comparison_report(self, model_results):
        """
        生成模型比较报告
        Args:
            model_results: Dict[str, Dict[str, float]] - 每个模型的评估结果
        """
        # 创建结果表格
        headers = ["Model"] + METRICS
        comparison_table = []
        
        for model_name, scores in model_results.items():
            row = [model_name]
            for metric in METRICS:
                row.append(f"{scores[metric]:.4f}")
            comparison_table.append(row)
        
        # 保存比较结果
        output_path = os.path.join(SAVE_DIR, "model_comparison.csv")
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(comparison_table)
        
        # 为每个指标生成可视化比较图
        for metric in METRICS:
            plt.figure(figsize=(10, 6))
            values = [results[metric] for results in model_results.values()]
            plt.bar(model_results.keys(), values)
            plt.title(f"{metric} Comparison")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(SAVE_DIR, f"{metric}_comparison.png"))
            plt.close()

def main():
    evaluator = QAEvaluator()
    _, avg_scores = evaluator.evaluate()
    
    print("\n评估结果:")
    for metric, score in avg_scores.items():
        print(f"{metric}: {score:.4f}")
    print(f"\n详细结果已保存至: {SAVE_DIR}")

if __name__ == "__main__":
    main()