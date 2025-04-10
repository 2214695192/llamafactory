from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
import os
from tqdm import tqdm
import time
from metrics import QAMetrics
from config import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 配置部分
BASE_MODEL_PATH = "/home/zqq/model/Qwen-7B-Chat"  # 基础模型
LORA_PATH = "/home/zqq/LLaMA-Factory/saves/Qwen-7B-Chat/lora/train_2025-03-25-18-47-44"  # LoRA权重路径
SAVE_DIR = os.path.join(os.path.dirname(LORA_PATH), 'qa_evaluation_results')  # 评估结果保存目录

class FinetunedQAEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("初始化系统...")
        
        print(f"正在从本地加载基础模型: {BASE_MODEL_PATH}")
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH, 
            trust_remote_code=True,
            use_fast=False
        )
        
        # 加载基础模型
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            trust_remote_code=True
        )
        
        # 加载LoRA权重
        print(f"加载LoRA权重: {LORA_PATH}")
        self.model = PeftModel.from_pretrained(
            self.model,
            LORA_PATH
        )
        self.model.eval()
        
        # 初始化评估指标
        self.metrics = QAMetrics()
        print("系统初始化完成!")

    def generate_answer(self, question):
        """生成答案"""
        try:
            # 记录开始时间
            start_time = time.time()
            
            response, history = self.model.chat(
                self.tokenizer,
                question,
                history=None,
                temperature=0.1
            )
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            return response, response_time
        except Exception as e:
            print(f"生成答案时出错: {str(e)}")
            return "", 0

    def evaluate_single_qa(self, question, reference):
        """评估单个问答对"""
        eval_start_time = time.time()  # 记录评估开始时间
        
        # 生成答案
        prediction, response_time = self.generate_answer(question)
        
        # 计算各项指标
        results = {
            'question': question,
            'prediction': prediction,
            'reference': reference,
            'response_time': response_time,
            'metrics': {
                'bleu': self.metrics.calculate_bleu(prediction, reference),
                'medical_f1': self.metrics.calculate_medical_f1(prediction, reference),
                'bert_score': self.metrics.calculate_bert_score(prediction, reference),
                'moverscore': self.metrics.calculate_moverscore(prediction, reference),
                'factcc': self.metrics.calculate_factcc(prediction, reference),
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
        
        total = len(qa_data)
        print(f"\n开始评估 {total} 个问答对...")
        
        # 评估每个问答对
        all_results = []
        for qa in tqdm(qa_data, 
                      desc="评估进度",
                      ncols=80,
                      bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
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
            'avg_response_time': np.mean([r['response_time'] for r in all_results]),
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
        plt.title('Finetuned Model Average Scores by Metric')
        plt.savefig(os.path.join(SAVE_DIR, 'scores_visualization.png'))
        plt.close()

def main():
    evaluator = FinetunedQAEvaluator()
    _, avg_scores = evaluator.evaluate()
    
    print("\n评估指标:")
    print("-" * 60)
    print("性能指标:")
    for metric in ['bleu', 'medical_f1', 'bert_score', 'moverscore', 'factcc', 'safety']:
        print(f"{metric:20}: {avg_scores[metric]:.4f}")
    
    print("\n时间指标:")
    print(f"{'响应时间':20}: {avg_scores['avg_response_time']:.2f} 秒")
    print(f"{'评估时间':20}: {avg_scores['avg_evaluation_time']:.2f} 秒")
    print("-" * 60)
    print(f"结果保存路径: {SAVE_DIR}")

if __name__ == "__main__":
    main() 