import requests
import json
import os
from tqdm import tqdm
import time
from metrics import QAMetrics
from config import *
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# API配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-jzspkglygboapudxoyefgedfrhogfflqarvzfafxouhgtpau"
TIMEOUT_SECONDS = 30

class APIQAEvaluator:
    def __init__(self):
        print("初始化系统...")
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        # 使用与本地模型相同的评估指标
        self.metrics = QAMetrics()
        # 设置特定的保存目录
        self.save_dir = os.path.join(SAVE_DIR, 'api_results')
        print("系统初始化完成!")

    def generate_answer(self, question):
        """通过API生成答案"""
        try:
            # 构建API请求
            payload = {
                "model": "Pro/deepseek-ai/DeepSeek-V3",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的医疗问答助手，请根据问题提供准确、专业的回答。"
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ],
                "stream": False,
                "max_tokens": 1000,
                "temperature": 0.1,
                "top_p": 0.1,
                "top_k": 1,
                "frequency_penalty": 0.0,
                "n": 1,
                "response_format": {"type": "text"}
            }
            
            # 记录开始时间
            start_time = time.time()
            
            # 发送API请求，增加超时时间
            response = requests.post(
                API_URL,
                headers=self.headers,
                json=payload,
                timeout=60  # 增加超时时间到60秒
            )
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                return answer, response_time
            else:
                print(f"API请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return "", response_time
            
        except requests.Timeout:
            print(f"请求超时")
            return "", 60  # 使用超时时间作为响应时间
        except Exception as e:
            print(f"生成答案时出错: {str(e)}")
            return "", 0

    def evaluate_single_qa(self, question, reference):
        """评估单个问答对"""
        eval_start_time = time.time()  # 记录评估开始时间
        
        print(f"\n正在评估问题: {question}")  # 添加问题日志
        # 生成答案
        prediction, response_time = self.generate_answer(question)
        
        # 记录预测结果
        print(f"生成的回答: {prediction[:100]}..." if len(prediction) > 100 else prediction)
        
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
        os.makedirs(self.save_dir, exist_ok=True)
        
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
        with open(os.path.join(self.save_dir, 'detailed_results.json'), 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=4)
        
        # 保存平均分数
        with open(os.path.join(self.save_dir, 'average_scores.json'), 'w', encoding='utf-8') as f:
            json.dump(avg_scores, f, ensure_ascii=False, indent=4)
        
        # 创建评分可视化
        scores_df = pd.DataFrame(avg_scores.items(), columns=['Metric', 'Score'])
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Score', y='Metric', data=scores_df)
        plt.title('API Model Average Scores by Metric')
        plt.savefig(os.path.join(self.save_dir, 'scores_visualization.png'))
        plt.close()

def main():
    evaluator = APIQAEvaluator()
    _, avg_scores = evaluator.evaluate()
    
    print("\n评估指标:")
    print("-" * 60)
    print("性能指标:")
    for metric in ['bleu', 'medical_f1', 'bert_score', 'moverscore', 'factcc', 'safety']:
        print(f"{metric:20}: {avg_scores[metric]:.4f}")
    
    print("\n时间指标:")
    print(f"{'API响应时间':20}: {avg_scores['avg_response_time']:.2f} 秒")
    print(f"{'总评估时间':20}: {avg_scores['avg_evaluation_time']:.2f} 秒")
    print(f"{'评估开销时间':20}: {(avg_scores['avg_evaluation_time'] - avg_scores['avg_response_time']):.2f} 秒")
    print("-" * 60)
    print(f"结果保存路径: {evaluator.save_dir}")

if __name__ == "__main__":
    main() 