import requests
import json
import os
from tqdm import tqdm
import time
from evaluation_utils import EvaluationMetrics

# 配置部分
API_URL = "https://api.siliconflow.cn/v1/chat/completions"  # 硅基流动API地址
API_KEY = "sk-jzspkglygboapudxoyefgedfrhogfflqarvzfafxouhgtpau"  # API密钥
DATA_PATH = "select.json"
SAVE_DIR = "api_model_results"  # 结果保存目录
TIMEOUT_SECONDS = 30  # API请求超时时间

class APIEvaluator:
    def __init__(self):
        print("初始化系统...")
        self.headers = {
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json"
        }
        self.metrics = EvaluationMetrics(SAVE_DIR)
        print("系统初始化完成!")

    def generate_answer(self, question, options):
        # 构建提示模板
        query = f"""你是一个专业的选择题答题助手。请仔细阅读以下选择题，并选择最合适的答案。

要求：
1. 仔细分析每个选项的内容
2. 只回答选项字母（A/B/C/D）
3. 确保你的选择是基于题目内容和选项的完整分析
4. 不要解释原因，只给出答案

问题：{question}

选项：
{chr(10).join(options)}

请直接回答选项字母："""
        
        try:
            # 构建API请求
            payload = {
                "model": "Qwen/Qwen2.5-14B-Instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是一个专业的选择题答题助手，擅长分析问题并给出准确的答案。你只会回答选项字母（A/B/C/D），不会解释原因。"
                    },
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                "stream": False,
                "max_tokens": 10,
                "temperature": 0.1,
                "top_p": 0.1,
                "top_k": 1,
                "frequency_penalty": 0.0,
                "n": 1,
                "response_format": {"type": "text"}
            }
            
            # 记录开始时间
            start_time = time.time()
            
            # 发送API请求
            response = requests.post(
                API_URL,
                headers=self.headers,
                json=payload,
                timeout=TIMEOUT_SECONDS
            )
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answer_text = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                
                # 提取答案
                answer = None
                for char in answer_text:
                    if char.upper() in ['A', 'B', 'C', 'D']:
                        answer = char.upper()
                
                return answer if answer else 'A', response_time
            else:
                print(f"API请求失败: {response.status_code}")
                print(f"错误信息: {response.text}")
                return 'A', response_time
            
        except requests.Timeout:
            print(f"请求超时")
            return 'A', TIMEOUT_SECONDS
        except Exception as e:
            print(f"生成答案时出错: {str(e)}")
            return 'A', 0

    def evaluate(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)
        
        total = len(questions)
        print(f"\n开始评估 {total} 道选择题...")
        
        for idx, question in enumerate(questions, 1):
            pred, response_time = self.generate_answer(question['question'], question['options'])
            self.metrics.add_prediction(pred, question['answer'], question, response_time)
            
            print(f"进度: {idx}/{total}, 当前正确率: {self.metrics.correct_count/idx:.2%}", end='\r')
        
        print("\n评估完成!")
        
        # 计算并保存结果
        results = self.metrics.calculate_metrics()
        self.metrics.save_results(results)
        return results

def main():
    evaluator = APIEvaluator()
    results = evaluator.evaluate(DATA_PATH)
    evaluator.metrics.print_results(results)

if __name__ == "__main__":
    main()