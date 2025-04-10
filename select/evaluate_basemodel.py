from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
import os
from tqdm import tqdm
import time
from evaluation_utils import EvaluationMetrics
import random

# 配置部分
MODEL_PATH = "/home/zqq/model/Qwen-7B-Chat"  # 基础模型路径
DATA_PATH = "select.json"
SAVE_DIR = "evaluate_basemodel_results"  # 结果保存目录

class BaseQwenEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("初始化系统...")
        
        # 加载模型和tokenizer
        print(f"加载模型: {MODEL_PATH}")
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
        
        self.metrics = EvaluationMetrics(SAVE_DIR)
        print("系统初始化完成!")

    def generate_answer(self, question, options):
        # 构建提示模板
        query = f"""请回答以下选择题。注意：
1. 只需要回答选项字母（A/B/C/D）
2. 基于你的知识作答

问题：{question}

选项：
{chr(10).join(options)}

请直接回答选项字母："""
        
        try:
            # 记录开始时间
            start_time = time.time()
            response, history = self.model.chat(
                self.tokenizer, 
                query,
                history=None,
                temperature=0.1
            )
            
            # 计算响应时间
            response_time = time.time() - start_time
            
            # 提取答案 - 增强版
            answer = None
            for char in response:
                if char.upper() in ['A', 'B', 'C', 'D']:
                    answer = char.upper()
                    break
            
            # 如果仍未找到答案，使用模型logits强制生成
            if not answer:
                input_ids = self.tokenizer.encode(query, return_tensors='pt').to(self.device)
                logits = self.model.generate(
                    input_ids=input_ids,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=1
                ).scores[0]
                answer = chr(65 + torch.argmax(logits[0, 65:69]).item())  # A=65,B=66,C=67,D=68
            
            return answer, response_time
            
        except Exception as e:
            print(f"生成答案时出错: {str(e)}")
            # 异常时使用随机选择
            return random.choice(['A','B','C','D']), 0

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
    evaluator = BaseQwenEvaluator()
    results = evaluator.evaluate(DATA_PATH)
    evaluator.metrics.print_results(results)

if __name__ == "__main__":
    main()