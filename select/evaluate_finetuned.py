from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
import os
import re  # 新增导入正则模块
from tqdm import tqdm
import time
from evaluation_utils import EvaluationMetrics
import random

# 配置部分（保持不变）
BASE_MODEL_PATH = "/home/zqq/model/Qwen-7B-Chat"
LORA_PATH = "/home/zqq/LLaMA-Factory/saves/Qwen-7B-Chat/lora/newlora"
DATA_PATH = "select.json"
SAVE_DIR = os.path.join(LORA_PATH, 'evaluate_finetuned_results')

class QwenEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading model from {BASE_MODEL_PATH}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH, 
            trust_remote_code=True,
            use_fast=False
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"Loading LoRA weights from {LORA_PATH}...")
        self.model = PeftModel.from_pretrained(
            self.model,
            LORA_PATH
        )
        self.model.eval()
        
        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        self.metrics = EvaluationMetrics(SAVE_DIR)
        print("Model loaded successfully!")

    def generate_answer(self, question, options):
        # --- 修改 Prompt: 加入 Few-Shot 示例 ---
        prompt = f"""你是一个专业的选择题答题与分析助手。请严格按以下步骤操作，并模仿示例的格式：
1. 仔细阅读题目和选项。
2. 进行深入的思考，分析每个选项的正确性。
3. **最后，必须只输出一个字符，即你选择的选项字母（A/B/C/D），并且该字母必须是整个回答的绝对最后一个字符，后面不能有任何其他内容（包括标点、空格或换行符）。**

**示例 1:**
问题：中国的首都是哪里？
选项：
A. 上海
B. 北京
C. 广州
D. 深圳
你的回答：
分析：题目问的是中国的首都。上海是经济中心，广州和深圳是重要城市，但中国的首都是北京。因此，选项B是正确的。其他选项描述的城市虽然重要，但不是首都。
B

**示例 2:**
问题：以下哪个是水果？
选项：
A. 胡萝卜
B. 土豆
C. 苹果
D. 白菜
你的回答：
思考过程：胡萝卜和土豆是蔬菜的根茎部分，白菜是叶菜。苹果是生长在树上，符合水果的特征。所以C是正确答案。A、B、D都不是水果。
C

**现在，请根据以下信息回答问题：**
问题：{question}
选项：
{chr(10).join(options)}
你的回答：
[在此处详细阐述你的思考过程：为什么选它，为什么其他选项不对]
[确保思考过程结束后，紧接着就是最终的单个选项字母]""" # <--- 确保这里没有多余的换行或空格

        start_time = time.time()
        response, _ = self.model.chat(
            self.tokenizer,
            prompt,
            history=None,
            temperature=0.05, # 使用非常低的温度，强制模型遵循格式
            top_p=0.5,       # 进一步限制随机性
            max_new_tokens=450, # 稍微增加以容纳示例和思考过程
            repetition_penalty=1.05 # 轻微惩罚重复
        )
        response_time = time.time() - start_time
        response_text = response.strip()

        # --- 提取逻辑：优先提取最后一个字符 ---
        # 我们仍然保留检查倒数第二个字符的逻辑作为备用，以防万一
        extracted_answer = None
        if response_text:
            last_char = response_text[-1]
            common_punctuation = ['.', '。', ',', '，', '!', '！', '?', '？', ';', '；', ' ', '\n']

            if last_char.upper() in ['A', 'B', 'C', 'D']:
                extracted_answer = last_char.upper()
            elif last_char in common_punctuation and len(response_text) >= 2:
                second_last_char = response_text[-2]
                if second_last_char.upper() in ['A', 'B', 'C', 'D']:
                    extracted_answer = second_last_char.upper()
                else:
                     print(f"\n警告: 最后是标点 '{last_char}', 但倒数第二个 '{second_last_char}' 也非选项。响应尾部:\n...{response_text[-100:]}\n---")
            else:
                 print(f"\n警告: 最后一个字符 '{last_char}' 非选项或预期标点。响应尾部:\n...{response_text[-100:]}\n---")
        else:
             print("\n警告: 模型响应为空。")

        return extracted_answer, response_time, response_text

    def evaluate(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)

        total = len(questions)
        print(f"\n开始评估 {total} 道选择题 (Few-Shot Prompt, 提取最后一个字符)...")

        correct_count = 0
        processed_count = 0
        extraction_failures = 0

        # 使用 tqdm 作为迭代器，它会处理进度条
        with tqdm(total=total, desc="评估进度") as pbar:
            for idx, question_data in enumerate(questions, 1):
                pred_answer, response_time, raw_response = self.generate_answer(
                    question_data['question'],
                    question_data['options']
                )

                processed_count += 1
                is_correct = False
                if pred_answer is None:
                    extraction_failures += 1
                elif pred_answer == question_data['answer']:
                    correct_count += 1
                    is_correct = True

                self.metrics.add_prediction(
                    pred_answer,
                    question_data['answer'],
                    question_data,
                    response_time
                )

                # 更新实时统计信息到 tqdm 的 postfix
                current_accuracy = (correct_count / (processed_count - extraction_failures) * 100) if (processed_count - extraction_failures) > 0 else 0
                extraction_failure_rate = (extraction_failures / processed_count * 100) if processed_count > 0 else 0
                pbar.set_postfix({
                    "准确率": f"{current_accuracy:.2f}%",
                    "提取失败率": f"{extraction_failure_rate:.2f}%",
                    "最后预测": f"{pred_answer}",
                    "耗时": f"{response_time:.2f}s"
                })
                pbar.update(1) # 更新进度条

        print(f"\n评估完成! 处理题目数: {processed_count}, 正确数: {correct_count}, 提取失败数: {extraction_failures}")

        # 计算并保存最终结果
        results = self.metrics.calculate_metrics()
        # 确保 metrics 能处理 None, 并且计算最终的准确率和失败率
        final_accuracy = (results.get('correct_count', 0) / (results.get('total_predictions', 0) - results.get('extraction_failures', extraction_failures)) * 100) if (results.get('total_predictions', 0) - results.get('extraction_failures', extraction_failures)) > 0 else 0
        final_failure_rate = (results.get('extraction_failures', extraction_failures) / results.get('total_predictions', 0) * 100) if results.get('total_predictions', 0) > 0 else 0

        results['extraction_failure_rate'] = final_failure_rate
        results['accuracy_excluding_failures'] = final_accuracy
        self.metrics.save_results(results)
        return results

def main():
    evaluator = QwenEvaluator()
    results = evaluator.evaluate(DATA_PATH)
    evaluator.metrics.print_results(results) # 确保打印时处理 None 值

if __name__ == "__main__":  
    main()