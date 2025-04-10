import os

# 基础配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 获取当前文件所在目录
MODEL_PATH = "/home/zqq/model/Qwen-7B-Chat"  # 本地模型路径
QA_DATA_PATH = os.path.join(BASE_DIR, "QA.json")  # QA数据路径，直接使用QA.json
SAVE_DIR = os.path.join(BASE_DIR, "evaluation_results")  # 评估结果主目录

# API配置
API_URL = "https://api.siliconflow.cn/v1/chat/completions"
API_KEY = "sk-jzspkglygboapudxoyefgedfrhogfflqarvzfafxouhgtpau"
TIMEOUT_SECONDS = 30

# 评估指标列表
METRICS = [
    'bleu',
    'medical_f1',
    'bert_score',
    'moverscore',
    'factcc',
    'safety',
    'avg_response_time'
] 