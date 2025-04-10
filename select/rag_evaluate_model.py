from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch
import json
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
import jieba
from tqdm import tqdm
import time
from evaluation_utils import EvaluationMetrics

# 配置部分
BASE_MODEL_PATH = "/home/zqq/model/Qwen-7B-Chat"
LORA_PATH = "/home/zqq/LLaMA-Factory/saves/Qwen-7B-Chat/lora/train_2025-04-08-18-53-22"
DATA_PATH = "select.json"
KNOWLEDGE_DIR = "knowledge"  # 知识库文件夹路径
EMBEDDING_MODEL = "BAAI/bge-large-zh"  # 中文embedding模型

class RAGQwenEvaluator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("初始化系统...")
        
        # 1. 加载Qwen模型和tokenizer
        print(f"加载基础模型: {BASE_MODEL_PATH}")
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
        
        print(f"加载LoRA权重: {LORA_PATH}")
        self.model = PeftModel.from_pretrained(
            self.model,
            LORA_PATH
        )
        self.model.eval()
        
        # 2. 初始化向量数据库
        print("初始化向量数据库...")
        self.initialize_vector_store()
        
        self.metrics = EvaluationMetrics(os.path.join(LORA_PATH, 'evaluation_results'))
        print("系统初始化完成!")

    def initialize_vector_store(self):
        """初始化向量数据库"""
        if os.path.exists("vector_store") and os.listdir("vector_store"):
            print("加载已存在的向量数据库...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': self.device}
            )
            self.vector_store = FAISS.load_local(
                "vector_store",
                self.embeddings,
                allow_dangerous_deserialization=True
            )
        else:
            print("创建新的向量数据库...")
            
            # 配置文本分割器
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=400,      # 一个比较适中的长度
                chunk_overlap=50,    # 保持一定的上下文
                separators=[
                    "\n# ",         # 一级标题
                    "\n## ",        # 二级标题
                    "\n### ",       # 三级标题
                    "\n\n",         # 段落
                    "\n",           # 换行
                    "。",           # 句号
                    "；",           # 分号
                    "！",           # 感叹号
                    "？",           # 问号
                    "，",           # 逗号
                    " ",           # 空格
                    ""             # 字符
                ],
                length_function=len,
            )
            
            # 加载并处理文档
            documents = []
            for filename in tqdm(os.listdir(KNOWLEDGE_DIR), desc="处理文件"):
                if filename.endswith('.md'):
                    file_path = os.path.join(KNOWLEDGE_DIR, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        # 分割文档
                        docs = text_splitter.create_documents(
                            texts=[content],
                            metadatas=[{'source': filename}]
                        )
                        documents.extend(docs)
                        
                        # 打印分割信息（调试用）
                        print(f"\n文件 {filename} 被分割为 {len(docs)} 个片段")
                        print(f"平均片段长度: {sum(len(d.page_content) for d in docs)/len(docs):.0f} 字符")
            
            print(f"\n共生成 {len(documents)} 个文档片段")
            
            # 创建向量数据库
            print("正在创建向量数据库...")
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs={'device': self.device}
            )
            
            # 批量处理
            texts = []
            metadatas = []
            for doc in tqdm(documents, desc="准备向量化", ncols=100):
                texts.append(doc.page_content)
                metadatas.append(doc.metadata)
            
            # 批量计算向量
            print("计算文档向量...")
            embeddings = self.embeddings.embed_documents(texts)
            
            # 创建索引
            print("构建FAISS索引...")
            self.vector_store = FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
            
            # 保存
            print("保存向量数据库...")
            self.vector_store.save_local("vector_store")
            print("向量数据库创建完成！")

    def get_relevant_context(self, question):
        """检索相关文档"""
        try:
            # 使用jieba分词增强检索效果
            words = jieba.cut(question)
            processed_question = " ".join(words)
            
            # 检索相关文档
            docs = self.vector_store.similarity_search_with_score(
                processed_question,
                k=3  # 检索前3个最相关的文档
            )
            
            # 构建上下文，包含相关性得分
            contexts = []
            for doc, score in docs:
                # 相关性得分转换为百分比（得分越小越相关）
                relevance = round((1 - score) * 100, 2)
                
                context = f"相关度: {relevance}%\n"
                if doc.metadata.get('source'):
                    source = os.path.basename(doc.metadata['source'])
                    context += f"来源: {source}\n"
                context += f"内容: {doc.page_content.strip()}\n"
                contexts.append(context)
            
            return "\n---\n".join(contexts)
            
        except Exception as e:
            print(f"检索知识库时出错: {str(e)}")
            return ""

    def generate_answer(self, question, options):
        # 获取相关上下文
        context = self.get_relevant_context(question)

        # 构建新的提示模板，要求解释
        query = f"""请基于以下参考资料和你的知识回答选择题。你需要：
1. 仔细阅读参考资料和问题。
2. 先明确给出你的选项字母（A/B/C/D）。
3. 接着，详细解释你为什么选择这个选项，说明你的推理过程或依据。

参考资料：
{context if context else "无"}

问题：{question}

选项：
{chr(10).join(options)}

请按要求回答：
选项：[在此填入A/B/C/D]
解释：[在此填入你的解释]"""

        # 记录开始时间
        start_time = time.time()

        # 调用模型生成答案，移除了 try...except
        response, history = self.model.chat(
            self.tokenizer,
            query,
            history=None,
            temperature=0.1 # 可以适当调整 temperature
        )

        # 计算响应时间
        response_time = time.time() - start_time

        # 提取答案 (注意：这个提取逻辑可能需要根据模型实际输出调整)
        # 简单的提取第一个出现的选项字母作为预测结果
        answer = None
        for char in response:
            if char.upper() in ['A', 'B', 'C', 'D']:
                answer = char.upper()
                break # 取第一个找到的字母

        # 如果没有找到明确的选项字母，可以设置一个默认值或标记
        if answer is None:
            print(f"\n警告: 未能从模型响应中提取明确选项: {response}")
            answer = 'A' # 或者 'N/A', None 等，取决于下游如何处理

        # 返回提取的答案字母和响应时间。完整的响应在 'response' 变量中。
        # 如果需要评估解释，需要修改 evaluate 方法来处理完整的 response。
        return answer, response_time

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
    evaluator = RAGQwenEvaluator()
    results = evaluator.evaluate(DATA_PATH)
    evaluator.metrics.print_results(results)

if __name__ == "__main__":
    main() 