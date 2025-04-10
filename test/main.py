import os
from typing import List
from knowledge_extractor import KnowledgeExtractor, Triple
from graph_builder import GraphBuilder
from config import NEO4J_CONFIG

def build_knowledge_graph(md_file_path: str, api_key: str, model_name: str):
    """构建知识图谱
    
    Args:
        md_file_path: Markdown 文件路径
        api_key: 硅基流动 API key
        model_name: 模型名称
    """
    # 初始化知识提取器
    extractor = KnowledgeExtractor(api_key=api_key, model_name=model_name)
    
    # 初始化图谱构建器
    graph_builder = GraphBuilder(
        uri=NEO4J_CONFIG["uri"],
        user=NEO4J_CONFIG["user"],
        password=NEO4J_CONFIG["password"]
    )
    
    # 读取 Markdown 文件
    with open(md_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 提取知识三元组
    triples = extractor.extract_triples(text)
    
    # 构建知识图谱
    graph_builder.build_graph(triples)
    
    print(f"成功构建知识图谱，共提取 {len(triples)} 个三元组")

if __name__ == "__main__":
    # 测试用例
    md_file_path = "test_data.md"
    api_key = "sk-jzspkglygboapudxoyefgedfrhogfflqarvzfafxouhgtpau"  # 你的 API key
    model_name = "Pro/deepseek-ai/DeepSeek-V3"  # 你的模型名称
    
    build_knowledge_graph(md_file_path, api_key, model_name) 