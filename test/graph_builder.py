from typing import List
from neo4j import GraphDatabase
from knowledge_extractor import Triple
from config import KNOWLEDGE_GRAPH_CONFIG

class GraphBuilder:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.node_types = KNOWLEDGE_GRAPH_CONFIG["node_types"]
        
    def close(self):
        self.driver.close()
        
    def build_graph(self, triples: List[Triple]):
        """构建知识图谱
        
        Args:
            triples: 知识三元组列表
        """
        with self.driver.session() as session:
            # 清空数据库
            session.run("MATCH (n) DETACH DELETE n")
            
            # 创建节点和关系
            for triple in triples:
                # 创建主体节点
                session.run(
                    "MERGE (s:Node {name: $name}) "
                    "SET s.type = $type",
                    name=triple.subject,
                    type=self._get_node_type(triple.subject)
                )
                
                # 创建客体节点
                session.run(
                    "MERGE (o:Node {name: $name}) "
                    "SET o.type = $type",
                    name=triple.object,
                    type=self._get_node_type(triple.object)
                )
                
                # 创建关系
                session.run(
                    "MATCH (s:Node {name: $subject}), (o:Node {name: $object}) "
                    "MERGE (s)-[r:RELATION {type: $relation}]->(o)",
                    subject=triple.subject,
                    object=triple.object,
                    relation=triple.relation
                )
                
    def _get_node_type(self, name: str) -> str:
        """根据节点名称推断节点类型
        
        Args:
            name: 节点名称
            
        Returns:
            str: 节点类型
        """
        # 检查节点名称是否直接匹配某个类型
        for node_type in self.node_types:
            if node_type in name:
                return node_type
        
        # 根据关键词推断节点类型
        keywords = {
            "疾病": ["病", "症", "炎"],
            "症状": ["痛", "不适", "感", "下降", "模糊"],
            "检查方法": ["检查", "测量", "镜"],
            "治疗方法": ["治疗", "手术", "用药"],
            "风险因素": ["因素", "风险", "原因"],
            "预防措施": ["预防", "避免", "建议"]
        }
        
        for node_type, type_keywords in keywords.items():
            for keyword in type_keywords:
                if keyword in name:
                    return node_type
        
        return "未知" 