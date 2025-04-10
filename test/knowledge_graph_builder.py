from typing import List, Dict, Any
from neo4j import GraphDatabase
from .config import NEO4J_CONFIG
from .knowledge_extractor import Triple

class KnowledgeGraphBuilder:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_CONFIG["uri"],
            auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"])
        )
        
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
        
    def build(self, triples: List[Triple]):
        """构建知识图谱"""
        with self.driver.session() as session:
            # 创建约束
            self._create_constraints(session)
            
            # 创建节点和关系
            for triple in triples:
                self._create_triple(session, triple)
                
    def _create_constraints(self, session):
        """创建数据库约束"""
        constraints = [
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Disease) ON (n.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Symptom) ON (n.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Examination) ON (n.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:Treatment) ON (n.name) IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (n:RiskFactor) ON (n.name) IS UNIQUE"
        ]
        
        for constraint in constraints:
            session.run(constraint)
            
    def _create_triple(self, session, triple: Triple):
        """创建单个三元组"""
        # 创建主体节点
        subject_query = f"""
        MERGE (s:{self._get_node_type(triple.subject)} {{name: $subject}})
        """
        
        # 创建客体节点
        object_query = f"""
        MERGE (o:{self._get_node_type(triple.object)} {{name: $object}})
        """
        
        # 创建关系
        relation_query = f"""
        MATCH (s:{self._get_node_type(triple.subject)} {{name: $subject}})
        MATCH (o:{self._get_node_type(triple.object)} {{name: $object}})
        MERGE (s)-[r:{triple.relation}]->(o)
        """
        
        # 执行查询
        session.run(subject_query, subject=triple.subject)
        session.run(object_query, object=triple.object)
        session.run(relation_query, subject=triple.subject, object=triple.object)
        
    def _get_node_type(self, name: str) -> str:
        """根据节点名称判断节点类型"""
        # 这里可以根据实际需求实现更复杂的判断逻辑
        if "症状" in name:
            return "Symptom"
        elif "检查" in name:
            return "Examination"
        elif "治疗" in name:
            return "Treatment"
        elif "风险" in name:
            return "RiskFactor"
        else:
            return "Disease" 