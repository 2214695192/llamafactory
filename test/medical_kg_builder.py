import os
import glob
from typing import List, Dict, Any
from neo4j import GraphDatabase
from knowledge_extractor import KnowledgeExtractor, Triple
from config import KNOWLEDGE_GRAPH_CONFIG, NEO4J_CONFIG, SILICON_CONFIG
from tqdm import tqdm
import time

class MedicalKnowledgeGraphBuilder:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        neo4j_uri: str = "bolt://10.26.2.130:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "Zq08171911",
        docs_dir: str = "test/docs/medical_knowledge"
    ):
        """初始化医学知识图谱构建器
        
        Args:
            api_key: SiliconFlow API密钥
            model_name: 模型名称
            neo4j_uri: Neo4j数据库URI
            neo4j_user: Neo4j用户名
            neo4j_password: Neo4j密码
            docs_dir: 医学文档目录
        """
        self.docs_dir = docs_dir
        self.extractor = KnowledgeExtractor(api_key=api_key, model_name=model_name)
        self.driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        print(f"✅ Connected to Neo4j at {neo4j_uri}")
        
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
        
    def read_markdown_files(self, docs_dir: str = None) -> List[Dict[str, str]]:
        """读取目录下的所有Markdown文件
        
        Args:
            docs_dir: Markdown文件目录，如果为None则使用默认目录
            
        Returns:
            文件内容列表，包含文件路径和内容
        """
        if docs_dir is None:
            docs_dir = self.docs_dir
            
        # 检查目录是否存在
        if not os.path.exists(docs_dir):
            print(f"\n⚠️ 目录不存在: {docs_dir}")
            print("请检查目录路径是否正确")
            return []
            
        # 获取所有 .md 文件
        md_files = glob.glob(os.path.join(docs_dir, "*.md"))
        
        if not md_files:
            print(f"\n⚠️ 目录 '{docs_dir}' 中未找到 .md 文件")
            print("请确保目录中包含 Markdown 文件")
            return []
            
        # 读取文件内容
        docs = []
        print(f"\n正在读取文件...")
        with tqdm(total=len(md_files), desc="读取文件", unit="个") as pbar:
            for md_file in md_files:
                try:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if not content.strip():
                            print(f"\n⚠️ 文件为空: {md_file}")
                            continue
                        docs.append({
                            'file_path': md_file,
                            'title': os.path.basename(md_file),
                            'content': content
                        })
                        pbar.set_postfix({"文件": os.path.basename(md_file)})
                except Exception as e:
                    print(f"\n❌ 读取文件失败: {md_file}")
                    print(f"错误信息: {str(e)}")
                pbar.update(1)
                
        return docs
        
    def create_constraints(self):
        """创建Neo4j约束"""
        with self.driver.session() as session:
            # 为每种节点类型创建唯一性约束
            for node_type in KNOWLEDGE_GRAPH_CONFIG["node_types"]:
                try:
                    # 使用更简单的约束语法
                    session.run(f"""
                    CREATE CONSTRAINT ON (n:{node_type})
                    ASSERT n.name IS UNIQUE
                    """)
                    print(f"✅ 成功创建约束: {node_type}")
                except Exception as e:
                    print(f"❌ 创建约束失败 {node_type}: {str(e)}")
                    # 如果约束已存在，继续执行
                    if "already exists" in str(e):
                        print(f"约束 {node_type} 已存在")
                    else:
                        raise e
                    
    def create_node(self, tx, label: str, name: str, properties: Dict = None):
        """创建节点
        
        Args:
            tx: 数据库事务
            label: 节点标签
            name: 节点名称
            properties: 节点属性
        """
        if properties is None:
            properties = {}
        query = (
            f"MERGE (n:{label} {{name: $name}}) "
            "SET n += $properties "
            "RETURN n"
        )
        result = tx.run(query, name=name, properties=properties)
        return result.single()[0]
        
    def create_relationship(self, tx, subject: str, subject_type: str,
                          predicate: str, object: str, object_type: str,
                          properties: Dict = None):
        """创建关系
        
        Args:
            tx: 数据库事务
            subject: 主体名称
            subject_type: 主体类型
            predicate: 关系类型
            object: 客体名称
            object_type: 客体类型
            properties: 关系属性
        """
        if properties is None:
            properties = {}
        query = (
            f"MATCH (a:{subject_type}), (b:{object_type}) "
            "WHERE a.name = $subject AND b.name = $object "
            f"MERGE (a)-[r:{predicate}]->(b) "
            "SET r += $properties "
            "RETURN type(r)"
        )
        result = tx.run(query, subject=subject, object=object, properties=properties)
        return result.single()[0]
        
    def build_knowledge_graph(self, docs_dir: str = None):
        """Build knowledge graph from documents"""
        try:
            start_time = time.time()
            
            # Create constraints
            self.create_constraints()
            
            # Read and process documents
            docs = self.read_markdown_files(docs_dir)
            total_docs = len(docs)
            
            # 检查是否有文档需要处理
            if total_docs == 0:
                print("\n⚠️ 未找到任何文档")
                print(f"请检查目录 '{docs_dir}' 是否存在并包含 .md 文件")
                return
                
            print(f"\n发现 {total_docs} 个文档需要处理")
            
            # Initialize counters
            total_triples = 0
            successful_docs = 0
            failed_docs = 0
            
            # Create progress bar for documents
            with tqdm(total=total_docs, desc="正在处理文档", unit="个") as doc_pbar:
                # Process each document
                for doc in docs:
                    try:
                        # Extract triples
                        triples = self.extractor.extract_triples(doc['content'])
                        triple_count = len(triples)
                        total_triples += triple_count
                        
                        # Update progress bar description
                        doc_pbar.set_postfix({
                            "文件": os.path.basename(doc['file_path']),
                            "三元组数": triple_count
                        })
                        
                        # Create nodes and relationships with progress bar
                        if triples:
                            with tqdm(total=triple_count, desc="正在创建三元组", unit="个", leave=False) as triple_pbar:
                                with self.driver.session() as session:
                                    for triple in triples:
                                        try:
                                            # Create subject node
                                            session.write_transaction(
                                                self.create_node,
                                                triple.subject_type,
                                                triple.subject,
                                                {'source_file': doc['file_path']}
                                            )
                                            
                                            # Create object node
                                            session.write_transaction(
                                                self.create_node,
                                                triple.object_type,
                                                triple.object,
                                                {'source_file': doc['file_path']}
                                            )
                                            
                                            # Create relationship
                                            session.write_transaction(
                                                self.create_relationship,
                                                triple.subject,
                                                triple.subject_type,
                                                triple.relation,
                                                triple.object,
                                                triple.object_type,
                                                {'source_file': doc['file_path']}
                                            )
                                            
                                            triple_pbar.update(1)
                                            
                                        except Exception as e:
                                            print(f"\n❌ 处理三元组失败: {triple}")
                                            print(f"错误信息: {str(e)}")
                                            continue
                        
                        successful_docs += 1
                        doc_pbar.update(1)
                        
                    except Exception as e:
                        print(f"\n❌ 处理文档失败: {doc['file_path']}")
                        print(f"错误信息: {str(e)}")
                        failed_docs += 1
                        doc_pbar.update(1)
                        continue
            
            # Calculate statistics
            end_time = time.time()
            total_time = end_time - start_time
            
            # Print summary
            print("\n=== 知识图谱构建统计 ===")
            print(f"总耗时: {total_time:.2f} 秒")
            print(f"处理文档数: {total_docs} 个")
            print(f"成功文档数: {successful_docs} 个")
            print(f"失败文档数: {failed_docs} 个")
            print(f"成功率: {(successful_docs / total_docs * 100):.2f}%" if total_docs > 0 else "成功率: 0.00%")
            print(f"总三元组数: {total_triples} 个")
            print(f"平均每文档三元组数: {(total_triples/total_docs):.2f} 个" if total_docs > 0 else "平均每文档三元组数: 0.00 个")
            print("\n✅ 知识图谱构建完成")
            
        except Exception as e:
            print(f"\n❌ 知识图谱构建失败: {str(e)}")
            raise

def main():
    """Main function"""
    try:
        # Configuration parameters
        api_key = "sk-jzspkglygboapudxoyefgedfrhogfflqarvzfafxouhgtpau"  # SiliconFlow API key
        model_name = "Pro/deepseek-ai/DeepSeek-V3"  # Use DeepSeek model
        neo4j_uri = "bolt://10.26.2.130:7687"  # Neo4j server address
        neo4j_user = "neo4j"
        neo4j_password = "Zq08171911"  # Neo4j password
        docs_dir = "test/docs/medical_knowledge"
        
        print("\n=== 开始构建医疗知识图谱 ===")
        print(f"Neo4j 服务器: {neo4j_uri}")
        print(f"文档目录: {docs_dir}")
        print(f"使用模型: {model_name}")
        
        # Build knowledge graph
        builder = MedicalKnowledgeGraphBuilder(
            api_key=api_key,
            model_name=model_name,
            neo4j_uri=neo4j_uri,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_password,
            docs_dir=docs_dir
        )
        
        try:
            # Test database connection
            print("\n正在测试数据库连接...")
            with builder.driver.session() as session:
                result = session.run("RETURN 1")
                result.single()
                print("✅ 数据库连接成功")
            
            # Build knowledge graph
            builder.build_knowledge_graph(docs_dir)
            print("\n✅ 知识图谱构建完成")
        except Exception as e:
            print(f"\n❌ 知识图谱构建失败: {str(e)}")
            if "Unable to retrieve routing information" in str(e):
                print("\n可能的原因:")
                print("1. Neo4j 服务器未启动")
                print("2. 服务器地址或端口不正确")
                print("3. 防火墙阻止了连接")
                print("\n建议:")
                print(f"1. 确认 Neo4j 服务在 {neo4j_uri} 运行")
                print("2. 检查服务器防火墙设置")
                print("3. 尝试使用 telnet 测试端口连通性")
        finally:
            # Close connection
            builder.close()
            
    except Exception as e:
        print(f"\n❌ 程序执行失败: {str(e)}")
        raise

if __name__ == "__main__":
    main() 