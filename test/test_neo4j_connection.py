from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError
from config import NEO4J_CONFIG

def test_connection():
    """测试Neo4j数据库连接"""
    try:
        print(f"尝试连接到 Neo4j 数据库: {NEO4J_CONFIG['uri']}")
        
        # 创建驱动实例
        driver = GraphDatabase.driver(
            NEO4J_CONFIG["uri"],
            auth=(NEO4J_CONFIG["user"], NEO4J_CONFIG["password"])
        )
        
        # 测试连接
        with driver.session() as session:
            # 执行简单查询
            result = session.run("RETURN 1 as num")
            record = result.single()
            if record and record["num"] == 1:
                print("✅ Neo4j连接成功!")
                
                # 测试数据库状态
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                component = result.single()
                print(f"\n数据库信息:")
                print(f"- 名称: {component['name']}")
                print(f"- 版本: {component['versions']}")
                print(f"- 版本类型: {component['edition']}")
                
                # 获取现有约束
                result = session.run("SHOW CONSTRAINTS")
                constraints = list(result)
                print(f"\n现有约束数量: {len(constraints)}")
                for constraint in constraints:
                    print(f"- {constraint}")
            
    except ServiceUnavailable as e:
        print(f"❌ Neo4j服务不可用: {str(e)}")
        print("请检查:")
        print("1. Neo4j服务是否正在运行")
        print("2. 端口7687是否可访问")
        print("3. 服务器防火墙设置")
    except AuthError as e:
        print(f"❌ Neo4j认证失败: {str(e)}")
        print("请检查:")
        print("1. 用户名是否正确")
        print("2. 密码是否正确")
    except Exception as e:
        print(f"❌ Neo4j连接失败: {str(e)}")
        print(f"错误类型: {type(e).__name__}")
    finally:
        if 'driver' in locals():
            driver.close()

if __name__ == "__main__":
    test_connection() 