"""
独立脚本：使用 GDS + APOC 对医疗知识图谱进行社区划分
前提：
  - Neo4j 数据库已运行，并包含由原 MedicalGraph 代码导入的图数据
  - Neo4j 中已安装 GDS 插件（图数据科学库）和 APOC 插件
  - Python 环境已安装 neo4j 驱动：pip install neo4j
"""
import time

from neo4j import GraphDatabase
from collections import defaultdict

# Neo4j 连接配置（请根据实际情况修改）
NEO4J_URI = "neo4j://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "12345678"
GRAPH_NAME = "medicalGraph"          # 内存图投影名称
COMMUNITY_PROPERTY = "louvain_community"   # 临时写入的属性名（如使用 write 模式）

class CommunityDetector:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def run_cypher(self, query, parameters=None):
        """执行 Cypher 查询并返回结果"""
        with self.driver.session() as session:
            result = session.run(query, parameters or {})
            return list(result)

    def check_plugins(self):
        """检查 GDS 和 APOC 插件是否可用"""
        try:
            # 检查 GDS
            gds_check = self.run_cypher("RETURN gds.version() AS version")
            if gds_check and gds_check[0]["version"]:
                print(f"✅ GDS 版本: {gds_check[0]['version']}")
            else:
                print("❌ GDS 未正确安装或不可用")
                return False
            # 检查 APOC
            apoc_check = self.run_cypher("RETURN apoc.version() AS version")
            if apoc_check and apoc_check[0]["version"]:
                print(f"✅ APOC 版本: {apoc_check[0]['version']}")
            else:
                print("❌ APOC 未正确安装或不可用")
                return False
            return True
        except Exception as e:
            print(f"❌ 插件检查失败: {e}")
            return False

    def drop_graph(self):
        """删除已存在的图投影（避免冲突）"""
        self.run_cypher(f"CALL gds.graph.drop('{GRAPH_NAME}', false) YIELD graphName")

    def project_graph(self):
        """创建全局图投影（使用默认有向图，Louvain会内部处理为无向）"""
        query = f"""
        CALL gds.graph.project(
            '{GRAPH_NAME}',
            '*',
            '*'
        )
        YIELD graphName, nodeCount, relationshipCount
        RETURN graphName, nodeCount, relationshipCount
        """
        result = self.run_cypher(query)
        if result:
            print(f"✅ 图投影创建成功：{result[0]['graphName']}，节点数 {result[0]['nodeCount']}，关系数 {result[0]['relationshipCount']}")
        else:
            print("❌ 图投影创建失败")

    def run_louvain_stream(self):
        """运行 Louvain 算法，流式返回社区划分结果"""
        query = f"""
        CALL gds.louvain.stream('{GRAPH_NAME}')
        YIELD nodeId, communityId
        RETURN gds.util.asNode(nodeId) AS node, communityId
        ORDER BY communityId, node.name
        """
        results = self.run_cypher(query)
        if not results:
            print("⚠️ 社区划分结果为空，请检查图数据")
        return results

    def group_communities(self, stream_results):
        """按 communityId 分组，每个社区包含节点列表（节点名称与标签）"""
        community_map = defaultdict(list)
        for record in stream_results:
            node = record["node"]        # 字典形式，包含 id, labels, properties
            community_id = record["communityId"]
            node_name = node.get("name", node.get("id", "未命名"))
            node_labels = node.get("labels", [])
            community_map[community_id].append({
                "name": node_name,
                "labels": node_labels,
                "node_id": node.get("id")
            })
        return community_map

    def print_communities(self, community_map):
        """打印每个社区的详细信息"""
        if not community_map:
            print("无社区数据")
            return
        print(f"\n{'='*60}")
        print(f"共发现 {len(community_map)} 个社区")
        for cid, nodes in sorted(community_map.items()):
            print(f"\n📌 社区 {cid}（共 {len(nodes)} 个节点）")
            # 可选择显示前 10 个节点，避免过长（根据实际需要调整）
            for i, node in enumerate(nodes[:10], 1):
                label_str = ",".join(node["labels"])
                print(f"   {i}. [{label_str}] {node['name']}")
            if len(nodes) > 10:
                print(f"   ... 以及其他 {len(nodes)-10} 个节点")
        print(f"\n{'='*60}")

    def run(self):
        start_time = time.time()
        """执行完整流程：检查插件 -> 清理旧投影 -> 创建新投影 -> 社区划分 -> 分组打印"""
        if not self.check_plugins():
            return
        self.drop_graph()
        self.project_graph()
        stream_results = self.run_louvain_stream()
        community_map = self.group_communities(stream_results)
        self.print_communities(community_map)
        # 可选：删除内存图投影（节省资源）
        self.drop_graph()
        print("✅ 社区划分完成，内存图投影已清理")
        end_time = time.time()
        print(f"⏱️ 总耗时: {end_time - start_time:.2f} 秒")

def main():
    detector = CommunityDetector(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    try:
        detector.run()
    finally:
        detector.close()

if __name__ == "__main__":
    main()