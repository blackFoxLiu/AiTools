import json
from typing import Dict, Any

from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path


class Neo4jQueryTools:
    """Neo4j 数据库查询工具类"""

    def __init__(self, uri: str="neo4j://localhost:7687", user: str="neo4j", password: str="12345678"):
        """
        初始化连接参数

        Args:
            uri: Neo4j 数据库 URI，如 "neo4j://localhost:7687"
            user: 用户名
            password: 密码
        """
        self.uri = uri
        self.user = user
        self.password = password

    @staticmethod
    def _to_json_compatible(value):
        """递归将 Neo4j 特定类型转换为 JSON 兼容的 Python 类型"""
        if value is None:
            return None
        if isinstance(value, Node):
            return {
                "element_id": value.id,
                "labels": list(value.labels),
                "properties": dict(value)
            }
        if isinstance(value, Relationship):
            return {
                "element_id": value.id,
                "type": value.type,
                "start_node_id": value.start_node.id,
                "end_node_id": value.end_node.id,
                "properties": dict(value)
            }
        if isinstance(value, Path):
            return {
                "nodes": [Neo4jQueryTools._to_json_compatible(node) for node in value.nodes],
                "relationships": [Neo4jQueryTools._to_json_compatible(rel) for rel in value.relationships]
            }
        if isinstance(value, (list, tuple)):
            return [Neo4jQueryTools._to_json_compatible(v) for v in value]
        if isinstance(value, dict):
            return {k: Neo4jQueryTools._to_json_compatible(v) for k, v in value.items()}
        # 基本类型直接返回
        return value

    def _query(self, cypher: str, parameters: dict = None, as_json: bool = True, indent: int = 2):
        """
        执行 Cypher 查询并返回结果

        Args:
            cypher: Cypher 查询语句
            parameters: 可选参数字典
            as_json: True 返回 JSON 字符串，False 返回 Python 字典列表
            indent: 当 as_json=True 时的 JSON 缩进空格数

        Returns:
            - as_json=True: str, JSON 格式的查询结果
            - as_json=False: list[dict], 每个字典对应一条记录
        """
        driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        try:
            with driver.session() as session:
                result = session.run(cypher, parameters or {})
                records = []
                for record in result:
                    rec_dict = {}
                    for key, value in record.items():
                        rec_dict[key] = self._to_json_compatible(value)
                    records.append(rec_dict)
                if as_json:
                    return json.dumps(records, indent=indent, ensure_ascii=False)
                return records
        finally:
            driver.close()

    def query(self, query_keyword: str, as_json: bool = False):
        cypher = f"""
                MATCH (n:Disease)-[r:has_symptom]->(s:Symptom)
                WHERE n.name CONTAINS "{query_keyword}"
                RETURN s.name AS 症状
                """
        return self._query(cypher, as_json=as_json)

# 使用示例
if __name__ == '__main__':
    client = Neo4jQueryTools()

    # 查询1：返回 JSON 字符串
    json_result = client.query(
        "头风",
        as_json=True
    )
    print(json_result)

    # 查询2：返回字典列表
    dict_result = client.query(
        "头风",
        as_json=False
    )
    print(dict_result)