import json
from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path


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
            "nodes": [_to_json_compatible(node) for node in value.nodes],
            "relationships": [_to_json_compatible(rel) for rel in value.relationships]
        }
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_json_compatible(v) for k, v in value.items()}
    # 其他基本类型 (str, int, float, bool) 或日期时间，直接返回
    # 若需要支持 datetime 可在此添加处理
    return value

def run_cypher_query(uri, user, password, query, parameters=None, as_json=True, indent=2):
    """
    执行 Cypher 查询，返回 JSON 字符串或字典列表。

    Args:
        uri, user, password: Neo4j 连接信息
        query: Cypher 查询语句
        parameters: 可选参数字典
        as_json: True 返回 JSON 字符串，False 返回 Python 字典列表
        indent: 当 as_json=True 时 JSON 缩进空格数

    Returns:
        - as_json=True: str, JSON 格式的查询结果
        - as_json=False: list[dict], 每个字典对应一条记录
    """
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            result = session.run(query, parameters or {})
            records = []
            for record in result:
                rec_dict = {}
                for key, value in record.items():
                    rec_dict[key] = _to_json_compatible(value)
                records.append(rec_dict)
            if as_json:
                return json.dumps(records, indent=indent, ensure_ascii=False)
            return records
    finally:
        driver.close()



if __name__ == '__main__':


    # 示例1：返回 JSON 字符串
    json_result = run_cypher_query(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="12345678",
        query="""
        MATCH (n:Disease)-[r:has_symptom]->(s:Symptom)
    WHERE n.name CONTAINS "头风"
    RETURN s.name as 症状
        """,
        as_json=True
    )



    print(json_result)

    # 示例2：返回字典列表（便于程序内处理）
    dict_result = run_cypher_query(
        uri="neo4j://localhost:7687",
        user="neo4j",
        password="12345678",
        query="""
        MATCH (n:Disease)-[r:has_symptom]->(s:Symptom)
    WHERE n.name CONTAINS "头风"
    RETURN s.name as 相关症状
        """,
        as_json=False
    )
    print(dict_result)