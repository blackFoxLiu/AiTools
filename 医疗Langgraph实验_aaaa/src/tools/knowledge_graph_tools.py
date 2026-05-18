import json
import logging
import os
import sys

from neo4j import GraphDatabase
from neo4j.graph import Node, Relationship, Path

# 导入独立子图构建函数
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 将项目根目录添加到 Python 的模块搜索路径中
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    from config.default_config import config
except ImportError:
    raise RuntimeError(f"导入模块失败")


# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Neo4jQueryTools:
    """Neo4j 数据库查询工具类"""

    def __init__(self, uri: str=config.NEO4J_URL, user: str=config.NEO4J_USER, password: str=config.NEO4J_PASSWORD):
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

    def query_disease(self, query_keyword: str, as_json: bool = False):
        cypher = f"""
                MATCH (n:Disease)-[r:has_symptom]->(s:Symptom)
                WHERE n.name CONTAINS "{query_keyword}"
                RETURN s.name AS 症状
                """
        return self._query(cypher, as_json=as_json)

    def query_medication_by_disease(self, query_keyword: str, as_json: bool = False):
        cypher = f"""
                MATCH (n:Disease)-[r:recommand_drug]->(d:Drug)
                WHERE n.name CONTAINS "{query_keyword}"
                RETURN d.name AS 药物名称
                """
        return self._query(cypher, as_json=as_json)

    def query_no_eat_by_disease(self, query_keyword: str, as_json: bool = False):
        cypher = f"""
                MATCH (n:Disease)-[r:recommand_drug]->(d:Drug)
                WHERE n.name CONTAINS "{query_keyword}"
                RETURN d.name AS 禁止食用食物
                """
        return self._query(cypher, as_json=as_json)

    def query_recommand_eat_by_disease(self, query_keyword: str, as_json: bool = False):
        cypher = f"""
                MATCH (n:Disease)-[r:recommand_eat]->(d:Food)
                WHERE n.name CONTAINS "{query_keyword}"
                RETURN d.name AS 可食用食物
                """
        return self._query(cypher, as_json=as_json)

    def check_disease_is_exists(self, query_keyword: str, as_json: bool = False):
        """
        查询当前疾病是否存在
        :param query_keyword:
        :param as_json:
        :return:
        """
        cypher = f"""
                MATCH(n: Disease) WHERE
                n.name contains "{query_keyword}"
                return count(n) as total
                """
        rst_json = self._query(cypher, as_json=as_json)
        return rst_json[0]["total"]

    def use_cypher(self, cypher: str, as_json: bool = False, parameters: dict = None):
        """执行一个 Cypher 语句，支持参数化查询"""
        result = None
        try:
            result = self._query(cypher, parameters=parameters, as_json=as_json)
        except Exception as e:
            logger.error(f"Cypher 执行错误。语句：{cypher}")
        return result

    def node_exists(self, label: str, name: str) -> bool:
        """
        查询节点是否存在

        Args:
            label: 节点标签（必须在白名单内）
            name: 节点 name 属性值

        Returns:
            True 如果存在至少一个匹配节点，否则 False
        """
        cypher = f"MATCH (n:{label} {{name: $name}}) RETURN n LIMIT 1"
        records = self._query(cypher, parameters={"name": name}, as_json=False)
        return len(records) > 0


    def get_node_info(self, label: str, name: str, as_json: bool = False):
        """
        查询指定标签和名称的节点的全部属性信息

        Args:
            label: 节点标签（必须在白名单内）
            name: 节点 name 属性值
            as_json: True 返回 JSON 字符串，False 返回 Python 字典

        Returns:
            as_json=True: str, JSON 格式的节点信息（包含全部属性）
            as_json=False: dict, 节点的属性字典，若节点不存在返回空字典
        """
        cypher = f"MATCH (n:{label} {{name: $name}}) RETURN n"
        records = self._query(cypher, parameters={"name": name}, as_json=False)

        if not records:
            # 节点不存在，返回空结构
            if as_json:
                return json.dumps({}, ensure_ascii=False, indent=2)
            return {}

        # records[0] 是第一条记录，其 'n' 键对应 _query 方法已转换好的 Node 字典
        node_dict = records[0].get("n")

        # 提取属性部分（_to_json_compatible 已处理好）
        properties = node_dict.get("properties", {}) if isinstance(node_dict, dict) else {}

        if as_json:
            return json.dumps(properties, ensure_ascii=False, indent=2)
        return properties


# 使用示例
if __name__ == '__main__':
    client = Neo4jQueryTools()

    # 查询1：返回 JSON 字符串
    json_result = client.check_disease_is_exists("喘息样支气管炎", False)
    print(json_result)

    cypher_code = "MERGE (d:Disease {name: '糖尿病1'}) ON CREATE SET d.desc = '代谢性疾病', d.cause = '胰岛素抵抗', d.prevent = '运动' ON MATCH SET d.desc = '代谢性疾病', d.cause = '胰岛素抵抗', d.prevent = '运动';"
    rst_ = client.use_cypher(cypher_code)
    print(rst_)