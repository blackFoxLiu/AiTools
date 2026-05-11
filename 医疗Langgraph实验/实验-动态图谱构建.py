"""
    当前命中知识图谱存在两种标签。
    basic_symptom(Node)  原子症状
    - name(属性) 症状的原子状态，符合** 医学本体论（如 SNOMED CT、ICD-11）** 中原子性的概念。
    - hit_cnt(属性) 查询命中次数

    disease_analysis  查询疾病标签
    - name 疾病名称
    - chief_complaint 主诉症状
    - complications 并发症
    - treatment_plan 治疗方案
    - disease_course 病程
    - body_system 人体系统
    - presenting_features 表现特征
    - symptom_nature 症状性质
"""

import logging
from functools import lru_cache
from typing import TypedDict, List, Any, Dict

from langchain_core.messages import BaseMessage
from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from knowledge_graph_tools import Neo4jQueryTools

# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

neo4j_tools = Neo4jQueryTools()


class DynamicKnowledgeState(TypedDict):
    messages: List[BaseMessage]
    disease_analysis: Dict[str, Any]   # 由 supervisor 决定

OLLAMA_CONFIG = {
    "model": "qwen3:8b",
    "base_url": "http://127.0.0.1:11434",
    "temperature": 0.0
}

# ==================== 模型获取（缓存）====================
@lru_cache(maxsize=1)
def get_base_chat_model() -> ChatOllama:
    return ChatOllama(
        model=OLLAMA_CONFIG["model"],
        base_url=OLLAMA_CONFIG["base_url"],
        temperature=OLLAMA_CONFIG["temperature"]
    )

llm = get_base_chat_model()

# 定义工具1：检查症状节点是否存在
def check_symptom_exists(name: str) -> bool:
    """查询知识图谱中是否已存在指定名称的 basic_symptom 节点。返回 True 或 False。"""
    logger.info(f"调用 check_symptom_exists，名称：{name}")
    cypher = f"MATCH (n:basic_symptom {{name: '{name}'}}) RETURN count(n) as total"
    client = Neo4jQueryTools()
    result = client.use_cypher(cypher)
    # 根据实际返回结构判断是否存在，请根据你的 Neo4jQueryTools 实现调整
    return bool(int(result[0]["total"]) >0 )

# 工具2：创建新的症状节点
def create_symptom_node(name: str) -> str:
    """创建新的 basic_symptom 节点，hit_cnt 默认为 1。返回执行结果。"""
    logger.info(f"调用 create_symptom_node，名称：{name}")
    cypher = f"CREATE (n:basic_symptom {{name: '{name}', hit_cnt: 1}})"
    client = Neo4jQueryTools()
    client.use_cypher(cypher)
    return f"节点 '{name}' 创建成功，hit_cnt = 1"

# 工具3：更新已有症状节点的命中次数
def update_symptom_hit_cnt(name: str) -> str:
    """将已有 basic_symptom 节点的 hit_cnt 属性增加 1。返回执行结果。"""
    logger.info(f"调用 update_symptom_hit_cnt，名称：{name}")
    cypher = f"MATCH (n:basic_symptom {{name: '{name}'}}) SET n.hit_cnt = n.hit_cnt + 1"
    client = Neo4jQueryTools()
    client.use_cypher(cypher)
    return f"节点 '{name}' 的 hit_cnt 已增加 1"

def check_basic_symptom_node(state:DynamicKnowledgeState):
    """
        查询当前的状态结点，检查是否已经存在。
        如果结点存在就执行更新操作，更新结点中的命中次数。
    :param state:
    :return:
    """
    logger.info("check_basic_symptom_node 进入")
    # 调用 LLM，分析当前 messages，输出 next_agent
    disease_analysis:list[Any] = state["disease_analysis"]["primitive_concept_symptom"]["chief_complaint_list"]
    for disease_name in disease_analysis:
        disease_hit_cnt = check_symptom_exists(disease_name)
        if disease_hit_cnt == 0:
            # 创建结点
            create_symptom_node(disease_name)
        else:
            update_symptom_hit_cnt(disease_name)
    return {"next_agent": ""}



if __name__ == '__main__':
    graph = StateGraph(DynamicKnowledgeState)
    graph.add_node("check_basic_symptom_node", check_basic_symptom_node)

    graph.add_edge(START, "check_basic_symptom_node")
    graph.add_edge("check_basic_symptom_node", END)

    app = graph.compile()

    tmp_dict = {
  "disease_name": "急性支气管炎",
  "primitive_concept_symptom": {
    "chief_complaint_list": ["咳嗽", "发热"]
  },
  "disease_course": {
    "primary_course": "急性",
    "alternative_courses": [],
    "reasoning": "病程描述中出现“起病较急”、“病程不超过3周”等时间提示，符合急性定义。"
  },
  "human_body_system": {
    "primary_system": "呼吸系统",
    "alternative_systems": ["免疫系统"],
    "reasoning": "主要症状咳嗽、发热与呼吸道感染相关；病程中若涉及全身炎症反应，免疫系统亦可受累。"
  },
  "manifestation_characteristics": {
    "primary_characteristic": "阵发性",
    "additional_characteristics": ["湿性咳嗽", "低热"],
    "reasoning": "典型表现为阵发性咳嗽，可伴咳痰（湿性）及不超过38.5℃的发热。"
  },
  "symptom_nature": {
    "primary_property": "保护性反射症状",
    "alternative_properties": ["病理性症状"],
    "reasoning": "咳嗽是气道清除异物和分泌物的保护性反射；同时持续炎症状态下的发热可视为病理性症状。"
  }
}

    messages = app.invoke({"disease_analysis": tmp_dict})
    print(messages)