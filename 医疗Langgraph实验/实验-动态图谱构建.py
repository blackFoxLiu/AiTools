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
    - human_body_system 人体系统
    - manifestation_characteristics 表现特征
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

# ==================== 新增 disease_analysis 节点相关函数 ====================
def check_disease_analysis_exists(name: str) -> bool:
    """检查 disease_analysis 节点是否已存在"""
    logger.info(f"调用 check_disease_analysis_exists，疾病：{name}")
    cypher = f"MATCH (n:disease_analysis {{name: '{name}'}}) RETURN count(n) as total"
    client = Neo4jQueryTools()
    result = client.use_cypher(cypher)
    return bool(int(result[0]["total"]) > 0)

def create_disease_analysis_node(data: Dict[str, Any]) -> str:
    """创建新的 disease_analysis 节点，将参数字典中的属性写入"""
    logger.info(f"调用 create_disease_analysis_node，疾病：{data['name']}")
    # 构建Cypher属性字符串，注意处理列表和字典类型（转为字符串存储，或保持原生）
    props = []
    for k, v in data.items():
        if v is None:
            continue
        # 将非字符串类型转为JSON字符串或保留表达式（简单起见转为字符串）
        if isinstance(v, (list, dict)):
            import json
            v_str = json.dumps(v, ensure_ascii=False)
        else:
            v_str = str(v).replace("'", "\\'")
        props.append(f"n.{k} = '{v_str}'")
    set_clause = ", ".join(props)
    # 使用 MERGE 保证创建时同时设置属性
    cypher = f"MERGE (n:disease_analysis {{name: '{data['name']}'}}) ON CREATE SET {set_clause} ON MATCH SET {set_clause}"
    client = Neo4jQueryTools()
    client.use_cypher(cypher)
    return f"疾病分析节点 '{data['name']}' 已存储（创建或更新）"

def update_disease_analysis_node(name: str, data: Dict[str, Any]) -> str:
    """更新已有 disease_analysis 节点的所有属性（覆盖）"""
    logger.info(f"调用 update_disease_analysis_node，疾病：{name}")
    props = []
    for k, v in data.items():
        if k == "name":  # name 不更新
            continue
        if v is None:
            continue
        if isinstance(v, (list, dict)):
            import json
            v_str = json.dumps(v, ensure_ascii=False)
        else:
            v_str = str(v).replace("'", "\\'")
        props.append(f"n.{k} = '{v_str}'")
    set_clause = ", ".join(props)
    cypher = f"MATCH (n:disease_analysis {{name: '{name}'}}) SET {set_clause}"
    client = Neo4jQueryTools()
    client.use_cypher(cypher)
    return f"疾病分析节点 '{name}' 已更新"

def process_disease_analysis(state: DynamicKnowledgeState):
    """
    将 disease_analysis 数据写入知识图谱的 disease_analysis 标签节点。
    如果节点不存在则创建，存在则更新（覆盖所有属性）。
    """
    logger.info("process_disease_analysis 进入")
    analysis = state["disease_analysis"]
    # 提取疾病名称
    disease_name = analysis.get("disease_name")
    if not disease_name:
        logger.error("disease_analysis 中缺少 disease_name 字段，无法存储")
        return {}

    # 构建要存入节点的属性字典（与图模型定义的属性对齐）
    node_data = {
        "name": disease_name,
        "chief_complaint": analysis.get("primitive_concept_symptom", {}).get("chief_complaint_list", []),
        "complications": analysis.get("complications", []),            # 若模型未提供则暂为空列表
        "treatment_plan": analysis.get("treatment_plan", ""),
        "disease_course": analysis.get("disease_course", {}).get("primary_course", ""),
        "human_body_system": analysis.get("human_body_system", {}).get("primary_system", ""),
        "manifestation_characteristics": analysis.get("manifestation_characteristics", {}).get("primary_characteristic", ""),
        "symptom_nature": analysis.get("symptom_nature", {}).get("primary_property", "")
    }

    exists = check_disease_analysis_exists(disease_name)
    if not exists:
        create_disease_analysis_node(node_data)
    else:
        update_disease_analysis_node(disease_name, node_data)

    # 返回空字典，不修改 state 其他内容（如需传递信息可设置 next_agent 等）
    return {}

# ==================== 症状间关系处理（无向边，每对只创建一条关系，权重增量0.5）====================
# 症状间关系权重步长（每次增加0.5）
SYMPTOM_SYMPTOM_WEIGHT_STEP = 0.5

def merge_symptom_relationship(symptom_a: str, symptom_b: str) -> str:
    """
    创建或更新从 symptom_a 到 symptom_b 的 :symptom 关系（无向语义，只创建一条有向边）。
    - 如果关系不存在，则创建并设置 weight = SYMPTOM_SYMPTOM_WEIGHT_STEP。
    - 如果已存在，则 weight 增加 SYMPTOM_SYMPTOM_WEIGHT_STEP。
    """
    logger.info(f"合并症状间关系：({symptom_a})-[:symptom]-({symptom_b})，权重增加 {SYMPTOM_SYMPTOM_WEIGHT_STEP}")
    cypher = f"""
    MATCH (a:basic_symptom {{name: '{symptom_a}'}}), (b:basic_symptom {{name: '{symptom_b}'}})
    MERGE (a)-[r:symptom]-(b)
    ON CREATE SET r.weight = {SYMPTOM_SYMPTOM_WEIGHT_STEP}
    ON MATCH SET r.weight = r.weight + {SYMPTOM_SYMPTOM_WEIGHT_STEP}
    """
    client = Neo4jQueryTools()
    client.use_cypher(cypher)
    return f"症状间关系 ({symptom_a})-[:symptom]-({symptom_b}) 已处理"

def process_symptom_relationships(state: DynamicKnowledgeState):
    """
    处理主诉症状列表中两两之间的 symptom 关系（无向，每对只创建一条有向边）。
    对于每一对不同的症状（i < j），创建从 symptoms[i] 到 symptoms[j] 的关系，
    权重每次调用增加 SYMPTOM_SYMPTOM_WEIGHT_STEP。
    """
    logger.info("process_symptom_relationships 进入")
    symptoms = state["disease_analysis"].get("primitive_concept_symptom", {}).get("chief_complaint_list", [])
    if len(symptoms) < 2:
        logger.info("症状数量不足2个，无需创建关系")
        return {}

    # 生成所有无序对，只创建一条有向边（从索引小的指向索引大的）
    for i in range(len(symptoms)):
        for j in range(i + 1, len(symptoms)):
            a = symptoms[i]
            b = symptoms[j]
            merge_symptom_relationship(a, b)   # 只创建 a->b 方向，视为无向
    return {}

# ==================== 症状-疾病关系处理（带权重累加）====================
# 症状-疾病关系权重步长（可独立配置）
SYMPTOM_DISEASE_WEIGHT_STEP = 1

def merge_symptom_disease_relationship(symptom_name: str, disease_name: str) -> str:
    """
    创建或更新从 symptom 节点到 disease_analysis 节点的 :include_disease 关系。
    - 如果关系不存在，则创建并设置 weight = SYMPTOM_DISEASE_WEIGHT_STEP。
    - 如果已存在，则 weight 增加 SYMPTOM_DISEASE_WEIGHT_STEP。
    方向：(symptom)-[:include_disease]->(disease)
    """
    logger.info(f"合并症状-疾病关系：({symptom_name})-[:include_disease]->({disease_name})，权重增加 {SYMPTOM_DISEASE_WEIGHT_STEP}")
    cypher = f"""
    MATCH (s:basic_symptom {{name: '{symptom_name}'}}), (d:disease_analysis {{name: '{disease_name}'}})
    MERGE (s)-[r:include_disease]->(d)
    ON CREATE SET r.weight = {SYMPTOM_DISEASE_WEIGHT_STEP}
    ON MATCH SET r.weight = r.weight + {SYMPTOM_DISEASE_WEIGHT_STEP}
    """
    client = Neo4jQueryTools()
    client.use_cypher(cypher)
    return f"症状-疾病关系 ({symptom_name})-[:include_disease]->({disease_name}) 已处理"

def process_symptom_disease_relations(state: DynamicKnowledgeState):
    """
    建立每个主诉症状与当前疾病之间的 include_disease 关系，并累加权重。
    注意：此节点应放在 process_disease_analysis 之后，确保疾病节点已存在。
    """
    logger.info("process_symptom_disease_relations 进入")
    analysis = state["disease_analysis"]
    disease_name = analysis.get("disease_name")
    if not disease_name:
        logger.error("disease_analysis 中缺少 disease_name 字段，无法建立症状-疾病关系")
        return {}

    symptoms = analysis.get("primitive_concept_symptom", {}).get("chief_complaint_list", [])
    if not symptoms:
        logger.info("无主诉症状，无需建立症状-疾病关系")
        return {}

    for symptom in symptoms:
        merge_symptom_disease_relationship(symptom, disease_name)
    return {}

# ==================== Graph 构建 ====================
if __name__ == '__main__':
    graph = StateGraph(DynamicKnowledgeState)
    graph.add_node("check_basic_symptom_node", check_basic_symptom_node)
    graph.add_node("process_symptom_relationships", process_symptom_relationships)
    graph.add_node("process_disease_analysis", process_disease_analysis)
    graph.add_node("process_symptom_disease_relations", process_symptom_disease_relations)

    graph.add_edge(START, "check_basic_symptom_node")
    graph.add_edge("check_basic_symptom_node", "process_symptom_relationships")
    graph.add_edge("process_symptom_relationships", "process_disease_analysis")
    graph.add_edge("process_disease_analysis", "process_symptom_disease_relations")
    graph.add_edge("process_symptom_disease_relations", END)

    app = graph.compile()

    # 测试数据
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