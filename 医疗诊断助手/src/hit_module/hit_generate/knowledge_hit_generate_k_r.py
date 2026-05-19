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
    - human_desc 人类提问描述
"""
import json
import logging
import os
import sys
from functools import lru_cache
from typing import TypedDict, Any, Dict

from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph

# ==================== 日志配置（保留原有方式）====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# 将项目根目录添加到 Python 的模块搜索路径中
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    from src.tools.knowledge_graph_tools import Neo4jQueryTools
    from src.tools.rag_module import KnowledgeBaseService
    from config.hit_config import config as hit_config
    from config.default_config import config as default_config
except ImportError:
    raise RuntimeError(f"导入模块失败")


class DynamicKnowledgeState(TypedDict):
    disease_analysis: Dict[str, Any]
    has_symptoms: Dict[str, Any]
    disease_severity: Dict[str, Any]


class KnowledgeGraphUpdateService:
    """
    知识图谱更新服务类
    负责将疾病分析结果写入 Neo4j，包括：
    - basic_symptom 节点的创建/命中次数更新
    - disease_analysis 节点的创建/全量更新
    - 症状间关系（无向，权重累加 0.5）
    - 症状->疾病关系（有向，权重累加 1）
    """

    # 关系权重步长
    SYMPTOM_SYMPTOM_WEIGHT_STEP = 0.5
    SYMPTOM_DISEASE_WEIGHT_STEP = 1

    def __init__(self):
        """初始化服务，构建工作流图"""
        # 保留原有模型获取（尽管未在图节点中使用，但为保持代码完整性）
        self._llm = self._get_base_chat_model()
        # 构建并编译工作流
        self.app = self._build_workflow()
        self.rag_service = KnowledgeBaseService(
            collection_name=hit_config.COLLECTION_NAME,
            persist_directory=hit_config.PERSIST_DIRCTORY
        )

    @lru_cache(maxsize=1)
    def _get_base_chat_model(self) -> ChatOllama:
        """获取 Ollama 模型实例（缓存）"""
        return ChatOllama(
            model=default_config.OLLAMA_CONFIG["model"],
            base_url=default_config.OLLAMA_CONFIG["base_url"],
            temperature=default_config.OLLAMA_CONFIG["temperature"]
        )

    # ---------- 原有工具函数（封装为实例方法）----------
    def check_symptom_exists(self, name: str) -> bool:
        """查询知识图谱中是否已存在指定名称的 basic_symptom 节点。返回 True 或 False。"""
        logger.info(f"调用 check_symptom_exists，名称：{name}")
        cypher = f"MATCH (n:basic_symptom {{name: '{name}'}}) RETURN count(n) as total"
        client = Neo4jQueryTools()
        result = client.use_cypher(cypher)
        if result is None:
            return False
        return bool(int(result[0]["total"]) > 0)

    def create_symptom_node(self, name: str) -> str:
        """创建新的 basic_symptom 节点，hit_cnt 默认为 1。返回执行结果。"""
        logger.info(f"调用 create_symptom_node，名称：{name}")
        cypher = f"CREATE (n:basic_symptom {{name: '{name}', hit_cnt: 1}})"
        client = Neo4jQueryTools()
        client.use_cypher(cypher)
        return f"节点 '{name}' 创建成功，hit_cnt = 1"

    def update_symptom_hit_cnt(self, name: str) -> str:
        """将已有 basic_symptom 节点的 hit_cnt 属性增加 1。返回执行结果。"""
        logger.info(f"调用 update_symptom_hit_cnt，名称：{name}")
        cypher = f"MATCH (n:basic_symptom {{name: '{name}'}}) SET n.hit_cnt = n.hit_cnt + 1"
        client = Neo4jQueryTools()
        client.use_cypher(cypher)
        return f"节点 '{name}' 的 hit_cnt 已增加 1"

    def check_disease_analysis_exists(self, name: str) -> bool:
        """检查 disease_analysis 节点是否已存在"""
        logger.info(f"调用 check_disease_analysis_exists，疾病：{name}")
        cypher = f"MATCH (n:disease_analysis {{name: '{name}'}}) RETURN count(n) as total"
        client = Neo4jQueryTools()
        result = client.use_cypher(cypher)
        if result is None:
            return False
        return bool(int(result[0]["total"]) > 0)

    def create_disease_analysis_node(self, data: Dict[str, Any]) -> str:
        """创建新的 disease_analysis 节点，将参数字典中的属性写入"""
        logger.info(f"调用 create_disease_analysis_node，疾病：{data['name']}")
        props = []
        for k, v in data.items():
            if v is None:
                continue
            if isinstance(v, (list, dict)):
                v_str = json.dumps(v, ensure_ascii=False)
            else:
                v_str = str(v).replace("'", "\\'")
            props.append(f"n.{k} = '{v_str}'")
        set_clause = ", ".join(props)
        cypher = f"MERGE (n:disease_analysis {{name: '{data['name']}'}}) ON CREATE SET {set_clause} ON MATCH SET {set_clause}"
        client = Neo4jQueryTools()
        client.use_cypher(cypher)
        return f"疾病分析节点 '{data['name']}' 已存储（创建或更新）"

    def update_disease_analysis_node(self, name: str, data: Dict[str, Any]) -> str:
        """更新已有 disease_analysis 节点的所有属性（覆盖）"""
        logger.info(f"调用 update_disease_analysis_node，疾病：{name}")
        props = []
        for k, v in data.items():
            if k == "name":
                continue
            if v is None:
                continue
            if isinstance(v, (list, dict)):
                v_str = json.dumps(v, ensure_ascii=False)
            else:
                v_str = str(v).replace("'", "\\'")
            props.append(f"n.{k} = '{v_str}'")
        set_clause = ", ".join(props)
        cypher = f"MATCH (n:disease_analysis {{name: '{name}'}}) SET {set_clause}"
        client = Neo4jQueryTools()
        client.use_cypher(cypher)
        return f"疾病分析节点 '{name}' 已更新"

    def merge_symptom_relationship(self, symptom_a: str, symptom_b: str) -> str:
        """创建或更新症状间关系（无向，权重累加）"""
        logger.info(f"合并症状间关系：({symptom_a})-[:symptom]-({symptom_b})，权重增加 {self.SYMPTOM_SYMPTOM_WEIGHT_STEP}")
        cypher = f"""
        MATCH (a:basic_symptom {{name: '{symptom_a}'}}), (b:basic_symptom {{name: '{symptom_b}'}})
        MERGE (a)-[r:symptom]-(b)
        ON CREATE SET r.weight = {self.SYMPTOM_SYMPTOM_WEIGHT_STEP}
        ON MATCH SET r.weight = r.weight + {self.SYMPTOM_SYMPTOM_WEIGHT_STEP}
        """
        client = Neo4jQueryTools()
        client.use_cypher(cypher)
        return f"症状间关系 ({symptom_a})-[:symptom]-({symptom_b}) 已处理"

    def merge_symptom_disease_relationship(self, symptom_name: str, disease_name: str) -> str:
        """创建或更新症状->疾病关系（有向，权重累加）"""
        logger.info(f"合并症状-疾病关系：({symptom_name})-[:include_disease]->({disease_name})，权重增加 {self.SYMPTOM_DISEASE_WEIGHT_STEP}")
        cypher = f"""
        MATCH (s:basic_symptom {{name: '{symptom_name}'}}), (d:disease_analysis {{name: '{disease_name}'}})
        MERGE (s)-[r:include_disease]->(d)
        ON CREATE SET r.weight = {self.SYMPTOM_DISEASE_WEIGHT_STEP}
        ON MATCH SET r.weight = r.weight + {self.SYMPTOM_DISEASE_WEIGHT_STEP}
        """
        client = Neo4jQueryTools()
        client.use_cypher(cypher)
        return f"症状-疾病关系 ({symptom_name})-[:include_disease]->({disease_name}) 已处理"

    # ---------- LangGraph 节点方法（实例方法，签名接收 state）----------
    def check_basic_symptom_node(self, state: DynamicKnowledgeState):
        """检查并更新/创建基础症状节点"""
        logger.info("check_basic_symptom_node 进入")
        disease_analysis: list[Any] = state.get("disease_analysis", {}).get("primitive_concept_symptom", {}).get("chief_complaint_list", [])
        for disease_name in disease_analysis:
            exists = self.check_symptom_exists(disease_name)
            if not exists:
                self.create_symptom_node(disease_name)
            else:
                self.update_symptom_hit_cnt(disease_name)
        return {}

    def process_symptom_relationships(self, state: DynamicKnowledgeState):
        """处理症状间关系（两两之间）"""
        logger.info("process_symptom_relationships 进入")
        symptoms = state["disease_analysis"].get("primitive_concept_symptom", {}).get("chief_complaint_list", [])
        if len(symptoms) < 2:
            logger.info("症状数量不足2个，无需创建关系")
            return {}
        for i in range(len(symptoms)):
            for j in range(i + 1, len(symptoms)):
                self.merge_symptom_relationship(symptoms[i], symptoms[j])
        return {}

    def process_disease_analysis(self, state: DynamicKnowledgeState):
        """将疾病分析数据写入 disease_analysis 节点（创建或更新）"""
        logger.info("process_disease_analysis 进入")
        analysis = state["disease_analysis"]
        disease_name = analysis.get("disease_name")
        if not disease_name:
            logger.error("disease_analysis 中缺少 disease_name 字段，无法存储")
            return {}

        node_data = {
            "name": disease_name,
            "chief_complaint": analysis.get("primitive_concept_symptom", {}).get("chief_complaint_list", []),
            "complications": analysis.get("complications", []),
            "treatment_plan": analysis.get("treatment_plan", ""),
            "disease_course": analysis.get("disease_course", {}).get("primary_course", ""),
            "human_body_system": analysis.get("human_body_system", {}).get("primary_system", ""),
            "manifestation_characteristics": analysis.get("manifestation_characteristics", {}).get("primary_characteristic", ""),
            "symptom_nature": analysis.get("symptom_nature", {}).get("primary_property", ""),
            "human_desc": analysis.get("human_desc", {})
        }

        exists = self.check_disease_analysis_exists(disease_name)
        if not exists:
            self.create_disease_analysis_node(node_data)
        else:
            self.update_disease_analysis_node(disease_name, node_data)
        return {}

    def process_symptom_disease_relations(self, state: DynamicKnowledgeState):
        """建立每个主诉症状与当前疾病之间的 include_disease 关系"""
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
            self.merge_symptom_disease_relationship(symptom, disease_name)
        return {}

    def generate_rag_search_hit(self, state: DynamicKnowledgeState):
        """
            设置 RAG 命中语义设计，症状命中（原函数未完成，保留逻辑不变）
        """
        chief_complaint_list = (
            state.get("disease_analysis", {})
            .get("primitive_concept_symptom", {})
            .get("chief_complaint_list", [])
        )
        rag_rst_list = []
        disease_severity_level = state.get("disease_severity", {}).get("severity_level", "")
        if disease_severity_level:
            rag_rst_list.append(f"疾病严重程度：{disease_severity_level}")

        if chief_complaint_list:
            rag_rst_list.append("符合医学本体论原子症状包括：" + "、".join(chief_complaint_list))

        associated_symptoms_list = state["has_symptoms"]["associated_symptoms"]
        if associated_symptoms_list:
            rag_rst_list.append("相关症状包括：" + "、".join(associated_symptoms_list))

        symptom_absent_list = state["has_symptoms"]["symptom_absent"]
        if associated_symptoms_list:   # 原代码此处判断有误，但保留原样
            rag_rst_list.append("不存在症状包括：" + "、".join(symptom_absent_list))

        disease_name = state["disease_analysis"]["disease_name"]
        rag_symptom = "。".join(rag_rst_list)
        if disease_name and rag_symptom:
            rag_rst_date = disease_name + "=>" + rag_symptom
            self.rag_service.upload_by_str(rag_rst_date, "search_hit_data")

        if state["disease_analysis"]["human_desc"]:
            rag_rst_date = disease_name + "=>" + state["disease_analysis"]["human_desc"]
            self.rag_service.upload_by_str(rag_rst_date, "search_intention_data")
        return {}

    # ---------- 构建工作流 ----------
    def _build_workflow(self):
        """构建 LangGraph 工作流（与原有顺序完全一致）"""
        graph = StateGraph(DynamicKnowledgeState)
        graph.add_node("check_basic_symptom_node", self.check_basic_symptom_node)
        graph.add_node("process_symptom_relationships", self.process_symptom_relationships)
        graph.add_node("process_disease_analysis", self.process_disease_analysis)
        graph.add_node("process_symptom_disease_relations", self.process_symptom_disease_relations)
        graph.add_node("generate_rag_search_hit", self.generate_rag_search_hit)

        graph.add_edge(START, "check_basic_symptom_node")
        graph.add_edge("check_basic_symptom_node", "process_symptom_relationships")
        graph.add_edge("process_symptom_relationships", "process_disease_analysis")
        graph.add_edge("process_disease_analysis", "process_symptom_disease_relations")
        graph.add_edge("process_symptom_disease_relations", "generate_rag_search_hit")
        graph.add_edge("generate_rag_search_hit", END)

        return graph.compile()

    # ---------- 对外运行接口 ----------
    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行知识图谱更新流程

        Args:
            initial_state: 初始状态，必须包含 "disease_analysis" 和 "has_symptoms" 字段，
                           结构与 DynamicKnowledgeState 一致。

        Returns:
            工作流最终状态字典
        """
        if "disease_analysis" not in initial_state or "has_symptoms" not in initial_state:
            raise ValueError("initial_state 必须包含 'disease_analysis' 和 'has_symptoms' 字段")
        return self.app.invoke(initial_state)


# ==================== 使用示例（与原测试逻辑完全一致）====================
if __name__ == '__main__':
    # 实例化服务
    service = KnowledgeGraphUpdateService()

    # 测试数据（与原代码完全相同）
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
        },
        "human_desc":"咳嗽，头疼"
    }

    tmp_has_symptoms = {
        "associated_symptoms": ["发热", "鼻塞", "咳嗽"],
        "symptom_absent": ["恶心", "呕吐", "腹泻"]
    }

    result = service.invoke({
        "disease_analysis": tmp_dict,
        "has_symptoms": tmp_has_symptoms,
        "disease_severity": {
            "severity_level": "危重",
            "reasoning": "生命体征不稳定（昏迷、呼吸衰竭、休克），需要ICU级别的生命支持设备（呼吸机、升压药），存在多器官功能不全，死亡率高。"
        }
    })
    print(result)