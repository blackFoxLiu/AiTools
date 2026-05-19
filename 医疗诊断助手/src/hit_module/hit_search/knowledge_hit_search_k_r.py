"""
知识图谱 + RAG 查询服务

输入格式与 KnowledgeGraphUpdateService 完全一致：
- disease_analysis: 包含疾病分析结果（disease_name, primitive_concept_symptom 等）
- has_symptoms: 包含 associated_symptoms（相关症状）和 symptom_absent（不存在症状）
- disease_severity: 疾病严重程度评估（可选）

输出：
- 从 Neo4j 中查询的疾病节点完整信息、症状节点命中次数、关系权重等
- 从 RAG 向量库中检索到的相似片段（基于疾病名称 + 症状描述）
"""

import json
import logging
import os
import sys
from functools import lru_cache
from typing import Dict, Any, List

from langchain_ollama import ChatOllama
from langgraph.constants import END, START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict

# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# 将项目根目录添加到 Python 的模块搜索路径中
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    from src.tools.knowledge_graph_tools import Neo4jQueryTools
    from src.tools.rag_module import KnowledgeBaseService
    from config.default_config import config as default_config
    from config.hit_config import config as hit_config
except ImportError:
    raise RuntimeError(f"导入模块失败")


class QueryState(TypedDict):
    """
        查询工作流状态，与更新服务的 DynamicKnowledgeState 兼容
    """
    disease_analysis: Dict[str, Any]
    has_symptoms: Dict[str, Any]
    disease_severity: Dict[str, Any]
    # 以下字段由查询节点填充
    symptom_nodes_info: Dict[str, Any]          # 症状节点信息
    disease_node_info: Dict[str, Any]           # 疾病节点信息
    symptom_relationships: List[Dict[str, Any]] # 症状间关系
    symptom_disease_rels: List[Dict[str, Any]]  # 症状->疾病关系
    rag_search_hits: List[str]                  # RAG 检索结果


class KnowledgeGraphQueryService:
    """
    知识图谱 + RAG 查询服务
    根据给定的疾病分析结果，从 Neo4j 和向量库中检索相关信息
    """

    def __init__(self):
        self._llm = self._get_base_chat_model()
        self.neo4j = Neo4jQueryTools()
        self.rag_service = KnowledgeBaseService(
            collection_name=hit_config.COLLECTION_NAME,
            persist_directory=hit_config.PERSIST_DIRCTORY
        )
        self.app = self._build_workflow()


    @lru_cache(maxsize=1)
    def _get_base_chat_model(self) -> ChatOllama:
        return ChatOllama(
            model=default_config.OLLAMA_CONFIG["model"],
            base_url=default_config.OLLAMA_CONFIG["base_url"],
            temperature=default_config.OLLAMA_CONFIG["temperature"]
        )

    # ---------- 查询节点方法 ----------
    def query_symptom_nodes(self, state: QueryState) -> Dict[str, Any]:
        """查询主诉症状中每个症状节点的完整信息（属性、命中次数）"""
        logger.info("query_symptom_nodes 进入")
        symptoms = state.get("disease_analysis", {}).get("primitive_concept_symptom", {}).get("chief_complaint_list", [])
        symptom_info = {}
        for sym in symptoms:
            # 检查症状节点是否存在
            exists = self.neo4j.node_exists("basic_symptom", sym)
            if exists:
                props = self.neo4j.get_node_info("basic_symptom", sym, as_json=False)
                symptom_info[sym] = props
            else:
                symptom_info[sym] = {"exists": False, "message": "节点不存在"}
        return {"symptom_nodes_info": symptom_info}

    def query_disease_node(self, state: QueryState) -> Dict[str, Any]:
        """查询疾病分析节点的完整属性"""
        logger.info("query_disease_node 进入")
        disease_name = state.get("disease_analysis", {}).get("disease_name")
        if not disease_name:
            logger.warning("未提供 disease_name，无法查询疾病节点")
            return {"disease_node_info": {}}
        exists = self.neo4j.node_exists("disease_analysis", disease_name)
        if exists:
            node_info = self.neo4j.get_node_info("disease_analysis", disease_name, as_json=False)
        else:
            node_info = {"exists": False, "message": "疾病分析节点不存在"}
        return {"disease_node_info": node_info}

    def query_symptom_relationships(self, state: QueryState) -> Dict[str, Any]:
        """查询主诉症状之间的 symptom 关系（权重）"""
        logger.info("query_symptom_relationships 进入")
        symptoms = state.get("disease_analysis", {}).get("primitive_concept_symptom", {}).get("chief_complaint_list", [])
        rels = []
        if len(symptoms) < 2:
            return {"symptom_relationships": rels}
        # 构建 Cypher 查询所有症状对之间的关系
        # 由于关系是无向的，我们使用 (a)-[r:symptom]-(b) 模式
        for i in range(len(symptoms)):
            for j in range(i+1, len(symptoms)):
                cypher = f"""
                MATCH (a:basic_symptom {{name: '{symptoms[i]}'}})-[r:symptom]-(b:basic_symptom {{name: '{symptoms[j]}'}})
                RETURN a.name as symptom_a, b.name as symptom_b, r.weight as weight
                """
                result = self.neo4j.use_cypher(cypher, as_json=False)
                if result:
                    rels.extend(result)
        return {"symptom_relationships": rels}

    def query_symptom_disease_relations(self, state: QueryState) -> Dict[str, Any]:
        """查询每个症状到当前疾病的 include_disease 关系（权重）"""
        logger.info("query_symptom_disease_relations 进入")
        disease_name = state.get("disease_analysis", {}).get("disease_name")
        symptoms = state.get("disease_analysis", {}).get("primitive_concept_symptom", {}).get("chief_complaint_list", [])
        rels = []
        if not disease_name or not symptoms:
            return {"symptom_disease_rels": rels}
        for sym in symptoms:
            cypher = f"""
            MATCH (s:basic_symptom {{name: '{sym}'}})-[r:include_disease]->(d:disease_analysis {{name: '{disease_name}'}})
            RETURN s.name as symptom, d.name as disease, r.weight as weight
            """
            result = self.neo4j.use_cypher(cypher, as_json=False)
            if result:
                rels.extend(result)
        return {"symptom_disease_rels": rels}

    def query_rag_hits(self, state: QueryState) -> Dict[str, Any]:
        """
        使用 RAG 检索与疾病和症状相关的历史命中片段
        构建查询文本：疾病名称 + 主诉症状 + 相关症状 + 严重程度
        """
        logger.info("query_rag_hits 进入")
        disease_name = state.get("disease_analysis", {}).get("disease_name", "")
        chief_symptoms = state.get("disease_analysis", {}).get("primitive_concept_symptom", {}).get("chief_complaint_list", [])
        associated = state.get("has_symptoms", {}).get("associated_symptoms", [])
        severity = state.get("disease_severity", {}).get("severity_level", "")

        query_parts = []
        if disease_name:
            query_parts.append(f"疾病：{disease_name}")
        if chief_symptoms:
            query_parts.append(f"主诉症状：{', '.join(chief_symptoms)}")
        if associated:
            query_parts.append(f"相关症状：{', '.join(associated)}")
        if severity:
            query_parts.append(f"严重程度：{severity}")

        query_text = "；".join(query_parts)
        if not query_text:
            return {"rag_search_hits": []}

        # 假设 KnowledgeBaseService 有 retrieve 方法，返回相似文档列表
        # 若实际接口不同请相应修改
        try:
            # 常见 RAG 服务接口：retrieve(query, top_k=5)
            hits = self.rag_service.query(query_text, top_k=3)
            # 假设 hits 是字符串列表或包含 content 字段的对象列表
            if hits and isinstance(hits[0], dict):
                hit_contents = [hit.get("text", str(hit)) for hit in hits]
            else:
                hit_contents = [str(hit) for hit in hits]
        except AttributeError:
            # 如果 retrieve 不存在，尝试 search 方法
            try:
                hits = self.rag_service.query(query_text, top_k=3)
                hit_contents = [hit.get("content", str(hit)) for hit in hits]
            except Exception as e:
                logger.error(f"RAG 检索失败: {e}")
                hit_contents = []
        except Exception as e:
            logger.error(f"RAG 检索失败: {e}")
            hit_contents = []

        return {"rag_search_hits": hit_contents}

    # ---------- 构建工作流 ----------
    def _build_workflow(self) -> StateGraph:
        graph = StateGraph(QueryState)
        graph.add_node("query_symptom_nodes", self.query_symptom_nodes)
        graph.add_node("query_disease_node", self.query_disease_node)
        graph.add_node("query_symptom_relationships", self.query_symptom_relationships)
        graph.add_node("query_symptom_disease_relations", self.query_symptom_disease_relations)
        graph.add_node("query_rag_hits", self.query_rag_hits)

        # 顺序执行所有查询节点
        graph.add_edge(START, "query_symptom_nodes")
        graph.add_edge("query_symptom_nodes", "query_disease_node")
        graph.add_edge("query_disease_node", "query_symptom_relationships")
        graph.add_edge("query_symptom_relationships", "query_symptom_disease_relations")
        graph.add_edge("query_symptom_disease_relations", "query_rag_hits")
        graph.add_edge("query_rag_hits", END)

        return graph.compile()

    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        执行查询流程

        Args:
            initial_state: 必须包含 disease_analysis 和 has_symptoms 字段

        Returns:
            包含所有查询结果的完整状态字典
        """
        if "disease_analysis" not in initial_state or "has_symptoms" not in initial_state:
            raise ValueError("initial_state 必须包含 'disease_analysis' 和 'has_symptoms' 字段")
        return self.app.invoke(initial_state)


# ==================== 使用示例（与更新服务输入完全一致）====================
if __name__ == '__main__':
    # 完全相同于 KnowledgeGraphUpdateService 的测试数据
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
        "human_desc": "咳嗽，头疼"
    }

    tmp_has_symptoms = {
        "associated_symptoms": ["发热", "鼻塞", "咳嗽"],
        "symptom_absent": ["恶心", "呕吐", "腹泻"]
    }

    query_service = KnowledgeGraphQueryService()
    result = query_service.invoke({
        "disease_analysis": tmp_dict,
        "has_symptoms": tmp_has_symptoms,
        "disease_severity": {
            "severity_level": "危重",
            "reasoning": "生命体征不稳定（昏迷、呼吸衰竭、休克），需要ICU级别的生命支持设备（呼吸机、升压药），存在多器官功能不全，死亡率高。"
        }
    })

    # 打印查询结果（美观输出）
    print("\n========== 查询结果 ==========")
    print("症状节点信息:")
    print(json.dumps(result.get("symptom_nodes_info", {}), indent=2, ensure_ascii=False))
    print("\n疾病节点信息:")
    print(json.dumps(result.get("disease_node_info", {}), indent=2, ensure_ascii=False))
    print("\n症状间关系:")
    print(json.dumps(result.get("symptom_relationships", []), indent=2, ensure_ascii=False))
    print("\n症状-疾病关系:")
    print(json.dumps(result.get("symptom_disease_rels", []), indent=2, ensure_ascii=False))
    print("\nRAG 检索命中:")
    for i, hit in enumerate(result.get("rag_search_hits", []), 1):
        print(f"{i}. {hit}")