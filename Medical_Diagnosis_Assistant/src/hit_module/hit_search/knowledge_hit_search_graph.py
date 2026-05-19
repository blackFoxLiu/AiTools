import json
import logging
from typing import List, Dict, Any, Optional, Union, Annotated, TypedDict

import logger
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages


import os
import sys
# 导入独立子图构建函数
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

# 将项目根目录添加到 Python 的模块搜索路径中
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    from config.default_config import config
    from src.utils.common_utils import read_prompt, safe_json_parse
    from src.hit_module.common.knowledge_hit_analysis_subgraph import create_disease_analysis_subgraph
    from src.hit_module.hit_search.knowledge_hit_search_k_r import KnowledgeGraphQueryService
except ImportError:
    raise RuntimeError(f"导入模块失败")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SearchClassState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    high_prob_diseases_analysis: List[Dict[str, Any]]
    has_symptoms: Dict[str, Any]
    disease_severity: Dict[str, Any]
    current_analysis: Dict[str, Any]
    last_severity: Dict[str, Any]
    # 新增：存储知识图谱查询结果列表
    kg_query_results: List[Dict[str, Any]]
    solution: str                     # 可选：原有解决方案


class DiseaseDataSearchService:
    """疾病症状分析服务类（重构版，使用独立子图模块 + 知识图谱结果处理节点）"""

    PROBABILITY_THRESHOLD = 70

    def __init__(
        self,
        prompt_paths: Optional[Dict[str, str]] = None,
        probability_threshold: int = 70
    ):
        self.prompt_paths = prompt_paths or config.PROMPT_PATHS
        self.probability_threshold = probability_threshold
        self.search_data_service = KnowledgeGraphQueryService()

        # 主工作流
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """构建主工作流：疾病分析子图 -> 动态知识图谱查询 -> 结果处理节点"""
        workflow = StateGraph(SearchClassState)

        # 创建并添加独立疾病分析子图
        disease_subgraph = create_disease_analysis_subgraph(
            prompt_paths=self.prompt_paths,
            probability_threshold=self.probability_threshold
        )
        workflow.add_node("loop_over_diseases_subgraph", disease_subgraph)

        # 动态知识图谱查询节点（原逻辑，现在会返回查询结果列表）
        workflow.add_node("search_rag_and_knowledge_graph", self._search_rag_and_knowledge_graph)

        # 处理知识图谱返回数据的节点
        workflow.add_node("process_kg_results", self._process_kg_results)

        # 边连接
        workflow.add_edge(START, "loop_over_diseases_subgraph")
        workflow.add_edge("loop_over_diseases_subgraph", "search_rag_and_knowledge_graph")
        workflow.add_edge("search_rag_and_knowledge_graph", "process_kg_results")
        workflow.add_edge("process_kg_results", END)

        return workflow.compile()

    def _search_rag_and_knowledge_graph(self, state: SearchClassState) -> Dict[str, Any]:
        """
        构建动态知识图谱：调用查询服务并收集结果
        原逻辑中未使用返回值，现将其存储到状态中
        """
        high_prob_diseases_analysis_list = state.get("high_prob_diseases_analysis", [])
        kg_results = []   # 收集所有查询结果

        for high_prob_diseases_analysis in high_prob_diseases_analysis_list:
            query_result = self.search_data_service.invoke({
                "disease_analysis": high_prob_diseases_analysis,
                "has_symptoms": state.get("has_symptoms", {}),
                "disease_severity": state.get("disease_severity", {})
            })
            kg_results.append(query_result)

            # 可选：将部分查询结果直接挂载到对应疾病分析中（便于后续节点使用）
            high_prob_diseases_analysis["_kg_query_snapshot"] = {
                "disease_node_exists": query_result.get("disease_node_info", {}).get("exists", False),
                "symptom_disease_rels_count": len(query_result.get("symptom_disease_rels", [])),
                "rag_hits_count": len(query_result.get("rag_search_hits", []))
            }

        # 返回更新后的分析列表和收集的查询结果
        return {
            "high_prob_diseases_analysis": high_prob_diseases_analysis_list,
            "kg_query_results": kg_results
        }

    def _process_kg_results(self, state: SearchClassState) -> Dict[str, Any]:
        """
        处理知识图谱查询返回的数据节点，生成结构化的格式化报告。
        使用给定的打印格式输出每个疾病的详细信息，并整合 RAG 参考片段和图谱置信度。
        """
        kg_results = state.get("kg_query_results", [])
        high_prob_list = state.get("high_prob_diseases_analysis", [])

        logger.info(f"进入结果处理节点，共收到 {len(kg_results)} 个疾病的图谱查询结果")

        # 构建格式化报告字符串
        report_lines = []
        report_lines.append("===== 高概率疾病分析报告 =====")

        for idx, disease_analysis in enumerate(high_prob_list):
            # 从知识图谱结果中提取增强信息（如果存在）
            if idx < len(kg_results):
                rag_hits = kg_results[idx].get("rag_search_hits", [])
                if rag_hits:
                    disease_analysis["kg_rag_reference"] = rag_hits[:2]  # 取前两条作为参考
                    logger.debug(f"为疾病 {disease_analysis.get('disease_name')} 添加了 RAG 参考片段")
                rels = kg_results[idx].get("symptom_disease_rels", [])
                if rels:
                    avg_weight = sum(r.get("weight", 0) for r in rels) / len(rels)
                    disease_analysis["graph_confidence"] = avg_weight
                    logger.debug(f"疾病 {disease_analysis.get('disease_name')} 图谱平均权重: {avg_weight:.3f}")

            # 按照指定格式组织疾病信息
            disease_name = disease_analysis.get('disease_name', '未知疾病')
            probability = disease_analysis.get('probability', 'N/A')

            # 提取嵌套字段（安全获取）
            primitive = disease_analysis.get('primitive_concept_symptom', {})
            disease_course = disease_analysis.get('disease_course', {})
            human_body = disease_analysis.get('human_body_system', {})
            manifestation = disease_analysis.get('manifestation_characteristics', {})
            symptom_nature = disease_analysis.get('symptom_nature', {})

            # 构建单个疾病的报告块
            block = f"""
                疾病名称：{disease_name} (概率 {probability})
                  原始概念症状: {json.dumps(primitive, ensure_ascii=False, indent=2)}
                  病程: {json.dumps(disease_course, ensure_ascii=False, indent=2)}
                  人体系统: {json.dumps(human_body, ensure_ascii=False, indent=2)}
                  表现特征: {json.dumps(manifestation, ensure_ascii=False, indent=2)}
                  症状性质: {json.dumps(symptom_nature, ensure_ascii=False, indent=2)}
            """

            # 添加增强字段（如果有）
            if "kg_rag_reference" in disease_analysis:
                block += f"\n  RAG参考片段: {disease_analysis['kg_rag_reference']}"
            if "graph_confidence" in disease_analysis:
                block += f"\n  图谱置信度: {disease_analysis['graph_confidence']:.3f}"

            block += "\n" + "-" * 60
            report_lines.append(block)

        # 将所有疾病报告合并
        formatted_report = "\n".join(report_lines)
        logger.info("格式化报告已生成：\n" + formatted_report)

        # 返回状态更新（包括修改后的 high_prob_list 和新生成的报告）
        return {
            "high_prob_diseases_analysis": high_prob_list,
            "solution": formatted_report
        }

    def run(self, input_data: Union[str, Dict]) -> Dict[str, Any]:
        """
        运行分析服务

        Args:
            input_data: 输入数据，可以是 JSON 字符串或字典。
                必须包含字段：
                - chief_complaint: 主诉
                - present_illness: 现病史（含 onset_time, pattern, treatments_received）
                - associated_symptoms: 伴随症状列表
                - symptom_absent: 阴性症状列表
                - existing_knowledge_supplement: 已有知识补充列表，每个元素为包含疾病名称、症状、原因、概率的字典或JSON字符串

        Returns:
            包含 high_prob_diseases_analysis 列表和 kg_query_results 列表的字典
        """
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data, ensure_ascii=False)
        else:
            input_str = input_data
        messages = [HumanMessage(content=input_str)]
        workflow_rst = self.workflow.invoke({"messages": messages})
        rst_data = {k: v for k, v in workflow_rst.items() if k != "messages"}
        return rst_data


# ==================== 使用示例 ====================
if __name__ == "__main__":
    service = DiseaseDataSearchService()
    user_input = {
        "chief_complaint": "咳嗽，头疼",
        "present_illness": {"onset_time": "今天早上", "pattern": "", "treatments_received": []},
        "associated_symptoms": ["咳嗽", "头疼"],
        "symptom_absent": [],
        "existing_knowledge_supplement": [
            "{\n  \"疾病名称\": \"咳嗽\",\n  \"疾病存在的症状\": \"变应性咳嗽，慢性咳嗽，小儿剧烈咳嗽，发作性咳嗽，气道高反应性，湿疹，持续性咳嗽，情绪性哮喘，发绀，昏睡，意识丧失，面色苍白，抽搐，以头昏为主的眩晕，喘息，夜间咳嗽，咳出黄色痰液，咽痛，苔黄腻，遗尿，尿频，尿流变细或中断，尿急，尿失禁，吐白沫痰，痰湿体质，哮鸣音，流黄鼻涕，声音嘶哑，‘咳嗽水’上瘾，鼻塞，胸闷，口苦，呼吸困难，痉挛性咳嗽，咳铁锈色痰，咳痰，气喘\",\n  \"疾病引起原因\": \"呼吸道疾病\",\n  \"疾病确定概率\": \"60%\"\n}",
            "{\n  \"疾病名称\": \"慢性咳嗽\",\n  \"疾病存在的症状\": \"哮鸣音, 咽痛\",\n  \"疾病引起原因\": \"1. 鼻部疾病（如鼻炎、鼻窦炎）导致鼻后滴流刺激咳嗽感受器；2. 胃食管反流性咳嗽（胃酸反流刺激气道）\",\n  \"疾病确定概率\": \"75%\"\n}",
            "{\n  \"疾病名称\": \"痰浊头痛\",\n  \"疾病存在的症状\": \"胸闷、清晨或上午头痛、头昏\",\n  \"疾病引起原因\": \"中医病因：饮食不节、嗜酒太过或过食辛辣肥甘导致脾失健运，痰浊中阻，清阳不升，浊阴上蒙。西医病因：可能与血管性头痛、紧张性头痛、颅内疾病等有关。\",\n  \"疾病确定概率\": \"60%\"\n}"
        ],
        "use_message": ["今天早上头疼，有些流鼻涕。无咳嗽"]
    }
    answer = service.run(user_input)
    print("===== 整体症状分类结果 =====")
    for analysis in answer.get("high_prob_diseases_analysis", []):
        print(f"疾病名称：{analysis['disease_name']} (概率 {analysis.get('probability', 'N/A')})")
        print(f"  原始概念症状: {analysis.get('primitive_concept_symptom', {})}")
        print(f"  病程: {analysis.get('disease_course', {})}")
        print(f"  人体系统: {analysis.get('human_body_system', {})}")
        print(f"  表现特征: {analysis.get('manifestation_characteristics', {})}")
        print(f"  症状性质: {analysis.get('symptom_nature', {})}")
        # 打印知识图谱增强字段（如有）
        if "kg_rag_reference" in analysis:
            print(f"  RAG参考片段: {analysis['kg_rag_reference']}")
        if "graph_confidence" in analysis:
            print(f"  图谱置信度: {analysis['graph_confidence']:.3f}")
        print("-" * 60)

    # 也可以单独查看所有知识图谱查询原始结果
    print("\n===== 知识图谱查询原始结果 =====")
    for idx, kg_res in enumerate(answer.get("kg_query_results", [])):
        print(f"疾病 {idx+1}:")
        print(f"  疾病节点信息: {kg_res.get('disease_node_info', {})}")
        print(f"  症状-疾病关系数量: {len(kg_res.get('symptom_disease_rels', []))}")
        print(f"  RAG命中数量: {len(kg_res.get('rag_search_hits', []))}")