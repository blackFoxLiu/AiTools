import json
import logging
from typing import List, Dict, Any, Union, Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph import StateGraph, START, END, add_messages

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import os
import sys
# 导入独立子图构建函数
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 将项目根目录添加到 Python 的模块搜索路径中
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    from hit_module.common.knowledge_hit_analysis_subgraph import create_disease_analysis_subgraph
    from hit_module.hit_generate.knowledge_hit_generate_k_r import KnowledgeGraphUpdateService
except ImportError:
    raise RuntimeError(f"导入模块失败")

class SearchClassState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    high_prob_diseases_analysis: List[Dict[str, Any]]
    has_symptoms: Dict[str, Any]
    disease_severity: Dict[str, Any]
    current_analysis: Dict[str, Any]
    last_severity: Dict[str, Any]

class DiseaseDataHitGenService:
    """疾病症状分析服务类（重构版，使用独立子图模块）"""

    PROBABILITY_THRESHOLD = 70

    def __init__(
        self,
        probability_threshold: int = 70
    ):
        self.probability_threshold = probability_threshold
        self.dynamic_knowledge = KnowledgeGraphUpdateService()

        # 主工作流
        self.workflow = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """构建主工作流：疾病分析子图 + 动态知识图谱构建"""
        workflow = StateGraph(SearchClassState)

        # 创建并添加独立疾病分析子图
        disease_subgraph = create_disease_analysis_subgraph(
            probability_threshold=self.probability_threshold
        )
        workflow.add_node("loop_over_diseases_subgraph", disease_subgraph)

        # 动态知识图谱构建节点
        workflow.add_node("build_dynamic_knowledge_graph", self._search_knowledge_graph)

        # 边连接
        workflow.add_edge(START, "loop_over_diseases_subgraph")
        workflow.add_edge("loop_over_diseases_subgraph", "build_dynamic_knowledge_graph")
        workflow.add_edge("build_dynamic_knowledge_graph", END)

        return workflow.compile()

    def _search_knowledge_graph(self, state: SearchClassState) -> Dict[str, Any]:
        """构建动态知识图谱（与原逻辑一致）"""
        high_prob_diseases_analysis_list = state.get("high_prob_diseases_analysis", [])
        for high_prob_diseases_analysis in high_prob_diseases_analysis_list:
            self.dynamic_knowledge.invoke({
                "disease_analysis": high_prob_diseases_analysis,
                "has_symptoms": state.get("has_symptoms", {}),
                "disease_severity": state.get("disease_severity", {})
            })
        return {}

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
            包含 high_prob_diseases_analysis 列表的字典，每个元素为疾病的详细分析结果
        """
        if isinstance(input_data, dict):
            input_str = json.dumps(input_data, ensure_ascii=False)
        else:
            input_str = input_data
        messages = [HumanMessage(content=input_str)]
        result = self.workflow.invoke({"messages": messages})
        return result


# ==================== 使用示例（与原代码相同）====================
if __name__ == "__main__":
    service = DiseaseDataHitGenService()
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
        print(f"疾病名称：{analysis['disease_name']} (概率 {analysis['probability']})")
        print(f"  原始概念症状: {analysis['primitive_concept_symptom']}")
        print(f"  病程: {analysis['disease_course']}")
        print(f"  人体系统: {analysis['human_body_system']}")
        print(f"  表现特征: {analysis['manifestation_characteristics']}")
        print(f"  症状性质: {analysis['symptom_nature']}")
        print("-" * 60)