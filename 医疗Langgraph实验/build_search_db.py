import json
import logging
import re
from typing import List, Dict, Any, Optional, Union, Annotated
from functools import lru_cache

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages

from common_utils import read_prompt, safe_json_parse
from build_dynamic_knowledge import KnowledgeGraphUpdateService

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SearchClassState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    high_prob_diseases_analysis: List[Dict[str, Any]]
    has_symptoms: Dict[str, Any]
    disease_severity: Dict[str, Any]
    current_analysis: Dict[str, Any]
    last_severity: Dict[str, Any]


# ---------- 子图专用状态（继承自父状态，增加迭代字段）----------
class DiseaseAnalysisSubgraphState(SearchClassState):
    current_index: int                     # 当前处理的疾病索引
    diseases_list: List[Dict[str, Any]]    # 高概率疾病列表（每个元素包含疾病信息）
    disease_desc: str                      # 预生成的症状描述
    human_message: List[str]               # 用户原始会话消息


class DiseaseAnalysisService:
    """疾病症状分析服务类（使用子图重构）"""

    DEFAULT_OLLAMA_CONFIG = {
        "model": "qwen3:8b",
        "base_url": "http://127.0.0.1:11434",
        "temperature": 0.0
    }

    DEFAULT_PROMPT_PATHS = {
        "primitive_concept_sys": "./prompt/prompt_primitive_concept_sys.txt",
        "primitive_concept_user": "./prompt/prompt_primitive_concept_user.txt",
        "disease_course_sys": "./prompt/prompt_tp_disease_course_sys.txt",
        "disease_course_user": "./prompt/prompt_tp_disease_course_user.txt",
        "human_body_system_sys": "./prompt/prompt_tp_human_body_system_sys.txt",
        "human_body_system_user": "./prompt/prompt_tp_human_body_system_user.txt",
        "manifestation_characteristics_sys": "./prompt/prompt_tp_manifestation_characteristics_sys.txt",
        "manifestation_characteristics_user": "./prompt/prompt_tp_manifestation_characteristics_user.txt",
        "symptom_nature_sys": "./prompt/prompt_tp_symptom_nature_sys.txt",
        "symptom_nature_user": "./prompt/prompt_tp_symptom_nature_user.txt",
        "disease_severity_sys": "./prompt/prompt_disease_severity_sys.txt",
        "disease_severity_user": "./prompt/prompt_disease_severity_user.txt",
    }

    PROBABILITY_THRESHOLD = 70

    def __init__(
        self,
        ollama_config: Optional[Dict[str, Any]] = None,
        prompt_paths: Optional[Dict[str, str]] = None,
        probability_threshold: int = 70
    ):
        self.ollama_config = ollama_config or self.DEFAULT_OLLAMA_CONFIG
        self.prompt_paths = prompt_paths or self.DEFAULT_PROMPT_PATHS
        self.probability_threshold = probability_threshold
        self.dynamic_knowledge = KnowledgeGraphUpdateService()
        self._chat_model = None

        # 主工作流（使用子图节点）
        self.workflow = self._build_workflow()

    @lru_cache(maxsize=1)
    def _get_chat_model(self) -> ChatOllama:
        return ChatOllama(
            model=self.ollama_config["model"],
            base_url=self.ollama_config["base_url"],
            temperature=self.ollama_config["temperature"]
        )

    # ---------- 原子 LLM 调用（保持不变，供子图内部使用）----------
    def _run_generate_disease_desc(self, chat_message: list, disease_name: str) -> str:
        logger.info("=== 进入 _run_generate_disease_desc ===")
        sys_prompt = "提供一个用户对自身病症状态的信息的会话列表，需要分析出当前内容，提问的核心要素信息。以常人的角度，会这样提问。总结成一个人类对病症的描述性文段。只输入描述性结果即可。"
        user_prompt = PromptTemplate.from_template("""
            会话列表：
            {chat_message}
            疾病名称:
            {disease_name}
        """).format(chat_message=chat_message, disease_name=disease_name)
        response = self._get_chat_model().invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        logger.info("=== 退出 _run_generate_disease_desc ===")
        return response.content

    def _run_disease_severity_node(self, input_text: str) -> Dict[str, Any]:
        sys_prompt = read_prompt(self.prompt_paths["disease_severity_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["disease_severity_user"])
        ).format(input_knowledge=input_text)
        response = self._get_chat_model().invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    def _run_primitive_concept_node(self, input_text: str) -> Dict[str, Any]:
        sys_prompt = read_prompt(self.prompt_paths["primitive_concept_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["primitive_concept_user"])
        ).format(input_knowledge=input_text)
        response = self._get_chat_model().invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    def _run_disease_course_node(self, input_text: str) -> Dict[str, Any]:
        sys_prompt = read_prompt(self.prompt_paths["disease_course_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["disease_course_user"])
        ).format(input_knowledge=input_text)
        response = self._get_chat_model().invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    def _run_human_body_system_node(self, input_text: str) -> Dict[str, Any]:
        sys_prompt = read_prompt(self.prompt_paths["human_body_system_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["human_body_system_user"])
        ).format(input_knowledge=input_text)
        response = self._get_chat_model().invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    def _run_manifestation_characteristics_node(self, input_text: str) -> Dict[str, Any]:
        sys_prompt = read_prompt(self.prompt_paths["manifestation_characteristics_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["manifestation_characteristics_user"])
        ).format(input_knowledge=input_text)
        response = self._get_chat_model().invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    def _run_symptom_nature_node(self, input_text: str) -> Dict[str, Any]:
        sys_prompt = read_prompt(self.prompt_paths["symptom_nature_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["symptom_nature_user"])
        ).format(input_knowledge=input_text)
        response = self._get_chat_model().invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    # ---------- 辅助方法 ----------
    @staticmethod
    def build_symptom_description(data: Union[str, Dict]) -> str:
        if isinstance(data, str):
            json_data = safe_json_parse(data)
        else:
            json_data = data
        chief = json_data.get("chief_complaint", "")
        pi = json_data.get("present_illness", {})
        onset = pi.get("onset_time", "")
        pattern = pi.get("pattern", "")
        treatments = ", ".join(pi.get("treatments_received", [])) if pi.get("treatments_received") else "无"
        associated = ", ".join(json_data.get("associated_symptoms", [])) if json_data.get("associated_symptoms") else "无"
        absent = ", ".join(json_data.get("symptom_absent", [])) if json_data.get("symptom_absent") else "无"
        return (f"主诉：{chief}。现病史：起病时间 {onset}，病程模式 {pattern if pattern else '未描述'}，"
                f"已接受治疗：{treatments}。伴随症状：{associated}。阴性症状：{absent}。")

    # ---------- 子图内部节点 ----------
    def _parse_input_node(self, state: DiseaseAnalysisSubgraphState) -> DiseaseAnalysisSubgraphState:
        """解析原始输入，生成疾病列表、症状描述等共享数据"""
        raw_input = state["messages"][-1].content
        if isinstance(raw_input, str):
            user_data = json.loads(raw_input)
        else:
            user_data = raw_input

        disease_desc = self.build_symptom_description(user_data)
        supplement_list = user_data.get("existing_knowledge_supplement", [])
        human_message = user_data.get("use_message", [])

        # 筛选高概率疾病
        diseases = []
        for item_str in supplement_list:
            try:
                disease_info = json.loads(item_str) if isinstance(item_str, str) else item_str
            except json.JSONDecodeError:
                continue
            prob_str = disease_info.get("疾病确定概率", "0%")
            match = re.search(r"(\d+)", prob_str)
            prob = int(match.group(1)) if match else 0
            if prob > self.probability_threshold:
                diseases.append({
                    "name": disease_info.get("疾病名称", "未知疾病"),
                    "symptoms": disease_info.get("疾病存在的症状", ""),
                    "causes": disease_info.get("疾病引起原因", ""),
                    "probability": prob
                })

        return {
            "diseases_list": diseases,
            "disease_desc": disease_desc,
            "human_message": human_message,
            "current_index": 0,
            "high_prob_diseases_analysis": [],   # 初始化结果列表
            "has_symptoms": {
                "associated_symptoms": user_data.get("associated_symptoms", []),
                "symptom_absent": user_data.get("symptom_absent", [])
            },
            "disease_severity": {}   # 临时占位，将在最后一个疾病分析后更新
        }

    def _analyze_single_disease_node(self, state: DiseaseAnalysisSubgraphState) -> DiseaseAnalysisSubgraphState:
        """处理当前索引指向的单个疾病：依次调用7个 LLM 分析节点，返回当前分析结果（但不更新累积列表）"""
        idx = state["current_index"]
        disease = state["diseases_list"][idx]
        disease_name = disease["name"]
        symptoms = disease["symptoms"]
        causes = disease["causes"]
        prob = disease["probability"]

        knowledge_text = (f"疾病名称：{disease_name}\n存在的症状：{symptoms}\n引起原因：{causes}。"
                          f"{state['disease_desc']}。RAG Chunk：")

        # 顺序执行七个分析节点（每个节点内部调用 LLM）
        primitive_res = self._run_primitive_concept_node(knowledge_text)
        severity_res = self._run_disease_severity_node(knowledge_text)
        course_res = self._run_disease_course_node(knowledge_text)
        body_res = self._run_human_body_system_node(knowledge_text)
        mani_res = self._run_manifestation_characteristics_node(knowledge_text)
        nature_res = self._run_symptom_nature_node(knowledge_text)
        human_desc = self._run_generate_disease_desc(state["human_message"], disease_name)

        current_analysis = {
            "disease_name": disease_name,
            "probability": f"{prob}%",
            "primitive_concept_symptom": primitive_res,
            "disease_course": course_res,
            "human_body_system": body_res,
            "manifestation_characteristics": mani_res,
            "symptom_nature": nature_res,
            "human_desc": human_desc
        }

        # 更新疾病严重程度（原逻辑：最后一次覆盖，此处暂存）
        return {"current_analysis": current_analysis, "last_severity": severity_res}

    def _update_results_node(self, state: DiseaseAnalysisSubgraphState) -> DiseaseAnalysisSubgraphState:
        """将当前分析结果追加到 high_prob_diseases_analysis，并更新 disease_severity 为最后一次调用的结果"""
        updated_analyses = state.get("high_prob_diseases_analysis", []) + [state["current_analysis"]]
        return {
            "high_prob_diseases_analysis": updated_analyses,
            "disease_severity": state.get("last_severity", {}),
            "current_index": state["current_index"] + 1
        }

    def _should_continue(self, state: DiseaseAnalysisSubgraphState) -> str:
        """判断是否还有下一个疾病需要处理"""
        if state["current_index"] < len(state["diseases_list"]):
            return "continue"
        else:
            return "end"

    # ---------- 构建子图 ----------
    def _build_disease_analysis_subgraph(self) -> StateGraph:
        """构建独立的疾病分析子图，内部包含循环处理每个高概率疾病的流程"""
        subgraph = StateGraph(DiseaseAnalysisSubgraphState)

        # 添加节点
        subgraph.add_node("parse_input", self._parse_input_node)
        subgraph.add_node("analyze_single", self._analyze_single_disease_node)
        subgraph.add_node("update_results", self._update_results_node)

        # 设置边
        subgraph.add_edge(START, "parse_input")
        subgraph.add_edge("parse_input", "analyze_single")
        subgraph.add_edge("analyze_single", "update_results")
        # 条件边：根据是否还有疾病决定是循环回到 analyze_single 还是结束
        subgraph.add_conditional_edges(
            "update_results",
            self._should_continue,
            {
                "continue": "analyze_single",
                "end": END
            }
        )

        return subgraph.compile()

    # ---------- 主工作流（使用子图）----------
    def _build_workflow(self):
        """构建主工作流，其中 loop_over_diseases_node 替换为子图节点"""
        workflow = StateGraph(SearchClassState)

        # 将子图添加为一个节点（子图的状态是 SearchClassState 的超集，但输入输出字段兼容）
        disease_subgraph = self._build_disease_analysis_subgraph()
        workflow.add_node("loop_over_diseases_subgraph", disease_subgraph)

        # 动态知识图谱构建节点（保持不变）
        workflow.add_node("build_dynamic_knowledge_graph", self._build_dynamic_knowledge_graph)

        # 边连接
        workflow.add_edge(START, "loop_over_diseases_subgraph")
        workflow.add_edge("loop_over_diseases_subgraph", "build_dynamic_knowledge_graph")
        workflow.add_edge("build_dynamic_knowledge_graph", END)

        return workflow.compile()

    def _build_dynamic_knowledge_graph(self, state: SearchClassState):
        """构建动态知识图谱（与原逻辑一致）"""
        high_prob_diseases_analysis_list = state.get("high_prob_diseases_analysis", [])
        for high_prob_diseases_analysis in high_prob_diseases_analysis_list:
            self.dynamic_knowledge.invoke({
                "disease_analysis": high_prob_diseases_analysis,
                "has_symptoms": state.get("has_symptoms", {}),
                "disease_severity": state.get("disease_severity", {})
            })
        return {}

    # ---------- 对外运行接口 ----------
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
    service = DiseaseAnalysisService()
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