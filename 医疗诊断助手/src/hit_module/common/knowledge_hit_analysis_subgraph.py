#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立疾病分析子图模块

功能：
- 接收用户症状数据和已有疾病补充列表（含概率）
- 筛选出概率高于阈值的高概率疾病
- 对每个疾病依次调用 7 个 LLM 分析节点（原始概念、病程、人体系统、表现特征、症状性质、严重程度、人类描述）
- 以子图（LangGraph）形式循环处理所有疾病，返回完整分析结果

依赖安装：
    pip install langchain-core langgraph langchain-ollama

本模块无需依赖外部项目文件（common_utils、build_dynamic_knowledge 等），已内置辅助函数。
"""
import json
import logging
import os
import re
import sys
from typing import List, Dict, Any, Optional, Annotated

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import TypedDict

# ---------- 路径设置 ----------
# 当前文件位于 src/discovery/ 或 src/hit_module/common/ 等二级目录下
# 向上两级到达项目根目录（即 src/ 的父目录）
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------- 导入项目内部模块 ----------
try:
    from src.utils.common_utils import get_base_chat_model, read_prompt, safe_json_parse
    from config.default_config import config as default_config
    from config.hit_config import config as hit_config
except ImportError as e:
    raise RuntimeError(f"导入模块失败: {e}") from e


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ========== 1. 状态定义 ==========
class DiseaseAnalysisSubgraphState(TypedDict):
    """子图内部状态（继承自父状态，增加迭代字段）"""
    messages: Annotated[List[BaseMessage], add_messages]   # 外部传入的原始消息（含用户输入）
    high_prob_diseases_analysis: List[Dict[str, Any]]      # 最终输出的疾病分析列表
    has_symptoms: Dict[str, Any]                           # 用户伴随/阴性症状
    disease_severity: Dict[str, Any]                       # 最后一个疾病的严重程度分析结果
    current_analysis: Dict[str, Any]                       # 当前正在处理的疾病分析结果
    last_severity: Dict[str, Any]                          # 当前疾病的严重程度（临时）
    current_index: int                                     # 当前处理的疾病索引
    diseases_list: List[Dict[str, Any]]                    # 高概率疾病列表
    disease_desc: str                                      # 预生成的症状描述文本
    human_message: List[str]                               # 用户原始会话消息


# ========== 3. 子图构建器类 ==========
class DiseaseAnalysisSubgraphBuilder:
    """
    疾病分析子图构建器

    封装了：
    - 解析输入 → 高概率疾病列表
    - 循环单疾病分析（7 个原子 LLM 节点）
    - 结果累积与迭代控制
    """

    def __init__(
        self,
        prompt_paths: Optional[Dict[str, str]] = None,
        probability_threshold: int = hit_config.PROBABILITY_THRESHOLD
    ):
        self.prompt_paths = prompt_paths or default_config.PROMPT_PATHS
        self.probability_threshold = probability_threshold

        self._chat_model = ChatOllama(
            model=default_config.OLLAMA_CONFIG["model"],
            base_url=default_config.OLLAMA_CONFIG["base_url"],
            temperature=default_config.OLLAMA_CONFIG["temperature"]
        )

    # ---------- 原子 LLM 调用节点 ----------
    def _run_generate_disease_desc(self, chat_messages: List[str], disease_name: str) -> str:
        """根据用户会话和疾病名称生成人类风格的疾病描述"""
        logger.info(f"生成疾病描述: {disease_name}")
        sys_prompt = "提供一个用户对自身病症状态的信息的会话列表，需要分析出当前内容，提问的核心要素信息。以常人的角度，会这样提问。总结成一个人类对病症的描述性文段。只输入描述性结果即可。"
        user_prompt = PromptTemplate.from_template("""
            会话列表：
            {chat_message}
            疾病名称:
            {disease_name}
        """).format(chat_message=chat_messages, disease_name=disease_name)
        response = self._chat_model.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content

    def _run_disease_severity_node(self, input_text: str) -> Dict[str, Any]:
        """严重程度分析"""
        sys_prompt = read_prompt(self.prompt_paths["disease_severity_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["disease_severity_user"])
        ).format(input_knowledge=input_text)
        response = self._chat_model.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    def _run_primitive_concept_node(self, input_text: str) -> Dict[str, Any]:
        """原始概念症状分析"""
        sys_prompt = read_prompt(self.prompt_paths["primitive_concept_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["primitive_concept_user"])
        ).format(input_knowledge=input_text)
        response = self._chat_model.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    def _run_disease_course_node(self, input_text: str) -> Dict[str, Any]:
        """病程分析"""
        sys_prompt = read_prompt(self.prompt_paths["disease_course_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["disease_course_user"])
        ).format(input_knowledge=input_text)
        response = self._chat_model.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    def _run_human_body_system_node(self, input_text: str) -> Dict[str, Any]:
        """人体系统累及分析"""
        sys_prompt = read_prompt(self.prompt_paths["human_body_system_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["human_body_system_user"])
        ).format(input_knowledge=input_text)
        response = self._chat_model.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    def _run_manifestation_characteristics_node(self, input_text: str) -> Dict[str, Any]:
        """表现特征分析"""
        sys_prompt = read_prompt(self.prompt_paths["manifestation_characteristics_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["manifestation_characteristics_user"])
        ).format(input_knowledge=input_text)
        response = self._chat_model.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    def _run_symptom_nature_node(self, input_text: str) -> Dict[str, Any]:
        """症状性质分析"""
        sys_prompt = read_prompt(self.prompt_paths["symptom_nature_sys"])
        user_prompt = PromptTemplate.from_template(
            read_prompt(self.prompt_paths["symptom_nature_user"])
        ).format(input_knowledge=input_text)
        response = self._chat_model.invoke([
            SystemMessage(content=sys_prompt),
            HumanMessage(content=user_prompt)
        ])
        return safe_json_parse(response.content)

    # ---------- 辅助方法 ----------
    @staticmethod
    def build_symptom_description(data: Dict[str, Any]) -> str:
        """从用户输入 JSON 中构建标准症状描述文本"""
        chief = data.get("chief_complaint", "")
        pi = data.get("present_illness", {})
        onset = pi.get("onset_time", "")
        pattern = pi.get("pattern", "")
        treatments = ", ".join(pi.get("treatments_received", [])) if pi.get("treatments_received") else "无"
        associated = ", ".join(data.get("associated_symptoms", [])) if data.get("associated_symptoms") else "无"
        absent = ", ".join(data.get("symptom_absent", [])) if data.get("symptom_absent") else "无"
        return (f"主诉：{chief}。现病史：起病时间 {onset}，病程模式 {pattern if pattern else '未描述'}，"
                f"已接受治疗：{treatments}。伴随症状：{associated}。阴性症状：{absent}。")

    # ---------- 子图节点实现 ----------
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

        # 筛选高概率疾病（概率 > 阈值）
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
            "high_prob_diseases_analysis": [],
            "has_symptoms": {
                "associated_symptoms": user_data.get("associated_symptoms", []),
                "symptom_absent": user_data.get("symptom_absent", [])
            },
            "disease_severity": {}
        }

    @staticmethod
    def get_disease_info(state: DiseaseAnalysisSubgraphState):
        """获取疾病信息（安全版本）"""
        # 1. 获取必要的状态字段
        diseases = state.get("diseases_list", [])
        idx = state.get("current_index", 0)  # 使用 current_index

        # 2. 边界检查（先检查再访问）
        if not diseases or (idx < 0 or idx >= len(diseases)):
            logger.warning("diseases_list 为空，使用占位疾病继续分析")
            diseases = [{"name": "未知疾病", "probability": 0}]
            idx = 0

        # 3. 安全获取疾病数据
        disease = diseases[idx]
        disease_name = disease.get("name", "未知疾病")
        symptoms = disease.get("symptoms", "无")
        causes = disease.get("causes", "未知")

        # 4. 构建返回信息（注意 disease_desc 可能缺失）
        disease_desc = state.get("disease_desc", "")

        info_text = (f"疾病名称：{disease_name}\n"
                     f"存在的症状：{symptoms}\n"
                     f"引起原因：{causes}。"
                     f"{disease_desc}。RAG Chunk：")

        # 正常情况返回带有疾病信息的 current_disease_info
        return {"current_disease_info": info_text}

    def _analyze_single_disease_node(self, state: DiseaseAnalysisSubgraphState) -> dict:
        diseases = state.get("diseases_list", [])
        idx = state.get("current_index", 0)
        human_message = state.get("human_message", "")

        # 处理空列表：构造一个占位疾病，让流程继续
        if not diseases or (idx < 0 or idx >= len(diseases)):
            logger.warning("diseases_list 为空，使用占位疾病继续分析")
            diseases = [{"name": "未知疾病", "probability": 0}]
            idx = 0

        disease = diseases[idx]
        disease_name = disease.get("name", "未知疾病")
        prob = disease.get("probability", 0)

        knowledge_text = self.get_disease_info(state)  # 确保该方法能处理 "未知疾病"

        # 并行/顺序调用分析节点（保持不变）
        primitive_res = self._run_primitive_concept_node(knowledge_text)
        severity_res = self._run_disease_severity_node(knowledge_text)
        course_res = self._run_disease_course_node(knowledge_text)
        body_res = self._run_human_body_system_node(knowledge_text)
        mani_res = self._run_manifestation_characteristics_node( knowledge_text)
        nature_res = self._run_symptom_nature_node(knowledge_text)

        human_desc = ""
        if disease_name and human_message:
            human_desc = self._run_generate_disease_desc(human_message, disease_name)
        # 可选：即使 disease_name 为 "未知疾病"，也可以生成描述（去掉 disease_name 条件或传 None）
        # 若希望无条件生成描述，可修改条件为 if human_message:

        current_analysis = {
            "disease_name": disease_name,
            "probability": f"{prob}%",
            "primitive_concept_symptom": primitive_res,
            "disease_course": course_res,
            "human_body_system": body_res,
            "manifestation_characteristics": mani_res,
            "symptom_nature": nature_res,
            "human_desc": human_desc,
        }

        return {
            "current_analysis": current_analysis,
            "last_severity": severity_res,
        }

    def _update_results_node(self, state: DiseaseAnalysisSubgraphState) -> DiseaseAnalysisSubgraphState:
        """将当前分析结果追加到结果列表，并更新疾病严重程度为最后一次调用的结果"""
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

    # ---------- 子图构建 ----------
    def build(self) -> StateGraph:
        """构建并编译完整的疾病分析子图"""
        subgraph = StateGraph(DiseaseAnalysisSubgraphState)

        subgraph.add_node("parse_input", self._parse_input_node)
        subgraph.add_node("analyze_single", self._analyze_single_disease_node)
        subgraph.add_node("update_results", self._update_results_node)

        subgraph.add_edge(START, "parse_input")
        subgraph.add_edge("parse_input", "analyze_single")
        subgraph.add_edge("analyze_single", "update_results")
        subgraph.add_conditional_edges(
            "update_results",
            self._should_continue,
            {
                "continue": "analyze_single",
                "end": END
            }
        )

        return subgraph.compile()


# ========== 4. 便捷函数 ==========
def create_disease_analysis_subgraph(
    prompt_paths: Optional[Dict[str, str]] = None,
    probability_threshold: int = 70
) -> StateGraph:
    """
    创建并返回一个编译好的疾病分析子图

    参数：
        ollama_config: Ollama 配置，如 {"model": "qwen3:8b", "base_url": "http://127.0.0.1:11434", "temperature": 0.0}
        prompt_paths: 各提示词文件路径字典，格式见 DEFAULT_PROMPT_PATHS
        probability_threshold: 疾病概率阈值（%），大于该值的疾病才被分析

    返回：
        编译后的 LangGraph StateGraph 实例，可直接 invoke
    """
    builder = DiseaseAnalysisSubgraphBuilder(
        prompt_paths=prompt_paths,
        probability_threshold=probability_threshold
    )
    return builder.build()


# ========== 5. 使用示例 ==========
if __name__ == "__main__":
    # 准备示例输入数据（格式与原始服务一致）
    test_input = {
        "chief_complaint": "咳嗽，头疼",
        "present_illness": {"onset_time": "今天早上", "pattern": "", "treatments_received": []},
        "associated_symptoms": ["咳嗽", "头疼"],
        "symptom_absent": [],
        "existing_knowledge_supplement": [
            "{\"疾病名称\": \"咳嗽\", \"疾病存在的症状\": \"咳嗽,咽痛\", \"疾病引起原因\": \"呼吸道感染\", \"疾病确定概率\": \"60%\"}",
            "{\"疾病名称\": \"慢性咳嗽\", \"疾病存在的症状\": \"哮鸣音,咽痛\", \"疾病引起原因\": \"鼻后滴流\", \"疾病确定概率\": \"75%\"}",
            "{\"疾病名称\": \"痰浊头痛\", \"疾病存在的症状\": \"胸闷,头昏\", \"疾病引起原因\": \"痰浊中阻\", \"疾病确定概率\": \"60%\"}"
        ],
        "use_message": ["今天早上头疼，有些流鼻涕。无咳嗽"]
    }

    # 构建子图（使用默认配置）
    subgraph = create_disease_analysis_subgraph(probability_threshold=70)

    # 运行子图（必须包装成 messages 列表）
    initial_state = {
        "messages": [HumanMessage(content=json.dumps(test_input, ensure_ascii=False))]
    }
    result = subgraph.invoke(initial_state)

    # 输出结果
    print("===== 疾病分析结果 =====")
    for analysis in result.get("high_prob_diseases_analysis", []):
        print(f"疾病名称：{analysis['disease_name']} (概率 {analysis['probability']})")
        print(f"  原始概念症状: {analysis['primitive_concept_symptom']}")
        print(f"  病程: {analysis['disease_course']}")
        print(f"  人体系统: {analysis['human_body_system']}")
        print(f"  表现特征: {analysis['manifestation_characteristics']}")
        print(f"  症状性质: {analysis['symptom_nature']}")
        print(f"  人类描述: {analysis['human_desc'][:100]}...")
        print("-" * 60)
    print(f"最终严重程度汇总: {result.get('disease_severity', {})}")