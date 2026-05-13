"""
医疗信息探寻查询图结构
"""

import json
import logging
from typing import Dict, Any, List, Annotated, TypedDict, Literal

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from build_search_db import DiseaseAnalysisService
from knowledge_graph_tools import Neo4jQueryTools
from rag_module import KnowledgeBaseService

from common_utils import read_prompt, safe_json_parse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMPT_PATHS = {
    "sufficiency_decision_sys": "./prompt/discover/prompt_sufficiency_decision_sys.txt",
    "sufficiency_decision_user": "./prompt/discover/prompt_sufficiency_decision_user.txt",
    "summary_disease_sys": "./prompt/discover/prompt_summary_disease_sys.txt",
    "summary_disease_user": "./prompt/discover/prompt_summary_disease_user.txt",
    "extract_disease_name_sys": "./prompt/discover/prompt_extract_disease_name_sys.txt",
    "extract_disease_name_user": "./prompt/discover/prompt_extract_disease_name_user.txt",
    "disease_analysis_sys": "./prompt/discover/prompt_disease_analysis_sys.txt",
    "disease_analysis_user": "./prompt/discover/prompt_disease_analysis_user.txt",
    "concomitant_symptoms_sys": "./prompt/discover/prompt_concomitant_symptoms_sys.txt",
    "concomitant_symptoms_user": "./prompt/discover/prompt_concomitant_symptoms_user.txt",
    "generate_solution_sys": "./prompt/discover/prompt_generate_solution_sys.txt",
    "generate_solution_user": "./prompt/discover/prompt_generate_solution_user.txt",
    "extract_disease_info_sys": "./prompt/discover/prompt_extract_disease_info_sys.txt",
    "extract_disease_info_user": "./prompt/discover/prompt_extract_disease_info_user.txt",
    "chief_complaint_sys": "./prompt/discover/prompt_chief_complaint_sys.txt",
    "chief_complaint_user": "./prompt/discover/prompt_chief_complaint_user.txt",
    "missing_questions_sys": "./prompt/discover/prompt_missing_questions_sys.txt",
    "missing_questions_user": "./prompt/discover/prompt_missing_questions_user.txt",
    "ask_question_by_symptoms_sys": "./prompt/discover/prompt_ask_question_by_symptoms_sys.txt",
    "ask_question_by_symptoms_user": "./prompt/discover/prompt_ask_question_by_symptoms_user.txt",
}

# 现病诱因 (TypedDict 版本)
class PresentIllness(TypedDict, total=False):
    onset_time: str  # 病开始时间
    pattern: str  # 持续/阵发/间歇性等
    treatments_received: List[str]  # 接受什么治疗，包括吃过什么药物，做过什么医学治疗


# 设置 State 状态
class MedicalInquiryState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # 核心问询字段
    chief_complaint: str  # 主诉：喊着最主要痛苦/问题
    present_illness: PresentIllness
    associated_symptoms: List[str]
    symptom_absent: List[str]
    existing_knowledge_supplement: Dict[str, Any]  # 当信息不充足时，使用当前数据信息，查询可能存在的病症，提供参考，从RAG检索到的医学知识
    missing_return_messages: str
    # 补充信息
    is_sufficient: bool  # stepA 判断结果：当前信息是否足够
    missing_info: List[str]  # 缺失的病症信息项（如“疼痛部位”“持续时间”）
    solution: str  # 生成的解决方案（诊断建议/下一步行动）
    need_more_info: bool  # stepC后再次判断的结果
    iteration_count: int  # 防止无限循环


# 定义结构化输出的类（数据契约）(BaseModel 版本)
class PresentIllnessSchema(BaseModel):
    """主诉信息结构化数据"""
    onset_time: str = Field(description="病症开始时间，或者提供的时间")
    pattern: str = Field(description="疼痛或发作规律，如持续/阵发/间歇性")
    treatments_received: List[str] = Field(description="接受过的治疗或服用的药物清单")


# ===== 新增：信息充分性判断的结构化输出类 =====
class SufficiencyDecision(BaseModel):
    """判断当前信息是否足够进行病症分析，并列出缺失的关键信息"""
    missing_info: List[str] = Field(description="若不足够，列出需要补充的医疗信息项（如'是否有发热或体温变化'、'是否有鼻窦压痛或鼻涕颜色异常'等）")
    confidence: float = Field(description="判断的置信度，0-1之间。应该尽可能重新的信息进行分析")

# ===== 新增：生成解决方案的结构化输出类 =====
class SolutionOutput(BaseModel):
    """生成的解决方案（诊断建议/下一步行动）"""
    possible_diagnoses: List[str] = Field(description="可能的疾病诊断列表")
    recommended_actions: List[str] = Field(description="建议的下一步行动，如检查、就医等")
    disclaimer: str = Field(description="免责声明，提醒用户仅供参考")


class MedicalDataDiscovery:

    def get_base_chat_model(self):
        """
        获取一个ollama会话模型
        :return:
        """
        return ChatOllama(
            model="qwen3:8b",
            base_url="http://127.0.0.1:11434",
            temperature=0.0
        )

    def __init__(self):
        # 初始化模型
        self.ollama_model = self.get_base_chat_model()
        # RAG 查询工具
        self.rag_service = KnowledgeBaseService()
        # 知识图谱查询工具
        self.knowledge_query_tools = Neo4jQueryTools()
        # 构建查询命中工具
        self.search_db_client = DiseaseAnalysisService()

    # 创建一个 Agent
    def get_llm_agent(self, sys_prompt: str, tools):
        agent = create_agent(
            model=self.ollama_model,
            tools=tools or [],
            system_prompt=sys_prompt
        )
        return agent

    def get_model_chief_complaint(self, state: MedicalInquiryState) -> Dict[str, Any]:
        # ----- 日志 -----
        logger.info("[节点] get_model_chief_complaint - 开始提取主诉")
        logger.info(f"  输入消息数: {len(state.get('messages', []))}")
        # ----------------
        tmp_messages = state["messages"]

        chief_complaint_sys = read_prompt(PROMPT_PATHS["chief_complaint_sys"])
        chief_complaint_user = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["chief_complaint_user"])).format(
            input_data = state["chief_complaint"],
            input_messages = tmp_messages[-1].content
        )
        system_messages = [SystemMessage(content=chief_complaint_sys)]
        user_messages = [HumanMessage(content=chief_complaint_user)]

        model_rsp_message = self.ollama_model.invoke(system_messages + user_messages)
        chief = model_rsp_message.content
        logger.info(f"  提取的主诉: {chief}")
        return {"chief_complaint": chief}

    def get_present_illness(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
            获取一个现在病症的状态
        :param state:
        :return:
        """
        # ----- 日志 -----
        logger.info("[节点] get_present_illness - 开始提取现病史")

        extract_disease_info_sys = read_prompt(PROMPT_PATHS["extract_disease_info_sys"])
        extract_disease_info_user = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["extract_disease_info_user"])).format(
            input_message = state["messages"][-1],
            input_present_illness = state["present_illness"]
        )

        extract_disease_info_sys_messages = [SystemMessage(content=extract_disease_info_sys)]
        prompt_sufficiency_decision_user_messages = [HumanMessage(content=extract_disease_info_user)]

        structured_llm = self.ollama_model.with_structured_output(PresentIllnessSchema)
        response = structured_llm.invoke(extract_disease_info_sys_messages+ prompt_sufficiency_decision_user_messages)

        present = {
            "onset_time": response.onset_time,
            "pattern": response.pattern,
            "treatments_received": response.treatments_received
        }
        logger.info(f"  提取的现病史: {present}")
        return {"present_illness": present}

    def get_concomitant_symptoms(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        获取并发症状，包括存在的伴随症状和不存在的症状
        :return: 包含 "associated_symptoms" 和 "symptom_absent" 的字典
        """
        logger.info("[节点] get_concomitant_symptoms - 开始提取伴随症状")
        tmp_messages: List[str] = state["messages"]

        prompt_concomitant_symptoms_sys = read_prompt(PROMPT_PATHS["concomitant_symptoms_sys"])
        prompt_concomitant_symptoms_user = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["concomitant_symptoms_user"])).format(
            input_message = tmp_messages[-1],
            input_associated_symptoms=state["associated_symptoms"],
            input_symptom_absent=state["symptom_absent"]
        )

        concomitant_symptoms_sys_messages = [SystemMessage(content=prompt_concomitant_symptoms_sys)]
        concomitant_symptoms_user_messages = [HumanMessage(content=prompt_concomitant_symptoms_user)]

        model_rsp_message = self.ollama_model.invoke(concomitant_symptoms_sys_messages + concomitant_symptoms_user_messages)
        content = model_rsp_message.content.strip()

        # 默认空结构
        result = {
            "associated_symptoms": [],
            "symptom_absent": []
        }

        # 解析 JSON
        try:
            # 尝试提取 JSON 部分（防止模型输出额外文字）
            import re
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)  # 使用 json.loads 更安全
                if isinstance(parsed, dict):
                    result["associated_symptoms"] = parsed.get("associated_symptoms", [])
                    result["symptom_absent"] = parsed.get("symptom_absent", [])
                    # 确保字段为列表
                    if not isinstance(result["associated_symptoms"], list):
                        result["associated_symptoms"] = []
                    if not isinstance(result["symptom_absent"], list):
                        result["symptom_absent"] = []
        except Exception as e:
            logger.warning(f"  解析伴随症状 JSON 失败: {e}, 原始内容: {content}")

        logger.info(f"  提取的伴随症状: {result['associated_symptoms']}")
        logger.info(f"  不存在的症状: {result['symptom_absent']}")
        return result

    # ===== 新增节点 1：信息充分性判断（意图识别）=====
    def classify_sufficiency(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        判断当前收集到的症状信息是否足够进行病症分析。
        输出 missing_info, confidence。
        """
        logger.info("[节点] classify_sufficiency - 判断信息充分性")
        chief = state.get("chief_complaint", "")
        present = state.get("present_illness", {})
        associated = state.get("associated_symptoms", [])
        symptom_absent = state.get("symptom_absent", [])
        existing_knowledge_sup = state.get("existing_knowledge_supplement", [])

        # 修改：将消息列表转为文本，避免直接打印消息对象
        messages_text = "\n".join([str(m.content) for m in state.get("messages", [])])

        prompt_medical_chat_answer_sys = read_prompt(PROMPT_PATHS["sufficiency_decision_sys"])
        prompt_medical_chat_answer_user = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["sufficiency_decision_user"])).format(
            input_user ={
                "chief_complaint":chief,
                "present_illness": present,
                "associated_symptoms": associated,
                "symptom_absent": symptom_absent,
                "existing_knowledge_supplement": existing_knowledge_sup,
            },
            input_messages=messages_text
        )

        system_messages = [SystemMessage(content=prompt_medical_chat_answer_sys)]
        user_messages = [HumanMessage(content=prompt_medical_chat_answer_user)]

        model_with_structure = self.ollama_model.with_structured_output(SufficiencyDecision)
        decision = model_with_structure.invoke(system_messages + user_messages)

        logger.info(f"  缺失信息: {decision.missing_info}")
        logger.info(f"  置信度: {decision.confidence}")

        return {
            "is_sufficient": True if decision.confidence >= 0.7 else False,
            "missing_info": decision.missing_info,
        }

    # ===== 新增节点 2：RAG 知识增强 =====
    def enhance_with_rag(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        使用当前已提取的症状信息查询医学知识库，将结果存入 existing_knowledge_supplement。
        此节点可在判断充分性之前或之后调用，增强后续节点的知识背景。
        """
        logger.info("[节点] enhance_with_rag - 查询RAG知识库")
        chief = state.get("chief_complaint", "")
        present_illness = state.get("present_illness", {})
        associated = state.get("associated_symptoms", [])
        symptom_absent = state.get("symptom_absent", [])

        query_parts = [f"主要症状：{chief}"]

        if present_illness.get("onset_time"):
            query_parts.append(f"发病时间：{present_illness['onset_time']}")
        if present_illness.get("pattern"):
            query_parts.append(f"发作规律：{present_illness['pattern']}")
        if present_illness.get("treatments_received"):
            query_parts.append(f"治疗/药物：{', '.join(present_illness['treatments_received'])}")
        if associated:
            query_parts.append(f"伴随症状：{', '.join(associated)}")
        if symptom_absent:
            query_parts.append(f"不存在症状：{', '.join(symptom_absent)}")

        query_str = "，".join(query_parts)
        logger.info("医疗信息充分，RAG检索问题："+query_str)
        # 调用 RAG 服务
        results = self.rag_service.query(query_str, top_k=5)  # 返回列表

        context_str = ""
        if results:
            for i, item in enumerate(results, 1):
                context_str += f"【参考{i}】{item['text'][:300]}...\n"
            logger.info(f"  检索到 {len(results)} 条结果")
        else:
            context_str = "未检索到相关知识。"
            logger.info(context_str)

        prompt_disease_analysis_sys = read_prompt(PROMPT_PATHS["disease_analysis_sys"])
        prompt_disease_analysis_user = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["disease_analysis_user"])).format(
            rag_text = results,
            knowledge_query_data=query_str
        )

        disease_analysis_sys_messages = [SystemMessage(content=prompt_disease_analysis_sys)]
        disease_analysis_user_messages = [HumanMessage(content=prompt_disease_analysis_user)]

        model_rsp_summary_disease_info = self.ollama_model.invoke(disease_analysis_sys_messages + disease_analysis_user_messages)

        json_existing_knowledge_supplement = safe_json_parse(model_rsp_summary_disease_info.content)
        json_existing_knowledge_supplement["疾病确定概率"] = "80%"
        # 不改变任何状态，后续会由 generate_missing_questions 生成问题
        return {"existing_knowledge_supplement": [json_existing_knowledge_supplement]}

    # ===== 新增节点 3：生成解决方案 =====
    def generate_solution(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        基于现有信息（包括RAG上下文）生成初步诊断建议和下一步行动。
        """
        logger.info("[节点] generate_solution - 生成解决方案")
        chief = state.get("chief_complaint", "")
        present = state.get("present_illness", {})
        associated = state.get("associated_symptoms", [])
        symptom_absent = state.get("symptom_absent", [])
        rag_ctx = state.get("existing_knowledge_supplement", "")
        logger.info(f"  主诉: {chief}")
        logger.info(f"  现病史: {present}")
        logger.info(f"  伴随症状: {associated}")

        input_data = f"""
            主诉：{chief}
            现病史：发病时间 {present.get('onset_time', '未知')}，发作规律 {present.get('pattern', '未知')}，已用药物/治疗 {', '.join(present.get('treatments_received', []))}
            伴随症状：{', '.join(associated) if associated else '无'}，不存在症状：{', '.join(symptom_absent) if symptom_absent else '无'}。
        """

        prompt_generate_solution_sys = read_prompt(PROMPT_PATHS["generate_solution_sys"])
        prompt_generate_solution_user = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["generate_solution_user"])).format(
            input_data = input_data,
            rag_ctx=rag_ctx
        )

        generate_solution_sys_messages = [SystemMessage(content=prompt_generate_solution_sys)]
        generate_solution_user_messages = [HumanMessage(content=prompt_generate_solution_user)]

        model_with_structure = self.ollama_model.with_structured_output(SolutionOutput)
        solution = model_with_structure.invoke(generate_solution_sys_messages+generate_solution_user_messages)

        # 将结构化的解决方案转换为字符串存入 state.solution
        solution_text = f"可能诊断：{', '.join(solution.possible_diagnoses)}\n建议行动：{', '.join(solution.recommended_actions)}\n免责声明：{solution.disclaimer}"
        logger.info(f"  生成的解决方案:\n{solution_text}")
        return {"solution": solution_text}

    # ===== 新增节点 4：生成缺失信息问题 =====
    def generate_missing_questions(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        根据 missing_info 列表，生成要询问用户的问题，并追加到 messages 中。
        返回的 messages 将通过 add_messages 自动合并。
        """
        logger.info("[节点] generate_missing_questions - 生成追问问题")
        missing = state.get("missing_info", [])
        logger.info(f"  缺失信息列表: {missing}")
        if not missing:
            logger.info("  无缺失信息，返回感谢语")
            return {"messages": [AIMessage(content="感谢您提供的信息，如果需要更多帮助，请继续描述。")]}

        tmp_missing_info = state.get("missing_info", [])
        tmp_messages = state.get("messages", [])
        tmp_existing_knowledge_supplement = state.get("existing_knowledge_supplement", [])

        # 将消息列表转为文本
        messages_text = "\n".join([str(m.content) for m in tmp_messages])

        prompt_missing_questions_sys = read_prompt(PROMPT_PATHS["missing_questions_sys"])
        prompt_missing_questions_user = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["missing_questions_user"])).format(
            messages_text = messages_text,
            missing_info=tmp_missing_info,
            existing_knowledge_supplement=tmp_existing_knowledge_supplement,
        )

        missing_questions_sys_messages = [SystemMessage(content=prompt_missing_questions_sys)]
        missing_questions_user_messages = [HumanMessage(content=prompt_missing_questions_user)]

        model_rsp_message = self.ollama_model.invoke(missing_questions_sys_messages + missing_questions_user_messages)

        content = model_rsp_message.content.strip()

        logger.info(f"生成的问题:\n{content}")
        return {"solution": content}

    # ===== 优化：重新判断信息充分性（复用 classify_sufficiency）=====
    def recheck_sufficiency(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        在提取完缺失信息后，再次判断信息是否充分。
        直接调用 classify_sufficiency 节点，但需要单独定义以避免循环调用。
        """
        logger.info("[节点] recheck_sufficiency - 再次判断信息充分性")
        return self.classify_sufficiency(state)

    # ===== 路由函数：根据信息是否充足决定下一步 =====
    @staticmethod
    def route_after_sufficiency(state: MedicalInquiryState) -> Literal["enhance_with_rag_node", "extract_missing_info_node"]:
        """条件边：信息充足则进入RAG增强并生成解决方案；不足则进入缺失信息提取流程"""
        choice = "enhance_with_rag_node" if state.get("is_sufficient", False) else "extract_missing_info_node"
        logger.info(f"\n[路由] classify_sufficiency → {choice}")
        return choice

    @staticmethod
    def route_after_recheck(state: MedicalInquiryState) -> Literal["generate_solution", "generate_missing_questions"]:
        """在重新判断后：信息充足则生成解决方案，否则继续追问缺失信息"""
        choice = "generate_solution_node" if state.get("is_sufficient", False) else "generate_missing_questions_node"
        logger.info(f"\n[路由] recheck_sufficiency → {choice}")
        return choice

    def rag_query(self, query, top_k=3):
        """ 封装一个RAG知识查询模块 """
        return self.rag_service.query(query, top_k=top_k)  # 返回列表

    def extract_missing_info_node(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        根据路由判断当前信息是否充分
        Step：根据当前的症状查询 RAG 中的数据。
        Step-1：根据主诉病症查询对应疾病
        Step-2：根据对应疾病查询对应并发症
        Step-3：根据根据RAG片段和疾病信息进行整理
        """
        chief_complaint = state["chief_complaint"]

        ask_question_by_symptoms_sys = read_prompt(PROMPT_PATHS["ask_question_by_symptoms_sys"])
        ask_question_by_symptoms_user = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["ask_question_by_symptoms_user"])).format(
            input_data=chief_complaint
        )

        ask_question_by_symptoms_sys_messages = [SystemMessage(content=ask_question_by_symptoms_sys)]
        ask_question_by_symptoms_user_messages = [HumanMessage(content=ask_question_by_symptoms_user)]

        # 修改：messages 必须是列表
        model_rsp_message = self.ollama_model.invoke(ask_question_by_symptoms_sys_messages+ ask_question_by_symptoms_user_messages)

        query = model_rsp_message.content

        rag_query_result = self.rag_query(query, top_k=3)

        summary_disease_list = list()
        for query_info in rag_query_result:
            # 修改：从字典中提取 text 字段
            rag_text = query_info.get("text", str(query_info))

            extract_disease_name_sys = read_prompt(PROMPT_PATHS["extract_disease_name_sys"])
            extract_disease_name_user = PromptTemplate.from_template(
                read_prompt(PROMPT_PATHS["extract_disease_name_user"])).format(
                rag_text = rag_text
            )
            extract_disease_name_sys_messages = [SystemMessage(content=extract_disease_name_sys)]
            extract_disease_name_user_messages = [HumanMessage(content=extract_disease_name_user)]

            model_rsp_message = self.ollama_model.invoke(extract_disease_name_sys_messages + extract_disease_name_user_messages)
            disease_name = model_rsp_message.content.strip()
            # 知识图谱查询的数据
            knowledge_query_data = self.knowledge_query_tools.query_disease(disease_name)

            prompt_chat_intent_choice_sys = read_prompt(PROMPT_PATHS["summary_disease_sys"])
            prompt_chat_intent_choice_user = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["summary_disease_user"])).format(
                knowledge_query_data=knowledge_query_data,
                rag_text = rag_text,
                symptom = {
                    "主诉症状": state.get("chief_complaint", ""),   # 主诉：喊着最主要痛苦/问题
                    "现病史": state.get("present_illness", {}),
                    "并发症": state.get("associated_symptoms", []),
                    "不存在症状": state.get("symptom_absent", [])
                }
            )
            chat_intent_choice_sys_messages = [SystemMessage(content=prompt_chat_intent_choice_sys)]
            chat_intent_choice_user_messages = [HumanMessage(content=prompt_chat_intent_choice_user)]

            model_rsp_summary_disease_info = self.ollama_model.invoke(chat_intent_choice_sys_messages + chat_intent_choice_user_messages)

            summary_disease_list.append(safe_json_parse(model_rsp_summary_disease_info.content))

        logger.info("== RAG分析后概率结果 == ")
        logger.info(summary_disease_list)

        # 不改变任何状态，后续会由 generate_missing_questions 生成问题
        return {"existing_knowledge_supplement": summary_disease_list}

    def build_search_db(self, state: MedicalInquiryState):
        """
            构建查询命中逻辑
        :return:
        """
        logger.info("进入 build_search_db")
        use_messages = list()
        state_messages = state.get("messages", [])
        for message in state_messages:
            if message.content:
                use_messages.append(message.content)

        # 执行分析
        build_search_state = self.search_db_client.run({
            "chief_complaint": state.get("chief_complaint", ""),
            "present_illness": state.get("present_illness", {}),
            "associated_symptoms": state.get("associated_symptoms", []),
            "symptom_absent": state.get("symptom_absent", []),
            "existing_knowledge_supplement": state.get("existing_knowledge_supplement", []),
            "use_message": use_messages
        })
        logger.info("构建查询命中")
        logger.info(build_search_state)
        return {}

    def get_graph(self):
        """
            构建诊断信息探寻图
        :return:
        """
        graph_builder = StateGraph(MedicalInquiryState)

        # 添加所有节点
        graph_builder.add_node("chief_complaint_node", self.get_model_chief_complaint)
        graph_builder.add_node("present_illness_node", self.get_present_illness)
        graph_builder.add_node("concomitant_symptoms_node", self.get_concomitant_symptoms)
        graph_builder.add_node("classify_sufficiency_node", self.classify_sufficiency)
        graph_builder.add_node("enhance_with_rag_node", self.enhance_with_rag)
        graph_builder.add_node("generate_solution_node", self.generate_solution)

        graph_builder.add_node("build_search_db_node", self.build_search_db)

        graph_builder.add_node("extract_missing_info_node", self.extract_missing_info_node)  # 替换 lambda
        graph_builder.add_node("recheck_sufficiency_node", self.recheck_sufficiency)
        graph_builder.add_node("generate_missing_questions_node", self.generate_missing_questions)

        # 设置边
        graph_builder.add_edge(START, "chief_complaint_node")
        graph_builder.add_edge("chief_complaint_node", "present_illness_node")
        graph_builder.add_edge("present_illness_node", "concomitant_symptoms_node")
        graph_builder.add_edge("concomitant_symptoms_node", "classify_sufficiency_node")

        # 条件分支：信息充分性判断后的路由
        graph_builder.add_conditional_edges(
            "classify_sufficiency_node",
            self.route_after_sufficiency,  # 现在是一个静态方法，可以直接用类名调用，但实例方法也可以
            {
                "enhance_with_rag_node": "enhance_with_rag_node",
                "extract_missing_info_node": "extract_missing_info_node"
            }
        )

        # 信息充分时，RAG增强后直接生成解决方案，然后结束
        graph_builder.add_edge("enhance_with_rag_node", "generate_solution_node")
        graph_builder.add_edge("generate_solution_node", "build_search_db_node")
        graph_builder.add_edge("build_search_db_node", END)

        # 信息不充分时，先进行缺失信息提取（占位），然后重新判断
        graph_builder.add_edge("extract_missing_info_node", "recheck_sufficiency_node")

        # 重新判断后的路由
        graph_builder.add_conditional_edges(
            "recheck_sufficiency_node",
            self.route_after_recheck,  # 静态方法
            {
                "generate_solution_node": "generate_solution_node",
                "generate_missing_questions_node": "generate_missing_questions_node"
            }
        )
        graph_builder.add_edge("generate_missing_questions_node", END)

        logger.info("[图编译完成] 辅助诊断助手 Agent 已初始化\n")

        # 编译图
        return graph_builder.compile()


    def get_medical_data_discovery(self, messages:BaseMessage, discovery_data:dict[str, object]):
        agent = self.get_graph()
        # 修改：提供完整的初始状态，避免 KeyError
        initial_state = {
            "messages": messages,
            "iteration_count": 0,
            "chief_complaint": discovery_data.get("chief_complaint", ""),
            "present_illness": discovery_data.get("present_illness", {}),
            "associated_symptoms": discovery_data.get("associated_symptoms", []),
            "symptom_absent": discovery_data.get("symptom_absent", []),
            "existing_knowledge_supplement": discovery_data.get("existing_knowledge_supplement", []),
            "is_sufficient": False,
            "missing_info": [],
            "solution": "",
            "need_more_info": False
        }
        return agent.invoke(initial_state)


# 测试代码
if __name__ == '__main__':
    messages = [HumanMessage(content="今天早上头疼，有些流鼻涕。无咳嗽,有鼻塞,无畏寒，不咽喉干燥")]
    logger.info("=== 开始测试 ===")
    graph_client = MedicalDataDiscovery()
    result = graph_client.get_medical_data_discovery(messages, {})

    print("\n=== 最终结果（完整状态） ===")
    for key, value in result.items():
        # 对 messages 字段进行特殊处理，避免打印过长内容
        if key == "messages":
            print(f"{key}: {[msg.content for msg in value]}")
        else:
            print(f"{key}: {value}")