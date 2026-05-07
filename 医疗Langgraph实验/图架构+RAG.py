from typing import Dict, Any, List, Optional, Annotated, TypedDict, Union, Literal

from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages, MessagesState
from pydantic import BaseModel, Field

from knowledge_graph_tools import Neo4jQueryTools
from rag_module import KnowledgeBaseService


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
    existing_knowledge_supplement: List[str]  # 当信息不充足时，使用当前数据信息，查询可能存在的病症，提供参考
    # 补充信息
    is_sufficient: bool  # stepA 判断结果：当前信息是否足够
    missing_info: List[str]  # 缺失的病症信息项（如“疼痛部位”“持续时间”）
    solution: str  # 生成的解决方案（诊断建议/下一步行动）
    rag_context: str  # 从RAG检索到的医学知识
    need_more_info: bool  # stepC后再次判断的结果
    iteration_count: int  # 防止无限循环


# 定义结构化输出的类（数据契约）(BaseModel 版本)
class PresentIllnessSchema(BaseModel):
    """主诉信息结构化数据"""
    onset_time: str = Field(description="病症开始时间")
    pattern: str = Field(description="疼痛或发作规律，如持续/阵发/间歇性")
    treatments_received: List[str] = Field(description="接受过的治疗或服用的药物清单")


# ===== 新增：信息充分性判断的结构化输出类 =====
class SufficiencyDecision(BaseModel):
    """判断当前信息是否足够进行病症分析，并列出缺失的关键信息"""
    missing_info: List[str] = Field(description="若不足够，列出需要补充的医疗信息项（如'疼痛部位'、'持续时间'等）")
    confidence: float = Field(description="判断的置信度，0-1之间")


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

    # 创建一个“提交答案”的工具
    @tool(args_schema=PresentIllnessSchema, return_direct=True)
    def submit_illness_info(
            onset_time: str,
            pattern: str,
            treatments_received: List[str]
    ) -> str:
        """将提取的病情信息提交给系统。你必须调用这个工具来输出你的答案。"""
        # 这个函数体不会被执行，仅用来生成 tool schema
        return "信息已接收"

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
        print("\n[节点] get_model_chief_complaint - 开始提取主诉")
        print(f"  输入消息数: {len(state.get('messages', []))}")
        # ----------------
        messages = state["messages"]
        tmp_chief_complaint = """
            根据用户聊天信息获取其中的病人最大的病痛信息。应该表现出是症状，或者最大痛苦。只返回症状，必须精简。
            例如： 【输入：早上有些发烧和头疼。 输出：发烧，头疼】
        """
        model_chief_complaint = self.get_llm_agent(tmp_chief_complaint, None)
        model_rsp_message = model_chief_complaint.invoke({"messages": messages})
        chief = model_rsp_message["messages"][-1].content
        print(f"  提取的主诉: {chief}")
        return {"chief_complaint": chief}

    def get_present_illness(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
            获取一个现在病症的状态
        :param state:
        :return:
        """
        # ----- 日志 -----
        print("\n[节点] get_present_illness - 开始提取现病史")
        # ----------------
        messages = state["messages"]
        tmp_chief_complaint = """
            你是一个专业的医疗信息提取助手。根据用户描述提取：
            - 病症开始时间 (onset_time)
            - 发作规律 (pattern)：持续/阵发/间歇性
            - 接受过的治疗或药物 (treatments_received)，列表形式

            提取后**必须调用 submit_illness_info 工具**，传入三个字段。调用一次后任务即完成。
            例如：【
            输入：
            今天早上有些咳嗽，曾经吃过一些感康
            输出：
            {
                'onset_time': '今天早上',
                'pattern': '阵发',   # 或空字符串，取决于模型判断
                'treatments_received': ['感康']
            }
            】
        """
        agent = self.get_llm_agent(tmp_chief_complaint, [self.submit_illness_info])
        # 限制递归深度
        response = agent.invoke({"messages": messages}, config={"recursion_limit": 10})

        for msg in response["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                args = msg.tool_calls[0]["args"]
                present = {
                    "onset_time": args.get("onset_time", ""),
                    "pattern": args.get("pattern", ""),
                    "treatments_received": args.get("treatments_received", [])
                }
                print(f"  提取的现病史: {present}")
                return {"present_illness": present}
        print("  现病史提取失败，返回空值")
        return {
            "present_illness": {"onset_time": "", "pattern": "", "treatments_received": []}
        }

    def get_concomitant_symptoms(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        获取并发症状
        :return:
        """
        # ----- 日志 -----
        print("\n[节点] get_concomitant_symptoms - 开始提取伴随症状")
        # ----------------
        messages = state["messages"]
        tmp_prompt = """
            根据提供的信息，提取数据中的主要病症之外的其他并发症（伴随症状），以 Python 列表形式输出，例如 ["浑身疼痛"]。如果没有其他症状，输出空列表 []。
            示例：【
            输入：早上发烧，浑身感觉疼痛
            输出：["浑身疼痛"]
            】
        """
        agent = self.get_llm_agent(tmp_prompt, None)
        model_rsp_message = agent.invoke({"messages": messages})
        content = model_rsp_message["messages"][-1].content.strip()

        # 将模型输出解析为列表
        try:
            symptoms_list = eval(content) if content.startswith('[') else []
            if not isinstance(symptoms_list, list):
                symptoms_list = []
        except:
            symptoms_list = []

        print(f"  提取的伴随症状: {symptoms_list}")
        return {"associated_symptoms": symptoms_list}

    class RAGQueryInput(BaseModel):
        """用于 RAG 知识库查询的输入参数"""
        chief_complaint: str = Field(description="主诉症状，如 '发烧,头痛'")
        onset_time: Optional[str] = Field(default="", description="病症开始时间")
        pattern: Optional[str] = Field(default="", description="发作规律：持续/阵发/间歇性")
        treatments_received: Optional[List[str]] = Field(default=[], description="接受过的治疗或药物")
        associated_symptoms: Optional[List[str]] = Field(default=[], description="伴随症状列表")

    # ===== 新增节点 1：信息充分性判断（意图识别）=====
    def classify_sufficiency(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        判断当前收集到的症状信息是否足够进行病症分析。
        输出 missing_info, confidence。
        """
        print("\n[节点] classify_sufficiency - 判断信息充分性")
        chief = state.get("chief_complaint", "")
        present = state.get("present_illness", {})
        associated = state.get("associated_symptoms", [])
        existing_knowledge_sup = state.get("existing_knowledge_supplement", [])

        print(f"  主诉: {chief}")
        print(f"  现病史: {present}")
        print(f"  伴随症状: {associated}")

        # 修改：将消息列表转为文本，避免直接打印消息对象
        messages_text = "\n".join([str(m.content) for m in state.get("messages", [])])

        prompt = f"""
你是一位医学信息评估专家。请根据以下患者信息判断是否足够做出初步的病症分析。

主诉：{chief}
现病史：发病时间 {present.get('onset_time', '未知')}，发作规律 {present.get('pattern', '未知')}，已接受治疗/药物 {', '.join(present.get('treatments_received', []))}
伴随症状：{', '.join(associated) if associated else '无'}。

当前用户症状为：{messages_text}

经过信息探寻分析数据：
{existing_knowledge_sup}

请回答：
1. 如果不足，请列出最关键的 1-3 个缺失信息项，根据当前的现病史选择需要补充的问题。
2. 给出判断的置信度（0-1）。
"""
        model_messages = [SystemMessage(content=prompt)]
        model_with_structure = self.ollama_model.with_structured_output(SufficiencyDecision)
        decision = model_with_structure.invoke(model_messages)

        print(f"  缺失信息: {decision.missing_info}")
        print(f"  置信度: {decision.confidence}")

        return {
            "is_sufficient": True if decision.confidence >= 0.7 else False,
            "missing_info": decision.missing_info,
        }

    # ===== 新增节点 2：RAG 知识增强 =====
    def enhance_with_rag(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        使用当前已提取的症状信息查询医学知识库，将结果存入 rag_context。
        此节点可在判断充分性之前或之后调用，增强后续节点的知识背景。
        """
        print("\n[节点] enhance_with_rag - 查询RAG知识库")
        chief = state.get("chief_complaint", "")
        present_illness = state.get("present_illness", {})
        associated = state.get("associated_symptoms", [])

        query_parts = [f"主要症状：{chief}"]

        if present_illness.get("onset_time"):
            query_parts.append(f"发病时间：{present_illness['onset_time']}")
        if present_illness.get("pattern"):
            query_parts.append(f"发作规律：{present_illness['pattern']}")
        if present_illness.get("treatments_received"):
            query_parts.append(f"治疗/药物：{', '.join(present_illness['treatments_received'])}")
        if associated:
            query_parts.append(f"伴随症状：{', '.join(associated)}")

        query_str = "，".join(query_parts)
        # 调用 RAG 服务
        results = self.rag_service.query(query_str, top_k=3)  # 返回列表

        context_str = ""
        if results:
            for i, item in enumerate(results, 1):
                context_str += f"【参考{i}】{item['text'][:300]}...\n"
            print(f"  检索到 {len(results)} 条结果")
        else:
            context_str = "未检索到相关知识。"
            print("  未检索到相关知识")

        return {"rag_context": context_str}

    # ===== 新增节点 3：生成解决方案 =====
    def generate_solution(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        基于现有信息（包括RAG上下文）生成初步诊断建议和下一步行动。
        """
        print("\n[节点] generate_solution - 生成解决方案")
        chief = state.get("chief_complaint", "")
        present = state.get("present_illness", {})
        associated = state.get("associated_symptoms", [])
        rag_ctx = state.get("rag_context", "")
        print(f"  主诉: {chief}")
        print(f"  现病史: {present}")
        print(f"  伴随症状: {associated}")

        prompt = f"""
    你是一位经验丰富的临床医生。请根据以下患者信息提供初步的医学建议。

    主诉：{chief}
    现病史：发病时间 {present.get('onset_time', '未知')}，发作规律 {present.get('pattern', '未知')}，已用药物/治疗 {', '.join(present.get('treatments_received', []))}
    伴随症状：{', '.join(associated) if associated else '无'}

    医学知识库参考：
    {rag_ctx}

    请以结构化形式输出：
    - 可能的疾病诊断（列出 2-3 个最可能的诊断）
    - 建议的下一步行动（如立即就医、自我监测、做某类检查等）
    - 免责声明（提示此信息不构成医疗建议）
    """
        model_with_structure = self.ollama_model.with_structured_output(SolutionOutput)
        solution = model_with_structure.invoke([HumanMessage(content=prompt)])

        # 将结构化的解决方案转换为字符串存入 state.solution
        solution_text = f"可能诊断：{', '.join(solution.possible_diagnoses)}\n建议行动：{', '.join(solution.recommended_actions)}\n免责声明：{solution.disclaimer}"
        print(f"  生成的解决方案:\n{solution_text}")
        return {"solution": solution_text}

    # ===== 新增节点 4：生成缺失信息问题 =====
    def generate_missing_questions(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        根据 missing_info 列表，生成要询问用户的问题，并追加到 messages 中。
        返回的 messages 将通过 add_messages 自动合并。
        """
        print("\n[节点] generate_missing_questions - 生成追问问题")
        missing = state.get("missing_info", [])
        print(f"  缺失信息列表: {missing}")
        if not missing:
            print("  无缺失信息，返回感谢语")
            return {"messages": [AIMessage(content="感谢您提供的信息，如果需要更多帮助，请继续描述。")]}

        tmp_missing_info = state.get("missing_info", [])
        tmp_messages = state.get("messages", [])
        tmp_existing_knowledge_supplement = state.get("existing_knowledge_supplement", [])

        # 将消息列表转为文本
        messages_text = "\n".join([str(m.content) for m in tmp_messages])

        prompt = f"""
        根据提供的病人询问信息中提到的内容和模型分析得到应该补充的内容进行整理，并生成响应内容，禁止使用提供数据外的信息进行回答。

        用户输入的询问数据：
        {messages_text}

        模型分析认为对疾病推理的缺失询问信息：
        {tmp_missing_info}

        可能病症补充分析数据：
        {tmp_existing_knowledge_supplement}


        输出：
        需要补充的临床提问信息：
        str——根据提供的** 模型分析认为对疾病推理的缺失询问信息 ** 进行推导总结。
        可能存在的病症
        名称：str
        常见症状：str
        诱因：str
        建议：str
        可能性：百分比

        """
        model_rsp_message = self.ollama_model.invoke([SystemMessage(content=prompt)])
        content = model_rsp_message.content.strip()

        print(f"  生成的问题:\n{content}")
        return {"messages": [AIMessage(content=content)]}

    # ===== 优化：重新判断信息充分性（复用 classify_sufficiency）=====
    def recheck_sufficiency(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        在提取完缺失信息后，再次判断信息是否充分。
        直接调用 classify_sufficiency 节点，但需要单独定义以避免循环调用。
        """
        print("\n[节点] recheck_sufficiency - 再次判断信息充分性")
        return self.classify_sufficiency(state)

    # ===== 路由函数：根据信息是否充足决定下一步 =====
    @staticmethod
    def route_after_sufficiency(state: MedicalInquiryState) -> Literal["enhance_with_rag", "extract_missing_info"]:
        """条件边：信息充足则进入RAG增强并生成解决方案；不足则进入缺失信息提取流程"""
        choice = "enhance_with_rag" if state.get("is_sufficient", False) else "extract_missing_info"
        print(f"\n[路由] classify_sufficiency → {choice}")
        return choice

    @staticmethod
    def route_after_recheck(state: MedicalInquiryState) -> Literal["generate_solution", "generate_missing_questions"]:
        """在重新判断后：信息充足则生成解决方案，否则继续追问缺失信息"""
        choice = "generate_solution" if state.get("is_sufficient", False) else "generate_missing_questions"
        print(f"\n[路由] recheck_sufficiency → {choice}")
        return choice

    def extract_missing_info_node(self, state: MedicalInquiryState) -> Dict[str, Any]:
        """
        根据路由判断当前信息是否充分
        Step：根据当前的症状查询 RAG 中的数据。
        Step-1：根据主诉病症查询对应疾病
        Step-2：根据对应疾病查询对应并发症
        Step-3：根据根据RAG片段和疾病信息进行整理
        """
        chief_complaint = state["chief_complaint"]

        tmp_chief_prompt = """
            根据输入的是与医疗疾病相关的主诉症状信息，将对应内容转换成的提问。
            例如：【
                输入：感觉头很痛
                输出：头疼可能是什么疾病？
            】
        """
        # 修改：messages 必须是列表
        messages = [HumanMessage(content=chief_complaint)]
        tmp_agent = self.get_llm_agent(tmp_chief_prompt, None)
        model_rsp_message = tmp_agent.invoke({"messages": messages})

        query = model_rsp_message["messages"][-1].content

        rag_query_result = self.rag_service.query(query, top_k=3)  # 返回列表

        summary_disease_list = list()
        for query_info in rag_query_result:
            # 修改：从字典中提取 text 字段
            rag_text = query_info.get("text", str(query_info))
            tmp_disease_keyword_prompt = f"""
                根据输入的数据信息，提取出其中的疾病名称，这个名称应该尽可能的精简，输出时只输出疾病名称，禁止输出其他内容。
                例如：【
                    输入：百日咳(pertussis, whoopingcough)是由百日咳杆菌所致的急性呼吸道传染病。
                    输出：百日咳。
                】
                用户输入为：{rag_text}
            """
            print("RAG知识片段信息：" + rag_text)
            model_rsp_message = self.ollama_model.invoke([SystemMessage(content=tmp_disease_keyword_prompt)])
            disease_name = model_rsp_message.content.strip()
            # 知识图谱查询的数据
            knowledge_query_data = self.knowledge_query_tools.query(disease_name)

            tmp_summary_disease_info = f"""
                根据提供的数据对信息进行整理。仅根据提供的数据进行整理和分析，禁止使用未提到的数据进行输出，内容应该尽量精简。

                输出内容JSON格式为：
                疾病名称：字符串
                疾病存在的症状：字符串
                疾病引起原因：字符串。

                下面是用户输入数据：
                症状：{knowledge_query_data}
                RAG文段数据：{rag_text}
            """
            model_rsp_summary_disease_info = self.ollama_model.invoke([SystemMessage(content=tmp_summary_disease_info)])
            summary_disease_list.append(model_rsp_summary_disease_info.content)

        print(summary_disease_list)

        # 不改变任何状态，后续会由 generate_missing_questions 生成问题
        return {"existing_knowledge_supplement": summary_disease_list}

    def get_graph(self):
        """
            构建诊断信息探寻图
        :return:
        """
        graph_builder = StateGraph(MedicalInquiryState)

        # 添加所有节点
        graph_builder.add_node("chief_complaint", self.get_model_chief_complaint)
        graph_builder.add_node("present_illness", self.get_present_illness)
        graph_builder.add_node("concomitant_symptoms", self.get_concomitant_symptoms)
        graph_builder.add_node("classify_sufficiency", self.classify_sufficiency)
        graph_builder.add_node("enhance_with_rag", self.enhance_with_rag)
        graph_builder.add_node("generate_solution", self.generate_solution)
        graph_builder.add_node("extract_missing_info", self.extract_missing_info_node)  # 替换 lambda
        graph_builder.add_node("recheck_sufficiency", self.recheck_sufficiency)
        graph_builder.add_node("generate_missing_questions", self.generate_missing_questions)

        # 设置边
        graph_builder.add_edge(START, "chief_complaint")
        graph_builder.add_edge("chief_complaint", "present_illness")
        graph_builder.add_edge("present_illness", "concomitant_symptoms")
        graph_builder.add_edge("concomitant_symptoms", "classify_sufficiency")

        # 条件分支：信息充分性判断后的路由
        graph_builder.add_conditional_edges(
            "classify_sufficiency",
            self.route_after_sufficiency,  # 现在是一个静态方法，可以直接用类名调用，但实例方法也可以
            {
                "enhance_with_rag": "enhance_with_rag",
                "extract_missing_info": "extract_missing_info"
            }
        )

        # 信息充分时，RAG增强后直接生成解决方案，然后结束
        graph_builder.add_edge("enhance_with_rag", "generate_solution")
        graph_builder.add_edge("generate_solution", END)

        # 信息不充分时，先进行缺失信息提取（占位），然后重新判断
        graph_builder.add_edge("extract_missing_info", "recheck_sufficiency")

        # 重新判断后的路由
        graph_builder.add_conditional_edges(
            "recheck_sufficiency",
            self.route_after_recheck,  # 静态方法
            {
                "generate_solution": "generate_solution",
                "generate_missing_questions": "generate_missing_questions"
            }
        )
        graph_builder.add_edge("generate_missing_questions", END)

        print("\n[图编译完成] 辅助诊断助手 Agent 已初始化\n")

        # 编译图
        return graph_builder.compile()


# 测试代码
if __name__ == '__main__':
    messages = [HumanMessage(content="今天早上头疼，有些流鼻涕")]
    print("=== 开始测试 ===")
    graph_client = MedicalDataDiscovery()
    agent = graph_client.get_graph()
    # 修改：提供完整的初始状态，避免 KeyError
    initial_state = {
        "messages": messages,
        "iteration_count": 0,
        "chief_complaint": "",
        "present_illness": {},
        "associated_symptoms": [],
        "existing_knowledge_supplement": [],
        "is_sufficient": False,
        "missing_info": [],
        "solution": "",
        "rag_context": "",
        "need_more_info": False
    }
    result = agent.invoke(initial_state)
    print("\n=== 最终结果 ===")
    print(f"主诉: {result.get('chief_complaint')}")
    print(f"现病史: {result.get('present_illness')}")
    print(f"伴随症状: {result.get('associated_symptoms')}")
    print(f"信息充分: {result.get('is_sufficient')}")
    print(f"缺失信息: {result.get('missing_info')}")
    print(f"解决方案: {result.get('solution')}")
    print(f"RAG上下文: {result.get('rag_context', '无')[:200]}...")