import logging
from typing import List, Dict, Any, Optional  # 优化：补充类型提示
from functools import lru_cache  # 优化：引入缓存装饰器

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import TypedDict, Annotated

from MedicalDataDiscovery import MedicalDataDiscovery
from knowledge_graph_tools import Neo4jQueryTools

# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 常量配置（优化：集中管理配置项）====================
OLLAMA_CONFIG = {
    "model": "qwen3:8b",
    "base_url": "http://127.0.0.1:11434",
    "temperature": 0.0
}

PROMPT_PATHS = {
    "intent_sys": "./prompt/prompt_medical_intent_info_sys.txt",
    "intent_user": "./prompt/prompt_medical_intent_info_user.txt"
}


# ==================== Graph state ====================
class MedicalChatState(TypedDict):
    """会话状态结构 """
    messages: Annotated[List[BaseMessage], add_messages]
    intentions: Annotated[List[HumanMessage], add_messages]
    discovery_data: Dict[str, Any]
    solution: str
    chat_intention_router: str


# ==================== 模型获取（优化：添加缓存避免重复创建）====================
@lru_cache(maxsize=1)
def get_base_chat_model() -> ChatOllama:
    """
    获取一个 Ollama 聊天模型实例（单例模式）
    优化：使用 lru_cache 确保全局只有一个实例，节省资源
    """
    return ChatOllama(
        model=OLLAMA_CONFIG["model"],
        base_url=OLLAMA_CONFIG["base_url"],
        temperature=OLLAMA_CONFIG["temperature"]
    )


# ==================== 提示词文件读取（优化：添加缓存和更好的异常处理）====================
@lru_cache(maxsize=5)
def read_prompt(file_path: str) -> str:
    """
    读取提示词文件，支持缓存
    优化：文件内容缓存后无需重复 I/O
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"错误：提示词文件不存在 - {file_path}")
    except Exception as e:
        logger.error(f"读取提示词文件失败：{e}")
    return ""


# ==================== 医疗对话助手工具类 ====================
class MedicalChat:
    """
    医疗对话助手工具类
    优化：完善类文档，职责清晰：管理 LangGraph 工作流及对话状态
    """

    def __init__(self):
        """初始化助手（优化：提前加载提示词和模型，避免运行时重复加载）"""
        logger.info("初始化 MedicalChat 助手实例")
        # 优化：将提示词内容预先读取为实例属性，避免每次调用都读文件
        self._intent_sys_prompt = read_prompt(PROMPT_PATHS["intent_sys"])
        self._intent_user_template = read_prompt(PROMPT_PATHS["intent_user"])
        self._model = get_base_chat_model()  # 复用单例模型
        # 优化：编译工作流，后续直接调用
        self._workflow_graph = self._build_graph()
        self._medical_data_discovery = MedicalDataDiscovery()
        self._knowledge_query_tools = Neo4jQueryTools()

    # ---------- 以下三个方法为工作流节点，逻辑保持原样（仅补充注释）----------
    def medical_intention_search(self, state: MedicalChatState) -> Dict[str, List[HumanMessage]]:
        """
        获取当前用户会话的意图信息
        优化：添加类型提示和更清晰的变量命名
        """
        tmp_messages = state.get("messages", [])
        tmp_intentions = state.get("intentions", [])
        # 使用实例属性的提示词和模型
        prompt_user = PromptTemplate.from_template(self._intent_user_template).format(
            session_text=tmp_messages,
            intent_history=tmp_intentions  # TODO 拆解
        )
        system_messages = [SystemMessage(content=self._intent_sys_prompt)]
        human_messages = [HumanMessage(content=prompt_user)]

        model_rsp_messages = self._model.invoke(system_messages + human_messages)
        new_intention = HumanMessage(content=model_rsp_messages.content)
        return {"intentions": [new_intention]}

    def medical_data_discovery(self, state: MedicalChatState) -> Dict[str, Dict[str, Any]]:
        """
        进行医疗信息探寻
        """
        tmp_messages = state.get("messages", [])
        tmp_discovery_data = state.get("discovery_data", {})
        medical_discovery_data = self._medical_data_discovery.get_medical_data_discovery(tmp_messages, tmp_discovery_data)
        logger.info(f"探寻结果: {medical_discovery_data}")  # 优化：用 logger 替代 print
        return {"discovery_data": medical_discovery_data}

    def answer_question(self, state: MedicalChatState) -> Dict[str, str]:
        """
        根据探寻信息和意图信息生成回答内容
        """
        tmp_messages = state.get("messages", [])
        tmp_discovery_data = state.get("discovery_data", {})
        # 优化：提取 prompt 为可读常量（原逻辑未改，仅调整格式）
        tmp_prompt = """
            根据输入的数据，根据用户的意图信息和想要知道的信息，生成用户想要知道的内容信息和需要提问的信息。
            输入格式为：
            提问回答信息：
            str——分点进行展示
            询问信息：
            str——根据意图信息和实际内容信息，判断需要用户回答的信息。
        """
        tmp_input_user_info = f"""
            用户历史会话信息：{tmp_messages}

            用户探寻知识信息：{tmp_discovery_data}
        """
        system_messages = [SystemMessage(content=tmp_prompt)]
        user_messages = [HumanMessage(content=tmp_input_user_info)]
        model_rsp_messages = self._model.invoke(system_messages + user_messages)
        logger.info("响应给用户的数据信息：" + model_rsp_messages.content)
        return {"solution": model_rsp_messages.content}

    def medication_data_discovery(self, state: MedicalChatState) -> Dict[str, Dict[str, Any]]:
        """
            根据当前的会话信息获取查询对应的药物内容
        :param state:
        :return:
        """
        tmp_messages = state.get("messages", [])
        tmp_disease_keyword = f"""
            获取当前用户的疾病信息，并仅输出用户确定的疾病信息：
        """
        tmp_input_user_info = f"""
            用户输出的会话信息为：
                {tmp_messages[-1]}
        """
        system_messages = [SystemMessage(content=tmp_disease_keyword)]
        user_messages = [HumanMessage(content=tmp_input_user_info)]
        model_rsp_messages = self._model.invoke(system_messages + user_messages)
        disease_name = model_rsp_messages.content
        logger.info("响应给用户的数据信息：" + disease_name)

        cnt_query = self._knowledge_query_tools.check_disease_is_exists(disease_name)
        if cnt_query == 0:
            medication = ""
            no_eat_food = ""
            recommend_food = ""
        else:
            medication = self._knowledge_query_tools.query_medication_by_disease(disease_name)
            no_eat_food = self._knowledge_query_tools.query_no_eat_by_disease(disease_name)
            recommend_food = self._knowledge_query_tools.query_recommand_eat_by_disease(disease_name)

        # 获取对应疾病的治疗方案——RAG逻辑
        results = self._medical_data_discovery.rag_service.query("疾病"+model_rsp_messages.content+"治疗方案？")
        context_str = ""
        if results:
            for i, item in enumerate(results, 1):
                context_str += f"【参考{i}】{item['text'][:300]}...\n"
            logger.info(f"  检索到 {len(results)} 条结果")
        else:
            context_str = "未检索到相关知识。"
            logger.info("未检索到相关知识")

        return {"discovery_data": {
            "疾病名称":model_rsp_messages,
            "RAG Chunk": context_str,
            "药物":medication,
            "不可使用食物": no_eat_food,
            "建议食用食物": recommend_food,
        }}

    def chat_intent_choice(self, state: MedicalChatState) -> Dict[str, Any]:
        """
        进行当前进行【症状探寻】和【药物探寻】的识别
        :param state:
        :return:
        """
        logger.info("chat_intent_choice 进入【症状信息】或【药物信息】探寻")
        tmp_messages = state.get("messages", [])
        logger.info(f"最新会话信息：{tmp_messages[-1]}")
        tmp_prompt = f"""
            根据当前的会话信息判断当前用户最新的会话信息，是需要使用进行【药物资料信息查询】，还是需要使用【病情症状查询】。如果信息的会话信息中提供的是具体的疾病信息，必须选择【medication_inquiry】
            输入内容只能是：
            `medication_inquiry` 或 `symptoms_inquiry`
        """
        tmp_input_user_info = f"""
            用户输出的会话信息为：
                {tmp_messages[-1]}
        """
        system_messages = [SystemMessage(content=tmp_prompt)]
        user_messages = [HumanMessage(content=tmp_input_user_info)]
        model_rsp_messages = self._model.invoke(system_messages + user_messages)
        logger.info("响应给用户的数据信息：" + model_rsp_messages.content)
        if model_rsp_messages.content == "symptoms_inquiry":
            return {"chat_intention_router": "symptoms_inquiry"}
        return {"chat_intention_router": "medication_inquiry"}

    def chat_medical_router(self, state: MedicalChatState) -> str:
        """
            根据当前会话信息，是需要进行【病症信息探寻】还是【药物信息探寻】
        :return:
        """
        intention_router = state.get("chat_intention_router", "symptoms_inquiry")
        logger.info("当前会话聊天意图探寻："+intention_router)
        return intention_router

    # ---------- 工作流构建（优化：提取为私有方法，提高可读性）----------
    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流（优化：明确返回类型，添加节点描述注释）"""
        workflow = StateGraph(MedicalChatState)

        # 添加节点
        # 判断当前结点【病情探寻】和【药物查询】
        workflow.add_node("chat_intent_choice", self.chat_intent_choice)

        # 药物信息探寻
        workflow.add_node("medication_data_discovery", self.medication_data_discovery)

        # 症状信息探寻
        workflow.add_node("medical_intention_search", self.medical_intention_search)
        workflow.add_node("medical_data_discovery", self.medical_data_discovery)
        workflow.add_node("answer_question", self.answer_question)

        # 【症状探寻】和【疾病探寻】路由
        workflow.add_conditional_edges(
            "chat_intent_choice",
            self.chat_medical_router,  # 静态方法
            {
                "medication_inquiry": "medication_data_discovery",
                "symptoms_inquiry": "medical_intention_search"
            }
        )

        # 添加边：START → 意图分析 → 数据探寻 → 回答生成 → END
        # 症状信息探寻路径
        workflow.add_edge(START, "chat_intent_choice")
        workflow.add_edge("medical_intention_search", "medical_data_discovery")
        workflow.add_edge("medical_data_discovery", "answer_question")

        # 药物信息探寻路径
        workflow.add_edge("medication_data_discovery", "answer_question")
        workflow.add_edge("answer_question", END)

        return workflow.compile()

    def _print_state_full(self, state: MedicalChatState, title: str = "State"):
        """完整打印 state 中的所有数据内容（谨慎使用，可能产生大量输出）"""
        logger.info(f"========== {title} ==========")

        # 1. messages
        msgs = state.get("messages", [])
        logger.info(f"messages 列表 (共 {len(msgs)} 条):")
        for i, msg in enumerate(msgs, 1):
            logger.info(f"  [{i}] {msg.__class__.__name__}: {msg.content}")

        # 2. intentions
        intents = state.get("intentions", [])
        logger.info(f"intentions 列表 (共 {len(intents)} 条):")
        for i, intent in enumerate(intents, 1):
            logger.info(f"  [{i}] {intent.content}")

        # 3. discovery_data
        disc_data = state.get("discovery_data", {})
        logger.info(f"discovery_data 字典 (共 {len(disc_data)} 个键):")
        for key, value in disc_data.items():
            # 如果 value 是字符串且过长，可截断；但按“完整”要求，这里直接打印
            logger.info(f"  {key}: {value}")

        # 4. solution
        solution = state.get("solution", "")
        logger.info(f"solution 内容:")
        logger.info(f"{solution}")

        # 5. chat_intention_router
        router = state.get("chat_intention_router", "未设置")
        logger.info(f"chat_intention_router: {router}")

        logger.info(f"========== {title} 结束 ==========")

    # ---------- 对外暴露的聊天接口（优化：新增便捷方法）----------
    def process_message(self, user_input: str, state: MedicalChatState) -> MedicalChatState:
        """
        处理单条用户消息，返回更新后的状态
        优化：封装单轮对话调用，便于集成
        """
        # 将用户消息添加到历史
        state.setdefault("messages", []).append(HumanMessage(content=user_input))
        # 调用工作流
        new_state = self._workflow_graph.invoke(state)
        # 将助手的回复作为 AIMessage 追加（工作流不会自动添加，需手动维护）
        self._print_state_full(new_state)
        answer = new_state.get("solution", "")
        if answer:
            new_state.setdefault("messages", []).append(AIMessage(content=answer))
        return new_state

    def run_interactive(self):
        """
        启动人工交互对话循环
        优化：将 main 中的交互循环封装为类方法，展示工具类的便捷使用
        """
        print("=" * 50)
        print("欢迎使用医疗对话助手（人工交互模式）")
        print("提示：输入 'exit' 或 'quit' 结束对话")
        print("=" * 50)

        # 初始化会话状态
        state: MedicalChatState = {
            "messages": [],
            "intentions": [],
            "discovery_data": {},
            "solution": ""
        }

        while True:
            user_input = input("\n[用户] >>> ").strip()
            if user_input.lower() in ("exit", "quit"):
                print("[系统] 对话已结束。")
                break
            if not user_input:
                print("[系统] 输入不能为空，请重新输入。")
                continue

            try:
                state = self.process_message(user_input, state)
                answer = state.get("solution", "")
                if not answer:
                    answer = "[系统] 抱歉，未能生成有效回答。"
                print("\n[助手]")
                print(answer)
            except Exception as e:
                logger.exception(f"处理消息时发生异常: {e}")  # 优化：记录完整堆栈
                print(f"[错误] 处理请求时发生异常：{e}")

        print("\n[系统] 感谢使用，再见！")


# ==================== 展示工具类用法 ====================
if __name__ == '__main__':
    # 优化：主函数仅负责实例化工具类并启动交互，代码清晰直观
    assistant = MedicalChat()
    assistant.run_interactive()