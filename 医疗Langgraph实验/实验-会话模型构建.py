import json
import logging
import os
import uuid
from datetime import datetime
from functools import lru_cache
from typing import List, Dict, Any, Optional

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import TypedDict, Annotated

# 导入你的自定义模块（请根据实际路径调整）
from MedicalDataDiscovery import MedicalDataDiscovery
from knowledge_graph_tools import Neo4jQueryTools

# 导入存储模块
from session_store import SessionStore

# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 常量配置 ====================
OLLAMA_CONFIG = {
    "model": "qwen3:8b",
    "base_url": "http://127.0.0.1:11434",
    "temperature": 0.0
}

PROMPT_PATHS = {
    "intent_sys": "./prompt/prompt_medical_intent_info_sys.txt",
    "intent_user": "./prompt/prompt_medical_intent_info_user.txt"
}

# ==================== Graph state（扩展 session_id）====================
class MedicalChatState(TypedDict):
    """会话状态结构，增加 session_id 用于持久化标识"""
    messages: Annotated[List[BaseMessage], add_messages]
    intentions: Annotated[List[HumanMessage], add_messages]
    discovery_data: Dict[str, Any]
    solution: str
    chat_intention_router: str
    session_id: str                     # 新增：会话唯一标识，用于 checkpointer

# ==================== 模型获取（缓存）====================
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

# ==================== 提示词文件读取（缓存）====================
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
    医疗对话助手，支持三种持久化模式：
    1. 无持久化：不传入任何存储参数
    2. 短期方案：传入 session_store (SessionStore 实例)
    3. 长期方案：传入 checkpointer (LangGraph 检查点实例，如 SqliteSaver)
    """

    def __init__(self, session_store: Optional[SessionStore] = None,
                 checkpointer: Optional[Any] = None):
        """
        :param session_store: SessionStore 实例（短期方案）
        :param checkpointer: LangGraph 检查点实例，需实现 get_tuple/put 等方法（长期方案）
        """
        logger.info("初始化 MedicalChat 助手实例")
        self._intent_sys_prompt = read_prompt(PROMPT_PATHS["intent_sys"])
        self._intent_user_template = read_prompt(PROMPT_PATHS["intent_user"])
        self._model = get_base_chat_model()
        self._medical_data_discovery = MedicalDataDiscovery()
        self._knowledge_query_tools = Neo4jQueryTools()

        # 持久化组件（二选一）
        self._session_store = session_store
        self._checkpointer = checkpointer

        if session_store and checkpointer:
            raise ValueError("不能同时指定 session_store 和 checkpointer，请选择一种持久化方式")
        if checkpointer is not None:
            logger.info("使用 LangGraph Checkpointer（长期方案）")
        elif session_store is not None:
            logger.info("使用 SessionStore（短期方案）")
        else:
            logger.info("不启用持久化（仅内存）")

        # 编译工作流（如果使用 checkpointer，则在编译时传入）
        self._workflow_graph = self._build_graph()

    # ---------- 工作流节点（保持原有逻辑，仅补充 session_id 到 state）----------
    def medical_intention_search(self, state: MedicalChatState) -> Dict[str, List[HumanMessage]]:
        """获取当前用户会话的意图信息"""
        tmp_messages = state.get("messages", [])
        tmp_intentions = state.get("intentions", [])
        prompt_user = PromptTemplate.from_template(self._intent_user_template).format(
            session_text=tmp_messages,
            intent_history=tmp_intentions
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
        logger.info(f"探寻结果: {medical_discovery_data}")
        return {"discovery_data": medical_discovery_data}

    def answer_question(self, state: MedicalChatState) -> Dict[str, str]:
        """
        根据探寻信息和意图信息生成回答内容
        """
        tmp_messages = state.get("messages", [])
        tmp_discovery_data = state.get("discovery_data", {})

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
        tmp_disease_keyword = """
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
            "疾病名称": model_rsp_messages,
            "RAG Chunk": context_str,
            "药物": medication,
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
        logger.info("当前会话聊天意图探寻：" + intention_router)
        return intention_router

    # ---------- 构建工作流（支持 Checkpointer）----------
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

        # 编译时若提供了 checkpointer，则传入
        if self._checkpointer is not None:
            return workflow.compile(checkpointer=self._checkpointer)
        else:
            return workflow.compile()

    # ---------- 核心交互接口：处理单条消息 ----------
    def process_message(self, user_input: str, session_id: Optional[str] = None) -> str:
        """
        处理用户消息，返回助手回复。
        - 若启用持久化且 session_id 为 None，则自动创建新会话。
        - 若启用持久化且 session_id 存在，则加载历史状态后继续。
        - 若不启用持久化，则需要外部手动管理状态（本方法不适用，请直接调用工作流）。
        """
        # 1. 根据持久化方式获取/创建 state 和 config
        if self._checkpointer is not None:
            # 长期方案：使用 LangGraph Checkpointer
            if session_id is None:
                session_id = str(uuid.uuid4())  # 生成新会话ID
                logger.info(f"创建新会话 (checkpointer): {session_id}")
            # 构造 config，LangGraph 会根据 thread_id 自动加载/保存状态
            config = {"configurable": {"thread_id": session_id}}
            # 为了遵循工作流输入，我们需要准备初始 state（如果是从 checkpointer 恢复，工作流内部会处理）
            # 但第一次调用时 state 应为空，但必须包含所有必需字段。
            # 我们可以先尝试从 checkpointer 中读取状态，如果不存在则创建空状态。
            checkpoint_tuple = self._checkpointer.get_tuple(config)
            if checkpoint_tuple is None:
                # 新建状态
                current_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "intentions": [],
                    "discovery_data": {},
                    "solution": "",
                    "chat_intention_router": "",  # 初始为空，符合业务需求
                    "session_id": session_id
                }
            else:
                # 恢复状态后追加用户消息
                current_state = checkpoint_tuple.checkpoint.values
                current_state.setdefault("messages", []).append(HumanMessage(content=user_input))
            # 调用工作流，checkpointer 会自动保存新状态
            final_state = self._workflow_graph.invoke(current_state, config=config)
            answer = final_state.get("solution", "")
            # 由于 checkpointer 已保存全状态，不需要额外持久化
            return answer

        elif self._session_store is not None:
            # 短期方案：使用 SessionStore
            if session_id is None:
                # 创建新会话
                session_id = self._session_store.create_session()
                logger.info(f"创建新会话 (SessionStore): {session_id}")
                # 初始状态下 chat_intention_router 必须为空字符串（而非默认路由）
                state = {
                    "messages": [],
                    "intentions": [],
                    "discovery_data": {},
                    "solution": "",
                    "chat_intention_router": "",  # 新增注释：初始为空，第一次对话时由意图识别节点决定路由
                    "session_id": session_id
                }
            else:
                restored = self._session_store.restore_state(session_id)
                if restored is None:
                    raise ValueError(f"会话 {session_id} 不存在")
                state = restored
                state["solution"] = ""

            # 追加用户消息
            state["messages"].append(HumanMessage(content=user_input))
            # 记录旧状态长度，用于增量保存
            old_msg_len = len(state["messages"]) - 1   # 刚添加的用户消息
            old_intent_len = len(state["intentions"])

            # 执行工作流（无 checkpointer）
            new_state = self._workflow_graph.invoke(state)

            # 将 AI 回复追加到 messages（工作流不会自动追加）
            answer = new_state.get("solution", "")
            if answer:
                ai_msg = AIMessage(content=answer)
                new_state.setdefault("messages", []).append(ai_msg)

            # 增量保存消息（原有逻辑，无需改动）
            for msg in new_state["messages"][old_msg_len:]:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                self._session_store.add_message(session_id, role, msg.content)

            # ========== 关键修改：保存意图轮次（一对一存储） ==========
            # 【新增】使用 add_intention_round 将本轮意图、发现数据、路由一并存储
            if len(new_state["intentions"]) > old_intent_len:
                # 获取本轮新增的意图（最后一个）
                new_intent = new_state["intentions"][-1]
                # 获取本轮对应的发现数据（工作流执行后的最新值）
                current_discovery = new_state.get("discovery_data", {})
                # 获取本轮对应的路由选择（可能为空或已有值）
                current_router = new_state.get("chat_intention_router", "")
                self._session_store.add_intention_round(
                    session_id,
                    intention_message=new_intent,
                    discovery_data=current_discovery,
                    router_value=current_router
                )
                logger.info(f"已保存本轮意图轮次: router={current_router}")

            return answer

        else:
            raise RuntimeError("未启用任何持久化方案，无法使用 process_message。请直接调用工作流或启用持久化。")

    # ---------- 交互式命令行（支持会话选择）----------
    def run_interactive(self):
        """启动命令行交互，支持根据持久化类型选择历史会话"""
        if self._checkpointer is None and self._session_store is None:
            print("错误：未启用任何持久化，无法运行交互模式。")
            return

        print("=" * 50)
        print("医疗对话助手")
        print("1. 新建会话")
        print("2. 加载历史会话")
        choice = input("请选择 (1/2): ").strip()

        session_id = None
        if choice == "2":
            # 列出历史会话
            if self._session_store:
                sessions = self._session_store.list_sessions()
            else:  # checkpointer 模式，需要从检查点中列出所有 thread_id（依赖于具体实现，这里简化）
                print("在 Checkpointer 模式下，暂不支持列出历史会话，将创建新会话。")
                sessions = []
            if not sessions:
                print("暂无历史会话，将创建新会话")
            else:
                print("\n历史会话列表：")
                for idx, sess in enumerate(sessions, 1):
                    print(f"{idx}. {sess['title']} (最后更新: {sess['updated_at']})")
                try:
                    idx = int(input("请输入序号: ")) - 1
                    session_id = sessions[idx]["session_id"]
                except (ValueError, IndexError):
                    print("无效输入，将创建新会话")

        print("\n输入 'exit' 退出对话")
        while True:
            user_input = input("[用户] >>> ").strip()
            if user_input.lower() in ("exit", "quit"):
                break
            if not user_input:
                continue
            try:
                answer = self.process_message(user_input, session_id)
                print(f"[助手]\n{answer}")
            except Exception as e:
                logger.exception(f"处理失败: {e}")
                print(f"错误: {e}")

    # 辅助：打印完整状态
    def _print_state_full(self, state: MedicalChatState, title: str = "State"):
        """完整打印 state 中的所有数据内容（谨慎使用，可能产生大量输出）"""
        logger.info(f"========== {title} ==========")
        msgs = state.get("messages", [])
        logger.info(f"messages 列表 (共 {len(msgs)} 条):")
        for i, msg in enumerate(msgs, 1):
            logger.info(f"  [{i}] {msg.__class__.__name__}: {msg.content}")
        intents = state.get("intentions", [])
        logger.info(f"intentions 列表 (共 {len(intents)} 条):")
        for i, intent in enumerate(intents, 1):
            logger.info(f"  [{i}] {intent.content}")
        disc_data = state.get("discovery_data", {})
        logger.info(f"discovery_data 字典 (共 {len(disc_data)} 个键):")
        for key, value in disc_data.items():
            logger.info(f"  {key}: {value}")
        solution = state.get("solution", "")
        logger.info(f"solution 内容:\n{solution}")
        router = state.get("chat_intention_router", "未设置")
        logger.info(f"chat_intention_router: {router}")
        logger.info(f"session_id: {state.get('session_id', '未设置')}")
        logger.info(f"========== {title} 结束 ==========")


# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 示例1：短期方案（SessionStore）
    print("=== 短期方案示例 ===")
    store = SessionStore()
    assistant1 = MedicalChat(session_store=store)
    assistant1.run_interactive()

    # 示例2：长期方案（LangGraph SqliteSaver）
    # 需要先安装 langgraph 并导入 SqliteSaver
    # from langgraph.checkpoint.sqlite import SqliteSaver
    # checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
    # assistant2 = MedicalChat(checkpointer=checkpointer)
    # assistant2.run_interactive()