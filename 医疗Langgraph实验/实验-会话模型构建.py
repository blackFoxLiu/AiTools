import operator
import uuid
from typing import List, Any, Annotated

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import TypedDict

from MedicalDataDiscovery import MedicalDataDiscovery
from common_utils import *
from knowledge_graph_tools import Neo4jQueryTools
# 导入存储模块
from session_store import SessionStore

# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROMPT_PATHS = {
    "intent_sys": "./prompt/prompt_medical_intent_info_sys.txt",
    "intent_user": "./prompt/prompt_medical_intent_info_user.txt",
    "chat_intent_choice_sys": "./prompt/chat/prompt_chat_intent_choice_sys.txt",
    "chat_intent_choice_user": "./prompt/chat/prompt_chat_intent_choice_user.txt",
    "medication_keyword_sys": "./prompt/chat/prompt_medication_keyword_sys.txt",
    "medication_keyword_user": "./prompt/chat/prompt_medication_keyword_user.txt",
    "medical_chat_answer_sys": "./prompt/chat/prompt_medical_chat_answer_sys.txt",
    "medical_chat_answer_user": "./prompt/chat/prompt_medical_chat_answer_user.txt",
}

# ==================== Graph state（使用纯字典存储意图）====================
class MedicalChatState(TypedDict):
    """会话状态结构，intentions 为纯字典列表，每个字典包含 main_intention 和 sub_operate"""
    messages: Annotated[List[BaseMessage], add_messages]
    intentions: Annotated[List[Dict[str, Any]], operator.add]   # 改为字典列表，使用 operator.add 追加
    discovery_data: Dict[str, Any]
    solution: str
    chat_intention_router: str
    session_id: str

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
        logger.info("初始化 MedicalChat 助手实例")
        self._intent_sys_prompt = read_prompt(PROMPT_PATHS["intent_sys"])
        self._intent_user_template = read_prompt(PROMPT_PATHS["intent_user"])
        self._model = get_base_chat_model()
        self._medical_data_discovery = MedicalDataDiscovery()
        self._knowledge_query_tools = Neo4jQueryTools()

        # 持久化组件（二选一）
        self._session_store = session_store
        self._checkpointer = checkpointer
        self.session_id = None

        if session_store and checkpointer:
            raise ValueError("不能同时指定 session_store 和 checkpointer，请选择一种持久化方式")
        if checkpointer is not None:
            logger.info("使用 LangGraph Checkpointer（长期方案）")
        elif session_store is not None:
            logger.info("使用 SessionStore（短期方案）")
        else:
            logger.info("不启用持久化（仅内存）")

        self._workflow_graph = self._build_graph()

    # ---------- 工作流节点 ----------
    def medical_intention_search_node(self, state: MedicalChatState) -> Dict[str, List[Dict[str, Any]]]:
        """
        获取当前用户会话的意图信息，返回字典格式的意图
        """
        logger.info("medical_intention_search_node-获取当前用户会话的意图信息")
        tmp_messages = state.get("messages", [])
        tmp_intentions = state.get("intentions", [])   # 现在是字典列表

        # 将意图历史转换为可读字符串
        intent_history_str = ""
        for i, intent in enumerate(tmp_intentions, 1):
            main_intention = intent.get("main_intention", "")
            sub_ops = intent.get("sub_operate", [])
            sub_str = "、".join(sub_ops) if sub_ops else ""
            intent_history_str += f"第{i}轮意图：{main_intention}（细分：{sub_str}）\n"

        prompt_user = PromptTemplate.from_template(self._intent_user_template).format(
            session_text=tmp_messages,
            intent_history=intent_history_str
        )
        system_messages = [SystemMessage(content=self._intent_sys_prompt)]
        human_messages = [HumanMessage(content=prompt_user)]

        model_rsp = self._model.invoke(system_messages + human_messages)
        content = model_rsp.content

        # 期望模型返回 JSON 字符串，解析为字典
        try:
            intention_dict = json.loads(content)
            if "main_intention" not in intention_dict:
                intention_dict["main_intention"] = content
            if "sub_operate" not in intention_dict:
                intention_dict["sub_operate"] = []
        except json.JSONDecodeError:
            # 降级处理：将整个回复作为 main_intention
            intention_dict = {
                "main_intention": content,
                "sub_operate": []
            }
            logger.error("intentions-JSONDecodeError，进行降级处理")
        return {"intentions": [intention_dict]}

    def medical_data_discovery_node(self, state: MedicalChatState) -> Dict[str, Dict[str, Any]]:
        """进行医疗信息探寻"""
        logger.info("medical_data_discovery_node-进行医疗信息探寻")
        tmp_messages = state.get("messages", [])
        tmp_discovery_data = state.get("discovery_data", {})
        medical_discovery_data = self._medical_data_discovery.get_medical_data_discovery(tmp_messages, tmp_discovery_data)

        rst_discovery_data = {k: v for k, v in medical_discovery_data.items() if k != "messages"}
        logger.info(f"探寻结果: {rst_discovery_data}")
        return {"discovery_data": rst_discovery_data}

    def answer_question_node(self, state: MedicalChatState) -> Dict[str, str]:
        """ 根据探寻信息和意图信息生成回答内容 """
        logger.info("answer_question_node-回答问题")
        tmp_messages = state.get("messages", [])
        tmp_discovery_data = state.get("discovery_data", {})
        tmp_intentions = state.get("intentions", [])

        latest_intention = tmp_intentions[-1] if tmp_intentions else {}
        main_intention = latest_intention.get("main_intention", "")

        prompt_medical_chat_answer_sys = read_prompt(PROMPT_PATHS["medical_chat_answer_sys"])
        prompt_medical_chat_answer_user = PromptTemplate.from_template(PROMPT_PATHS["medical_chat_answer_user"]).format(
            input_messages = tmp_messages,
            input_discovery_data = tmp_discovery_data,
            input_intention = main_intention
        )

        system_messages = [SystemMessage(content=prompt_medical_chat_answer_sys)]
        user_messages = [HumanMessage(content=prompt_medical_chat_answer_user)]

        model_rsp = self._model.invoke(system_messages + user_messages)
        logger.info("响应给用户的数据信息：" + model_rsp.content)
        return {"solution": model_rsp.content}

    def medication_data_discovery_node(self, state: MedicalChatState) -> Dict[str, Dict[str, Any]]:
        """根据当前的会话信息获取对应的药物内容"""
        tmp_messages = state.get("messages", [])

        prompt_medication_keyword_sys = read_prompt(PROMPT_PATHS["medication_keyword_sys"])
        prompt_medication_keyword_user = PromptTemplate.from_template(PROMPT_PATHS["medication_keyword_user"]).format(
            input_messages = tmp_messages[-1]
        )
        system_messages = [SystemMessage(content=prompt_medication_keyword_sys)]
        user_messages = [HumanMessage(content=prompt_medication_keyword_user)]
        model_rsp = self._model.invoke(system_messages + user_messages)
        disease_name = model_rsp.content.strip()
        logger.info(f"识别疾病：{disease_name}")

        cnt_query = self._knowledge_query_tools.check_disease_is_exists(disease_name)
        if cnt_query == 0:
            medication = ""
            no_eat_food = ""
            recommend_food = ""
        else:
            medication = self._knowledge_query_tools.query_medication_by_disease(disease_name)
            no_eat_food = self._knowledge_query_tools.query_no_eat_by_disease(disease_name)
            recommend_food = self._knowledge_query_tools.query_recommand_eat_by_disease(disease_name)

        results = self._medical_data_discovery.rag_service.query(f"疾病{disease_name}治疗方案？")
        context_str = ""
        if results:
            for i, item in enumerate(results, 1):
                context_str += f"【参考{i}】{item['text']}\n"
            logger.info(f"检索到 {len(results)} 条结果")
        else:
            context_str = "未检索到相关知识。"
            logger.info("未检索到相关知识")

        return {"discovery_data": {
            "疾病名称": disease_name,
            "RAG Chunk": context_str,
            "药物": medication,
            "不可使用食物": no_eat_food,
            "建议食用食物": recommend_food,
        }}

    def chat_intent_choice_node(self, state: MedicalChatState) -> Dict[str, Any]:
        """进行当前进行【症状探寻】和【药物探寻】的识别"""
        logger.info("chat_intent_choice_node 进入【症状信息】或【药物信息】探寻")
        tmp_messages = state.get("messages", [])
        logger.info(f"最新会话信息：{tmp_messages[-1]}")

        prompt_chat_intent_choice_sys = read_prompt(PROMPT_PATHS["chat_intent_choice_sys"])
        prompt_chat_intent_choice_user = PromptTemplate.from_template(PROMPT_PATHS["chat_intent_choice_user"]).format(
            input_messages = tmp_messages[-1]
        )
        system_messages = [SystemMessage(content=prompt_chat_intent_choice_sys)]
        user_messages = [HumanMessage(content=prompt_chat_intent_choice_user)]
        model_rsp = self._model.invoke(system_messages + user_messages)

        logger.info("响应给用户的数据信息：" + model_rsp.content)
        if model_rsp.content == "symptoms_inquiry":
            return {"chat_intention_router": "symptoms_inquiry"}
        return {"chat_intention_router": "medication_inquiry"}

    def chat_medical_router(self, state: MedicalChatState) -> str:
        """
        路由选择工具，选择 药物信息探寻 和 症状信息探寻
        :param state:
        :return:
        """
        logger.info("chat_medical_router 进行症状验证和药物验证")

        intention_router = state.get("chat_intention_router", "symptoms_inquiry")
        logger.info("当前会话聊天意图探寻：" + intention_router)
        return intention_router

    # ---------- 构建工作流 ----------
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(MedicalChatState)

        # 添加节点
        # 判断当前结点【病情探寻】和【药物查询】
        workflow.add_node("chat_intent_choice_node", self.chat_intent_choice_node)

        # 药物信息探寻
        workflow.add_node("medication_data_discovery_node", self.medication_data_discovery_node)

        # 症状信息探寻
        workflow.add_node("medical_intention_search_node", self.medical_intention_search_node)
        workflow.add_node("medical_data_discovery_node", self.medical_data_discovery_node)
        workflow.add_node("answer_question_node", self.answer_question_node)

        # 【症状探寻】和【疾病探寻】路由
        workflow.add_conditional_edges(
            "chat_intent_choice_node",
            self.chat_medical_router,
            {
                "medication_inquiry": "medication_data_discovery_node",
                "symptoms_inquiry": "medical_intention_search_node"
            }
        )

        # 添加边：START → 意图分析 → 数据探寻 → 回答生成 → END
        # 症状信息探寻路径
        workflow.add_edge(START, "chat_intent_choice_node")
        workflow.add_edge("medical_intention_search_node", "medical_data_discovery_node")
        workflow.add_edge("medical_data_discovery_node", "answer_question_node")

        # 药物信息探寻路径
        workflow.add_edge("medication_data_discovery_node", "answer_question_node")
        workflow.add_edge("answer_question_node", END)

        # 编译时若提供了 checkpointer，则传入
        if self._checkpointer is not None:
            return workflow.compile(checkpointer=self._checkpointer)
        else:
            return workflow.compile()

    # ---------- 核心交互接口 ----------
    def process_message(self, user_input: str) -> str:
        """
        处理用户消息，返回助手回复。
        支持 SessionStore 和 Checkpointer 两种持久化方式。
        """
        # 长期方案：Checkpointer
        if self._checkpointer is not None:
            if self.session_id is None:
                self.session_id = str(uuid.uuid4())
                logger.info(f"创建新会话 (checkpointer): {self.session_id}")
            config = {"configurable": {"thread_id": self.session_id}}
            checkpoint_tuple = self._checkpointer.get_tuple(config)
            if checkpoint_tuple is None:
                current_state = {
                    "messages": [HumanMessage(content=user_input)],
                    "intentions": [],
                    "discovery_data": {},
                    "solution": "",
                    "chat_intention_router": "",
                    "session_id": self.session_id
                }
            else:
                # 恢复状态后追加用户消息
                current_state = checkpoint_tuple.checkpoint.values
                current_state.setdefault("messages", []).append(HumanMessage(content=user_input))
            # 调用工作流，checkpointer 会自动保存新状态
            final_state = self._workflow_graph.invoke(current_state, config=config)
            return final_state.get("solution", "")

        # 短期方案：SessionStore
        elif self._session_store is not None:
            # 短期方案：使用 SessionStore
            if self.session_id is None:
                self.session_id = self._session_store.create_session()
                logger.info(f"创建新会话 (SessionStore): {self.session_id}")
                # 初始状态下 chat_intention_router 必须为空字符串（而非默认路由）
                state = {
                    "messages": [],
                    "intentions": [],
                    "discovery_data": {},
                    "solution": "",
                    "chat_intention_router": "",
                    "session_id": self.session_id
                }
            else:
                restored = self._session_store.restore_state(self.session_id)
                if restored is None:
                    raise ValueError(f"会话 {self.session_id} 不存在")
                state = restored
                state["solution"] = ""

            # 追加用户消息
            state["messages"].append(HumanMessage(content=user_input))
            old_intent_len = len(state["intentions"])   # intentions 现在是字典列表

            # 执行工作流
            new_state = self._workflow_graph.invoke(state)

            # 获取回答
            answer = new_state.get("solution", "")
            if answer:
                ai_msg = AIMessage(content=answer)
                new_state.setdefault("messages", []).append(ai_msg)

            # 增量保存消息
            old_msg_len = len(state["messages"]) - 1   # 刚添加的用户消息前长度
            for msg in new_state["messages"][old_msg_len:]:
                role = "user" if isinstance(msg, HumanMessage) else "assistant"
                self._session_store.add_message(self.session_id, role, msg.content)

            # 保存意图轮次
            if len(new_state["intentions"]) > old_intent_len:
                new_intent_dict = new_state["intentions"][-1]
                current_discovery = new_state.get("discovery_data", {})
                current_router = new_state.get("chat_intention_router", "")
                self._session_store.add_intention_round(
                    self.session_id,
                    intention_dict=new_intent_dict,
                    discovery_data=current_discovery,
                    router_value=current_router
                )
                logger.info(f"已保存本轮意图轮次: router={current_router}")

            return answer
        else:
            raise RuntimeError("未启用任何持久化方案，无法使用 process_message。")

    # ---------- 交互式命令行 ----------
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

        if choice == "2":
            # 列出历史会话
            if self._session_store:
                sessions = self._session_store.list_sessions()
            else:
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
                    self.session_id = sessions[idx]["session_id"]
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
                answer = self.process_message(user_input)
                print(f"[助手]\n{answer}")
            except Exception as e:
                logger.exception(f"处理失败: {e}")
                print(f"错误: {e}")


# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 短期方案（SessionStore）
    print("=== 短期方案示例 ===")
    store = SessionStore()
    assistant1 = MedicalChat(session_store=store)
    assistant1.run_interactive()

    # 长期方案（LangGraph SqliteSaver）需要额外安装，此处注释
    # from langgraph.checkpoint.sqlite import SqliteSaver
    # checkpointer = SqliteSaver.from_conn_string("checkpoints.db")
    # assistant2 = MedicalChat(checkpointer=checkpointer)
    # assistant2.run_interactive()