import logging
from tkinter import Image
from typing import List

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END, add_messages
from typing_extensions import TypedDict, Annotated

from MedicalDataDiscovery import get_medical_data_discovery

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Graph state
class MedicalChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intentions: Annotated[List[HumanMessage], add_messages]
    discovery_data: TypedDict
    question:str


def get_base_chat_model():
    """
    获取一个ollama会话模型
    :return:
    """
    return ChatOllama(
        model="qwen3:8b",
        base_url="http://127.0.0.1:11434",
        temperature=0.0
    )

# 读取提示词文件
def read_prompt(file_path: str) -> str:
    """读取提示词文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"错误：提示词文件不存在 - {file_path}")
    except Exception as e:
        logger.error(f"读取提示词文件失败：{e}")
    return ""

class MedicalChat:

    def __init__(self):
        logger.info("初始化")

    def medical_intention_search(self, state: MedicalChatState):
        """
            获取当前用户会话的意图信息
        :param state:
        :return:
        """
        tmp_messages = state.get("messages", [])
        tmp_intentions = state.get("intentions", [])
        ollama_model = get_base_chat_model()
        prompt_medical_intent_info_sys = read_prompt("./prompt/prompt_medical_intent_info_sys.txt")

        prompt_medical_intent_info_user = PromptTemplate.from_template(read_prompt("./prompt/prompt_medical_intent_info_user.txt")).format(
            session_text=tmp_messages,
            intent_history=tmp_intentions # TODO 拆解
        )

        system_messages = [SystemMessage(content=prompt_medical_intent_info_sys)]
        human_messages = [HumanMessage(content=prompt_medical_intent_info_user)]

        model_rsp_messages = ollama_model.invoke(system_messages + human_messages)
        new_intention = HumanMessage(content=model_rsp_messages.content)
        return {"intentions": [new_intention]}   # 注意 add_messages 会追加


    def medical_data_discovery(self, state: MedicalChatState):
        """
            进行医疗信息探寻
        :param state:
        :return:
        """
        tmp_messages = state.get("messages", [])
        medical_discovery_data = get_medical_data_discovery(tmp_messages)
        print(medical_discovery_data)

        return {"discovery_data": medical_discovery_data}


    def answer_question(self, state: MedicalChatState):
        """
            根据探寻信息和意图信息生成回答内容
        :param state:
        :return:
        """
        tmp_messages = state.get("messages", [])
        tmp_discovery_data = state.get("discovery_data", [])
        tmp_prompt = f"""
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
        ollama_model = get_base_chat_model()
        model_rsp_messages = ollama_model.invoke(system_messages+user_messages)
        logger.info("响应给用户的数据信息："+model_rsp_messages.content)
        return {"question": model_rsp_messages.content}


    def get_graph(self):
        # Build workflow
        workflow = StateGraph(MedicalChatState)

        # 医疗意图分析
        workflow.add_node("medical_intention_search", self.medical_intention_search)
        workflow.add_node("medical_data_discovery", self.medical_data_discovery)
        workflow.add_node("answer_question", self.answer_question)

        # Add edges to connect nodes
        workflow.add_edge(START, "medical_intention_search")
        workflow.add_edge("medical_intention_search", "medical_data_discovery")
        workflow.add_edge("medical_data_discovery", "answer_question")
        workflow.add_edge("answer_question", END)

        # Compile
        return workflow.compile()



# ==================== 新增人工对话功能 ====================
from langchain_core.messages import AIMessage

if __name__ == '__main__':

    """
    启动人工对话模式，支持多轮交互。
    用户输入问题后，系统调用原有的医疗问答流程，输出回答并询问补充信息。
    输入 'exit' 或 'quit' 退出对话。
    """
    print("=" * 50)
    print("欢迎使用医疗对话助手（人工交互模式）")
    print("提示：输入 'exit' 或 'quit' 结束对话")
    print("=" * 50)

    # 初始化会话状态（与 MedicalChatState 结构一致）
    state = {
        "messages": [],          # 对话历史（HumanMessage 和 AIMessage）
        "intentions": [],        # 意图分析历史（自动维护）
        "discovery_data": {},    # 医疗知识探寻结果
        "question": ""           # 上一次助手的完整回答
    }

    MedicalChat = MedicalChat()
    workflow_graph = MedicalChat.get_graph()
    while True:
        # 获取用户输入
        user_input = input("\n[用户] >>> ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("[系统] 对话已结束。")
            break
        if not user_input:
            print("[系统] 输入不能为空，请重新输入。")
            continue

        # 将用户消息添加到历史中
        state["messages"].append(HumanMessage(content=user_input))

        # 调用原有工作流，传入当前状态
        try:
            result = workflow_graph.invoke(state)
        except Exception as e:
            print(f"[错误] 处理请求时发生异常：{e}")
            continue

        # 更新状态为工作流返回的最新状态
        state.update(result)

        # 获取助手的回答（包含“提问回答信息”和“询问信息”）
        answer = state.get("question", "")
        if not answer:
            answer = "[系统] 抱歉，未能生成有效回答。"

        # 显示助手的回答
        print("\n[助手]")
        print(answer)

        # 将助手的回复作为 AIMessage 追加到消息历史中
        # （原有工作流不会自动添加 AIMessage，此处手动维护以保证多轮连贯性）
        state["messages"].append(AIMessage(content=answer))

    print("\n[系统] 感谢使用，再见！")
