from typing import Dict, Any, List, Optional, Annotated, TypedDict, Union

from langchain.tools import tool
from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages, MessagesState
from pydantic import BaseModel, Field

# 初始化模型
ollama_model = ChatOllama(
    model="qwen3:8b",
    base_url="http://127.0.0.1:11434",
    temperature=0.0
)

# 现病诱因 (TypedDict 版本)
class PresentIllness(TypedDict, total=False):
    onset_time: str              # 病开始时间
    pattern: str                 # 持续/阵发/间歇性等
    treatments_received: List[str]       # 接受什么治疗，包括吃过什么药物，做过什么医学治疗

# 设置 State 状态
class MedicalInquiryState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # 核心问询字段
    chief_complaint: str   # 主诉：喊着最主要痛苦/问题
    present_illness: PresentIllness
    associated_symptoms: List[str]

# 定义结构化输出的类（数据契约）(BaseModel 版本)
class PresentIllnessSchema(BaseModel):
    """主诉信息结构化数据"""
    onset_time: str = Field(description="病症开始时间")
    pattern: str = Field(description="疼痛或发作规律，如持续/阵发/间歇性")
    treatments_received: List[str] = Field(description="接受过的治疗或服用的药物清单")

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
def get_llm_agent(sys_prompt: str, tools):
    agent = create_agent(
        model=ollama_model,
        tools=tools or [],
        system_prompt=sys_prompt
    )
    return agent

def get_model_chief_complaint(state: MedicalInquiryState) -> str:
    messages = state["messages"]
    tmp_chief_complaint = """
		根据用户聊天信息获取其中的病人最大的病痛信息。应该表现出是症状，或者最大痛苦。只返回症状，必须精简。
        例如： 输入：早上有些发烧和头疼。 输出：发烧，头疼
    """
    model_chief_complaint = get_llm_agent(tmp_chief_complaint, None)
    model_rsp_message = model_chief_complaint.invoke({"messages": messages})
    return {"chief_complaint": model_rsp_message["messages"][-1].content}


def get_present_illness(state: MedicalInquiryState) -> Dict[str, Any]:
    """
        获取一个现在病症的状态
    :param state:
    :return:
    """
    messages = state["messages"]
    tmp_chief_complaint = """
        你是一个专业的医疗信息提取助手。根据用户描述提取：
        - 病症开始时间 (onset_time)
        - 发作规律 (pattern)：持续/阵发/间歇性
        - 接受过的治疗或药物 (treatments_received)，列表形式
        
        提取后**必须调用 submit_illness_info 工具**，传入三个字段。调用一次后任务即完成。
        输入：
        今天早上有些咳嗽，曾经吃过一些感康
        输出：
        {
            'onset_time': '今天早上',
            'pattern': '阵发',   # 或空字符串，取决于模型判断
            'treatments_received': ['感康']
        }
    """
    agent = get_llm_agent(tmp_chief_complaint, [submit_illness_info])
    # 限制递归深度为 2（Human → AI → Tool → 结束）
    # TODO 设置递归限制
    response = agent.invoke({"messages": messages}, config={"recursion_limit": 10})

    for msg in response["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            args = msg.tool_calls[0]["args"]
            return {"present_illness": {
                "onset_time": args.get("onset_time", ""),
                "pattern": args.get("pattern", ""),
                "treatments_received": args.get("treatments_received", [])
            }}
    return {
        "present_illness": {"onset_time": "", "pattern": "", "treatments_received": []}
    }

def get_concomitant_symptoms(state: MedicalInquiryState) -> str:
    """
    获取一个并发症状
    :return:
    """
    messages = state["messages"]
    tmp_prompt = """
        根据提供的信息，提取数据中的主要病症之外的其他并发症（伴随症状），以 Python 列表形式输出，例如 ["浑身疼痛"]。如果没有其他症状，输出空列表 []。
        输入：早上发烧，浑身感觉疼痛
        输出：["浑身疼痛"]
    """
    agent = get_llm_agent(tmp_prompt, None)
    model_rsp_message = agent.invoke({"messages": messages})
    content = model_rsp_message["messages"][-1].content.strip()

    # 将模型输出解析为列表
    try:
        symptoms_list = eval(content) if content.startswith('[') else []
        if not isinstance(symptoms_list, list):
            symptoms_list = []
    except:
        symptoms_list = []

    return {"associated_symptoms": symptoms_list}

agent_builder = StateGraph(MedicalInquiryState)
agent_builder.add_node("chief_complaint", get_model_chief_complaint)
agent_builder.add_node("present_illness", get_present_illness)
agent_builder.add_node("concomitant_symptoms", get_concomitant_symptoms)

agent_builder.add_edge(START, "chief_complaint")
agent_builder.add_edge("chief_complaint", "present_illness")
agent_builder.add_edge("present_illness", "concomitant_symptoms")
agent_builder.add_edge("concomitant_symptoms", END)

agent = agent_builder.compile()

messages = [HumanMessage(content="今天早上有些咳嗽，头疼，曾经吃过一些感康")]
messages = agent.invoke({"messages": messages})
print(messages)