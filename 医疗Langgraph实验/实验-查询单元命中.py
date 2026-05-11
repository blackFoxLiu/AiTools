import json
from typing import List, Dict, Any, Annotated

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END, add_messages
from IPython.display import Image, display

from common_utils import read_prompt
from functools import lru_cache  # 优化：引入缓存装饰器


# Graph state
class SearchClassState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    disease_course: Dict[str, Any]
    human_body_system: Dict[str, Any]
    manifestation_characteristics: Dict[str, Any]
    symptom_nature: Dict[str, Any]
    primitive_concept_symptom: List[str]


OLLAMA_CONFIG = {
    "model": "qwen3:8b",
    "base_url": "http://127.0.0.1:11434",
    "temperature": 0.0
}

PROMPT_PATHS = {
    "primitive_concept_sys": "./prompt/prompt_primitive_concept_sys.txt",
    "primitive_concept_user": "./prompt/prompt_primitive_concept_user.txt",
    "disease_course_sys": "./prompt/prompt_tp_disease_course_sys.txt",
    "disease_course_user": "./prompt/prompt_tp_disease_course_user.txt",
    "human_body_system_sys": "./prompt/prompt_tp_human_body_system_sys.txt",
    "human_body_system_user": "./prompt/prompt_tp_human_body_system_user.txt",
    "manifestation_characteristics_sys": "./prompt/prompt_tp_manifestation_characteristics_sys.txt",
    "manifestation_characteristics_user": "./prompt/prompt_tp_manifestation_characteristics_user.txt",
    "symptom_nature_sys": "./prompt/prompt_tp_symptom_nature_sys.txt",
    "symptom_nature_user": "./prompt/prompt_tp_symptom_nature_user.txt"
}

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

def primitive_concept_node(state: SearchClassState):
    """
        病症提取——在医学信息学、术语标准化（如 SNOMED CT、ICD-11、医学本体论）标准化
    :param state:
    :return:
    """
    tmp_system_prompt = read_prompt(PROMPT_PATHS["primitive_concept_sys"])
    tmp_user_prompt = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["primitive_concept_user"])).format(
        input_knowledge=state["messages"][-1]
    )
    model_rsp_messages = get_base_chat_model().invoke([
        SystemMessage(content=tmp_system_prompt),
        HumanMessage(content=tmp_user_prompt)
    ])
    return {"primitive_concept_symptom": model_rsp_messages}

def disease_course_node(state: SearchClassState):
    """
        病症提取——按病程特点（时间轴）：判断急性、亚急性、慢性，这对病因推断很有价值。
    :param state:
    :return:
    """
    tmp_system_prompt = read_prompt(PROMPT_PATHS["disease_course_sys"])
    tmp_user_prompt = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["disease_course_user"])).format(
        input_knowledge=state["messages"][-1]
    )
    model_rsp_messages = get_base_chat_model().invoke([
        SystemMessage(content=tmp_system_prompt),
        HumanMessage(content=tmp_user_prompt)
    ])
    return {"disease_course": model_rsp_messages}

def human_body_system_node(state: SearchClassState):
    """
        病症提取——按人体系统（定位诊断）：确定症状主要来源于哪个器官/系统（如呼吸、消化、循环、神经等）。
    :param state:
    :return:
    """
    tmp_system_prompt = read_prompt(PROMPT_PATHS["human_body_system_sys"])
    tmp_user_prompt = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["human_body_system_user"])).format(
        input_knowledge=state["messages"][-1]
    )
    model_rsp_messages = get_base_chat_model().invoke([
        SystemMessage(content=tmp_system_prompt),
        HumanMessage(content=tmp_user_prompt)
    ])
    return {"human_body_system": model_rsp_messages}

def manifestation_characteristics_node(state: SearchClassState):
    """
        病症提取——按症状性质（定性诊断）：区分是炎症、反射、结构性、功能性还是心理性等。
    :param state:
    :return:
    """
    tmp_system_prompt = read_prompt(PROMPT_PATHS["manifestation_characteristics_sys"])
    tmp_user_prompt = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["manifestation_characteristics_user"])).format(
        input_knowledge=state["messages"][-1]
    )
    model_rsp_messages = get_base_chat_model().invoke([
        SystemMessage(content=tmp_system_prompt),
        HumanMessage(content=tmp_user_prompt)
    ])
    return {"manifestation_characteristics": model_rsp_messages}

def symptom_nature_node(state: SearchClassState):
    """
        病症提取——按表现特征（现象学）：如干/湿、持续性/间歇性、伴随体征等。
    :param state:
    :return:
    """
    tmp_system_prompt = read_prompt(PROMPT_PATHS["symptom_nature_sys"])
    tmp_user_prompt = PromptTemplate.from_template(read_prompt(PROMPT_PATHS["symptom_nature_user"])).format(
        input_knowledge=state["messages"][-1]
    )
    model_rsp_messages = get_base_chat_model().invoke([
        SystemMessage(content=tmp_system_prompt),
        HumanMessage(content=tmp_user_prompt)
    ])
    return {"symptom_nature": model_rsp_messages}


# Build workflow
workflow = StateGraph(SearchClassState)

# 创建节点
workflow.add_node("primitive_concept_node", primitive_concept_node)
workflow.add_node("disease_course_node", disease_course_node)
workflow.add_node("human_body_system_node", human_body_system_node)
workflow.add_node("manifestation_characteristics_node", manifestation_characteristics_node)
workflow.add_node("symptom_nature_node", symptom_nature_node)

# 创建边
workflow.add_edge(START, "primitive_concept_node")
workflow.add_edge("primitive_concept_node", "disease_course_node")
workflow.add_edge("disease_course_node", "human_body_system_node")
workflow.add_edge("human_body_system_node", "manifestation_characteristics_node")
workflow.add_edge("manifestation_characteristics_node", "symptom_nature_node")
workflow.add_edge("symptom_nature_node", END)

# Compile
chain = workflow.compile()

user_input = {
  "chief_complaint": "鼻塞，打喷嚏，流清涕",
  "present_illness": {
    "onset_time": "昨日傍晚",
    "pattern": "间歇性",
    "treatments_received": []
  },
  "associated_symptoms": [
    "鼻塞",
    "清水样鼻涕",
    "打喷嚏",
    "声音嘶哑",
    "头部昏沉",
    "轻度乏力",
    "睡眠质量差",
    "被迫张口呼吸",
    "精神一般",
    "食欲稍减"
  ],
  "symptom_absent": [
    "发热",
    "肌肉酸痛",
    "畏寒",
    "咳嗽咳痰"
  ],
  "missing_info": [
    "是否有明确的过敏原接触史（如花粉、尘螨、宠物皮屑等）",
    "是否伴有鼻痒（已提及但需确认是否为持续性）",
    "是否使用过抗组胺药或其他药物（现病史中提到'已接受治疗/药物'但未明确药物种类）"
  ],
  "solution": {
    "possible_diagnoses": [
      "急性上呼吸道感染（病毒性）",
      "过敏性鼻炎",
      "急性鼻咽炎"
    ],
    "suggested_actions": {
      "self_monitoring": "观察症状变化，记录鼻塞、打喷嚏频率及睡眠质量改善情况。若3天内症状无缓解或出现发热、咳嗽等新症状，需及时就医。",
      "symptom_management": "可使用生理盐水鼻腔冲洗保持鼻腔湿润，配合使用抗组胺药物（如氯雷他定）缓解鼻塞和流涕，声音嘶哑可含服润喉糖或使用溶菌酶含片。",
      "environment_adjustment": "保持室内空气湿润（40-60%湿度），避免接触烟酒及刺激性气体，睡前抬高枕头减轻被迫张口呼吸。"
    },
    "disclaimer": "以上信息仅供参考，不能替代专业医疗建议。具体诊断和治疗方案需由执业医师根据面诊情况制定。若症状持续加重或出现新症状，请立即就医。"
  }
}

user_input_str = json.dumps(user_input, ensure_ascii=False, indent=2)
messages = [HumanMessage(content=user_input_str)]
# 调用工作流
answer = chain.invoke({"messages": messages})

print(answer.get("disease_course", ""))
print(answer.get("human_body_system", ""))
print(answer.get("manifestation_characteristics", ""))
print(answer.get("symptom_nature", ""))
print(answer.get("primitive_concept_symptom", ""))

# Invoke
print("Initial joke:")