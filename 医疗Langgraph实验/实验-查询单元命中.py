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
    "chief_complaint": "咳嗽，头疼",
    "present_illness": {
        "onset_time": "今天早上",
        "pattern": "",
        "treatments_received": []
    },
    "associated_symptoms": [
        "咳嗽",
        "头疼"
    ],
    "symptom_absent": [],
    "existing_knowledge_supplement": [
        "{\n  \"疾病名称\": \"咳嗽\",\n  \"疾病存在的症状\": \"变应性咳嗽，慢性咳嗽，小儿剧烈咳嗽，发作性咳嗽，气道高反应性，湿疹，持续性咳嗽，情绪性哮喘，发绀，昏睡，意识丧失，面色苍白，抽搐，以头昏为主的眩晕，喘息，夜间咳嗽，咳出黄色痰液，咽痛，苔黄腻，遗尿，尿频，尿流变细或中断，尿急，尿失禁，吐白沫痰，痰湿体质，哮鸣音，流黄鼻涕，声音嘶哑，‘咳嗽水’上瘾，鼻塞，胸闷，口苦，呼吸困难，痉挛性咳嗽，咳铁锈色痰，咳痰，气喘\",\n  \"疾病引起原因\": \"呼吸道疾病\",\n  \"疾病确定概率\": \"60%\"\n}",
        "{\n  \"疾病名称\": \"慢性咳嗽\",\n  \"疾病存在的症状\": \"哮鸣音, 咽痛\",\n  \"疾病引起原因\": \"1. 鼻部疾病（如鼻炎、鼻窦炎）导致鼻后滴流刺激咳嗽感受器；2. 胃食管反流性咳嗽（胃酸反流刺激气道）\",\n  \"疾病确定概率\": \"60%\"\n}",
        "{\n  \"疾病名称\": \"痰浊头痛\",\n  \"疾病存在的症状\": \"胸闷、清晨或上午头痛、头昏\",\n  \"疾病引起原因\": \"中医病因：饮食不节、嗜酒太过或过食辛辣肥甘导致脾失健运，痰浊中阻，清阳不升，浊阴上蒙。西医病因：可能与血管性头痛、紧张性头痛、颅内疾病等有关。\",\n  \"疾病确定概率\": \"60%\"\n}"
    ]
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
