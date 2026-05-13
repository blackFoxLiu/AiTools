#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
多子 Agent 示例：使用 LangGraph 的 create_agent 实现工具调用
使用本地 Ollama (qwen3:8b) 和 Neo4j 查询工具
兼容 langchain 1.2.17 + langgraph
"""

from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

from knowledge_graph_tools import Neo4jQueryTools

# ==================== 1. 配置 LLM ====================
DEFAULT_OLLAMA_CONFIG = {
    "model": "qwen3:8b",
    "base_url": "http://127.0.0.1:11434",
    "temperature": 0.0,
}

try:
    llm = ChatOllama(**DEFAULT_OLLAMA_CONFIG)
    # 正确的连通性测试（必须传入消息列表）
    llm.invoke([HumanMessage(content="ping")])
    print("✅ LLM 连接成功")
except Exception as e:
    print(f"❌ Ollama 连接失败，请确保服务已启动且模型已拉取。\n错误: {e}")
    exit(1)


@tool
def query_disease_info(disease_name: str) -> str:
    """
    根据疾病名称，从 Neo4j 知识图谱中查询该疾病的详细信息。
    参数:
        disease_name: 疾病的中文名称，例如 "二硫化碳中毒"
    """
    print(f"🔍 查询疾病: {disease_name}")

    # 1. 构建生成 Cypher 的提示词
    cypher_generation_prompt = f"""
你是一个 Neo4j Cypher 专家。根据用户给出的疾病名称，生成一条查询语句。
数据库结构：
- 疾病节点标签为 `Disease`，该标签唯一。
- `Disease` 节点包含属性 `name`（疾病名称）。

要求：
- 只输出 Cypher 语句，不要包含任何解释、注释、反引号或其他文字。
- 查询应返回该疾病节点的所有属性（或足够的信息）。
- 示例：用户输入“头风”，输出：MATCH (n:Disease) WHERE n.name = '头风' RETURN n LIMIT 25;

用户输入的疾病名称：{disease_name}
输出 Cypher（仅语句）：
"""

    # 2. 调用 LLM 生成 Cypher（增加清理逻辑）
    try:
        response = llm.invoke([
            SystemMessage(content="你是一个专业的 Neo4j 查询生成器。严格按要求输出纯 Cypher 语句，不要包含 Markdown 代码块标记。"),
            HumanMessage(content=cypher_generation_prompt)
        ])
        cypher_query = response.content.strip()
        # 移除可能出现的 markdown 代码块标记
        if cypher_query.startswith("```"):
            cypher_query = cypher_query.strip("`").strip()
            if cypher_query.lower().startswith("cypher"):
                cypher_query = cypher_query[6:].strip()
        print(f"📝 生成的 Cypher: {cypher_query}")
    except Exception as e:
        return f"生成 Cypher 语句时出错: {e}"

    # 3. 执行 Cypher 查询
    try:
        neo4j_client = Neo4jQueryTools()
        result = neo4j_client.use_cypher(cypher_query)
        if not result or len(result.strip()) == 0:
            return f"未找到关于“{disease_name}”的信息。"
        print(f"✅ 查询完成，返回长度: {len(result)} 字符")
        return result
    except Exception as e:
        return f"执行 Neo4j 查询时出错: {e}\n生成的 Cypher 语句: {cypher_query}"



@tool
def insert_ss_node(name: str) -> str:
    """
    向 Neo4j 知识图谱中插入一个标签为 SS 的节点，并设置其 name 属性。
    参数:
        name: 节点的名称属性值。
    """
    print(f"🔍 插入节点: SS, name={name}")

    # 直接构建 Cypher 插入语句
    cypher_query = f"CREATE (n:SS {{name: '{name}'}}) RETURN n"
    print(f"📝 生成的 Cypher: {cypher_query}")

    try:
        neo4j_client = Neo4jQueryTools()
        result = neo4j_client.use_cypher(cypher_query)
        if not result or len(result.strip()) == 0:
            return f"插入节点“{name}”失败，未返回结果。"
        print(f"✅ 插入完成，返回长度: {len(result)} 字符")
        return f"成功插入节点: SS {{name: '{name}'}}，结果: {result}"
    except Exception as e:
        return f"执行 Neo4j 插入时出错: {e}\n生成的 Cypher 语句: {cypher_query}"

# ==================== 2. 创建主 Agent（使用 LangGraph） ====================
# 定义系统提示（create_agent 使用 system_message 参数）
system_prompt = (
    "你是一个智能助理，可以调用工具查询疾病信息。\n"
    "当用户询问某种疾病时，请使用 query_disease_info 工具来获取数据，工具参数 disease_name 就是疾病的中文名称。\n"
    "当用户要求进行插入数据时，请使用 insert_ss_node 工具来插入一个节点，工具参数 name 就是的中文名称。\n"
    "例如：用户问“查询二硫化碳中毒”，你应当调用 query_disease_info(disease_name='二硫化碳中毒')。"
    "例如：用户问“查询二硫化碳中毒”，你应当调用 insert_ss_node(name='巴拉巴拉')。"
)

# 创建 ReAct Agent
supervisor_agent = create_agent(
    model=llm,
    tools=[query_disease_info, insert_ss_node],
    debug = False,
    system_prompt=system_prompt   # 注意：参数名可能是 state_modifier 或 messages_modifier，根据版本
)


# ==================== 3. 测试运行 ====================
if __name__ == "__main__":
    test_questions = [
        "请帮我查询二硫化碳中毒的相关信息",
        "帮我插入一个结点名称叫胖胖龙"
    ]

    for q in test_questions:
        print("\n" + "=" * 60)
        print(f"👤 用户: {q}")
        try:
            # LangGraph agent 的 invoke 输入格式
            response = supervisor_agent.invoke(
                {"messages": [("human", q)]}
            )
            # 提取最后一条消息的回答内容
            answer = response["messages"][-1].content
            print(f"🤖 助手: {answer}")
        except Exception as e:
            print(f"⚠️ 调用出错: {e}")