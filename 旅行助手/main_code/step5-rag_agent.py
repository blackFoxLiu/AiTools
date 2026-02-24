#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅行助手RAG响应
"""

import logging
import os
import sys

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma

# 设置日志
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

import configparser

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

# 文件信息配置
model_name = config['model_config']['model_name']
model_url = config['model_config']['model_url']
rag_file_path = config['file_path']['rag_file_path']
chroma_db_path = config['file_path']['chroma_db_path']

# 编码模型
embedding_model_nme = config['embedding_model']['model_name']


# 初始化 Ollama LLM
llm = Ollama(
    model="model_name",
    base_url="model_url",
    temperature=0.0,
)

# 初始化嵌入模型
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_nme,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 自定义提示模板
prompt_template = """[INST]<<SYS>>
You are a helpful AI assistant.
<</SYS>>

请根据以下上下文信息回答问题。如果上下文信息不足以回答问题，请说明。

上下文:
{context}

问题: {question}
[/INST]"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# 添加问题分析函数
def analyze_question_intent(question, retrieved_docs, llm):
    """分析大模型对问题的理解"""
    analysis_prompt = f"""
    请分析以下用户问题，并回答：
    1. 用户可能在询问什么？
    2. 问题的核心意图是什么？
    3. 基于检索到的文档，这个问题是否容易回答？

    用户问题: "{question}"

    检索到的相关文档片段数量: {len(retrieved_docs)}

    请用简洁的语言回答。
    """

    try:
        analysis_response = llm.invoke(analysis_prompt)
        return analysis_response
    except Exception as e:
        return f"问题分析失败: {str(e)}"


def load_and_process_documents(directory_path):
    """加载和处理文档"""
    # 使用 DirectoryLoader 加载文本文件
    loader = DirectoryLoader(
        directory_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={'encoding': 'utf-8'}
    )
    documents = loader.load()

    # 文本分割
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],  # 明确指定只按换行符切分
        chunk_size=100,  # 设置一个较大的值，确保不会因为长度限制而进一步切分行内内容
        chunk_overlap=20,
        length_function=len,
    )

    splits = text_splitter.split_documents(documents)
    print(f"文档分割为 {len(splits)} 个片段")
    return splits


def setup_vector_store(documents, persist_directory=chroma_db_path):
    """设置向量数据库"""
    # 如果已存在持久化数据，则加载
    if os.path.exists(persist_directory):
        print("加载已存在的向量数据库...")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding_model
        )
    else:
        print("创建新的向量数据库...")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        vector_store.persist()

    return vector_store


def main():
    # 1. 加载和处理文档
    documents = load_and_process_documents(rag_file_path)

    # 2. 设置向量数据库
    vector_store = setup_vector_store(documents)

    # 3. 创建检索器
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )

    # 4. 创建 RetrievalQA 链
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    while True:
        # 5. 查询示例 - 修改为实际的问题
        query = input()

        # 记录用户原始问题
        print('\n' + '=' * 60)
        print(f"🎯 用户原始问题: \"{query}\"")
        print('=' * 60)

        # 方式1: 仅检索相似文档（对应原代码的 retrieve）
        print('\n' + '-' * 10 + '检索结果' + '-' * 10)
        similar_docs = retriever.get_relevant_documents(query)

        # 记录检索到的文档信息
        print(f"📚 检索到 {len(similar_docs)} 个相关文档片段")

        # 添加问题意图分析
        print('\n' + '🔍 问题意图分析:')
        intent_analysis = analyze_question_intent(query, similar_docs, llm)
        print(intent_analysis)

        for i, doc in enumerate(similar_docs):
            print('\n' + '*' * 10 + f'片段 {i} 开始' + '*' * 10)
            print(doc.page_content)
            print('*' * 10 + f'片段 {i} 结束' + '*' * 10)
        print('-' * 10 + '检索结果' + '-' * 10)

        # 方式2: 完整问答（对应原代码的 query）
        print("\n" + "=" * 50)
        print("完整RAG回答:")
        print("=" * 50)

        # 记录大模型实际处理的问题
        print(f"\n🤖 大模型实际处理的问题: \"{query}\"")

        result = qa_chain({"query": query})

        # 显示使用的源文档
        print("\n📋 使用的源文档:")
        for i, source_doc in enumerate(result["source_documents"]):
            print(f"文档 {i + 1}: {source_doc.page_content[:200]}...")

        # 添加回答质量评估
        print('\n' + '📊 回答质量评估:')
        evaluation_prompt = f"""
        请评估以下问答对的质量：
    
        问题: "{query}"
        回答: "{result['result']}"
    
        请从以下维度评估：
        1. 回答的相关性
        2. 回答的完整性
        3. 是否基于提供的上下文
    
        用简洁的语言给出评估。
        """

        try:
            evaluation_response = llm.invoke(evaluation_prompt)
            print(evaluation_response)
        except Exception as e:
            print(f"评估失败: {str(e)}")

        print(f"\n💬 最终回答: {result['result']}")


if __name__ == "__main__":
    main()