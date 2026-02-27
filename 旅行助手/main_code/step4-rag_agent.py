#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅行助手RAG响应系统
"""

import configparser
import logging
import os
import sys
from typing import List

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores import Chroma

# 尝试导入自定义工具函数，若失败则使用占位（实际需保证该函数存在）
try:
    from utils.common_tools import get_prompt_str
except ImportError:
    logger.debug("警告：导入 get_prompt_str 失败，请确保 utils.common_tools 可用")
    # 提供一个简单的占位实现，避免运行时崩溃

    def get_prompt_str(file_path: str) -> str:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise RuntimeError(f"无法读取提示文件 {file_path}: {e}")

# 日志配置（只添加一次StreamHandler）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(logging.StreamHandler(sys.stdout))

# 配置加载
config = configparser.ConfigParser()
CONFIG_PATH = 'config.ini'
if not os.path.exists(CONFIG_PATH):
    raise FileNotFoundError(f"配置文件 {CONFIG_PATH} 不存在")

config.read(CONFIG_PATH, encoding='utf-8')

# 配置项读取（带默认值防止缺失）
DEFAULT_CONFIG = {
    'openai_model_config.model_name': 'llama2',
    'openai_model_config.base_url': 'http://localhost:11434',
    'embedding_model.model_name': 'sentence-transformers/all-MiniLM-L6-v2',
    'file_path.chroma_db_path': './chroma_db',
    'file_path.rag_file_path': './data',
    'file_path.rag_eva_resp_prompt': './prompts/evaluation.txt',
    'file_path.rag_analysis_prompt': './prompts/analysis.txt',
    'file_path.rag_chat_prompt': './prompts/chat.txt',
    'text_splitter.chunk_size': '400',
    'text_splitter.chunk_overlap': '20',
    'text_splitter.separators': '\n',  # 多个分隔符用逗号分隔
    'retriever.k': '16',
}


def get_config(section_key: str, default: str = None) -> str:
    """安全获取配置值，若缺失则返回默认值或抛出异常"""
    try:
        section, key = section_key.split('.', 1)
        return config.get(section, key)
    except (configparser.NoSectionError, configparser.NoOptionError):
        if default is not None:
            return default
        else:
            raise ValueError(f"配置项 {section_key} 缺失且无默认值")


# 加载配置
model_name = get_config('openai_model_config.model_name', DEFAULT_CONFIG['openai_model_config.model_name'])
base_url = get_config('openai_model_config.base_url', DEFAULT_CONFIG['openai_model_config.base_url'])
embedding_model_name = get_config('embedding_model.model_name', DEFAULT_CONFIG['embedding_model.model_name'])
chroma_db_path = get_config('file_path.chroma_db_path', DEFAULT_CONFIG['file_path.chroma_db_path'])
rag_file_path = get_config('file_path.rag_file_path', DEFAULT_CONFIG['file_path.rag_file_path'])
rag_eva_resp_prompt = get_config('file_path.rag_eva_resp_prompt', DEFAULT_CONFIG['file_path.rag_eva_resp_prompt'])
rag_analysis_prompt = get_config('file_path.rag_analysis_prompt', DEFAULT_CONFIG['file_path.rag_analysis_prompt'])
rag_chat_prompt = get_config('file_path.rag_chat_prompt', DEFAULT_CONFIG['file_path.rag_chat_prompt'])

# 文本分割参数
chunk_size = int(get_config('text_splitter.chunk_size', DEFAULT_CONFIG['text_splitter.chunk_size']))
chunk_overlap = int(get_config('text_splitter.chunk_overlap', DEFAULT_CONFIG['text_splitter.chunk_overlap']))
separators = get_config('text_splitter.separators', DEFAULT_CONFIG['text_splitter.separators']).split(',')

# 检索参数
retriever_k = int(get_config('retriever.k', DEFAULT_CONFIG['retriever.k']))

# 初始化LLM
llm = Ollama(
    model=model_name,
    base_url=base_url,
    temperature=0.0,
)

# 初始化嵌入模型（设备可配置）
embedding_model = HuggingFaceEmbeddings(
    model_name=embedding_model_name,
    model_kwargs={'device': 'cpu'},  # 若需GPU可改为'cuda'
    encode_kwargs={'normalize_embeddings': True}
)


class RAGApplication:
    """RAG应用封装"""

    def __init__(self, llm, embedding_model, vector_store_path: str, data_path: str):
        self.llm = llm
        self.embedding_model = embedding_model
        self.vector_store_path = vector_store_path
        self.data_path = data_path
        self.vector_store = None
        self.retriever = None
        self.qa_chain = None
        self._initialize()

    def _initialize(self):
        """初始化向量库和检索链"""
        # 尝试加载已存在的向量库
        if os.path.exists(self.vector_store_path) and os.listdir(self.vector_store_path):
            logger.info("加载已存在的向量数据库...")
            self.vector_store = Chroma(
                persist_directory=self.vector_store_path,
                embedding_function=self.embedding_model
            )
        else:
            logger.info("创建新的向量数据库...")
            documents = self._load_documents()
            splits = self._split_documents(documents)
            self.vector_store = Chroma.from_documents(
                documents=splits,
                embedding=self.embedding_model,
                persist_directory=self.vector_store_path
            )
            self.vector_store.persist()
            logger.info(f"向量数据库已创建并持久化至 {self.vector_store_path}")

        # 创建检索器
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": retriever_k}
        )

        # 加载对话提示模板
        chat_prompt_template = get_prompt_str(rag_chat_prompt)
        chat_prompt = PromptTemplate(
            template=chat_prompt_template,
            input_variables=["context", "question"]
        )

        # 创建QA链
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            chain_type_kwargs={"prompt": chat_prompt},
            return_source_documents=True
        )

    def _load_documents(self) -> List[Document]:
        """加载原始文档"""
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"文档路径不存在: {self.data_path}")

        loader = DirectoryLoader(
            self.data_path,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'}
        )
        documents = loader.load()
        logger.info(f"加载了 {len(documents)} 个文档")
        return documents

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """将文档分割为块"""
        text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        splits = text_splitter.split_documents(documents)
        logger.info(f"文档分割为 {len(splits)} 个片段")
        return splits

    def analyze_question_intent(self, question: str, retrieved_docs: List[Document]) -> str:
        """分析问题意图"""
        # 将检索到的文档内容合并为字符串
        docs_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
        try:
            analysis_template = get_prompt_str(rag_analysis_prompt)
            analysis_prompt = PromptTemplate(
                template=analysis_template,
                input_variables=["question", "retrieved_docs"]
            )
            formatted_prompt = analysis_prompt.format(question=question, retrieved_docs=len(docs_text))
            response = self.llm.invoke(formatted_prompt)
            return response
        except Exception as e:
            logger.error(f"问题意图分析失败: {e}")
            return f"分析失败: {str(e)}"

    def evaluate_response(self, question: str, answer: str) -> str:
        """评估回答质量"""
        try:
            evaluation_template = get_prompt_str(rag_eva_resp_prompt)
            # 注意：假设模板中使用了 {query} 和 {result}
            evaluation_prompt = evaluation_template.format(query=question, result=answer)
            response = self.llm.invoke(evaluation_prompt)
            return response
        except Exception as e:
            logger.error(f"回答质量评估失败: {e}")
            return f"评估失败: {str(e)}"

    def retrieve_similar_docs(self, question: str) -> List[Document]:
        """检索相似文档"""
        return self.retriever.get_relevant_documents(question)

    def answer_question(self, question: str) -> dict:
        """生成完整问答结果"""
        return self.qa_chain({"query": question})


def main():
    """主交互循环"""
    # 创建RAG应用实例
    app = RAGApplication(llm, embedding_model, chroma_db_path, rag_file_path)

    logger.debug("\n=== 旅行助手RAG系统已启动（输入 'exit' 或 'quit' 退出）===\n")
    while True:
        try:
            query = input("请输入您的问题: ").strip()
            if query.lower() in ('exit', 'quit'):
                logger.debug("感谢使用，再见！")
                break
            if not query:
                continue

            logger.debug('\n' + '=' * 60)
            logger.debug(f"🎯 用户原始问题: \"{query}\"")
            logger.debug('=' * 60)

            # 检索相似文档
            logger.debug('\n' + '-' * 10 + '检索结果' + '-' * 10)
            similar_docs = app.retrieve_similar_docs(query)
            logger.debug(f"📚 检索到 {len(similar_docs)} 个相关文档片段")

            # 意图分析
            logger.debug('\n' + '🔍 问题意图分析:')
            intent = app.analyze_question_intent(query, similar_docs)
            logger.debug(intent)

            # 显示检索到的文档片段
            for i, doc in enumerate(similar_docs):
                logger.debug('\n' + '*' * 10 + f'片段 {i} 开始' + '*' * 10)
                logger.debug(doc.page_content)
                logger.debug('*' * 10 + f'片段 {i} 结束' + '*' * 10)
            logger.debug('-' * 10 + '检索结果' + '-' * 10)

            result = app.answer_question(query)

            # 显示源文档
            logger.debug("\n📋 使用的源文档:")
            for i, src_doc in enumerate(result["source_documents"]):
                preview = src_doc.page_content[:200] + ('...' if len(src_doc.page_content) > 200 else '')
                logger.debug(f"文档 {i + 1}: {preview}")

            # 回答质量评估
            logger.debug('\n' + '📊 回答质量评估:')
            eval_result = app.evaluate_response(query, result['result'])
            logger.debug(eval_result)

            # 完整RAG回答
            logger.debug("\n" + "=" * 50)
            logger.debug("完整RAG回答:")
            logger.debug("=" * 50)

            print(f"\n💬 最终回答: {result['result']}")

        except KeyboardInterrupt:
            logger.debug("\n\n程序被用户中断，退出。")
            break
        except Exception as e:
            logger.error(f"处理问题时发生错误: {e}", exc_info=True)
            logger.debug(f"处理失败: {e}")


if __name__ == "__main__":
    main()