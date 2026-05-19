"""
单例模式管理器 - 统一管理所有全局唯一实例
路径: src/container/container.py
"""

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from FlagEmbedding import BGEM3FlagModel

from pathlib import Path

# ---------- 项目根目录定位（用于资源/数据路径）----------
_PROJECT_ROOT = Path(__file__).parent.parent.parent   # 即项目根目录
_DATA_DIR = _PROJECT_ROOT / "data"
_DATABASES_DIR = _DATA_DIR / "databases"


import os
import sys
# 导入独立子图构建函数
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 将项目根目录添加到 Python 的模块搜索路径中
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    from src.tools.knowledge_graph_tools import Neo4jQueryTools
    from src.tools.rag_module import KnowledgeBaseService
    from config.default_config import config
    from src.hit_module.hit_generate.knowledge_hit_generate_graph import DiseaseDataHitGenService
    from src.hit_module.hit_search.knowledge_hit_search_graph import DiseaseDataSearchService
except ImportError:
    raise RuntimeError(f"导入模块失败")


# ========== ChatOllama 单例 ==========
_ollama_instance = None
def get_base_chat_model():
    global _ollama_instance
    if _ollama_instance is None:
        _ollama_instance = ChatOllama(
            model=config.OLLAMA_CONFIG["model"],
            base_url=config.OLLAMA_CONFIG["base_url"],
            temperature=config.OLLAMA_CONFIG["temperature"]
        )
    return _ollama_instance


# ========== HuggingFaceEmbeddings 单例 ==========
_hf_embedding_instance = None
def get_hf_embedding():
    global _hf_embedding_instance
    if _hf_embedding_instance is None:
        _hf_embedding_instance = HuggingFaceEmbeddings(
            model_name=config.coarse_model_path,
            model_kwargs={'device': config.device},
            encode_kwargs={'normalize_embeddings': True}
        )
    return _hf_embedding_instance


# ========== BGEM3FlagModel 单例 ==========
_bge_m3_instance = None
def get_bge_m3_model():
    global _bge_m3_instance
    if _bge_m3_instance is None:
        _bge_m3_instance = BGEM3FlagModel(
            config.fine_model_path,
            use_fp16=config.use_gpu
        )
    return _bge_m3_instance


# ========== Neo4jQueryTools 单例 ==========
_neo4j_tools = None
def get_neo4j_tools():
    global _neo4j_tools
    if _neo4j_tools is None:
        _neo4j_tools = Neo4jQueryTools()
    return _neo4j_tools


# ========== KnowledgeBaseService 单例（RAG向量库） ==========
_kb_service = None
def get_knowledge_base_service():
    global _kb_service
    if _kb_service is None:
        # 数据库统一放在 data/databases/search_hit/ 下
        persist_path = _DATABASES_DIR / "search_hit"
        # 确保目录存在
        persist_path.mkdir(parents=True, exist_ok=True)
        _kb_service = KnowledgeBaseService(
            collection_name="search_hit",
            persist_directory=str(persist_path)
        )
    return _kb_service


# ========== DiseaseDataHitGenService 单例 ==========
_disease_hit_gen = None
def get_disease_hit_gen_service():
    global _disease_hit_gen
    if _disease_hit_gen is None:
        _disease_hit_gen = DiseaseDataHitGenService()
    return _disease_hit_gen


# ========== 8. DiseaseDataSearchService 单例 ==========
_disease_search = None
def get_disease_search_service():
    global _disease_search
    if _disease_search is None:
        _disease_search = DiseaseDataSearchService()
    return _disease_search


# ========== 服务容器（统一访问入口） ==========
class ServiceContainer:
    """服务容器，以属性方式统一访问所有单例"""

    @property
    def llm(self):
        return get_base_chat_model()

    @property
    def hf_embedding(self):
        return get_hf_embedding()

    @property
    def bge_m3(self):
        return get_bge_m3_model()

    @property
    def neo4j(self):
        return get_neo4j_tools()

    @property
    def rag(self):
        return get_knowledge_base_service()

    @property
    def hit_gen(self):
        return get_disease_hit_gen_service()

    @property
    def hit_search(self):
        return get_disease_search_service()


# 全局容器实例（可直接导入使用）
service_container = ServiceContainer()