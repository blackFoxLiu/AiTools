"""
知识库服务：向量上传、删除、查询（粗排+精排）
依赖：pip install langchain-chroma langchain-community dashscope FlagEmbedding
"""

import hashlib
import os
import warnings
from datetime import datetime
from typing import List, Dict, Any, Optional

import logger
import numpy as np
import torch
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from FlagEmbedding import BGEM3FlagModel


# ==================== 配置 ====================
class Config:
    persist_directory = "./chroma_db"
    collection_name = "knowledge_base"
    md5_path = "./output/uploaded_md5.txt"
    chunk_size = 800
    chunk_overlap = 50
    separators = ["\n\n"] # , "。", "！", "？", "；", " ", "", "\n"
    max_split_char_number = 800

    # 嵌入模型配置
    coarse_model_path = "D:/endedingModel/bge-base-zh-v1___5"   # 粗排模型
    fine_model_path = "D:/endedingModel/bge-m3"  # 精排模型（BGE-M3）
    use_gpu = True                     # 是否使用GPU（BGE-M3建议开启）
    rerank_ratio = 2                   # 精排时粗排召回的倍数：最终返回 top_k，粗排召回 top_k * rerank_ratio
    fine_weight = 1.0                  # 精排分数权重（可保留为1）
    coarse_weight = 0.0                # 粗排分数权重（设为0则完全使用精排分数）


config = Config()

# ==================== MD5 辅助函数（保持不变） ====================
def get_string_md5(input_str: str, encoding='utf-8') -> str:
    str_bytes = input_str.encode(encoding=encoding)
    md5_obj = hashlib.md5()
    md5_obj.update(str_bytes)
    return md5_obj.hexdigest()

def check_md5(md5_str: str) -> bool:
    if not os.path.exists(config.md5_path):
        open(config.md5_path, 'w', encoding='utf-8').close()
        return False
    with open(config.md5_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip() == md5_str:
                return True
    return False

def save_md5(md5_str: str):
    with open(config.md5_path, 'a', encoding='utf-8') as f:
        f.write(md5_str + '\n')

def remove_md5(md5_str: str):
    if not os.path.exists(config.md5_path):
        return
    lines = open(config.md5_path, 'r', encoding='utf-8').readlines()
    with open(config.md5_path, 'w', encoding='utf-8') as f:
        for line in lines:
            if line.strip() != md5_str:
                f.write(line)


# ==================== 知识库服务类（优化版：粗排+精排） ====================
class KnowledgeBaseService:
    def __init__(self, collection_name: str = None, persist_directory: str = None):
        # 使用传入的参数，若未传则使用 config 中的默认值
        self.collection_name = collection_name or config.collection_name
        self.persist_directory = persist_directory or config.persist_directory

        os.makedirs(self.persist_directory, exist_ok=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # 1. 粗排模型（保持不变）
        self.coarse_embedding = HuggingFaceEmbeddings(
            model_name=config.coarse_model_path,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )

        # 2. 精排模型（保持不变）
        self.fine_model = None
        try:
            self.fine_model = BGEM3FlagModel(
                config.fine_model_path,
                use_fp16=config.use_gpu
            )
            print("[信息] 精排模型 BGE-M3 加载成功")
        except Exception as e:
            logger.error(f"[警告] 精排模型加载失败，将回退到仅粗排模式。错误：{e}")

        # 3. 向量数据库（使用实例变量）
        self.chroma = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.coarse_embedding,
            persist_directory=self.persist_directory,
        )

        # 4. 文本分割器（保持不变）
        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=config.separators,
            length_function=len,
        )

    # ------------------- 上传（与原始逻辑相同，使用粗排模型） -------------------
    def upload_by_str(self, data: str, filename: str) -> str:
        content_md5 = get_string_md5(data)
        if check_md5(content_md5):
            return "[跳过] 内容已经存在知识库中"

        # 文本切分
        if len(data) > config.max_split_char_number:
            chunks = self.spliter.split_text(data)
        else:
            chunks = [data]

        file_md5 = get_string_md5(filename)
        chunk_ids = [f"{file_md5}_{i}" for i in range(len(chunks))]

        base_metadata = {
            "source": filename,
            "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "operator": "FOX",
            "content_md5": content_md5,
        }
        metadatas = [base_metadata.copy() for _ in chunks]

        # 分批添加，避免超过 Chroma 的最大批次限制（5461）
        batch_size = 5000  # 安全阈值
        total = len(chunks)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            self.chroma.add_texts(
                texts=chunks[start:end],
                metadatas=metadatas[start:end],
                ids=chunk_ids[start:end]
            )

        save_md5(content_md5)
        return f"[成功] 已将 {len(chunks)} 个片段载入向量库，ID前缀: {file_md5}"

    def upload_by_directory(self, dir_path: str, extensions: List[str] = None) -> str:
        if not os.path.isdir(dir_path):
            return f"[错误] 目录不存在: {dir_path}"
        if extensions is None:
            extensions = ['.txt']

        results = []
        total_files = 0
        success_files = 0
        skipped_files = 0
        error_files = 0

        for root, dirs, files in os.walk(dir_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    total_files += 1
                    file_path = os.path.join(root, file)
                    try:
                        data = None
                        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
                            try:
                                with open(file_path, 'r', encoding=encoding) as f:
                                    data = f.read()
                                break
                            except UnicodeDecodeError:
                                continue
                        if data is None:
                            raise ValueError("无法以常见编码读取文件")
                    except Exception as e:
                        error_files += 1
                        results.append(f"[错误] 读取文件失败 {file_path}: {e}")
                        continue

                    res = self.upload_by_str(data, file)
                    if "成功" in res:
                        success_files += 1
                    elif "跳过" in res:
                        skipped_files += 1
                    results.append(f"{file}: {res}")

        summary = (f"目录处理完成：共发现 {total_files} 个文件，"
                   f"新增 {success_files} 个，跳过 {skipped_files} 个，错误 {error_files} 个")
        results.insert(0, summary)
        return "\n".join(results)

    # ------------------- 查询（粗排+精排优化） -------------------
    def query(self, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        两阶段检索：
        1. 粗排：用原模型召回 top_k * rerank_ratio 个候选
        2. 精排：用 BGE-M3 对候选重新计算相似度并排序，返回最终 top_k
        """
        if top_k <= 0:
            return []

        # 阶段1：粗排召回
        candidate_k = top_k * config.rerank_ratio
        # 获取查询向量（粗排）
        query_vector = self.coarse_embedding.embed_query(query_text)
        native_collection = self.chroma._collection
        coarse_results = native_collection.query(
            query_embeddings=[query_vector],
            n_results=candidate_k,
            include=["documents", "metadatas", "distances"]
        )

        ids = coarse_results['ids'][0]
        docs = coarse_results['documents'][0]
        metas = coarse_results['metadatas'][0]
        coarse_scores = coarse_results['distances'][0]   # L2距离，越小越相似

        if not ids:
            return []

        # 如果没有精排模型，直接返回粗排结果
        if self.fine_model is None:
            output = []
            for i, doc_id in enumerate(ids):
                output.append({
                    "id": doc_id,
                    "text": docs[i],
                    "metadata": metas[i] if metas[i] else {},
                    "score": coarse_scores[i],
                })
            return output[:top_k]

        # 阶段2：精排重排序
        # 计算查询的精细向量（BGE-M3 支持 dense 向量）
        # BGE-M3 的 encode 返回对象，包含 dense_vecs
        query_fine_vec = self.fine_model.encode(
            [query_text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs'][0]  # shape: (embed_dim,)

        # 计算所有候选文档的精细向量
        doc_fine_vecs = self.fine_model.encode(
            docs,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )['dense_vecs']  # shape: (num_candidates, embed_dim)

        # 计算余弦相似度（BGE-M3 输出的向量已归一化，可直接点积）
        similarities = np.dot(doc_fine_vecs, query_fine_vec)  # 点积即余弦相似度
        # 将距离转换为相似度（分数越高越相似），用于排序
        # 粗排分数是 L2 距离，越小越好，我们保留但不参与最终排序（若需要混合可配置）
        combined_scores = similarities  # 直接使用精排相似度

        # 按精排相似度降序排序
        sorted_indices = np.argsort(combined_scores)[::-1]  # 从大到小

        # 组装最终结果
        final_results = []
        for idx in sorted_indices[:top_k]:
            original_pos = idx
            final_results.append({
                "id": ids[original_pos],
                "text": docs[original_pos],
                "metadata": metas[original_pos] if metas[original_pos] else {},
                "score": float(combined_scores[original_pos]),  # 精排相似度，范围[-1,1]
                "coarse_score": coarse_scores[original_pos],    # 保留粗排分数供分析
            })
        return final_results

    # ------------------- 删除（保持不变） -------------------
    def delete_by_ids(self, ids: List[str]) -> str:
        if not ids:
            return "[跳过] 未提供任何ID"
        try:
            self.chroma.delete(ids=ids)
            return f"[成功] 已删除 {len(ids)} 条向量记录"
        except Exception as e:
            return f"[失败] {e}"

    def delete_by_metadata(self, where: Dict[str, Any]) -> str:
        if not where:
            return "[跳过] 未提供任何过滤条件"
        try:
            results = self.chroma.get(where=where)
            ids_to_delete = results['ids']
            if not ids_to_delete:
                return "[跳过] 未找到符合条件的记录"
            self.chroma.delete(ids=ids_to_delete)
            content_md5s = set()
            for meta in results['metadatas']:
                if meta and 'content_md5' in meta:
                    content_md5s.add(meta['content_md5'])
            for md5 in content_md5s:
                remain = self.chroma.get(where={"content_md5": md5})
                if not remain['ids']:
                    remove_md5(md5)
            return f"[成功] 删除 {len(ids_to_delete)} 条记录，并清理了对应的MD5"
        except Exception as e:
            return f"[失败] {e}"

    def delete_by_source(self, source: str) -> str:
        return self.delete_by_metadata({"source": source})

    def insert_string_to_library(text: str, library_name: str, persist_root: str = "./lib"):
        """
        将字符串作为一个整体，插入到指定的向量库中（若库不存在则自动创建）
        :param text: 待插入的文本
        :param library_name: 库名称（将作为 collection_name，并自动创建子目录）
        :param persist_root: 所有库的根目录，实际存储路径为 {persist_root}/{library_name}
        """
        lib_path = os.path.join(persist_root, library_name)
        service = KnowledgeBaseService(collection_name=library_name, persist_directory=lib_path)

        # 生成唯一 ID（使用文本 MD5，也可自定义）
        content_md5 = get_string_md5(text)

        # 强制作为一个整体插入（不切分）
        service.chroma.add_texts(
            texts=[text],
            ids=[content_md5],
            metadatas=[{"source": "user_input", "time": str(datetime.now())}]
        )
        # 可选：调用 persist() 确保写入磁盘（langchain_chroma 通常自动持久化，但显式调用更安全）
        service.chroma.persist()
        print(f"[成功] 文本已插入库 '{library_name}'，ID: {content_md5}")


# ==================== 示例使用 ====================
if __name__ == '__main__':

    # 创建一个指向特定集合和目录的服务实例
    custom_service = KnowledgeBaseService(
        collection_name="my_special_lib",
        persist_directory="./my_custom_db"
    )
    custom_service.upload_by_str("这是自定义库中的内容", "custom.txt")

    # 查询该库
    results = custom_service.query("自定义库查询测试", top_k=2)

    # service = KnowledgeBaseService()
    # 上传示例
    # print(service.upload_by_str("周杰轮222", "testfile"))
    # print(service.upload_by_str("周杰轮222", "testfile"))  # 重复上传应跳过

    # 目录上传示例
    # print(service.upload_by_directory(r"C:\Users\13187\Desktop\output", extensions=[".txt", ".md", ".jsonl"]))

    # 查询示例（自动启用精排重排序）
    # results = service.query("结痂、呼吸困难、舌口及咽部烧灼感、疤痕形成、出血倾向、发音障碍", top_k=3)
    # print("\n精排后的查询结果：")
    for item in results:
        print(f"ID: {item['id']}")
        print(f"文本: {item['text']}")
        print(f"元数据: {item['metadata']}")
        print(f"精排相似度: {item['score']:.4f}")
        print(f"粗排距离: {item['coarse_score']:.4f}")
        print("-" * 50)