# module/utils/default_config.py
class Config:
    persist_directory = "D:/PythonProjects/AiTools/Medical_Diagnosis_Assistant/data/databases/rag_db/chroma_db"
    collection_name = "knowledge_base"
    md5_path = "/Medical_Diagnosis_Assistant/data/output/chroma_db_uploaded_md5.txt"
    chunk_size = 800
    chunk_overlap = 50
    separators = ["\n\n"] # , "。", "！", "？", "；", " ", "", "\n"
    max_split_char_number = 800

    # 嵌入模
    coarse_model_path = "D:/endedingModel/bge-base-zh-v1___5"   # 粗排模型
    fine_model_path = "D:/endedingModel/bge-m3"  # 精排模型（BGE-M3）
    use_gpu = True                     # 是否使用GPU（BGE-M3建议开启）
    rerank_ratio = 2                   # 精排时粗排召回的倍数：最终返回 top_k，粗排召回 top_k * rerank_ratio
    fine_weight = 1.0                  # 精排分数权重（可保留为1）
    coarse_weight = 0.0                # 粗排分数权重（设为0则完全使用精排分数）

config = Config()