class Config:
    # BERT 文件路径
    ROBERTA_BERT = "D:/endedingModel/chinese-roberta-wwm-ext"
    # 训练数据文件结点
    DATA_PATH = "/Medical_Diagnosis_Assistant/data/output/medical_fine_ranked.jsonl"
    # 保存模型结点
    SAVE_MODEL_PATH = "/Medical_Diagnosis_Assistant/model_checkpoint"
    # 最大序列长度
    MAX_LENGTH = 128
    # 训练轮数
    NUM_EPOCHS = 20
    # 批次大小
    BATCH_SIZE = 8
    # 学习率
    LEARNING_RATE = 2e-5
    # 预热步数
    WARMUP_STEPS = 10
    # 权重衰减系数
    WEIGHT_DECAY = 0.01
    # 评估间隔步数
    EVAL_STEPS = 10
    # 保存检查点间隔步数
    SAVE_STEPS = 10
    # 早停耐心值（连续无改善轮数）
    EARLY_STOPPING_PATIENCE = 3
    # 训练集比例
    TRAIN_RATIO = 0.8
    # 验证集比例
    DEV_RATIO = 0.1
config = Config()