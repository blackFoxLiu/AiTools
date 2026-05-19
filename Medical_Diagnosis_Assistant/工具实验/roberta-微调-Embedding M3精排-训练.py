# -*- coding: utf-8 -*-
"""
使用中文RoBERTa模型进行Query-Document相关性二分类微调（精排任务）
数据格式：每行一个JSON数组，数组内每个元素为：
    {"query": "...", "pos_desc": "...", "neg_desc": "...", "label": 1}
自动构造正例(label=1)和负例(label=0)，并划分训练/验证/测试集
"""
import sys

import torch
from torch.utils.data import Dataset, random_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
import logging
import os
import json
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

# 将项目根目录添加到 Python 的模块搜索路径中
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    from config.bert_config import config as bert_config
except ImportError:
    raise RuntimeError(f"导入模块失败")


class QueryDocDataset(Dataset):
    """Query-Document相关性数据集，预先完成tokenization，不进行静态填充"""

    def __init__(self, tokenizer, data_file: str, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = self._load_and_tokenize(data_file)
        logger.info(f"从 {data_file} 加载并处理了 {len(self.examples)} 个样本")

    def _load_and_tokenize(self, data_file: str):
        """加载原始数据并立即进行tokenization（不填充）"""
        raw_data = self._load_json(data_file)
        examples = []

        for idx, item in enumerate(raw_data):
            # 安全获取字段，跳过无效数据
            query = item.get('query')
            pos_doc = item.get('pos_desc')
            neg_doc = item.get('neg_desc')

            if not query or not pos_doc or not neg_doc:
                logger.warning(f"跳过第 {idx} 条数据：缺少必要字段 (query/pos_desc/neg_desc) -> {item}")
                continue

            # 正例
            pos_enc = self.tokenizer(
                query, pos_doc,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors='pt'
            )
            examples.append({
                'input_ids': pos_enc['input_ids'][0],
                'attention_mask': pos_enc['attention_mask'][0],
                'labels': torch.tensor(1, dtype=torch.long)
            })

            # 负例
            neg_enc = self.tokenizer(
                query, neg_doc,
                truncation=True,
                max_length=self.max_length,
                padding=False,
                return_tensors='pt'
            )
            examples.append({
                'input_ids': neg_enc['input_ids'][0],
                'attention_mask': neg_enc['attention_mask'][0],
                'labels': torch.tensor(0, dtype=torch.long)
            })

        return examples

    def _load_json(self, data_file: str):
        """加载JSON文件，支持每行一个JSON数组（数组内为对象）"""
        result = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    # 如果是数组，遍历每个元素
                    if isinstance(obj, list):
                        for sub_obj in obj:
                            if isinstance(sub_obj, dict):
                                # 只保留包含必要字段的字典
                                if all(k in sub_obj for k in ('query', 'pos_desc', 'neg_desc')):
                                    result.append(sub_obj)
                                else:
                                    logger.warning(f"第 {line_no} 行数组中元素缺少必要字段：{sub_obj}")
                            else:
                                logger.warning(f"第 {line_no} 行数组包含非字典元素：{sub_obj}")
                    elif isinstance(obj, dict):
                        if all(k in obj for k in ('query', 'pos_desc', 'neg_desc')):
                            result.append(obj)
                        else:
                            logger.warning(f"第 {line_no} 行对象缺少必要字段：{obj}")
                    else:
                        logger.warning(f"第 {line_no} 行不是JSON对象或数组：{obj}")
                except json.JSONDecodeError as e:
                    logger.warning(f"第 {line_no} 行JSON解析失败：{line[:50]}... 错误：{e}")
        logger.info(f"成功加载 {len(result)} 条有效数据记录")
        return result

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    prec = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': prec,
        'recall': rec
    }


def main():
    # ==================== 配置区域 ====================
    # 预训练模型路径
    model_name = bert_config.ROBERTA_BERT
    # 训练数据文件路径
    data_file = bert_config.DATA_PATH
    # 模型保存目录
    output_dir = bert_config.SAVE_MODEL_PATH
    # 最大序列长度
    max_length = bert_config.MAX_LENGTH
    # 训练轮数
    num_epochs = bert_config.NUM_EPOCHS
    # 批次大小
    batch_size = bert_config.BATCH_SIZE
    # 学习率
    learning_rate = bert_config.LEARNING_RATE
    # 预热步数
    warmup_steps = bert_config.WARMUP_STEPS
    # 权重衰减系数
    weight_decay = bert_config.WEIGHT_DECAY
    # 评估间隔步数
    eval_steps = bert_config.EVAL_STEPS
    # 保存检查点间隔步数
    save_steps = bert_config.SAVE_STEPS
    # 早停耐心值
    early_stopping_patience = bert_config.EARLY_STOPPING_PATIENCE
    # 训练集比例
    train_ratio = bert_config.TRAIN_RATIO
    # 验证集比例
    dev_ratio = bert_config.DEV_RATIO
    # =================================================

    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_name):
        logger.error(f"模型路径不存在: {model_name}")
        return
    if not os.path.exists(data_file):
        logger.error(f"数据文件不存在: {data_file}")
        return

    logger.info(f"加载本地tokenizer和模型: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        problem_type="single_label_classification"
    )

    logger.info("加载并预处理数据集...")
    full_dataset = QueryDocDataset(tokenizer, data_file, max_length=max_length)

    if len(full_dataset) == 0:
        logger.error("数据集为空，请检查数据文件")
        return

    total_len = len(full_dataset)
    train_len = int(total_len * train_ratio)
    dev_len = int(total_len * dev_ratio)
    test_len = total_len - train_len - dev_len
    train_dataset, dev_dataset, test_dataset = random_split(
        full_dataset,
        [train_len, dev_len, test_len],
        generator=torch.Generator().manual_seed(42)
    )

    logger.info(f"总样本数: {total_len}")
    logger.info(f"训练集: {len(train_dataset)}, 验证集: {len(dev_dataset)}, 测试集: {len(test_dataset)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        logging_dir=f'{output_dir}/logs',
        logging_steps=5,
        eval_strategy='steps',
        eval_steps=eval_steps,
        save_strategy='steps',
        save_steps=save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        greater_is_better=True,
        learning_rate=learning_rate,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        report_to='none',
        seed=42,
        remove_unused_columns=False,
    )

    # 动态填充
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=early_stopping_patience)],
        data_collator=data_collator,
    )

    logger.info("开始训练模型...")
    trainer.train()

    logger.info(f"保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    logger.info("在测试集上评估最佳模型...")
    test_metrics = trainer.evaluate(eval_dataset=test_dataset)
    test_metrics = {f"test_{k}": v for k, v in test_metrics.items()}
    logger.info(f"测试集结果: {test_metrics}")

    logger.info("训练完成！")


if __name__ == "__main__":
    main()