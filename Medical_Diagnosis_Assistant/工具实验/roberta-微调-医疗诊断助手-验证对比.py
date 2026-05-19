# -*- coding: utf-8 -*-
"""
对比原始 RoBERTa 模型与微调后的精排模型在 Query-Document 相关性二分类上的效果。

用法:
    1. 交互模式: python compare_models.py
    2. 命令行模式: python compare_models.py --query "查询文本" --doc "文档文本"

输出两个模型的预测类别 (0/1) 以及正类 (label=1) 的概率。
"""
import os
import sys

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import logging

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))

# 将项目根目录添加到 Python 的模块搜索路径中
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    from config.bert_config import config as bert_config
except ImportError:
    raise RuntimeError(f"导入模块失败")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path, num_labels=2, device=None):
    """
    加载模型和分词器。

    Args:
        model_path: 模型路径（本地目录或 HuggingFace 模型名）
        num_labels: 分类类别数（仅对原始模型有效，微调模型会忽略此参数）
        device: torch.device

    Returns:
        model, tokenizer
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 尝试加载为微调后的完整模型（包含分类头配置）
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        logger.info(f"成功从 {model_path} 加载微调后的模型（使用其自身配置）")
    except Exception as e:
        # 如果失败，则当作原始预训练模型处理，需手动指定 num_labels
        logger.info(f"未找到微调模型配置，尝试作为原始模型加载: {e}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        logger.info(f"从 {model_path} 加载原始模型，并添加 {num_labels} 分类头")

    model.to(device)
    model.eval()
    return model, tokenizer, device


def predict(query, document, model, tokenizer, device, max_length=128):
    """
    对单个 Query-Document 对进行预测，返回预测类别和正类概率。

    Returns:
        pred_label (int): 0 或 1
        prob_positive (float): 属于类别 1 的概率
    """
    inputs = tokenizer(
        query, document,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)  # (batch, 2)
        prob_positive = probs[0, 1].item()
        pred_label = torch.argmax(probs, dim=-1).item()

    return pred_label, prob_positive


def main():
    parser = argparse.ArgumentParser(description="对比原始模型与微调模型的相关性预测效果")
    parser.add_argument("--query", type=str, help="查询文本")
    parser.add_argument("--doc", type=str, help="文档文本")
    parser.add_argument("--base_model", type=str,
                        default=bert_config.ROBERTA_BERT,
                        help="原始预训练模型路径或名称")
    parser.add_argument("--finetuned_model", type=str,
                        default=bert_config.SAVE_MODEL_PATH,
                        help="微调后的模型目录")
    parser.add_argument("--max_length", type=int, default=128, help="输入最大长度")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")

    # 加载两个模型
    base_model, base_tokenizer, device = load_model_and_tokenizer(args.base_model, num_labels=2, device=device)
    ft_model, ft_tokenizer, _ = load_model_and_tokenizer(args.finetuned_model, device=device)

    # 交互模式或单次预测
    if args.query and args.doc:
        queries_docs = [(args.query, args.doc)]
    else:
        logger.info("未提供 --query 和 --doc，进入交互模式（输入 'quit' 退出）")
        queries_docs = []
        while True:
            q = input("请输入 Query: ").strip()
            if q.lower() == 'quit':
                break
            d = input("请输入 Document: ").strip()
            if d.lower() == 'quit':
                break
            queries_docs.append((q, d))

    # 对每对进行预测并输出对比结果
    for idx, (query, doc) in enumerate(queries_docs):
        print(f"\n========== 样本 {idx+1} ==========")
        print(f"Query: {query}")
        print(f"Document: {doc[:200]}{'...' if len(doc) > 200 else ''}")

        # 原始模型预测
        base_label, base_prob = predict(query, doc, base_model, base_tokenizer, device, args.max_length)
        print(f"\n【原始模型】")
        print(f"  预测类别: {base_label} ({'相关' if base_label == 1 else '不相关'})")
        print(f"  正类概率: {base_prob:.4f}")

        # 微调模型预测
        ft_label, ft_prob = predict(query, doc, ft_model, ft_tokenizer, device, args.max_length)
        print(f"\n【微调模型】")
        print(f"  预测类别: {ft_label} ({'相关' if ft_label == 1 else '不相关'})")
        print(f"  正类概率: {ft_prob:.4f}")

        # 简要对比
        print(f"\n-> 结论: 原始模型预测为 {'相关' if base_label == 1 else '不相关'}，微调模型预测为 {'相关' if ft_label == 1 else '不相关'}")


if __name__ == "__main__":
    """
        输入问题【query】和【document】之后，输入【quit】进行退出。
    """
    main()