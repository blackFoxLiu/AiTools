# -*- coding: utf-8 -*-
import json
import logging
import re
from functools import lru_cache
from typing import Optional, Dict

from langchain_ollama import ChatOllama

import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))  # 根据实际文件位置调整层级
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

try:
    from utils.json_model_tools import safe_json_by_model
except ImportError:
    raise RuntimeError(f"导入模块失败")



# 获取日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==================== 常量配置 ====================
OLLAMA_CONFIG = {
    "model": "qwen3:8b",
    "base_url": "http://127.0.0.1:11434",
    "temperature": 0.0
}

# ==================== 模型获取（缓存）====================
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

@lru_cache(maxsize=5)
def read_prompt(file_path: str) -> str:
    """读取提示词文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"错误：提示词文件不存在 - {file_path}")
    except Exception as e:
        logger.error(f"读取提示词文件失败：{e}")
    return ""

def safe_json_parse(json_str: str) -> Optional[Dict]:
    """安全解析JSON字符串，处理可能的格式问题"""
    cleaned = clean_value(json_str)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 尝试修复常见错误：将单引号替换为双引号
        try:
            fixed = safe_json_by_model(cleaned.replace("'", '"'))
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败：{e}\n内容：{cleaned[:200]}")
            logger.error(f"JSON 字符串：{cleaned}")
            return {}


def clean_value(value):
    """递归清理数据中的字符串：去除 ```json、``` 以及所有空白字符"""
    if isinstance(value, str):
        # 移除标记
        value = value.replace("```json", "").replace("```", "")
        # 移除所有空白字符（空格、制表符、换行等）
        value = re.sub(r'\s+', '', value)
        return value
    elif isinstance(value, dict):
        # 如果是字典，递归清理每个键值对
        return {k: clean_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        # 如果是列表，递归清理每个元素
        return [clean_value(item) for item in value]
    else:
        # 其他类型（数字、布尔等）直接返回
        return value
