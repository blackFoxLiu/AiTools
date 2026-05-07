# -*- coding: utf-8 -*-
import json
import logging
import re
from typing import Optional, Dict

# 获取日志
def get_logger():
    """
        获取logger
    :return:
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)


# 日志配置（只添加一次StreamHandler）
logger = get_logger()


# 读取提示词文件
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
    # 去除可能的 markdown 代码块标记
    cleaned = clean_value(json_str)
    try:
        return json.loads(cleaned), True
    except json.JSONDecodeError:
        # 尝试修复常见错误：将单引号替换为双引号
        try:
            fixed = cleaned.replace("'", '"')
            return json.loads(fixed), True
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败：{e}\n内容：{cleaned[:200]}")
            logger.error(f"JSON 字符串：{cleaned}")
            return {}, False


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
