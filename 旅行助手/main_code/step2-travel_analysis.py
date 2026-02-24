#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅行助手统计分析脚本
"""

import argparse
import json
import logging
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm import tqdm

# 尝试导入自定义验证函数，若失败则提供占位函数
try:
    from utils.checkJson_travel_analysis import check_travel_analysis
    from utils.common_tools import read_json_file, get_prompt_str, use_model, str2json, get_timestamp
except ImportError:
    print("警告：未找到 common_tools、checkJson_travel_analysis，使用默认验证（始终通过）")

import configparser

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

# ---------- 默认配置 ----------
DEFAULT_INPUT_JSON_PATH = config['file_path']['crawler_file']
DEFAULT_PROMPT_TRAVEL_PATH = config['file_path']['travel_travel_prompt']
DEFAULT_PROMPT_LABEL_PATH = config['file_path']['travel_label_prompt']
DEFAULT_OUTPUT_PATH = config['model_output']['travel_analysis_output_path']  # 建议配置为 .json 后缀

# 模型配置
model_name = config['model_config']['model_name']
DEFAULT_OLLAMA_BASE_URL = config['model_config']['model_url']
DEFAULT_OLLAMA_API_KEY = config['model_config']['api_key']

# 模型重试次数
MAX_RETRIES_TRAVEL = int(config['travel_analysis_retry']['MAX_RETRIES_TRAVEL'])
MAX_RETRIES_LABEL = int(config['travel_analysis_retry']['MAX_RETRIES_LABEL'])
DESC_MIN_LENGTH = int(config['travel_analysis_retry']['DESC_MIN_LENGTH'])

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# ----------------------------


def process_single_record(
    record: Dict[str, Any],
    client: OpenAI,
    prompt_travel: str,
    prompt_label: str
) -> Optional[Dict[str, Any]]:
    """
    处理单条记录：进行旅行分析和景点标注，返回增强后的字典；若失败返回 None。
    """
    video_url = record.get("video_url", "")
    desc = record.get("desc", "")

    # 跳过条件：有视频 或 描述太短
    if video_url or len(desc) < DESC_MIN_LENGTH:
        logger.debug(f"跳过记录: video_url存在或描述过短 (len={len(desc)})")
        return None

    # ---------- 第一步：旅行分析（带重试）----------
    travel_result = None
    try:
        travel_output = use_model(client, prompt_travel, desc)
        travel_dict = str2json(travel_output)
        if travel_dict is None:
            logger.warning(f"旅行分析 JSON 解析失败")
            return travel_result
        main_scenic_list = travel_dict.get("main_scenic_list", [])
        if not isinstance(main_scenic_list, list) or len(main_scenic_list) == 0:
            logger.warning(f"旅行分析结果中 main_scenic_list 为空")
            return None
        travel_result = travel_dict
    except Exception as e:
        logger.error(f"旅行分析调用异常: {e}")

    if travel_result is None:
        logger.error(f"记录 {record.get('note_id', '未知')} 旅行分析失败，跳过")
        return None

    # ---------- 第二步：对每个景点进行标注（独立重试，失败跳过该景点）----------
    main_scenic_list = travel_result.get("main_scenic_list", [])
    for main_scenic in main_scenic_list:
        main_scenic_name = main_scenic.get("main_scenic", "")
        if len(main_scenic_name) == 0:
            continue
        spot_text = f"{main_scenic_name}--{desc}"
        label_success = False
        for attempt in range(MAX_RETRIES_LABEL):
            try:
                label_output = use_model(client, prompt_label, spot_text)
                label_dict = str2json(label_output)
                if label_dict is None:
                    logger.warning(f"景点标注 JSON 解析失败，重试 {attempt+1}/{MAX_RETRIES_LABEL}")
                    continue
                label1 = label_dict.get("最倾向一级标签", "")
                label2 = label_dict.get("最倾向二级标签", "")
                main_scenic["tendency_label_1"] = label1
                main_scenic["tendency_label_2"] = label2
                label_success = True
                break
            except Exception as e:
                logger.error(f"景点标注异常 (attempt {attempt+1}): {e}")

        if not label_success:
            logger.warning(f"景点 '{main_scenic_name}' 标注失败，跳过该景点标签")

    # ---------- 第三步：补充原始记录字段 ----------
    travel_result["desc"] = desc
    travel_result["liked_count"] = record.get("liked_count", "")
    travel_result["collected_count"] = record.get("collected_count", "")
    travel_result["comment_count"] = record.get("comment_count", "")
    travel_result["share_count"] = record.get("share_count", "")
    travel_result["user_id"] = record.get("user_id", "")
    travel_result["note_id"] = record.get("note_id", "")

    # ---------- 第四步：验证结果格式 ----------
    try:
        is_valid = check_travel_analysis(travel_result)
    except Exception as e:
        logger.error(f"验证函数执行异常: {e}")
        return None
    if not is_valid:
        logger.warning(f"验证失败")
        return None

    return travel_result


def main():
    parser = argparse.ArgumentParser(description="旅行文案分析与景点标注工具")
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT_JSON_PATH,
                        help=f"输入JSON文件路径 (默认: {DEFAULT_INPUT_JSON_PATH})")
    parser.add_argument("--output", type=str, default=None,
                        help=f"输出JSON文件路径，若不指定则使用配置文件中的路径 (默认: {DEFAULT_OUTPUT_PATH})")
    parser.add_argument("--prompt_travel", type=str, default=DEFAULT_PROMPT_TRAVEL_PATH,
                        help=f"旅行分析提示词文件路径 (默认: {DEFAULT_PROMPT_TRAVEL_PATH})")
    parser.add_argument("--prompt_label", type=str, default=DEFAULT_PROMPT_LABEL_PATH,
                        help=f"景点标注提示词文件路径 (默认: {DEFAULT_PROMPT_LABEL_PATH})")
    parser.add_argument("--ollama_url", type=str, default=DEFAULT_OLLAMA_BASE_URL,
                        help=f"Ollama服务地址 (默认: {DEFAULT_OLLAMA_BASE_URL})")
    parser.add_argument("--ollama_api_key", type=str, default=DEFAULT_OLLAMA_API_KEY,
                        help=f"Ollama API密钥 (默认: {DEFAULT_OLLAMA_API_KEY})")
    args = parser.parse_args()

    output_path = args.output if args.output else DEFAULT_OUTPUT_PATH

    # 1. 读取数据
    data = read_json_file(args.input_json)
    logger.info(f"成功读取 {len(data)} 条记录")

    # 2. 初始化 OpenAI 客户端
    client = OpenAI(base_url=args.ollama_url, api_key=args.ollama_api_key)

    # 3. 预读取提示词文件
    prompt_travel = get_prompt_str(args.prompt_travel)
    prompt_label = get_prompt_str(args.prompt_label)

    success_count = 0
    results: List[Dict[str, Any]] = []  # 收集所有成功处理的结果

    # 4. 逐条处理并收集
    for record in tqdm(data, desc="处理进度"):
        result = process_single_record(record, client, prompt_travel, prompt_label)
        if result is not None:
            results.append(result)
            success_count += 1
            logger.debug(f"成功处理一条记录，累计成功数: {success_count}")
        else:
            logger.debug(f"记录处理失败，已跳过")

    # 5. 一次性写入完整的 JSON 数组文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)  # indent=2 使输出更可读

    logger.info(f"处理完成，成功写入 {success_count} 条记录到 {output_path}")


if __name__ == "__main__":
    main()