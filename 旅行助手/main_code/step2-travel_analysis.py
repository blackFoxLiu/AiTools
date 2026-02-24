#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅行助手统计分析脚本
"""

import argparse
import json
import logging
from typing import Any, Dict, Optional

from openai import OpenAI
from tqdm import tqdm

# 尝试导入自定义验证函数，若失败则提供占位函数
try:
    # 假设这些自定义函数存在
    from utils.checkJson_travel_analysis import check_travel_analysis
    from utils.common_tools import read_json_file, get_prompt_str, use_model, str2json, get_timestamp
except ImportError:
    print("警告：未找到 common_tools、checkJson_travel_analysis，使用默认验证（始终通过）")


import configparser

config = configparser.ConfigParser()
config.read('config.ini', encoding='utf-8')

# ---------- 默认配置 ----------
# DEFAULT_OUTPUT_TXT_PATH = "C:/Users/13187/Desktop/travelAnalysis_" + get_timestamp("%Y%m%d-%H%M%S") + ".txt"

# 文件信息配置
DEFAULT_INPUT_JSON_PATH = config['file_path']['crawler_file']
DEFAULT_PROMPT_TRAVEL_PATH = config['file_path']['travel_travel_prompt']
DEFAULT_PROMPT_LABEL_PATH = config['file_path']['travel_label_prompt']
DEFAULT_OUTPUT_TXT_PATH = config['model_output']['travel_analysis_output_path']

# 模型配置
model_name = config['model_config']['model_name']
DEFAULT_OLLAMA_BASE_URL = config['model_config']['model_url']
DEFAULT_OLLAMA_API_KEY = config['model_config']['api_key']

# 模型重试次数
MAX_RETRIES_TRAVEL = config['travel_analysis_retry']['MAX_RETRIES_TRAVEL']      # 旅行分析模型调用最大重试次数
MAX_RETRIES_LABEL = config['travel_analysis_retry']['MAX_RETRIES_LABEL']       # 景点标注模型调用最大重试次数
DESC_MIN_LENGTH = config['travel_analysis_retry']['DESC_MIN_LENGTH']        # 描述最短长度要求

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
    for attempt in range(MAX_RETRIES_TRAVEL):
        try:
            travel_output = use_model(client, prompt_travel, desc)
            travel_dict = str2json(travel_output)
            if travel_dict is None:
                logger.warning(f"旅行分析 JSON 解析失败，重试 {attempt+1}/{MAX_RETRIES_TRAVEL}")
                continue
            # 检查是否包含 journeys 字段且非空
            journeys = travel_dict.get("journeys", [])
            if not isinstance(journeys, list) or len(journeys) == 0:
                logger.warning(f"旅行分析结果中 journeys 为空，重试 {attempt+1}/{MAX_RETRIES_TRAVEL}")
                continue
            travel_result = travel_dict
            break
        except Exception as e:
            logger.error(f"旅行分析调用异常 (attempt {attempt+1}): {e}")

    if travel_result is None:
        logger.error(f"记录 {record.get('note_id', '未知')} 旅行分析失败，跳过")
        return None

    # ---------- 第二步：对每个景点进行标注（独立重试，失败跳过该景点）----------
    journeys = travel_result.get("journeys", [])
    for spot in journeys:
        if not isinstance(spot, dict):
            continue
        scenic_intro = spot.get("scenic_intro", "")
        scenic_name = spot.get("scenic", "")
        if not scenic_intro:  # 无介绍则跳过标注
            continue

        # 构建标注输入文本
        spot_text = f"{scenic_name}--{scenic_intro}"

        label_success = False
        for attempt in range(MAX_RETRIES_LABEL):
            try:
                label_output = use_model(client, prompt_label, spot_text)
                label_dict = str2json(label_output)
                if label_dict is None:
                    logger.warning(f"景点标注 JSON 解析失败，重试 {attempt+1}/{MAX_RETRIES_LABEL}")
                    continue
                # 提取标签
                label1 = label_dict.get("最倾向一级标签", "")
                label2 = label_dict.get("最倾向二级标签", "")
                spot["tendency_label_1"] = label1
                spot["tendency_label_2"] = label2
                label_success = True
                break
            except Exception as e:
                logger.error(f"景点标注异常 (attempt {attempt+1}): {e}")

        if not label_success:
            logger.warning(f"景点 '{scenic_name}' 标注失败，跳过该景点标签")

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
        is_valid, errors = check_travel_analysis(travel_result)
    except Exception as e:
        logger.error(f"验证函数执行异常: {e}")
        return None

    if not is_valid:
        for err in errors:
            logger.warning(f"验证失败: {err}")
        return None

    return travel_result


def main():
    parser = argparse.ArgumentParser(description="旅行文案分析与景点标注工具")
    parser.add_argument("--input_json", type=str, default=DEFAULT_INPUT_JSON_PATH,
                        help=f"输入JSON文件路径 (默认: {DEFAULT_INPUT_JSON_PATH})")
    parser.add_argument("--output_txt", type=str, default=None,
                        help=f"输出TXT文件路径，若不指定则自动生成带时间戳的文件名 (默认自动生成)")
    parser.add_argument("--prompt_travel", type=str, default=DEFAULT_PROMPT_TRAVEL_PATH,
                        help=f"旅行分析提示词文件路径 (默认: {DEFAULT_PROMPT_TRAVEL_PATH})")
    parser.add_argument("--prompt_label", type=str, default=DEFAULT_PROMPT_LABEL_PATH,
                        help=f"景点标注提示词文件路径 (默认: {DEFAULT_PROMPT_LABEL_PATH})")
    parser.add_argument("--ollama_url", type=str, default=DEFAULT_OLLAMA_BASE_URL,
                        help=f"Ollama服务地址 (默认: {DEFAULT_OLLAMA_BASE_URL})")
    parser.add_argument("--ollama_api_key", type=str, default=DEFAULT_OLLAMA_API_KEY,
                        help=f"Ollama API密钥 (默认: {DEFAULT_OLLAMA_API_KEY})")
    args = parser.parse_args()

    # 如果未指定输出文件路径，则使用带时间戳的默认路径
    output_txt_path = args.output_txt if args.output_txt else DEFAULT_OUTPUT_TXT_PATH

    # 1. 读取数据
    data = read_json_file(args.input_json)
    logger.info(f"成功读取 {len(data)} 条记录")

    # 2. 初始化 OpenAI 客户端（指向本地 Ollama）
    client = OpenAI(base_url=args.ollama_url, api_key=args.ollama_api_key)

    # 3. 预读取提示词文件
    prompt_travel = get_prompt_str(args.prompt_travel)
    prompt_label = get_prompt_str(args.prompt_label)

    success_count = 0
    # 4. 打开输出文件（追加模式）
    with open(output_txt_path, 'a', encoding='utf-8') as f:
        for record in tqdm(data, desc="处理进度"):
            result = process_single_record(record, client, prompt_travel, prompt_label)
            if result is not None:
                # 将字典转为 JSON 字符串（确保双引号）
                json_str = json.dumps(result, ensure_ascii=False)
                f.write(json_str + ',\n')
                f.flush()
                success_count += 1
                logger.debug(f"成功写入一条记录，累计成功数: {success_count}")
            else:
                logger.debug(f"记录处理失败，已跳过")

    logger.info(f"处理完成，成功写入 {success_count} 条记录到 {output_txt_path}")


if __name__ == "__main__":
    main()