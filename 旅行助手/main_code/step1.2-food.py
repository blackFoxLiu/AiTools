#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
旅行助手数据处理脚本
读取包含旅行描述的JSON文件，使用两个提示词模型进行分析和优化，
并将结果保存为JSON数组文件。
"""

import argparse
import json
from datetime import datetime
from typing import Dict, Optional

from openai import OpenAI
from tqdm import tqdm

# 尝试导入自定义验证函数，若失败则提供占位函数
try:
    from utils.checkJson_food import checkJson_food
    from utils.common_tools import read_prompt, read_json_file
except ImportError:
    print("警告：未找到 checkJson_travel_tools，使用默认验证（始终通过）")


def get_timestamp(format_str: str = "%Y%m%d-%H%M%S") -> str:
    """返回当前时间戳字符串"""
    return datetime.now().strftime(format_str)


def safe_json_parse(json_str: str) -> Optional[Dict]:
    """安全解析JSON字符串，处理可能的格式问题"""
    # 去除可能的 markdown 代码块标记
    cleaned = json_str.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 尝试修复常见错误：将单引号替换为双引号
        try:
            fixed = cleaned.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError as e:
            print(f"JSON解析失败：{e}\n内容：{cleaned[:200]}")
            return None


# ---------------------------- 模型调用 ----------------------------
def get_ollama_client(base_url: str = "http://localhost:11434/v1",
                      api_key: str = "ollama") -> OpenAI:
    """创建Ollama客户端"""
    return OpenAI(base_url=base_url, api_key=api_key)


def call_model(client: OpenAI, system_prompt: str, user_content: str,
               model: str = "deepseek-r1:14b", temperature: float = 0.1,
               max_tokens: int = 2048) -> str:
    """调用模型并返回内容"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"模型调用失败：{e}")
        return ""


# ---------------------------- 核心处理 ----------------------------
def process_record(record: Dict, client: OpenAI,
                   prompt_tools: str) -> Optional[Dict]:
    """
    处理单条记录
    返回优化后的结果字典，若跳过则返回None
    """
    desc = record.get("desc", "")
    if not desc:
        print("警告：记录缺少 desc 字段，跳过")
        return None

    # 第一步：工具分析
    food_response = call_model(client, prompt_tools, desc)
    if not food_response:
        return None

    food_data = safe_json_parse(food_response)
    if not food_data:
        return None

    # 补充额外字段
    food_data["node_id"] = record.get("note_id", "")

    # 验证
    if not checkJson_food(food_data):
        print("验证失败，跳过")
        return None

    return food_data


def main():
    import configparser

    config = configparser.ConfigParser()
    config.read('config.ini', encoding='utf-8')

    # 文件信息配置
    crawler_file_path = config['file_path']['CRAWLER_FILE']
    food_prompt_path = config['file_path']['FOOD_PROMPT']
    food_output_path = config['model_output']['FOOD_OUTPUT_PATH']

    # 模型配置
    model_name = config['model_config']['model_name']
    model_url = config['model_config']['model_url']
    api_key = config['model_config']['api_key']

    parser = argparse.ArgumentParser(description="旅行助手数据处理")
    parser.add_argument("--input", "-i", default=crawler_file_path,
                        help="输入JSON文件路径")
    parser.add_argument("--prompt-food", "-pt", default=food_prompt_path,
                        help="美食分析提示词文件路径")
    parser.add_argument("--output", "-o", default=food_output_path,
                        help="输出文件路径（默认自动生成）")
    parser.add_argument("--model", default=model_name,
                        help="模型名称")
    parser.add_argument("--base-url", default=model_url,
                        help="Ollama API地址")
    parser.add_argument("--api-key", default=api_key,
                        help="API密钥")
    args = parser.parse_args()

    # 读取提示词
    prompt_food = read_prompt(args.prompt_food)

    # 读取输入数据
    data = read_json_file(args.input)
    if not data:
        return
    print(f"成功读取 {len(data)} 条记录")

    # 初始化客户端
    client = get_ollama_client(args.base_url, args.api_key)

    results = []
    success_count = 0
    error_count = 0

    # 处理每条记录
    for record in tqdm(data, desc="处理进度"):
        try:
            result = process_record(record, client, prompt_food)
            if result:
                results.append(result)
                success_count += 1
                tqdm.write(f"成功处理 {success_count} 条")
                print(result)
            else:
                error_count += 1
        except Exception as e:
            error_count += 1
            tqdm.write(f"处理记录时发生异常：{e}")

    # 写入结果
    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n处理完成！成功：{success_count} 条，失败：{error_count} 条")
        print(f"结果已保存至：{args.output}")
    except Exception as e:
        print(f"写入输出文件失败：{e}")


if __name__ == "__main__":
    main()