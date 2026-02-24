import json
import os
import time
from datetime import datetime

from openai import OpenAI


def read_json_file(file_path):
    """读取JSON文件并处理每条数据"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 未找到")
        return []
    except json.JSONDecodeError:
        print(f"错误: {file_path} 不是有效的JSON文件")
        return []
    except Exception as e:
        print(f"读取文件时发生错误: {str(e)}")
        return []


# 删除文件
def del_file(file_path):
    try:
        # 检查文件是否存在于指定路径
        if os.path.exists(file_path):
            print(f"文件 {file_path} 存在。")
            # 删除文件
            os.remove(file_path)
            print(f"文件 {file_path} 已成功删除。")
        else:
            print(f"文件 {file_path} 不存在。")
    except Exception as e:
        print(f"操作失败：{str(e)}")


# 获取时间戳
def get_timestamp(timestamp_format):
    timestamp = time.time()
    # 将时间戳转换为日期时间对象
    dt = datetime.fromtimestamp(timestamp)

    if len(timestamp_format) == 0:
        timestamp_format = "%Y-%m-%d %H:%M:%S"
    # 将日期时间对象格式化为字符串
    return dt.strftime(timestamp_format)  # 示例格式，可根据需求调整


# 读取 JSON 文件从文件路径中获取，将JSON字符串转换为Python字典
def str2json(json_str):
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"解析失败：{e}")
        return None


# 读取提示词
def get_prompt_str(file_path):
    try:
        with open(file_path, 'r', encoding='UTF-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"文件 {file_path} 不存在。")
    except PermissionError:
        print(f"没有权限读取文件 {file_path}。")


# 模型调用
def use_model(client, prompt, user_val):
    model_result = client.chat.completions.create(
        model="deepseek-r1:14b",  # 请确保此处模型名与 Ollama 使用的名称一致
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_val}
        ],
        # 深度思考模式相关参数
        temperature=0.1,  # 降低随机性，使思考更严谨
        max_tokens=2048
    )
    return model_result.choices[0].message.content.replace("```json", "").replace("```", "")


# 获取模型
def get_model():
    return OpenAI(
        base_url="http://localhost:11434/v1",  # Ollama的本地API地址
        api_key="ollama"  # 此处可填写任意非空字符串，因为本地服务通常无需鉴权[citation:6]
    )


# 读取提示词文件
def read_prompt(file_path: str) -> str:
    """读取提示词文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"错误：提示词文件不存在 - {file_path}")
    except Exception as e:
        print(f"读取提示词文件失败：{e}")
    return ""