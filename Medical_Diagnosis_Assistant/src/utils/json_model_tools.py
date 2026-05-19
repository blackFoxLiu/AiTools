#!/usr/bin/env python3
"""
JSON 修复工具 - 基于微调后的 Qwen2.5-Coder-1.5B-Instruct 模型。

提供简单的函数接口 safe_json_by_model()，内部自动加载并缓存微调模型。
直接使用微调后的合并模型（最新模型），无需对比原始模型。

使用方法：
    from json_repair_tool import safe_json_by_model
    result = safe_json_by_model('{"name": "Alice", "age": 30,}')
    logger.info(result)   # 输出修复后的 JSON 字符串
"""
import logging
import os
import re
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 将项目根目录添加到 Python 的模块搜索路径中
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    from tools.knowledge_graph_tools import Neo4jQueryTools
    from tools.rag_module import KnowledgeBaseService
except ImportError:
    raise RuntimeError(f"导入模块失败")

# ==================== 日志配置 ====================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================== 默认配置 =====================

DEFAULT_MODEL_PATH = "D:/PythonProjects/AiTools/JSON模型微调/qwen_json_fixer_unsloth/merged_model"  # 微调后的合并模型路径
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.1
DEFAULT_DO_SAMPLE = False

# 示例数据（用于上下文学习，帮助模型理解修复风格）
_EXAMPLE_NORM_JSON = '{"rY9tL\\nZ": "nZ5tP\\nM", "mT4nKq": 8472, "pW9jR": true, "sX7nT": ["mP5kL", 2947, false, null, {"eQ8nP": "wT3rL", "vX9jK": true}], "fK7nQ": {"qT4mL": 8240.0, "rY9wJ": [true, 1847, null], "kP8nW": {"pQ5mL": 82.47, "wT7kN": "mR8nL"}}, "vX3nK": "mP8kR\\tP"}'
_EXAMPLE_ERR_JSON = """`{'rY9tL\\nZ': "nZ5tP\\nM', 'mT4nKq': 8472 'pW9jR': True, 'sX7nT': ['mP5kL', 2947, False, None, {'eQ8nP': 'wT3rL', 'vX9jK': True}], 'fK7nQ': {'qT4mL': 8240.0, 'rY9wJ': [True, 1847, None], 'kP8nW': {'pQ5mL': 82.47, 'wT7kN': 'mR8nL'}}}, 'vX3nK': 'mP8kR\\tP'}`"""

# ===================== 全局模型缓存（单例） =====================
_model = None
_tokenizer = None


def _get_device_and_dtype():
    """自动选择设备与数据类型"""
    if torch.cuda.is_available():
        return "cuda", torch.float16
    else:
        return "cpu", torch.float32


def _load_model(model_path: str):
    """加载模型和分词器（内部使用）"""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return  # 已加载，直接返回

    logger.info(f"正在加载微调模型: {model_path}")
    device, dtype = _get_device_and_dtype()
    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    _tokenizer = AutoTokenizer.from_pretrained(model_path)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token
    _tokenizer.padding_side = "right"
    _model.eval()
    logger.info("模型加载完成。")


def _build_chat_prompt(user_input: str) -> str:
    """构建 Qwen Chat 格式的 prompt，包含一个 JSON 修复示例"""
    system_and_example = f"""你是一个专业的JSON修复助手。你的任务是修复用户提供的错误JSON，严格遵守以下规则：
1. 不增加、不删除任何键值对，只修正格式错误（如引号、逗号、括号、大小写等）。
2. 输出必须是纯JSON对象，不要包含任何额外解释、注释或代码块标记。
3. 参考下面给出的示例修复风格。

示例：
输入（错误JSON）：
{_EXAMPLE_ERR_JSON}

输出（正确JSON）：
{_EXAMPLE_NORM_JSON}

现在请修复用户提供的错误JSON："""
    full_user_content = system_and_example + "\n" + user_input
    return f"<|im_start|>user\n{full_user_content}<|im_end|>\n<|im_start|>assistant\n"


def _clean_response(text: str) -> str:
    """清理模型输出，提取 JSON 内容"""
    # 去掉可能存在的 markdown 代码块
    cleaned = re.sub(r'^```json\s*', '', text)
    cleaned = re.sub(r'```$', '', cleaned)
    cleaned = cleaned.strip()
    return cleaned


def safe_json_by_model(bad_json: str,
                model_path: str = DEFAULT_MODEL_PATH,
                max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                temperature: float = DEFAULT_TEMPERATURE,
                do_sample: bool = DEFAULT_DO_SAMPLE,
                verbose: bool = False) -> str:
    """
    修复错误的 JSON 字符串。

    参数：
        bad_json (str): 包含格式错误的 JSON 字符串。
        model_path (str): 微调模型的路径（默认使用合并模型目录）。
        max_new_tokens (int): 生成的最大 token 数。
        temperature (float): 生成温度（仅在 do_sample=True 时生效）。
        do_sample (bool): 是否随机采样（设为 False 使用贪心解码）。
        verbose (bool): 是否打印详细信息（如 prompt、原始输出等）。

    返回：
        str: 修复后的 JSON 字符串（纯 JSON 格式）。
    """
    # 1. 加载模型（全局单例，避免重复加载）
    _load_model(model_path)

    # 2. 构建 prompt
    prompt = _build_chat_prompt(bad_json)
    if verbose:
        logger.info(f"生成的 Prompt (前200字符):\n{prompt[:200]}...\n")

    # 3. 生成回答
    inputs = _tokenizer(prompt, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=_tokenizer.eos_token_id,
            use_cache=True,
        )
    full_response = _tokenizer.decode(outputs[0], skip_special_tokens=False)

    # 4. 提取 assistant 部分
    if "<|im_start|>assistant\n" in full_response:
        assistant_part = full_response.split("<|im_start|>assistant\n")[-1]
        assistant_part = assistant_part.split("<|im_end|>")[0].strip()
    else:
        assistant_part = full_response[len(prompt):].strip()

    if verbose:
        logger.info(f"原始模型输出:\n{assistant_part}\n")

    # 5. 清理输出并返回
    cleaned = _clean_response(assistant_part)
    return cleaned


# ===================== 可选：对比功能（同时使用原始模型和微调模型） =====================
def compare_models(bad_json: str,
                   base_model_path: str = "D:/Model/Qwen2.5-Coder-1.5B-Instruct",
                   merged_model_path: str = DEFAULT_MODEL_PATH,
                   **kwargs) -> dict:
    """
    同时使用原始模型和微调模型修复 JSON，返回两者的结果。

    参数：
        bad_json (str): 错误的 JSON 字符串。
        base_model_path (str): 原始未微调模型的路径。
        merged_model_path (str): 微调合并模型的路径。
        **kwargs: 其他生成参数（max_new_tokens, temperature 等）。

    返回：
        dict: {"base": 原始模型输出, "merged": 微调模型输出}
    """
    # 临时加载两个模型（不干扰全局单例）
    base_model, base_tokenizer = _load_model_explicit(base_model_path)
    merged_model, merged_tokenizer = _load_model_explicit(merged_model_path)

    prompt = _build_chat_prompt(bad_json)

    def _gen(model, tokenizer):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS),
                temperature=kwargs.get("temperature", DEFAULT_TEMPERATURE),
                do_sample=kwargs.get("do_sample", DEFAULT_DO_SAMPLE),
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        full = tokenizer.decode(outputs[0], skip_special_tokens=False)
        if "<|im_start|>assistant\n" in full:
            assistant = full.split("<|im_start|>assistant\n")[-1].split("<|im_end|>")[0].strip()
        else:
            assistant = full[len(prompt):].strip()
        return _clean_response(assistant)

    base_ans = _gen(base_model, base_tokenizer)
    merged_ans = _gen(merged_model, merged_tokenizer)

    # 清理显存
    del base_model, merged_model
    torch.cuda.empty_cache()

    return {"base": base_ans, "merged": merged_ans}


def _load_model_explicit(model_path: str):
    """显式加载模型（不缓存），供 compare_models 使用"""
    device, dtype = _get_device_and_dtype()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto" if device == "cuda" else None,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model.eval()
    return model, tokenizer


# ===================== 命令行入口（可选） =====================
if __name__ == "__main__":
    # 示例用法
    test_json = '{"name": "Alice", "age": 30,}'   # 尾部多余逗号
    logger.info("测试 JSON:", test_json)
    result = safe_json_by_model(test_json, verbose=True)
    logger.info("\n修复结果:", result)

    # 也可以使用对比功能
    # comparison = compare_models(test_json)
    # logger.info("原始模型:", comparison["base"])
    # logger.info("微调模型:", comparison["merged"])