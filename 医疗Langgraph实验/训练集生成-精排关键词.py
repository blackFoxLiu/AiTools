# -*- coding: utf-8 -*-
import json
import time

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from tqdm import tqdm

from api_req_model import call_model
from common_utils import read_prompt, get_logger, safe_json_parse

logger = get_logger()


if __name__ == '__main__':
    # 使用示例
    input_file = "output/rag_blocks_tmp.jsonl"
    output_file = "./output/medical_fine_ranked" + str(int(time.time())) + ".jsonl"

    # 判断知识片段是否作为有效知识片段
    PROMPT_FINE_TRAIN_KEYWORD_SYS_FILE_PATH = "./prompt/prompt_fine_train_keyword_data_generate_sys.txt"
    PROMPT_FINE_TRAIN_KEYWORD_USER_FILE_PATH = "./prompt/prompt_fine_train_keyword_data_generate_user.txt"
    prompt_fine_train_keyword_data_generate_sys = read_prompt(PROMPT_FINE_TRAIN_KEYWORD_SYS_FILE_PATH)

    # 初始化本地模型客户端
    local_client = ChatOllama(model="deepseek-r1:14b", base_url="http://localhost:11434")

    # 读取所有行（非空）以便 tqdm 显示总数
    with open(input_file, 'r', encoding='utf-8') as read_file:
        lines = [line.strip() for line in read_file if line.strip()]

    with open(output_file, 'w', encoding='utf-8') as write_file:
        # 使用 tqdm 包装迭代器，显示进度
        for line in tqdm(lines, desc="生成关键词微调数据", unit="条"):
            prompt_fine_train_keyword_data_generate_user = (
                PromptTemplate.from_template(read_prompt(PROMPT_FINE_TRAIN_KEYWORD_USER_FILE_PATH))
                .format(input_user=line)
            )
            for retry_idx in range(3):
                _, model_answer = call_model(
                    client=local_client,
                    system_prompt=prompt_fine_train_keyword_data_generate_sys,
                    user_content=prompt_fine_train_keyword_data_generate_user,
                    prefer_remote=True
                )
                safe_json, judge_safe_json = safe_json_parse(model_answer)
                if judge_safe_json:
                    # 写入生成的训练样本（JSON 字符串）
                    save_line = json.dumps(safe_json, ensure_ascii=False) + '\n'
                    write_file.write(save_line)
                    write_file.flush()
                    logger.info(f"关键词微调数据生成成功: {line[:50]}...")
                    break
                logger.error(f"关键词微调数据生成失败，第{retry_idx}次")