# default_config.py
# 远程模型配置（可放在项目根目录，或通过环境变量覆盖）

import os

# MiniMax 远程模型配置
REMOTE_CONFIG = {
    "api_key": os.getenv("MINIMAX_API_KEY", "sk-cp-mC5XmJWKwt_FFLCUAkv6CmE2n6PuS9_-Q3vrP1rLCEamVW9di8Y4P1MD3qts_aVokA2dxmT2Y1aeOUpEQONtwSbrURIwrPc8ldP8f8dfBkE"),  # 建议使用环境变量
    "base_url": os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
    "model": os.getenv("REMOTE_MODEL", "MiniMax-M2.7"),
    "max_tokens": 8192,
    "thinking_budget": 2048,
}

# MiniMax 远程模型列表配置
REMOTE_CONFIGS = [
    {
        "api_key": os.getenv("MINIMAX_API_KEY", "sk-cp-mC5XmJWKwt_FFLCUAkv6CmE2n6PuS9_-Q3vrP1rLCEamVW9di8Y4P1MD3qts_aVokA2dxmT2YaeOUpEQONtwSbrURIwrPc8ldP8f8dfBkE"),  # 建议使用环境变量
        "base_url": os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
        "model": os.getenv("REMOTE_MODEL", "MiniMax-M2.7"),
        "max_tokens": 8192,
        "thinking_budget": 2048,
    },{
        "api_key": os.getenv("MINIMAX_API_KEY", "sk-cp-yk7KuLSWe6Yfa-J68APc78gsDjzB1fOkrgn4aqtsovZaa2a17eJqLozZ67NEX_R5bGiV8YFmFlYb-9DoHUFher8QiQ9cCvGXyAUc9S8v4"),  # 建议使用环境变量
        "base_url": os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/anthropic"),
        "model": os.getenv("REMOTE_MODEL", "MiniMax-M2.7"),
        "max_tokens": 8192,
        "thinking_budget": 2048,
    }
]

# 本地 Ollama 默认配置
LOCAL_CONFIGS = [
    {
        "model": "deepseek-r1:14b",
        "temperature": 0.0,
        "max_tokens": 2048,
    },{
        "model": "deepseek-r1:14b",
        "temperature": 0.0,
        "max_tokens": 2048,
    }
]

# 本地 Ollama 默认配置
LOCAL_CONFIG = [
    {
        "model": "deepseek-r1:14b",
        "temperature": 0.0,
        "max_tokens": 2048,
    }
]