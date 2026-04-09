import logging
import time
from functools import wraps
from typing import Optional, Tuple, Dict, Any
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage


# 导入远程配置
from model_config import REMOTE_CONFIG, LOCAL_CONFIG

# 配置日志
logger = logging.getLogger(__name__)


# ========================= 设计模式核心结构 =========================

class ModelInvoker:
    """模型调用器抽象基类（策略接口）"""

    def invoke(self, system_prompt: str, user_content: str, **kwargs) -> Tuple[str, str]:
        raise NotImplementedError


class RemoteModelInvoker(ModelInvoker):
    """远程模型调用器（MiniMax M2.7）"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def invoke(self, system_prompt: str, user_content: str, **kwargs) -> Tuple[str, str]:
        try:
            from anthropic import Anthropic
            client = Anthropic(api_key=self.config["api_key"], base_url=self.config["base_url"])

            response = client.messages.create(
                model=self.config["model"],
                max_tokens=self.config["max_tokens"],
                system=system_prompt,
                messages=[{"role": "user", "content": [{"type": "text", "text": user_content}]}],
                thinking={"type": "enabled", "budget_tokens": self.config["thinking_budget"]}
            )

            reasoning = ""
            final_answer = ""
            for block in response.content:
                if block.type == "thinking":
                    reasoning = block.thinking
                elif block.type == "text":
                    final_answer = block.text
            return reasoning, final_answer
        except Exception as e:
            logger.error(f"远程模型调用失败: {e}")
            raise


class LocalModelInvoker(ModelInvoker):
    """本地 Ollama 模型调用器（已开启深度思考）"""

    def __init__(self, client: ChatOllama, config: Dict[str, Any]):
        self.client = client
        self.config = config

    def invoke(self, system_prompt: str, user_content: str, **kwargs) -> Tuple[str, str]:
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
            # 🔥 显式开启深度思考：添加 thinking=True 参数
            bound_client = self.client.bind(
                model=self.config["model"],
                options = {
                    "num_predict": self.config["max_tokens"],
                    "temperature": self.config["temperature"],
                    "thinking": True   # 启用思考过程（对 deepseek-r1 等模型有效）
                }
            )
            response = bound_client.invoke(messages)
            # 尝试从 additional_kwargs 中获取推理内容（不同模型可能字段不同）
            reasoning = response.additional_kwargs.get("reasoning_content", "")
            # 如果没有 reasoning_content，尝试其他常见字段
            if not reasoning:
                reasoning = response.additional_kwargs.get("thinking", "")
            final_answer = response.content
            return reasoning, final_answer
        except Exception as e:
            logger.error(f"本地模型调用失败: {e}")
            raise


class FallbackModelInvoker(ModelInvoker):
    """带降级策略的调用器（责任链模式）"""

    def __init__(self, primary: ModelInvoker, fallback: Optional[ModelInvoker] = None):
        self.primary = primary
        self.fallback = fallback

    def invoke(self, system_prompt: str, user_content: str, **kwargs) -> Tuple[str, str]:
        try:
            return self.primary.invoke(system_prompt, user_content, **kwargs)
        except Exception as e:
            if self.fallback:
                logger.warning(f"主调用器失败，切换到备用调用器: {e}")
                return self.fallback.invoke(system_prompt, user_content, **kwargs)
            else:
                logger.error(f"无可用备用调用器，调用失败: {e}")
                return "", ""

def timed(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} 耗时 {elapsed:.3f} 秒")
        return result
    return wrapper


# ========================= 兼容原接口的入口函数 =========================
@timed
def call_model(
        client: Optional[ChatOllama] = None,
        system_prompt: str = "",
        user_content: str = "",
        model: str = None,  # 本地模型名，若不指定则使用 LOCAL_CONFIG
        temperature: float = None,  # 若不指定则使用 LOCAL_CONFIG
        max_tokens: int = None,  # 若不指定则使用 LOCAL_CONFIG
        prefer_remote: bool = True,
        remote_config: Optional[Dict[str, Any]] = None,  # 若不指定则使用全局 REMOTE_CONFIG
) -> Tuple[str, str]:
    """
    统一模型调用入口（兼容原有签名，增加远程优先和回退能力）
    """
    # 1. 决定远程配置
    effective_remote_config = remote_config if remote_config is not None else REMOTE_CONFIG

    # 2. 决定本地配置（合并参数和默认值）
    local_model = model or LOCAL_CONFIG["model"]
    local_temp = temperature if temperature is not None else LOCAL_CONFIG["temperature"]
    local_max_tokens = max_tokens if max_tokens is not None else LOCAL_CONFIG["max_tokens"]
    local_config = {
        "model": local_model,
        "temperature": local_temp,
        "max_tokens": local_max_tokens,
    }

    # 3. 构建调用器链
    remote_invoker = None
    if prefer_remote and effective_remote_config.get("api_key"):
        remote_invoker = RemoteModelInvoker(config=effective_remote_config)

    local_invoker = None
    if client is not None:
        local_invoker = LocalModelInvoker(client=client, config=local_config)

    if prefer_remote and remote_invoker:
        if local_invoker:
            invoker = FallbackModelInvoker(primary=remote_invoker, fallback=local_invoker)
        else:
            invoker = remote_invoker  # 无回退，远程失败即返回空
    elif local_invoker:
        invoker = local_invoker
    else:
        raise ValueError("没有可用的模型调用器：请提供 client（本地）或启用远程配置")

    return invoker.invoke(system_prompt, user_content)


# ========================= 使用示例 =========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system_info = "你是游戏助手"
    user_info = "请做出决策"

    # 示例1：仅使用远程（无本地回退），配置从 model_config.py 自动加载
    # reasoning, answer = call_model(
    #     system_prompt="你是游戏助手",  # 实际使用时替换为您的 system_info
    #     user_content="请做出决策",  # 实际使用时替换为您的 user_info
    #     prefer_remote=True,
    #     # remote_config 不传，则自动使用 model_config.REMOTE_CONFIG
    # )
    # print(f"远程结果:\n思考: {reasoning}\n答案: {answer}")

    # 示例2：远程优先 + 本地回退（需要提供 client）
    # from langchain_ollama import ChatOllama
    # local_client = ChatOllama(model="deepseek-r1:14b")
    # reasoning, answer = call_model(
    #     client=local_client,
    #     system_prompt=system_info,
    #     user_content=user_info,
    #     prefer_remote=True
    # )
    # print(answer)

    # 示例3：仅使用本地（完全兼容原接口）
    local_client = ChatOllama(model="deepseek-r1:14b")
    reasoning, answer = call_model(
        client=local_client,
        system_prompt=system_info,
        user_content=user_info,
        prefer_remote=False
    )