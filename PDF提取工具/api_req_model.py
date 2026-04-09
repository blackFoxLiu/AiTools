import logging
import time
from functools import wraps
from typing import Optional, Tuple, Dict, Any, List
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

    def __init__(self, config: Dict[str, Any], name: str = "Remote"):
        self.config = config
        self.name = name

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
            logger.info(f"{self.name} 调用成功")
            return reasoning, final_answer
        except Exception as e:
            logger.error(f"{self.name} 调用失败: {e}")
            raise


class LocalModelInvoker(ModelInvoker):
    """本地 Ollama 模型调用器（已开启深度思考）"""

    def __init__(self, client: ChatOllama, config: Dict[str, Any]):
        self.client = client
        self.config = config

    def invoke(self, system_prompt: str, user_content: str, **kwargs) -> Tuple[str, str]:
        try:
            messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
            bound_client = self.client.bind(
                model=self.config["model"],
                options={
                    "num_predict": self.config["max_tokens"],
                    "temperature": self.config["temperature"],
                    "thinking": True
                }
            )
            response = bound_client.invoke(messages)
            reasoning = response.additional_kwargs.get("reasoning_content", "")
            if not reasoning:
                reasoning = response.additional_kwargs.get("thinking", "")
            final_answer = response.content
            logger.info("本地模型调用成功")
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


# ========================= 兼容原接口的入口函数（支持多远端，无需修改参数） =========================
@timed
def call_model(
        client: Optional[ChatOllama] = None,
        system_prompt: str = "",
        user_content: str = "",
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        prefer_remote: bool = True,
        remote_config: Optional[Dict[str, Any]] = None,   # 保留此参数以兼容旧代码
) -> Tuple[str, str]:
    """
    统一模型调用入口。
    支持多个远端配置（通过 model_config.REMOTE_CONFIGS 列表定义）+ 本地回退的责任链模式。
    调用顺序：remote1 -> remote2 -> ... -> local。
    若显式传入 remote_config，则仅使用该配置作为唯一远端（兼容旧行为）。
    否则自动从 model_config 中读取远端配置列表（支持单/多远端）。
    """
    # 1. 获取远端配置列表（优先使用显式传入的 remote_config，其次尝试全局列表，最后回退到旧的单字典）
    remote_configs: List[Dict[str, Any]] = []
    if remote_config is not None:
        # 显式传入单个配置，保持旧行为（仅此一个远端）
        remote_configs = [remote_config]
    else:
        # 尝试从 model_config 导入 REMOTE_CONFIGS（列表）
        try:
            from model_config import REMOTE_CONFIGS
            if isinstance(REMOTE_CONFIGS, list) and REMOTE_CONFIGS:
                remote_configs = REMOTE_CONFIGS
        except ImportError:
            pass
        # 如果没有 REMOTE_CONFIGS，则尝试使用原有的 REMOTE_CONFIG（单个字典）
        if not remote_configs and isinstance(REMOTE_CONFIG, dict) and REMOTE_CONFIG.get("api_key"):
            remote_configs = [REMOTE_CONFIG]

    # 2. 准备本地配置
    local_model = model or LOCAL_CONFIG["model"]
    local_temp = temperature if temperature is not None else LOCAL_CONFIG["temperature"]
    local_max_tokens = max_tokens if max_tokens is not None else LOCAL_CONFIG["max_tokens"]
    local_config = {
        "model": local_model,
        "temperature": local_temp,
        "max_tokens": local_max_tokens,
    }

    # 3. 构建责任链（从最末端开始）
    # 本地调用器作为最终回退
    local_invoker = None
    if client is not None:
        local_invoker = LocalModelInvoker(client=client, config=local_config)

    # 远程调用器列表（只保留有效的）
    remote_invokers = []
    if prefer_remote:
        for idx, cfg in enumerate(remote_configs):
            if cfg and cfg.get("api_key"):
                remote_invokers.append(RemoteModelInvoker(config=cfg, name=f"Remote-{idx+1}"))

    # 构建链: 第一个远程 -> 第二个远程 -> ... -> 本地
    chain = local_invoker
    for inv in reversed(remote_invokers):
        chain = FallbackModelInvoker(primary=inv, fallback=chain)

    if chain is None:
        raise ValueError("没有可用的模型调用器：请提供 client（本地）或配置有效的远端")

    return chain.invoke(system_prompt, user_content)


# ========================= 使用示例 =========================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    system_info = "你是游戏助手"
    user_info = "请做出决策"

    # 示例1：仅使用本地（prefer_remote=False）
    local_client = ChatOllama(model="deepseek-r1:14b")
    reasoning, answer = call_model(
        client=local_client,
        system_prompt=system_info,
        user_content=user_info,
        prefer_remote=False
    )
    print("仅本地结果:", answer)

    # 示例2：自动使用 model_config 中定义的所有远端（若 REMOTE_CONFIGS 包含两个配置）+ 本地回退
    # 无需修改调用代码
    reasoning, answer = call_model(
        client=local_client,
        system_prompt=system_info,
        user_content=user_info,
        prefer_remote=True
    )
    print("多远端+本地结果:", answer)

    # 示例3：兼容旧代码，显式传入单个 remote_config
    # reasoning, answer = call_model(
    #     client=local_client,
    #     system_prompt=system_info,
    #     user_content=user_info,
    #     prefer_remote=True,
    #     remote_config={"api_key": "xxx", ...}   # 此时只使用这一个远端
    # )