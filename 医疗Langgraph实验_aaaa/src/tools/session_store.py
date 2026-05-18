"""
模型会话存储机制设计，用于存储医疗助手会话过程中产生的会话信息、意图信息、知识探寻信息
统一使用纯字典格式存储意图，避免 LangChain 消息对象导致的序列化问题
"""

import json
import logging
import os
import sys
import uuid
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Union

from langchain_core.messages import HumanMessage, AIMessage

# 导入独立子图构建函数
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if root_path not in sys.path:
    sys.path.append(root_path)

try:
    from config.default_config import config as default_config
except ImportError:
    raise RuntimeError("导入模块失败")

logger = logging.getLogger(__name__)

class SessionStore:
    """
    会话持久化存储管理器（基于 JSON 文件）
    每个意图轮次（intention round）包含：
        - intention: {"main_intention": str, "sub_operate": List[str]}
        - discovery_data: Dict[str, Any]
        - chat_intention_router: str
        - timestamp: str

    线程安全（使用 RLock 保护文件读写）
    """

    def __init__(self, storage_dir: Optional[Union[str, Path]] = None):
        """
        :param storage_dir: 会话存储目录，默认从配置读取，若无则使用 "./sessions"
        """
        if storage_dir is None:
            storage_dir = getattr(default_config, "MEDICAL_CHAT_SESSION", "./sessions")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()  # 文件操作锁

    def _get_session_path(self, session_id: str) -> Path:
        """获取会话文件的完整路径（Path 对象）"""
        return self.storage_dir / f"{session_id}.json"

    def _generate_session_id(self) -> str:
        """生成唯一会话 ID（时间戳+随机串）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = str(uuid.uuid4())[:8]
        return f"{timestamp}_{rand_suffix}"

    @staticmethod
    def _make_serializable(obj: Any) -> Any:
        """
        递归将对象转换为 JSON 可序列化的形式。
        支持：Pydantic v1/v2、dataclass、常见容器、以及任意具有 model_dump/dict 方法的对象。
        """
        # Pydantic v2
        if hasattr(obj, "model_dump") and callable(obj.model_dump):
            return obj.model_dump()
        # Pydantic v1
        if hasattr(obj, "dict") and callable(obj.dict):
            return obj.dict()
        # dataclass
        if hasattr(obj, "__dataclass_fields__"):
            return {k: SessionStore._make_serializable(v) for k, v in obj.__dict__.items()}
        # 自定义对象：尝试转为 __dict__
        if hasattr(obj, "__dict__"):
            return {k: SessionStore._make_serializable(v) for k, v in obj.__dict__.items()}
        # 列表 / 元组
        if isinstance(obj, (list, tuple)):
            return [SessionStore._make_serializable(item) for item in obj]
        # 字典
        if isinstance(obj, dict):
            return {k: SessionStore._make_serializable(v) for k, v in obj.items()}
        # 基础类型直接返回
        return obj

    def _save_session(self, session_id: str, data: Dict[str, Any]) -> None:
        """
        原子写入会话数据（使用临时文件 + 替换，线程安全）
        """
        # 清洗数据，移除不可序列化部分
        cleaned_data = self._make_serializable(data)

        file_path = self._get_session_path(session_id)
        temp_path = file_path.with_suffix(".tmp")

        with self._lock:
            # 写入临时文件
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
            # 原子替换
            shutil.move(str(temp_path), str(file_path))

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        加载指定会话的完整数据，若文件损坏或不存在则返回 None
        """
        file_path = self._get_session_path(session_id)
        if not file_path.exists():
            return None

        with self._lock:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                # 备份损坏的文件，便于人工排查
                backup_path = file_path.with_suffix(".corrupted")
                shutil.copy(file_path, backup_path)
                logger.error(
                    f"会话 {session_id} JSON 解析失败，已备份至 {backup_path}，错误: {e}"
                )
                return None
            except OSError as e:
                logger.error(f"读取会话文件 {session_id} 失败: {e}")
                return None

    @staticmethod
    def _extract_title_from_intentions(intentions_rounds: List[Dict[str, Any]]) -> str:
        """从意图轮次列表中提取标题，使用最后一轮的 main_intention"""
        if not intentions_rounds:
            return "未命名会话"
        latest = intentions_rounds[-1]
        intention_dict = latest.get("intention", {})
        main_intention = intention_dict.get("main_intention", "")
        if len(main_intention) > 50:
            main_intention = main_intention[:47] + "..."
        return main_intention if main_intention else "未命名会话"

    def create_session(self) -> str:
        """创建新会话，返回 session_id"""
        session_id = self._generate_session_id()
        session_data = {
            "session_id": session_id,
            "title": "未命名会话",
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "intentions": [],  # 每个元素: {"intention": {...}, "discovery_data": {...}, "chat_intention_router": str, "timestamp": str}
        }
        self._save_session(session_id, session_data)
        logger.info(f"创建新会话: {session_id}")
        return session_id

    def update_title_from_intentions(self, session_id: str) -> None:
        """
        根据当前会话的所有意图轮次重新计算标题并更新
        """
        session_data = self.load_session(session_id)
        if not session_data:
            logger.error(f"会话 {session_id} 不存在，无法更新标题")
            return
        new_title = self._extract_title_from_intentions(session_data.get("intentions", []))
        if new_title != session_data.get("title"):
            session_data["title"] = new_title
            session_data["updated_at"] = datetime.now().isoformat()
            self._save_session(session_id, session_data)
            logger.info(f"更新会话标题: {session_id} -> {new_title}")

    def add_message(self, session_id: str, role: str, content: str) -> None:
        """添加一条对话消息（user 或 assistant）"""
        session_data = self.load_session(session_id)
        if not session_data:
            raise ValueError(f"会话 {session_id} 不存在")
        session_data["messages"].append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        session_data["updated_at"] = datetime.now().isoformat()
        self._save_session(session_id, session_data)

    def add_intention_round(self, session_id: str,
                            intention_dict: Dict[str, Any],
                            discovery_data: Dict[str, Any],
                            router_value: str) -> None:
        """
        添加一轮完整的交互（意图字典 + 发现数据 + 路由选择）
        :param intention_dict: 格式 {"main_intention": str, "sub_operate": List[str]}
        :param discovery_data: 该轮对应的探寻数据（字典）
        :param router_value: 该轮的路由标识，如 "symptoms_inquiry" 或 "medication_inquiry"
        """
        session_data = self.load_session(session_id)
        if not session_data:
            raise ValueError(f"会话 {session_id} 不存在")

        round_data = {
            "intention": intention_dict,
            "discovery_data": discovery_data,
            "chat_intention_router": router_value,
            "timestamp": datetime.now().isoformat()
        }
        session_data["intentions"].append(round_data)
        session_data["updated_at"] = datetime.now().isoformat()
        self._save_session(session_id, session_data)

        # 更新标题
        self.update_title_from_intentions(session_id)

    def get_last_round(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        获取最新一轮的意图、发现数据、路由
        """
        session_data = self.load_session(session_id)
        if not session_data or not session_data.get("intentions"):
            return None
        return session_data["intentions"][-1]

    def restore_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        从持久化存储恢复 MedicalChatState 所需的字段。
        返回格式：
            messages: List[BaseMessage] (HumanMessage/AIMessage)
            intentions: List[Dict[str, Any]]   # 纯字典列表，每个元素包含 main_intention, sub_operate
            discovery_data: Dict[str, Any]     # 最后一轮的发现数据
            chat_intention_router: str         # 最后一轮的路由
            session_id: str
            solution: "" (占位)
        """
        session_data = self.load_session(session_id)
        if not session_data:
            return None

        # 恢复消息（转换为 LangChain 消息对象）
        messages = []
        for msg in session_data.get("messages", []):
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            else:
                messages.append(AIMessage(content=content))

        # 恢复意图轮次，提取每个轮次的 intention 字典
        intentions = []  # 纯字典列表
        last_discovery = {}
        last_router = ""
        for round_data in session_data.get("intentions", []):
            intentions.append(round_data.get("intention", {}))
            last_discovery = round_data.get("discovery_data", {})
            last_router = round_data.get("chat_intention_router", "")

        return {
            "messages": messages,
            "intentions": intentions,
            "discovery_data": last_discovery,
            "solution": "",
            "chat_intention_router": last_router,
            "session_id": session_id
        }

    def list_sessions(self) -> List[Dict[str, str]]:
        """
        列出所有会话元信息（session_id, title, updated_at）
        """
        sessions = []
        for file_path in self.storage_dir.glob("*.json"):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data["session_id"],
                        "title": data.get("title", "未命名"),
                        "updated_at": data.get("updated_at", "")
                    })
            except (json.JSONDecodeError, KeyError, OSError) as e:
                logger.warning(f"跳过损坏的会话文件 {file_path.name}: {e}")
                continue
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)

    def clean_old_sessions(self, days: int = 30) -> int:
        """
        删除超过指定天数未更新的会话文件
        :param days: 保留天数，默认30天
        :return: 删除的文件数量
        """
        now = datetime.now().timestamp()
        deleted = 0
        for file_path in self.storage_dir.glob("*.json"):
            if now - file_path.stat().st_mtime > days * 86400:
                try:
                    file_path.unlink()
                    deleted += 1
                    logger.info(f"已删除过期会话: {file_path.name}")
                except OSError as e:
                    logger.error(f"删除会话文件 {file_path.name} 失败: {e}")
        return deleted


# 示例运行（用于测试）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("=" * 60)
    print("SessionStore 功能演示（纯字典意图存储 + 优化版）")
    print("=" * 60)

    store = SessionStore()
    session_id = store.create_session()
    print(f"创建会话: {session_id}")

    # 添加消息
    store.add_message(session_id, "user", "今天早上头疼，有些流鼻涕。")
    store.add_message(session_id, "assistant", "请问是否有鼻塞、发热？")

    # 添加意图轮次
    intention_dict = {
        "main_intention": "了解头痛伴流鼻涕的可能病因",
        "sub_operate": ["头痛伴流鼻涕可能是什么疾病", "是否需要就医", "如何进行一般护理"]
    }
    discovery = {"症状": "头痛、流鼻涕", "排除": "无发热"}
    store.add_intention_round(session_id, intention_dict, discovery, "symptoms_inquiry")

    # 恢复状态
    restored = store.restore_state(session_id)
    if restored:
        print(f"恢复的消息数: {len(restored['messages'])}")
        print(f"恢复的意图字典: {restored['intentions'][0]}")
        print(f"恢复的 discovery_data: {restored['discovery_data']}")
    else:
        print("恢复失败")