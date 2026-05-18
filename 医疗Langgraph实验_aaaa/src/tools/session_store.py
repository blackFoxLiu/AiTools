"""
模型会话存储机制设计，用于存储医疗助手会话过程中产生的会话信息、意图信息、知识探寻信息
统一使用纯字典格式存储意图，避免 LangChain 消息对象导致的序列化问题
"""
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage

# 导入独立子图构建函数
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# 将项目根目录添加到 Python 的模块搜索路径中
if root_path not in sys.path:
    sys.path.append(root_path)
try:
    from config.default_config import config as default_config
except ImportError:
    raise RuntimeError(f"导入模块失败")

logger = logging.getLogger(__name__)

class SessionStore:
    """
    会话持久化存储管理器（基于 JSON 文件）
    每个意图轮次（intention round）包含：
        - intention: {"main_intention": str, "sub_operate": List[str]}
        - discovery_data: Dict[str, Any]
        - chat_intention_router: str
        - timestamp: str
    """

    def __init__(self, storage_dir: str = default_config.MEDICAL_CHAT_SESSION):
        self.storage_dir = storage_dir
        os.makedirs(storage_dir, exist_ok=True)

    def _get_session_path(self, session_id: str) -> str:
        """获取会话文件的完整路径"""
        return os.path.join(self.storage_dir, f"{session_id}.json")

    def _generate_session_id(self) -> str:
        """生成唯一会话 ID（时间戳+随机串）"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rand_suffix = str(uuid.uuid4())[:8]
        return f"{timestamp}_{rand_suffix}"

    def _extract_title_from_intentions(self, intentions_rounds: List[Dict[str, Any]]) -> str:
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
            "intentions": []   # 每个元素: {"intention": {...}, "discovery_data": {...}, "chat_intention_router": str, "timestamp": str}
        }
        self._save_session(session_id, session_data)
        logger.info(f"创建新会话: {session_id}")
        return session_id

    def _save_session(self, session_id: str, data: Dict[str, Any]):
        """将会话数据写入文件（所有数据均为可序列化类型）"""
        file_path = self._get_session_path(session_id)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """加载指定会话的完整数据，若不存在则返回 None"""
        file_path = self._get_session_path(session_id)
        if not os.path.exists(file_path):
            logger.warning(f"会话文件不存在: {session_id}")
            return None
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def update_title_from_intentions(self, session_id: str):
        """根据当前会话的所有意图轮次重新计算标题并更新"""
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

    def add_message(self, session_id: str, role: str, content: str):
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
                            router_value: str):
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
        for msg in session_data["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # 恢复意图轮次，提取每个轮次的 intention 字典
        intentions = []          # 纯字典列表
        last_discovery = {}
        last_router = ""
        for round_data in session_data.get("intentions", []):
            intentions.append(round_data["intention"])
            last_discovery = round_data.get("discovery_data", {})
            last_router = round_data.get("chat_intention_router", "")

        return {
            "messages": messages,
            "intentions": intentions,           # 纯字典列表，与工作流状态一致
            "discovery_data": last_discovery,
            "solution": "",
            "chat_intention_router": last_router,
            "session_id": session_id
        }

    def list_sessions(self) -> List[Dict[str, str]]:
        sessions = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.storage_dir, filename)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        sessions.append({
                            "session_id": data["session_id"],
                            "title": data["title"],
                            "updated_at": data["updated_at"]
                        })
                except (json.JSONDecodeError, KeyError, OSError) as e:
                    logger.warning(f"跳过损坏的会话文件 {filename}: {e}")
                    continue
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)


# 示例运行（用于测试）
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    print("=" * 60)
    print("SessionStore 功能演示（纯字典意图存储）")
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
    print(f"恢复的消息数: {len(restored['messages'])}")
    print(f"恢复的意图字典: {restored['intentions'][0]}")
    print(f"恢复的 discovery_data: {restored['discovery_data']}")