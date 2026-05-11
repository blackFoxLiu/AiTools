import json
import logging
import os
import uuid
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)


class SessionStore:
    """
    会话持久化存储管理器（基于 JSON 文件）
    每个意图（intention）对应一个发现数据（discovery_data）和一个路由选择（chat_intention_router）
    """

    def __init__(self, storage_dir: str = "./chat_sessions"):
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

    def _extract_title_from_intentions(self, intentions: List[Dict[str, Any]]) -> str:
        """
        从意图列表中提取标题，优先使用最后一个意图的 main_intention
        intentions 是包含 intention 字段的字典列表
        """
        if not intentions:
            return "未命名会话"

        latest = intentions[-1]
        intention_obj = latest.get("intention", {})
        # 兼容旧格式或新格式
        if isinstance(intention_obj, dict):
            main_intention = intention_obj.get("main_intention", "")
        else:
            # 如果是旧格式的 HumanMessage，尝试解析
            try:
                intent_data = json.loads(intention_obj.content)
                main_intention = intent_data.get("main_intention", "")
            except:
                main_intention = ""
        if len(main_intention) > 50:
            main_intention = main_intention[:47] + "..."
        return main_intention if main_intention else "未命名会话"

    def create_session(self, initial_intentions: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        创建新会话
        initial_intentions: 可选的初始意图列表，每个元素包含 intention, discovery_data, chat_intention_router
        """
        session_id = self._generate_session_id()
        title = "未命名会话"
        if initial_intentions:
            title = self._extract_title_from_intentions(initial_intentions)

        session_data = {
            "session_id": session_id,
            "title": title,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "messages": [],
            "intentions": []   # 每个元素: {"intention": {...}, "discovery_data": {...}, "chat_intention_router": str, "timestamp": str}
        }
        if initial_intentions:
            session_data["intentions"] = initial_intentions
        self._save_session(session_id, session_data)
        logger.info(f"创建新会话: {session_id} -> {title}")
        return session_id

    def _save_session(self, session_id: str, data: Dict[str, Any]):
        """将会话数据写入文件"""
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
        """根据当前会话的所有意图重新计算标题并更新"""
        session_data = self.load_session(session_id)
        if not session_data:
            logger.error(f"会话 {session_id} 不存在，无法更新标题")
            return
        new_title = self._extract_title_from_intentions(session_data["intentions"])
        if new_title != session_data["title"]:
            session_data["title"] = new_title
            session_data["updated_at"] = datetime.now().isoformat()
            self._save_session(session_id, session_data)
            logger.info(f"更新会话标题: {session_id} -> {new_title}")

    def add_message(self, session_id: str, role: str, content: str):
        """
        添加一条对话消息。
        role: 'user' 或 'assistant'
        """
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
                            intention_message: HumanMessage,
                            discovery_data: Dict[str, Any],
                            router_value: str):
        """
        添加一轮完整的交互（意图 + 发现数据 + 路由选择）
        :param intention_message: 意图提取出的 HumanMessage，其 content 应为 JSON 字符串
        :param discovery_data: 该轮对应的探寻数据（字典）
        :param router_value: 该轮的路由标识，如 "symptoms_inquiry" 或 "medication_inquiry"
        """
        session_data = self.load_session(session_id)
        if not session_data:
            raise ValueError(f"会话 {session_id} 不存在")

        # 解析意图内容
        try:
            intent_data = json.loads(intention_message.content)
            main_intention = intent_data.get("main_intention", "")
            sub_operate = intent_data.get("sub_operate", [])
        except json.JSONDecodeError:
            main_intention = ""
            sub_operate = []
            logger.warning("意图内容不是合法 JSON，原样保存 raw_content")
            intent_data = {"raw_content": intention_message.content}

        round_data = {
            "intention": {
                "main_intention": main_intention,
                "sub_operate": sub_operate,
                "raw_content": intention_message.content
            },
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
        为兼容原有 MedicalChatState 结构（顶层有 discovery_data 和 chat_intention_router），
        这里取最后一轮的 discovery_data 和 router 填充到顶层。
        同时返回完整的 intentions 列表（HumanMessage 对象列表）。
        """
        session_data = self.load_session(session_id)
        if not session_data:
            return None

        # 恢复消息
        messages = []
        for msg in session_data["messages"]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                from langchain_core.messages import AIMessage
                messages.append(AIMessage(content=msg["content"]))

        # 恢复意图轮次数据，并生成 intentions 列表（仅 HumanMessage）
        intentions = []
        last_discovery = {}
        last_router = ""
        for round_data in session_data.get("intentions", []):
            # 将意图转为 HumanMessage
            intent_content = round_data["intention"].get("raw_content", "")
            if not intent_content:
                # 兼容旧格式：重建 JSON
                intent_content = json.dumps({
                    "main_intention": round_data["intention"].get("main_intention", ""),
                    "sub_operate": round_data["intention"].get("sub_operate", [])
                })
            intentions.append(HumanMessage(content=intent_content))
            # 记录最后一轮的数据
            last_discovery = round_data.get("discovery_data", {})
            last_router = round_data.get("chat_intention_router", "")

        return {
            "messages": messages,
            "intentions": intentions,          # 列表中的 HumanMessage 顺序与存储顺序一致
            "discovery_data": last_discovery,  # 兼容：当前工作流所需的最新发现数据
            "solution": "",
            "chat_intention_router": last_router,  # 兼容：当前工作流所需的最新路由
            "session_id": session_id
        }

    def list_sessions(self) -> List[Dict[str, str]]:
        sessions = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(self.storage_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    sessions.append({
                        "session_id": data["session_id"],
                        "title": data["title"],
                        "updated_at": data["updated_at"]
                    })
        return sorted(sessions, key=lambda x: x["updated_at"], reverse=True)


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SessionStore 调用示例（__main__.py）

演示 SessionStore 的完整使用流程：
1. 创建新会话
2. 添加用户和助手消息
3. 添加意图并自动更新标题
4. 列出所有会话
5. 恢复会话状态
6. 手动更新标题
"""

import json
import logging
from langchain_core.messages import HumanMessage

# 配置日志（便于观察输出）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def demo_session_store():
    """展示 SessionStore 的核心功能"""
    print("=" * 60)
    print("SessionStore 功能演示")
    print("=" * 60)

    # 1. 初始化存储管理器（使用临时目录，便于清理）
    store = SessionStore(storage_dir="./demo_sessions")
    print("\n[1] 初始化 SessionStore，存储目录：./demo_sessions")

    # 2. 创建一个新会话（无初始意图）
    session_id = store.create_session()
    print(f"\n[2] 创建新会话: {session_id}")
    print(f"    初始标题: {store.load_session(session_id)['title']}")

    # 3. 模拟用户与助手的对话，存储消息
    print("\n[3] 添加对话消息...")
    store.add_message(session_id, "user", "今天早上头疼，有些流鼻涕。无咳嗽")
    store.add_message(session_id, "assistant", "提问回答信息：1. 鼻源性头痛 ...")
    store.add_message(session_id, "user", "无鼻塞、无畏寒、无发热、无咽痛")
    store.add_message(session_id, "assistant", "根据您的描述，可能是风寒头痛或鼻源性头痛...")
    print("    已添加 4 条消息")

    # 4. 添加意图（通常在每次意图提取后调用）
    print("\n[4] 添加意图...")
    intention_msg_1 = HumanMessage(
        content='{"main_intention": "了解头痛伴流鼻涕的可能病因、危险信号及处理建议", "sub_operate": ["头痛伴流鼻涕可能是什么疾病", "是否需要就医", "如何进行一般护理"]}'
    )
    store.add_intention(session_id, intention_msg_1)
    print("    添加第一个意图，标题应自动更新")

    # 查看更新后的标题
    session_data = store.load_session(session_id)
    print(f"    当前标题: {session_data['title']}")

    # 5. 再次添加另一个意图（模拟用户进一步追问）
    intention_msg_2 = HumanMessage(
        content='{"main_intention": "明确头痛伴流鼻涕的当前病因、是否需要用药", "sub_operate": ["无鼻塞发热的可能疾病", "用药建议", "护理措施"]}'
    )
    store.add_intention(session_id, intention_msg_2)
    session_data = store.load_session(session_id)
    print(f"\n[5] 添加第二个意图后，标题再次更新: {session_data['title']}")

    # 6. 列出所有会话摘要
    print("\n[6] 列出所有会话摘要:")
    for sess in store.list_sessions():
        print(f"    ID: {sess['session_id']}")
        print(f"    标题: {sess['title']}")
        print(f"    更新时间: {sess['updated_at']}\n")

    # 7. 恢复状态（用于继续对话）
    print("[7] 从存储恢复状态（restore_state）:")
    restored = store.restore_state(session_id)
    if restored:
        print(f"    恢复成功，包含 {len(restored['messages'])} 条消息，{len(restored['intentions'])} 个意图")
        print(f"    最后一条消息: {restored['messages'][-1].content[:80]}...")

    # 8. 手动更新标题（通常不需要，由 add_intention 自动处理）
    print("\n[8] 手动更新标题示例（一般无需手动调用）:")
    # 模拟一个更短的意图
    short_intention = HumanMessage(content='{"main_intention": "头痛咨询", "sub_operate": []}')
    store.update_title_from_intentions(session_id, [short_intention])
    session_data = store.load_session(session_id)
    print(f"    手动更新后的标题: {session_data['title']}")

    # 9. 加载不存在的会话（演示容错）
    print("\n[9] 尝试加载不存在的会话:")
    missing = store.load_session("non_existent_id_123")
    print(f"    结果: {missing}")

    print("\n" + "=" * 60)
    print("演示完成。生成的会话文件保存在 ./demo_sessions 目录下，可手动删除。")
    print("=" * 60)


if __name__ == "__main__":
    demo_session_store()