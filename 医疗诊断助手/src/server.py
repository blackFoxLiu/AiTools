# server.py
import sys
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 使用绝对路径导入项目模块
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

# 将项目根目录加入 sys.path，以便导入 src 下的模块
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 现在可以正确导入 src 下的模块
from src.tools.session_store import SessionStore
from src.main import MedicalChat

# ------------------------------------------------------------
# 2. 创建 FastAPI 应用
# ------------------------------------------------------------
app = FastAPI(title="智医助手API", version="1.0.0")

# 跨域配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------------------------------------
# 3. 全局会话存储（使用项目 data 目录下的 chat_sessions）
# ------------------------------------------------------------
data_dir = project_root / "data" / "chat_sessions"
data_dir.mkdir(parents=True, exist_ok=True)          # 确保目录存在
session_store = SessionStore(storage_dir=str(data_dir))

# ------------------------------------------------------------
# 4. Pydantic 模型
# ------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    answer: str

class SessionCreateResponse(BaseModel):
    session_id: str

# ------------------------------------------------------------
# 5. API 路由
# ------------------------------------------------------------
@app.post("/api/sessions", response_model=SessionCreateResponse)
async def create_session():
    session_id = session_store.create_session()
    return SessionCreateResponse(session_id=session_id)

@app.get("/api/sessions")
async def list_sessions():
    return session_store.list_sessions()

@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    session_data = session_store.load_session(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail="会话不存在")
    messages = []
    for msg in session_data.get("messages", []):
        messages.append({
            "role": msg["role"],
            "content": msg["content"],
            "timestamp": msg.get("timestamp", datetime.now().isoformat())
        })
    return messages

@app.post("/api/sessions/{session_id}/chat", response_model=ChatResponse)
async def chat(session_id: str, req: ChatRequest):
    session_data = session_store.load_session(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail=f"会话 {session_id} 不存在")

    medical_assistant = MedicalChat(session_store=session_store)
    medical_assistant.session_id = session_id

    try:
        answer = medical_assistant.process_message(req.message)
        return ChatResponse(answer=answer)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"对话处理失败: {str(e)}")

# ------------------------------------------------------------
# 6. 静态文件挂载（使用项目 resources/static 目录，绝对路径）
# ------------------------------------------------------------
static_dir = project_root / "resources" / "static"
if static_dir.exists() and static_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="static")
else:
    print(f"错误: 静态目录 {static_dir} 不存在，请创建并将 index.html 放入其中")

# ------------------------------------------------------------
# 7. 启动入口
# C:\Users\13187\anaconda3\envs\langgraph_env\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8808 --reload
# ------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    # 若仍遇到权限错误，请以管理员身份运行，或换用 8000 端口
    uvicorn.run(app, host="0.0.0.0", port=8808)