from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import asyncio
import uvicorn
from tools.client import MCPClient
import json
from datetime import datetime
import os

app = FastAPI(title="OpenKG-ToolAgent API", version="1.0")

# 挂载输出目录
app.mount(
    "/output",
    StaticFiles(directory=os.path.expanduser("~/OpenKG-ToolAgent/output")),
    name="output"
)

# 挂载模型目录（新增）
app.mount(
    "/files",
    StaticFiles(directory=os.path.expanduser("~/OpenKG-ToolAgent/output/models")),
    name="models"
)

client = MCPClient()
loop = asyncio.get_event_loop()

@app.on_event("startup")
async def startup_event():
    """在 FastAPI 启动时连接 MCP 工具服务器"""
    await client.connect_to_server("tools/server.py")

@app.on_event("shutdown")
async def shutdown_event():
    """关闭连接"""
    await client.clean()

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    """
    模拟 OpenAI 的 /v1/chat/completions 接口。
    外部应用可以直接用 OpenAI SDK 调用此端点。
    """
    body = await request.json()
    messages = body.get("messages", [])
    model = body.get("model", "openkg-toolagent")
    stream = body.get("stream", False)

    # 拼接用户消息
    user_message = ""
    for msg in messages:
        if msg["role"] == "user":
            user_message += msg["content"] + "\n"

    async def response_generator():
        """流式输出（兼容 OpenAI 的 SSE 格式）"""
        try:
            # 直接调用内部逻辑
            result = await client.process_query(user_message)
            completion_id = f"cmpl-{datetime.now().timestamp()}"
            chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": int(datetime.now().timestamp()),
                "model": model,
                "choices": [{
                    "delta": {"content": result},
                    "index": 0,
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            err = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(err)}\n\n"

    # 如果需要流式输出
    if stream:
        return StreamingResponse(response_generator(), media_type="text/event-stream")

    # 否则直接返回完整结果
    result = await client.process_query(user_message)
    completion_id = f"cmpl-{datetime.now().timestamp()}"
    response = {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(datetime.now().timestamp()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": result
            },
            "finish_reason": "stop"
        }]
    }
    return JSONResponse(content=response)

if __name__ == "__main__":
    uvicorn.run("server_api:app", host="0.0.0.0", port=8000, reload=True)