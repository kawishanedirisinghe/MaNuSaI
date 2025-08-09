import asyncio
import json
import os
import threading
import time
import queue
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.middleware.wsgi import WSGIMiddleware

from app.logger import log_queue, logger

# Import the existing Flask app and helpers
from main import app as flask_app  # noqa: E402
from main import run_async_task, run_flow_async_task, running_tasks  # noqa: E402
from main import PROCESS_TIMEOUT, FILE_CHECK_INTERVAL  # noqa: E402


app = FastAPI(title="OpenManus ASGI Server")

# Mount the Flask app at root
app.mount("/", WSGIMiddleware(flask_app))


async def _stream_task_over_websocket(
    websocket: WebSocket,
    task_thread: threading.Thread,
    task_id: str,
) -> None:
    start_time = time.time()
    while task_thread.is_alive() or not log_queue.empty():
        # Stop signal
        if running_tasks.get(task_id, {}).get("stop_flag", False):
            try:
                await websocket.send_text("Task stopped by user.\n")
            except Exception:
                pass
            break

        # Timeout
        if time.time() - start_time > PROCESS_TIMEOUT:
            try:
                await websocket.send_text("0303030")
            except Exception:
                pass
            break

        new_content: str = ""
        try:
            new_content = log_queue.get(timeout=0.1)
        except queue.Empty:
            pass

        if new_content:
            try:
                await websocket.send_text(new_content)
            except Exception:
                break
        else:
            await asyncio.sleep(FILE_CHECK_INTERVAL)

    # Final sentinel
    try:
        await websocket.send_text("0303030")
    except Exception:
        pass


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    try:
        # Expect an initial JSON payload from client
        payload_text = await websocket.receive_text()
        data = json.loads(payload_text or "{}")
        message: str = data.get("message", "")
        task_id: str = data.get("task_id") or f"task_{os.urandom(4).hex()}"
        uploaded_files = data.get("uploaded_files", [])
        mood: str = data.get("mood", "default")
        chat_id: str = data.get("chat_id") or task_id

        if not message:
            await websocket.send_text(json.dumps({"error": "No message provided"}))
            await websocket.close()
            return

        # Launch background task using existing function
        task_thread = threading.Thread(
            target=run_async_task,
            args=(message, task_id),
            daemon=True,
        )
        task_thread.start()

        # Stream results over WS
        await _stream_task_over_websocket(websocket, task_thread, task_id)

    except WebSocketDisconnect:
        # Client disconnected; signal stop if task exists
        try:
            if "task_id" in locals() and task_id in running_tasks:
                running_tasks[task_id]["stop_flag"] = True
        except Exception:
            pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


@app.websocket("/ws/flow")
async def websocket_flow(websocket: WebSocket):
    await websocket.accept()
    try:
        payload_text = await websocket.receive_text()
        data = json.loads(payload_text or "{}")
        message: str = data.get("message", "")
        task_id: str = data.get("task_id") or f"task_{os.urandom(4).hex()}"
        uploaded_files = data.get("uploaded_files", [])

        if not message:
            await websocket.send_text(json.dumps({"error": "No message provided"}))
            await websocket.close()
            return

        # Launch background flow task
        task_thread = threading.Thread(
            target=run_flow_async_task,
            args=(message, task_id),
            daemon=True,
        )
        task_thread.start()

        await _stream_task_over_websocket(websocket, task_thread, task_id)

    except WebSocketDisconnect:
        try:
            if "task_id" in locals() and task_id in running_tasks:
                running_tasks[task_id]["stop_flag"] = True
        except Exception:
            pass
    except Exception as e:
        try:
            await websocket.send_text(json.dumps({"error": str(e)}))
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


# Optional: programmatic launcher for local dev with TLS via env vars
if __name__ == "__main__":
    import uvicorn

    ssl_certfile: Optional[str] = os.getenv("SSL_CERTFILE")
    ssl_keyfile: Optional[str] = os.getenv("SSL_KEYFILE")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "3000")),
        reload=False,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
    )