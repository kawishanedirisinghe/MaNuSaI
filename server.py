import asyncio
import json
import os
import threading
import time
import queue
from typing import Optional
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.middleware.wsgi import WSGIMiddleware

from app.logger import log_queue, logger

# Import the existing Flask app and helpers
from main import app as flask_app  # noqa: E402
from main import run_async_task, run_flow_async_task, running_tasks  # noqa: E402
from main import PROCESS_TIMEOUT, FILE_CHECK_INTERVAL  # noqa: E402
from main import load_chat_by_id, extract_text_from_file, trim_context, save_chat_history  # noqa: E402


app = FastAPI(title="OpenManus ASGI Server")

# Mount the Flask app at root
app.mount("/", WSGIMiddleware(flask_app))


async def _stream_task_over_websocket(
    websocket: WebSocket,
    task_thread: threading.Thread,
    task_id: str,
    *,
    save_meta: Optional[dict] = None,
) -> None:
    start_time = time.time()
    full_response_parts: list[str] = []
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
            full_response_parts.append(new_content)
            try:
                await websocket.send_text(new_content)
            except Exception:
                break
        else:
            await asyncio.sleep(FILE_CHECK_INTERVAL)

    # Persist chat history if requested
    if save_meta:
        try:
            elapsed = time.time() - start_time
            chat_data = {
                "id": save_meta.get("chat_id") or task_id,
                "timestamp": datetime.now().isoformat(),
                "user_message": trim_context(save_meta.get("user_message", "")),
                "agent_response": "".join(full_response_parts),
                "agent_type": save_meta.get("agent_type", "manus"),
                "uploaded_files": save_meta.get("uploaded_files", []),
                "mood": save_meta.get("mood", "default"),
                "elapsed_seconds": round(elapsed, 3),
            }
            save_chat_history(chat_data)
        except Exception as e:
            logger.error(f"Failed to save chat history (WS): {e}")

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

        # Build file context
        file_context = ""
        if uploaded_files:
            file_context = "\n\nUploaded files context:\n"
            for file_info in uploaded_files:
                try:
                    filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], file_info.get('filename', ''))
                    if filepath and os.path.exists(filepath):
                        snippet = extract_text_from_file(filepath)
                        if len(snippet) > 4000:
                            snippet = snippet[:4000]
                        file_context += f"\n--- {file_info.get('original_name') or os.path.basename(filepath)} ---\n{snippet}\n"
                except Exception as e:
                    logger.error(f"WS file read error {file_info}: {e}")

        # History context
        history_context = ""
        history_item = load_chat_by_id(chat_id)
        if history_item:
            try:
                prev_user = history_item.get('user_message') or ''
                prev_agent = history_item.get('agent_response') or ''
                if prev_user or prev_agent:
                    history_context = (
                        f"\n\nPrevious conversation context (trimmed):\n[User]\n{trim_context(prev_user)}\n[Agent]\n{trim_context(prev_agent)}\n"
                    )
            except Exception as e:
                logger.error(f"WS build history context error: {e}")

        # Mood prefix
        mood_prefix = f"[Agent Mood: {mood}]\n" if mood and mood != "default" else ""
        full_message = trim_context(mood_prefix + message + file_context + history_context)

        # Launch background task using existing function
        task_thread = threading.Thread(
            target=run_async_task,
            args=(full_message, task_id),
            daemon=True,
        )
        task_thread.start()

        # Stream results over WS and persist
        await _stream_task_over_websocket(
            websocket,
            task_thread,
            task_id,
            save_meta={
                "chat_id": chat_id,
                "user_message": message,
                "uploaded_files": uploaded_files,
                "mood": mood,
                "agent_type": "manus",
            },
        )

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

        # Build file context (smaller limit for flow)
        file_context = ""
        if uploaded_files:
            file_context = "\n\nUploaded files context:\n"
            for file_info in uploaded_files:
                try:
                    filepath = os.path.join(flask_app.config['UPLOAD_FOLDER'], file_info.get('filename', ''))
                    if filepath and os.path.exists(filepath):
                        snippet = extract_text_from_file(filepath)
                        if len(snippet) > 2000:
                            snippet = snippet[:2000]
                        file_context += f"\n--- {file_info.get('original_name') or os.path.basename(filepath)} ---\n{snippet}\n"
                except Exception as e:
                    logger.error(f"WS flow file read error {file_info}: {e}")

        full_message = message + file_context

        # Launch background flow task
        task_thread = threading.Thread(
            target=run_flow_async_task,
            args=(full_message, task_id),
            daemon=True,
        )
        task_thread.start()

        await _stream_task_over_websocket(
            websocket,
            task_thread,
            task_id,
            save_meta={
                "chat_id": task_id,
                "user_message": message,
                "uploaded_files": uploaded_files,
                "agent_type": "flow",
            },
        )

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