"""ClusterRL backend.  Run:  uvicorn main:app --port 8000"""
import asyncio
import sys
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Make the real-data package importable when run from this directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from session import Session  # noqa: E402

app = FastAPI(title="ClusterRL backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    session = Session()
    sender = asyncio.create_task(session.run(ws))
    try:
        while True:
            session.handle(await ws.receive_json())
    except WebSocketDisconnect:
        pass
    finally:
        sender.cancel()


# After `npm run build` the backend can serve the UI itself (single-port demo).
dist = Path(__file__).resolve().parent.parent / "frontend" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="ui")
