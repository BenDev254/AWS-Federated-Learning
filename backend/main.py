# main.py

from fastapi import FastAPI, WebSocket
from router import app as router_app
from models import Base
from db.database import engine
from fastapi.middleware.cors import CORSMiddleware

Base.metadata.create_all(bind=engine)

app = FastAPI()  # ✅ this is the actual main app

# ✅ Apply CORS to the main app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with your frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Mount the internal app that has all your routes
app.mount("/", router_app)

# ✅ WebSocket on the main app
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("WebSocket connection established.")
    while True:
        try:
            message = await websocket.receive_text()
            await websocket.send_text(f"Echo: {message}")
        except Exception:
            break
