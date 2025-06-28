from fastapi import FastAPI, WebSocket
from router import app as router_app
from models import Base
from db.database import engine
from fastapi.middleware.cors import CORSMiddleware

Base.metadata.create_all(bind=engine)

app = router_app

# ✅ Allow all origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# ✅ Add this WebSocket endpoint
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
