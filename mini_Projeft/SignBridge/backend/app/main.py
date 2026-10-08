"""
SignBridge backend (FastAPI).

Start it from the backend folder:
    uvicorn app.main:app --reload --port 8000
Interactive API docs:  http://127.0.0.1:8000/docs
The app starts even if the dataset or the trained model is missing - it then reports what is missing.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import recognition, speech, system

app = FastAPI(title="SignBridge API", version="1.0.0",
              description="Two-way ISL communication assistant (closed-set, 100-sentence prototype).")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"], allow_headers=["*"],
)
app.include_router(system.router)
app.include_router(recognition.router)
app.include_router(speech.router)


@app.get("/")
def root():
    return {"name": "SignBridge API", "docs": "/docs", "status": "/api/status"}
