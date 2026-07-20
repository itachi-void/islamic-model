import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.api.routes import router

app = FastAPI(
    title="Islamic AI Engine",
    version="2.0.0",
    description="Unified RAG Search & Chat Engine for Holy Quran and Sahih Al-Bukhari"
)

app.include_router(router)

# Mount Frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")