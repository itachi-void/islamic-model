import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.api.routes import router

app = FastAPI(
    title="Islamic AI Engine",
    version="2.0.0",
    description="Unified RAG Search & Chat Engine for Holy Quran and Sahih Al-Bukhari"
)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.endswith((".js", ".css", ".html")):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.include_router(router)

# Mount Frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")