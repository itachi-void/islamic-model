import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.api.routes import router, get_pipeline, get_hadith_service, get_chat_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm all heavy singletons at startup so first user request is instant."""
    try:
        logger.info("[STARTUP] Pre-warming pipeline...")
        get_pipeline()
        logger.info("[STARTUP] Pre-warming hadith service...")
        get_hadith_service()
        logger.info("[STARTUP] Pre-warming chat service...")
        get_chat_service()
        logger.info("[STARTUP] All services ready.")
    except Exception as e:
        logger.error(f"[STARTUP] Warm-up failed (non-fatal): {e}")
    yield

app = FastAPI(
    title="Islamic AI Engine",
    version="2.0.0",
    description="Unified RAG Search & Chat Engine for Holy Quran and Sahih Al-Bukhari",
    lifespan=lifespan
)


@app.get("/health")
def health():
    return {"status": "ok", "pipeline": "ready"}

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