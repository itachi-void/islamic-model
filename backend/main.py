import os
import logging
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.api.routes import router, get_pipeline, get_hadith_service, get_chat_service

logger = logging.getLogger(__name__)

# Global warmup state
_warmup_done = False
_warmup_lock = threading.Lock()


def _background_warmup():
    """
    Run in a daemon thread at startup.
    Warms up all heavy singletons WITHOUT blocking WSGI from accepting requests.
    PythonAnywhere free tier has 1 worker — if warmup is sync/blocking, first request
    hangs until warmup finishes (15s+). Background thread releases the worker immediately.
    """
    global _warmup_done
    try:
        logger.info("[WARMUP] Starting background service initialization...")
        get_pipeline()
        logger.info("[WARMUP] Pipeline ready.")
        get_hadith_service()
        logger.info("[WARMUP] Hadith service ready.")
        get_chat_service()
        logger.info("[WARMUP] Chat service ready.")
        with _warmup_lock:
            _warmup_done = True
        logger.info("[WARMUP] All services warmed up successfully.")
    except Exception as e:
        logger.error(f"[WARMUP] Background warmup failed (non-fatal): {e}")
        with _warmup_lock:
            _warmup_done = True  # Mark done even on failure so requests proceed


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Kicks off background warmup thread immediately at WSGI startup.
    WSGI worker is free to handle requests within milliseconds.
    First request to /chat will trigger lazy-init if warmup isn't done yet (still fast via inverted index).
    """
    t = threading.Thread(target=_background_warmup, daemon=True, name="warmup-thread")
    t.start()
    logger.info("[STARTUP] Background warmup thread started.")
    yield


app = FastAPI(
    title="Islamic AI Engine",
    version="2.0.0",
    description="Unified RAG Search & Chat Engine for Holy Quran and Sahih Al-Bukhari",
    lifespan=lifespan
)


@app.get("/health")
def health():
    global _warmup_done
    with _warmup_lock:
        ready = _warmup_done
    return {"status": "ok", "pipeline_ready": ready}


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