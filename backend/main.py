import os
import logging
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI, Response
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
            _warmup_done = True


@asynccontextmanager
async def lifespan(app: FastAPI):
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


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return Response(status_code=204)


app.include_router(router)

# Mount Frontend static files
frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="static")


# WSGI Adapter for PythonAnywhere / uWSGI servers
try:
    from a2wsgi import ASGIMiddleware
    _raw_wsgi = ASGIMiddleware(app)

    def wsgi_app(environ, start_response):
        def safe_start_response(status, headers, exc_info=None):
            try:
                return start_response(status, headers, exc_info)
            except TypeError:
                return start_response(status, headers)
        return _raw_wsgi(environ, safe_start_response)

except ImportError:
    wsgi_app = app