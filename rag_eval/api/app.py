"""App factory: settings -> components (wiring) -> routes, and the UI at ``/``."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse

from rag_eval.api.jobs import EvalJob
from rag_eval.api.routes import router
from rag_eval.api.state import AppState
from rag_eval.config import Settings
from rag_eval.wiring import Components, build_components

UI_FILE = Path(__file__).parent / "ui" / "index.html"


def create_app(components: Components | None = None, settings: Settings | None = None) -> FastAPI:
    """``components`` is injectable for tests; by default everything is built from the environment."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        c = components or build_components(settings or Settings.from_env())
        app.state.rag = AppState(c, EvalJob(c))
        yield

    app = FastAPI(title="Frostfield Creamery assistant", version="2.0.0", lifespan=lifespan)

    @app.exception_handler(ValueError)
    async def value_error(_: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(UI_FILE, media_type="text/html")

    app.include_router(router)
    return app
