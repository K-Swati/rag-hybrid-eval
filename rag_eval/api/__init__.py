"""HTTP layer: FastAPI app, routes, schemas and the web UI (``ui/index.html``, served at ``/``)."""

from rag_eval.api.app import create_app

app = create_app()

__all__ = ["app", "create_app"]
