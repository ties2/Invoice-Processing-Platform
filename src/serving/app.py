"""FastAPI application entrypoint (IPP-009).

Creates the app, ensures tables exist, and mounts the router. Run locally with:
    uvicorn src.serving.app:app --reload
"""

from fastapi import FastAPI

from src.serving import db_models  # noqa: F401  (registers ORM tables)
from src.serving.api import router
from src.serving.database import Base, engine


def create_app() -> FastAPI:
    Base.metadata.create_all(bind=engine)
    app = FastAPI(
        title="Invoice Processing Platform",
        description="OCR + extraction + confidence routing with a human-in-the-loop feedback loop.",
        version="0.1.0",
    )
    app.include_router(router)
    return app


app = create_app()
