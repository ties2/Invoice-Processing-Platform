"""Shared test fixtures.

Tests run against an in-memory SQLite database instead of Postgres, so the whole
suite is hermetic and needs no running services. The ORM models are backend-
agnostic, so this is a faithful test of the real data layer.
"""

import os
import tempfile

os.environ.setdefault("POSTGRES_USER", "test")
os.environ.setdefault("POSTGRES_PASSWORD", "test")
os.environ.setdefault("POSTGRES_DB", "test")
os.environ.setdefault("POSTGRES_HOST", "localhost")
os.environ["LOCAL_ARTIFACT_DIR"] = tempfile.mkdtemp()

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import src.serving.database as database
from src.serving.database import Base


def _memory_engine():
    # StaticPool shares ONE in-memory connection across threads, so the tables
    # created here are visible to requests TestClient runs on another thread.
    return create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


@pytest.fixture
def db_session():
    engine = _memory_engine()
    database.engine = engine
    database.SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    import src.serving.db_models  # noqa: F401  register tables
    Base.metadata.create_all(engine)
    session = database.SessionLocal()
    yield session
    session.close()


@pytest.fixture
def client():
    engine = _memory_engine()
    database.engine = engine
    database.SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    import src.serving.db_models  # noqa: F401
    Base.metadata.create_all(engine)

    # Fresh pipeline (clear the lru_cache so tests don't share state).
    from src.serving.pipeline_factory import get_pipeline
    get_pipeline.cache_clear()

    from fastapi.testclient import TestClient
    from src.serving.app import create_app
    return TestClient(create_app())


@pytest.fixture
def clean_invoice_bytes():
    return (
        b"KPN B.V.\n"
        b"Invoice Number: INV-2026-1234\n"
        b"Date: 12-08-2026\n"
        b"Subtotal                    1000,00\n"
        b"VAT 21%                       210,00\n"
        b"Total                       1210,00\n"
        b"Currency: EUR\n"
    )
