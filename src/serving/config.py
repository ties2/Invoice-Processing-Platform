"""Application settings.

Twelve-factor config: everything comes from environment variables (or a local
.env file in development). No secrets or paths are hardcoded. This mirrors the
reference MSP project's config style, adapted to the invoice-processing domain.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- App ---
    app_env: str = "development"

    # --- Database (PostgreSQL) ---
    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str
    postgres_port: int = 5432

    # --- Artifact storage (the raw PDF/image files live here) ---
    ARTIFACT_STORAGE_BACKEND: str = "local"
    LOCAL_ARTIFACT_DIR: str = "/app/invoice_artifacts/"

    # --- Confidence routing thresholds (the heart of the Zenvoices-style flow) ---
    # Above AUTO_THRESHOLD  -> book automatically.
    # Below REVIEW_THRESHOLD -> always send to a human.
    # In between            -> human review (safe default).
    AUTO_BOOK_THRESHOLD: float = 0.95
    REVIEW_THRESHOLD: float = 0.80

    # --- Grafana (optional) ---
    gf_security_admin_user: str | None = None
    gf_security_admin_password: str | None = None

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
