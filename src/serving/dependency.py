"""Dependency injection for the storage backend (IPP-002).

This is the single seam where "which storage backend" is decided. Swapping to
S3 means adding one branch here and one class in storage.py — no other file
changes.

NOTE: the reference MSP project had a bug here — it passed the *backend name*
("local") as `base_dir` instead of the configured directory. Fixed below by
using `settings.LOCAL_ARTIFACT_DIR`.
"""

from src.serving.config import settings
from src.serving.storage import ArtifactStorage, LocalArtifactStorage


def get_artifact_storage() -> ArtifactStorage:
    if settings.ARTIFACT_STORAGE_BACKEND == "local":
        return LocalArtifactStorage(base_dir=settings.LOCAL_ARTIFACT_DIR)
    # elif settings.ARTIFACT_STORAGE_BACKEND == "s3":
    #     return S3ArtifactStorage(bucket=settings.S3_BUCKET)
    raise ValueError(
        f"Unsupported storage backend: {settings.ARTIFACT_STORAGE_BACKEND}"
    )
