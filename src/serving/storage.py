"""Artifact storage abstraction (IPP-002).

Reused almost verbatim from the reference MSP project — this is the whole point
of the "infrastructure is decoupled from the domain" design: the same storage
layer that held `.joblib` model artifacts now holds raw invoice `PDF`/image
bytes, with zero changes to its logic.

One concrete backend today (local disk), behind a `Protocol` so an S3/GCS
backend is a new class, not a rewrite. Includes path-traversal protection so a
malicious `artifact_uri` cannot escape the storage root.
"""

import os
from typing import Protocol


class StorageError(Exception):
    pass


class ArtifactNotFoundError(StorageError):
    pass


class InvalidURIError(StorageError):
    pass


class PathTraversalError(StorageError):
    pass


class ArtifactStorage(Protocol):
    """Interface every storage backend must satisfy."""

    def save(self, uri: str, content: bytes) -> None: ...
    def get(self, uri: str) -> bytes: ...
    def exists(self, uri: str) -> bool: ...
    def delete(self, uri: str) -> None: ...


class LocalArtifactStorage:
    """Local-disk backend. URIs look like `local://<supplier>/<id>/invoice.pdf`."""

    SCHEME = "local://"

    def __init__(self, base_dir: str):
        # Absolute path so the traversal check below is reliable.
        self.base_dir = os.path.abspath(base_dir)
        os.makedirs(self.base_dir, exist_ok=True)

    def _resolve_path(self, uri: str) -> str:
        if not uri.startswith(self.SCHEME):
            raise InvalidURIError(f"Invalid URI scheme for LocalStorage: {uri}")

        # Strip the scheme and any leading slashes so os.path.join does NOT treat
        # the remainder as an absolute path.
        relative_path = uri[len(self.SCHEME):].lstrip("/")
        full_path = os.path.abspath(os.path.join(self.base_dir, relative_path))

        # Security: reject anything that resolves outside the storage root.
        if not full_path.startswith(self.base_dir):
            raise PathTraversalError("Path traversal attempt detected")

        return full_path

    def save(self, uri: str, content: bytes) -> None:
        path = self._resolve_path(uri)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content)

    def get(self, uri: str) -> bytes:
        path = self._resolve_path(uri)
        if not os.path.exists(path):
            raise ArtifactNotFoundError(f"Artifact not found: {uri}")
        with open(path, "rb") as f:
            return f.read()

    def exists(self, uri: str) -> bool:
        return os.path.exists(self._resolve_path(uri))

    def delete(self, uri: str) -> None:
        path = self._resolve_path(uri)
        if not os.path.exists(path):
            raise ArtifactNotFoundError(f"Artifact not found: {uri}")
        os.remove(path)
