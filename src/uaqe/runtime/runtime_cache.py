"""
UAQE Phase E.1 Runtime Cache Manager
Provides deterministic cryptographic SHA-256 cache management for prebuilt reconstructed
TFLite models and static FlatBuffer offset metadata.
"""

from __future__ import annotations

import os
import sys
import json
import hashlib
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path


RUNTIME_CACHE_VERSION = "1.0.0"


class RuntimeCacheManager:
    """Manages persistent cryptographic cache for reconstructed models and static metadata."""

    def __init__(self, cache_dir: str = "output/phase_e1/runtime_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.version = RUNTIME_CACHE_VERSION

    @staticmethod
    def compute_sha256_file(file_path: Union[str, Path]) -> str:
        """Computes SHA-256 of a file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def compute_sha256_bytes(data: Union[bytes, bytearray]) -> str:
        """Computes SHA-256 of raw bytes."""
        hasher = hashlib.sha256()
        hasher.update(data)
        return hasher.hexdigest()

    def generate_cache_key(self, archive_sha256: str) -> str:
        """Generates deterministic cache key from source archive hash and runtime version."""
        composite = f"{archive_sha256}_v{self.version}"
        return hashlib.sha256(composite.encode("utf-8")).hexdigest()

    def get_cached_model_path(self, archive_sha256: str) -> Path:
        """Returns the expected filesystem path for a cached TFLite model."""
        key = self.generate_cache_key(archive_sha256)
        return self.cache_dir / f"model_{key}.tflite"

    def get_cached_metadata_path(self, archive_sha256: str) -> Path:
        """Returns the expected filesystem path for cached metadata."""
        key = self.generate_cache_key(archive_sha256)
        return self.cache_dir / f"meta_{key}.json"

    def has_cached_model(self, archive_sha256: str) -> bool:
        """Checks if a valid, uncorrupted cached model exists for the given archive SHA-256."""
        model_path = self.get_cached_model_path(archive_sha256)
        meta_path = self.get_cached_metadata_path(archive_sha256)

        if not model_path.exists() or not meta_path.exists():
            return False

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            if meta.get("archive_sha256") != archive_sha256:
                return False
            if meta.get("runtime_cache_version") != self.version:
                return False

            expected_model_hash = meta.get("model_sha256")
            actual_model_hash = self.compute_sha256_file(model_path)
            if actual_model_hash != expected_model_hash:
                # Corrupted cache file
                return False

            return True
        except Exception:
            return False

    def load_cached_model(self, archive_sha256: str) -> Optional[bytes]:
        """Loads and verifies cached model bytes if valid, else returns None."""
        if not self.has_cached_model(archive_sha256):
            return None

        model_path = self.get_cached_model_path(archive_sha256)
        try:
            with open(model_path, "rb") as f:
                return f.read()
        except Exception:
            return None

    def store_cached_model(
        self,
        archive_sha256: str,
        model_bytes: Union[bytes, bytearray],
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """Stores reconstructed model bytes and cryptographically seals metadata."""
        model_path = self.get_cached_model_path(archive_sha256)
        meta_path = self.get_cached_metadata_path(archive_sha256)

        model_hash = self.compute_sha256_bytes(model_bytes)

        # Write model
        with open(model_path, "wb") as f:
            f.write(model_bytes)

        # Write metadata
        meta = {
            "archive_sha256": archive_sha256,
            "runtime_cache_version": self.version,
            "model_sha256": model_hash,
            "model_size_bytes": len(model_bytes),
            "created_at": extra_metadata.get("timestamp") if extra_metadata else None,
            "extra": extra_metadata or {}
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        return model_path

    def invalidate(self, archive_sha256: str) -> None:
        """Deletes cache files associated with a specific archive hash."""
        model_path = self.get_cached_model_path(archive_sha256)
        meta_path = self.get_cached_metadata_path(archive_sha256)
        if model_path.exists():
            model_path.unlink()
        if meta_path.exists():
            meta_path.unlink()

    def clear_all(self) -> int:
        """Clears all cached files in the cache directory."""
        count = 0
        for p in self.cache_dir.glob("*"):
            if p.is_file():
                p.unlink()
                count += 1
        return count
