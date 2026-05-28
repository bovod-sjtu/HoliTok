from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import quote, urljoin, urlparse, urlunparse
from urllib.request import Request, urlopen

from .presets import (
    checkpoint_filename,
    checkpoint_sha256,
    is_preset,
    semantic_checkpoint_filename,
    semantic_checkpoint_sha256,
)


DEFAULT_CHECKPOINT_SOURCE = "https://huggingface.co/bovod-sjtu/HoliTok"
DEFAULT_HF_REPO_ID = DEFAULT_CHECKPOINT_SOURCE
CHECKPOINT_SOURCE_ENV = "HOLITOK_CHECKPOINT_SOURCE"
HF_REPO_ENV = "HOLITOK_HF_REPO_ID"
CACHE_ENV = "HOLITOK_CACHE_DIR"
DOWNLOAD_TIMEOUT_ENV = "HOLITOK_DOWNLOAD_TIMEOUT"
DEFAULT_DOWNLOAD_TIMEOUT = 60.0


def _is_url(source: str) -> bool:
    parsed = urlparse(source)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _cache_root(cache_dir: str | Path | None) -> Path:
    if cache_dir is not None:
        return Path(cache_dir).expanduser()
    if os.environ.get(CACHE_ENV):
        return Path(os.environ[CACHE_ENV]).expanduser()
    return Path.home() / ".cache" / "holitok"


def _source_cache_dir(source: str, cache_dir: str | Path | None) -> Path:
    source_key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return _cache_root(cache_dir) / "checkpoints" / source_key


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_marker(path: Path) -> Path:
    return path.with_name(path.name + ".sha256")


def _download_timeout() -> float:
    value = os.environ.get(DOWNLOAD_TIMEOUT_ENV)
    if value is None:
        return DEFAULT_DOWNLOAD_TIMEOUT
    try:
        timeout = float(value)
    except ValueError as exc:
        raise ValueError(f"{DOWNLOAD_TIMEOUT_ENV} must be a number of seconds") from exc
    if timeout <= 0:
        raise ValueError(f"{DOWNLOAD_TIMEOUT_ENV} must be positive")
    return timeout


def _cached_file_is_valid(path: Path, expected_sha256: str | None) -> bool:
    if not path.is_file():
        return False
    if expected_sha256 is None:
        return True

    marker = _sha256_marker(path)
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == expected_sha256:
        return True

    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        return False
    marker.write_text(f"{expected_sha256}\n", encoding="utf-8")
    return True


def _checkpoint_url(source: str, filename: str, revision: str | None = None) -> str:
    parsed = urlparse(source)
    parts = [part for part in parsed.path.split("/") if part]
    quoted_filename = quote(filename, safe="/")
    revision = revision or "main"

    if len(parts) >= 2 and parts[0] == "a":
        path = f"/api/a/{parts[1]}/resolve/{quoted_filename}"
        return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))

    if len(parts) >= 3 and parts[0] == "api" and parts[1] == "a":
        path = f"/api/a/{parts[2]}/resolve/{quoted_filename}"
        return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))

    if parsed.path.endswith(".pt"):
        return source

    if parsed.netloc == "huggingface.co" and len(parts) >= 2:
        if len(parts) >= 4 and parts[2] in {"tree", "resolve"}:
            revision = parts[3]
        path = f"/{parts[0]}/{parts[1]}/resolve/{quote(revision, safe='')}/{quoted_filename}"
        return urlunparse(parsed._replace(path=path, params="", query="", fragment=""))

    return urljoin(source.rstrip("/") + "/", quoted_filename)


def _download_url_checkpoint(
    *,
    source: str,
    filename: str,
    expected_sha256: str | None,
    revision: str | None,
    cache_dir: str | Path | None,
    local_files_only: bool,
) -> str:
    target = _source_cache_dir(source, cache_dir) / filename
    if _cached_file_is_valid(target, expected_sha256):
        return str(target)

    if local_files_only:
        raise FileNotFoundError(f"Checkpoint is not cached locally: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    checkpoint_url = _checkpoint_url(source, filename, revision=revision)
    request = Request(checkpoint_url, headers={"User-Agent": "holitok"})

    try:
        with urlopen(request, timeout=_download_timeout()) as response, tmp_target.open("wb") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        os.replace(tmp_target, target)
    finally:
        if tmp_target.exists():
            tmp_target.unlink()

    if not _cached_file_is_valid(target, expected_sha256):
        if target.exists():
            target.unlink()
        raise RuntimeError(f"Downloaded checkpoint failed SHA256 verification: {checkpoint_url}")

    return str(target)


def _resolve_preset_artifact(
    model_name: str | None,
    path: str | Path | None,
    *,
    filename_fn,
    sha256_fn,
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> str | None:
    if path is not None:
        return str(path)
    if model_name is None or not is_preset(model_name):
        return None

    source = (
        repo_id
        or os.environ.get(CHECKPOINT_SOURCE_ENV)
        or os.environ.get(HF_REPO_ENV)
        or DEFAULT_CHECKPOINT_SOURCE
    )
    if not source:
        raise ValueError(
            "checkpoint is required unless a checkpoint source is configured. "
            f"Pass repo_id=..., set {CHECKPOINT_SOURCE_ENV}, or set {HF_REPO_ENV}."
        )

    filename = filename_fn(model_name)
    expected_sha256 = sha256_fn(model_name)

    if _is_url(source):
        return _download_url_checkpoint(
            source=source,
            filename=filename,
            expected_sha256=expected_sha256,
            revision=revision,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError(
            "Loading HoliTok checkpoints from Hugging Face requires `huggingface_hub`. "
            "Install it with `pip install huggingface_hub`."
        ) from exc

    return hf_hub_download(
        repo_id=source,
        filename=filename,
        revision=revision,
        token=token,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
        local_files_only=local_files_only,
    )


def resolve_checkpoint(
    model_name: str | None,
    checkpoint: str | Path | None,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> str | None:
    return _resolve_preset_artifact(
        model_name,
        checkpoint,
        filename_fn=checkpoint_filename,
        sha256_fn=checkpoint_sha256,
        repo_id=repo_id,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )


def resolve_semantic_checkpoint(
    model_name: str | None,
    checkpoint: str | Path | None,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | bool | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
) -> str | None:
    return _resolve_preset_artifact(
        model_name,
        checkpoint,
        filename_fn=semantic_checkpoint_filename,
        sha256_fn=semantic_checkpoint_sha256,
        repo_id=repo_id,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
