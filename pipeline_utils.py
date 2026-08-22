from __future__ import annotations

import fnmatch
import json
import logging
import os
import random
import re
import shutil
import threading
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError
import xxhash


def setup_logging(common: Dict[str, Any], stage_name: str) -> logging.Logger:
    log_dir = Path(common["logging"]["dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(stage_name)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    if not root.handlers:
        debug = logging.FileHandler(log_dir / common["logging"]["debug_log"], encoding="utf-8")
        debug.setLevel(logging.DEBUG)
        debug.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        root.addHandler(debug)
    return logger


def retry_kwargs(common: Dict[str, Any], op: str) -> Dict[str, Any]:
    return dict(common["retry"][op])


def runtime_settings(common: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve and validate the bounded file-processing runtime profile."""
    configured = common.get("runtime") or {}
    settings = {
        "file_workers": int(configured.get("file_workers", 2)),
        "download_workers": int(configured.get("download_workers", 1)),
        "upload_workers": int(configured.get("upload_workers", 1)),
        "download_queue_size": int(configured.get("download_queue_size", 0)),
        "upload_queue_size": int(configured.get("upload_queue_size", 0)),
        "max_inflight_files": int(configured.get("max_inflight_files", 2)),
        "batch_rows": int(configured.get("batch_rows", 4096)),
        "min_free_disk_gb": float(configured.get("min_free_disk_gb", 5.0)),
        "local_cache_dir": configured.get("local_cache_dir"),
        "local_temp_dir": configured.get("local_temp_dir"),
    }
    for name in (
        "file_workers", "download_workers", "upload_workers",
        "download_queue_size", "upload_queue_size",
        "max_inflight_files", "batch_rows",
    ):
        if settings[name] <= 0:
            if name.endswith("queue_size") and settings[name] == 0:
                continue
            raise ValueError(f"runtime.{name} must be > 0")
    if settings["min_free_disk_gb"] < 0:
        raise ValueError("runtime.min_free_disk_gb must be >= 0")
    return settings


class _BoundedIOGate:
    """Limit active I/O and bound callers waiting behind that limit."""

    def __init__(self, workers: int, queue_size: int) -> None:
        self._admission = threading.BoundedSemaphore(workers + queue_size)
        self._active = threading.BoundedSemaphore(workers)

    @contextmanager
    def slot(self) -> Iterator[None]:
        self._admission.acquire()
        try:
            self._active.acquire()
            try:
                yield
            finally:
                self._active.release()
        finally:
            self._admission.release()


_IO_GATES: Dict[tuple[int, str], _BoundedIOGate] = {}
_IO_GATES_LOCK = threading.Lock()


def _io_gate(common: Dict[str, Any], operation: str) -> _BoundedIOGate:
    settings = runtime_settings(common)
    if operation == "download":
        workers = settings["download_workers"]
        queue_size = settings["download_queue_size"]
    elif operation == "upload":
        workers = settings["upload_workers"]
        queue_size = settings["upload_queue_size"]
    else:
        raise ValueError(f"Unsupported I/O operation: {operation!r}")
    key = (id(common), operation)
    with _IO_GATES_LOCK:
        gate = _IO_GATES.get(key)
        if gate is None:
            gate = _BoundedIOGate(workers, queue_size)
            _IO_GATES[key] = gate
    return gate


def local_work_root(common: Dict[str, Any]) -> Path:
    """Return the configured temporary work root without creating it."""
    settings = runtime_settings(common)
    value = settings["local_temp_dir"] or common["storage"]["local_work_dir"]
    return Path(value).expanduser()


def ensure_disk_space(common: Dict[str, Any], path: str | Path | None = None) -> Dict[str, Any]:
    """Fail before processing when the target filesystem is nearly full."""
    settings = runtime_settings(common)
    target = (Path(path) if path is not None else local_work_root(common)).expanduser().resolve()
    probe = target
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    free_gb = usage.free / (1024 ** 3)
    required_gb = settings["min_free_disk_gb"]
    if free_gb < required_gb:
        raise OSError(
            f"Insufficient free disk space at {probe}: "
            f"{free_gb:.2f} GiB available, {required_gb:.2f} GiB required."
        )
    return {
        "path": str(probe),
        "free_bytes": int(usage.free),
        "free_gib": round(free_gb, 3),
        "required_free_gib": required_gb,
    }


def file_detail(path: str | Path, algorithm: str = "xxh3_128", remote_path: str | None = None) -> Dict[str, Any]:
    """Return a streaming checksum and byte count for a local artifact."""
    local = Path(path)
    if not local.is_file():
        raise FileNotFoundError(local)
    if algorithm == "xxh3_128":
        digest = xxhash.xxh3_128()
    elif algorithm == "xxh64":
        digest = xxhash.xxh64()
    elif algorithm == "sha256":
        import hashlib
        digest = hashlib.sha256()
    else:
        raise ValueError(f"Unsupported file checksum algorithm: {algorithm!r}")
    with local.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    detail = {
        "algorithm": algorithm,
        "digest": digest.hexdigest(),
        "bytes": local.stat().st_size,
    }
    if remote_path is not None:
        detail["path"] = remote_path
    return detail


def retry_call(fn, *args, op_name: str, **kwargs):
    max_attempts = int(kwargs.pop("max_attempts"))
    base_delay = float(kwargs.pop("base_delay_seconds"))
    max_delay = float(kwargs.pop("max_delay_seconds"))
    jitter = float(kwargs.pop("jitter_seconds"))
    last: Optional[Exception] = None
    for attempt in range(1, max_attempts + 1):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last = exc
            logging.getLogger("retry").warning("Attempt %d/%d for %s failed: %s", attempt, max_attempts, op_name, exc)
            if attempt == max_attempts:
                break
            delay = min(max_delay, base_delay * (2 ** (attempt - 1))) + random.uniform(0, jitter)
            time.sleep(delay)
    raise RuntimeError(f"{op_name} failed after {max_attempts} attempts") from last


def hf_token(common: Dict[str, Any]) -> str:
    env = common["huggingface"]["token_env_var"]
    token = os.environ.get(env)
    if not token:
        raise ValueError(f"Missing Hugging Face token in environment variable {env!r}")
    return token


def ensure_repo(api: HfApi, repo_id: str, token: str, common: Dict[str, Any], must_exist: bool = False) -> None:
    try:
        api.repo_info(repo_id=repo_id, repo_type="dataset", token=token)
        return
    except RepositoryNotFoundError:
        if must_exist:
            raise
    if not common["huggingface"].get("create_missing_output_repos", True):
        raise RepositoryNotFoundError(f"Dataset repository does not exist: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=bool(common["huggingface"].get("private", True)),
        exist_ok=True,
        token=token,
    )


def upload_file(api: HfApi, repo_id: str, local_path: Path, remote_path: str, token: str, common: Dict[str, Any]) -> None:
    with _io_gate(common, "upload").slot():
        retry_call(
            api.upload_file,
            path_or_fileobj=str(local_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
            token=token,
            op_name=f"upload:{repo_id}:{remote_path}",
            **retry_kwargs(common, "upload"),
        )


def upload_folder(api: HfApi, repo_id: str, local_dir: Path, remote_dir: str, token: str, common: Dict[str, Any]) -> None:
    """Publish a directory in one Hub commit to avoid commit-rate exhaustion."""
    with _io_gate(common, "upload").slot():
        retry_call(
            lambda: api.upload_folder(
                repo_id=repo_id,
                repo_type="dataset",
                folder_path=str(local_dir),
                path_in_repo=remote_dir,
                token=token,
                commit_message=f"Upload {remote_dir}",
            ),
            op_name=f"upload-folder:{repo_id}:{remote_dir}",
            **retry_kwargs(common, "upload"),
        )


def download_file(
    repo_id: str,
    filename: str,
    revision: str,
    token: str,
    common: Dict[str, Any],
    local_dir: str | Path | None = None,
) -> Path:
    settings = runtime_settings(common)
    download_options = retry_kwargs(common, "download")
    if settings["local_cache_dir"]:
        download_options["cache_dir"] = str(Path(settings["local_cache_dir"]).expanduser())
    with _io_gate(common, "download").slot():
        return Path(
            retry_call(
                hf_hub_download,
                repo_id=repo_id,
                filename=filename,
                revision=revision,
                repo_type="dataset",
                token=token,
                local_dir=str(local_dir) if local_dir else None,
                op_name=f"download:{repo_id}:{filename}",
                **download_options,
            )
        )


def list_repo_files(api: HfApi, repo_id: str, token: str, common: Dict[str, Any]) -> List[str]:
    with _io_gate(common, "download").slot():
        return list(
            retry_call(
                api.list_repo_files,
                repo_id=repo_id,
                repo_type="dataset",
                token=token,
                op_name=f"list:{repo_id}",
                **retry_kwargs(common, "download"),
            )
        )


def matches_any(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, p) for p in patterns)


def slug(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-")
    if not value:
        raise ValueError("Value cannot be converted to a non-empty slug")
    return value


def write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")


def read_remote_json(repo_id: str, path: str, token: str, common: Dict[str, Any]) -> Dict[str, Any]:
    local = download_file(repo_id, path, "main", token, common)
    return json.loads(local.read_text(encoding="utf-8"))


def write_failure_manifest(
    *,
    api: HfApi,
    repo_id: str,
    token: str,
    common: Dict[str, Any],
    local_root: Path,
    manifest_remote: str,
    artifact_contract: Dict[str, Any],
    stage: int,
    source_key: str,
    exc: Exception,
) -> None:
    """Persist a retryable failure record without marking a unit committed."""
    failure_dir = local_root / "failures" / f"stage{stage}" / source_key
    failure_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "artifact_contract": artifact_contract,
        "version": 1,
        "stage": stage,
        "source_key": source_key,
        "processing_status": "failed",
        "failure": {
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        },
        "error_count": 1,
        "errors_by_reason": {type(exc).__name__: 1},
        "counts": {
            "seen": 0,
            "accepted": 0,
            "rejected": 0,
            "duplicates": 0,
            "errors": 1,
            "rejected_by_reason": {},
            "duplicate_by_reason": {},
            "errors_by_reason": {type(exc).__name__: 1},
        },
        "failed_at": utc_now(),
    }
    local_manifest = failure_dir / "manifest.json"
    write_json(local_manifest, manifest)
    try:
        upload_file(api, repo_id, local_manifest, manifest_remote, token, common)
    except Exception:
        logging.getLogger("failure_manifest").exception(
            "Could not upload failure manifest %s", manifest_remote
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
